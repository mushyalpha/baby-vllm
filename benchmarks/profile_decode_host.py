#!/usr/bin/env python3
"""Where does a steady decode step go, host side and device side?

``profile_day7.py`` says a batch-64 step is 7.09 ms of wall clock, 6.19 ms of
kernel time in Nsight, and calls the difference "scheduling overhead". That
difference mixes work the host can remove with idle time inside the graph it
cannot. This splits them.

For each measured engine step it records:

  wall         engine.step(), host clock, ends after the sampled tokens are on the host
  schedule     Scheduler.schedule()
  prepare      the runner's input preparation for the graph (whichever of
               prepare_static_inputs / prepare_decode_inputs the code has)
  launch       host time inside ModelRunner.replay (queueing the graph)
  update       Scheduler.update_from_output()
  rest         wall minus the four above: waiting for the GPU, the sampler or
               the token copy, and engine glue
  graph_gpu    CUDA events around replay: device time of the graph itself
  exposed      wall minus graph_gpu: everything a step costs beyond the graph.
               This is the number host-side work can attack.

Run it before and after a change on the same pod, same flags. ``--cprofile``
adds Python function totals over the measured steps.

The published shape is Qwen2.5-7B, bf16, H100, batch 64, prompt 128. Anything
else is written with publishable=false.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

PHASES = ("schedule", "prepare", "launch", "update")


class StepProbe:
    """Wraps engine components in place. restore() undoes it."""

    def __init__(self, engine, use_events: bool):
        self.engine = engine
        self.use_events = use_events
        self.times: dict[str, float] = {p: 0.0 for p in PHASES}
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._restore: list[tuple[object, str, object]] = []
        self.prepare_name = None

    def _wrap(self, obj, attr: str, phase: str, events: bool = False):
        original = getattr(obj, attr)

        def timed(*args, **kwargs):
            if events:
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
            t0 = time.perf_counter()
            out = original(*args, **kwargs)
            self.times[phase] += time.perf_counter() - t0
            if events:
                b.record()
                self.events.append((a, b))
            return out

        self._restore.append((obj, attr, original))
        setattr(obj, attr, timed)

    def install(self):
        runner = self.engine.model_runner
        self._wrap(self.engine.scheduler, "schedule", "schedule")
        self._wrap(self.engine.scheduler, "update_from_output", "update")
        for name in ("prepare_decode_inputs", "prepare_static_inputs"):
            if hasattr(runner, name):
                self.prepare_name = name
                self._wrap(runner, name, "prepare")
                break
        self._wrap(runner, "replay", "launch", events=self.use_events)

    def restore(self):
        for obj, attr, original in reversed(self._restore):
            setattr(obj, attr, original)
        self._restore.clear()

    def take(self) -> dict[str, float]:
        out = dict(self.times)
        self.times = {p: 0.0 for p in PHASES}
        return out


def run_trial(engine, prompt_ids, batch, warmup, steps, device, cprofile: bool):
    from babyvllm.sequence import SamplingParams

    use_events = device.type == "cuda"
    engine.reset()
    total = warmup + steps + 5
    for _ in range(batch):
        engine.add_request(
            list(prompt_ids),
            SamplingParams(temperature=0.0, max_tokens=total, ignore_eos=True),
        )
    # Prefill steps, until every request has its first token.
    while True:
        outs = engine.step()
        if len(outs) == batch and all(o.new_token_ids for o in outs):
            break
    for _ in range(warmup):
        engine.step()

    probe = StepProbe(engine, use_events)
    probe.install()
    prof = cProfile.Profile() if cprofile else None
    rows = []
    try:
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(steps):
            probe.take()
            probe.events.clear()
            t0 = time.perf_counter()
            if prof:
                prof.enable()
            engine.step()
            if prof:
                prof.disable()
            wall = time.perf_counter() - t0
            phases = probe.take()
            row = {"wall": wall * 1e3, **{k: v * 1e3 for k, v in phases.items()}}
            row["rest"] = row["wall"] - sum(row[p] for p in PHASES)
            if use_events and probe.events:
                # The step already synchronised on the sampled tokens.
                row["graph_gpu"] = sum(a.elapsed_time(b) for a, b in probe.events)
                row["exposed"] = row["wall"] - row["graph_gpu"]
            rows.append(row)
            probe.events.clear()
    finally:
        probe.restore()
    return rows, prof, probe.prepare_name


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {k: statistics.median(r[k] for r in rows) for k in keys}


def ci95(values: list[float]) -> float:
    return 1.96 * statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cprofile", action="store_true", help="Python function totals over the measured steps")
    p.add_argument("--label", default="run", help="Free text stored in the JSON, e.g. a commit")
    p.add_argument("--out", default="decode_host_profile.json")
    args = p.parse_args()

    from benchmark_vs_vllm import gpu_name, load_tokenizer, publishable

    from babyvllm.config import SchedulerConfig
    from babyvllm.llm import LLM
    from babyvllm.worker.model_runner import BATCH_BUCKETS, CTX_BUCKETS, ceil_to_bucket

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if ceil_to_bucket(args.batch, BATCH_BUCKETS) is None:
        raise SystemExit(f"batch {args.batch} is above the largest graph bucket {max(BATCH_BUCKETS)}")
    name = gpu_name()
    can_publish = publishable(name, args.model, "bfloat16") and device.type == "cuda" and args.repeats >= 3

    llm = LLM(
        args.model,
        scheduler_config=SchedulerConfig(
            max_num_seqs=max(args.batch, 1),
            max_num_batched_tokens=max(8192, args.batch * args.prompt_len),
        ),
        verbose=True,
    )
    if llm.model_runner.use_cuda_graphs:
        llm.model_runner.capture_all(warmup_steps=3)
        torch.cuda.synchronize()
    tok = load_tokenizer(args.model)
    filler = tok.encode("The quick brown fox jumps over the lazy dog. ", add_special_tokens=False) or [1]
    prompt = (filler * (args.prompt_len // len(filler) + 1))[: args.prompt_len]
    last_seq = args.prompt_len + args.warmup + args.steps + 1
    if last_seq > max(CTX_BUCKETS):
        raise SystemExit(f"sequence would reach {last_seq}, past the largest context bucket {max(CTX_BUCKETS)}")

    trials = []
    prepare_name = None
    for r in range(1, args.repeats + 1):
        rows, prof, prepare_name = run_trial(
            llm.engine, prompt, args.batch, args.warmup, args.steps, device, args.cprofile and r == 1
        )
        s = summarize(rows)
        trials.append(s)
        print(f"trial {r}: " + " | ".join(f"{k} {v:.3f}" for k, v in s.items()), flush=True)
        if prof is not None:
            buf = io.StringIO()
            pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(18)
            print(buf.getvalue(), flush=True)

    result = {k: {"p50_of_trials": statistics.median(t[k] for t in trials),
                  "ci95": ci95([t[k] for t in trials])} for k in trials[0]}
    print("\nms per decode step, median over steps, then over trials (± 95% interval)")
    for k, v in result.items():
        print(f"  {k:<10} {v['p50_of_trials']:7.3f} ± {v['ci95']:.3f}")

    payload = {
        "metadata": {
            "label": args.label,
            "gpu": name,
            "model": args.model,
            "dtype": "bfloat16",
            "publishable": can_publish,
            "batch": args.batch,
            "prompt_len": args.prompt_len,
            "warmup_steps": args.warmup,
            "steps": args.steps,
            "repeats": args.repeats,
            "prepare_function": prepare_name,
            "pytorch_version": torch.__version__,
            "command": "python " + " ".join(sys.argv),
        },
        "ms_per_step": result,
        "trials": trials,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
