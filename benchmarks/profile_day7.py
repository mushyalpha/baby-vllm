#!/usr/bin/env python3

from __future__ import annotations
import argparse
import json
import math
import os
import sqlite3
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch



# KERNEL CLASSIFIER

GEMM_PATTERNS = [
    "gemm", "Gemm", "GEMM",
    "gemv", "Gemv", "GEMV",
    "cutlass", "Cutlass", "CUTLASS",
    "cublas", "cublasLt", "sm80_xmma", "sm90_xmma",
    "ampere_", "volta_", "turing_", "hopper_",
    "nvjet", "splitK",
]

ATTN_PATTERNS = [
    "paged_decode_attn",
    "flash_", "Flash", "fmha",
    "sdpa", "efficient_attention",
    "sm80_fmha", "sm90_fmha",
]

def classify_kernel(name: str) -> str:
    for p in ATTN_PATTERNS:
        if p in name: return "attention"
    for p in GEMM_PATTERNS:
        if p in name: return "GEMM"
    if "Memcpy" in name or "Memset" in name:
        return "memory"
    return "elementwise"



# PARSE AN NSYS .SQLITE EXPORT

def clean_wall_from_timing(path: str) -> dict:
    """Wall-clock p50 from a timing-only run. Never from an nsys trace."""
    with open(path) as f:
        data = json.load(f)
    if "p50_mean" in data:
        wall = float(data["p50_mean"])
        ci95 = float(data.get("p50_ci95") or 0.0)
    elif "step_ms_p50" in data:
        wall = float(data["step_ms_p50"])
        ci95 = 0.0
    else:
        print(f"ERROR: {path} has no p50_mean or step_ms_p50", file=sys.stderr)
        sys.exit(1)
    return {"wall_p50_ms": wall, "ci95": ci95}


def parse_nsys_sqlite(path: str, timing_json: str | None = None) -> dict:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Export with: nsys export -t sqlite {path.replace('.sqlite', '.nsys-rep')}")
        sys.exit(1)

    conn = sqlite3.connect(path)

    # Fetch NVTX decode step windows
    windows = []
    try:
        windows = conn.execute("""
            SELECT start, end FROM NVTX_EVENTS
            WHERE text LIKE 'decode_step_%'
        """).fetchall()
    except sqlite3.OperationalError:
        pass

    if not windows:
        try:
            windows = conn.execute("""
                SELECT n.start, n.end
                FROM NVTX_EVENTS n
                JOIN StringIds s ON n.textId = s.id
                WHERE s.value LIKE 'decode_step_%'
            """).fetchall()
        except sqlite3.OperationalError:
            print("Warning: Could not read NVTX_EVENTS.")

    # Fetch kernels
    try:
        rows = conn.execute("""
            SELECT s.value AS name, k.start, k.end
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            ORDER BY k.start
        """).fetchall()
    except sqlite3.OperationalError:
        try:
            rows = conn.execute("""
                SELECT s.value AS name, k.start, k.end
                FROM CUPTI_ACTIVITY_KIND_KERNEL k
                JOIN StringIds s ON k.demangledName = s.id
                ORDER BY k.start
            """).fetchall()
        except sqlite3.OperationalError:
            print("ERROR: Could not find kernel table. Check nsys version.")
            conn.close()
            sys.exit(1)

    # Fetch memcpys
    try:
        mcpys = conn.execute("""
            SELECT copyKind, start, end
            FROM CUPTI_ACTIVITY_KIND_MEMCPY
        """).fetchall()
        for kind, start, end in mcpys:
            name = "Memcpy H2D" if kind == 1 else "Memcpy D2H" if kind == 2 else "Memcpy D2D" if kind == 8 else "Memcpy"
            rows.append((name, start, end))
    except sqlite3.OperationalError:
        pass

    # Fetch memsets
    try:
        msets = conn.execute("""
            SELECT start, end
            FROM CUPTI_ACTIVITY_KIND_MEMSET
        """).fetchall()
        for start, end in msets:
            rows.append(("Memset", start, end))
    except sqlite3.OperationalError:
        pass

    conn.close()

    if not rows:
        print("No kernel rows found. Did you use --cuda-graph-trace=node?")
        sys.exit(1)

    measured_rows = []
    num_steps = len(windows)
    
    pre_launch_ms = 0.0
    in_graph_idle_ms = 0.0
    tail_ms = 0.0

    if windows:
        rows.sort(key=lambda x: x[1])
        for w_start, w_end in windows:
            w_kernels = [r for r in rows if r[1] >= w_start and r[2] <= w_end]
            if not w_kernels:
                continue
                
            w_kernels.sort(key=lambda x: x[1])
            
            for r in w_kernels:
                measured_rows.append((r[0], r[2] - r[1]))
                
            pre_launch_ms += (w_kernels[0][1] - w_start) / 1e6
            tail_ms += (w_end - w_kernels[-1][2]) / 1e6
            
            for i in range(1, len(w_kernels)):
                gap = w_kernels[i][1] - w_kernels[i-1][2]
                if gap > 0:
                    in_graph_idle_ms += gap / 1e6
    else:
        print("Warning: No decode_step NVTX ranges found. Falling back to all kernels.")
        measured_rows = [(r[0], r[2] - r[1]) for r in rows]
        num_steps = 1

    if not measured_rows:
        print("Error: NVTX ranges found, but no kernels fell inside them. Check synchronization.")
        sys.exit(1)

    buckets: dict[str, list[float]] = defaultdict(list)
    for name, dur_ns in measured_rows:
        cat = classify_kernel(name or "")
        buckets[cat].append(dur_ns / 1e6)

    total_ms = sum(sum(v) for v in buckets.values()) / num_steps
    avg_kernels_per_step = len(measured_rows) / num_steps

    print(f"\n{'='*65}")
    print(f"Day 7 Kernel Breakdown (Avg per decode step over {num_steps} steps)")
    print(f"{'='*65}")
    print(f"{'Category':<16} {'Count/Step':>12} {'ms/Step':>10} {'% of active':>10} {'Avg µs':>10}")
    print(f"{'-'*65}")
    
    for cat in ["GEMM", "elementwise", "attention", "memory"]:
        times = buckets.get(cat, [])
        if not times:
            continue
        cat_total = sum(times) / num_steps
        cat_count = len(times) / num_steps
        avg_us = (sum(times) / len(times)) * 1000 if len(times) else 0
        pct = 100.0 * cat_total / total_ms if total_ms else 0
        print(f"{cat:<16} {cat_count:>12.1f} {cat_total:>10.2f} {pct:>9.1f}% {avg_us:>10.1f}")
    
    print(f"{'-'*65}")
    print(f"{'ACTIVE GPU':<16} {avg_kernels_per_step:>12.1f} {total_ms:>10.2f} {'100.0':>9}%")

    avg_pre_launch = pre_launch_ms / num_steps if num_steps else 0.0
    avg_in_graph_idle = in_graph_idle_ms / num_steps if num_steps else 0.0
    avg_tail = tail_ms / num_steps if num_steps else 0.0
    # Sum of NVTX window time. --cuda-graph-trace=node inflates this; it is
    # not decode latency. Latency comes from the timing pass.
    nsys_window_ms = total_ms + avg_pre_launch + avg_in_graph_idle + avg_tail

    print(f"\n{'='*65}")
    print("nsys-observed gaps (profiler overhead included; not latency)")
    print(f"{'='*65}")
    print(f"pre-launch:         {avg_pre_launch:>6.2f} ms")
    print(f"in-graph gaps:      {avg_in_graph_idle:>6.2f} ms")
    print(f"tail:               {avg_tail:>6.2f} ms")
    print(f"nsys NVTX window:   {nsys_window_ms:>6.2f} ms")
    print(f"active GPU kernels: {total_ms:>6.2f} ms")
    print("Do not report the NVTX window as wall time.")

    print(f"\n{'='*65}")
    print("Top 20 elementwise kernels by name (Avg per step):")
    print(f"{'='*65}")
    ew_by_name: dict[str, list[float]] = defaultdict(list)
    for name, dur_ns in measured_rows:
        if classify_kernel(name or "") == "elementwise":
            short = (name or "unknown").split("<")[0].strip()
            ew_by_name[short].append(dur_ns / 1e6)
    
    sorted_ew = sorted(ew_by_name.items(), key=lambda kv: -sum(kv[1]))
    print(f"{'Kernel':<40} {'Count/Step':>12} {'ms/Step':>9}")
    print(f"{'-'*65}")
    for name, times in sorted_ew[:20]:
        cat_total = sum(times) / num_steps
        cat_count = len(times) / num_steps
        print(f"{name[:40]:<40} {cat_count:>12.1f} {cat_total:>9.3f}")

    gemm_ms = sum(buckets.get("GEMM", [])) / num_steps if num_steps else 0
    if gemm_ms > 0:
        eff_bw = 15.23 / (gemm_ms / 1000)
        print(f"\n{'='*65}")
        print(f"Effective GEMM Bandwidth: {eff_bw:.1f} GB/s (assuming 15.23 GB weights)")
        print(f"{'='*65}")

    result = {
        "measured_steps": num_steps,
        "avg_kernels_per_step": avg_kernels_per_step,
        "active_gpu_kernel_time_ms": total_ms,
        "avg_active_ms_per_step": total_ms,
        "nsys_nvtx_window_ms": nsys_window_ms,
        "idle_ms_nsys": {
            "pre_launch": avg_pre_launch,
            "in_graph_idle": avg_in_graph_idle,
            "tail": avg_tail,
        },
        "breakdown_per_step": {
            cat: {"count": len(times)/num_steps, "ms": sum(times)/num_steps}
            for cat, times in buckets.items()
        },
    }

    if timing_json:
        clean = clean_wall_from_timing(timing_json)
        overhead = clean["wall_p50_ms"] - total_ms
        result["timing_json"] = timing_json
        result["clean_wall_p50_ms"] = clean["wall_p50_ms"]
        result["clean_wall_p50_ci95_ms"] = clean["ci95"]
        result["clean_scheduling_overhead_ms"] = overhead
        print(f"\n{'='*65}")
        print("Clean overhead (timing pass − nsys kernel time)")
        print(f"{'='*65}")
        print(f"clean wall p50:  {clean['wall_p50_ms']:>8.3f} ± {clean['ci95']:.3f} ms")
        print(f"active kernels:  {total_ms:>8.3f} ms")
        print(f"scheduling:      {overhead:>8.3f} ms")
        print("Overhead uses the timing-pass p50, not the nsys NVTX window.")

    return result



# STEADY-STATE DECODE DRIVER

def summarize_trials(values: list[float]) -> dict:
    """Mean, sample std, and normal 95% CI half-width across independent trials."""
    n = len(values)
    if n == 0:
        raise ValueError("no trials")
    mean = sum(values) / n
    if n > 1:
        std = statistics.stdev(values)
        ci95 = 1.96 * (std / math.sqrt(n))
    else:
        std = 0.0
        ci95 = 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "min": min(values),
        "max": max(values),
        "n": n,
    }


def fit_decode_window(
    context: int,
    warmup_steps: int,
    decode_steps: int,
    max_seq_len: int,
) -> tuple[int, int, int, str | None]:
    """Keep every timed step inside the largest CUDA-graph context bucket.

    On the last timed step the sequence length is
    ``prompt + warmup_steps + decode_steps``. A prompt of 2048 with any
    warmup already steps past that bucket, and the runner then drops to
    eager mid-measurement.
    """

    def last_seq(prompt: int, warmup: int, steps: int) -> int:
        return prompt + warmup + steps

    if last_seq(context, warmup_steps, decode_steps) <= max_seq_len:
        return context, warmup_steps, decode_steps, None

    room = max_seq_len - context - warmup_steps
    if room >= 1:
        note = (
            f"Prompt {context} with {warmup_steps} warmup and {decode_steps} timed "
            f"steps would reach seq_len {last_seq(context, warmup_steps, decode_steps)}, "
            f"past CUDA-graph bucket {max_seq_len}. Timing {room} steps so the run "
            f"stays on the graph (seq_len {context + warmup_steps + 1}–"
            f"{context + warmup_steps + room})."
        )
        return context, warmup_steps, room, note

    if context + 1 < max_seq_len:
        steps = 1
        warmup = min(warmup_steps, max_seq_len - context - steps)
        note = (
            f"Prompt {context} does not leave room for {warmup_steps} warmup steps "
            f"inside bucket {max_seq_len}. Using warmup={warmup}, decode_steps={steps}."
        )
        return context, warmup, steps, note

    prompt = max_seq_len - warmup_steps - decode_steps
    if prompt < 1:
        raise SystemExit(
            f"warmup_steps ({warmup_steps}) + decode_steps ({decode_steps}) "
            f"exceed the CUDA-graph bucket ({max_seq_len})."
        )
    note = (
        f"Prompt {context} is already past the last graphed decode step "
        f"(bucket {max_seq_len}). Prefilling {prompt} tokens with {warmup_steps} "
        f"warmup and {decode_steps} timed steps so the window ends at seq_len "
        f"{max_seq_len}."
    )
    return prompt, warmup_steps, decode_steps, note


def _disable_inductor_cudagraphs() -> None:
    # ModelRunner captures the decode graph. A second Inductor graph inside
    # that region would nest captures, and autotune would change the kernels
    # between warmup and capture.
    import torch._inductor.config as inductor_config
    triton_cfg = getattr(inductor_config, "triton", None)
    if triton_cfg is not None and hasattr(triton_cfg, "cudagraphs"):
        triton_cfg.cudagraphs = False
    if hasattr(inductor_config, "max_autotune"):
        inductor_config.max_autotune = False


def compile_decoder_layers(model) -> int:
    _disable_inductor_cudagraphs()
    layers = model.model.layers
    for i in range(len(layers)):
        layers[i] = torch.compile(layers[i], backend="inductor", dynamic=False)
    return len(layers)


def _step_sample(step_times: list[float], batch: int) -> dict:
    p50 = float(statistics.median(step_times))
    n = len(step_times)
    mean = sum(step_times) / n
    std = statistics.stdev(step_times) if n > 1 else 0.0
    return {
        "steps": n,
        "step_ms_p50": p50,
        "step_ms_mean": mean,
        "step_ms_std": std,
        "step_ms_min": min(step_times),
        "step_ms_max": max(step_times),
        "tok_s": batch / (p50 / 1e3),
    }


def _timed_decode(engine, prompt_ids: list[int], batch: int, warmup_steps: int, decode_steps: int, repeat: int) -> list[float]:
    from babyvllm.sequence import SamplingParams

    engine.reset()
    total_gen = warmup_steps + decode_steps + 5
    for _ in range(batch):
        engine.add_request(
            list(prompt_ids),
            SamplingParams(temperature=0.0, max_tokens=total_gen, ignore_eos=True),
        )

    while True:
        outputs = engine.step()
        if len(outputs) == batch and all(len(out.new_token_ids) > 0 for out in outputs):
            break

    for _ in range(warmup_steps):
        engine.step()

    step_times = []
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"MEASURE_B{batch}_R{repeat}")
    try:
        for i in range(decode_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            torch.cuda.nvtx.range_push(f"decode_step_{i}")
            engine.step()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            step_times.append((time.perf_counter() - t0) * 1e3)
    finally:
        torch.cuda.nvtx.range_pop()
    return step_times


def run_steady_decode(
    model_name: str,
    batch: int,
    context: int,
    decode_steps: int,
    warmup_steps: int,
    repeats: int,
    compile_model: bool,
) -> dict:
    from babyvllm.config import SchedulerConfig
    from babyvllm.llm import LLM
    from babyvllm.worker.model_runner import BATCH_BUCKETS, CTX_BUCKETS, ceil_to_bucket

    sched = SchedulerConfig(
        max_num_seqs=max(batch, 256),
        # Tokens one step may run. 64 × 1920 = 122_880, and that eager prefill
        # asks gate_up for an 8.67 GiB activation after the KV cache has taken
        # the rest of an 80 GB H100. Decode is one token per sequence either way.
        max_num_batched_tokens=8192,
    )

    llm = LLM(model_name, scheduler_config=sched, verbose=True)
    runner = llm.model_runner

    if compile_model:
        n_layers = compile_decoder_layers(runner.model)
        print(
            f"torch.compile inductor on {n_layers} decoder layers. "
            "CUDA graph capture stays in ModelRunner.",
            flush=True,
        )

    if runner.use_cuda_graphs and ceil_to_bucket(batch, BATCH_BUCKETS) is None:
        raise SystemExit(
            f"batch {batch} is above the largest CUDA-graph bucket ({max(BATCH_BUCKETS)})."
        )

    prompt_len = context
    warmup = warmup_steps
    steps = decode_steps
    fit_note = None
    if runner.use_cuda_graphs:
        prompt_len, warmup, steps, fit_note = fit_decode_window(
            context, warmup_steps, decode_steps, max(CTX_BUCKETS)
        )
        if fit_note:
            print(fit_note, flush=True)
        print(
            "Pre-capturing CUDA graphs for every batch/context bucket "
            "so timed steps do not pay for capture.",
            flush=True,
        )
        runner.capture_all(warmup_steps=3)
        torch.cuda.synchronize()
        print(f"Captured {len(runner._graphs)} graphs.", flush=True)

    tokenizer = llm.tokenizer
    filler = tokenizer.encode("The quick brown fox jumps over the lazy dog. ") or [1]
    prompt_ids = (filler * ((prompt_len // len(filler)) + 1))[:prompt_len]

    trials = []
    for repeat in range(1, repeats + 1):
        print(
            f"\nTrial {repeat}/{repeats} — B={batch}, prompt={prompt_len}, "
            f"warmup={warmup}, timed={steps}",
            flush=True,
        )
        step_times = _timed_decode(
            llm.engine, prompt_ids, batch, warmup, steps, repeat
        )
        sample = _step_sample(step_times, batch)
        sample["repeat"] = repeat
        trials.append(sample)
        print(
            f"  p50 {sample['step_ms_p50']:.3f} ms | mean {sample['step_ms_mean']:.3f} ms",
            flush=True,
        )

    p50_stats = summarize_trials([t["step_ms_p50"] for t in trials])
    mean_stats = summarize_trials([t["step_ms_mean"] for t in trials])
    seq_first = prompt_len + warmup + 1
    seq_last = prompt_len + warmup + steps

    print(
        f"\nDecode step p50 — B={batch}, prompt={prompt_len} "
        f"(requested ctx {context}), repeats={repeats}\n"
        f"  p50:  {p50_stats['mean']:.3f} ± {p50_stats['ci95']:.3f} ms (95% CI)\n"
        f"  mean: {mean_stats['mean']:.3f} ± {mean_stats['ci95']:.3f} ms\n"
        f"  seq_len {seq_first}–{seq_last}",
        flush=True,
    )

    return {
        "model": model_name,
        "batch": batch,
        "context": context,
        "prompt_len": prompt_len,
        "seq_len_first": seq_first,
        "seq_len_last": seq_last,
        "ctx_bucket_first": ceil_to_bucket(seq_first, CTX_BUCKETS),
        "ctx_bucket_last": ceil_to_bucket(seq_last, CTX_BUCKETS),
        "decode_steps": steps,
        "decode_steps_requested": decode_steps,
        "warmup_steps": warmup,
        "warmup_steps_requested": warmup_steps,
        "repeats": repeats,
        "compile": compile_model,
        "cuda_graphs": runner.use_cuda_graphs,
        "graphs_precaptured": bool(runner.use_cuda_graphs),
        "primary_metric": "p50",
        "ci_method": "normal_1.96_se",
        "p50_mean": p50_stats["mean"],
        "p50_std": p50_stats["std"],
        "p50_ci95": p50_stats["ci95"],
        "p50_min": p50_stats["min"],
        "p50_max": p50_stats["max"],
        "mean_of_means": mean_stats["mean"],
        "mean_std": mean_stats["std"],
        "mean_ci95": mean_stats["ci95"],
        "step_ms_p50": p50_stats["mean"],
        "step_ms_mean": mean_stats["mean"],
        "tok_s": batch / (p50_stats["mean"] / 1e3),
        "fit_note": fit_note,
        "trials": trials,
    }


def _write_json(path: str, payload: dict) -> None:
    if not path or path == "/dev/null":
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}", flush=True)


def write_profile_output(path: str, result: dict) -> None:
    _write_json(path, result)
    trials = result.get("trials")
    if not trials or not path.endswith(".json"):
        return
    stem = path[:-5]
    meta_keys = (
        "model", "batch", "context", "prompt_len", "seq_len_first", "seq_len_last",
        "compile", "cuda_graphs", "graphs_precaptured", "decode_steps", "warmup_steps",
    )
    meta = {k: result[k] for k in meta_keys if k in result}
    for trial in trials:
        payload = dict(meta)
        payload.update(trial)
        _write_json(f"{stem}_run{trial['repeat']}.json", payload)


def resolve_out(args: argparse.Namespace) -> str:
    if args.out:
        return args.out
    if args.parse:
        return "day7_breakdown.json"
    suffix = "_compile" if args.compile else ""
    return f"day7/timing_b{args.batch}_ctx{args.context}{suffix}.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--decode-steps", type=int, default=50)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3,
                   help="Independent timing trials. 95%% CI is 1.96 * s / sqrt(n) on the trial medians.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile(backend=inductor) on each decoder layer. "
                        "Inductor CUDA graphs stay off; ModelRunner captures the decode graph.")
    p.add_argument("--parse", type=str, default=None)
    p.add_argument("--timing-json", default=None,
                   help="Timing-pass JSON. With --parse, scheduling overhead uses this wall time.")
    p.add_argument("--out", default=None,
                   help="Default is day7/timing_b{batch}_ctx{context}[_compile].json")
    return p.parse_args()

def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        print("ERROR: --repeats must be >= 1", file=sys.stderr)
        return 2
    if args.parse:
        result = parse_nsys_sqlite(args.parse, timing_json=args.timing_json)
    else:
        result = run_steady_decode(
            args.model,
            args.batch,
            args.context,
            args.decode_steps,
            args.warmup_steps,
            args.repeats,
            args.compile,
        )

    write_profile_output(resolve_out(args), result)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
