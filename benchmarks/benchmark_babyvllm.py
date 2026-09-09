from __future__ import annotations

import argparse
import json
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from babyvllm.config import SchedulerConfig
from babyvllm.llm import LLM
from babyvllm.sequence import SamplingParams


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    t = k - lo
    return xs[lo] * (1.0 - t) + xs[hi] * t


def prompt_ids(tokenizer, length: int) -> list[int]:
    filler = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    if not filler:
        filler = [1]
    out: list[int] = []
    while len(out) < length:
        out.extend(filler)
    return out[:length]


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_once(
    llm: LLM,
    prompts: list[list[int]],
    gen_lens: list[int],
    max_num_seqs: int,
) -> dict:
    llm.engine.reset()
    llm.engine.scheduler.max_num_seqs = max_num_seqs
    device = llm.model_runner.device

    for ids, n_gen in zip(prompts, gen_lens):
        llm.engine.add_request(
            ids,
            SamplingParams(temperature=0.0, max_tokens=n_gen, ignore_eos=True),
        )

    ttft: dict[int, float] = {}
    last_token_t: dict[int, float] = {}
    itls: list[float] = []
    generated = 0
    steps = 0

    sync(device)
    t_submit = time.perf_counter()
    while llm.engine.has_unfinished_requests():
        sync(device)
        outputs = llm.engine.step()
        sync(device)
        t_after = time.perf_counter()
        steps += 1
        for out in outputs:
            n = len(out.new_token_ids)
            if not n:
                continue
            generated += n
            if out.seq_id not in ttft:
                ttft[out.seq_id] = t_after - t_submit
            if out.seq_id in last_token_t:
                itls.append(t_after - last_token_t[out.seq_id])
            last_token_t[out.seq_id] = t_after

    elapsed = time.perf_counter() - t_submit
    ttfts = list(ttft.values())
    return {
        "max_num_seqs": max_num_seqs,
        "requests": len(prompts),
        "steps": steps,
        "gen_tokens": generated,
        "elapsed_s": elapsed,
        "tok_s": generated / elapsed if elapsed > 0 else 0.0,
        "ttft_p50_ms": (percentile(ttfts, 50) or 0.0) * 1e3,
        "ttft_p99_ms": (percentile(ttfts, 99) or 0.0) * 1e3,
        "itl_p50_ms": (percentile(itls, 50) or 0.0) * 1e3,
        "itl_p99_ms": (percentile(itls, 99) or 0.0) * 1e3,
    }


def run_hf(
    model_name: str,
    prompts: list[list[int]],
    gen_lens: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    from transformers import AutoModelForCausalLM

    print("\nHuggingFace sequential generate (one request at a time)...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()

    generated = 0
    sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, (ids, n_gen) in enumerate(zip(prompts, gen_lens), 1):
            inp = torch.tensor([ids], device=device)
            out = hf.generate(
                inp,
                max_new_tokens=n_gen,
                do_sample=False,
                use_cache=True,
            )
            generated += out.shape[1] - inp.shape[1]
            print(f"  HF {i}/{len(prompts)}  {time.perf_counter() - t0:.1f}s", flush=True)
    sync(device)
    elapsed = time.perf_counter() - t0
    del hf
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "engine": "huggingface_sequential",
        "gen_tokens": generated,
        "elapsed_s": elapsed,
        "tok_s": generated / elapsed if elapsed > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark baby-vLLM offline throughput")
    p.add_argument("--model", default="Qwen/Qwen2-0.5B")
    p.add_argument("--num-requests", type=int, default=32)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--seqs", default="1,4,8,16,32", help="max_num_seqs sweep")
    p.add_argument("--max-batched-tokens", type=int, default=2048)
    p.add_argument("--mixed", action="store_true", help="ShareGPT-like random lengths")
    p.add_argument("--vs-hf", action="store_true", help="Also time sequential HuggingFace")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--nsys-focus", action="store_true")
    p.add_argument("--out", default="babyvllm_bench.json")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.num_requests = 8
        args.prompt_len = 64
        args.max_tokens = 16
        args.seqs = "1,8"
        args.vs_hf = False
    if args.nsys_focus:
        args.num_requests = 4
        args.prompt_len = 64
        args.max_tokens = 16
        args.seqs = "4"
        args.vs_hf = False

    print(
        f"Loading baby-vLLM  model={args.model}  "
        f"requests={args.num_requests} prompt={args.prompt_len} gen={args.max_tokens}",
        flush=True,
    )
    sched = SchedulerConfig(
        max_num_seqs=max(parse_int_list(args.seqs)),
        max_num_batched_tokens=args.max_batched_tokens,
    )
    llm = LLM(args.model, scheduler_config=sched, verbose=True)
    device = llm.model_runner.device
    tokenizer = llm.tokenizer
    profile = llm.model_runner.kv_profile or {}

    rng = torch.Generator().manual_seed(args.seed)
    prompts: list[list[int]] = []
    gen_lens: list[int] = []
    for _ in range(args.num_requests):
        if args.mixed:
            p_len = int(torch.randint(32, args.prompt_len + 1, (1,), generator=rng).item())
            g_len = int(torch.randint(8, args.max_tokens + 1, (1,), generator=rng).item())
        else:
            p_len = args.prompt_len
            g_len = args.max_tokens
        prompts.append(prompt_ids(tokenizer, p_len))
        gen_lens.append(g_len)

    print("\n========== Engine ==========")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"device={device}  dtype={llm.model_runner.dtype}")
    print(f"KV profile: {json.dumps(profile, indent=2)}")
    print("============================\n")

    print("Warmup...", flush=True)
    run_once(llm, prompts[: min(2, len(prompts))], gen_lens[: min(2, len(prompts))], max_num_seqs=1)

    results = []
    for n_seqs in parse_int_list(args.seqs):
        print(f"\n--- baby-vLLM max_num_seqs={n_seqs} ---", flush=True)
        row = run_once(llm, prompts, gen_lens, max_num_seqs=n_seqs)
        results.append(row)
        print(
            f"  {row['tok_s']:.1f} tok/s  "
            f"{row['gen_tokens']} tokens in {row['elapsed_s']:.2f}s  "
            f"{row['steps']} steps  "
            f"TTFT p50/p99 {row['ttft_p50_ms']:.1f}/{row['ttft_p99_ms']:.1f} ms  "
            f"ITL p50/p99 {row['itl_p50_ms']:.2f}/{row['itl_p99_ms']:.2f} ms",
            flush=True,
        )

    hf_row = None
    if args.vs_hf:
        hf_row = run_hf(
            args.model, prompts, gen_lens, device, llm.model_runner.dtype
        )
        print(
            f"-> HF sequential: {hf_row['tok_s']:.2f} tok/s  ({hf_row['elapsed_s']:.1f}s)",
            flush=True,
        )

    print("\n========== Results ==========")
    print(
        f"{'seqs':>6} {'tok/s':>10} {'TTFT p50':>10} {'TTFT p99':>10} "
        f"{'ITL p50':>10} {'ITL p99':>10} {'steps':>7}"
    )
    for row in results:
        print(
            f"{row['max_num_seqs']:6d} {row['tok_s']:10.1f} "
            f"{row['ttft_p50_ms']:10.1f} {row['ttft_p99_ms']:10.1f} "
            f"{row['itl_p50_ms']:10.2f} {row['itl_p99_ms']:10.2f} "
            f"{row['steps']:7d}"
        )
    if hf_row:
        best = max(results, key=lambda r: r["tok_s"])
        print(
            f"\nHF sequential: {hf_row['tok_s']:.1f} tok/s  |  "
            f"baby-vLLM best: {best['tok_s']:.1f} tok/s  "
            f"({best['tok_s'] / hf_row['tok_s']:.2f}x) at max_num_seqs={best['max_num_seqs']}"
        )
        print(
            "This is vs one-request-at-a-time Transformers, not vs vLLM. "
            "The remaining gap to vLLM is the Python paged-attention gather."
        )

    payload = {
        "model": args.model,
        "device": str(device),
        "kv_profile": profile,
        "workload": {
            "num_requests": args.num_requests,
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "mixed": args.mixed,
        },
        "babyvllm": results,
        "huggingface_sequential": hf_row,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
