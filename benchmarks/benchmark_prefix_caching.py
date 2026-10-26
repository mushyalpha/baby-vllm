#!/usr/bin/env python3
"""Prefix caching on and off, same engine, same pod: time to first token and hit rate.

Workload: every request is one shared system prompt of --system-lens tokens
followed by its own --suffix-len unique tokens. Token ids are exact. The system
prompt is a multiple of the 16-token block size so every cached block is a
whole system-prompt block.

Two arrival patterns, each run with caching off and on:

  sequential  One request at a time, --n-seq requests, one new token each.
              With caching on, request 0 is cold and the rest hit. Reported
              separately: "cold first" and "warm" (median of requests 1..n-1).
  burst       --burst requests submitted together onto an empty cache, each
              generating --burst-max-tokens. Per-request time to first token,
              total time, and the peak number of KV blocks in use. With caching
              on, the first request fills the system prompt blocks and the
              others share them inside the same scheduling step.

One LLM instance serves both modes. Only KVCacheManager.enable_prefix_caching
changes, and the cache is reset before every run. Modes alternate order each
trial so drift does not land on one side. Every cell has one untimed warmup.

Hit rate is KVCacheManager's own counter: cached prompt tokens over prompt
tokens queried at admission. The script also prints the rate the workload
should give from block arithmetic, so a counter bug cannot pass quietly.

The published number is Qwen2.5-7B, bf16, H100. Anything else is written with
publishable=false.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from benchmark_vs_vllm import (  # noqa: E402
    PUBLISHED_MODEL,
    gpu_name,
    load_tokenizer,
    package_version,
    publishable,
    summarize_trials,
    sync,
)

PREFILL_TOKEN_CAP = 8192
SALT_BASE = 1000


# ---------------------------------------------------------------- pure helpers


def make_shared_prefix_prompts(
    base: list[int],
    n: int,
    system_len: int,
    suffix_len: int,
    vocab_size: int,
    salt: int = 0,
) -> list[list[int]]:
    """n prompts of system_len + suffix_len tokens sharing the first system_len.

    ``salt`` changes the very first token, so two salts share no block even
    though block hashes are chained from the first one.
    """
    if n < 1 or system_len < 1 or suffix_len < 2:
        raise ValueError("need n >= 1, system_len >= 1, suffix_len >= 2")
    if not base:
        base = [10]
    span = min(vocab_size, 1000) - 11
    if span < 2 or n > span * span:
        raise ValueError("vocab_size is too small to build distinct suffixes")
    system = (list(base) * (system_len // len(base) + 1))[:system_len]
    system[0] = SALT_BASE + salt
    if system[0] >= vocab_size:
        raise ValueError("salt is outside the vocabulary")
    prompts = []
    for i in range(n):
        tail = (list(base) * (suffix_len // len(base) + 1))[:suffix_len]
        tail[0] = 10 + (i % span)
        tail[1] = 10 + ((i // span) % span)
        prompts.append(system + tail)
    return prompts


def cached_tokens_per_hit(system_len: int, prompt_len: int, block_size: int) -> int:
    """Tokens a warm request can take from the cache.

    Whole blocks of the shared prefix, and never the block holding the last
    prompt token, so at least one token is always computed.
    """
    max_blocks = (prompt_len - 1) // block_size
    return min(system_len // block_size, max_blocks) * block_size


def expected_hit_rate(
    system_len: int, suffix_len: int, block_size: int, n_requests: int, first_is_cold: bool = True
) -> float:
    """Block arithmetic for a run where request 0 fills the cache."""
    prompt_len = system_len + suffix_len
    hit = cached_tokens_per_hit(system_len, prompt_len, block_size)
    cold = 1 if first_is_cold else 0
    return (n_requests - cold) * hit / (n_requests * prompt_len)


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def ids_match(reference: list[list[int]], other: list[list[int]]) -> dict[str, Any]:
    bad = [i for i, (a, b) in enumerate(zip(reference, other)) if a != b]
    return {
        "match": len(reference) == len(other) and not bad,
        "sequences": len(reference),
        "mismatched_sequences": len(bad) + abs(len(reference) - len(other)),
    }


# ------------------------------------------------------------------ engine side


def set_caching(engine, on: bool) -> None:
    engine.kv_cache_manager.enable_prefix_caching = on
    engine.kv_cache_manager.reset()


def run_requests(engine, prompts: list[list[int]], max_tokens: int, device: torch.device) -> dict[str, Any]:
    """Submit every prompt now and step until all finish.

    A request's time to first token is the time from submission to the end of
    the step that returned its first token. The step returns after the sampler
    has copied the token to the host, so this is wall clock, not launch time.
    """
    from babyvllm.sequence import SamplingParams

    kv = engine.kv_cache_manager
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    hit0, queried0 = kv.num_hit_tokens, kv.num_queried_tokens

    # Blocks are freed in the step a request finishes, so peak usage has to be
    # read just before the forward pass of each step.
    runner = engine.model_runner
    original = runner.execute_model
    peak = 0

    def watched(scheduler_output):
        nonlocal peak
        peak = max(peak, len(kv.used_block_ids))
        return original(scheduler_output)

    runner.execute_model = watched
    try:
        order = [engine.add_request(list(p), copy.copy(params)) for p in prompts]
        tokens: dict[int, list[int]] = {i: [] for i in order}
        first: dict[int, float] = {}
        sync(device)
        start = time.perf_counter()
        while engine.has_unfinished_requests():
            outs = engine.step()
            now = time.perf_counter()
            for out in outs:
                tokens[out.seq_id].extend(out.new_token_ids)
                if out.new_token_ids and out.seq_id not in first:
                    first[out.seq_id] = now - start
        sync(device)
        total = time.perf_counter() - start
    finally:
        runner.execute_model = original

    return {
        "ttft_ms": [first[i] * 1000.0 for i in order],
        "total_ms": total * 1000.0,
        "ids": [tokens[i] for i in order],
        "peak_blocks": peak,
        "hit_tokens": kv.num_hit_tokens - hit0,
        # The manager only counts queries while caching is on, so take the
        # prompt tokens from the prompts. Otherwise "off" reads as zero.
        "queried_tokens": sum(len(p) for p in prompts),
        "manager_queried_tokens": kv.num_queried_tokens - queried0,
    }


def rate(hit: int, queried: int) -> float:
    return hit / queried if queried else 0.0


def run_sequential(engine, prompts, device) -> dict[str, Any]:
    ttfts: list[float] = []
    ids: list[list[int]] = []
    hit_after_first = queried_after_first = hit = queried = 0
    for i, prompt in enumerate(prompts):
        r = run_requests(engine, [prompt], 1, device)
        ttfts.append(r["ttft_ms"][0])
        ids.append(r["ids"][0])
        hit += r["hit_tokens"]
        queried += r["queried_tokens"]
        if i == 0:
            hit_after_first, queried_after_first = hit, queried
    return {
        "cold_first_ttft_ms": ttfts[0],
        "warm_ttft_ms": float(statistics.median(ttfts[1:])) if len(ttfts) > 1 else ttfts[0],
        "hit_rate": rate(hit, queried),
        "warm_hit_rate": rate(hit - hit_after_first, queried - queried_after_first),
        "ids": ids,
    }


def run_burst(engine, prompts, max_tokens, device) -> dict[str, Any]:
    r = run_requests(engine, prompts, max_tokens, device)
    t = r["ttft_ms"]
    return {
        "ttft_mean_ms": float(statistics.fmean(t)),
        "ttft_p50_ms": percentile(t, 0.5),
        "ttft_max_ms": max(t),
        "total_ms": r["total_ms"],
        "peak_blocks": r["peak_blocks"],
        "hit_rate": rate(r["hit_tokens"], r["queried_tokens"]),
        "prefill_tokens_computed": r["queried_tokens"] - r["hit_tokens"],
        "ids": r["ids"],
    }


def one_cell(engine, mode: str, args, base, vocab, device, system_len, salt) -> dict[str, Any]:
    on = mode == "on"
    seq_prompts = make_shared_prefix_prompts(base, args.n_seq, system_len, args.suffix_len, vocab, salt)
    burst_prompts = make_shared_prefix_prompts(base, args.burst, system_len, args.suffix_len, vocab, salt + 1)
    set_caching(engine, on)
    seq = run_sequential(engine, seq_prompts, device)
    set_caching(engine, on)
    burst = run_burst(engine, burst_prompts, args.burst_max_tokens, device)
    set_caching(engine, False)
    return {"sequential": seq, "burst": burst}


def aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for scenario in ("sequential", "burst"):
        keys = [k for k in trials[0][scenario] if k != "ids"]
        out[scenario] = {
            k: {
                "median": round(summarize_trials([t[scenario][k] for t in trials])["median"], 4),
                "ci95": round(summarize_trials([t[scenario][k] for t in trials])["ci95"], 4),
            }
            for k in keys
        }
    return out


def run_length(engine, args, base, vocab, device, system_len: int) -> dict[str, Any]:
    bs = engine.block_size
    print(f"\nsystem prompt {system_len} tokens + {args.suffix_len} unique", flush=True)
    for mode in ("off", "on"):
        one_cell(engine, mode, args, base, vocab, device, system_len, salt=0)  # warmup

    per_mode: dict[str, list[dict[str, Any]]] = {"off": [], "on": []}
    for trial in range(args.repeats):
        order = ("off", "on") if trial % 2 == 0 else ("on", "off")
        for mode in order:
            cell = one_cell(engine, mode, args, base, vocab, device, system_len, salt=2 + 2 * trial)
            per_mode[mode].append(cell)
            s, b = cell["sequential"], cell["burst"]
            print(
                f"  trial {trial + 1} {mode:>3}: seq warm {s['warm_ttft_ms']:7.1f} ms "
                f"(cold {s['cold_first_ttft_ms']:7.1f}) | burst mean {b['ttft_mean_ms']:7.1f} ms "
                f"last {b['ttft_max_ms']:7.1f} ms | blocks {b['peak_blocks']:>5} "
                f"| hit {b['hit_rate'] * 100:5.1f}%",
                flush=True,
            )

    off_ids = per_mode["off"][0]
    on_ids = per_mode["on"][0]
    return {
        "system_len": system_len,
        "suffix_len": args.suffix_len,
        "expected_hit_rate_sequential": round(
            expected_hit_rate(system_len, args.suffix_len, bs, args.n_seq), 4
        ),
        "expected_hit_rate_burst": round(
            expected_hit_rate(system_len, args.suffix_len, bs, args.burst), 4
        ),
        "off": aggregate(per_mode["off"]),
        "on": aggregate(per_mode["on"]),
        "greedy_match_sequential": ids_match(off_ids["sequential"]["ids"], on_ids["sequential"]["ids"]),
        "greedy_match_burst": ids_match(off_ids["burst"]["ids"], on_ids["burst"]["ids"]),
        "trials": {
            mode: [
                {sc: {k: v for k, v in cell[sc].items() if k != "ids"} for sc in cell}
                for cell in cells
            ]
            for mode, cells in per_mode.items()
        },
    }


def print_tables(results: list[dict[str, Any]], args, name: str, can_publish: bool) -> None:
    def m(row, scenario, key):
        c = row[scenario][key]
        return f"{c['median']:.1f} ± {c['ci95']:.1f}"

    print("\n" + "=" * 60)
    print(f"Prefix caching, {args.model}, bf16, {name}")
    print(f"{args.n_seq} sequential requests; burst of {args.burst} x {args.burst_max_tokens} tokens; "
          f"{args.repeats} trials, median ± 95% interval")
    if not can_publish:
        print("These numbers are not the published H100 ladder.")
    print("=" * 60)
    print("\nTime to first token, one request at a time (ms)\n")
    print("| System prompt | Off | On, cold first | On, warm | Speedup | Hit rate (warm) | Expected |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        off, on = r["off"], r["on"]
        speed = off["sequential"]["warm_ttft_ms"]["median"] / on["sequential"]["warm_ttft_ms"]["median"]
        print(
            f"| {r['system_len']} | {m(off, 'sequential', 'warm_ttft_ms')} "
            f"| {m(on, 'sequential', 'cold_first_ttft_ms')} | {m(on, 'sequential', 'warm_ttft_ms')} "
            f"| {speed:.2f}x | {on['sequential']['warm_hit_rate']['median'] * 100:.1f}% "
            f"| {r['expected_hit_rate_sequential'] * 100:.1f}% overall |"
        )
    print(f"\nBurst of {args.burst} requests onto an empty cache\n")
    print("| System prompt | Mean TTFT off | Mean TTFT on | Last TTFT off | Last TTFT on | Peak KV blocks off → on | Hit rate on | Expected |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        off, on = r["off"]["burst"], r["on"]["burst"]
        print(
            f"| {r['system_len']} | {off['ttft_mean_ms']['median']:.1f} | {on['ttft_mean_ms']['median']:.1f} "
            f"| {off['ttft_max_ms']['median']:.1f} | {on['ttft_max_ms']['median']:.1f} "
            f"| {off['peak_blocks']['median']:.0f} → {on['peak_blocks']['median']:.0f} "
            f"| {on['hit_rate']['median'] * 100:.1f}% | {r['expected_hit_rate_burst'] * 100:.1f}% |"
        )
    bad = [r["system_len"] for r in results
           if not (r["greedy_match_sequential"]["match"] and r["greedy_match_burst"]["match"])]
    print("\nGreedy tokens identical with caching on and off: "
          + ("yes at every length" if not bad else f"NO at system length(s) {bad}"))


def parse_args():
    p = argparse.ArgumentParser(description="Prefix caching on/off: TTFT and hit rate")
    p.add_argument("--model", default=PUBLISHED_MODEL)
    p.add_argument("--system-lens", default="128,512,1024,1536",
                   help="Comma-separated shared system prompt lengths in tokens")
    p.add_argument("--suffix-len", type=int, default=64)
    p.add_argument("--n-seq", type=int, default=16, help="Sequential requests per trial")
    p.add_argument("--burst", type=int, default=32, help="Requests submitted together")
    p.add_argument("--burst-max-tokens", type=int, default=32)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="prefix_caching.json")
    return p.parse_args()


def main():
    args = parse_args()
    lens = [int(x) for x in args.system_lens.split(",") if x.strip()]
    if not lens or any(n < 1 for n in lens):
        raise SystemExit("--system-lens must be positive integers")
    if args.repeats < 1 or args.n_seq < 2 or args.burst < 2:
        raise SystemExit("need --repeats >= 1, --n-seq >= 2, --burst >= 2")
    longest = max(lens) + args.suffix_len + args.burst_max_tokens
    from babyvllm.worker.model_runner import CTX_BUCKETS
    if longest > max(CTX_BUCKETS):
        raise SystemExit(f"longest sequence {longest} is past the largest CUDA graph bucket {max(CTX_BUCKETS)}")
    budget = max(PREFILL_TOKEN_CAP, max(lens) + args.suffix_len)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    name = gpu_name()
    can_publish = publishable(name, args.model, "bfloat16") and device.type == "cuda" and args.repeats >= 3
    print(f"Device {device} ({name}) | model {args.model} | publishable={can_publish}", flush=True)

    from babyvllm.config import SchedulerConfig
    from babyvllm.llm import LLM

    torch.manual_seed(0)
    llm = LLM(
        args.model,
        scheduler_config=SchedulerConfig(max_num_seqs=max(args.burst, 1), max_num_batched_tokens=budget),
        device=device,
    )
    engine = llm.engine
    tokenizer = load_tokenizer(args.model)
    base = tokenizer.encode("The quick brown fox jumps over the lazy dog. ", add_special_tokens=False)
    vocab = llm.model_config.vocab_size

    results = [run_length(engine, args, base, vocab, device, n) for n in lens]
    print_tables(results, args, name, can_publish)

    payload = {
        "metadata": {
            "gpu": name,
            "model": args.model,
            "dtype": "bfloat16",
            "publishable": can_publish,
            "block_size": engine.block_size,
            "system_lens": lens,
            "suffix_len": args.suffix_len,
            "n_seq": args.n_seq,
            "burst": args.burst,
            "burst_max_tokens": args.burst_max_tokens,
            "repeats": args.repeats,
            "prefill_budget": budget,
            "pytorch_version": torch.__version__,
            "hf_version": package_version("transformers"),
            "babyvllm_prefill_attn": os.environ.get("BABYVLLM_PREFILL_ATTN", "triton"),
            "command": "python " + " ".join(sys.argv),
            "definition": {
                "warm_ttft_ms": "sequential: median over requests 1..n-1 of wall clock to the first token, one request in flight",
                "cold_first_ttft_ms": "sequential: request 0 on an empty cache",
                "ttft_mean_ms / ttft_max_ms": "burst: mean and max over requests of submit-to-first-token wall clock",
                "hit_rate": "KVCacheManager counter: cached prompt tokens / prompt tokens queried at admission",
                "warm_hit_rate": "sequential hit rate over requests 1..n-1",
                "peak_blocks": "most KV blocks held at the start of any step",
            },
        },
        "lengths": results,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
