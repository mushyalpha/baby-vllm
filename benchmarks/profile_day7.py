#!/usr/bin/env python3
"""Day 7 profiling harness: measure where the 8.21 ms goes.

This script drives baby-vLLM through steady-state decode steps (graphs + Triton)
and emits NVTX markers so nsys can bucket kernel time.

Usage:
  1. Timing only (no nsys):
       python benchmarks/profile_day7.py --model Qwen/Qwen2.5-7B

  2. Full nsys capture with CUDA-graph-node visibility:
       nsys profile --cuda-graph-trace=node \
         -t cuda,nvtx \
         -o day7_b1_ctx128 \
         python benchmarks/profile_day7.py --model Qwen/Qwen2.5-7B

  3. Parse the .sqlite from nsys:
       python benchmarks/profile_day7.py --parse day7_b1_ctx128.sqlite
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch


# ---------------------------------------------------------------------------
# Kernel classifier
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Parse an nsys .sqlite export
# ---------------------------------------------------------------------------

def parse_nsys_sqlite(path: str) -> dict:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found. Export with: nsys export -t sqlite {path.replace('.sqlite', '.nsys-rep')}")
        sys.exit(1)

    conn = sqlite3.connect(path)

    # 1. Fetch NVTX decode step windows
    windows = []
    try:
        windows = conn.execute("""
            SELECT n.start, n.end
            FROM NVTX_EVENTS n
            JOIN StringIds s ON n.textId = s.id
            WHERE s.value LIKE 'decode_step_%'
        """).fetchall()
    except sqlite3.OperationalError:
        print("Warning: Could not read NVTX_EVENTS. Schema might differ.")

    # 2. Fetch kernels (JOIN with StringIds to get actual names)
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

    # Fetch Memcpys
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

    # Fetch Memsets
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

    # 3. Filter kernels to only those inside the NVTX measurement windows
    # and compute idle/gap times
    measured_rows = []
    num_steps = len(windows)
    
    pre_launch_ms = 0.0
    in_graph_idle_ms = 0.0
    tail_ms = 0.0

    if windows:
        rows.sort(key=lambda x: x[1])
        for w_start, w_end in windows:
            # find kernels in this window
            w_kernels = [r for r in rows if r[1] >= w_start and r[2] <= w_end]
            if not w_kernels:
                continue
                
            w_kernels.sort(key=lambda x: x[1])
            
            # accumulate measured rows for category breakdown
            for r in w_kernels:
                measured_rows.append((r[0], r[2] - r[1]))
                
            pre_launch_ms += (w_kernels[0][1] - w_start) / 1e6
            tail_ms += (w_end - w_kernels[-1][2]) / 1e6
            
            # calculate gaps between consecutive kernels
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

    # 4. Bucket and average per step
    buckets: dict[str, list[float]] = defaultdict(list)
    for name, dur_ns in measured_rows:
        cat = classify_kernel(name or "")
        buckets[cat].append(dur_ns / 1e6)  # ms

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
    wall_ms = total_ms + avg_pre_launch + avg_in_graph_idle + avg_tail

    print(f"\n{'='*65}")
    print(f"Idle / CPU Time (Avg per step)")
    print(f"{'='*65}")
    print(f"pre-launch:    {avg_pre_launch:>6.2f} ms  (CPU scheduling before first kernel)")
    print(f"in-graph gaps: {avg_in_graph_idle:>6.2f} ms  (idle time between GPU kernels)")
    print(f"tail:          {avg_tail:>6.2f} ms  (sampling readback, detokenize, Python)")
    print(f"{'-'*65}")
    print(f"WALL TIME:     {wall_ms:>6.2f} ms")

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

    return {
        "measured_steps": num_steps,
        "avg_kernels_per_step": avg_kernels_per_step,
        "avg_active_ms_per_step": total_ms,
        "avg_wall_ms_per_step": wall_ms,
        "idle_ms": {
            "pre_launch": avg_pre_launch,
            "in_graph_idle": avg_in_graph_idle,
            "tail": avg_tail,
        },
        "breakdown_per_step": {
            cat: {"count": len(times)/num_steps, "ms": sum(times)/num_steps}
            for cat, times in buckets.items()
        },
    }


# ---------------------------------------------------------------------------
# Steady-state decode driver
# ---------------------------------------------------------------------------

def run_steady_decode(model_name: str, batch: int, context: int, decode_steps: int, warmup_steps: int) -> dict:
    from babyvllm.config import SchedulerConfig
    from babyvllm.llm import LLM
    from babyvllm.sequence import SamplingParams

    sched = SchedulerConfig(
        max_num_seqs=max(batch, 256),
        max_num_batched_tokens=max(batch * context, 8192),
    )
    
    llm = LLM(model_name, scheduler_config=sched, verbose=True)
    tokenizer = llm.tokenizer
    
    filler = tokenizer.encode("The quick brown fox jumps over the lazy dog. ") or [1]
    prompt_ids = (filler * ((context // len(filler)) + 1))[:context]
    
    total_gen = warmup_steps + decode_steps + 5
    for _ in range(batch):
        llm.engine.add_request(
            list(prompt_ids),
            SamplingParams(temperature=0.0, max_tokens=total_gen, ignore_eos=True),
        )
    
    # Warmup
    prefill_done = False
    warmup_done = 0
    while warmup_done < warmup_steps:
        llm.engine.step()
        if not prefill_done:
            prefill_done = True
            continue
        warmup_done += 1
    
    # Measurement
    step_times = []
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"MEASURE_B{batch}_CTX{context}")
    
    for i in range(decode_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        torch.cuda.nvtx.range_push(f"decode_step_{i}")
        llm.engine.step()
        torch.cuda.synchronize()  # Wait for all kernels to finish
        torch.cuda.nvtx.range_pop() # Now close the window
        
        t1 = time.perf_counter()
        step_times.append((t1 - t0) * 1e3)
    
    torch.cuda.nvtx.range_pop()
    
    step_times.sort()
    n = len(step_times)
    p50 = step_times[n // 2]
    
    result = {
        "model": model_name, "batch": batch, "context": context,
        "decode_steps": decode_steps, "warmup_steps": warmup_steps,
        "step_ms_mean": sum(step_times) / n, "step_ms_p50": p50,
        "tok_s": batch / (p50 / 1e3),
    }
    
    print(f"\nDecode Step Time — B={batch}, ctx≈{context} | p50: {p50:.2f} ms")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--decode-steps", type=int, default=50)
    p.add_argument("--warmup-steps", type=int, default=20)
    p.add_argument("--parse", type=str, default=None)
    p.add_argument("--out", default="day7_profile.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.parse:
        result = parse_nsys_sqlite(args.parse)
    else:
        result = run_steady_decode(args.model, args.batch, args.context, args.decode_steps, args.warmup_steps)
    
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
