#!/usr/bin/env python3
"""One attention layer, multi-token steps: per-sequence torch SDPA loop vs the Triton paged prefill kernel.

Model dims come from the checkpoint's config.json (BABYVLLM_MODEL, default
Qwen2.5-7B). Each shape is a list of (cached context, new query tokens).
Reports median time per layer call over CUDA events, the per-forward estimate
(× num_layers), peak extra memory, and the max abs difference between paths.

Usage, on the GPU:
    python benchmarks/bench_prefill_attn.py [--out prefill_attn_bench.json]
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch

from babyvllm.config import load_model_config
from babyvllm.kernels.paged_prefill import paged_prefill_attention
from babyvllm.layers.attention import _paged_attention_torch
from babyvllm.worker.context import AttentionMetadata

SHAPES = {
    "b64 x 128 fresh prefill": [(0, 128)] * 64,
    "b1 x 2048 fresh prefill": [(0, 2048)],
    "b8 x (1024 cached + 128)": [(1024, 128)] * 8,
    "b1 x (2048 cached + 4096)": [(2048, 4096)],
    "1 x 512 prefill + 63 decodes at 1000": [(0, 512)] + [(1000, 1)] * 63,
}


def build(specs, num_heads, num_kv_heads, head_dim, block_size):
    seq_lens = [c + q for c, q in specs]
    q_lens = [q for _, q in specs]
    blocks = [(s + block_size - 1) // block_size for s in seq_lens]
    total = sum(blocks)
    perm = torch.randperm(total).tolist()
    tables, i = [], 0
    for n in blocks:
        tables.append(perm[i:i + n] + [0] * (max(blocks) - n))
        i += n
    kv_cache = torch.randn(2, total, block_size, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    q = torch.randn(sum(q_lens), num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    qsl = [0]
    for n in q_lens:
        qsl.append(qsl[-1] + n)
    ctx = [c for c, _ in specs]
    md = AttentionMetadata(
        slot_mapping=torch.zeros(sum(q_lens), dtype=torch.int64, device="cuda"),
        block_tables=torch.tensor(tables, dtype=torch.int32, device="cuda"),
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device="cuda"),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device="cuda"),
        context_lens=torch.tensor(ctx, dtype=torch.int32, device="cuda"),
        query_start_loc_cpu=qsl,
        seq_lens_cpu=seq_lens,
        context_lens_cpu=ctx,
        max_query_len=max(q_lens),
        max_seq_len=max(seq_lens),
    )
    return q, kv_cache, md


def time_fn(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def peak_extra_bytes(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    try:
        fn()
    except torch.cuda.OutOfMemoryError:
        return None
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default="prefill_attn_bench.json")
    args = ap.parse_args()

    cfg = load_model_config()
    num_heads, num_kv_heads, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    q_per_kv = num_heads // num_kv_heads
    scale = head_dim ** -0.5
    print(f"GPU {torch.cuda.get_device_name(0)}, torch {torch.__version__}", flush=True)
    print(f"{num_heads} q heads, {num_kv_heads} kv heads, head dim {head_dim}, {cfg.num_hidden_layers} layers", flush=True)

    rows = []
    for name, specs in SHAPES.items():
        q, kv_cache, md = build(specs, num_heads, num_kv_heads, head_dim, cfg.block_size)
        old = lambda: _paged_attention_torch(q, kv_cache, md, scale, q_per_kv)
        new = lambda: paged_prefill_attention(q, kv_cache, md, scale)

        old_mem = peak_extra_bytes(old)
        new_mem = peak_extra_bytes(new)
        new_ms = time_fn(new, args.warmup, args.iters)
        if old_mem is None:
            old_ms, diff = None, None
        else:
            old_ms = time_fn(old, args.warmup, args.iters)
            diff = float((old().float() - new().float()).abs().max())
        row = {
            "shape": name,
            "query_tokens": sum(qn for _, qn in specs),
            "torch_ms": old_ms,
            "triton_ms": new_ms,
            "speedup": None if old_ms is None else old_ms / new_ms,
            "torch_forward_ms": None if old_ms is None else old_ms * cfg.num_hidden_layers,
            "triton_forward_ms": new_ms * cfg.num_hidden_layers,
            "torch_peak_extra_mib": None if old_mem is None else old_mem / 2**20,
            "triton_peak_extra_mib": new_mem / 2**20,
            "max_abs_diff": diff,
        }
        rows.append(row)
        old_s = "OOM" if old_ms is None else f"{old_ms:.3f} ms"
        old_m = "OOM" if old_mem is None else f"{old_mem / 2**20:.0f} MiB"
        print(
            f"{name:40s} torch {old_s:>11s} {old_m:>9s} | triton {new_ms:.3f} ms {new_mem / 2**20:.0f} MiB"
            + ("" if old_ms is None else f" | {old_ms / new_ms:.1f}x, diff {diff:.3g}"),
            flush=True,
        )
        del q, kv_cache, md
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "model": os.environ.get("BABYVLLM_MODEL", "Qwen/Qwen2.5-7B"),
            "rows": rows,
        }, f, indent=2)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
