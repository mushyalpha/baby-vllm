#!/usr/bin/env python3
"""Paged decode attention: single-pass kernel vs flash-decoding splits.

Model dims come from the checkpoint's config.json (BABYVLLM_MODEL, default
Qwen2.5-7B). Every row of a batch is at the full context, and the block table
is the width of the 2048 graph bucket. One CUDA graph replays the call once
per decoder layer, each layer on its own KV cache as in the model, so the KV
does not stay resident in L2 between calls. Reports per-layer time, the
per-forward estimate (× num_layers), achieved KV bandwidth, and the max abs
difference from the single-pass kernel.

Usage, on the GPU:
    python benchmarks/bench_decode_attn.py [--out decode_attn_bench.json]
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
from babyvllm.kernels.decode_split import (
    DECODE_PARTITION_ENV,
    DEFAULT_PARTITION_BLOCKS,
    decode_split_plan,
)
from babyvllm.kernels.paged_attn import paged_decode_attention

BATCHES = [1, 2, 4, 8, 16, 32, 64]
CONTEXTS = [256, 2048]
# 0 is the single-pass kernel; "auto" is what the engine runs.
PARTITIONS = [0, "auto", 2, 4, 8, 16, 32]


def cdiv(a, b):
    return -(-a // b)


def build(batch, ctx, num_layers, num_heads, num_kv_heads, head_dim, block_size, width):
    blocks = cdiv(ctx, block_size)
    total = batch * blocks
    caches, tables = [], []
    for _ in range(num_layers):
        caches.append(torch.randn(2, total, block_size, num_kv_heads, head_dim,
                                  device="cuda", dtype=torch.bfloat16))
        bt = torch.zeros(batch, width, dtype=torch.int32, device="cuda")
        bt[:, :blocks] = torch.randperm(total, device="cuda").view(batch, blocks).to(torch.int32)
        tables.append(bt)
    q = torch.randn(batch, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    seq_lens = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")
    return q, caches, tables, seq_lens


def time_graph(fn, warmup, iters):
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    del graph
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="decode_attn_bench.json")
    args = ap.parse_args()

    os.environ.pop(DECODE_PARTITION_ENV, None)
    cfg = load_model_config()
    num_heads, num_kv_heads, head_dim = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    layers, block_size = cfg.num_hidden_layers, cfg.block_size
    width = cdiv(max(CONTEXTS), block_size) + 1
    scale = head_dim ** -0.5
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"GPU {torch.cuda.get_device_name(0)}, {sms} SMs, torch {torch.__version__}", flush=True)
    print(f"{num_heads} q heads, {num_kv_heads} kv heads, head dim {head_dim}, {layers} layers, "
          f"block {block_size}, default partition {DEFAULT_PARTITION_BLOCKS} blocks", flush=True)

    rows = []
    for ctx in CONTEXTS:
        for batch in BATCHES:
            q, caches, tables, seq_lens = build(
                batch, ctx, layers, num_heads, num_kv_heads, head_dim, block_size, width,
            )
            kv_bytes = batch * ctx * num_kv_heads * head_dim * 2 * 2
            ref = paged_decode_attention(q, caches[0], tables[0], seq_lens, scale,
                                         max_seq_len=ctx, partition_blocks=0).float()
            base_us = None
            for part in PARTITIONS:
                pb = None if part == "auto" else part
                splits, per = decode_split_plan(batch, num_kv_heads, block_size, width, ctx, sms, pb)

                def fwd():
                    for i in range(layers):
                        paged_decode_attention(q, caches[i], tables[i], seq_lens, scale,
                                               max_seq_len=ctx, partition_blocks=pb)

                us = time_graph(fwd, args.warmup, args.iters) * 1e3 / layers
                out = paged_decode_attention(q, caches[0], tables[0], seq_lens, scale,
                                             max_seq_len=ctx, partition_blocks=pb)
                diff = float((out.float() - ref).abs().max())
                if part == 0:
                    base_us = us
                row = {
                    "batch": batch,
                    "context": ctx,
                    "partition": part,
                    "num_splits": splits,
                    "blocks_per_split": per,
                    "ctas": batch * num_kv_heads * splits,
                    "per_layer_us": us,
                    "forward_ms": us * layers / 1e3,
                    "kv_gb_s": kv_bytes / (us * 1e-6) / 1e9,
                    "speedup_vs_single_pass": base_us / us,
                    "max_abs_diff_vs_single_pass": diff,
                }
                rows.append(row)
                print(
                    f"b{batch:<3d} ctx {ctx:<5d} partition {str(part):>4s} → {splits:>3d} splits "
                    f"{row['ctas']:>5d} CTAs | {us:8.2f} µs/layer {row['forward_ms']:6.3f} ms/fwd "
                    f"{row['kv_gb_s']:7.0f} GB/s | {row['speedup_vs_single_pass']:5.2f}x diff {diff:.2g}",
                    flush=True,
                )
            del q, caches, tables, seq_lens, ref
            torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump({
            "gpu": torch.cuda.get_device_name(0),
            "sms": sms,
            "torch": torch.__version__,
            "model": os.environ.get("BABYVLLM_MODEL", "Qwen/Qwen2.5-7B"),
            "default_partition_blocks": DEFAULT_PARTITION_BLOCKS,
            "rows": rows,
        }, f, indent=2)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
