#!/usr/bin/env python3
"""Decode step time for Day 6 Triton paged attention.

    python benchmarks/bench_triton_decode.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from babyvllm.config import SchedulerConfig
from babyvllm.llm import LLM
from babyvllm.sequence import SamplingParams


MODEL = os.environ.get("BABYVLLM_MODEL", "/root/huggingface/Qwen2-0.5B")
POINTS = ((1, 128), (1, 2048), (64, 2048))
GEN = 16


def prompt_ids(tokenizer, length: int) -> list[int]:
    filler = tokenizer.encode("The quick brown fox jumps over the lazy dog. ")
    if not filler:
        filler = [1]
    out: list[int] = []
    while len(out) < length:
        out.extend(filler)
    return out[:length]


def time_point(llm: LLM, batch: int, ctx: int) -> None:
    tok = llm.tokenizer
    prompts = [prompt_ids(tok, ctx) for _ in range(batch)]
    llm.engine.reset()
    llm.engine.scheduler.max_num_seqs = batch
    device = llm.model_runner.device
    for ids in prompts:
        llm.engine.add_request(
            ids,
            SamplingParams(temperature=0.0, max_tokens=GEN, ignore_eos=True),
        )

    # First step is prefill (and graph capture). Throw it away.
    if device.type == "cuda":
        torch.cuda.synchronize()
    llm.engine.step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    while llm.engine.has_unfinished_requests():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.engine.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

    if not times:
        print(f"B={batch:<2} ctx={ctx:<4}  no decode steps", flush=True)
        return
    times.sort()
    p50 = times[len(times) // 2]
    mean = sum(times) / len(times)
    print(
        f"B={batch:<2} ctx={ctx:<4}  decode_steps={len(times):<3}  "
        f"p50={p50:.2f} ms  mean={mean:.2f} ms  min={times[0]:.2f} ms",
        flush=True,
    )


def main() -> int:
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    try:
        from babyvllm.kernels.paged_attn import paged_decode_attention  # noqa: F401
        print("kernel: Triton paged_decode_attention", flush=True)
    except ImportError:
        print("kernel: MISSING — still on Day 5 SDPA path", flush=True)

    sched = SchedulerConfig(max_num_seqs=64, max_num_batched_tokens=131072)
    print(f"Loading {MODEL} ...", flush=True)
    llm = LLM(MODEL, scheduler_config=sched, verbose=True)
    print("========== decode step times ==========", flush=True)
    for batch, ctx in POINTS:
        time_point(llm, batch, ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
