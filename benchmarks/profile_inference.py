#!/usr/bin/env python3
"""Prefill vs decode anatomy on a single GPU (RTX 3090 / 24GB).

Isolates prefill and decode, sweeps batch size and context length, and reports
throughput, step time, estimated HBM bandwidth, and a 3090 roofline.

    python benchmarks/profile_inference.py
    python benchmarks/profile_inference.py --quick
    python benchmarks/profile_inference.py --nsys-focus

Nsight Systems (optional, after a smoke run works):

    nsys profile -t cuda,nvtx,osrt -o nsys_prefill_decode \\
      python benchmarks/profile_inference.py --nsys-focus
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

import torch
from transformers import AutoModelForCausalLM


# RTX 3090 GA102, dense tensor-core peak (no sparsity).
RTX_3090_FP16_TFLOPS = 71.0
RTX_3090_HBM_GB_S = 936.0
RTX_3090_RIDGE_FLOP_PER_BYTE = RTX_3090_FP16_TFLOPS * 1e12 / (RTX_3090_HBM_GB_S * 1e9)


@dataclass
class RunResult:
    phase: str
    batch: int
    context: int
    decode_tokens: int
    ok: bool
    skip_reason: str | None = None
    latency_ms: float | None = None
    tok_s: float | None = None
    tpot_ms: float | None = None
    peak_mem_gb: float | None = None
    est_hbm_gb_s: float | None = None
    est_hbm_pct_peak: float | None = None
    power_w: float | None = None
    tokens_per_joule: float | None = None


@dataclass
class SessionInfo:
    gpu_name: str
    gpu_memory_gb: float
    model: str
    dtype: str
    attn: str
    num_params_b: float
    weight_gb: float
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_size: int
    kv_bytes_per_token: int
    kv_kib_per_token: float
    max_kv_tokens_est: int
    ridge_flop_per_byte: float
    decode_memory_bound_until_batch: int
    results: list[RunResult] = field(default_factory=list)


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def gpu_power_w() -> float | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def bytes_per_dtype(dtype: torch.dtype) -> int:
    if dtype in (torch.float16, torch.bfloat16):
        return 2
    if dtype == torch.float32:
        return 4
    raise ValueError(f"unsupported dtype {dtype}")


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def kv_bytes_per_token(cfg, dtype: torch.dtype) -> int:
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return (
        2
        * cfg.num_hidden_layers
        * cfg.num_key_value_heads
        * head_dim
        * bytes_per_dtype(dtype)
    )


def estimate_decode_bytes(weight_bytes: int, kv_bpt: int, batch: int, context: int) -> int:
    """Bytes touched in one decode step: all weights + full KV cache read."""
    return weight_bytes + kv_bpt * batch * context


def estimate_prefill_flops(num_params: int, batch: int, seq: int) -> float:
    return 2.0 * num_params * batch * seq


def estimate_decode_flops(num_params: int, batch: int) -> float:
    return 2.0 * num_params * batch


@torch.inference_mode()
def prefill(model, input_ids: torch.Tensor):
    with torch.cuda.nvtx.range("prefill"):
        out = model(input_ids=input_ids, use_cache=True)
    return out.logits, out.past_key_values


@torch.inference_mode()
def decode_n(
    model,
    batch: int,
    start_len: int,
    n_tokens: int,
    past,
    next_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    for t in range(n_tokens):
        position_ids = torch.full(
            (batch, 1), start_len + t, device=device, dtype=torch.long
        )
        with torch.cuda.nvtx.range("decode"):
            out = model(
                input_ids=next_ids,
                position_ids=position_ids,
                past_key_values=past,
                use_cache=True,
            )
        past = out.past_key_values
        next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return next_ids


def time_cuda(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def load_model(model_name: str, dtype: torch.dtype, attn: str, device: torch.device):
    kwargs = dict(
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    try:
        kwargs["attn_implementation"] = attn
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device)
    model.eval()
    return model


def run_prefill(
    model,
    input_ids: torch.Tensor,
    warmup: int,
    repeats: int,
    num_params: int,
) -> dict:
    def one():
        prefill(model, input_ids)

    for _ in range(warmup):
        one()
    torch.cuda.reset_peak_memory_stats()
    p0 = gpu_power_w()
    elapsed = time_cuda(lambda: [one() for _ in range(repeats)]) / repeats
    p1 = gpu_power_w()
    b, s = input_ids.shape
    tokens = b * s
    power = None if p0 is None or p1 is None else 0.5 * (p0 + p1)
    flops = estimate_prefill_flops(num_params, b, s)
    return {
        "latency_ms": elapsed * 1e3,
        "tok_s": tokens / elapsed,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "power_w": power,
        "tokens_per_joule": (tokens / (power * elapsed)) if power and power > 0 else None,
        "tflops": (flops / elapsed) / 1e12,
    }


def run_decode(
    model,
    input_ids: torch.Tensor,
    decode_tokens: int,
    warmup_tokens: int,
    weight_bytes: int,
    kv_bpt: int,
    num_params: int,
    device: torch.device,
) -> dict:
    batch, ctx = input_ids.shape

    def fresh_cache():
        logits, past = prefill(model, input_ids)
        next_ids = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        return past, next_ids

    past, next_ids = fresh_cache()
    if warmup_tokens:
        decode_n(model, batch, ctx, warmup_tokens, past, next_ids, device)
        del past
        torch.cuda.empty_cache()
        past, next_ids = fresh_cache()

    def one():
        decode_n(model, batch, ctx, decode_tokens, past, next_ids, device)

    torch.cuda.reset_peak_memory_stats()
    p0 = gpu_power_w()
    elapsed = time_cuda(one)
    p1 = gpu_power_w()
    power = None if p0 is None or p1 is None else 0.5 * (p0 + p1)
    tokens = batch * decode_tokens
    bytes_moved = estimate_decode_bytes(weight_bytes, kv_bpt, batch, ctx) * decode_tokens
    hbm_gb_s = (bytes_moved / elapsed) / 1e9
    return {
        "latency_ms": elapsed * 1e3,
        "tok_s": tokens / elapsed,
        "tpot_ms": (elapsed / decode_tokens) * 1e3,
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "est_hbm_gb_s": hbm_gb_s,
        "est_hbm_pct_peak": 100.0 * hbm_gb_s / RTX_3090_HBM_GB_S,
        "power_w": power,
        "tokens_per_joule": (tokens / (power * elapsed)) if power and power > 0 else None,
        "tflops": (estimate_decode_flops(num_params, batch) * decode_tokens / elapsed) / 1e12,
    }


def print_roofline(info: SessionInfo) -> None:
    print("\n========== Roofline (RTX 3090) ==========")
    print(f"GPU:                 {info.gpu_name}")
    print(f"Peak FP16 tensor:    {RTX_3090_FP16_TFLOPS:.0f} TFLOP/s")
    print(f"Peak HBM:            {RTX_3090_HBM_GB_S:.0f} GB/s")
    print(f"Ridge point:         {info.ridge_flop_per_byte:.0f} FLOP/byte")
    print(f"Params:              {info.num_params_b:.2f}B  ({info.weight_gb:.2f} GB weights)")
    print(f"KV bytes/token:      {info.kv_bytes_per_token}  ({info.kv_kib_per_token:.1f} KiB)")
    print(f"KV capacity est:     {info.max_kv_tokens_est:,} tokens in leftover VRAM")
    print(
        f"Decode is memory-bound until batch ≈ {info.decode_memory_bound_until_batch} "
        f"(AI ≈ batch; ridge ≈ {info.ridge_flop_per_byte:.0f})"
    )
    print("Prefill AI ≈ prompt length, so it is compute-bound almost immediately.")
    print("==========================================\n")


def print_table(results: list[RunResult]) -> None:
    cols = (
        f"{'phase':<8} {'B':>3} {'ctx':>5} {'tok/s':>10} {'ms':>9} "
        f"{'TPOT':>8} {'HBM%':>7} {'memGB':>7} {'W':>6}"
    )
    print(cols)
    print("-" * len(cols))
    for r in results:
        if not r.ok:
            print(
                f"{r.phase:<8} {r.batch:>3} {r.context:>5} "
                f"{'SKIP':>10} {r.skip_reason}"
            )
            continue
        print(
            f"{r.phase:<8} {r.batch:>3} {r.context:>5} "
            f"{r.tok_s:10.1f} {r.latency_ms:9.1f} "
            f"{(r.tpot_ms or 0):8.2f} {(r.est_hbm_pct_peak or 0):6.1f}% "
            f"{r.peak_mem_gb:7.2f} {(r.power_w or 0):6.0f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefill/decode profiler for a 24GB GPU")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    p.add_argument("--attn", default="sdpa", help="sdpa | eager | flash_attention_2")
    p.add_argument("--batches", default="1,4,8,16")
    p.add_argument("--contexts", default="128,512,1024,2048")
    p.add_argument("--decode-tokens", type=int, default=64)
    p.add_argument("--warmup-prefill", type=int, default=2)
    p.add_argument("--repeats-prefill", type=int, default=5)
    p.add_argument("--warmup-decode", type=int, default=8)
    p.add_argument("--quick", action="store_true", help="Tiny sweep for a smoke test")
    p.add_argument(
        "--nsys-focus",
        action="store_true",
        help="One prefill + one decode at B=1, ctx=512 (for nsys)",
    )
    p.add_argument("--out", default="profile_results.json")
    p.add_argument(
        "--fallback-model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Used only if 7B fails to load",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("CUDA is required. This script is meant to run on a RunPod GPU pod.", file=sys.stderr)
        return 1

    if args.quick:
        args.batches = "1,4"
        args.contexts = "128,512"
        args.decode_tokens = 16
        args.warmup_prefill = 1
        args.repeats_prefill = 2
        args.warmup_decode = 4
    if args.nsys_focus:
        args.batches = "1"
        args.contexts = "512"
        args.decode_tokens = 32
        args.warmup_prefill = 1
        args.repeats_prefill = 1
        args.warmup_decode = 4

    device = torch.device("cuda")
    dtype = getattr(torch, args.dtype)
    props = torch.cuda.get_device_properties(0)
    gpu_name = torch.cuda.get_device_name(0)
    gpu_gb = props.total_memory / 1e9

    print(f"GPU: {gpu_name}  {gpu_gb:.1f} GB")
    print(f"Model: {args.model}  dtype={args.dtype}  attn={args.attn}")
    print("Loading weights (first run downloads ~15 GB for 7B)...", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_name = args.model
    try:
        model = load_model(model_name, dtype, args.attn, device)
    except torch.cuda.OutOfMemoryError:
        print(f"{model_name} did not fit. Falling back to {args.fallback_model}.", flush=True)
        torch.cuda.empty_cache()
        model_name = args.fallback_model
        model = load_model(model_name, dtype, args.attn, device)

    cfg = model.config
    num_params = count_params(model)
    weight_bytes = num_params * bytes_per_dtype(dtype)
    kv_bpt = kv_bytes_per_token(cfg, dtype)
    allocated = torch.cuda.memory_allocated()
    leftover = props.total_memory * 0.90 - allocated
    max_kv_tokens = max(0, int(leftover / kv_bpt))

    info = SessionInfo(
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_gb,
        model=model_name,
        dtype=args.dtype,
        attn=args.attn,
        num_params_b=num_params / 1e9,
        weight_gb=weight_bytes / 1e9,
        n_layers=cfg.num_hidden_layers,
        n_heads=cfg.num_attention_heads,
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        hidden_size=cfg.hidden_size,
        kv_bytes_per_token=kv_bpt,
        kv_kib_per_token=kv_bpt / 1024,
        max_kv_tokens_est=max_kv_tokens,
        ridge_flop_per_byte=RTX_3090_RIDGE_FLOP_PER_BYTE,
        decode_memory_bound_until_batch=max(1, round(RTX_3090_RIDGE_FLOP_PER_BYTE)),
    )
    print_roofline(info)

    vocab = min(cfg.vocab_size, 32000)
    batches = parse_int_list(args.batches)
    contexts = parse_int_list(args.contexts)

    for ctx in contexts:
        for batch in batches:
            print(f"\n--- batch={batch}  context={ctx} ---", flush=True)
            torch.cuda.empty_cache()
            try:
                input_ids = torch.randint(1, vocab, (batch, ctx), device=device)

                with torch.cuda.nvtx.range(f"bench_prefill_b{batch}_c{ctx}"):
                    pre = run_prefill(
                        model,
                        input_ids,
                        warmup=args.warmup_prefill,
                        repeats=args.repeats_prefill,
                        num_params=num_params,
                    )
                info.results.append(
                    RunResult(
                        phase="prefill",
                        batch=batch,
                        context=ctx,
                        decode_tokens=0,
                        ok=True,
                        latency_ms=pre["latency_ms"],
                        tok_s=pre["tok_s"],
                        peak_mem_gb=pre["peak_mem_gb"],
                        power_w=pre["power_w"],
                        tokens_per_joule=pre["tokens_per_joule"],
                    )
                )
                print(
                    f"  prefill  {pre['tok_s']:.1f} tok/s  "
                    f"{pre['latency_ms']:.1f} ms  "
                    f"{pre['tflops']:.1f} TFLOP/s  "
                    f"mem {pre['peak_mem_gb']:.2f} GB",
                    flush=True,
                )

                with torch.cuda.nvtx.range(f"bench_decode_b{batch}_c{ctx}"):
                    dec = run_decode(
                        model,
                        input_ids,
                        decode_tokens=args.decode_tokens,
                        warmup_tokens=args.warmup_decode,
                        weight_bytes=weight_bytes,
                        kv_bpt=kv_bpt,
                        num_params=num_params,
                        device=device,
                    )
                info.results.append(
                    RunResult(
                        phase="decode",
                        batch=batch,
                        context=ctx,
                        decode_tokens=args.decode_tokens,
                        ok=True,
                        latency_ms=dec["latency_ms"],
                        tok_s=dec["tok_s"],
                        tpot_ms=dec["tpot_ms"],
                        peak_mem_gb=dec["peak_mem_gb"],
                        est_hbm_gb_s=dec["est_hbm_gb_s"],
                        est_hbm_pct_peak=dec["est_hbm_pct_peak"],
                        power_w=dec["power_w"],
                        tokens_per_joule=dec["tokens_per_joule"],
                    )
                )
                print(
                    f"  decode   {dec['tok_s']:.1f} tok/s  "
                    f"TPOT {dec['tpot_ms']:.2f} ms  "
                    f"HBM {dec['est_hbm_pct_peak']:.1f}% of 936 GB/s  "
                    f"mem {dec['peak_mem_gb']:.2f} GB",
                    flush=True,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("  OOM — skipped", flush=True)
                info.results.append(
                    RunResult(
                        phase="both",
                        batch=batch,
                        context=ctx,
                        decode_tokens=args.decode_tokens,
                        ok=False,
                        skip_reason="oom",
                    )
                )

    print("\n========== Results ==========")
    print_table(info.results)

    decode_rows = [r for r in info.results if r.ok and r.phase == "decode"]
    if decode_rows:
        best = max(decode_rows, key=lambda r: r.tok_s or 0)
        low_b = min(decode_rows, key=lambda r: r.batch)
        print(
            f"\nHeadline: decode at B={best.batch}, ctx={best.context} "
            f"reached {best.tok_s:.0f} tok/s "
            f"({best.est_hbm_pct_peak:.0f}% of 3090 HBM peak)."
        )
        print(
            f"Batch-1 decode TPOT {low_b.tpot_ms:.2f} ms, "
            f"estimated {low_b.est_hbm_pct_peak:.0f}% of peak HBM. "
            "If that number is well below 50%, the gap is CPU/kernel-launch overhead, not FLOPs."
        )

    payload = asdict(info)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {os.path.abspath(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
