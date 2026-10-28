#!/usr/bin/env python3
"""Offline throughput and time-to-first-token for three engines on one GPU.

HuggingFace Transformers (one batched generate), baby-vLLM, and vLLM's
offline LLM.generate. No HTTP server. The published comparison is
Qwen2.5-7B, bf16, H100. Any other GPU or model is written with
publishable=false.

What this script forces, because the defaults would not be a comparison:

- Exact token ids, one distinct first token per request. Re-decoding text
  does not preserve prompt length, and identical prompts are a prefix-cache hit.
- vLLM prefix caching off. Current vLLM defaults it on.
- The same prefill token budget on both engines, capped at 8192. A 64 x 1920
  eager prefill asks for an 8.67 GiB activation after the KV cache is allocated.
- CUDA graphs for the batches in the sweep captured before the timer.
  baby-vLLM otherwise captures on the first decode step.
- Greedy decoding that actually emits max_tokens (HF stops on EOS unless
  min_new_tokens is set; vLLM and baby-vLLM ignore EOS).
- Three timed trials after one warmup. The reported point is the median.
  The 95% interval is 1.96 * sample standard deviation / sqrt(n).
- vLLM chunked prefill left at the engine default. The token budget holds
  this grid's largest prefill, so the default scheduler does not split it.
  Forcing the flag off makes vLLM warn that Qwen2 outputs may be wrong.
- Greedy ids from the first trial, compared with HuggingFace.

TTFT is the wall clock of a warm 1-token generate. Throughput is new tokens
divided by the wall clock of a warm generate of --max-tokens, prefill included.
That is not the Day 7 steady-state decode rate.
"""

from __future__ import annotations

import argparse
import copy
import gc
import inspect
import json
import os
import statistics
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

# 64 x 1920 scheduled as one eager prefill materialises an 8.67 GiB gate_up
# activation. Decode never needs that. Keep every step under this cap.
PREFILL_TOKEN_CAP = 8192
PUBLISHED_MODEL = "Qwen/Qwen2.5-7B"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def publishable(gpu: str, model: str, dtype: str) -> bool:
    """The only number that can go on the H100 ladder."""
    return (
        "H100" in gpu.upper()
        and model.rstrip("/").endswith("Qwen2.5-7B")
        and dtype == "bfloat16"
    )


def prefill_budget(batch_sizes: list[int], prompt_len: int, cap: int = PREFILL_TOKEN_CAP) -> int:
    if prompt_len < 1:
        raise ValueError("prompt_len must be positive")
    if not batch_sizes:
        raise ValueError("batch_sizes is empty")
    if prompt_len > cap:
        raise ValueError(
            f"prompt_len {prompt_len} is above the prefill cap {cap}. "
            "baby-vLLM rejects a request longer than max_num_batched_tokens."
        )
    need = max(batch_sizes) * prompt_len
    return min(max(need, prompt_len), cap)


def make_prompt_token_ids(
    base: list[int],
    n: int,
    prompt_len: int,
    vocab_size: int,
) -> list[list[int]]:
    """n prompts of exactly prompt_len tokens, with distinct prefixes.

    Token 0 carries the request index. A shared tail is not a prefix, so
    automatic prefix caching has nothing to reuse even if a flag is ignored.
    """
    if n < 1 or prompt_len < 1:
        raise ValueError("n and prompt_len must be positive")
    if vocab_size < 12:
        raise ValueError("vocab_size is too small to build distinct prompts")
    if not base:
        base = [10]
    tiled = (list(base) * (prompt_len // len(base) + 1))[:prompt_len]
    span = min(vocab_size, 1000) - 11
    prompts: list[list[int]] = []
    for i in range(n):
        ids = list(tiled)
        ids[0] = 10 + (i % span)
        if n > span and prompt_len > 1:
            ids[1] = 10 + ((i // span) % span)
        prompts.append(ids)
    return prompts


def graph_keys(
    batch_sizes: list[int],
    prompt_len: int,
    max_tokens: int,
    batch_buckets: list[int],
    ctx_buckets: list[int],
) -> tuple[list[tuple[int, int]], list[int]]:
    """(batch bucket, context bucket) pairs decode will replay, largest first.

    The second list is batch sizes that sit above every graph bucket and will
    run eager. Lengths are the sequence length on the first decode step and
    on the last, so a run that crosses 256 captures both context buckets
    before the timer starts.
    """

    def ceil(value: int, buckets: list[int]) -> int | None:
        for bucket in buckets:
            if value <= bucket:
                return bucket
        return None

    first = prompt_len + 1
    last = prompt_len + max_tokens
    ctxs: list[int] = []
    for length in (first, last):
        ctx = ceil(length, ctx_buckets)
        if ctx is None:
            raise ValueError(
                f"sequence length {length} is past the CUDA graph context "
                f"buckets {ctx_buckets}. Decode would leave the graph."
            )
        if ctx not in ctxs:
            ctxs.append(ctx)

    keys: list[tuple[int, int]] = []
    eager: list[int] = []
    for batch in batch_sizes:
        bb = ceil(batch, batch_buckets)
        if bb is None:
            eager.append(batch)
            continue
        for ctx in ctxs:
            keys.append((bb, ctx))
    # Largest graph first so the shared pool is sized for it.
    keys.sort(key=lambda key: (key[1], key[0]), reverse=True)
    uniq: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for key in keys:
        if key not in seen:
            seen.add(key)
            uniq.append(key)
    return uniq, eager


def decode_tok_s(
    generated: int,
    batch: int,
    elapsed_s: float,
    ttft_s: float,
    tokens_at_ttft: int,
) -> float | None:
    """Steady decode rate once every request has its first token.

    Only defined when the prefill step emitted exactly one token per request.
    """
    if tokens_at_ttft != batch:
        return None
    decode_s = elapsed_s - ttft_s
    decode_tokens = generated - batch
    if decode_s <= 0 or decode_tokens < 0:
        return None
    return decode_tokens / decode_s


def summarize_trials(values: list[float]) -> dict[str, float]:
    """Median, and a normal 95% interval from the sample standard deviation."""
    n = len(values)
    if n == 0:
        raise ValueError("no trials")
    median = float(statistics.median(values))
    mean = float(statistics.fmean(values))
    if n < 2:
        std = 0.0
        ci95 = 0.0
    else:
        std = float(statistics.stdev(values))
        ci95 = 1.96 * std / (n ** 0.5)
    return {"n": n, "median": median, "mean": mean, "std": std, "ci95": ci95}


def compare_token_ids(
    reference: list[list[int]] | None,
    other: list[list[int]] | None,
) -> dict[str, Any]:
    """Greedy-id check. Position is the first token that differs, 0-based."""
    if not reference or not other:
        return {"match": False, "reason": "missing"}
    if len(reference) != len(other):
        return {
            "match": False,
            "reason": "batch_size",
            "reference_sequences": len(reference),
            "other_sequences": len(other),
        }
    mismatched = 0
    first: dict[str, int] | None = None
    for i, (left, right) in enumerate(zip(reference, other)):
        left_ids = [int(token) for token in left]
        right_ids = [int(token) for token in right]
        if left_ids == right_ids:
            continue
        mismatched += 1
        if first is None:
            pos = next((k for k, (a, b) in enumerate(zip(left_ids, right_ids)) if a != b), None)
            if pos is None:
                pos = min(len(left_ids), len(right_ids))
            first = {"sequence": i, "position": pos}
    return {
        "match": mismatched == 0,
        "sequences": len(reference),
        "mismatched_sequences": mismatched,
        "first_mismatch": first,
    }


def pack_trials(
    trials: list[dict[str, Any]],
    greedy_ids: list[list[int]],
    stable: bool,
) -> dict[str, Any]:
    tok = summarize_trials([t["tok_s"] for t in trials])
    ttft = summarize_trials([t["ttft_ms"] for t in trials])
    row: dict[str, Any] = {
        "tok_s": round(tok["median"], 2),
        "tok_s_ci95": round(tok["ci95"], 2),
        "ttft_ms": round(ttft["median"], 2),
        "ttft_ms_ci95": round(ttft["ci95"], 2),
        "elapsed_s": round(float(statistics.median([t["elapsed_s"] for t in trials])), 4),
        "total_tokens": trials[0]["total_tokens"],
        "n": tok["n"],
        "trials": trials,
        "greedy_ids": greedy_ids,
        "greedy_ids_stable": stable,
    }
    decode_vals = [t["decode_tok_s"] for t in trials if isinstance(t.get("decode_tok_s"), (int, float))]
    if len(decode_vals) == len(trials):
        dec = summarize_trials(decode_vals)
        row["decode_tok_s"] = round(dec["median"], 2)
        row["decode_tok_s_ci95"] = round(dec["ci95"], 2)
    return row


def vllm_engine_kwargs(
    parameters: set[str],
    *,
    model: str,
    dtype: str,
    max_batch: int,
    max_model_len: int,
    batched_tokens: int,
    forwards_kwargs: bool = False,
) -> dict[str, Any]:
    """Kwargs for vLLM's offline LLM.

    Current vLLM takes enable_prefix_caching through ``**kwargs`` and defaults
    it to True. A filter that keeps only the explicit signature would drop the
    flag and time a cached prefill. When the constructor does not forward
    unknown names, a missing fairness flag is a hard stop.

    Chunked prefill is not passed. vLLM keeps its default. Forcing it off
    makes current vLLM warn that Qwen2 outputs may be wrong, and this grid's
    prefill already fits in ``batched_tokens``.
    """
    wanted: dict[str, Any] = {
        "model": model,
        "dtype": dtype,
        "enable_prefix_caching": False,
        "gpu_memory_utilization": 0.9,
        "max_model_len": max_model_len,
        "max_num_seqs": max_batch,
        "max_num_batched_tokens": batched_tokens,
        "enforce_eager": False,
        "trust_remote_code": True,
        "tensor_parallel_size": 1,
        "seed": 0,
    }
    if forwards_kwargs:
        return wanted
    required = ("enable_prefix_caching", "enforce_eager", "gpu_memory_utilization")
    missing = [name for name in required if name not in parameters]
    if missing:
        raise RuntimeError(
            "This vLLM build has no "
            + ", ".join(missing)
            + " argument. Refusing to run a comparison that might cache prompts "
            "or disable CUDA graphs."
        )
    return {name: value for name, value in wanted.items() if name in parameters}


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "Apple Silicon (MPS)"
    return "CPU"


def package_version(name: str) -> str:
    try:
        module = __import__(name)
    except Exception:
        return "not installed"
    return getattr(module, "__version__", "unknown")


def release(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _cell(tok_s: float, ttft_ms: float, elapsed_s: float, total_tokens: int, **extra: Any) -> dict[str, Any]:
    row = {
        "tok_s": round(tok_s, 2),
        "ttft_ms": round(ttft_ms, 2),
        "elapsed_s": round(elapsed_s, 4),
        "total_tokens": total_tokens,
    }
    row.update(extra)
    return row


def _oom_cell(exc: BaseException) -> dict[str, Any]:
    text = str(exc).strip().splitlines()
    return {"error": "oom", "message": text[-1] if text else "out of memory"}


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prompt_ids_for(tokenizer, n: int, prompt_len: int) -> list[list[int]]:
    base = tokenizer.encode(
        "The quick brown fox jumps over the lazy dog. ",
        add_special_tokens=False,
    )
    vocab = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab < 12:
        vocab = max((max(base) + 1 if base else 12), 12)
    return make_prompt_token_ids(base, n, prompt_len, vocab)


def time_call(device: torch.device, fn):
    sync(device)
    start = time.perf_counter()
    out = fn()
    sync(device)
    return out, time.perf_counter() - start


def run_benchmark_hf(
    model_name: str,
    dtype: torch.dtype,
    device: torch.device,
    batch_sizes: list[int],
    prompt_len: int,
    max_tokens: int,
    tokenizer,
    repeats: int,
) -> dict[int, dict[str, Any]]:
    print("\n" + "=" * 60)
    print(" [1/3] HuggingFace Transformers, one batched generate, SDPA")
    print("=" * 60, flush=True)

    from transformers import AutoModelForCausalLM

    last_exc: Exception | None = None
    hf_model = None
    for extra in (
        {"dtype": dtype, "attn_implementation": "sdpa"},
        {"torch_dtype": dtype, "attn_implementation": "sdpa"},
        {"dtype": dtype},
        {"torch_dtype": dtype},
    ):
        load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True, **extra}
        if device.type == "cuda":
            load_kwargs["device_map"] = {"": 0 if device.index is None else device.index}
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            break
        except TypeError as exc:
            last_exc = exc
    if hf_model is None:
        raise RuntimeError("Could not load the HuggingFace model") from last_exc
    if device.type != "cuda":
        hf_model = hf_model.to(device)
    hf_model.eval()
    # min_new_tokens stops EOS from ending the sequence early. Without it HF
    # generates fewer tokens than the other two engines and the tok/s is not a rate.
    hf_model.generation_config.pad_token_id = tokenizer.pad_token_id

    def generate(ids: list[list[int]], n_new: int):
        input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        # Every row is exactly prompt_len, so the mask is all ones. Without it,
        # transformers warns that pad and EOS are the same token and may guess.
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            return hf_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=n_new,
                min_new_tokens=n_new,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

    print("Warming up HF...", flush=True)
    warm = prompt_ids_for(tokenizer, 1, min(prompt_len, 32))
    generate(warm, 1)
    sync(device)

    results: dict[int, dict[str, Any]] = {}
    for batch_size in batch_sizes:
        print(f"HF: batch {batch_size}...", flush=True)
        try:
            prompts = prompt_ids_for(tokenizer, batch_size, prompt_len)
            generate(prompts, 1)
            sync(device)
            trials: list[dict[str, Any]] = []
            first_ids: list[list[int]] | None = None
            last_ids: list[list[int]] | None = None
            for trial in range(1, repeats + 1):
                _, ttft_s = time_call(device, lambda: generate(prompts, 1))
                outputs, elapsed = time_call(device, lambda: generate(prompts, max_tokens))
                ids = outputs[:, prompt_len:].detach().cpu().tolist()
                new_tokens = sum(len(row) for row in ids)
                tok_s = new_tokens / elapsed if elapsed > 0 else 0.0
                trial_row = _cell(tok_s, ttft_s * 1000.0, elapsed, new_tokens)
                trials.append(trial_row)
                _warn_token_count("HF", batch_size, new_tokens, batch_size * max_tokens)
                _print_cell(batch_size, trial_row, trial=trial)
                if first_ids is None:
                    first_ids = ids
                last_ids = ids
            assert first_ids is not None and last_ids is not None
            results[batch_size] = pack_trials(trials, first_ids, first_ids == last_ids)
            _print_cell(batch_size, results[batch_size])
        except torch.cuda.OutOfMemoryError as exc:
            results[batch_size] = _oom_cell(exc)
            print(f"   batch {batch_size}: out of memory", flush=True)
            release(device)
    del hf_model
    release(device)
    return results


def _warn_token_count(engine: str, batch: int, got: int, expected: int) -> None:
    if got != expected:
        print(
            f"   warning: {engine} batch {batch} produced {got} tokens, expected {expected}",
            flush=True,
        )


def _print_cell(batch: int, row: dict[str, Any], trial: int | None = None) -> None:
    label = f"trial {trial} " if trial is not None else ""
    tok = f"{row['tok_s']:.2f}"
    ttft = f"{row['ttft_ms']:.2f}"
    if "tok_s_ci95" in row:
        tok += f" ± {row['tok_s_ci95']:.2f}"
    if "ttft_ms_ci95" in row:
        ttft += f" ± {row['ttft_ms_ci95']:.2f}"
    print(
        f"   {label}batch {batch:>2}: {tok:>8} tok/s | TTFT {ttft:>8} ms "
        f"({row['total_tokens']} tokens in {row['elapsed_s']:.2f}s)",
        flush=True,
    )


def _run_baby_once(llm, prompts: list[list[int]], max_tokens: int, batch_size: int, device: torch.device):
    from babyvllm.sequence import SamplingParams

    llm.engine.reset()
    llm.engine.scheduler.max_num_seqs = batch_size
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    order: list[int] = []
    for token_ids in prompts:
        order.append(llm.engine.add_request(token_ids, copy.copy(params)))

    seen: set[int] = set()
    by_seq = {seq_id: [] for seq_id in order}
    tokens_at_ttft = 0
    ttft_s = None
    generated = 0
    sync(device)
    start = time.perf_counter()
    while llm.engine.has_unfinished_requests():
        step_out = llm.engine.step()
        now = time.perf_counter()
        for out in step_out:
            n = len(out.new_token_ids)
            generated += n
            by_seq[out.seq_id].extend(out.new_token_ids)
            if n and out.seq_id not in seen:
                seen.add(out.seq_id)
        if ttft_s is None and len(seen) == batch_size:
            ttft_s = now - start
            tokens_at_ttft = generated
    sync(device)
    elapsed = time.perf_counter() - start
    return generated, elapsed, ttft_s, tokens_at_ttft, [by_seq[seq_id] for seq_id in order]


def run_benchmark_babyvllm(
    model_name: str,
    device: torch.device,
    batch_sizes: list[int],
    prompt_len: int,
    max_tokens: int,
    tokenizer,
    budget: int,
    repeats: int,
) -> dict[int, dict[str, Any]]:
    print("\n" + "=" * 60)
    print(" [2/3] baby-vLLM")
    print("=" * 60, flush=True)

    from babyvllm.config import SchedulerConfig
    from babyvllm.llm import LLM as BabyLLM
    from babyvllm.worker.model_runner import BATCH_BUCKETS, CTX_BUCKETS

    sched_cfg = SchedulerConfig(
        max_num_seqs=max(batch_sizes),
        max_num_batched_tokens=budget,
    )
    llm = BabyLLM(model_name, scheduler_config=sched_cfg, device=device, verbose=False)
    if device.type == "cuda" and llm.model_runner.dtype != torch.bfloat16:
        raise RuntimeError(
            f"baby-vLLM on CUDA is bf16 (got {llm.model_runner.dtype}). "
            "Refusing a comparison against a different dtype."
        )

    if device.type == "cuda" and llm.model_runner.use_cuda_graphs:
        keys, eager = graph_keys(batch_sizes, prompt_len, max_tokens, BATCH_BUCKETS, CTX_BUCKETS)
        if eager:
            print(f"Batches {eager} are above the graph buckets and will run eager.", flush=True)
        for batch_bucket, ctx_bucket in keys:
            print(f"Capturing CUDA graph batch={batch_bucket} ctx={ctx_bucket}", flush=True)
            llm.model_runner.capture_one(batch_bucket, ctx_bucket)
        sync(device)
    elif device.type == "cuda":
        print("CUDA graphs are off. Decode will be eager.", flush=True)

    print("Warming up baby-vLLM...", flush=True)
    warm = prompt_ids_for(tokenizer, 1, min(prompt_len, 32))
    _run_baby_once(llm, warm, 1, 1, device)

    results: dict[int, dict[str, Any]] = {}
    for batch_size in batch_sizes:
        print(f"baby-vLLM: batch {batch_size}...", flush=True)
        try:
            prompts = prompt_ids_for(tokenizer, batch_size, prompt_len)
            _run_baby_once(llm, prompts, 1, batch_size, device)
            trials: list[dict[str, Any]] = []
            first_ids: list[list[int]] | None = None
            last_ids: list[list[int]] | None = None
            for trial in range(1, repeats + 1):
                # Second value is the synced wall clock of the 1-token run.
                _generated, ttft_s, _in_run, _at, _one_ids = _run_baby_once(
                    llm, prompts, 1, batch_size, device
                )
                generated, elapsed, in_run_ttft, tokens_at_ttft, ids = _run_baby_once(
                    llm, prompts, max_tokens, batch_size, device
                )
                tok_s = generated / elapsed if elapsed > 0 else 0.0
                extra: dict[str, Any] = {}
                if in_run_ttft is not None:
                    extra["ttft_in_run_ms"] = round(in_run_ttft * 1000.0, 2)
                    rate = decode_tok_s(generated, batch_size, elapsed, in_run_ttft, tokens_at_ttft)
                    if rate is not None:
                        extra["decode_tok_s"] = round(rate, 2)
                trial_row = _cell(tok_s, ttft_s * 1000.0, elapsed, generated, **extra)
                trials.append(trial_row)
                _warn_token_count("baby-vLLM", batch_size, generated, batch_size * max_tokens)
                _print_cell(batch_size, trial_row, trial=trial)
                if first_ids is None:
                    first_ids = ids
                last_ids = ids
            assert first_ids is not None and last_ids is not None
            results[batch_size] = pack_trials(trials, first_ids, first_ids == last_ids)
            _print_cell(batch_size, results[batch_size])
        except torch.cuda.OutOfMemoryError as exc:
            results[batch_size] = _oom_cell(exc)
            print(f"   batch {batch_size}: out of memory", flush=True)
            release(device)

    # Drop the graph pool before vLLM profiles free memory.
    llm.model_runner._graphs.clear()
    llm.model_runner._static.clear()
    llm.model_runner._pool = None
    del llm
    release(device)
    return results


def _sampling_params(cls, n_tokens: int):
    fields = inspect.signature(cls).parameters
    if "ignore_eos" not in fields:
        raise RuntimeError("This vLLM SamplingParams has no ignore_eos. Refusing an early-EOS run.")
    kwargs = {"temperature": 0.0, "max_tokens": n_tokens, "ignore_eos": True}
    if "min_tokens" in fields:
        kwargs["min_tokens"] = n_tokens
    return cls(**kwargs)


def _vllm_generate(llm, token_ids: list[list[int]], params):
    prompts = [{"prompt_token_ids": ids} for ids in token_ids]
    try:
        return llm.generate(prompts, params, use_tqdm=False)
    except TypeError:
        return llm.generate(prompts, params)


def _count_vllm_tokens(outputs) -> int:
    total = 0
    for out in outputs:
        if not getattr(out, "outputs", None):
            continue
        total += len(out.outputs[0].token_ids)
    return total


def _shutdown_vllm(llm) -> None:
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    shutdown = getattr(core, "shutdown", None)
    if shutdown is not None:
        try:
            shutdown()
        except Exception:
            pass


def _vllm_ids(outputs) -> list[list[int]]:
    return [list(out.outputs[0].token_ids) for out in outputs]


def run_benchmark_vllm(
    model_name: str,
    dtype_str: str,
    device: torch.device,
    batch_sizes: list[int],
    prompt_len: int,
    max_tokens: int,
    tokenizer,
    budget: int,
    repeats: int,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    print("\n" + "=" * 60)
    print(" [3/3] vLLM offline LLM.generate")
    print("=" * 60, flush=True)

    try:
        from vllm import LLM as VLLM
        from vllm import SamplingParams as VLLMSamplingParams
    except ImportError:
        print("vLLM is not installed. Skipping vLLM.", flush=True)
        return {}, {"installed": False}

    max_model_len = prompt_len + max_tokens
    signature = inspect.signature(VLLM.__init__)
    parameters = set(signature.parameters) - {"self"}
    forwards = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    kwargs = vllm_engine_kwargs(
        parameters,
        model=model_name,
        dtype=dtype_str,
        max_batch=max(batch_sizes),
        max_model_len=max_model_len,
        batched_tokens=budget,
        forwards_kwargs=forwards,
    )
    meta = {
        "installed": True,
        "kwargs": {key: value for key, value in kwargs.items() if key != "model"},
        "prefix_caching": kwargs.get("enable_prefix_caching"),
        "chunked_prefill": "engine default",
        "enforce_eager": kwargs.get("enforce_eager"),
    }
    print(
        "Chunked prefill is left at vLLM's default. "
        f"Token budget {kwargs.get('max_num_batched_tokens', budget)} holds this grid's largest prefill.",
        flush=True,
    )
    print(f"vLLM settings: {meta['kwargs']}", flush=True)
    llm = VLLM(**kwargs)

    print("Warming up vLLM...", flush=True)
    warm = prompt_ids_for(tokenizer, 1, min(prompt_len, 32))
    _vllm_generate(llm, warm, _sampling_params(VLLMSamplingParams, 1))
    sync(device)

    results: dict[int, dict[str, Any]] = {}
    for batch_size in batch_sizes:
        print(f"vLLM: batch {batch_size}...", flush=True)
        try:
            prompts = prompt_ids_for(tokenizer, batch_size, prompt_len)
            one = _sampling_params(VLLMSamplingParams, 1)
            full = _sampling_params(VLLMSamplingParams, max_tokens)
            _vllm_generate(llm, prompts, one)
            sync(device)
            trials: list[dict[str, Any]] = []
            first_ids: list[list[int]] | None = None
            last_ids: list[list[int]] | None = None
            for trial in range(1, repeats + 1):
                ttft_out, ttft_s = time_call(device, lambda: _vllm_generate(llm, prompts, one))
                outputs, elapsed = time_call(device, lambda: _vllm_generate(llm, prompts, full))
                new_tokens = _count_vllm_tokens(outputs)
                one_tokens = _count_vllm_tokens(ttft_out)
                ids = _vllm_ids(outputs)
                tok_s = new_tokens / elapsed if elapsed > 0 else 0.0
                trial_row = _cell(tok_s, ttft_s * 1000.0, elapsed, new_tokens, ttft_tokens=one_tokens)
                trials.append(trial_row)
                _warn_token_count("vLLM", batch_size, new_tokens, batch_size * max_tokens)
                if one_tokens != batch_size:
                    print(
                        f"   warning: vLLM 1-token run returned {one_tokens} tokens, expected {batch_size}",
                        flush=True,
                    )
                _print_cell(batch_size, trial_row, trial=trial)
                if first_ids is None:
                    first_ids = ids
                last_ids = ids
            assert first_ids is not None and last_ids is not None
            results[batch_size] = pack_trials(trials, first_ids, first_ids == last_ids)
            _print_cell(batch_size, results[batch_size])
        except torch.cuda.OutOfMemoryError as exc:
            results[batch_size] = _oom_cell(exc)
            print(f"   batch {batch_size}: out of memory", flush=True)
            release(device)

    _shutdown_vllm(llm)
    del llm
    release(device)
    return results, meta


def _fmt(row: dict[str, Any] | None, metric: str) -> str:
    if not row or "error" in row:
        return "OOM" if row and row.get("error") == "oom" else "—"
    value = row.get(metric)
    if not isinstance(value, (int, float)):
        return "—"
    ci = row.get(f"{metric}_ci95")
    if isinstance(ci, (int, float)):
        return f"{value:.2f} ± {ci:.2f}"
    return f"{value:.2f}"


def print_markdown_results(
    batch_sizes: list[int],
    hf_res: dict[int, dict[str, Any]],
    baby_res: dict[int, dict[str, Any]],
    vllm_res: dict[int, dict[str, Any]],
    *,
    gpu: str,
    model_name: str,
    dtype_str: str,
    prompt_len: int,
    max_tokens: int,
    budget: int,
    vllm_version: str,
    cmd_str: str,
    is_publishable: bool,
) -> None:
    headers = ["Engine"] + [f"Batch {b}" for b in batch_sizes]

    def table(metric: str) -> None:
        print(" | ".join(headers))
        print(" | ".join(["---"] * len(headers)))
        rows = [
            ("HF transformers", hf_res),
            ("baby-vllm", baby_res),
            (f"vLLM {vllm_version}", vllm_res),
        ]
        for name, res in rows:
            cells = [name] + [_fmt(res.get(b), metric) for b in batch_sizes]
            print(" | ".join(cells))

    print("\n### Output throughput (tokens / second, prefill included)\n")
    table("tok_s")
    print("\n### Time to first token (ms, wall clock of a 1-token generate)\n")
    table("ttft_ms")

    baby_decode = [baby_res.get(b, {}).get("decode_tok_s") for b in batch_sizes]
    if any(isinstance(v, (int, float)) for v in baby_decode):
        print("\n### baby-vLLM decode-only tokens / second\n")
        print("Median across the same trials, with the prefill step removed. It is not the 21 October steady-state grid.")
        print(" | ".join(headers))
        print(" | ".join(["---"] * len(headers)))
        cells = ["baby-vllm"] + [_fmt(baby_res.get(b), "decode_tok_s") for b in batch_sizes]
        print(" | ".join(cells))

    ratios = []
    for b in batch_sizes:
        baby = baby_res.get(b, {}).get("tok_s")
        other = vllm_res.get(b, {}).get("tok_s")
        if isinstance(baby, (int, float)) and isinstance(other, (int, float)) and other > 0:
            ratios.append(f"{b}: {100.0 * baby / other:.1f}%")
    if ratios:
        print("\nbaby-vLLM throughput relative to vLLM: " + ", ".join(ratios))

    print("\n### Greedy token ids against HuggingFace\n")
    for batch in batch_sizes:
        reference = hf_res.get(batch, {}).get("greedy_ids")
        for name, res in (("baby-vllm", baby_res), (f"vLLM {vllm_version}", vllm_res)):
            report = compare_token_ids(reference, res.get(batch, {}).get("greedy_ids"))
            stable = res.get(batch, {}).get("greedy_ids_stable")
            if report.get("match"):
                text = f"batch {batch} {name}: match across {report['sequences']} sequences"
            elif report.get("reason") == "missing":
                text = f"batch {batch} {name}: not compared"
            else:
                where = report.get("first_mismatch")
                text = (
                    f"batch {batch} {name}: {report.get('mismatched_sequences', 0)} "
                    f"of {report.get('sequences', '?')} sequences differ, first at {where}"
                )
            if stable is False:
                text += ". Ids changed across trials"
            print(text)
        if hf_res.get(batch, {}).get("greedy_ids_stable") is False:
            print(f"batch {batch} HF transformers: ids changed across trials")

    print("\n" + "-" * 60)
    print(f"GPU: {gpu}")
    print(f"Model: {model_name} ({dtype_str})")
    print(f"Workload: prompt_len={prompt_len}, max_tokens={max_tokens}, greedy, ignore EOS")
    print(f"Prefill budget: {budget} tokens (cap {PREFILL_TOKEN_CAP})")
    print("Chunked prefill: vLLM default. This budget holds the largest prefill in the grid.")
    print("Reported point: median. 95% interval: 1.96 * sample std / sqrt(n).")
    print(
        f"Versions: vLLM {vllm_version}, HF transformers {package_version('transformers')}, "
        f"PyTorch {torch.__version__}"
    )
    print(f"Command: {cmd_str}")
    if is_publishable:
        print("Publishable: yes. Qwen2.5-7B, bf16, H100.")
    else:
        print("Publishable: no. The H100 ladder is Qwen2.5-7B, bf16, H100 only.")
    print("-" * 60)
    print(
        "Throughput includes prefill. The 21 October figures (170 tok/s at batch 1, "
        "9024 tok/s at batch 64) are steady-state decode around context 128–258, "
        "from a different harness. Do not put those numbers in this table."
    )
    if os.environ.get("BABYVLLM_PREFILL_ATTN", "triton") == "torch":
        prefill = ("baby-vLLM prefill (BABYVLLM_PREFILL_ATTN=torch) gathers each "
                   "sequence's KV blocks in Python and runs SDPA on that sequence.")
    else:
        prefill = ("baby-vLLM prefill runs the Triton varlen paged prefill kernel "
                   "over the whole batch, reading K/V from the paged cache.")
    print(
        f"{prefill} Decode replays a CUDA graph and the Triton "
        "paged-attention kernel. Each step copies the sampled ids back with "
        "tokens.cpu().tolist() before the next step is scheduled."
    )


def parse_args():
    p = argparse.ArgumentParser(description="HF vs baby-vLLM vs vLLM offline benchmark")
    p.add_argument("--model", default=PUBLISHED_MODEL, help="Hugging Face model id or local path")
    p.add_argument("--batches", default="1,8,32,64", help="Comma-separated batch sizes")
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None, help="cuda, cpu, or mps. Default: cuda if present.")
    p.add_argument("--repeats", type=int, default=3, help="Timed trials after warmup. The table reports the median.")
    p.add_argument("--skip-hf", action="store_true")
    p.add_argument("--skip-babyvllm", action="store_true")
    p.add_argument("--skip-vllm", action="store_true")
    p.add_argument("--out", default="benchmark_comparison.json")
    return p.parse_args()


def _json_ready(rows: dict[int, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(batch): row for batch, row in rows.items()}


def main():
    args = parse_args()
    batch_sizes = [int(part.strip()) for part in args.batches.split(",") if part.strip()]
    if not batch_sizes or any(b < 1 for b in batch_sizes):
        raise SystemExit("--batches must be a comma-separated list of positive integers")
    if args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type == "cuda" and args.dtype != "bfloat16":
        raise SystemExit(
            "baby-vLLM on CUDA is bf16. Pass --dtype bfloat16 so HF and vLLM match it."
        )

    budget = prefill_budget(batch_sizes, args.prompt_len)
    need = max(batch_sizes) * args.prompt_len
    if need > budget:
        print(
            f"Batch {max(batch_sizes)} x prompt {args.prompt_len} = {need} tokens, "
            f"above the prefill cap {budget}. Prefill is split across steps. "
            "TTFT is still the wall clock until one new token per request.",
            flush=True,
        )

    name = gpu_name()
    vllm_ver = package_version("vllm")
    cmd_str = "python " + " ".join(sys.argv)
    can_publish = (
        publishable(name, args.model, args.dtype)
        and device.type == "cuda"
        and args.repeats >= 3
    )

    print("=" * 60)
    print("HF vs baby-vLLM vs vLLM")
    print("=" * 60)
    print(f"Device:   {device} ({name})")
    print(f"Model:    {args.model} | dtype: {args.dtype}")
    print(f"Batches:  {batch_sizes}")
    print(f"Workload: prompt_len={args.prompt_len}, max_tokens={args.max_tokens}")
    print(f"Repeats:  {args.repeats} timed trials, median and 95% interval")
    print(f"Budget:   {budget} tokens per step")
    print(f"Versions: vLLM {vllm_ver}, HF {package_version('transformers')}, PyTorch {torch.__version__}")
    if not can_publish:
        print("These numbers are not the published H100 ladder.")
    print("=" * 60, flush=True)

    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    tokenizer = load_tokenizer(args.model)
    # Fail in the prompt builder before any engine allocates if the ids are wrong.
    probe = prompt_ids_for(tokenizer, max(batch_sizes), args.prompt_len)
    if len(probe) != max(batch_sizes) or any(len(ids) != args.prompt_len for ids in probe):
        raise SystemExit("prompt builder did not return exact lengths")
    if len({tuple(ids) for ids in probe}) != len(probe):
        raise SystemExit("prompt builder produced duplicate prompts")

    hf_results: dict[int, dict[str, Any]] = {}
    if not args.skip_hf:
        hf_results = run_benchmark_hf(
            args.model, getattr(torch, args.dtype), device, batch_sizes,
            args.prompt_len, args.max_tokens, tokenizer, args.repeats,
        )

    baby_results: dict[int, dict[str, Any]] = {}
    if not args.skip_babyvllm:
        baby_results = run_benchmark_babyvllm(
            args.model, device, batch_sizes, args.prompt_len, args.max_tokens,
            tokenizer, budget, args.repeats,
        )

    vllm_results: dict[int, dict[str, Any]] = {}
    vllm_meta: dict[str, Any] = {"installed": vllm_ver != "not installed"}
    if not args.skip_vllm:
        vllm_results, vllm_meta = run_benchmark_vllm(
            args.model, args.dtype, device, batch_sizes, args.prompt_len, args.max_tokens,
            tokenizer, budget, args.repeats,
        )

    print_markdown_results(
        batch_sizes, hf_results, baby_results, vllm_results,
        gpu=name, model_name=args.model, dtype_str=args.dtype,
        prompt_len=args.prompt_len, max_tokens=args.max_tokens, budget=budget,
        vllm_version=vllm_ver, cmd_str=cmd_str, is_publishable=can_publish,
    )

    payload = {
        "metadata": {
            "gpu": name,
            "model": args.model,
            "dtype": args.dtype,
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "batches": batch_sizes,
            "repeats": args.repeats,
            "prefill_budget": budget,
            "publishable": can_publish,
            "vllm_version": vllm_ver,
            "hf_version": package_version("transformers"),
            "pytorch_version": torch.__version__,
            "babyvllm_prefill_attn": os.environ.get("BABYVLLM_PREFILL_ATTN", "triton"),
            "babyvllm_decode_partition": os.environ.get("BABYVLLM_DECODE_PARTITION", "auto"),
            "command": cmd_str,
            "vllm": vllm_meta,
            "definition": {
                "tok_s": "median of new tokens / wall clock of generate(max_tokens), prefill included",
                "tok_s_ci95": "1.96 * sample std / sqrt(n) across timed trials",
                "ttft_ms": "median wall clock of a separate warm generate of 1 new token",
                "decode_tok_s": "baby-vLLM only: median of tokens after the first per request, same run",
                "greedy_ids": "token ids from trial 1, compared with HuggingFace",
            },
        },
        "hf_transformers": _json_ready(hf_results),
        "baby_vllm": _json_ready(baby_results),
        "vllm": _json_ready(vllm_results),
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
