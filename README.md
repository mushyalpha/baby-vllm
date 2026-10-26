<p align="center">
  <img src="assets/baby_vllm_logo.jpg" alt="Baby-vLLM logo" width="400">
</p>

# Baby-vLLM

A lightweight vLLM-style inference engine built from scratch.

## Key Features

* **Continuous batching**. Mix prefills and decodes in the same step
* **Lean codebase**. Clean implementation in ~2,500 lines of Python
* **Paged KV cache**. Block allocation, preemption, recompute when memory is tight
* **Prefix caching**. Reference-counted, hash-chained 16-token blocks, vLLM v1 style (off by default)
* **Triton kernels**. Paged decode attention, varlen paged prefill attention, fused RMSNorm, RoPE + KV store, SiLU × mul
* **CUDA graphs**. Decode steps replay captured graphs

## Installation

```bash
pip install git+https://github.com/mushyalpha/baby-vllm.git
```

## Model Download

To download the model weights manually:

```bash
hf download Qwen/Qwen2.5-0.5B --local-dir ~/huggingface/Qwen2.5-0.5B
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface:

```python
from babyvllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen2.5-0.5B")
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Baby-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0].text
```

## Benchmark

Qwen2.5-7B, bf16, one H100 80GB HBM3, 26 October 2026. All three engines ran on the same pod. Prompt 128, 64 new tokens, greedy. Three trials. The point is the median, and the ± is a 95% interval. Throughput includes prefill. Time to first token is a separate one-token generate.

baby-vLLM matches vLLM 0.32.0 throughput at batch 64 and is ahead at batch 1, 8 and 32. Its first token arrives sooner at every batch size: 171 ms against 221 ms at batch 64. Greedy tokens match HuggingFace at every batch size.

Tokens per second. Prefill is included.

| Engine | Batch 1 | Batch 8 | Batch 32 | Batch 64 |
|---|---:|---:|---:|---:|
| HuggingFace | 88 | 708 | 2,595 | 4,437 |
| baby-vLLM | 170.0 ± 0.1 | 1,269 ± 4 | 4,174 ± 15 | 6,575 ± 2 |
| vLLM 0.32.0 | 164.1 ± 1.8 | 1,188 ± 11 | 3,954 ± 53 | 6,581 ± 235 |

Time to first token, milliseconds.

| Engine | Batch 1 | Batch 8 | Batch 32 | Batch 64 |
|---|---:|---:|---:|---:|
| HuggingFace | 13.5 ± 0.2 | 29.7 ± 0.2 | 104.0 ± 2.0 | 203.3 ± 2.3 |
| baby-vLLM | 7.7 ± 0.2 | 21.4 ± 0.3 | 82.9 ± 3.6 | 171.0 ± 2.6 |
| vLLM 0.32.0 | 17.9 ± 4.3 | 47.4 ± 5.7 | 120.7 ± 7.7 | 221.2 ± 12.1 |

HuggingFace's first throughput trial was still cold, so that row is the median only. PyTorch 2.13.0+cu130, vLLM 0.32.0, transformers 5.19.0. Prefix caching was off in both engines. Chunked prefill stayed at vLLM's default. Sequences stop at 192 tokens, so this does not exercise long-context attention.

baby-vLLM wins on prefill and still loses on decode. At batch 64, generate time minus time to first token is about 450 ms for baby-vLLM and about 400 ms for vLLM, roughly 0.8 ms per decode step.

```bash
bash benchmarks/run_h100_comparison.sh Qwen/Qwen2.5-7B 1,8,32,64 128 64
```

### Paged prefill attention

Multi-token steps run a Triton varlen kernel (`babyvllm/kernels/paged_prefill.py`). It reads K and V straight from the paged cache for the whole batch, handles a cached prefix plus new query tokens, and does causal masking and grouped-query attention in one launch per layer. The previous path looped over sequences in Python, gathered each one's blocks, and called SDPA. A prefix hit sent it to the math backend with an explicit mask.

Same pod, same PyTorch, only `BABYVLLM_PREFILL_ATTN` changed:

| Prefill attention | Batch 1 | Batch 8 | Batch 32 | Batch 64 |
|---|---:|---:|---:|---:|
| torch loop, tok/s | 168.5 | 1,243 | 3,928 | 6,069 |
| Triton kernel, tok/s | 170.0 | 1,269 | 4,174 | 6,575 |
| torch loop, first token (ms) | 11.4 | 30.7 | 117.6 | 231.4 |
| Triton kernel, first token (ms) | 7.7 | 21.4 | 82.9 | 171.0 |

One attention layer in isolation, from `benchmarks/bench_prefill_attn.py` on the same PyTorch 2.13:

| Shape | torch loop | Triton | Extra peak memory, torch → Triton |
|---|---:|---:|---:|
| 64 × 128 fresh prefill | 6.44 ms | 0.13 ms | 59 → 56 MiB |
| 1 × 2048 fresh prefill | 0.26 ms | 0.23 ms | 64 → 14 MiB |
| 8 × (1024 cached + 128 new) | 2.19 ms | 0.13 ms | 128 → 7 MiB |
| 1 × (2048 cached + 4096 new) | 19.26 ms | 1.26 ms | 6,512 → 28 MiB |
| 1 × 512 prefill + 63 decodes at 1000 | 12.05 ms | 0.35 ms | 60 → 4 MiB |

A single fresh sequence was already on flash SDPA, so it barely moves. The prefix-hit rows are where the old path fell back to the math backend: 15× slower and 6.5 GiB of scratch for one long sequence. The torch time for many-sequence shapes includes the Python loop's launch overhead. Inside the engine part of that hides behind the GEMMs, which is why batch 64 saved 60 ms end to end, not the 177 ms the per-layer number suggests. The kernel passes 23 GPU parity tests against the torch path (`tests/test_paged_prefill.py`). Raw results are in `results_26Oct2026_H100/`.

## Decode on one H100

The numbers below are the fused engine, Qwen2.5-7B in bf16, on one H100 80GB, measured on 21 October 2026. Each cell is three timing trials with graphs captured before the clock starts. The figure is the median step time, with a 95% confidence interval across the three trial medians. Kernel counts come from a separate Nsight Systems pass and are not the latency measurement.

The quick start above uses Qwen2.5-0.5B so it can run on a laptop. That model is not in this table.

| Shape | Timed sequence | Step time (p50) | Tokens/s | Kernels / step |
|---|---|---|---|---|
| 1 × 128 | 159–258 | 5.879 ± 0.009 ms | 170 | 329 |
| 1 × 2048 | 1949–2048 | 8.021 ± 0.002 ms | 125 | 329 |
| 64 × 128 | 159–258 | 7.092 ± 0.007 ms | 9,024 | 329 |
| 64 × 1920 | 1951–2048 | 10.081 ± 0.001 ms | 6,348 | 329 |

Every shape launches the same 329 kernels. The matmuls stay near 5.1–5.4 ms. What grows with context is paged attention, from 0.26 ms at batch 1 with a short cache to 2.86 ms at batch 64 near length 2048. Batching 64 sequences adds less than 2 ms at the short context, which is why tokens per second go from 170 to 9,024.

The long batch-64 prefill is chunked at 8,192 tokens. A single step of 64 × 1,920 tokens asks the gate/up projection for an 8.67 GiB activation, and the KV cache had already left only about 4 GiB free. Decode itself stays on the CUDA graph.

The write-up of the engine, the fusions, and what to build next is `kernel_fusion_report/baby_vllm_report.md`.