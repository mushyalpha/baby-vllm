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

Qwen2.5-7B, bf16, one H100 80GB HBM3. Each table is one pod with both engines on PyTorch 2.13.0+cu130 and vLLM 0.32.0. 64 new tokens, greedy, three trials. The point is the median, and the ± is a 95% interval. Throughput includes prefill. Time to first token is a separate one-token generate.

27 October 2026, prompt 128, after the decode host fix (greedy argmax inside the CUDA graph, decode inputs edited in place). baby-vLLM is ahead at every batch, and greedy tokens match HuggingFace.

| Tokens/s | Batch 1 | Batch 8 | Batch 32 | Batch 64 |
|---|---:|---:|---:|---:|
| baby-vLLM | 169.4 ± 0.4 | 1,283 ± 1 | 4,319 ± 11 | 6,969 ± 21 |
| vLLM 0.32.0 | 167.3 ± 0.1 | 1,203 ± 18 | 4,261 ± 50 | 6,882 ± 52 |
| First token, baby-vLLM (ms) | 9.7 | 21.6 | 82.8 | 168.6 |
| First token, vLLM (ms) | 9.8 | 31.4 | 100.1 | 184.8 |

28 October 2026, prompt 1,920, with flash-decoding (decode attention split along the sequence when the batch leaves SMs idle). Greedy tokens are identical to vLLM's for every sequence. Batch 64 is the one cell where vLLM is clearly ahead, mostly in the first token.

| Tokens/s | Batch 1 | Batch 8 | Batch 64 |
|---|---:|---:|---:|
| baby-vLLM, single-pass decode attention | 119.0 ± 7.8 | 603 ± 25 | 1,132 ± 20 |
| baby-vLLM, flash-decoding (default) | 149.3 ± 0.1 | 672 ± 29 | 1,136 ± 22 |
| vLLM 0.32.0 | 149.3 ± 0.3 | 694 ± 18 | 1,257 ± 8 |
| First token, baby-vLLM (ms) | 45.4 | 350.5 | 2,903 |
| First token, vLLM (ms) | 47.4 | 335.5 | 2,689 |

Prefix caching (`enable_prefix_caching=True`, off by default), 27 October: with a 1,536-token shared system prompt, the first token goes from 40.6 to 8.9 ms, and a burst of 32 requests needs 288 KV blocks instead of 3,264.

```bash
bash benchmarks/run_h100_decode_and_prefix.sh   # 27 October
bash benchmarks/run_h100_flash_decode.sh        # 28 October
```

### 26 October, before the decode host fix

Prompt 128. baby-vLLM matched vLLM 0.32.0 throughput at batch 64 and was ahead at batch 1, 8 and 32. Tokens per second, prefill included:

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

On 26 October baby-vLLM won on prefill and lost on decode, about 0.8 ms per step at batch 64. The 27 October host fix is what closed that.

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

This table is the Day 7 engine on PyTorch 2.8. On 28 October, after the decode host fix and flash-decoding, on PyTorch 2.13, the same cells are 5.871, 5.953, 6.621 and 9.094 ms. Near length 2048 at batch 1, decode attention went from 2.15 to 0.27 ms per step.

The write-up of the engine, the fusions, and what is left (FP8 weights, then tensor parallelism at TP = 2) is `kernel_fusion_report/baby_vllm_report.md`. The PDF is `kernel_fusion_report/Baby_vLLM.pdf`. One row per day is in `BENCHMARKS.md`.