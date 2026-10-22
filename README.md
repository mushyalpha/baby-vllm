<p align="center">
  <img src="assets/baby_vllm_logo.jpg" alt="Baby-vLLM logo" width="400">
</p>

# Baby-vLLM

A lightweight vLLM-style inference engine built from scratch.

## Key Features

* **Continuous batching**. Mix prefills and decodes in the same step
* **Lean codebase**. Clean implementation in ~1,400 lines of Python
* **Paged KV cache**. Block allocation, preemption, recompute when memory is tight

## Installation

```bash
pip install git+https://github.com/mushyalpha/baby-vllm.git
```

## Model Download

To download the model weights manually:

```bash
huggingface-cli download --resume-download Qwen/Qwen2-0.5B \
  --local-dir ~/huggingface/Qwen2-0.5B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface:

```python
from babyvllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen2-0.5B")
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Baby-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0].text
```

## Benchmark

Qwen2.5-7B, bf16, one H100 80GB HBM3. Prompt 128, 64 new tokens, greedy. Three trials. The point is the median, and the ± is a 95% interval. Throughput includes prefill. Time to first token is a separate one-token generate.

baby-vLLM matches vLLM at batch 1 and 8. At batch 64 it is about 10% slower, and the first token arrives 23 ms later. Greedy tokens match HuggingFace at every batch size.

Tokens per second. Prefill is included.

| Engine | Batch 1 | Batch 8 | Batch 32 | Batch 64 |
|---|---:|---:|---:|---:|
| HuggingFace | 76 | 621 | 2,054 | 4,049 |
| baby-vLLM | 168.0 ± 0.1 | 1,239 ± 12 | 3,872 ± 90 | 6,025 ± 39 |
| vLLM 0.32.0 | 163.7 ± 1.6 | 1,198 ± 34 | 4,121 ± 73 | 6,659 ± 236 |

Time to first token, milliseconds.

| Engine | Batch 1 | Batch 8 | Batch 32 | Batch 64 |
|---|---:|---:|---:|---:|
| HuggingFace | 13.8 ± 2.0 | 30.2 ± 0.4 | 103.7 ± 1.4 | 203.2 ± 2.9 |
| baby-vLLM | 11.7 ± 0.9 | 32.5 ± 2.0 | 117.8 ± 0.9 | 230.3 ± 1.3 |
| vLLM 0.32.0 | 17.2 ± 3.9 | 42.0 ± 11.2 | 110.1 ± 5.1 | 206.9 ± 17.5 |

HuggingFace's first throughput trial was still cold, so that row is the median only. PyTorch 2.13.0+cu130, vLLM 0.32.0, transformers 5.19.0. Chunked prefill stayed at vLLM's default.

```bash
bash benchmarks/run_h100_comparison.sh Qwen/Qwen2.5-7B 1,8,32,64 128 64
```

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