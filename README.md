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

`benchmarks/benchmark_throughput.py` for a HuggingFace vs Baby-vLLM throughput comparison.

```bash
python benchmarks/benchmark_throughput.py
```