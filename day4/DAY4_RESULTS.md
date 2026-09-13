# Day 4/45 — Roofline profiling: decode is memory-bound (measured)

**Goal:** Prove that decode is memory bound and prefill is compute bound using a real Roofline analysis.

## The Experiment

I built a PyTorch roofline profiler for **Qwen2.5-7B** on an **H100**.

First, I calculated the analytical floor for decode step latency before touching the GPU: **4.55 ms/step** (~220 tok/s at batch 1). 
Then, I measured the actual HuggingFace PyTorch implementation. I only achieved **28% of that theoretical floor** (~61 tok/s).

I expected batching to close this gap, but profiling showed that while GPU occupancy climbed from 65% → 89% (as batch size grew from 1 to 64), the kernels were stuck at about 40% bandwidth efficiency. Prefill, however, hit 45–55% MFU as expected once the GPU was saturated.

## Key Numbers (Qwen2.5-7B BF16, H100 SXM5 @ 3.35 TB/s)

| Case | Analytical floor | Measured | % of floor |
| --- | --- | --- | --- |
| B=1, ctx=128 | 220 tok/s | 61 tok/s | 28% |
| B=64, ctx=128 | 13,655 tok/s | 4,478 tok/s | 33% |
| B=64, ctx=2048 | 9,425 tok/s | 2,878 tok/s | 31% |

## Where the time went (Decode)

Using `torch.profiler`, the step time breaks down as:
- **~59% Other** (framework / CPU launch overhead / Python)
- **~34–38% Linear** (GEMM)
- **3–8% Attention** (grows with context)

The GPU is busy 65–89% of the step, but almost 60% of the *wall time* is lost to launch overhead and Python bookkeeping.

## Energy (NVML)

Batching is an energy strategy.
- B=1, ctx=128: ~3.7M Joules/Mtok
- B=64, ctx=128: ~77K Joules/Mtok

Batch 64 gives **36× more tokens/joule** than Batch 1.

## Conclusion & Hand-off

The 28-33% efficiency against the H100 bandwidth roofline is our **engineering loss budget**. To claw it back, we need to eliminate Python overhead (CUDA graphs), write proper memory management, and build a real Paged Attention kernel.
