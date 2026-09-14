# Day 6/45: Paged Attention & The CPU Sync Trap

**Goal:** Eliminate the massive `gather_kv_batched` tensor copies by writing a custom Triton kernel that natively reads paged KV cache blocks, driving our decode step time closer to the 4.55 ms HBM floor.

## The Plot Twist (The Opus Review)
Going into Day 6, our baseline decode step time for `Qwen2.5-7B` at Batch=1 was 9.55 ms. I assumed this was due to the PyTorch attention mechanism struggling with uncoalesced memory reads. 

However, a brutal architectural review by Claude Opus revealed a harsh truth: **The 9.5 ms at B=1 was NOT caused by the attention mechanism.** 

At B=1 with a 128-token context, the entire 28-layer KV cache is only 7.3 MB. Moving 7.3 MB takes roughly 2 microseconds on an H100. You cannot optimize a 5 millisecond gap by speeding up a 2 microsecond operation. The math demanded a different culprit. 

The review pointed directly at hidden CPU synchronizations executing *outside* of our CUDA graph captures.

## Phase 1: Eradicating the CPU Syncs (The "Day 5.5" Baseline)
Before we could measure any Triton kernel, we had to stop Python from blocking the GPU. We found two massive bottlenecks:

1. **The Sampler Loop:** `sampler.py` was calling `torch.argmax(logit).item()` for every single sequence. At B=64, this forced 64 blocking GPU-to-CPU synchronizations per step.
   * **The Fix:** Batched the `argmax` and `multinomial` math on the GPU over the whole tensor, replacing the loop with a single `.cpu().tolist()` call at the very end.
2. **Pageable H2D Copies:** `prepare_static_inputs` was calling `torch.tensor(..., device="cuda")` in a loop to format the block tables. At B=64, this triggered 130+ pageable, blocking Host-to-Device memory copies.
   * **The Fix:** Introduced pinned CPU staging buffers (`torch.zeros(..., pin_memory=True)`). The Python lists are now formatted onto a flat CPU buffer first, and a single `.copy_(non_blocking=True)` streams it to the GPU asynchronously.

We ran a "Day 5.5" baseline with these fixes applied. At B=64 (context 1920), the step time plummeted from 20.90 ms down to **15.35 ms**. We had just clawed back **5.5 ms** of pure Python overhead.

## Phase 2: The Triton Paged Attention Kernel
With the framework overhead cleared, we finally deployed the Triton kernel to solve the memory-bandwidth bottleneck at high batch sizes. 

**Kernel Design:**
* **Native Block Lookups:** The kernel reads `seq_lens` as a device tensor and fetches physical blocks directly through the `block_tables` mapping, completely eliminating the need for PyTorch to manually gather and clone the KV cache into contiguous memory.
* **Tensor Cores:** Swapped SIMT `tl.sum` for `tl.dot`, padding the query block size to 16 to utilize the H100's MMA Tensor Cores.
* **Correctness & Safety:** Handled `-inf` masking carefully to avoid `NaN` pollution in the softmax on padded rows, zero-initialized the output buffers to prevent stale CUDA graph memory leaks, and added a strict `pytest` GPU parity gate against PyTorch SDPA.

## The Results (Qwen2.5-7B, H100 SXM5 @ 3.35 TB/s)

We measured the pure decode step time using the Inter-Token Latency (ITL p50) metric, isolating it from the massive prefill (TTFT) times.

### B=64, ctx=1920 (The Heavyweight Match)
This is where Paged Attention shines. The kernel saves us from reading and writing over 7 GB of gather buffers.
* **Day 5 (GQA Bug):** 186.86 ms
* **Day 5.5 (No CPU syncs, PyTorch SDPA):** 15.35 ms
* **Day 6 (Triton Paged Attention):** **11.86 ms**

**Result:** The Triton kernel successfully shaved another 3.5 ms off the step time, landing perfectly within Opus's predicted 11-14 ms window.

### B=1, ctx=128 (The Speed Limit)
* **Day 5 (Original graphs):** 9.55 ms
* **Day 5.5 (No CPU syncs, PyTorch SDPA):** 8.87 ms
* **Day 6 (Triton Paged Attention):** **8.21 ms**

**Result:** As predicted, Triton barely moves the needle here. The attention math is virtually free. The 8.21 ms floor is hard-capped by the sheer volume of kernel launches.

## Conclusion & Day 7 Handoff
We have officially conquered the memory-bandwidth bottleneck of the Attention mechanism. Our B=64 performance is extremely lean. 

However, our B=1 latency (8.21 ms) remains almost double the theoretical HBM floor (4.55 ms). The H100 is still spending more time launching kernels than doing math, because our engine executes ~1,240 separate, unfused elementwise PyTorch kernels per step (RMSNorm, RoPE, Silu).

For Day 7, we must leave the Attention block and write a custom **Fused GEMM + RoPE + Silu + RMSNorm kernel**. Once we fuse those elementwise operations into the Linear layers, the kernel count will plummet, and we will finally shatter the 5 ms barrier.
