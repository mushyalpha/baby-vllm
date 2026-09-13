# Baby-vLLM Project Context (Days 1 - 5)

## Day 4: Roofline Profiling
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


---

## Day 5 Narrative
# Day 5/45 of AI Inference Engineering: CUDA Graphs

Today I used CUDA graphs to double the serving speed of my custom "baby-vLLM" engine for a single user. 
But when I scaled up to 64 users, I accidentally uncovered a hidden PyTorch bug copying 105 GB of memory for no reason. 🧵👇

**Some context:** 
Earlier this week, I built a custom "baby-vLLM" engine with the goal of getting a first-principled understanding of why vLLM is powerful. 
I intentionally started bare-bones so I could add the optimizations myself, layer by layer. 

**The Day 4 Problem:**
Yesterday, I hit my first major bottleneck. Even with my custom engine, the GPU was spending 60% of its time doing absolutely nothing. 
Why? Because Python was sending 1,951 tiny, separate instructions to the GPU for every single word generated. 
The GPU is so incredibly fast that it would finish the math in a microsecond, and then sit around waiting for Python to send the next instruction. 
Out of a 21.2 ms step, the H100 spent 12.7 ms just staring at the wall. You don't rent a $30,000 GPU to wait on Python.

**The Day 5 Solution:**
Today I applied CUDA graphs to my custom-built engine to solve this. Instead of sending 1,951 separate instructions, I locked all the memory into static buffers and captured the whole decode step into a single CUDA graph.
It worked miraculously. 
This slashed the total time down to 9.5 ms. The idle waiting time dropped from 12.7 ms to just 1.2 ms, and the serving speed more than doubled from 47 tokens/sec to 105 tokens/sec.

The absolute physical memory bandwidth limit of the H100 for a 7B model is 4.5 ms. By dropping my step time to 9.5 ms, I am now operating at nearly 50% of the theoretical hardware limit.

**But surprisingly...**
When I simulated a heavy production load (Batch 64, or 64 concurrent users), my shiny new engine completely choked. 
It took 187 ms per step. That is 8x slower than the basic, unoptimized HuggingFace code I started with on Day 4. What happened?

I profiled it and found a catastrophe. 
A specific PyTorch attention setting (`enable_gqa=True`) was secretly triggering a fallback that needlessly duplicated memory. PyTorch was blindly copying 105 Gigabytes of data back and forth inside the GPU on every single step.

I applied a 2-line fix. By manually "folding" the tensor shape, I completely bypassed PyTorch's memory expansion. 
The latency instantly plummeted from a disastrous 187 ms down to an incredible 20.9 ms. I fixed an 8x slowdown with a reshape.

**So for Day 6:**
CUDA graphs are done. We've squeezed everything we can out of standard PyTorch. 
Tomorrow, I leave Python behind. I'll be writing a custom Triton kernel to handle Paged Attention natively on the GPU hardware.


---

## Day 5 Results Summary
# Day 5/45 — CUDA Graphs: final results

**Date:** 10 September 2026  
**Question of the series:** why is running this model so expensive, and what in the code is wasting the GPU.

The GPU was not slow. Python was making it wait. CUDA graphs fixed the waiting. Then PyTorch was secretly copying ~105 GB per token at batch 64; a reshape stopped that. What is left is inside the kernels.

---

## Setup

| | |
|---|---|
| GPU | NVIDIA H100 80GB HBM3 (rated 3.35 TB/s) |
| Torch | 2.8.0+cu128 |
| Models | Qwen2.5-7B (7.616e9 params, 28 layers, 4 KV heads, head_dim 128) and Qwen2.5-0.5B |
| Engine | baby-vLLM (Day 3) + static buffers + CUDA-graph capture (Day 5) |
| Timing | CUDA events, warmup 10 / timed 50, **median** ms |
| Floor | `(weights + B·ctx·KV) / 3.35e12` using **actual ctx**, not the bucket |
| Weight traffic | 15.23 GB/step (7B BF16) |
| KV bytes/token (7B) | 57,344 |
| Graph pool | 0.661 GB, 14 graphs keyed by `(batch_bucket, ctx_bucket)` |
| Buckets | batch `{1,2,4,8,16,32,64}`, ctx `{256, 2048}` |
| Long-ctx cell | measured **ctx=1920** (bucket 2048). True 2048 does not fit warmup+timed in that bucket. |
| HuggingFace column | Day 4, same GPU/model. ctx=1920 uses HF ctx=2048. Not re-measured on Day 5. |

Raw JSON: `results_7B (1).json`, `results_0.5B (1).json`, `profile_day5.json`, `results_gqa_fold_b64.json`.

---

## The story in four beats

**1. One user, one token — the GPU is bored.**  
A normal Python decode step sent the GPU **1,951 instructions** for every token. Each kernel was microseconds. Then the GPU waited for the next note. On a **21.2 ms** step, **12.7 ms was idle** (~60%). You were paying H100 rent for waiting.

**2. CUDA graphs — give it all the instructions at once.**  
Record the step once, replay as **2 CPU launches**. Same kernels, same math. **21.2 → 9.5 ms**, **47 → 105 tokens/sec**, **2.22×**. Kernel time barely moved (8.56 → 8.35 ms). Graphs delete **gaps**, not work.

**3. Batch 64 — PyTorch copies 105 GB for no reason.**  
64 users at once. Graphs did nothing (**187 ms** vs 191 ms static). HuggingFace on the same GPU: **22.6 ms**. We were **8× slower** than the baseline we set out to beat. Cause: `enable_gqa=True` plus an additive mask selects the math backend, which implements GQA as `repeat_interleave` on K/V. ~105 GB/step of avoidable traffic.

**4. Diagnostic (same cell, q-fold, not a full matrix re-run).**  
Fold the GQA group into the query-length dim. Do not expand K/V. Same cell: **187 → 20.9 ms**. HuggingFace is 22.6 ms. Padding 33→64 fell from **+49.5% to +4.8%**. Keep 187 in the published table; 20.9 is the one-line callout.

---

## Headline — 7B, batch 1, ctx=128

| | static eager (launch loop) | CUDA graph |
|---|---|---|
| ms / decode step | 21.2 | **9.5** |
| tokens / sec | 47 | **105** |
| CPU launches / step | **1,951** | **2** |
| kernels executed / step | 1,950 | 1,865 |
| Σ kernel time (nsys) | 8.56 ms | 8.35 ms |
| idle gaps (CUDA-event wall − kernel) | **12.7 ms** | **1.2 ms** |
| GPU busy (kernel / CUDA-event wall) | **40%** | **88%** |
| % of HBM floor (4.55 ms) | 21% | **48%** |
| speedup vs static | — | **2.22×** |

Idle accounting: nsys inflates wall in proportion to kernels traced. Kernel *durations* are good. Take ms from CUDA events, kernel time from nsys, idle = wall − kernel. Do not quote nsys busy 27% → 70%.

Kernel time alone is 8.5 ms against a 4.55 ms floor → **54% bandwidth efficiency** inside the kernels. A perfect graph on this implementation cannot beat ~54% of floor. We measured **47.7%** — about **88% of the available graph win**. Graphs are done.

---

## Day 4 correction

Day 4 reported “59% launch overhead” and “40% kernel bandwidth efficiency” as two findings. They were **one finding double-counted** — both used wall time as the denominator.

Measured properly at B=1/ctx=128 on baby-vLLM:

| Quantity | Value |
|---|---|
| Analytical floor | 4.55 ms |
| Σ kernel time | 8.5 ms |
| Kernel bandwidth efficiency | 4.55 / 8.5 = **54%** |
| Idle gaps | 12.7 ms (60% of the step) |
| Step (measured) | 21.2 ms |

Day 4 measured HuggingFace; Day 5 measures baby-vLLM with 1,951 launches/step. Same mechanism; magnitudes are not transferable.

---

## nsys — B=1, ctx=128 (static eager vs graph)

`--trace=cuda,nvtx --cuda-graph-trace=node`. 20 NVTX `decode_step`s.

| | static eager | graph |
|---|---|---|
| CUDA-event wall | 21.235 ms | 9.545 ms |
| nsys NVTX wall (inflated) | 30.34 ms | ~12.5 ms |
| Σ kernel / step | 8.557 ms | 8.349 ms |
| idle / step | 12.678 ms | 1.196 ms |
| busy | 40.3% | 87.5% |
| CPU launches / step | 1951 | 2 |
| kernels / step | 1949 | 1864 |

### GEMM is already on the HBM roof

The old top-two table (2.82 + 1.26 = 4.08 ms) implied **3.73 TB/s**, above the 3.35 TB/s peak — physically impossible. Summing **all** `nvjet*` / `gemv*` variants:

| | eager | graph |
|---|---|---|
| GEMM time | 5.39 ms (7 variants, 252 kernels) | 5.47 ms (8 variants, 242 kernels) |
| Implied bandwidth | **2.82 TB/s = 84% of peak** | 2.79 TB/s = 83% of peak |
| Other kernels | 3.17 ms | 2.88 ms |

GEMMs are not the Day 6 target at B=1. The addressable leftover is **~1,240 unfused elementwise kernels (~2.9 ms)**.

### Chart 2 encoding (absolute ms, not %)

| Bar | GEMM | Other kernels | Idle | Total |
|---|---|---|---|---|
| static eager | 5.4 | 3.2 | **12.7** | 21.2 |
| graph | 5.5 | 2.9 | **1.2** | 9.5 |

Kernel blocks are the same height. The gray idle block collapses. That is the graph result.

---

## Full 7B matrix (median ms)

`graph_vs_static` is the Day 5 number. HuggingFace is Day 4. Compile at low batch is contaminated (same process as hand-rolled graphs) — do not lead with it.

| B | ctx | floor | HF eager | day3 | static | graph | compile | graph/static | graph % floor |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 128 | 4.549 | 14.43 | 20.44 | 21.24 | **9.55** | 44.71 | **2.22×** | **47.7%** |
| 1 | 1920 | 4.580 | 14.33 | 21.51 | 22.82 | 12.15 | 45.85 | 1.88× | 37.7% |
| 4 | 128 | 4.555 | 14.49 | 40.29 | 22.52 | 10.86 | 45.39 | 2.07× | 41.9% |
| 4 | 1920 | 4.678 | 14.27 | 40.59 | 24.30 | 20.64 | 50.08 | 1.18× | 22.7% |
| 16 | 128 | 4.582 | 14.54 | 115.92 | 23.86 | 15.93 | 48.03 | 1.50× | 28.8% |
| 16 | 1920 | 5.072 | 14.47 | 114.96 | 58.63 | 54.66 | 66.59 | 1.07× | 9.3% |
| 64 | 128 | 4.687 | 14.31 | 407.21 | 37.62 | 33.55 | 57.69 | 1.12× | 14.0% |
| 64 | 1920 | 6.650 | **22.57** | 402.49 | 190.88 | **186.86** | 165.38 | 1.02× | 3.6% |

At B=64/ctx=1920 the published graph number is **186.86 ms vs HF 22.57 ms = 8.3× slower than HuggingFace**. Graphs recover 2%. That cell is copy-bound, not launch-bound.

Chart 3 plots **ctx=128 only** so B=1 is 2.22×, not the 2.05× average of (2.22 + 1.88)/2.

---

## Batch 64 diagnostic — GQA materialize

`enable_gqa=True` does **not** save memory on the math backend. An explicit additive mask disqualifies flash. The fallback does `key.repeat_interleave` / `value.repeat_interleave`.

Arithmetic at B=64, ctx bucket 2048:

| Term | Bytes | @ peak |
|---|---|---|
| Weights | 15.23 GB | 4.5 ms |
| Gather + permute + contiguous | ~22.5 GB | 6.7 ms |
| GQA expand 4→28 heads, K+V, 28 layers (write+read) | **~105 GB** | **31 ms** |
| Total at peak | ~143 GB | ~43 ms |

Measured 187 ms ⇒ those copies run at ~23% of peak (uncoalesced strided copies). The 187 ms is explained.

**Fix (in `static_attention.py`):** fold the query group dim instead of expanding KV. Mathematically identical. Use `.reshape`, not `.view`, on the SDPA output (non-contiguous).

```text
q: [B, 28, 1, 128] → [B, 4, 7, 128]
k, v: [B, 4, ctx, 128]  — untouched
mask: [B, 1, 1, ctx] broadcasts over the group dim
```

### Same cell, after the fold (not a full matrix re-run)

File: `results_gqa_fold_b64.json`. GPU: H100 80GB HBM3.

| | Published (enable_gqa) | After q-fold |
|---|---|---|
| static eager | 190.88 ms | **23.85 ms** |
| CUDA graph | **186.86 ms** | **20.90 ms** |
| graph / static | 1.02× | 1.14× |
| % of floor | 3.6% | **31.8%** |
| vs HF 22.57 ms | 8.3× slower | about even |

**187 → 20.9 ms.** Day 3 path still ~401 ms (Python loop + its own `repeat_interleave`).

nsys at B=64 was **not** captured: the diagnostic pod had no `/opt/nsight-systems`. `profile_b64_old.json` is torch.profiler fallback and is not usable. The timing drop is the proof.

Keep **186.86 ms** in the published table. Report 20.9 ms as a one-line “fixed cell” callout.

---

## Padding waste (graph path, ctx=128 → bucket 256)

Padding is nearly free when you are weight-bound, and linear when you are attention-bound.

| | padded rows | overhead **before** q-fold | overhead **after** q-fold |
|---|---|---|---|
| 7B B=3→4 (25% padded) | 1 | +3.5% | +0.3% |
| 7B B=33→64 (48% padded) | 31 | **+49.5%** | **+4.8%** |
| 0.5B B=3→4 | 1 | +1.9% | — |
| 0.5B B=33→64 | 31 | +31.0% | — |

Pre-registered Day 6 prediction was: after the attention kernel lands, 33→64 padding should fall from ~49% toward ~10%. The q-fold already landed **4.8%**. The remaining Day 6 job is paged attention without a gather buffer, not padding.

---

## 0.5B — bound by kernel count, not bytes

| B | ctx | floor | static | graph | graph/static | % floor |
|---|---|---|---|---|---|---|
| 1 | 128 | 0.295 | 18.68 | **3.94** | **4.74×** | 7.5% |
| 1 | 1920 | 0.302 | 17.79 | 4.63 | 3.84× | 6.5% |
| 4 | 128 | 0.297 | 17.73 | 4.51 | 3.93× | 6.6% |
| 4 | 1920 | 0.323 | 19.25 | 6.94 | 2.77× | 4.7% |
| 16 | 128 | 0.302 | 20.20 | 5.92 | 3.41× | 5.1% |
| 16 | 1920 | 0.408 | 20.11 | 15.53 | 1.29× | 2.6% |
| 64 | 128 | 0.325 | 21.63 | 10.56 | 2.05× | 3.1% |
| 64 | 1920 | 0.746 | 55.19 | 51.01 | 1.08× | 1.5% |

After graphs, 0.5B B=1 is **3.94 ms vs a 0.295 ms floor** (7.5% of floor). Scale launches: 1,951 × 24/28 ≈ 1,670 kernels. 3.94 ms / 1,670 ≈ **2.36 µs per kernel** — about the minimum wall-clock of a trivial kernel on an H100.

The small model is not bandwidth-bound or launch-bound after graphs. It is bound by **how many kernels exist**. That is fusion (Day 7).

---

## What we did *not* get

- nsys at B=64 (no Nsight on the diagnostic pod). Do not cite `profile_b64_old.json`.
- HuggingFace re-measured on Day 5 (`hf_eager_ms` is null in the JSON; table uses Day 4).
- A clean `torch.compile` column. Same-process graphs poison `reduce-overhead`. Ignore compile at low batch. At B=64/ctx=1920 compile beat graph (165 vs 187) — do not lead with that.
- Pool reserve is still **9.66 GB reserved / 0.661 GB used**. About 9 GB of KV cache given away. Day 6: reserve `measured_pool × 2 + 0.5 GB` after a calibration run.

---

## Day 6 — numbers already committed

1. **Triton paged decode attention.** `seq_lens` as a device tensor, blocks read through the block table, group dim folded into query length, no mask materialization, no gather buffer. Targets: B=64/ctx=2048 **187 → under 40 ms** (q-fold already at 21 ms; kernel should hold that without the gather). Padding 33→64 already **4.8%**. Drop ctx buckets if the kernel makes them unnecessary.
2. **Correctness gate** first: exact token-ID match vs SDPA for 64 greedy steps, plus one bucket-switch replay.
3. **Reference:** `flash_attn_with_kvcache` or vLLM’s kernel.
4. **Free:** fix the 9 GB pool over-reserve; finer batch buckets if still padding-bound.

**Day 7 (fusion):** ~1,240 elementwise kernels, ~2.1–2.9 ms. At B=1, 9.5 ms toward ~7.4 ms (48% → ~61% of floor).

---

## Files

| File | What it is |
|---|---|
| `results_7B (1).json` | Published 7B matrix (includes 186.86 ms cell) |
| `results_0.5B (1).json` | 0.5B matrix |
| `profile_day5.json` | nsys B=1/ctx=128, idle = CUDA-event − kernel |
| `results_gqa_fold_b64.json` | Diagnostic: same B=64/ctx=1920 cell after q-fold (**20.90 ms**) |
| `charts/chart_1_roofline.png` | tok/s vs batch: static eager, graph, HF |
| `charts/chart_2_other.png` | Absolute ms: GEMM / other kernels / idle |
| `charts/chart_3_speedup.png` | Speedup vs batch, **ctx=128 only** |
| `day5-cuda-graphs.html` | Animation: shared ms axis, idle 12.7 → 1.2, 47 → 105 tok/s |
| `poster_day5.png` | Freeze frame of that animation |
| `static_attention.py` | Q-fold (`.reshape`); do not use `enable_gqa=True` with an additive mask |


---
# Codebase

sequence.py

```python
import itertools
from enum import Enum, auto
from dataclasses import dataclass, field

@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 256
    stop_tokens: set[int] = field(default_factory=set)


class SequenceState(Enum):
    WAITING = auto()
    RUNNING = auto()
    DONE = auto()

class SequenceFinishReason(str, Enum):
    STOP = "stop"
    LENGTH = "length"
    ABORT = "abort"

class Sequence:
    _counter = itertools.count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams | None = None):
        if not token_ids:
            raise ValueError("Prompt cannot be empty")
        self.seq_id = next(Sequence._counter)
        self.token_ids = list(token_ids)
        self.sampling_params = sampling_params or SamplingParams()
        self.status = SequenceState.WAITING
        self.finish_reason = None
        self._prompt_len = len(token_ids)
        self.num_computed_tokens: int = 0


    def __len__(self):
        return len(self.token_ids)

    def __getitem__(self, item):
        return self.token_ids[item]

    def __repr__(self) -> str:
        return (f"Sequence(id={self.seq_id}, {self.status.name}, "
                f"prompt={self._prompt_len}, total={len(self)}, "
                f"computed={self.num_computed_tokens})")

    @property
    def output_token_ids(self) -> list[int]:
        return self.token_ids[self._prompt_len:]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(token_id)

    def advance_computed(self, n: int) -> None:
        self.num_computed_tokens += n
        if self.num_computed_tokens > len(self.token_ids):
            raise ValueError(f"Computed tokens ({self.num_computed_tokens}) exceeds sequence length ({len(self.token_ids)})")

    @property
    def prompt_len(self) -> int:
        return self._prompt_len

    @property
    def generated_token_len(self) -> int:
        return len(self.token_ids) - self.prompt_len

    @property
    def num_tokens_to_compute(self) -> int:
        return len(self.token_ids) - self.num_computed_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceState.DONE

    @property
    def last_token_id(self):
        return self.token_ids[-1]

    def finish(self, reason: SequenceFinishReason) -> None:
        self.status = SequenceState.DONE
        self.finish_reason = reason

    def reset_for_recompute(self) -> None:
        self.status = SequenceState.WAITING
        self.num_computed_tokens = 0
```

kv_cache_manager.py

```python
from collections import deque
from babyvllm.sequence import Sequence

_DUMMY_BLOCK = 0

class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0, "num_blocks must be > 0"
        assert block_size > 0, "block_size must be > 0"
        
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_block_ids = deque(range(num_blocks))

        self.used_block_ids = set()
        self.request_blocks: dict[int, list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def reset(self):
        self.free_block_ids = deque(range(self.num_blocks))
        self.used_block_ids.clear()
        self.request_blocks.clear()

    def _check_invariants(self):
        assert self.num_free_blocks + len(self.used_block_ids) == self.num_blocks

    def get_block_table(self, seq: Sequence) -> list[int]:
        return self.request_blocks.get(seq.seq_id, [])

    def get_slot_mapping(self, seq: Sequence, num_new_tokens: int) -> list[int]:
        table = self.request_blocks.get(seq.seq_id, [])
        start = seq.num_computed_tokens
        out = []
        for pos in range(start, start + num_new_tokens):
            b, o = divmod(pos, self.block_size)
            assert b < len(table), f"seq {seq.seq_id} pos {pos} has no block"
            out.append(table[b] * self.block_size + o)
        return out

    def num_blocks_needed(self, seq: Sequence, num_new_tokens: int) -> int:
        target_num_computed_tokens = seq.num_computed_tokens + num_new_tokens
        target_num_logical_blocks = (target_num_computed_tokens + self.block_size - 1) // self.block_size
        return max(0, target_num_logical_blocks - len(self.request_blocks.get(seq.seq_id, [])))

    def can_allocate(self, seq: Sequence, num_new_tokens: int) -> bool:
        return self.num_free_blocks >= self.num_blocks_needed(seq, num_new_tokens)

    def allocate_slots(self, seq: Sequence, num_new_tokens: int) -> None:
        needed = self.num_blocks_needed(seq, num_new_tokens)
        
        assert self.can_allocate(seq, num_new_tokens), f"OOM: cannot allocate for seq {seq.seq_id}. Free: {self.num_free_blocks}, needed: {needed}"

        if seq.seq_id not in self.request_blocks:
            self.request_blocks[seq.seq_id] = []

        for _ in range(needed):
            block_id = self.free_block_ids.popleft()
            self.used_block_ids.add(block_id)
            self.request_blocks[seq.seq_id].append(block_id)

    def free(self, seq: Sequence): 
        assert seq.seq_id in self.request_blocks, f"free() called on seq {seq.seq_id} with no allocated blocks. Double-free or free-without-allocate."

        for block_id in self.request_blocks[seq.seq_id]:
            self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)

        del self.request_blocks[seq.seq_id]

    def free_if_allocated(self, seq: Sequence):
        if seq.seq_id in self.request_blocks:
            self.free(seq)
```

config.py

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_hidden_layers: int = 24
    num_attention_heads: int = 14
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 32768
    block_size: int = 16
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True

@dataclass
class CacheConfig:
    block_size: int = 16
    num_gpu_blocks: int = 100
    num_cpu_blocks: int = 100

@dataclass
class SchedulerConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
```

scheduler.py

```python
from collections import deque
from dataclasses import dataclass
from babyvllm.sequence import Sequence, SequenceState, SequenceFinishReason
from babyvllm.kv_cache_manager import KVCacheManager

@dataclass
class SchedulerOutput:
    scheduled_sequences: list[Sequence]
    num_scheduled_tokens: dict[int, int]
    block_tables: dict[int, list[int]]
    slot_mappings: dict[int, list[int]]
    preempted_seqs: list[Sequence]

class Scheduler:
    def __init__(self, kv_cache_manager: KVCacheManager, max_num_batched_tokens: int = 2048, max_num_seqs: int = 256):
        
        
        self.kv_cache_manager = kv_cache_manager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished_seqs: list[Sequence] = []

    def _finish(self, seq: Sequence, reason: SequenceFinishReason) -> None:
        seq.finish(reason)
        self.free_seq(seq)
        self.finished_seqs.append(seq)

    def pop_finished(self) -> list[Sequence]:
        out, self.finished_seqs = self.finished_seqs, []
        return out

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def add_seq(self, seq: Sequence):
        if len(seq) > self.max_num_batched_tokens:
            self._finish(seq, SequenceFinishReason.LENGTH)
            return
            
        if self.kv_cache_manager.num_blocks_needed(seq, len(seq)) > self.kv_cache_manager.num_blocks:
            self._finish(seq, SequenceFinishReason.LENGTH)
            return
            
        seq.status = SequenceState.WAITING
        self.waiting.append(seq)

    def admit_seq(self, seq: Sequence):
        popped = self.waiting.popleft()
        assert popped == seq
        seq.status = SequenceState.RUNNING
        self.running.append(seq)

    def abort_seq(self, seq: Sequence):
        self._finish(seq, SequenceFinishReason.ABORT)

    def free_seq(self, seq: Sequence):
        if seq in self.running:
            self.running.remove(seq)
        elif seq in self.waiting:
            self.waiting.remove(seq)
        
        self.kv_cache_manager.free_if_allocated(seq)

    def update_from_output(self, scheduler_output: SchedulerOutput, model_output: dict[int, int]):
        for seq in scheduler_output.scheduled_sequences:
            num_scheduled = scheduler_output.num_scheduled_tokens[seq.seq_id]
            seq.advance_computed(num_scheduled)
            if seq.seq_id in model_output:
                token_id = model_output[seq.seq_id]
                if token_id in seq.sampling_params.stop_tokens:
                    self._finish(seq, SequenceFinishReason.STOP)
                else:
                    seq.append_token(token_id)
                    if seq.generated_token_len >= seq.sampling_params.max_tokens:
                        self._finish(seq, SequenceFinishReason.LENGTH)

    def schedule(self) -> SchedulerOutput:
        budget = self.max_num_batched_tokens
        scheduled_sequences: list[Sequence] = []
        num_scheduled_tokens: dict[int, int] = {}
        block_tables: dict[int, list[int]] = {}
        slot_mappings: dict[int, list[int]] = {}

        running_seqs = list(self.running)
        preempted_seqs = []
        for seq in running_seqs:
            if seq.status != SequenceState.RUNNING:
                continue
            needed = seq.num_tokens_to_compute
            
            if budget < needed or len(scheduled_sequences) >= self.max_num_seqs:
                break
                
            while not self.kv_cache_manager.can_allocate(seq, needed):
                if len(self.running) > 1 and self.running[-1] is not seq:
                    victim = self.running[-1]
                    victim.reset_for_recompute()
                    self.running.remove(victim)
                    preempted_seqs.append(victim)
                    self.kv_cache_manager.free_if_allocated(victim)
                else:
                    seq.reset_for_recompute()
                    self.running.remove(seq)
                    preempted_seqs.append(seq)
                    self.kv_cache_manager.free_if_allocated(seq)
                    break
            else:
                self.kv_cache_manager.allocate_slots(seq, needed)
                scheduled_sequences.append(seq)
                num_scheduled_tokens[seq.seq_id] = needed
                block_tables[seq.seq_id] = list(self.kv_cache_manager.get_block_table(seq))
                slot_mappings[seq.seq_id] = self.kv_cache_manager.get_slot_mapping(seq, needed)
                budget -= needed

        for seq in reversed(preempted_seqs):
            self.waiting.appendleft(seq)

        while self.waiting:
            seq = self.waiting[0]
            if len(scheduled_sequences) >= self.max_num_seqs:
                break
                
            needed = seq.num_tokens_to_compute
            
            if needed > self.max_num_batched_tokens:
                self._finish(seq, SequenceFinishReason.LENGTH)
                continue
            
            if self.kv_cache_manager.num_blocks_needed(seq, needed) > self.kv_cache_manager.num_blocks:
                self._finish(seq, SequenceFinishReason.LENGTH)
                continue
            
            if budget >= needed and self.kv_cache_manager.can_allocate(seq, needed):
                self.kv_cache_manager.allocate_slots(seq, needed)
                self.admit_seq(seq)
                scheduled_sequences.append(seq)
                num_scheduled_tokens[seq.seq_id] = needed
                block_tables[seq.seq_id] = list(self.kv_cache_manager.get_block_table(seq))
                slot_mappings[seq.seq_id] = self.kv_cache_manager.get_slot_mapping(seq, needed)
                budget -= needed
            else:
                break

        return SchedulerOutput(
            scheduled_sequences=scheduled_sequences,
            num_scheduled_tokens=num_scheduled_tokens,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            preempted_seqs=preempted_seqs,
        )
```

engine.py

```python
from dataclasses import dataclass

from babyvllm.sequence import (
    Sequence,
    SamplingParams,
    SequenceFinishReason,
)
from babyvllm.scheduler import Scheduler, SchedulerOutput
from babyvllm.kv_cache_manager import KVCacheManager
from babyvllm.worker.model_runner import nvtx_range


@dataclass
class RequestOutput:
    seq_id: int
    new_token_ids: list[int]
    finish_reason: SequenceFinishReason | None = None
    text: str = ""

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None


class LLMEngine:
    def __init__(
        self,
        model_runner,
        max_num_batched_tokens: int = 8,
        max_num_seqs: int = 4,
    ):
        self.model_runner = model_runner
        num_blocks = getattr(model_runner, "num_blocks", None)
        if not num_blocks:
            num_blocks = model_runner.determine_num_blocks()

        self.kv_cache_manager = KVCacheManager(
            num_blocks=num_blocks,
            block_size=model_runner.block_size,
        )

        self.scheduler = Scheduler(
            kv_cache_manager=self.kv_cache_manager,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        )

        self.sequences: dict[int, Sequence] = {}

        self.block_size = model_runner.block_size

    def add_request(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> int:

        seq = Sequence(
            token_ids=token_ids,
            sampling_params=sampling_params,
        )

        self.sequences[seq.seq_id] = seq
        self.scheduler.add_seq(seq)

        return seq.seq_id

    def abort_request(self, seq_id: int) -> None:
        seq = self.sequences.get(seq_id)
        if seq is None:
            return

        if not seq.is_finished:
            self.scheduler.abort_seq(seq)

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    def step(self) -> list[RequestOutput]:
        if not self.has_unfinished_requests() and not self.scheduler.finished_seqs:
            return []

        scheduler_output = None
        sampled_tokens = {}
        if self.has_unfinished_requests():
            with nvtx_range("schedule"):
                scheduler_output = self.scheduler.schedule()

            made_progress = (
                bool(scheduler_output.scheduled_sequences) or 
                bool(self.scheduler.finished_seqs) or 
                bool(scheduler_output.preempted_seqs)
            )

            if not made_progress:
                raise RuntimeError(
                    "Engine stalled: unfinished requests exist, "
                    "but scheduler scheduled no work, preempted nothing, and no sequences finished."
                )

            if scheduler_output.scheduled_sequences:
                sampled_tokens = self.model_runner.execute_model(
                    scheduler_output
                )

                if not isinstance(sampled_tokens, dict):
                    raise RuntimeError(
                        "ModelRunner returned incorrect type"
                    )

                with nvtx_range("update"):
                    self.scheduler.update_from_output(
                        scheduler_output,
                        sampled_tokens,
                    )

        outputs = []
        handled_seq_ids = set()
        
        if scheduler_output and scheduler_output.scheduled_sequences:
            for seq in scheduler_output.scheduled_sequences:
                if seq.seq_id in sampled_tokens:
                    token = sampled_tokens[seq.seq_id]
                    new_ids = [] if token in seq.sampling_params.stop_tokens else [token]
                    outputs.append(
                        RequestOutput(
                            seq_id=seq.seq_id,
                            new_token_ids=new_ids,
                            finish_reason=seq.finish_reason,
                        )
                    )
                    handled_seq_ids.add(seq.seq_id)

        for seq in self.scheduler.pop_finished():
            if seq.seq_id not in handled_seq_ids:
                outputs.append(
                    RequestOutput(
                        seq_id=seq.seq_id,
                        new_token_ids=[],
                        finish_reason=seq.finish_reason,
                    )
                )
            self.sequences.pop(seq.seq_id, None)

        return outputs

    def run(self, max_steps: int = 100_000) -> dict[int, RequestOutput]:
        steps = 0
        final_outputs: dict[int, RequestOutput] = {}
        
        while self.has_unfinished_requests():
            step_outputs = self.step()
            for out in step_outputs:
                if out.seq_id not in final_outputs:
                    final_outputs[out.seq_id] = RequestOutput(
                        seq_id=out.seq_id,
                        new_token_ids=[],
                        finish_reason=None
                    )
                final_outputs[out.seq_id].new_token_ids.extend(out.new_token_ids)
                if out.finished:
                    final_outputs[out.seq_id].finish_reason = out.finish_reason
            
            steps += 1
            if steps >= max_steps:
                raise RuntimeError(f"Engine.run() hit max_steps ({max_steps}). Possible livelock.")

        return final_outputs

    def reset(self) -> None:
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self.scheduler.finished_seqs.clear()
        self.kv_cache_manager.reset()
        self.sequences.clear()
```

llm.py

```python
import os
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

from babyvllm.config import ModelConfig, CacheConfig, SchedulerConfig
from babyvllm.engine import LLMEngine
from babyvllm.worker.model_runner import ModelRunner, pick_device
from babyvllm.worker.loader import load_model
from babyvllm.sequence import SamplingParams

class LLM:
    def __init__(
        self,
        model_name: str,
        cache_config: CacheConfig = None,
        scheduler_config: SchedulerConfig = None,
        device=None,
        verbose: bool = False,
        use_cuda_graphs: bool | None = None,
    ):
        self.model_name = model_name
        self.verbose = verbose
        device = torch.device(device) if device is not None else pick_device()
        self._log(f"Loading {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        try:
            from transformers import GenerationConfig
            gen_cfg = GenerationConfig.from_pretrained(model_name)
            if isinstance(gen_cfg.eos_token_id, list):
                self.eos_token_id = set(gen_cfg.eos_token_id)
            elif isinstance(gen_cfg.eos_token_id, int):
                self.eos_token_id = {gen_cfg.eos_token_id}
            else:
                self.eos_token_id = set()
        except Exception:
            if self.tokenizer.eos_token_id is not None:
                self.eos_token_id = {self.tokenizer.eos_token_id}
            else:
                self.eos_token_id = set()
        
        if os.path.isdir(model_name):
            path = model_name
        else:
            self._log("Downloading / resolving checkpoint...")
            path = snapshot_download(repo_id=model_name)
            self._log(f"Checkpoint ready at {path}")
        
        import json
        with open(f"{path}/config.json", "r") as f:
            hf_cfg = json.load(f)
            
        self.model_config = ModelConfig(
            vocab_size=hf_cfg.get("vocab_size", 151936),
            hidden_size=hf_cfg.get("hidden_size", 896),
            intermediate_size=hf_cfg.get("intermediate_size", 4864),
            num_hidden_layers=hf_cfg.get("num_hidden_layers", 24),
            num_attention_heads=hf_cfg.get("num_attention_heads", 14),
            num_key_value_heads=hf_cfg.get("num_key_value_heads", 2),
            rms_norm_eps=hf_cfg.get("rms_norm_eps", 1e-6),
            max_position_embeddings=hf_cfg.get("max_position_embeddings", 32768),
            rope_theta=hf_cfg.get("rope_theta", 1000000.0),
            tie_word_embeddings=hf_cfg.get("tie_word_embeddings", True),
        )
        
        sched_cfg = scheduler_config or SchedulerConfig()
        
        self._log("Building model...")
        self.model_runner = ModelRunner(
            self.model_config,
            cache_config=cache_config,
            scheduler_config=sched_cfg,
            device=device,
            use_cuda_graphs=use_cuda_graphs,
        )
        self._log("Loading weights...")
        load_model(self.model_runner.model, path)

        num_blocks = self.model_runner.allocate_kv_cache()
        profile = self.model_runner.kv_profile or {}
        if "peak_memory_gb" in profile:
            self._log(
                f"KV profile: peak {profile['peak_memory_gb']:.2f} GB / "
                f"{profile['total_memory_gb']:.2f} GB, "
                f"{profile['available_for_kv_gb']:.2f} GB free for cache → "
                f"{num_blocks} blocks "
                f"({profile['max_kv_tokens']} tokens)"
            )
        else:
            self._log(f"KV cache: {num_blocks} blocks (CPU/MPS fallback)")

        self.engine = LLMEngine(
            model_runner=self.model_runner,
            max_num_batched_tokens=sched_cfg.max_num_batched_tokens,
            max_num_seqs=sched_cfg.max_num_seqs
        )
        if self.model_runner.use_cuda_graphs:
            self._log("CUDA graphs enabled for decode")
        self._log("Engine ready.")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def generate(self, prompts: list[str], sampling_params=None):
        import copy
        if sampling_params is None:
            sampling_params = SamplingParams(temperature=0.0)

        for prompt in prompts:
            req_params = copy.copy(sampling_params)
            if not req_params.ignore_eos:
                req_params.stop_tokens = set(req_params.stop_tokens) | self.eos_token_id
            
            token_ids = self.tokenizer(prompt).input_ids
            self.engine.add_request(token_ids, req_params)
            
        outputs = list(self.engine.run().values())
        for out in outputs:
            out.text = self.tokenizer.decode(out.new_token_ids)
            
        return outputs
```

attention.py

```python
from babyvllm.kernels.paged_attn import paged_decode_attention
import torch
import torch.nn as nn
import torch.nn.functional as F

from babyvllm.worker.context import get_forward_context

def store_kvcache(key: torch.Tensor,
                  value: torch.Tensor,
                  kv_cache: torch.Tensor,
                  slot_mapping: torch.Tensor) -> None:
    _, _, _, num_kv_heads, head_dim = kv_cache.shape
    k_flat = kv_cache[0].view(-1, num_kv_heads, head_dim)
    v_flat = kv_cache[1].view(-1, num_kv_heads, head_dim)
    k_flat.index_copy_(0, slot_mapping, key.to(k_flat.dtype))
    v_flat.index_copy_(0, slot_mapping, value.to(v_flat.dtype))


class Attention(nn.Module):
    """Paged attention over a flattened (varlen) token batch."""

    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int,
                 scale: float | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.scale = scale if scale is not None else head_dim ** -0.5
        self.kv_cache: torch.Tensor | None = None

    def forward(self, q, k, v):
        ctx = get_forward_context()
        if self.kv_cache is None or ctx is None:
            return _naive_causal_attention(q, k, v, self.scale,
                                           self.num_queries_per_kv)
        md = ctx.attn_metadata
        store_kvcache(k, v, self.kv_cache, md.slot_mapping)
        if md.max_query_len == 1:
            return paged_decode_attention(q, self.kv_cache, md.block_tables, md.seq_lens, self.scale)
        return _paged_attention_torch(q, self.kv_cache, md, self.scale,
                                      self.num_queries_per_kv)


def gather_kv_batched(
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    ctx_bucket: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_blocks_needed = (ctx_bucket + block_size - 1) // block_size
    n_blocks_needed = min(n_blocks_needed, block_tables.shape[1])
    B = block_tables.shape[0]
    n_kv_heads = kv_cache.shape[3]
    head_dim = kv_cache.shape[4]

    bt = block_tables[:, :n_blocks_needed]
    bt_flat = bt.reshape(-1)

    k_blocks = kv_cache[0][bt_flat]
    v_blocks = kv_cache[1][bt_flat]

    gathered = n_blocks_needed * block_size
    k = k_blocks.reshape(B, gathered, n_kv_heads, head_dim)
    v = v_blocks.reshape(B, gathered, n_kv_heads, head_dim)

    k = k[:, :ctx_bucket].permute(0, 2, 1, 3)
    v = v[:, :ctx_bucket].permute(0, 2, 1, 3)
    return k, v


def make_decode_mask(
    seq_lens: torch.Tensor,
    ctx_bucket: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(ctx_bucket, device=device)
    seq_lens_long = seq_lens.to(torch.int64)
    valid = positions[None, :] < seq_lens_long[:, None]
    additive = torch.zeros(
        seq_lens.shape[0], 1, 1, ctx_bucket, dtype=dtype, device=device
    )
    return additive.masked_fill(~valid[:, None, None, :], float("-inf"))


def _static_decode_attention(
    q, kv_cache, md, scale, num_heads, num_kv_heads, head_dim, num_queries_per_kv,
):
    B = md.seq_lens.shape[0]
    ctx_bucket = md.max_seq_len
    block_size = kv_cache.shape[2]

    K, V = gather_kv_batched(kv_cache, md.block_tables, ctx_bucket, block_size)
    Q = q.reshape(B, num_heads, 1, head_dim)
    mask = make_decode_mask(md.seq_lens, K.shape[2], q.device, q.dtype)

    group = num_queries_per_kv
    if group > 1:
        Q = Q.reshape(B, num_kv_heads, group, head_dim)
    out = F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=mask,
        scale=scale,
    )
    if group > 1:
        out = out.reshape(B, num_heads, 1, head_dim)
    return out.squeeze(2).reshape(-1, num_heads, head_dim)


def _paged_attention_torch(query, kv_cache, md, scale, num_queries_per_kv):
    block_size = kv_cache.shape[2]
    num_kv_heads, head_dim = kv_cache.shape[3], kv_cache.shape[4]
    out = torch.empty_like(query)

    for i, (start, end) in enumerate(zip(md.query_start_loc_cpu[:-1],
                                         md.query_start_loc_cpu[1:])):
        seq_len = md.seq_lens_cpu[i]
        ctx_len = md.context_lens_cpu[i]
        q_len = end - start

        num_blocks = (seq_len + block_size - 1) // block_size
        blocks = md.block_tables[i, :num_blocks]

        k_i = kv_cache[0][blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        v_i = kv_cache[1][blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        if num_queries_per_kv > 1:
            k_i = k_i.repeat_interleave(num_queries_per_kv, dim=1)
            v_i = v_i.repeat_interleave(num_queries_per_kv, dim=1)

        q_i = query[start:end].transpose(0, 1)
        k_i = k_i.transpose(0, 1)
        v_i = v_i.transpose(0, 1)

        if ctx_len == 0 and q_len == seq_len:
            o = F.scaled_dot_product_attention(q_i, k_i, v_i,
                                               is_causal=True, scale=scale)
        else:
            mask = torch.ones(q_len, seq_len, dtype=torch.bool, device=query.device)
            mask = torch.tril(mask, diagonal=ctx_len)
            o = F.scaled_dot_product_attention(q_i, k_i, v_i,
                                               attn_mask=mask, scale=scale)
        out[start:end] = o.transpose(0, 1)
    return out


def _naive_causal_attention(q, k, v, scale, num_queries_per_kv):
    """Single-sequence, no-cache reference. Only for parity tests."""
    if num_queries_per_kv > 1:
        k = k.repeat_interleave(num_queries_per_kv, dim=1)
        v = v.repeat_interleave(num_queries_per_kv, dim=1)
    o = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1),
                                       v.transpose(0, 1), is_causal=True, scale=scale)
    return o.transpose(0, 1)
```

layernorm.py

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)
```

rotary.py

```python
import torch
import torch.nn as nn

def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype):
        cos = self.cos_cached[0, 0, positions].to(dtype)
        sin = self.sin_cached[0, 0, positions].to(dtype)
        return cos, sin
```

sampler.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from babyvllm.worker.context import AttentionMetadata

class Sampler(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        logits: torch.Tensor,
        attn_metadata: AttentionMetadata,
        scheduler_output,
    ) -> dict[int, int]:
        sampled_tokens = {}
        for i, seq in enumerate(scheduler_output.scheduled_sequences):
            params = seq.sampling_params
            logit = logits[i].float()
            
            if params is None or params.temperature == 0.0:
                token = torch.argmax(logit).item()
            else:
                probs = F.softmax(logit / params.temperature, dim=-1)
                if params.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > params.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    probs[indices_to_remove] = 0.0
                    probs = probs / probs.sum()
                token = torch.multinomial(probs, num_samples=1).item()
            sampled_tokens[seq.seq_id] = token
            
        return sampled_tokens
```

qwen2.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from babyvllm.layers.layernorm import RMSNorm
from babyvllm.layers.rotary import RotaryEmbedding, apply_rope
from babyvllm.layers.attention import Attention

class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Qwen2Attention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)
        self.attn = Attention(self.num_heads, self.head_dim, self.num_kv_heads)

    def forward(self, x, cos, sin):
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        
        q, k = apply_rope(q, k, cos, sin)
        
        o = self.attn(q, k, v)
        return self.o_proj(o.reshape(-1, self.num_heads * self.head_dim))

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen2Attention(cfg, layer_idx)
        self.mlp = Qwen2MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin)
        x = residual + x
        
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x
        return x

class Qwen2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)
        ])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(
            cfg.hidden_size // cfg.num_attention_heads,
            max_position_embeddings=cfg.max_position_embeddings,
            base=cfg.rope_theta
        )

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x

class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = Qwen2Model(cfg)
        if cfg.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def compute_logits(self, hidden):
        w = self.model.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        return F.linear(hidden, w)

    def forward(self, input_ids, positions):
        return self.model(input_ids, positions)
```

context.py

```python
from dataclasses import dataclass
from contextlib import contextmanager
import torch

@dataclass
class AttentionMetadata:
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    query_start_loc_cpu: list[int]
    seq_lens_cpu: list[int]
    context_lens_cpu: list[int]
    max_query_len: int
    max_seq_len: int

@dataclass
class ForwardContext:
    attn_metadata: AttentionMetadata

_FORWARD_CTX: ForwardContext | None = None

@contextmanager
def set_forward_context(attn_metadata):
    global _FORWARD_CTX
    prev = _FORWARD_CTX
    _FORWARD_CTX = ForwardContext(attn_metadata)
    try:
        yield
    finally:
        _FORWARD_CTX = prev

def get_forward_context() -> ForwardContext | None:
    return _FORWARD_CTX
```

loader.py

```python
import os, glob, torch
from safetensors import safe_open

def load_model(model: torch.nn.Module, path: str):
    params = dict(model.named_parameters())
    loaded = set()
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as fp:
            for name in fp.keys():
                if name == "lm_head.weight" and model.lm_head is None:
                    continue
                if name not in params:
                    raise KeyError(f"unexpected checkpoint key: {name}")
                p = params[name]
                w = fp.get_tensor(name)
                assert p.shape == w.shape, (name, p.shape, w.shape)
                p.data.copy_(w.to(p.dtype))
                loaded.add(name)
    missing = set(params) - loaded
    if missing:
        raise RuntimeError(f"uninitialised params: {sorted(missing)}")
```

model_runner.py

```python
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from babyvllm.layers.attention import Attention
from babyvllm.models.qwen2 import Qwen2ForCausalLM
from babyvllm.worker.context import AttentionMetadata, set_forward_context
from babyvllm.worker.graph_runner import CUDAGraphRunner
from babyvllm.layers.sampler import Sampler


BATCH_BUCKETS = [1, 2, 4, 8, 16, 32, 64]
CTX_BUCKETS = [256, 2048]


@contextmanager
def nvtx_range(name: str):
    enabled = torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def ceil_to_bucket(value: int, buckets: list[int]) -> int | None:
    for b in buckets:
        if value <= b:
            return b
    return None


def compute_pool_reserve(
    max_batch_bucket: int,
    max_ctx_bucket: int,
    n_kv_heads: int,
    head_dim: int,
    n_layers: int,
    bytes_per_element: int = 2,
) -> int:
    gather_per_layer = (
        max_batch_bucket * n_kv_heads * max_ctx_bucket * head_dim
        * 2
        * bytes_per_element
    )
    total_gather = gather_per_layer * n_layers
    slack = 2 * 1024 * 1024 * 1024
    return total_gather + slack


@dataclass
class StaticBuffers:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    context_lens: torch.Tensor
    slot_mapping: torch.Tensor
    query_start_loc: torch.Tensor
    logits: torch.Tensor


class ModelRunner:
    def __init__(
        self,
        model_config,
        cache_config=None,
        scheduler_config=None,
        device=None,
        use_cuda_graphs: bool | None = None,
    ):
        self.model_config = model_config
        self.cache_config = cache_config
        self.scheduler_config = scheduler_config
        self.block_size = cache_config.block_size if cache_config else getattr(model_config, 'block_size', 16)
        self.device = torch.device(device) if device is not None else pick_device()
        self.dtype = pick_dtype(self.device)
        if use_cuda_graphs is None:
            self.use_cuda_graphs = self.device.type == "cuda"
        else:
            self.use_cuda_graphs = bool(use_cuda_graphs) and self.device.type == "cuda"

        torch.set_default_dtype(self.dtype)
        with torch.device(self.device):
            self.model = Qwen2ForCausalLM(model_config)
        self.model.eval()
        torch.set_default_dtype(torch.float32)

        self.sampler = Sampler(model_config.vocab_size).to(self.device)
        self.num_blocks: int | None = None
        self.kv_profile: dict | None = None
        self._dummy_block = 0
        self._static: dict[tuple[int, int], StaticBuffers] = {}
        self._graphs: dict[tuple[int, int], CUDAGraphRunner] = {}
        self._pool = None
        self._seq_lens_cpu: list[int] = []
        max_ctx = max(CTX_BUCKETS)
        self.max_blocks_per_seq = (max_ctx + self.block_size - 1) // self.block_size + 1

    def _bytes_per_token(self) -> int:
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
        bytes_per_element = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        return (
            2
            * self.model_config.num_key_value_heads
            * head_dim
            * self.model_config.num_hidden_layers
            * bytes_per_element
        )

    def determine_num_blocks(self) -> int:
        if self.device.type != "cuda":
            num_blocks = self.cache_config.num_cpu_blocks if self.cache_config else 100
            self.kv_profile = {
                "num_blocks": num_blocks,
                "block_size": self.block_size,
                "bytes_per_token": self._bytes_per_token(),
            }
            return num_blocks

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        max_batched_tokens = (
            self.scheduler_config.max_num_batched_tokens if self.scheduler_config else 2048
        )
        dummy_input = torch.zeros(max_batched_tokens, dtype=torch.int64, device=self.device)
        dummy_positions = torch.arange(max_batched_tokens, dtype=torch.int64, device=self.device)

        with torch.inference_mode():
            hidden_states = self.model(dummy_input, dummy_positions)
            _ = self.model.compute_logits(hidden_states)
            del hidden_states

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        peak_memory = torch.cuda.max_memory_allocated()
        allocated = torch.cuda.memory_allocated()
        free_bytes, total_memory = torch.cuda.mem_get_info()
        utilization = 0.9
        slack = 256 * 1024 * 1024
        pool_reserve = 0
        if self.use_cuda_graphs:
            head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
            pool_reserve = compute_pool_reserve(
                max(BATCH_BUCKETS),
                max(CTX_BUCKETS),
                self.model_config.num_key_value_heads,
                head_dim,
                self.model_config.num_hidden_layers,
                2 if self.dtype in (torch.float16, torch.bfloat16) else 4,
            )
        budget = int(total_memory * utilization) - allocated
        available_for_kv = min(budget, int(free_bytes) - slack - pool_reserve)
        bytes_per_token = self._bytes_per_token()
        bytes_per_block = bytes_per_token * self.block_size
        num_blocks = max(1, int(available_for_kv / bytes_per_block)) if available_for_kv > 0 else 1

        self.kv_profile = {
            "peak_memory_gb": peak_memory / 1e9,
            "allocated_gb": allocated / 1e9,
            "free_memory_gb": free_bytes / 1e9,
            "total_memory_gb": total_memory / 1e9,
            "utilization": utilization,
            "available_for_kv_gb": max(0.0, available_for_kv) / 1e9,
            "bytes_per_token": bytes_per_token,
            "block_size": self.block_size,
            "num_blocks": num_blocks,
            "max_kv_tokens": num_blocks * self.block_size,
        }
        return num_blocks

    def _alloc_kv_tensors(self, num_blocks: int) -> None:
        num_kv_heads = self.model_config.num_key_value_heads
        head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.kv_cache = torch.empty(
                    2,
                    num_blocks,
                    self.block_size,
                    num_kv_heads,
                    head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )

    def allocate_kv_cache(self) -> int:
        if self.num_blocks is not None:
            return self.num_blocks

        usable = self.determine_num_blocks()
        alloc = usable + 1 if self.use_cuda_graphs else usable
        while True:
            try:
                self._alloc_kv_tensors(alloc)
                break
            except torch.cuda.OutOfMemoryError:
                for module in self.model.modules():
                    if isinstance(module, Attention):
                        module.kv_cache = None
                torch.cuda.empty_cache()
                if alloc <= 1:
                    raise
                alloc = max(1, alloc // 2)
                if self.kv_profile is not None:
                    usable_now = alloc - 1 if self.use_cuda_graphs and alloc > 1 else alloc
                    self.kv_profile["num_blocks"] = usable_now
                    self.kv_profile["max_kv_tokens"] = usable_now * self.block_size
                    self.kv_profile["oom_backoff"] = True

        if self.use_cuda_graphs and alloc > 1:
            self._dummy_block = alloc - 1
            self.num_blocks = alloc - 1
        else:
            self._dummy_block = 0
            self.num_blocks = alloc
        if self.kv_profile is not None:
            self.kv_profile["num_blocks"] = self.num_blocks
            self.kv_profile["max_kv_tokens"] = self.num_blocks * self.block_size
        return self.num_blocks

    def prepare_inputs(self, scheduler_output) -> tuple[torch.Tensor, torch.Tensor, AttentionMetadata]:
        slot_mapping = []
        seq_lens = []
        context_lens = []
        block_tables = []
        query_lens = []
        input_tokens = []
        positions_list = []
        
        max_query_len = 0
        max_seq_len = 0
        max_num_blocks = 0

        for seq in scheduler_output.scheduled_sequences:
            needed = scheduler_output.num_scheduled_tokens[seq.seq_id]
            start_idx = seq.num_computed_tokens
            
            query_tokens = seq.token_ids[start_idx : start_idx + needed]
            input_tokens.extend(query_tokens)
            query_lens.append(needed)
            
            ctx_len = seq.num_computed_tokens
            context_lens.append(ctx_len)
            
            seq_len = ctx_len + needed
            seq_lens.append(seq_len)
            
            max_query_len = max(max_query_len, needed)
            max_seq_len = max(max_seq_len, seq_len)
            
            b_table = scheduler_output.block_tables[seq.seq_id]
            block_tables.append(b_table)
            max_num_blocks = max(max_num_blocks, len(b_table))
            
            s_map = scheduler_output.slot_mappings[seq.seq_id]
            slot_mapping.extend(s_map)
            
            positions_list.extend(range(ctx_len, ctx_len + needed))
            
        padded_block_tables = []
        for bt in block_tables:
            padded_block_tables.append(bt + [0] * (max_num_blocks - len(bt)))

        query_start_loc = [0]
        curr = 0
        for qlen in query_lens:
            curr += qlen
            query_start_loc.append(curr)

        attn_metadata = AttentionMetadata(
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.int64, device=self.device),
            query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=self.device),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            context_lens=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
            block_tables=torch.tensor(padded_block_tables, dtype=torch.int32, device=self.device),
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            seq_lens_cpu=seq_lens,
            context_lens_cpu=context_lens,
            query_start_loc_cpu=query_start_loc,
        )
        
        flat_input_ids = torch.tensor(input_tokens, dtype=torch.int64, device=self.device)
        positions = torch.tensor(positions_list, dtype=torch.int64, device=self.device)
        return flat_input_ids, positions, attn_metadata

    def _can_graph_decode(self, scheduler_output) -> bool:
        if not self.use_cuda_graphs:
            return False
        seqs = scheduler_output.scheduled_sequences
        if not seqs or self.num_blocks is None:
            return False
        for seq in seqs:
            if scheduler_output.num_scheduled_tokens[seq.seq_id] != 1:
                return False
        if ceil_to_bucket(len(seqs), BATCH_BUCKETS) is None:
            return False
        max_seq = 0
        for seq in seqs:
            max_seq = max(max_seq, seq.num_computed_tokens + 1)
        return ceil_to_bucket(max_seq, CTX_BUCKETS) is not None

    def _ensure_static_buffers(self, batch_bucket: int, ctx_bucket: int) -> StaticBuffers:
        key = (batch_bucket, ctx_bucket)
        if key in self._static:
            return self._static[key]

        B = batch_bucket
        MB = self.max_blocks_per_seq
        V = self.model_config.vocab_size
        dev = self.device
        dummy_slot = self._dummy_block * self.block_size
        buf = StaticBuffers(
            input_ids=torch.zeros(B, dtype=torch.int64, device=dev),
            position_ids=torch.zeros(B, dtype=torch.int64, device=dev),
            block_tables=torch.full((B, MB), self._dummy_block, dtype=torch.int32, device=dev),
            seq_lens=torch.zeros(B, dtype=torch.int32, device=dev),
            context_lens=torch.zeros(B, dtype=torch.int32, device=dev),
            slot_mapping=torch.full((B,), dummy_slot, dtype=torch.int64, device=dev),
            query_start_loc=torch.arange(B + 1, dtype=torch.int32, device=dev),
            logits=torch.zeros(B, V, dtype=torch.float32, device=dev),
        )
        self._static[key] = buf
        return buf

    def prepare_static_inputs(
        self,
        token_ids: list[int],
        positions: list[int],
        block_tables: list[list[int]],
        seq_lens: list[int],
        slot_mappings: list[int],
        batch_bucket: int,
        ctx_bucket: int,
    ) -> StaticBuffers:
        buf = self._ensure_static_buffers(batch_bucket, ctx_bucket)
        B_actual = len(token_ids)
        dummy_slot = self._dummy_block * self.block_size

        self._seq_lens_cpu = list(seq_lens) + [0] * (batch_bucket - B_actual)

        buf.input_ids[:B_actual].copy_(
            torch.tensor(token_ids, dtype=torch.int64, device=self.device)
        )
        buf.position_ids[:B_actual].copy_(
            torch.tensor(positions, dtype=torch.int64, device=self.device)
        )
        buf.seq_lens[:B_actual].copy_(
            torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        )
        buf.context_lens[:B_actual].copy_(
            torch.tensor([s - 1 for s in seq_lens], dtype=torch.int32, device=self.device)
        )
        buf.slot_mapping[:B_actual].copy_(
            torch.tensor(slot_mappings, dtype=torch.int64, device=self.device)
        )

        if B_actual < batch_bucket:
            buf.input_ids[B_actual:].zero_()
            buf.position_ids[B_actual:].zero_()
            buf.seq_lens[B_actual:].zero_()
            buf.context_lens[B_actual:].zero_()
            buf.slot_mapping[B_actual:].fill_(dummy_slot)
            buf.block_tables[B_actual:].fill_(self._dummy_block)

        for i, bt in enumerate(block_tables):
            n = len(bt)
            buf.block_tables[i, :n].copy_(
                torch.tensor(bt, dtype=torch.int32, device=self.device)
            )
            if n < self.max_blocks_per_seq:
                buf.block_tables[i, n:].fill_(self._dummy_block)

        return buf

    @torch.inference_mode()
    def _forward_static(
        self,
        buf: StaticBuffers,
        batch_bucket: int,
        ctx_bucket: int,
    ) -> torch.Tensor:
        B = batch_bucket
        seq_lens_cpu = (
            self._seq_lens_cpu[:B] if len(self._seq_lens_cpu) >= B else [0] * B
        )
        context_lens_cpu = [max(0, s - 1) for s in seq_lens_cpu]
        attn_metadata = AttentionMetadata(
            slot_mapping=buf.slot_mapping[:B],
            block_tables=buf.block_tables[:B],
            query_start_loc=buf.query_start_loc[: B + 1],
            seq_lens=buf.seq_lens[:B],
            context_lens=buf.context_lens[:B],
            max_query_len=1,
            max_seq_len=ctx_bucket,
            query_start_loc_cpu=list(range(B + 1)),
            seq_lens_cpu=seq_lens_cpu,
            context_lens_cpu=context_lens_cpu,
        )

        with set_forward_context(attn_metadata):
            hidden = self.model(buf.input_ids[:B], buf.position_ids[:B])

        logits = self.model.compute_logits(hidden)
        buf.logits[:B].copy_(logits)
        return buf.logits

    def capture_one(self, batch_bucket: int, ctx_bucket: int, warmup_steps: int = 3):
        key = (batch_bucket, ctx_bucket)
        if key in self._graphs:
            return self._graphs[key]

        buf = self._ensure_static_buffers(batch_bucket, ctx_bucket)
        if len(self._seq_lens_cpu) < batch_bucket:
            self._seq_lens_cpu = [1] * batch_bucket

        side_stream = torch.cuda.Stream()
        with torch.cuda.stream(side_stream):
            for _ in range(warmup_steps):
                self._forward_static(buf, batch_bucket, ctx_bucket)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        runner = CUDAGraphRunner()
        runner.capture(
            self._forward_static, buf, batch_bucket, ctx_bucket, pool=self._pool
        )
        self._graphs[key] = runner
        if self._pool is None:
            self._pool = runner.pool
        return runner

    def capture_all(self, warmup_steps: int = 3):
        for cb in reversed(CTX_BUCKETS):
            for bb in reversed(BATCH_BUCKETS):
                self.capture_one(bb, cb, warmup_steps=warmup_steps)

    def replay(self, buf: StaticBuffers, batch_bucket: int, ctx_bucket: int) -> torch.Tensor:
        key = (batch_bucket, ctx_bucket)
        if key not in self._graphs:
            raise RuntimeError(f"No graph captured for {key}. Call capture_one() first.")
        self._graphs[key].replay()
        return buf.logits

    def _execute_decode_graph(self, scheduler_output) -> dict[int, int]:
        seqs = scheduler_output.scheduled_sequences
        token_ids = []
        positions = []
        block_tables = []
        seq_lens = []
        slot_mappings = []
        for seq in seqs:
            start = seq.num_computed_tokens
            token_ids.append(seq.token_ids[start])
            positions.append(start)
            seq_lens.append(start + 1)
            block_tables.append(scheduler_output.block_tables[seq.seq_id])
            slot_mappings.append(scheduler_output.slot_mappings[seq.seq_id][0])

        bb = ceil_to_bucket(len(seqs), BATCH_BUCKETS)
        cb = ceil_to_bucket(max(seq_lens), CTX_BUCKETS)
        assert bb is not None and cb is not None

        with nvtx_range("prepare_inputs"):
            buf = self.prepare_static_inputs(
                token_ids, positions, block_tables, seq_lens, slot_mappings, bb, cb
            )

        with nvtx_range("forward"):
            self.capture_one(bb, cb)
            self.replay(buf, bb, cb)

        with nvtx_range("sample"):
            sampled_tokens = self.sampler(
                buf.logits[: len(seqs)], None, scheduler_output
            )
        return sampled_tokens

    @torch.inference_mode()
    def _execute_eager(self, scheduler_output) -> dict[int, int]:
        with nvtx_range("prepare_inputs"):
            input_ids, positions, attn_metadata = self.prepare_inputs(scheduler_output)

        with nvtx_range("forward"), set_forward_context(attn_metadata):
            hidden_states = self.model(input_ids, positions)

        last_indices = [end - 1 for end in attn_metadata.query_start_loc_cpu[1:]]
        hidden_states = hidden_states[last_indices]

        with nvtx_range("sample"):
            logits = self.model.compute_logits(hidden_states)
            sampled_tokens = self.sampler(logits, attn_metadata, scheduler_output)
        return sampled_tokens

    @torch.inference_mode()
    def execute_model(self, scheduler_output) -> dict[int, int]:
        if self._can_graph_decode(scheduler_output):
            #decode
            return self._execute_decode_graph(scheduler_output)
        return self._execute_eager(scheduler_output)
```

test_attention.py

```python
import torch
import torch.nn.functional as F

from babyvllm.layers.attention import Attention, _naive_causal_attention, store_kvcache
from babyvllm.worker.context import AttentionMetadata, set_forward_context

def test_prefill_parity():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    q_len = 10
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    
    q = torch.randn(q_len, num_heads, head_dim)
    k = torch.randn(q_len, num_kv_heads, head_dim)
    v = torch.randn(q_len, num_kv_heads, head_dim)
    
    expected = attn(q, k, v)
    
    block_size = 4
    num_blocks = (q_len + block_size - 1) // block_size
    kv_cache = torch.empty(2, num_blocks, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    md = AttentionMetadata(
        slot_mapping=torch.arange(q_len),
        block_tables=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, q_len], dtype=torch.int32),
        seq_lens=torch.tensor([q_len], dtype=torch.int32),
        context_lens=torch.tensor([0], dtype=torch.int32),
        query_start_loc_cpu=[0, q_len],
        seq_lens_cpu=[q_len],
        context_lens_cpu=[0],
        max_query_len=q_len,
        max_seq_len=q_len,
    )
    
    with set_forward_context(md):
        actual = attn(q, k, v)
        
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

def test_prefill_then_decode_parity():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    
    q_all = torch.randn(11, num_heads, head_dim)
    k_all = torch.randn(11, num_kv_heads, head_dim)
    v_all = torch.randn(11, num_kv_heads, head_dim)
    
    expected = _naive_causal_attention(q_all, k_all, v_all, attn.scale, attn.num_queries_per_kv)
    
    block_size = 4
    kv_cache = torch.empty(2, 3, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    store_kvcache(k_all[:10], v_all[:10], kv_cache, torch.arange(10))
    
    q_decode = q_all[10:11]
    k_decode = k_all[10:11]
    v_decode = v_all[10:11]
    
    md = AttentionMetadata(
        slot_mapping=torch.tensor([10]),
        block_tables=torch.tensor([[0, 1, 2]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([11], dtype=torch.int32),
        context_lens=torch.tensor([10], dtype=torch.int32),
        query_start_loc_cpu=[0, 1],
        seq_lens_cpu=[11],
        context_lens_cpu=[10],
        max_query_len=1,
        max_seq_len=11,
    )
    
    with set_forward_context(md):
        actual_decode = attn(q_decode, k_decode, v_decode)
        
    torch.testing.assert_close(actual_decode[0], expected[10], atol=1e-5, rtol=1e-5)


def test_batched_decode():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    block_size = 4

    attn = Attention(num_heads, head_dim, num_kv_heads)

    q0 = torch.randn(7, num_heads, head_dim)
    k0 = torch.randn(7, num_kv_heads, head_dim)
    v0 = torch.randn(7, num_kv_heads, head_dim)
    q1 = torch.randn(6, num_heads, head_dim)
    k1 = torch.randn(6, num_kv_heads, head_dim)
    v1 = torch.randn(6, num_kv_heads, head_dim)

    exp0 = _naive_causal_attention(q0, k0, v0, attn.scale, attn.num_queries_per_kv)
    exp1 = _naive_causal_attention(q1, k1, v1, attn.scale, attn.num_queries_per_kv)

    kv_cache = torch.empty(2, 8, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache

    store_kvcache(k0[:6], v0[:6], kv_cache, torch.arange(6))
    store_kvcache(k1[:5], v1[:5], kv_cache, torch.arange(16, 21))

    q = torch.cat([q0[6:7], q1[5:6]], dim=0)
    k = torch.cat([k0[6:7], k1[5:6]], dim=0)
    v = torch.cat([v0[6:7], v1[5:6]], dim=0)

    md = AttentionMetadata(
        slot_mapping=torch.tensor([6, 21]),
        block_tables=torch.tensor([
            [0, 1],
            [4, 5],
        ], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([7, 6], dtype=torch.int32),
        context_lens=torch.tensor([6, 5], dtype=torch.int32),
        query_start_loc_cpu=[0, 1, 2],
        seq_lens_cpu=[7, 6],
        context_lens_cpu=[6, 5],
        max_query_len=1,
        max_seq_len=7,
    )

    with set_forward_context(md):
        actual = attn(q, k, v)

    torch.testing.assert_close(actual[0], exp0[6], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[1], exp1[5], atol=1e-5, rtol=1e-5)


def test_batched_mixed():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    block_size = 4
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    kv_cache = torch.empty(2, 6, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    q0 = torch.randn(5, num_heads, head_dim)
    k0 = torch.randn(5, num_kv_heads, head_dim)
    v0 = torch.randn(5, num_kv_heads, head_dim)
    
    k1_ctx = torch.randn(3, num_kv_heads, head_dim)
    v1_ctx = torch.randn(3, num_kv_heads, head_dim)
    store_kvcache(k1_ctx, v1_ctx, kv_cache, torch.arange(8, 11))
    
    q1 = torch.randn(1, num_heads, head_dim)
    k1 = torch.randn(1, num_kv_heads, head_dim)
    v1 = torch.randn(1, num_kv_heads, head_dim)
    
    k2_ctx = torch.randn(8, num_kv_heads, head_dim)
    v2_ctx = torch.randn(8, num_kv_heads, head_dim)
    store_kvcache(k2_ctx, v2_ctx, kv_cache, torch.arange(12, 20))
    
    q2 = torch.randn(1, num_heads, head_dim)
    k2 = torch.randn(1, num_kv_heads, head_dim)
    v2 = torch.randn(1, num_kv_heads, head_dim)
    
    q_batch = torch.cat([q0, q1, q2], dim=0)
    k_batch = torch.cat([k0, k1, k2], dim=0)
    v_batch = torch.cat([v0, v1, v2], dim=0)
    
    md = AttentionMetadata(
        slot_mapping=torch.tensor([0,1,2,3,4, 11, 20]),
        block_tables=torch.tensor([
            [0, 1, -1],
            [2, -1, -1],
            [3, 4, 5]
        ], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 5, 6, 7], dtype=torch.int32),
        seq_lens=torch.tensor([5, 4, 9], dtype=torch.int32),
        context_lens=torch.tensor([0, 3, 8], dtype=torch.int32),
        query_start_loc_cpu=[0, 5, 6, 7],
        seq_lens_cpu=[5, 4, 9],
        context_lens_cpu=[0, 3, 8],
        max_query_len=5,
        max_seq_len=9,
    )
    
    with set_forward_context(md):
        actual = attn(q_batch, k_batch, v_batch)
        
    exp0 = _naive_causal_attention(q0, k0, v0, attn.scale, attn.num_queries_per_kv)
    
    exp1 = _naive_causal_attention(
        torch.cat([torch.zeros(3, num_heads, head_dim), q1], dim=0),
        torch.cat([k1_ctx, k1], dim=0),
        torch.cat([v1_ctx, v1], dim=0),
        attn.scale, attn.num_queries_per_kv
    )
    
    exp2 = _naive_causal_attention(
        torch.cat([torch.zeros(8, num_heads, head_dim), q2], dim=0),
        torch.cat([k2_ctx, k2], dim=0),
        torch.cat([v2_ctx, v2], dim=0),
        attn.scale, attn.num_queries_per_kv
    )
    
    torch.testing.assert_close(actual[0:5], exp0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[5:6], exp1[3:4], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(actual[6:7], exp2[8:9], atol=1e-5, rtol=1e-5)

def test_non_contiguous():
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    block_size = 2
    
    attn = Attention(num_heads, head_dim, num_kv_heads)
    
    q_all = torch.randn(5, num_heads, head_dim)
    k_all = torch.randn(5, num_kv_heads, head_dim)
    v_all = torch.randn(5, num_kv_heads, head_dim)
    
    expected = _naive_causal_attention(q_all, k_all, v_all, attn.scale, attn.num_queries_per_kv)
    
    kv_cache = torch.empty(2, 10, block_size, num_kv_heads, head_dim)
    attn.kv_cache = kv_cache
    
    slot_mapping = torch.tensor([
        7*2 + 0, 7*2 + 1,
        2*2 + 0, 2*2 + 1,
        5*2 + 0
    ])
    
    md = AttentionMetadata(
        slot_mapping=slot_mapping,
        block_tables=torch.tensor([[7, 2, 5]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        context_lens=torch.tensor([0], dtype=torch.int32),
        query_start_loc_cpu=[0, 5],
        seq_lens_cpu=[5],
        context_lens_cpu=[0],
        max_query_len=5,
        max_seq_len=5,
    )
    
    with set_forward_context(md):
        actual = attn(q_all, k_all, v_all)
        
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    test_prefill_parity()
    test_prefill_then_decode_parity()
    test_batched_decode()
    test_batched_mixed()
    test_non_contiguous()
    print("All attention tests passed!")
```

test_generation.py

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from babyvllm.llm import LLM
from babyvllm.sequence import SamplingParams

def test_e2e_generation():
    MODEL = "Qwen/Qwen2-0.5B"
    prompts = [
        "The capital of France is",
        "A recipe for chocolate cake:"
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    
    hf_outputs = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out_ids = hf_model.generate(ids, max_new_tokens=10, do_sample=False)
        out_ids = out_ids[0][ids.shape[1]:]
        hf_outputs.append(out_ids.tolist())
        
    my_llm = LLM(MODEL)
    params = SamplingParams(temperature=0.0, max_tokens=10)
    my_outputs = my_llm.generate(prompts, sampling_params=params)
    
    my_outputs_sorted = sorted(my_outputs, key=lambda x: x.seq_id)
    
    for i in range(len(prompts)):
        print(f"--- Prompt: {prompts[i]} ---")
        print(f"HF Output: {hf_outputs[i]}")
        print(f"My Output: {my_outputs_sorted[i].new_token_ids}")
        assert hf_outputs[i] == my_outputs_sorted[i].new_token_ids
        
    print("End-to-end greedy generation test passed!")

if __name__ == "__main__":
    test_e2e_generation()
```

test_parity.py

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

from babyvllm.config import ModelConfig
from babyvllm.models.qwen2 import Qwen2ForCausalLM
from babyvllm.worker.loader import load_model

def test_parity():
    MODEL = "Qwen/Qwen2-0.5B"
    hf = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    
    path = snapshot_download(repo_id=MODEL)
    
    cfg = ModelConfig()
    mine = Qwen2ForCausalLM(cfg)
    load_model(mine, path)
    mine.eval()

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok("The capital of France is", return_tensors="pt").input_ids[0]
    
    with torch.inference_mode():
        ref = hf(ids[None]).logits[0]
        hidden = mine(ids, torch.arange(len(ids)))
        got = mine.compute_logits(hidden)
        
    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
    print("Logit parity test passed!")

if __name__ == "__main__":
    test_parity()
```

test_sampler.py

```python
import torch
import torch.nn.functional as F
from unittest.mock import patch
from dataclasses import dataclass

from babyvllm.layers.sampler import Sampler
from babyvllm.sequence import Sequence, SamplingParams

def brute_force_top_p(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_sum = 0.0
    allowed_indices = []
    
    for i in range(len(sorted_probs)):
        allowed_indices.append(sorted_indices[i].item())
        cum_sum += sorted_probs[i].item()
        if cum_sum > top_p:
            break
            
    out_probs = torch.zeros_like(probs)
    for idx in allowed_indices:
        out_probs[idx] = probs[idx]
        
    return out_probs / out_probs.sum()

@dataclass
class MockSchedulerOutput:
    scheduled_sequences: list

def test_top_p_sampling():
    torch.manual_seed(42)
    sampler = Sampler(vocab_size=1000)
    
    for _ in range(100):
        logits = torch.randn(1, 1000)
        probs = F.softmax(logits, dim=-1)
        
        top_p = torch.rand(1).item() * 0.9 + 0.1
        expected = brute_force_top_p(probs[0], top_p)
        
        seq = Sequence([1], sampling_params=SamplingParams(top_p=top_p, temperature=1.0))
        scheduler_out = MockSchedulerOutput([seq])
        
        with patch("torch.multinomial") as mock_multinomial:
            mock_multinomial.return_value = torch.tensor([[0]])
            sampler(logits, None, scheduler_out)
            actual_probs = mock_multinomial.call_args[0][0]
            
        torch.testing.assert_close(actual_probs, expected, rtol=1e-5, atol=1e-5)

def test_top_p_edge_cases():
    sampler = Sampler(vocab_size=4)
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    logits = torch.log(probs)
    
    # top_p = 0.1
    expected = brute_force_top_p(probs[0], 0.1)
    seq = Sequence([1], sampling_params=SamplingParams(top_p=0.1, temperature=1.0))
    scheduler_out = MockSchedulerOutput([seq])
    with patch("torch.multinomial") as mock_multinomial:
        mock_multinomial.return_value = torch.tensor([[0]])
        sampler(logits, None, scheduler_out)
        actual = mock_multinomial.call_args[0][0]
    torch.testing.assert_close(actual, expected)
    
    # top_p = 1.0
    expected = brute_force_top_p(probs[0], 1.0)
    seq = Sequence([1], sampling_params=SamplingParams(top_p=1.0, temperature=1.0))
    scheduler_out = MockSchedulerOutput([seq])
    with patch("torch.multinomial") as mock_multinomial:
        mock_multinomial.return_value = torch.tensor([[0]])
        sampler(logits, None, scheduler_out)
        actual = mock_multinomial.call_args[0][0]
    torch.testing.assert_close(actual, expected)
```

graph_runner.py

```python
from __future__ import annotations

import torch


class CUDAGraphRunner:

    def __init__(self):
        self._graph: torch.cuda.CUDAGraph | None = None

    @property
    def pool(self):
        if self._graph is None:
            raise RuntimeError("Call capture() first")
        return self._graph.pool()

    def capture(self, fn, buf, batch_bucket: int, ctx_bucket: int, pool=None):
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=pool):
            fn(buf, batch_bucket, ctx_bucket)

    def replay(self):
        if self._graph is None:
            raise RuntimeError("Call capture() before replay()")
        self._graph.replay()
```

test_cuda_graphs.py

```python
from babyvllm.worker.model_runner import BATCH_BUCKETS, CTX_BUCKETS, ceil_to_bucket


def test_ceil_to_bucket():
    assert ceil_to_bucket(1, BATCH_BUCKETS) == 1
    assert ceil_to_bucket(3, BATCH_BUCKETS) == 4
    assert ceil_to_bucket(64, BATCH_BUCKETS) == 64
    assert ceil_to_bucket(65, BATCH_BUCKETS) is None
    assert ceil_to_bucket(128, CTX_BUCKETS) == 256
    assert ceil_to_bucket(256, CTX_BUCKETS) == 256
    assert ceil_to_bucket(257, CTX_BUCKETS) == 2048
    assert ceil_to_bucket(2049, CTX_BUCKETS) is None
```

paged_attn.py

```python
import torch
import triton
import triton.language as tl

@triton.jit
def paged_decode_attn_kernel(
    q_ptr, k_cache, v_cache, out_ptr,
    block_tables, seq_lens, scale,
    stride_qs, stride_qh, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_os, stride_oh, stride_od,
    stride_bts, stride_btb, stride_sl,
    Q_PER_KV: tl.constexpr, Q_BLOCK: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    s = tl.program_id(0)
    kvh = tl.program_id(1)
    
    seq_len = tl.load(seq_lens + s * stride_sl)
    if seq_len == 0:
        return

    offs_q = tl.arange(0, Q_BLOCK)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_q < Q_PER_KV

    q = tl.load(
        q_ptr
        + s * stride_qs
        + (kvh * Q_PER_KV + offs_q)[:, None] * stride_qh
        + offs_d[None, :] * stride_qd,
        mask=q_mask[:, None],
        other=0.0,
    )

    m = tl.full([Q_BLOCK], float("-inf"), dtype=tl.float32)
    l = tl.zeros([Q_BLOCK], dtype=tl.float32)
    acc = tl.zeros([Q_BLOCK, HEAD_DIM], dtype=tl.float32)

    n_blocks = tl.minimum(tl.cdiv(seq_len, BLOCK), MAX_BLOCKS)
    
    bt_ptr = block_tables + s * stride_bts
    phys = tl.load(bt_ptr)
    
    offs_n = tl.arange(0, BLOCK)
    
    for b in range(0, n_blocks):
        # Load next phys early to hide latency (software pipelining block table)
        next_phys = tl.load(bt_ptr + (b + 1) * stride_btb, mask=(b + 1) < n_blocks, other=0)
        
        k = tl.load(
            k_cache
            + phys * stride_kb
            + offs_n[:, None] * stride_kt
            + kvh * stride_kh
            + offs_d[None, :] * stride_kd,
        )
        v = tl.load(
            v_cache
            + phys * stride_vb
            + offs_n[:, None] * stride_vt
            + kvh * stride_vh
            + offs_d[None, :] * stride_vd,
        )
        
        # Use tl.dot which runs on Tensor Cores!
        scores = tl.dot(q, tl.trans(k)) * scale
        
        token = b * BLOCK + offs_n
        mask = (token[None, :] < seq_len) & q_mask[:, None]
        scores = tl.where(mask, scores, float("-inf"))

        m_new = tl.maximum(m, tl.max(scores, axis=1))
        alpha = tl.exp(m - m_new)
        p = tl.exp(scores - m_new[:, None])
        
        # Use tl.dot for the value accumulation as well!
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        
        l = l * alpha + tl.sum(p, axis=1)
        m = m_new
        
        phys = next_phys

    l_safe = tl.where(l == 0, 1.0, l)
    out = acc / l_safe[:, None]
    
    tl.store(
        out_ptr
        + s * stride_os
        + (kvh * Q_PER_KV + offs_q)[:, None] * stride_oh
        + offs_d[None, :] * stride_od,
        out.to(out_ptr.dtype.element_ty),
        mask=q_mask[:, None],
    )


def paged_decode_attention(q, kv_cache, block_tables, seq_lens, scale):
    B, n_heads, head_dim = q.shape
    n_kv_heads = kv_cache.shape[3]
    block_size = kv_cache.shape[2]
    
    q_per_kv = n_heads // n_kv_heads
    
    # Pad Q_BLOCK to 16 for Tensor Cores (tl.dot requires >= 16)
    q_block = 16 if q_per_kv <= 16 else triton.next_power_of_2(q_per_kv)
    
    out = torch.empty_like(q)
    k_cache = kv_cache[0]
    v_cache = kv_cache[1]
    
    grid = (B, n_kv_heads)
    
    paged_decode_attn_kernel[grid](
        q, k_cache, v_cache, out,
        block_tables, seq_lens, float(scale),
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        block_tables.stride(0), block_tables.stride(1), seq_lens.stride(0),
        Q_PER_KV=q_per_kv,
        Q_BLOCK=q_block,
        HEAD_DIM=head_dim,
        BLOCK=block_size,
        MAX_BLOCKS=block_tables.shape[1],
        num_warps=4,
    )
    return out
```

test_scheduler.py

```python
import pytest
from babyvllm.scheduler import Scheduler
from babyvllm.sequence import Sequence
from babyvllm.kv_cache_manager import KVCacheManager

def test_scheduler_preempt_no_double_schedule():
    kv_cache = KVCacheManager(num_blocks=2, block_size=1)
    scheduler = Scheduler(kv_cache_manager=kv_cache, max_num_batched_tokens=32, max_num_seqs=2)
    
    seq1 = Sequence([1])
    seq2 = Sequence([2])
    
    scheduler.add_seq(seq1)
    scheduler.add_seq(seq2)
    
    out1 = scheduler.schedule()
    assert len(out1.scheduled_sequences) == 2
    assert len(scheduler.running) == 2
    
    scheduler.update_from_output(out1, {seq1.seq_id: 3, seq2.seq_id: 4})
    
    out2 = scheduler.schedule()
    
    try:
        scheduler.update_from_output(out2, {seq1.seq_id: 5})
    except ValueError as e:
        pytest.fail(f"Double scheduling bug occurred: {e}")
        
    assert len(out2.scheduled_sequences) == 1

import random
from babyvllm.engine import LLMEngine
from babyvllm.sequence import SamplingParams, SequenceFinishReason

class FakeModelRunner:
    def __init__(self, block_size=16):
        self.block_size = block_size
        self.num_blocks = 100
        self.eos_id = 999
        
    def determine_num_blocks(self):
        return self.num_blocks
        
    def execute_model(self, scheduler_output):
        sampled_tokens = {}
        for seq in scheduler_output.scheduled_sequences:
            token = (seq.seq_id + len(seq)) % 1000
            if len(seq) >= 15:
                token = self.eos_id
            sampled_tokens[seq.seq_id] = token
        return sampled_tokens

import itertools

def run_stress_test(num_blocks):
    Sequence._counter = itertools.count()
    runner = FakeModelRunner(block_size=4)
    runner.num_blocks = num_blocks
    engine = LLMEngine(model_runner=runner, max_num_batched_tokens=32, max_num_seqs=8)
    
    original_schedule = engine.scheduler.schedule
    def hooked_schedule():
        out = original_schedule()
        seq_ids = [s.seq_id for s in out.scheduled_sequences]
        assert len(seq_ids) == len(set(seq_ids)), "Duplicate sequence in scheduled_sequences!"
        
        for seq in out.scheduled_sequences:
            needed = out.num_scheduled_tokens[seq.seq_id]
            assert len(out.slot_mappings[seq.seq_id]) == needed, "Slot mapping length mismatch!"
        return out
    engine.scheduler.schedule = hooked_schedule
    
    random.seed(42)
    for i in range(20):
        prompt_len = random.randint(1, 10)
        prompt = [random.randint(0, 100) for _ in range(prompt_len)]
        max_tokens = random.randint(5, 20)
        engine.add_request(prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=True))
        
    outputs = {}
    while engine.has_unfinished_requests():
        step_outputs = engine.step()
        for out in step_outputs:
            if out.seq_id not in outputs:
                outputs[out.seq_id] = []
            outputs[out.seq_id].extend(out.new_token_ids)
            
        assert engine.kv_cache_manager.num_free_blocks + len(engine.kv_cache_manager.used_block_ids) == runner.num_blocks
        
        for seq in engine.scheduler.running:
            assert seq.num_computed_tokens <= len(seq)
            
    return outputs

def test_invariant_stress_test():
    out_large = run_stress_test(num_blocks=1000)
    out_small = run_stress_test(num_blocks=10)
    
    assert len(out_large) == 20
    assert len(out_small) == 20
    
    large_vals = [out_large[k] for k in sorted(out_large.keys())]
    small_vals = [out_small[k] for k in sorted(out_small.keys())]
    assert large_vals == small_vals, "Outputs mismatch under preemption!"

def test_eos_stop():
    runner = FakeModelRunner(block_size=4)
    engine = LLMEngine(model_runner=runner, max_num_batched_tokens=32, max_num_seqs=8)
    
    prompt = [1, 2, 3]
    engine.add_request(prompt, SamplingParams(max_tokens=50, stop_tokens={runner.eos_id}, ignore_eos=False))
    
    outputs = engine.run()
    seq_out = list(outputs.values())[0]
    
    assert seq_out.finish_reason == SequenceFinishReason.STOP
    assert runner.eos_id not in seq_out.new_token_ids
```
