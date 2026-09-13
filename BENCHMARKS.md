# Benchmarks

**Rule:** every published number is **Qwen2.5-7B, bf16, H100 SXM 80GB** (3.352 TB/s HBM, 132 SMs). `Qwen2.5-0.5B` is CPU tests / kernel iteration only and never appears in a table here.

Days 2–3 and 6 ran on 0.5B because baby-vLLM's config was hardcoded to it; Days 4–5 ran on 7B because that's the model worth an H100. Those numbers were never on the same ladder. From this file forward the engine reads config from the checkpoint, and later days append one row to the table below.

HBM floor uses 15.23 GB bf16 weights. B=1 ctx=128 → **4.55 ms/step**. B=64 ctx=2048 adds ~7.5 GB KV read → ~6.8 ms.

Day 4 HF Transformers numbers were measured in `mistral7B-inference` (archived). Day 5 CUDA-graph numbers came from a static harness in that same repo (`static_engine.py`), not from `baby-vllm` itself. Day 6 folded graphs into `babyvllm/worker/model_runner.py`. The empty baby-vllm rows are Day 7's re-baseline: one model, one repo, one harness.

| path | B=1 ctx=128 | B=1 ctx=2048 | B=64 ctx=2048 | %floor (B=1) |
| --- | --- | --- | --- | --- |
| HBM floor (15.23 GB weights, 3.352 TB/s) | 4.55 ms | 4.58 ms | ~6.8 ms (weights + 7.5 GB KV) | 100% |
| HF Transformers (Day 4) | 14.43 | 14.33 | 22.57 | 32% |
| baby-vLLM eager, torch paged attn | | | | |
| baby-vLLM + CUDA graphs (Day 5 mechanism, now in-repo) | | | | |
| baby-vLLM + graphs + Triton paged decode (Day 6 mechanism) | | | | |
| vLLM (sanity anchor, still owed from Day 4) | | | | |

Day 5's published 9.5 ms was the static harness. When the in-repo graph runner is measured on 7B: if it lands near 9.5 ms, Day 5's harness numbers reproduce in the engine proper; if not, that delta is a finding.

B=64 step times are not quotable until the sampler's per-seq `.item()` (64 host syncs/step) is removed.

See `PROJECT_STATE.md` for the rule, 7B facts, and open kernel work.
