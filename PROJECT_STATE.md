# baby-vLLM — project state (authoritative; supersedes older notes)

## Measurement substrate (RULE)
- Published numbers: **Qwen2.5-7B, bf16, H100 SXM 80GB (3.352 TB/s HBM, 132 SMs)**. No exceptions.
- Dev/test model: **Qwen2.5-0.5B** (fp32 on CPU). Never appears in a post table.
- Days 2–3, 6 were measured on 0.5B before this rule existed. Treated as mechanism-only; re-baselined on 7B on Day 7.

## Framing decisions
- Keep 0.5B for iteration speed, but run headline numbers on 7B bf16.
- `baby-vllm` is the only live repo. `mistral7B-inference` is archived Day 4 roofline + Day 5 harness.
- `ModelConfig` is always read from checkpoint `config.json` (`ModelConfig.from_hf`). Nothing is hardcoded to a model.
- **Every number that appears in a post is Qwen2.5-7B, bf16, H100 SXM. Qwen2.5-0.5B exists only for CPU unit tests and kernel-correctness iteration. It never appears in a published table.**

## Qwen2.5-7B facts (derive everything from these)
- 28 layers, hidden 3584, 28 Q heads, 4 KV heads, head_dim 128, intermediate 18944, vocab 152064, untied lm_head
- GQA ratio 7 (same as 0.5B)
- Weights ≈ 15.23 GB bf16 → B=1 decode floor **4.55 ms/step ≈ 220 tok/s**
- KV/token = 2·28·4·128·2 B = **56 KiB**; block(16) = 896 KiB
- B=64, ctx=2048 KV read/step = 7.5 GB → floor ≈ 6.8 ms

## Repo
- `baby-vllm` is the only live repo. `mistral7B-inference` = archived Day 4 roofline + Day 5 harness.
- ModelConfig is read from checkpoint config.json (`ModelConfig.from_hf`). Nothing is hardcoded to a model.

## Ladder (7B, B=1 ctx=128, ms/step) — append one row per day
| day | path | ms | %floor |
| 4 | HF Transformers | 14.43 | 32% |
| 5 | static harness + CUDA graphs (other repo) | 9.55 | 48% |
| 7 | baby-vllm eager | TBD | |
| 7 | baby-vllm + graphs | TBD | |
| 7 | baby-vllm + graphs + Triton | TBD | |

## Tracker
| day | work | status | notes |
| 5 | CUDA graphs | ✅ Done (7B, static harness in mistral7B-inference) | 21.2→9.5 ms B=1; 48% floor; 1951→2 launches |
| 6 | Triton paged decode kernel | ✅ Done (0.5B — mechanism only) | graph-replay bug fixed; kernel latency-bound not BW-bound; needs 7B re-run |
| 7 | 7B unification + re-baseline ladder | ⬜ Next | config.from_hf; one table HF/eager/graphs/Triton/vLLM on 7B; ncu on kernel |

## Tests
- Default CI: `pytest -m "not gpu"` — synthetic + Qwen2.5-0.5B fp32 CPU (`test_parity.py`, `test_generation.py`).
- 7B GPU: `BABYVLLM_RUN_7B=1 pytest tests/test_parity_7b.py -m gpu`

## Open items
- Day 6 kernel: latency-bound (128 blocks / 132 SMs, ~6% occ, 153 regs at hd=64; expect spills at hd=128). Fix order: ncu → control kernel → tl.dot w/ Q padded to 16 → break block-table pointer chase.
- vLLM sanity anchor still owed from Day 4.
- Sampler does per-seq `.item()` (64 syncs/step at B=64) — remove before quoting B=64 step times.
