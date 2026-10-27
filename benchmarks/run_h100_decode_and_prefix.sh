#!/usr/bin/env bash
# One H100 pod, Qwen2.5-7B, bf16. Produces every number for two write-ups:
#   1. batch-64 decode host overhead: profile before and after, then the
#      three-engine comparison again.
#   2. prefix caching on and off with a shared system prompt.
#
# Same pod for everything. The "before" profile runs the code at BASELINE_REF
# from a throwaway git worktree, so before and after differ only in the code.
# Order is before, after, before, after so drift cannot sit on one side.
#
# Does not apt-install anything and does not pip-install torch.
# Usage: bash benchmarks/run_h100_decode_and_prefix.sh
#   BASELINE_REF=2ee611c   commit that has the old decode path
#   SKIP_VLLM=1            skip the comparison (profile and prefix only)
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/workspace/hf}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
BASELINE_REF="${BASELINE_REF:-2ee611c}"
OUT="results_$(date +%d%b%Y)_H100"
mkdir -p "$OUT"

python - << 'PY'
import torch
name = torch.cuda.get_device_name(0)
print("GPU:", name, "| torch", torch.__version__)
assert "H100" in name, f"published numbers need an H100, got {name}"
PY

python -c "import pytest, accelerate" 2>/dev/null || python -m pip install -q pytest accelerate

echo "== 0. GPU unit tests (Qwen2.5-0.5B, correctness only, nothing here is published) =="
python -m pytest tests/test_decode_graph.py tests/test_paged_prefill.py -q -m gpu 2>&1 | tee "$OUT/gpu_tests.log"

echo "== 1. baseline worktree at $BASELINE_REF =="
BASE_DIR="$(mktemp -d)/baseline"
git worktree add --detach "$BASE_DIR" "$BASELINE_REF"
trap 'git worktree remove --force "$BASE_DIR" || true' EXIT
cp benchmarks/profile_decode_host.py "$BASE_DIR/benchmarks/"
cp benchmarks/benchmark_vs_vllm.py "$BASE_DIR/benchmarks/"

echo "== 2. decode step profile, before / after / before / after =="
for round in 1 2; do
  (cd "$BASE_DIR" && python benchmarks/profile_decode_host.py --model "$MODEL" \
      --label "before $BASELINE_REF round $round" \
      --out "$OLDPWD/$OUT/decode_profile_before_r$round.json" $([ "$round" = 1 ] && echo --cprofile)) \
      2>&1 | tee "$OUT/decode_profile_before_r$round.log"
  python benchmarks/profile_decode_host.py --model "$MODEL" \
      --label "after $(git rev-parse --short HEAD)+worktree round $round" \
      --out "$OUT/decode_profile_after_r$round.json" $([ "$round" = 1 ] && echo --cprofile) \
      2>&1 | tee "$OUT/decode_profile_after_r$round.log"
done

if [ -z "${SKIP_VLLM:-}" ]; then
  echo "== 3. HF vs baby-vLLM vs vLLM, same command as the 26 October run =="
  bash benchmarks/run_h100_comparison.sh "$MODEL" 1,8,32,64 128 64 2>&1 | tee "$OUT/comparison.log"
  mv h100_vllm_comparison_r3.json "$OUT/comparison_graph_argmax_incremental.json"
fi

echo "== 4. prefix caching on and off =="
python benchmarks/benchmark_prefix_caching.py --model "$MODEL" \
  --system-lens 128,512,1024,1536 --suffix-len 64 --n-seq 16 --burst 32 \
  --burst-max-tokens 32 --repeats 3 \
  --out "$OUT/prefix_caching.json" 2>&1 | tee "$OUT/prefix_caching.log"

echo "Done. Results in $OUT/"
