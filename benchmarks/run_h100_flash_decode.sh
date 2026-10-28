#!/usr/bin/env bash
# Flash-decoding (split paged decode attention): before vs after.
# Qwen2.5-7B, bf16, one H100. Same pod, same commit, same PyTorch for every
# number: BABYVLLM_DECODE_PARTITION=0 is the single-pass kernel (before),
# "auto" splits when batch × KV heads leaves SMs idle (after).
# vLLM is installed first, because installing it changes torch.
#
#   1. GPU tests for the kernel and the decode graph (Qwen2.5-0.5B).
#   2. bench_decode_attn.py: one layer, every batch bucket, partition sweep.
#   3. run_day7_profile.sh per mode: timing pass + nsys kernel breakdown at
#      the Day 7 shapes plus batch 8, order rotated across two rounds.
#   4. benchmark_vs_vllm.py at a 1920-token prompt: baby-vLLM per mode, vLLM once.
#
# Usage: bash benchmarks/run_h100_flash_decode.sh 2>&1 | tee run_flash_decode.log
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/workspace/hf}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
OUT="results_$(date +%d%b%Y)_H100_flash_decode"
SHAPES="${SHAPES:-1:128 1:2048 8:1920 64:128 64:1920}"
mkdir -p "$OUT"

python -c "import pytest, accelerate" 2>/dev/null || python -m pip install -q pytest accelerate
python -c "import babyvllm" 2>/dev/null || python -m pip install -q -e .
if ! python -c "import vllm" 2>/dev/null; then
  # The RunPod image ships Debian's PyJWT without a pip RECORD.
  python -m pip install -q --ignore-installed PyJWT
  python -m pip install vllm || python -m pip install --ignore-installed vllm
fi

python - << 'PY'
import torch, vllm
name = torch.cuda.get_device_name(0)
print("GPU:", name, "| torch", torch.__version__, "| vLLM", vllm.__version__)
assert "H100" in name, f"published numbers need an H100, got {name}"
PY

echo "== 1. GPU tests =="
python -m pytest tests/test_decode_split.py tests/test_paged_attn.py tests/test_decode_graph.py \
  tests/test_cuda_graphs.py -q 2>&1 | tee "$OUT/gpu_tests.log"

echo "== 2. one-layer decode attention sweep =="
BABYVLLM_MODEL="$MODEL" python benchmarks/bench_decode_attn.py --out "$OUT/decode_attn_bench.json" \
  2>&1 | tee "$OUT/decode_attn_bench.log"

echo "== 3. decode step profile, single-pass (0) vs split (auto) =="
for round in 1 2; do
  if [ "$round" = 1 ]; then modes="0 auto"; else modes="auto 0"; fi
  for mode in $modes; do
    tag="partition_${mode}_r$round"
    BABYVLLM_DECODE_PARTITION=$mode OUTDIR="$OUT/$tag" SHAPES="$SHAPES" \
      bash benchmarks/run_day7_profile.sh "$MODEL" 2>&1 | tee "$OUT/$tag.log"
  done
done

echo "== 4. end to end at a 1920-token prompt: baby-vLLM per mode, then vLLM once =="
for mode in 0 auto; do
  BABYVLLM_DECODE_PARTITION=$mode python benchmarks/benchmark_vs_vllm.py --model "$MODEL" \
    --batches 1,8,64 --prompt-len 1920 --max-tokens 64 --repeats 3 --dtype bfloat16 \
    --skip-hf --skip-vllm --out "$OUT/comparison_baby_partition_${mode}.json" \
    2>&1 | tee "$OUT/comparison_baby_partition_${mode}.log"
done
python benchmarks/benchmark_vs_vllm.py --model "$MODEL" \
  --batches 1,8,64 --prompt-len 1920 --max-tokens 64 --repeats 3 --dtype bfloat16 \
  --skip-hf --skip-babyvllm --out "$OUT/comparison_vllm.json" \
  2>&1 | tee "$OUT/comparison_vllm.log"

echo "Done. Results in $OUT/"
