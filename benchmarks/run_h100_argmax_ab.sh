#!/usr/bin/env bash
# Greedy argmax inside the decode graph: fp32 vs bf16 vs Triton, batch 64.
# Qwen2.5-7B, bf16, one H100. Same pod, same PyTorch for every number:
# vLLM is installed first, because installing it changes torch.
#
#   1. GPU tests for the kernel and all three graph modes (Qwen2.5-0.5B).
#   2. profile_decode_host.py per mode, two rounds, order rotated.
#   3. benchmark_vs_vllm.py at batch 64: baby-vLLM once per mode, vLLM once.
#
# Usage: bash benchmarks/run_h100_argmax_ab.sh 2>&1 | tee run_argmax.log
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/workspace/hf}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PIP_BREAK_SYSTEM_PACKAGES=1

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
OUT="results_$(date +%d%b%Y)_H100_argmax"
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
python -m pytest tests/test_argmax_kernel.py tests/test_decode_graph.py -q -m gpu 2>&1 | tee "$OUT/gpu_tests.log"

echo "== 2. decode step profile per argmax mode =="
for round in 1 2; do
  if [ "$round" = 1 ]; then modes="fp32 bf16 triton"; else modes="triton bf16 fp32"; fi
  for mode in $modes; do
    BABYVLLM_GREEDY_ARGMAX=$mode python benchmarks/profile_decode_host.py --model "$MODEL" \
      --label "$mode round $round" --out "$OUT/decode_profile_${mode}_r$round.json" \
      2>&1 | tee "$OUT/decode_profile_${mode}_r$round.log"
  done
done

echo "== 3. batch 64 throughput: baby-vLLM per mode, then vLLM once =="
for mode in fp32 bf16 triton; do
  BABYVLLM_GREEDY_ARGMAX=$mode python benchmarks/benchmark_vs_vllm.py --model "$MODEL" \
    --batches 64 --prompt-len 128 --max-tokens 64 --repeats 3 --dtype bfloat16 \
    --skip-hf --skip-vllm --out "$OUT/comparison_baby_${mode}.json" \
    2>&1 | tee "$OUT/comparison_baby_${mode}.log"
done
python benchmarks/benchmark_vs_vllm.py --model "$MODEL" \
  --batches 64 --prompt-len 128 --max-tokens 64 --repeats 3 --dtype bfloat16 \
  --skip-hf --skip-babyvllm --out "$OUT/comparison_vllm.json" \
  2>&1 | tee "$OUT/comparison_vllm.log"

echo "Done. Results in $OUT/"
