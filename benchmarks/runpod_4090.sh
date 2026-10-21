#!/usr/bin/env bash
# Same comparison as run_h100_comparison.sh.
# A 4090 result is a smoke test. It is not a published baby-vLLM number.
# Does not apt-install anything and does not pip-install torch.
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-/workspace/hf}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

python - << 'PY'
import importlib.util
import subprocess
import sys

print("torch before:", end=" ")
subprocess.check_call(
    [sys.executable, "-c", "import torch; print(torch.__version__, 'cuda', torch.version.cuda)"]
)
missing = [
    pkg for pkg in ("transformers", "accelerate", "safetensors", "huggingface_hub")
    if importlib.util.find_spec(pkg) is None
]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
if importlib.util.find_spec("vllm") is None:
    print("Installing vLLM. This does not install torch first.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "vllm"])
    print("torch after vLLM install:", end=" ")
    subprocess.check_call(
        [sys.executable, "-c", "import torch; print(torch.__version__, 'cuda', torch.version.cuda)"]
    )
PY

MODEL="${1:-Qwen/Qwen2.5-7B}"
BATCHES="${2:-1,8,32,64}"
PROMPT_LEN="${3:-128}"
MAX_TOKENS="${4:-64}"

echo "Model: $MODEL | Batches: $BATCHES | Prompt: $PROMPT_LEN | Max tokens: $MAX_TOKENS"
echo "This script does not publish. The ladder is Qwen2.5-7B, bf16, H100."

python benchmarks/benchmark_vs_vllm.py \
  --model "$MODEL" \
  --batches "$BATCHES" \
  --prompt-len "$PROMPT_LEN" \
  --max-tokens "$MAX_TOKENS" \
  --dtype bfloat16 \
  --out benchmark_comparison.json
