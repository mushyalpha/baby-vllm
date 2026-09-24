set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install -q --upgrade pip
python -m pip install -q "transformers>=4.44" accelerate safetensors huggingface_hub

MODE="${1:-full}"
case "$MODE" in
  quick)
    python benchmarks/benchmark_babyvllm.py --quick --out babyvllm_bench.json
    ;;
  nsys)
    python benchmarks/benchmark_babyvllm.py --nsys-focus --out babyvllm_bench.json
    ;;
  full)
    python benchmarks/benchmark_babyvllm.py --vs-hf --out babyvllm_bench.json
    ;;
  *)
    echo "usage: $0 [quick|nsys|full]" >&2
    exit 1
    ;;
esac
