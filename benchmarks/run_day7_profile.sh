#!/usr/bin/env bash
# Day 7 Profiling Session — run on H100 pod

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B}"
OUTDIR="day7"
mkdir -p "$OUTDIR"

# Auto-discover nsys if it's missing from the fresh pod PATH
if ! command -v nsys &> /dev/null; then
    echo "nsys not in PATH. Searching for it..."
    NSYS_PATH=$(find /opt/nvidia -name "nsys" -type f -executable 2>/dev/null | head -n 1 || true)
    if [ -n "$NSYS_PATH" ]; then
        export PATH=$PATH:$(dirname "$NSYS_PATH")
        echo "Found nsys and added to PATH: $(dirname "$NSYS_PATH")"
    else
        echo "Error: nsys not found. Please install Nsight Systems."
        exit 1
    fi
fi

echo "============================================"
echo "Day 7 Profiling Session"
echo "Model: $MODEL"
echo "Output: $OUTDIR/"
echo "============================================"

# ──────────────────────────────────────────────
# Step 1: Timing-only baselines (no nsys overhead)
# ──────────────────────────────────────────────
echo ""
echo "──── Step 1: Timing-only baselines ────"

echo "→ B=1, ctx=128 (your 4.55 ms floor comparison)"
python benchmarks/profile_day7.py \
    --model "$MODEL" \
    --batch 1 --context 128 \
    --decode-steps 100 --warmup-steps 30 \
    --out "$OUTDIR/timing_b1_ctx128.json"

# ──────────────────────────────────────────────
# Step 2: nsys capture — B=1, ctx=128
# ──────────────────────────────────────────────
echo ""
echo "──── Step 2: nsys capture B=1, ctx=128 ────"
echo "(--cuda-graph-trace=node expands graph replay into individual kernels)"

nsys profile \
    --cuda-graph-trace=node \
    -t cuda,nvtx \
    --force-overwrite=true \
    -o "$OUTDIR/nsys_b1_ctx128" \
    python benchmarks/profile_day7.py \
        --model "$MODEL" \
        --batch 1 --context 128 \
        --decode-steps 30 --warmup-steps 20 \
        --out /dev/null

# ──────────────────────────────────────────────
# Step 3: Export .sqlite and parse
# ──────────────────────────────────────────────
echo ""
echo "──── Step 3: Export and parse ────"

echo "→ Exporting B=1 .nsys-rep → .sqlite"
nsys export -t sqlite \
    --force-overwrite=true \
    -o "$OUTDIR/nsys_b1_ctx128.sqlite" \
    "$OUTDIR/nsys_b1_ctx128.nsys-rep"

echo ""
echo "========== B=1, ctx=128 KERNEL BREAKDOWN =========="
python benchmarks/profile_day7.py \
    --parse "$OUTDIR/nsys_b1_ctx128.sqlite" \
    --out "$OUTDIR/breakdown_b1_ctx128.json"

echo ""
echo "============================================"
echo "Day 7 profiling complete. Files in $OUTDIR/"
echo "============================================"
