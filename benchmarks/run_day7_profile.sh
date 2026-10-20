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

# batch:context. Override with SHAPES="1:128" to reproduce the original session.
# 64:1920 is the Day 6 point whose memory-sizing pass used to OOM.
# 1:2048 is the longest context the decode graphs capture.
if [[ -n "${SHAPES:-}" ]]; then
    read -r -a SHAPE_LIST <<< "$SHAPES"
else
    SHAPE_LIST=("1:128" "64:1920" "1:2048")
fi

echo "============================================"
echo "Day 7 Profiling Session"
echo "Model: $MODEL"
echo "Output: $OUTDIR/"
echo "Shapes: ${SHAPE_LIST[*]}"
echo "============================================"

for shape in "${SHAPE_LIST[@]}"; do
    BATCH="${shape%%:*}"
    CONTEXT="${shape##*:}"
    TAG="b${BATCH}_ctx${CONTEXT}"

    echo ""
    echo "──── Timing-only  B=${BATCH} ctx=${CONTEXT} ────"
    python benchmarks/profile_day7.py \
        --model "$MODEL" \
        --batch "$BATCH" --context "$CONTEXT" \
        --decode-steps 100 --warmup-steps 30 \
        --out "$OUTDIR/timing_${TAG}.json"

    echo ""
    echo "──── nsys capture  B=${BATCH} ctx=${CONTEXT} ────"
    nsys profile \
        --cuda-graph-trace=node \
        -t cuda,nvtx \
        --force-overwrite=true \
        -o "$OUTDIR/nsys_${TAG}" \
        python benchmarks/profile_day7.py \
            --model "$MODEL" \
            --batch "$BATCH" --context "$CONTEXT" \
            --decode-steps 30 --warmup-steps 20 \
            --out /dev/null

    echo "→ Exporting nsys_${TAG}.nsys-rep → .sqlite"
    nsys export -t sqlite \
        --force-overwrite=true \
        -o "$OUTDIR/nsys_${TAG}.sqlite" \
        "$OUTDIR/nsys_${TAG}.nsys-rep"

    echo ""
    echo "========== B=${BATCH}, ctx=${CONTEXT} KERNEL BREAKDOWN =========="
    python benchmarks/profile_day7.py \
        --parse "$OUTDIR/nsys_${TAG}.sqlite" \
        --out "$OUTDIR/breakdown_${TAG}.json"
done

echo ""
echo "============================================"
echo "Day 7 profiling complete. Files in $OUTDIR/"
echo "============================================"
