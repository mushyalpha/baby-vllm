#!/usr/bin/env bash
# Day 7 profiling — run on the H100.
#
# Two passes per shape. They are not interchangeable:
#   Pass 1  timing only (no nsys). The p50 from this pass is the latency.
#   Pass 2  nsys, for kernel durations and kernel counts only.
#           --cuda-graph-trace=node inflates wall time, so the NVTX window
#           is never the reported latency.
#   Then    clean scheduling overhead = pass-1 p50 − pass-2 active kernel time.
#
# Output names include the shape and, for the Inductor baseline, a _compile
# suffix, so one run cannot overwrite another:
#   day7/timing_b1_ctx128.json
#   day7/timing_b1_ctx128_run1.json
#   day7/nsys_b1_ctx128.sqlite
#   day7/breakdown_b1_ctx128.json
#
# Knobs:
#   SHAPES="1:128"    subset of the 2×2 grid
#   REPEATS=5         timing trials (default 3). The nsys pass stays at 1.
#   RUN_COMPILE=1     also run every shape with --compile
#   OUTDIR=day7

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B}"
OUTDIR="${OUTDIR:-day7}"
REPEATS="${REPEATS:-3}"
RUN_COMPILE="${RUN_COMPILE:-0}"
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

# batch:context. 1:2048 and 64:1920 sit on the 2048-token CUDA-graph bucket.
# profile_day7.py shortens the timed window when warmup + decode would walk
# off that bucket (a prompt of 2048 cannot itself be a graphed decode step).
if [[ -n "${SHAPES:-}" ]]; then
    read -r -a SHAPE_LIST <<< "$SHAPES"
else
    SHAPE_LIST=("1:128" "1:2048" "64:128" "64:1920")
fi

run_shapes() {
    local suffix="$1"
    local label="fused"
    if [[ "$suffix" == "_compile" ]]; then
        label="torch.compile inductor"
    fi

    local shape BATCH CONTEXT TAG timing_out
    for shape in "${SHAPE_LIST[@]}"; do
        BATCH="${shape%%:*}"
        CONTEXT="${shape##*:}"
        TAG="b${BATCH}_ctx${CONTEXT}"
        timing_out="$OUTDIR/timing_${TAG}${suffix}.json"

        echo ""
        echo "──── ${label}  timing  B=${BATCH} ctx=${CONTEXT}  repeats=${REPEATS} ────"
        # ${suffix:+--compile} is empty for the fused pass and --compile for
        # the Inductor pass. An empty array under `set -u` is not portable.
        python benchmarks/profile_day7.py \
            --model "$MODEL" \
            --batch "$BATCH" --context "$CONTEXT" \
            --decode-steps 100 --warmup-steps 30 \
            --repeats "$REPEATS" \
            ${suffix:+--compile} \
            --out "$timing_out"

        echo ""
        echo "──── ${label}  nsys  B=${BATCH} ctx=${CONTEXT}  (kernel times only) ────"
        # repeats=1: the trace is for kernel durations, not a latency sample.
        # --out /dev/null drops the perturbed step times so they cannot be
        # mistaken for the timing pass.
        nsys profile \
            --cuda-graph-trace=node \
            -t cuda,nvtx \
            --force-overwrite=true \
            -o "$OUTDIR/nsys_${TAG}${suffix}" \
            python benchmarks/profile_day7.py \
                --model "$MODEL" \
                --batch "$BATCH" --context "$CONTEXT" \
                --decode-steps 30 --warmup-steps 20 \
                --repeats 1 \
                ${suffix:+--compile} \
                --out /dev/null

        echo "→ Exporting nsys_${TAG}${suffix}.nsys-rep → .sqlite"
        nsys export -t sqlite \
            --force-overwrite=true \
            -o "$OUTDIR/nsys_${TAG}${suffix}.sqlite" \
            "$OUTDIR/nsys_${TAG}${suffix}.nsys-rep"

        echo ""
        echo "========== B=${BATCH}, ctx=${CONTEXT} (${label}) KERNEL BREAKDOWN =========="
        python benchmarks/profile_day7.py \
            --parse "$OUTDIR/nsys_${TAG}${suffix}.sqlite" \
            --timing-json "$timing_out" \
            --out "$OUTDIR/breakdown_${TAG}${suffix}.json"
    done
}

echo "============================================"
echo "Day 7 Profiling Session"
echo "Model: $MODEL"
echo "Output: $OUTDIR/"
echo "Shapes: ${SHAPE_LIST[*]}"
echo "Timing repeats: $REPEATS (nsys repeats: 1)"
if [[ "$RUN_COMPILE" == "1" ]]; then
    echo "Modes: fused, torch.compile inductor"
else
    echo "Modes: fused"
    echo "Inductor baseline is off. Re-run with RUN_COMPILE=1 to add --compile."
fi
echo "============================================"

run_shapes ""
if [[ "$RUN_COMPILE" == "1" ]]; then
    run_shapes "_compile"
fi

echo ""
echo "============================================"
echo "Day 7 profiling complete. Files in $OUTDIR/"
echo "============================================"
