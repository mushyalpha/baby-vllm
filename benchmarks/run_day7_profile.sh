#!/usr/bin/env bash
# Day 7 Profiling Session — run on H100 pod
#
# This script runs the full Day 7 profiling session:
#   1. Timing-only baseline (no nsys overhead)
#   2. nsys capture B=1,ctx=128 with --cuda-graph-trace=node
#   3. nsys capture B=64,ctx=1920 (optional, cheap)
#   4. Export .sqlite and parse kernel breakdown
#
# Usage:
#   chmod +x benchmarks/run_day7_profile.sh
#   bash benchmarks/run_day7_profile.sh [MODEL]
#
# Defaults to Qwen/Qwen2.5-7B. Pass a local path if checkpoint is pre-downloaded.

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-7B}"
OUTDIR="day7"
mkdir -p "$OUTDIR"

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

echo ""
echo "→ B=64, ctx=1920 (your 6.8 ms floor comparison)"
python benchmarks/profile_day7.py \
    --model "$MODEL" \
    --batch 64 --context 1920 \
    --decode-steps 50 --warmup-steps 20 \
    --out "$OUTDIR/timing_b64_ctx1920.json"

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
# Step 3: nsys capture — B=64, ctx=1920
# ──────────────────────────────────────────────
echo ""
echo "──── Step 3: nsys capture B=64, ctx=1920 ────"

nsys profile \
    --cuda-graph-trace=node \
    -t cuda,nvtx \
    --force-overwrite=true \
    -o "$OUTDIR/nsys_b64_ctx1920" \
    python benchmarks/profile_day7.py \
        --model "$MODEL" \
        --batch 64 --context 1920 \
        --decode-steps 20 --warmup-steps 10 \
        --out /dev/null

# ──────────────────────────────────────────────
# Step 4: Export .sqlite and parse
# ──────────────────────────────────────────────
echo ""
echo "──── Step 4: Export and parse ────"

echo "→ Exporting B=1 .nsys-rep → .sqlite"
nsys export -t sqlite \
    --force-overwrite=true \
    -o "$OUTDIR/nsys_b1_ctx128.sqlite" \
    "$OUTDIR/nsys_b1_ctx128.nsys-rep"

echo "→ Exporting B=64 .nsys-rep → .sqlite"
nsys export -t sqlite \
    --force-overwrite=true \
    -o "$OUTDIR/nsys_b64_ctx1920.sqlite" \
    "$OUTDIR/nsys_b64_ctx1920.nsys-rep"

echo ""
echo "========== B=1, ctx=128 KERNEL BREAKDOWN =========="
python benchmarks/profile_day7.py \
    --parse "$OUTDIR/nsys_b1_ctx128.sqlite" \
    --out "$OUTDIR/breakdown_b1_ctx128.json"

echo ""
echo "========== B=64, ctx=1920 KERNEL BREAKDOWN =========="
python benchmarks/profile_day7.py \
    --parse "$OUTDIR/nsys_b64_ctx1920.sqlite" \
    --out "$OUTDIR/breakdown_b64_ctx1920.json"

# ──────────────────────────────────────────────
# Done
# ──────────────────────────────────────────────
echo ""
echo "============================================"
echo "Day 7 profiling complete. Files in $OUTDIR/"
echo ""
echo "Key files:"
echo "  $OUTDIR/timing_b1_ctx128.json      — raw step times (no nsys overhead)"
echo "  $OUTDIR/timing_b64_ctx1920.json     — raw step times (no nsys overhead)"
echo "  $OUTDIR/breakdown_b1_ctx128.json    — kernel-level split (GEMM/ew/attn)"
echo "  $OUTDIR/breakdown_b64_ctx1920.json  — kernel-level split (GEMM/ew/attn)"
echo "  $OUTDIR/nsys_b1_ctx128.nsys-rep     — open in Nsight Systems GUI"
echo "  $OUTDIR/nsys_b64_ctx1920.nsys-rep   — open in Nsight Systems GUI"
echo ""
echo "Next: read the breakdown tables above."
echo "  - If elementwise > 2ms → fuse RMSNorm+residual first"
echo "  - If GEMM dominates + you have >84 GEMMs → merge QKV/GateUp first"
echo "  - If idle > 1ms inside graph → something is serializing"
echo "============================================"
