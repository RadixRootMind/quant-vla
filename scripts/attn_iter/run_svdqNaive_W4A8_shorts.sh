#!/bin/bash
# Shorts-only orchestrator for SVDQ-naive W4A8 (block_size=0 / per-row only).
# Build 3 short-suite packs in parallel (GPUs 0-2), then eval all 3 in parallel
# on GPUs 0-7 packed with distinct PORT_BASE.
set -uo pipefail
REPO="${QUANTVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCR="$REPO/scripts/attn_iter"
LOG="$REPO/logs/svdqNaive_W4A8_b0"
mkdir -p "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/orchestrator.log"; }

# ===== Stage 1: build 3 shorts in parallel =====
say "Stage 1 — build 3 short packs (object→GPU0, spatial→GPU1, goal→GPU2), block_size=0"
SUITE=object  GPU=0 bash "$SCR/build_svdqNaive_W4A8.sh" > "$LOG/build_object.log"  2>&1 &
PID_O=$!
SUITE=spatial GPU=1 bash "$SCR/build_svdqNaive_W4A8.sh" > "$LOG/build_spatial.log" 2>&1 &
PID_S=$!
SUITE=goal    GPU=2 bash "$SCR/build_svdqNaive_W4A8.sh" > "$LOG/build_goal.log"    2>&1 &
PID_G=$!
wait $PID_O; RC_O=$?
wait $PID_S; RC_S=$?
wait $PID_G; RC_G=$?
say "Stage 1 build rc: object=$RC_O spatial=$RC_S goal=$RC_G"
[ $RC_O -ne 0 ] || [ $RC_S -ne 0 ] || [ $RC_G -ne 0 ] && { say "FAIL build"; exit 1; }

# ===== Stage 2: 3 shorts in parallel, shard=8 each =====
say "Stage 2 — 3 shorts in parallel (shard=8 each on GPUs 0-7, packed)"
SUITE=object  GPUS="0,1,2,3,4,5,6,7" PORT_BASE=9100 OFF=10 TRIALS=10 \
  bash "$SCR/eval_svdqNaive_W4A8.sh" > "$LOG/eval_object.log" 2>&1 &
PID_O=$!
SUITE=spatial GPUS="0,1,2,3,4,5,6,7" PORT_BASE=9200 OFF=10 TRIALS=10 \
  bash "$SCR/eval_svdqNaive_W4A8.sh" > "$LOG/eval_spatial.log" 2>&1 &
PID_S=$!
SUITE=goal    GPUS="0,1,2,3,4,5,6,7" PORT_BASE=9300 OFF=10 TRIALS=10 \
  bash "$SCR/eval_svdqNaive_W4A8.sh" > "$LOG/eval_goal.log" 2>&1 &
PID_G=$!
wait $PID_O; RC_O=$?
wait $PID_S; RC_S=$?
wait $PID_G; RC_G=$?
say "Stage 2 shorts rc: object=$RC_O spatial=$RC_S goal=$RC_G"

# ===== Summary =====
say ""
say "=== SVDQ-naive W4A8 (block_size=0) shorts summary off=10 ==="
for suite in object spatial goal; do
    f="$REPO/results/attn_iter_svdqNaive_W4A8/${suite}_off10_10tr/merged_summary.json"
    if [ -f "$f" ]; then
        rate=$(python -c "import json; print(round(100*json.load(open('$f'))['total_success_rate'],1))" 2>/dev/null || echo "?")
        say "  $suite: ${rate}%"
    else
        say "  $suite: MISSING"
    fi
done
