#!/bin/bash
# SVDQ-naive W4A4 (block_size=0, per-row only) — shorts-only.
set -uo pipefail
REPO="${QUANTVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCR="$REPO/scripts/attn_iter"
LOG="$REPO/logs/svdqNaive_W4A4"
mkdir -p "$LOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/orchestrator.log"; }

TAG="svdqNaive_W4A4"

# Stage 1: build 3 shorts in parallel, W4A4
say "Stage 1 — build 3 short packs ($TAG, block_size=0)"
SUITE=object  GPU=0 A_BITS=4 PACK_TAG="$TAG" \
  bash "$SCR/build_svdqNaive_generic.sh" > "$LOG/build_object.log"  2>&1 &
PID_O=$!
SUITE=spatial GPU=1 A_BITS=4 PACK_TAG="$TAG" \
  bash "$SCR/build_svdqNaive_generic.sh" > "$LOG/build_spatial.log" 2>&1 &
PID_S=$!
SUITE=goal    GPU=2 A_BITS=4 PACK_TAG="$TAG" \
  bash "$SCR/build_svdqNaive_generic.sh" > "$LOG/build_goal.log"    2>&1 &
PID_G=$!
wait $PID_O; RC_O=$?
wait $PID_S; RC_S=$?
wait $PID_G; RC_G=$?
say "Stage 1 build rc: object=$RC_O spatial=$RC_S goal=$RC_G"
[ $RC_O -ne 0 ] || [ $RC_S -ne 0 ] || [ $RC_G -ne 0 ] && { say "FAIL build"; exit 1; }

# Stage 2: eval shorts in parallel (shard=8 each, 8 GPUs packed)
say "Stage 2 — 3 shorts in parallel (shard=8 each on GPUs 0-7)"
SUITE=object  GPUS="0,1,2,3,4,5,6,7" PORT_BASE=9500 OFF=10 TRIALS=10 \
  LABEL="$TAG" EVAL_ABITS=4 \
  bash "$SCR/eval_svdqNaive_generic.sh" > "$LOG/eval_object.log" 2>&1 &
PID_O=$!
SUITE=spatial GPUS="0,1,2,3,4,5,6,7" PORT_BASE=9600 OFF=10 TRIALS=10 \
  LABEL="$TAG" EVAL_ABITS=4 \
  bash "$SCR/eval_svdqNaive_generic.sh" > "$LOG/eval_spatial.log" 2>&1 &
PID_S=$!
SUITE=goal    GPUS="0,1,2,3,4,5,6,7" PORT_BASE=9700 OFF=10 TRIALS=10 \
  LABEL="$TAG" EVAL_ABITS=4 \
  bash "$SCR/eval_svdqNaive_generic.sh" > "$LOG/eval_goal.log" 2>&1 &
PID_G=$!
wait $PID_O; RC_O=$?
wait $PID_S; RC_S=$?
wait $PID_G; RC_G=$?
say "Stage 2 shorts rc: object=$RC_O spatial=$RC_S goal=$RC_G"

# Summary
say ""
say "=== $TAG (block_size=0) shorts off=10 ==="
for s in object spatial goal; do
  f="$REPO/results/attn_iter_${TAG}/${s}_off10_10tr/merged_summary.json"
  if [ -f "$f" ]; then
    rate=$(python -c "import json; print(round(100*json.load(open('$f'))['total_success_rate'],1))" 2>/dev/null || echo "?")
    say "  $s: ${rate}%"
  else
    say "  $s: MISSING"
  fi
done
