#!/bin/bash
# Top-level orchestrator: build naive SVDQuant W4A8 packs (4 suites) and
# eval per user spec —
#   Phase 1: 3 shorts (object, spatial, goal) in parallel, each shard=8,
#            packed on GPUs 0-7 (distinct PORT_BASE per suite).
#   Phase 2: long alone, shard=5 on GPUs 0-4.
#
# Build stage runs 4 suites in parallel on GPUs 0-3 (one suite per GPU).
# Each suite does Step A (DiT-side) then Step B (LLM-side) sequentially.
set -uo pipefail
REPO="${QUANTVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCR="$REPO/scripts/attn_iter"
LOG="$REPO/logs/svdqNaive_W4A8"
mkdir -p "$LOG"

say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG/orchestrator.log"; }

# ===== Stage 1: build all 4 suites' packs in parallel =====
say "Stage 1 — build 4 packs (object→GPU0, spatial→GPU1, goal→GPU2, long→GPU3)"
SUITE=object  GPU=0 bash "$SCR/build_svdqNaive_W4A8.sh" > "$LOG/build_object.log"  2>&1 &
PID_O=$!
SUITE=spatial GPU=1 bash "$SCR/build_svdqNaive_W4A8.sh" > "$LOG/build_spatial.log" 2>&1 &
PID_S=$!
SUITE=goal    GPU=2 bash "$SCR/build_svdqNaive_W4A8.sh" > "$LOG/build_goal.log"    2>&1 &
PID_G=$!
SUITE=long    GPU=3 bash "$SCR/build_svdqNaive_W4A8.sh" > "$LOG/build_long.log"    2>&1 &
PID_L=$!

wait $PID_O; RC_O=$?
wait $PID_S; RC_S=$?
wait $PID_G; RC_G=$?
wait $PID_L; RC_L=$?
say "Stage 1 build rc: object=$RC_O spatial=$RC_S goal=$RC_G long=$RC_L"
if [ $RC_O -ne 0 ] || [ $RC_S -ne 0 ] || [ $RC_G -ne 0 ] || [ $RC_L -ne 0 ]; then
    say "[FAIL] one or more builds failed — see $LOG/build_*.log"
    exit 1
fi

# ===== Stage 2: 3 shorts in parallel, shard=8 each, packed on GPUs 0-7 =====
# Three suites × 8 shards on the same 8 GPUs → 3 inference servers per GPU,
# differentiated by PORT_BASE. Memory: pack-multiple-evals.
say "Stage 2 — 3 shorts in parallel (shard=8 each on GPUs 0-7)"
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

# ===== Stage 3: long alone, shard=5 on GPUs 0-4 =====
say "Stage 3 — long (shard=5 on GPUs 0-4)"
SUITE=long GPUS="0,1,2,3,4" PORT_BASE=9400 OFF=10 TRIALS=10 \
  bash "$SCR/eval_svdqNaive_W4A8.sh" > "$LOG/eval_long.log" 2>&1
RC_L=$?
say "Stage 3 long rc=$RC_L"

# ===== Final summary =====
say ""
say "=== SVDQ-naive W4A8 4-suite (off=10) summary ==="
for suite in object spatial goal long; do
    f="$REPO/results/attn_iter_svdqNaive_W4A8/${suite}_off10_10tr/merged_summary.json"
    if [ -f "$f" ]; then
        rate=$(python -c "import json; print(round(100*json.load(open('$f'))['total_success_rate'],1))" 2>/dev/null || echo "?")
        say "  $suite: ${rate}%"
    else
        say "  $suite: MISSING"
    fi
done
