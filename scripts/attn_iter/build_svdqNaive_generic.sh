#!/bin/bash
# Build "naive SVDQuant W4A8" packs for one suite — both DiT and LLM sides.
#
# SVDQuant naive (paper-style):
#   1. SmoothQuant scale (alpha=0.5)
#   2. SVD-r16 low-rank head of W_s
#   3. RTN on residual (no GPTQ error compensation)
#   4. Single-bucket act_scale_table  (no per-step)
#
# Build pipeline per suite:
#   Step A (DiT side):   --target-side dit, output → ${SUITE}_${PACK_TAG:-svdqNaive_W4A8}_DiT/
#   Step B (LLM side):   --target-side llm, base-pack from Step A,
#                        output → ${SUITE}_${PACK_TAG:-svdqNaive_W4A8}/ (final, both sides)
#
# Args:
#   SUITE  = object | spatial | goal | long  (required)
#   GPU    = GPU index (default 0)
set -uo pipefail
REPO="${QUANTVLA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck disable=SC1091
source "${REPO}/scripts/common_paths.sh"
QUANTVLA_CONDA_ENV="${QUANTVLA_CONDA_ENV:-$(quantvla_default_conda_env)}"
export QUANTVLA_CONDA_ENV

quantvla_activate_env "${QUANTVLA_CONDA_ENV}"
quantvla_export_pythonpath
quantvla_setup_cache_dirs
quantvla_setup_libero_config
PY="$(command -v python)"
CKPT_ROOT="${CHECKPOINTS_ROOT:?set CHECKPOINTS_ROOT to your checkpoints dir}"
cd "$REPO"

SUITE="${SUITE:?must set SUITE}"
GPU="${GPU:-0}"

case "$SUITE" in
    object)  TASK="libero_object";  DCFG="examples.Libero.custom_data_config:LiberoDataConfig";        CKPT="$CKPT_ROOT/gr00t-n1.5-libero-object-posttrain"  ;;
    spatial) TASK="libero_spatial"; DCFG="examples.Libero.custom_data_config:LiberoDataConfig";        CKPT="$CKPT_ROOT/gr00t-n1.5-libero-spatial-posttrain" ;;
    goal)    TASK="libero_goal";    DCFG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"; CKPT="$CKPT_ROOT/gr00t-n1.5-libero-goal-posttrain"    ;;
    long)    TASK="libero_10";      DCFG="examples.Libero.custom_data_config:LiberoDataConfig";        CKPT="$CKPT_ROOT/gr00t-n1.5-libero-long-posttrain"   ;;
    *) echo "unknown SUITE=$SUITE"; exit 1 ;;
esac

BASE="$REPO/results/multisuite_packs/${SUITE}_DiT_a2lite_RTN_full_cal10_damp05/quantized.pt"
[ -f "$BASE" ] || { echo "[err] missing base pack $BASE"; exit 1; }

DIT_OUT="$REPO/results/multisuite_packs/${SUITE}_${PACK_TAG:-svdqNaive_W4A8}_DiT/quantized.pt"
LLM_OUT="$REPO/results/multisuite_packs/${SUITE}_${PACK_TAG:-svdqNaive_W4A8}/quantized.pt"
mkdir -p "$(dirname "$DIT_OUT")" "$(dirname "$LLM_OUT")"

LOG_DIT="$(dirname "$DIT_OUT")/build.log"
LOG_LLM="$(dirname "$LLM_OUT")/build.log"

# ---------- Step A: DiT-side SVDQ-naive (W4A8, no GPTQ, no perstep) ----------
if [ -f "$DIT_OUT" ]; then
    echo "[skip] $DIT_OUT already exists"
else
    echo "[build][$SUITE] Step A: DiT-side SVDQ-naive on GPU $GPU"
    env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PY" tools/build_dit_svdquant_weights.py \
        --checkpoint "$CKPT" \
        --base-pack  "$BASE" \
        --output-path "$DIT_OUT" \
        --task-suite-name "$TASK" \
        --data-config "$DCFG" \
        --target-side dit \
        --quant-type-DiT original \
        --use-rtn \
        --single-step-scale \
        --w-bits 4 --a-bits ${A_BITS:-8} \
        --svd-rank ${SVD_RANK:-16} \
        --num-samples 10 --token-cap 1024 --num-steps 8 \
        --sq-alpha 0.5 \
        --gptq-block-size 0 > "$LOG_DIT" 2>&1
    rc=$?
    echo "[build][$SUITE] Step A rc=$rc"
    [ "$rc" != 0 ] && exit $rc
fi

# ---------- Step B: LLM-side SVDQ-naive (W4A8), on top of Step A ----------
if [ -f "$LLM_OUT" ]; then
    echo "[skip] $LLM_OUT already exists"
else
    echo "[build][$SUITE] Step B: LLM-side SVDQ-naive on GPU $GPU"
    env CUDA_VISIBLE_DEVICES="$GPU" \
      "$PY" tools/build_dit_svdquant_weights.py \
        --checkpoint "$CKPT" \
        --base-pack  "$DIT_OUT" \
        --output-path "$LLM_OUT" \
        --task-suite-name "$TASK" \
        --data-config "$DCFG" \
        --target-side llm \
        --quant-type-DiT original \
        --use-rtn \
        --w-bits 4 --a-bits ${A_BITS:-8} \
        --svd-rank ${SVD_RANK:-16} \
        --num-samples 10 --token-cap 1024 \
        --sq-alpha 0.5 \
        --gptq-block-size 0 > "$LOG_LLM" 2>&1
    rc=$?
    echo "[build][$SUITE] Step B rc=$rc"
    [ "$rc" != 0 ] && exit $rc
fi

echo "[build][$SUITE] DONE → $LLM_OUT"
