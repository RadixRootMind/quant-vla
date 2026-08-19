#!/bin/bash
#
# Run pi0.5/OpenPI on LIBERO through this repo's multi-GPU benchmark harness.
#
# Required local pieces:
#   - OPENPI_ROOT points to a Physical-Intelligence/openpi checkout.
#   - OPENPI_PY points to the Python used by that checkout.
#   - The LIBERO eval Python (defaults to QUANTVLA_CONDA_ENV) has openpi_client installed.
#
# Methods:
#   fp16       : OpenPI checkpoint as-is.
#   pure_gptq  : GPTQ runtime with a pure GPTQ pack.
#   gptq  : GPTQ runtime with a (rotation-baked) GPTQ pack.
#
# For gptq/pure_gptq, OPENPI_CHECKPOINT must be a converted PyTorch OpenPI
# checkpoint directory containing model.safetensors.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_paths.sh"

QUANTVLA_CONDA_ENV="${QUANTVLA_CONDA_ENV:-$(quantvla_default_conda_env)}"
export QUANTVLA_CONDA_ENV

quantvla_activate_env "${QUANTVLA_CONDA_ENV}"
quantvla_export_pythonpath
quantvla_setup_cache_dirs
quantvla_setup_libero_config

RUNNER_PY="$(quantvla_python_bin)"

export PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
unset VIRTUAL_ENV

METHOD="${METHOD:-fp16}"
SUITE="${SUITE:-long}"
WBITS="${WBITS:-4}"
ABITS="${ABITS:-8}"
GPU_LIST="${GPU_LIST:-0}"
PORT_BASE="${PORT_BASE:-8000}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-5}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
OPENPI_ROOT="${OPENPI_ROOT:-${HOME}/openpi}"
OPENPI_PY="${OPENPI_PY:-${OPENPI_ROOT}/.venv/bin/python}"
OPENPI_CONFIG="${OPENPI_CONFIG:-pi05_libero}"
OPENPI_CHECKPOINT="${OPENPI_CHECKPOINT:-gs://openpi-assets/checkpoints/pi05_libero}"
OPENPI_DEVICE="${OPENPI_DEVICE:-cuda}"

case "$SUITE" in
    long)    TASK_SUITE="libero_10" ;;
    spatial) TASK_SUITE="libero_spatial" ;;
    goal)    TASK_SUITE="libero_goal" ;;
    object)  TASK_SUITE="libero_object" ;;
    libero_10|libero_spatial|libero_goal|libero_object|libero_90)
        TASK_SUITE="$SUITE" ;;
    *) echo "unknown suite ${SUITE}" >&2; exit 1 ;;
esac

if [ ! -x "${OPENPI_PY}" ]; then
    echo "[pi05] ERROR: OPENPI_PY is not executable: ${OPENPI_PY}" >&2
    echo "[pi05] Install openpi first, or set OPENPI_ROOT/OPENPI_PY." >&2
    exit 2
fi

if [ "$METHOD" != "fp16" ] && [[ "${OPENPI_CHECKPOINT}" == gs://* ]]; then
    echo "[pi05] ERROR: ${METHOD} needs a converted PyTorch checkpoint directory, not ${OPENPI_CHECKPOINT}" >&2
    echo "[pi05] Convert gs://openpi-assets/checkpoints/pi05_libero to a directory containing model.safetensors." >&2
    exit 2
fi

OPENPI_DUQUANT_INCLUDE_DEFAULT='.*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*'
OPENPI_DUQUANT_EXCLUDE_DEFAULT='(?:^|\.)(vision_tower|vision_model|embeddings|embed_tokens|norm|layernorm|lm_head)(?:\.|$)'

LABEL="${METHOD}_pi05_${TASK_SUITE}_w${WBITS}a${ABITS}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${QUANTVLA_ROOT}/results/pi05_libero/${LABEL}}"
mkdir -p "${OUTPUT_ROOT}/logs"

# OpenPI PyTorch names are different from GR00T. This default covers Gemma
# attention/MLP layers plus the flow/action projections while excluding vision
# tower, embeddings, norms, and heads. Override after a dry-run if needed.
OPENPI_INCLUDE_DEFAULT='.*(paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|(?:state_proj|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)).*'
OPENPI_EXCLUDE_DEFAULT='(?:^|\.)(vision_tower|vision_model|embeddings|embed_tokens|norm|layernorm|lm_head)(?:\.|$)'

case "$METHOD" in
    fp16)
        unset GR00T_GPTQ GR00T_GPTQ_PATH GR00T_GPTQ_INCLUDE GR00T_GPTQ_EXCLUDE
        unset GR00T_DUQUANT_INCLUDE GR00T_DUQUANT_EXCLUDE GR00T_ATM_ENABLE GR00T_OHB_ENABLE
        ;;
    pure_gptq|gptq)
        QUANT_PATH="${OPENPI_GPTQ_PATH:-${GR00T_GPTQ_PATH_OVERRIDE:-}}"
        if [ -z "${QUANT_PATH}" ]; then
            echo "[pi05] ERROR: set OPENPI_GPTQ_PATH to the pi05 ${METHOD} pack (.pt)" >&2
            exit 2
        fi
        if [ ! -f "${QUANT_PATH}" ]; then
            echo "[pi05] ERROR: missing pack ${QUANT_PATH}" >&2
            exit 2
        fi
        export GR00T_GPTQ=1
        export GR00T_GPTQ_PATH="${QUANT_PATH}"
        export GR00T_GPTQ_INCLUDE="${OPENPI_GPTQ_INCLUDE:-${OPENPI_INCLUDE_DEFAULT}}"
        export GR00T_GPTQ_EXCLUDE="${OPENPI_GPTQ_EXCLUDE:-${OPENPI_EXCLUDE_DEFAULT}}"
        export GR00T_GPTQ_WBITS_DEFAULT="${WBITS}"
        export GR00T_GPTQ_ABITS="${ABITS}"
        export GR00T_GPTQ_MISSING="${GR00T_GPTQ_MISSING:-error}"
        unset GR00T_ATM_ENABLE GR00T_OHB_ENABLE
        ;;
    hybrid)
        # Asymmetric: PaliGemma LLM via DuQuant runtime (A2-lite default),
        # Gemma expert via GPTQ pack (SVDQuant custom/original).
        # Caller MUST set OPENPI_GPTQ_PATH to the expert-only pack
        # and OPENPI_GPTQ_INCLUDE / OPENPI_DUQUANT_INCLUDE to non-overlapping
        # regexes (one for expert, one for PaliGemma).
        QUANT_PATH="${OPENPI_GPTQ_PATH:-${GR00T_GPTQ_PATH_OVERRIDE:-}}"
        if [ -z "${QUANT_PATH}" ] || [ ! -f "${QUANT_PATH}" ]; then
            echo "[pi05] ERROR (hybrid): OPENPI_GPTQ_PATH must point at expert pack .pt" >&2
            exit 2
        fi
        export GR00T_GPTQ=1
        export GR00T_GPTQ_PATH="${QUANT_PATH}"
        export GR00T_GPTQ_INCLUDE="${OPENPI_GPTQ_INCLUDE:?hybrid requires OPENPI_GPTQ_INCLUDE (expert regex)}"
        export GR00T_GPTQ_EXCLUDE="${OPENPI_GPTQ_EXCLUDE:-${OPENPI_EXCLUDE_DEFAULT}}"
        export GR00T_GPTQ_WBITS_DEFAULT="${WBITS}"
        export GR00T_GPTQ_ABITS="${ABITS}"
        export GR00T_GPTQ_MISSING="${GR00T_GPTQ_MISSING:-fallback}"
        # DuQuant on PaliGemma LLM only
        export GR00T_DUQUANT_INCLUDE="${OPENPI_DUQUANT_INCLUDE:?hybrid requires OPENPI_DUQUANT_INCLUDE (paligemma regex)}"
        export GR00T_DUQUANT_EXCLUDE="${OPENPI_DUQUANT_EXCLUDE:-${OPENPI_DUQUANT_EXCLUDE_DEFAULT}}"
        export GR00T_DUQUANT_BLOCK="${GR00T_DUQUANT_BLOCK:-64}"
        export GR00T_DUQUANT_BLOCK_OUT="${GR00T_DUQUANT_BLOCK_OUT:-64}"
        export GR00T_DUQUANT_PERMUTE="${GR00T_DUQUANT_PERMUTE:-1}"
        export GR00T_DUQUANT_ROW_ROT="${GR00T_DUQUANT_ROW_ROT:-restore}"
        export GR00T_DUQUANT_ACT_PCT="${GR00T_DUQUANT_ACT_PCT:-99.9}"
        export GR00T_DUQUANT_CALIB_STEPS="${GR00T_DUQUANT_CALIB_STEPS:-32}"
        export GR00T_DUQUANT_LS="${GR00T_DUQUANT_LS:-0.15}"
        export GR00T_DUQUANT_PERM_SCORE="${GR00T_DUQUANT_PERM_SCORE:-weight}"
        export GR00T_DUQUANT_ROT_MODE="${GR00T_DUQUANT_ROT_MODE:-svd_hadamard}"
        export GR00T_DUQUANT_ACT_SCALE_MODE="${GR00T_DUQUANT_ACT_SCALE_MODE:-percentile}"
        unset GR00T_ATM_ENABLE GR00T_OHB_ENABLE
        ;;
    rtn)
        unset GR00T_GPTQ GR00T_GPTQ_PATH GR00T_GPTQ_INCLUDE GR00T_GPTQ_EXCLUDE
        unset GR00T_DUQUANT_INCLUDE GR00T_DUQUANT_EXCLUDE GR00T_ATM_ENABLE GR00T_OHB_ENABLE
        export GR00T_RTN=1
        export GR00T_RTN_INCLUDE="${OPENPI_RTN_INCLUDE:-${OPENPI_DUQUANT_INCLUDE_DEFAULT}}"
        export GR00T_RTN_EXCLUDE="${OPENPI_RTN_EXCLUDE:-${OPENPI_DUQUANT_EXCLUDE_DEFAULT}}"
        export GR00T_RTN_GROUP_SIZE="${GR00T_RTN_GROUP_SIZE:-64}"
        export GR00T_RTN_WBITS="${GR00T_RTN_WBITS:-${WBITS}}"
        export GR00T_RTN_ABITS="${GR00T_RTN_ABITS:-${ABITS}}"
        ;;
    duquant)
        unset GR00T_GPTQ GR00T_GPTQ_PATH GR00T_GPTQ_INCLUDE GR00T_GPTQ_EXCLUDE
        # NOTE: GR00T_ATM_* / GR00T_OHB_* are intentionally NOT unset here so
        # callers can opt into pi0.5 ATM/OHB by exporting
        #   GR00T_ATM_SCOPE=pi05 GR00T_ATM_ALPHA_PATH=… GR00T_ATM_ENABLE=1 GR00T_OHB_ENABLE=1
        # before invoking this script. The inference service routes scope=pi05
        # through gr00t.atm.pi05_atm; scope=dit is a no-op on pi0.5.
        if [ "${GR00T_ATM_SCOPE:-}" != "pi05" ]; then
            unset GR00T_ATM_ENABLE GR00T_OHB_ENABLE
        fi
        export GR00T_DUQUANT_INCLUDE="${OPENPI_DUQUANT_INCLUDE:-${OPENPI_DUQUANT_INCLUDE_DEFAULT}}"
        export GR00T_DUQUANT_EXCLUDE="${OPENPI_DUQUANT_EXCLUDE:-${OPENPI_DUQUANT_EXCLUDE_DEFAULT}}"
        export GR00T_DUQUANT_BLOCK="${GR00T_DUQUANT_BLOCK:-64}"
        export GR00T_DUQUANT_BLOCK_OUT="${GR00T_DUQUANT_BLOCK_OUT:-64}"
        export GR00T_DUQUANT_PERMUTE="${GR00T_DUQUANT_PERMUTE:-1}"
        export GR00T_DUQUANT_ROW_ROT="${GR00T_DUQUANT_ROW_ROT:-restore}"
        export GR00T_DUQUANT_ACT_PCT="${GR00T_DUQUANT_ACT_PCT:-99.9}"
        export GR00T_DUQUANT_CALIB_STEPS="${GR00T_DUQUANT_CALIB_STEPS:-32}"
        export GR00T_DUQUANT_LS="${GR00T_DUQUANT_LS:-0.15}"
        export GR00T_DUQUANT_PERM_SCORE="${GR00T_DUQUANT_PERM_SCORE:-act_weight}"
        export GR00T_DUQUANT_ROT_MODE="${GR00T_DUQUANT_ROT_MODE:-svd_hadamard}"
        export GR00T_DUQUANT_ACT_SCALE_MODE="${GR00T_DUQUANT_ACT_SCALE_MODE:-percentile}"
        if [ -n "${OPENPI_DUQUANT_ACT_STATS_PATH:-${GR00T_DUQUANT_ACT_STATS_PATH:-}}" ]; then
            export GR00T_DUQUANT_ACT_STATS_PATH="${OPENPI_DUQUANT_ACT_STATS_PATH:-${GR00T_DUQUANT_ACT_STATS_PATH}}"
        fi
        # Per-cell A2 default needs act-stats; warn if perm_score wants act stats but file is missing.
        if [ "${GR00T_DUQUANT_PERM_SCORE}" != "weight" ] && [ -z "${GR00T_DUQUANT_ACT_STATS_PATH:-}" ]; then
            echo "[pi05] WARN: GR00T_DUQUANT_PERM_SCORE=${GR00T_DUQUANT_PERM_SCORE} expects GR00T_DUQUANT_ACT_STATS_PATH; permutation will fall back to weight-energy." >&2
        fi
        ;;
    *) echo "unknown METHOD ${METHOD}" >&2; exit 1 ;;
esac

export METHOD
export TASK_SUITE
export GPU_LIST
export NUM_TRIALS_PER_TASK NUM_STEPS_WAIT DENOISING_STEPS
export WBITS ABITS
export OUTPUT_ROOT
export PACKDIR="${OUTPUT_ROOT}/packdir"
export PORT_BASE
export MODEL_PATH="${OPENPI_CHECKPOINT}"
export DATA_CONFIG="${OPENPI_CONFIG}"
export BENCHMARK_LABEL="${LABEL}"
export GROOT_PY="${RUNNER_PY}"
export LIBERO_PY="${RUNNER_PY}"
export INFERENCE_PY="${OPENPI_PY}"
export INFERENCE_SCRIPT="${QUANTVLA_ROOT}/scripts/openpi_inference_service.py"
export INFERENCE_CWD="${OPENPI_ROOT}"
export INFERENCE_EXTRA_ARGS_JSON="[\"--device\",\"${OPENPI_DEVICE}\"]"
export EVAL_EXTRA_ARGS_JSON="[\"--policy-backend\",\"openpi_ws\",\"--max-steps-profile\",\"openpi\",\"--replan-steps\",\"${REPLAN_STEPS}\"]"

echo "================================================================"
echo "[pi05] ${LABEL}"
echo "  task_suite=${TASK_SUITE}"
echo "  method=${METHOD}"
echo "  checkpoint=${OPENPI_CHECKPOINT}"
echo "  openpi=${OPENPI_ROOT}"
echo "  output=${OUTPUT_ROOT}"
echo "  gpus=${GPU_LIST}"
echo "  trials=${NUM_TRIALS_PER_TASK}"
echo "================================================================"

if "${RUNNER_PY}" -u "${QUANTVLA_ROOT}/scripts/run_libero_duquant_benchmark_multi_gpu.py" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/run.log"; then
    echo "[pi05 ${LABEL}] exit=0"
else
    status=$?
    echo "[pi05 ${LABEL}] exit=${status}" >&2
    echo "[pi05 ${LABEL}] last run.log lines:" >&2
    tail -n 120 "${OUTPUT_ROOT}/logs/run.log" >&2 || true
    exit "${status}"
fi
