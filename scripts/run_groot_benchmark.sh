#!/bin/bash
#
# GR00T-N1.5 LIBERO benchmark — orthogonal quantization interface.
#
# Two orthogonal axes, chosen independently per side (LLM = Eagle language model,
# DiT = action-head transformer):
#
#   *_QUANT  ∈ {gptq, rtn, duquant, none}    which quantizer wraps that side
#   *_ROT    ∈ {none, svd, svd_hadamard}     DuQuant rotation (duquant side only;
#                                            for a gptq side the rotation is baked
#                                            into the pack at build time)
#
# Switches:
#   LLM_QUANT   (default duquant)   DIT_QUANT   (default gptq)
#   LLM_ROT     (default svd_hadamard)  DIT_ROT (default svd_hadamard)
#   DIT_ATTN    (default 0)          1 = also quantize DiT attn1 (q/k/v/out), 0 = MLP only
#   DIT_PERSTEP (default 1)          informational; per-step act scale is baked into the GPTQ pack
#   LLM_GPTQ_PACK / DIT_GPTQ_PACK   required when that side's *_QUANT=gptq
#
# Presets (set PRESET to skip the switches):
#   quantvla        default DuQuant + ATM + OHB on LLM + DiT-MLP
#   quantvla_w4a8   QuantVLA legacy W4A8 route:
#                  DuQuant(SVD) + ATM + OHB on LLM + DiT-MLP
#   pure_gptq  GPTQ pack (no rotation) on LLM + DiT-MLP — needs GPTQ_PACK
#   fp16       no quantization
#
# Reproduce README E2 (default switches): LLM duquant(svd_hadamard) + DiT gptq pack.
#
# One invocation runs ONE config on a chosen GPU set.

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

PYTHON_BIN="$(quantvla_python_bin)"

export PYTHONNOUSERSITE=1 HF_HUB_DISABLE_XET=1 NO_ALBUMENTATIONS_UPDATE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
unset VIRTUAL_ENV

SUITE="${SUITE:?must set SUITE: long|spatial|goal|object}"
WBITS="${WBITS:-4}"
ABITS="${ABITS:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
PORT_BASE="${PORT_BASE:-5800}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-40}"
NUM_STEPS_WAIT="${NUM_STEPS_WAIT:-10}"
DENOISING_STEPS="${DENOISING_STEPS:-8}"

case "$SUITE" in
    long)    TASK_SUITE="libero_10";     CKPT_NAME="libero-long-posttrain" ;;
    spatial) TASK_SUITE="libero_spatial"; CKPT_NAME="libero-spatial-posttrain" ;;
    goal)    TASK_SUITE="libero_goal";    CKPT_NAME="libero-goal-posttrain" ;;
    object)  TASK_SUITE="libero_object";  CKPT_NAME="libero-object-posttrain" ;;
    *) echo "unknown suite $SUITE" >&2; exit 1 ;;
esac
CHECKPOINT="${CHECKPOINT:-${CHECKPOINTS_ROOT:?set CHECKPOINTS_ROOT to your checkpoints dir}/gr00t-n1.5-${CKPT_NAME}}"
DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
[ "$SUITE" = "goal" ] && DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"

if [[ "${CHECKPOINT}" == /* || "${CHECKPOINT}" == ./* || "${CHECKPOINT}" == ../* ]]; then
    if [[ ! -d "${CHECKPOINT}" ]]; then
        echo "[bench] ERROR: GR00T checkpoint directory not found: ${CHECKPOINT}" >&2
        echo "[bench]        download it with:" >&2
        echo "[bench]        hf download youliangtan/gr00t-n1.5-${CKPT_NAME} --local-dir \"${CHECKPOINT}\"" >&2
        echo "[bench]        or pass --groot-checkpoint /path/to/checkpoint." >&2
        exit 2
    fi
    if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
        echo "[bench] ERROR: GR00T checkpoint does not look complete: ${CHECKPOINT}" >&2
        echo "[bench]        missing required file: ${CHECKPOINT}/config.json" >&2
        echo "[bench]        re-download the checkpoint or pass --groot-checkpoint /path/to/checkpoint." >&2
        exit 2
    fi
fi

# ---- Regex building blocks ----
LLM_RE='backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
DIT_MLP_RE='action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)'
DIT_ATTN_RE='action_head\.model\.transformer_blocks\.\d+\.attn1\.(to_q|to_k|to_v|to_out\.0)'
EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)'

# ---- Presets expand into the orthogonal switches ----
PRESET="${PRESET:-}"
ATM_ON=0
case "$PRESET" in
    quantvla)  LLM_QUANT=duquant; DIT_QUANT=duquant; LLM_ROT=svd_hadamard; DIT_ROT=svd_hadamard; ATM_ON=1 ;;
    quantvla_w4a8)
        LLM_QUANT=duquant
        DIT_QUANT=duquant
        LLM_ROT=svd
        DIT_ROT=svd
        DIT_ATTN="${DIT_ATTN:-0}"
        ATM_ON=1
        ;;
    pure_gptq) LLM_QUANT=gptq;    DIT_QUANT=gptq;    LLM_ROT=none;         DIT_ROT=none ;;
    fp16)      LLM_QUANT=none;    DIT_QUANT=none ;;
    "")        : ;;  # use switches directly
    *) echo "unknown PRESET $PRESET" >&2; exit 1 ;;
esac

LLM_QUANT="${LLM_QUANT:-duquant}"        # gptq | rtn | duquant | none
DIT_QUANT="${DIT_QUANT:-gptq}"
LLM_ROT="${LLM_ROT:-svd_hadamard}"       # none | svd | svd_hadamard
DIT_ROT="${DIT_ROT:-svd_hadamard}"
DIT_ATTN="${DIT_ATTN:-0}"
DIT_PERSTEP="${DIT_PERSTEP:-1}"

DIT_RE="$DIT_MLP_RE"
[ "$DIT_ATTN" = "1" ] && DIT_RE="(${DIT_ATTN_RE}|${DIT_MLP_RE})"

LABEL="${SUITE}_llm-${LLM_QUANT}-${LLM_ROT}_dit-${DIT_QUANT}-${DIT_ROT}_w${WBITS}a${ABITS}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${QUANTVLA_ROOT}/results/groot_bench/${LABEL}}"
mkdir -p "${OUTPUT_ROOT}/logs"
LABEL=$(basename "${OUTPUT_ROOT}")

echo "================================================================"
echo "[bench] ${LABEL}  GPU_LIST=${GPU_LIST}  trials=${NUM_TRIALS_PER_TASK}"
echo "  LLM=${LLM_QUANT}/${LLM_ROT}   DiT=${DIT_QUANT}/${DIT_ROT} (attn=${DIT_ATTN} perstep=${DIT_PERSTEP})"
echo "  ckpt=${CHECKPOINT}"
echo "  output=${OUTPUT_ROOT}"
echo "================================================================"

# ---- Accumulate per-quantizer include regexes from the side assignments ----
DUQUANT_PARTS=(); RTN_PARTS=(); GPTQ_PARTS=()
DUQUANT_ROT=""; GPTQ_PACK=""
assign_side() {  # $1=quant $2=side_regex $3=rot $4=gptq_pack_env
    case "$1" in
        duquant) DUQUANT_PARTS+=("$2"); DUQUANT_ROT="${DUQUANT_ROT:-$3}" ;;
        rtn)     RTN_PARTS+=("$2") ;;
        gptq)    GPTQ_PARTS+=("$2"); GPTQ_PACK="${GPTQ_PACK:-${!4:-}}" ;;
        none)    : ;;
        *) echo "unknown quant '$1'" >&2; exit 1 ;;
    esac
}
assign_side "$LLM_QUANT" "$LLM_RE" "$LLM_ROT" LLM_GPTQ_PACK
assign_side "$DIT_QUANT" "$DIT_RE" "$DIT_ROT" DIT_GPTQ_PACK

join_or() { local IFS='|'; echo "$*"; }

# Clear any inherited quant env so this run is hermetic.
unset GR00T_DUQUANT_INCLUDE GR00T_DUQUANT_EXCLUDE GR00T_DUQUANT_ROT_MODE \
      GR00T_RTN GR00T_RTN_INCLUDE GR00T_RTN_EXCLUDE \
      GR00T_GPTQ GR00T_GPTQ_PATH GR00T_GPTQ_INCLUDE GR00T_GPTQ_EXCLUDE \
      GR00T_HYBRID_QUANT GR00T_ATM_ENABLE GR00T_OHB_ENABLE 2>/dev/null || true

have_duquant=0; have_rtn=0; have_gptq=0

if [ "${#DUQUANT_PARTS[@]}" -gt 0 ]; then
    have_duquant=1
    export GR00T_DUQUANT_INCLUDE=".*($(join_or "${DUQUANT_PARTS[@]}")).*"
    export GR00T_DUQUANT_EXCLUDE="$EXCLUDE"
    export GR00T_DUQUANT_BLOCK=64 GR00T_DUQUANT_BLOCK_OUT=64
    export GR00T_DUQUANT_PERMUTE=0 GR00T_DUQUANT_ROW_ROT=restore
    export GR00T_DUQUANT_ACT_PCT=99.9 GR00T_DUQUANT_CALIB_STEPS=32 GR00T_DUQUANT_LS=0.15
    export GR00T_DUQUANT_PERM_SCORE=weight
    export GR00T_DUQUANT_ROT_MODE="${DUQUANT_ROT:-svd}"
    [ "${DUQUANT_ROT}" = "none" ] && export GR00T_DUQUANT_ROW_ROT=0
fi

if [ "${#RTN_PARTS[@]}" -gt 0 ]; then
    have_rtn=1
    export GR00T_RTN=1
    export GR00T_RTN_INCLUDE=".*($(join_or "${RTN_PARTS[@]}")).*"
    export GR00T_RTN_EXCLUDE="$EXCLUDE"
    export GR00T_RTN_WBITS="$WBITS" GR00T_RTN_ABITS="$ABITS" GR00T_RTN_GROUP_SIZE=64
fi

if [ "${#GPTQ_PARTS[@]}" -gt 0 ]; then
    have_gptq=1
    # Caller override (used by baseline eval scripts): GR00T_GPTQ_PATH_OVERRIDE etc.
    GPTQ_PACK="${GR00T_GPTQ_PATH_OVERRIDE:-$GPTQ_PACK}"
    if [ -z "$GPTQ_PACK" ] || [ ! -f "$GPTQ_PACK" ]; then
        echo "[bench] ERROR: GPTQ mode needs a quantized pack file, but it was not found." >&2
        echo "[bench]        resolved pack path: ${GPTQ_PACK:-<empty>}" >&2
        echo "[bench]        fix: build/download the pack, or pass --groot-gptq-pack /path/to/quantized.pt" >&2
        exit 1
    fi
    export GR00T_GPTQ=1
    export GR00T_GPTQ_PATH="$GPTQ_PACK"
    export GR00T_GPTQ_INCLUDE="${GR00T_GPTQ_INCLUDE_OVERRIDE:-.*($(join_or "${GPTQ_PARTS[@]}")).*}"
    export GR00T_GPTQ_EXCLUDE="${GR00T_GPTQ_EXCLUDE_OVERRIDE:-$EXCLUDE}"
    export GR00T_GPTQ_WBITS_DEFAULT="$WBITS" GR00T_GPTQ_ABITS="$ABITS"
    export GR00T_GPTQ_MISSING=fallback
fi

# ATM/OHB (quantvla preset only).
if [ "$ATM_ON" = "1" ]; then
    export GR00T_ATM_ENABLE=1 GR00T_ATM_SCOPE=dit
    export GR00T_ATM_ALPHA_PATH="${QUANTVLA_ROOT}/atm_alpha_beta_${SUITE}.json"
    export GR00T_OHB_ENABLE=1 GR00T_OHB_FALLBACK=1.0 GR00T_OHB_SCOPE=dit
fi

# METHOD signals the multi-GPU harness which DuQuant-env handling to use.
if [ "$have_gptq" = "1" ]; then
    METHOD=gptq
    [ "$have_duquant" = "1" ] && export GR00T_HYBRID_QUANT=1
elif [ "$have_duquant" = "1" ]; then
    METHOD=duquant
elif [ "$have_rtn" = "1" ]; then
    METHOD=rtn
else
    METHOD=fp16
fi

export METHOD TASK_SUITE GPU_LIST PORT_BASE
export NUM_TRIALS_PER_TASK NUM_STEPS_WAIT DENOISING_STEPS WBITS ABITS
export OUTPUT_ROOT
export PACKDIR="${OUTPUT_ROOT}/packdir"
export MODEL_PATH="${CHECKPOINT}"
export BENCHMARK_LABEL="${LABEL}"

if "${PYTHON_BIN}" -u "${QUANTVLA_ROOT}/scripts/run_libero_duquant_benchmark_multi_gpu.py" \
    2>&1 | tee "${OUTPUT_ROOT}/logs/run.log"; then
    echo "[bench ${LABEL}] exit=0"
else
    status=$?
    echo "[bench ${LABEL}] exit=${status}" >&2
    echo "[bench ${LABEL}] last run.log lines:" >&2
    tail -n 120 "${OUTPUT_ROOT}/logs/run.log" >&2 || true
    exit "${status}"
fi
