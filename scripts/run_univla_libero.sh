#!/bin/bash
# UniVLA LIBERO runner for Awesome-quant-vla.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_paths.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_univla_libero.sh [options]

Actions:
  --action plan|run|result

Options:
  --profile NAME
  --suite object|spatial|goal|long
  --checkpoint PATH
  --action-decoder PATH
  --python PATH
  --gpus 0[,1]
  --trials N
  --init-offset N
  --output-root PATH
  --checkpoints-root PATH
  --libero-root PATH
  --center-crop True|False
  --save-video True|False
  --seed N
EOF
}

die() {
    echo "[univla] ERROR: $*" >&2
    exit 2
}

suite_to_task_suite() {
    case "$1" in
        long) echo "libero_10" ;;
        spatial) echo "libero_spatial" ;;
        goal) echo "libero_goal" ;;
        object) echo "libero_object" ;;
        libero_10|libero_spatial|libero_goal|libero_object|libero_90) echo "$1" ;;
        *) die "unknown suite '$1'" ;;
    esac
}

suite_to_short() {
    case "$1" in
        libero_10) echo "long" ;;
        libero_spatial) echo "spatial" ;;
        libero_goal) echo "goal" ;;
        libero_object) echo "object" ;;
        libero_90) echo "90" ;;
        *) echo "$1" ;;
    esac
}

print_result() {
    local summary="$1/merged_summary.json"
    [[ -f "$summary" ]] || die "summary not found: $summary"
    "$PYTHON_BIN" - "$summary" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"{100 * data['total_success_rate']:.1f} %")
PY
}

PROFILE="${PROFILE:-univla_fp16}"
ACTION="${ACTION:-run}"
SUITE="${SUITE:-spatial}"
GPU_LIST="${GPU_LIST:-0}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
GR00T_EVAL_INIT_OFFSET="${GR00T_EVAL_INIT_OFFSET:-10}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${HOME}/ckpts}"
AWESOME_QVLA_ROOT="${AWESOME_QVLA_ROOT:-${QUANTVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
QUANTVLA_ROOT="${QUANTVLA_ROOT:-${AWESOME_QVLA_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
UNIVLA_CHECKPOINT="${UNIVLA_CHECKPOINT:-}"
UNIVLA_ACTION_DECODER="${UNIVLA_ACTION_DECODER:-}"
UNIVLA_PYTHON="${UNIVLA_PYTHON:-${OPENVLA_PYTHON:-python}}"
LIBERO_ROOT_ARG="${LIBERO_ROOT:-}"
CENTER_CROP="${CENTER_CROP:-True}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
SEED="${SEED:-7}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --action|--mode) ACTION="${2:?missing --action value}"; shift 2 ;;
        --profile) PROFILE="${2:?missing --profile value}"; shift 2 ;;
        --suite) SUITE="${2:?missing --suite value}"; shift 2 ;;
        --checkpoint|--univla-checkpoint) UNIVLA_CHECKPOINT="${2:?missing checkpoint value}"; shift 2 ;;
        --action-decoder|--univla-action-decoder) UNIVLA_ACTION_DECODER="${2:?missing action decoder value}"; shift 2 ;;
        --python|--univla-python) UNIVLA_PYTHON="${2:?missing python value}"; shift 2 ;;
        --gpus|--gpu-list) GPU_LIST="${2:?missing --gpus value}"; shift 2 ;;
        --trials) NUM_TRIALS_PER_TASK="${2:?missing --trials value}"; shift 2 ;;
        --init-offset) GR00T_EVAL_INIT_OFFSET="${2:?missing --init-offset value}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:?missing --output-root value}"; shift 2 ;;
        --checkpoints-root) CHECKPOINTS_ROOT="${2:?missing --checkpoints-root value}"; shift 2 ;;
        --libero-root) LIBERO_ROOT_ARG="${2:?missing --libero-root value}"; shift 2 ;;
        --center-crop) CENTER_CROP="${2:?missing --center-crop value}"; shift 2 ;;
        --save-video) SAVE_VIDEO="${2:?missing --save-video value}"; shift 2 ;;
        --seed) SEED="${2:?missing --seed value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option '$1'" ;;
    esac
done

case "$ACTION" in
    plan|run|result) ;;
    *) die "unknown action '$ACTION'" ;;
esac

BACKEND_ROOT="${QUANTVLA_ROOT}/third_party/univla"
[[ -d "$BACKEND_ROOT" ]] || die "UniVLA backend root not found: $BACKEND_ROOT"

TASK_SUITE="$(suite_to_task_suite "$SUITE")"
SUITE_SHORT="$(suite_to_short "$TASK_SUITE")"
SUITE="$SUITE_SHORT"
OUTPUT_ROOT="${OUTPUT_ROOT:-${QUANTVLA_ROOT}/results/awesome_quant_vla/${PROFILE}_${SUITE}}"
LOG_DIR="${OUTPUT_ROOT}/logs/eval"
PYTHON_BIN="$UNIVLA_PYTHON"

if [[ -z "$UNIVLA_CHECKPOINT" ]]; then
    UNIVLA_CHECKPOINT="${CHECKPOINTS_ROOT}/univla-7b-224-sft-libero"
fi
if [[ -z "$UNIVLA_ACTION_DECODER" ]]; then
    UNIVLA_ACTION_DECODER="${UNIVLA_CHECKPOINT}/action_decoder.pt"
fi

FIRST_GPU="${GPU_LIST%%,*}"
export CUDA_VISIBLE_DEVICES="$FIRST_GPU"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$FIRST_GPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export UNIVLA_ATTN_IMPL="${UNIVLA_ATTN_IMPL:-${OPENVLA_ATTN_IMPL:-eager}}"
export PYTHONPATH="${BACKEND_ROOT}:${QUANTVLA_ROOT}/tools:${QUANTVLA_ROOT}${LIBERO_ROOT_ARG:+:${LIBERO_ROOT_ARG}}:${PYTHONPATH:-}"
if [[ -n "$LIBERO_ROOT_ARG" ]]; then
    export LIBERO_ROOT="$LIBERO_ROOT_ARG"
fi

print_plan() {
    cat <<EOF
================================================================
[univla] plan
  profile        : ${PROFILE}
  action         : ${ACTION}
  suite          : ${SUITE} (${TASK_SUITE})
  project root   : ${QUANTVLA_ROOT}
  backend root   : ${BACKEND_ROOT}
  python         : ${PYTHON_BIN}
  checkpoint     : ${UNIVLA_CHECKPOINT}
  action decoder : ${UNIVLA_ACTION_DECODER}
  output         : ${OUTPUT_ROOT}
  gpus           : ${GPU_LIST}
  trials/task    : ${NUM_TRIALS_PER_TASK}
  steps wait     : ${GR00T_EVAL_INIT_OFFSET}
  center crop    : ${CENTER_CROP}
  save video     : ${SAVE_VIDEO}
  attention impl : ${UNIVLA_ATTN_IMPL}
================================================================
EOF
}

run_parse() {
    "$PYTHON_BIN" "${QUANTVLA_ROOT}/tools/qvla/parse_openvla_eval.py" \
        --log-dir "$LOG_DIR" \
        --output-root "$OUTPUT_ROOT" \
        --task-suite-name "$TASK_SUITE" \
        --benchmark "${PROFILE}_${SUITE}" \
        --quantization "FP16" \
        --gpus "$GPU_LIST"
}

run_eval() {
    [[ -d "$UNIVLA_CHECKPOINT" ]] || die "checkpoint not found: $UNIVLA_CHECKPOINT"
    [[ -f "$UNIVLA_ACTION_DECODER" ]] || die "action decoder not found: $UNIVLA_ACTION_DECODER"
    mkdir -p "$LOG_DIR"
    pushd "$BACKEND_ROOT" >/dev/null
    "$PYTHON_BIN" "experiments/robot/libero/run_libero_eval.py" \
        --model_family openvla \
        --pretrained_checkpoint "$UNIVLA_CHECKPOINT" \
        --action_decoder_path "$UNIVLA_ACTION_DECODER" \
        --task_suite_name "$TASK_SUITE" \
        --num_trials_per_task "$NUM_TRIALS_PER_TASK" \
        --num_steps_wait "$GR00T_EVAL_INIT_OFFSET" \
        --local_log_dir "$LOG_DIR" \
        --center_crop "$CENTER_CROP" \
        --save_video "$SAVE_VIDEO" \
        --run_id_note "${PROFILE}_${SUITE}" \
        --use_wandb False \
        --seed "$SEED"
    popd >/dev/null
    run_parse
}

if [[ "$ACTION" != "result" ]]; then
    print_plan
fi

case "$ACTION" in
    plan) exit 0 ;;
    result) print_result "$OUTPUT_ROOT" ;;
    run) run_eval ;;
esac

