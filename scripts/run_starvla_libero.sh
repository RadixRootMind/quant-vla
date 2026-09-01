#!/bin/bash
# StarVLA LIBERO runner for quant-vla.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_paths.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_starvla_libero.sh [options]

Actions:
  --action plan|run|result

Options:
  --profile NAME
  --suite object|spatial|goal|long
  --checkpoint PATH
  --python PATH
  --gpus 0[,1]
  --trials N
  --init-offset N
  --port-base PORT
  --output-root PATH
  --checkpoints-root PATH
  --libero-root PATH
  --unnorm-key KEY
  --use-bf16 0|1
  --save-video True|False
  --max-tasks N
  --seed N
EOF
}

die() {
    echo "[starvla] ERROR: $*" >&2
    exit 2
}

bool_true() {
    case "${1:-}" in
        1|true|True|TRUE|yes|Yes|YES|on|On|ON) return 0 ;;
        *) return 1 ;;
    esac
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

wait_for_port() {
    local host="$1"
    local port="$2"
    "$PYTHON_BIN" - "$host" "$port" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + 600
last_error = None
while time.time() < deadline:
    sock = socket.socket()
    sock.settimeout(2)
    try:
        sock.connect((host, port))
        sock.close()
        sys.exit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(2)
    finally:
        try:
            sock.close()
        except Exception:
            pass
print(f"port {host}:{port} not reachable: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

PROFILE="${PROFILE:-starvla_oft_fp16}"
ACTION="${ACTION:-run}"
SUITE="${SUITE:-spatial}"
GPU_LIST="${GPU_LIST:-0}"
PORT_BASE="${PORT_BASE:-8200}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-10}"
GR00T_EVAL_INIT_OFFSET="${GR00T_EVAL_INIT_OFFSET:-10}"
CHECKPOINTS_ROOT="${CHECKPOINTS_ROOT:-${HOME}/ckpts}"
AWESOME_QVLA_ROOT="${AWESOME_QVLA_ROOT:-${QUANTVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
QUANTVLA_ROOT="${QUANTVLA_ROOT:-${AWESOME_QVLA_ROOT}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
STARVLA_CHECKPOINT="${STARVLA_CHECKPOINT:-}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
STARVLA_UNNORM_KEY="${STARVLA_UNNORM_KEY:-}"
STARVLA_USE_BF16="${STARVLA_USE_BF16:-1}"
SAVE_VIDEO="${SAVE_VIDEO:-False}"
MAX_TASKS="${MAX_TASKS:--1}"
LIBERO_ROOT_ARG="${LIBERO_ROOT:-}"
SEED="${SEED:-7}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --action|--mode) ACTION="${2:?missing --action value}"; shift 2 ;;
        --profile) PROFILE="${2:?missing --profile value}"; shift 2 ;;
        --suite) SUITE="${2:?missing --suite value}"; shift 2 ;;
        --checkpoint|--starvla-checkpoint) STARVLA_CHECKPOINT="${2:?missing checkpoint value}"; shift 2 ;;
        --python|--starvla-python) STARVLA_PYTHON="${2:?missing python value}"; shift 2 ;;
        --gpus|--gpu-list) GPU_LIST="${2:?missing --gpus value}"; shift 2 ;;
        --trials) NUM_TRIALS_PER_TASK="${2:?missing --trials value}"; shift 2 ;;
        --init-offset) GR00T_EVAL_INIT_OFFSET="${2:?missing --init-offset value}"; shift 2 ;;
        --port-base) PORT_BASE="${2:?missing --port-base value}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:?missing --output-root value}"; shift 2 ;;
        --checkpoints-root) CHECKPOINTS_ROOT="${2:?missing --checkpoints-root value}"; shift 2 ;;
        --libero-root) LIBERO_ROOT_ARG="${2:?missing --libero-root value}"; shift 2 ;;
        --unnorm-key|--starvla-unnorm-key) STARVLA_UNNORM_KEY="${2:?missing --unnorm-key value}"; shift 2 ;;
        --use-bf16|--starvla-use-bf16) STARVLA_USE_BF16="${2:?missing --use-bf16 value}"; shift 2 ;;
        --save-video) SAVE_VIDEO="${2:?missing --save-video value}"; shift 2 ;;
        --max-tasks) MAX_TASKS="${2:?missing --max-tasks value}"; shift 2 ;;
        --seed) SEED="${2:?missing --seed value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option '$1'" ;;
    esac
done

case "$ACTION" in
    plan|run|result) ;;
    *) die "unknown action '$ACTION'" ;;
esac

BACKEND_ROOT="${QUANTVLA_ROOT}/third_party/starvla"
[[ -d "$BACKEND_ROOT" ]] || die "StarVLA backend root not found: $BACKEND_ROOT"

TASK_SUITE="$(suite_to_task_suite "$SUITE")"
SUITE_SHORT="$(suite_to_short "$TASK_SUITE")"
SUITE="$SUITE_SHORT"
OUTPUT_ROOT="${OUTPUT_ROOT:-${QUANTVLA_ROOT}/results/awesome_quant_vla/${PROFILE}_${SUITE}}"
LOG_DIR="${OUTPUT_ROOT}/logs"
EVAL_LOG="${LOG_DIR}/eval_stdout.log"
SERVER_LOG="${LOG_DIR}/server.log"
ROLLOUT_DIR="${OUTPUT_ROOT}/rollouts"
PYTHON_BIN="$STARVLA_PYTHON"

if [[ -z "$STARVLA_CHECKPOINT" ]]; then
    STARVLA_CHECKPOINT="${CHECKPOINTS_ROOT}/starvla/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt"
fi

FIRST_GPU="${GPU_LIST%%,*}"
export CUDA_VISIBLE_DEVICES="$FIRST_GPU"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-$FIRST_GPU}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export NO_ALBUMENTATIONS_UPDATE="${NO_ALBUMENTATIONS_UPDATE:-1}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:-1}"
export STARVLA_BACKEND_ROOT="$BACKEND_ROOT"

if [[ -n "$LIBERO_ROOT_ARG" ]]; then
    export LIBERO_ROOT="$LIBERO_ROOT_ARG"
fi
quantvla_setup_cache_dirs
quantvla_setup_libero_config
quantvla_export_pythonpath
export PYTHONPATH="${BACKEND_ROOT}:${QUANTVLA_ROOT}/tools:${PYTHONPATH:-}"

print_plan() {
    cat <<EOF
================================================================
[starvla] plan
  profile        : ${PROFILE}
  action         : ${ACTION}
  suite          : ${SUITE} (${TASK_SUITE})
  project root   : ${QUANTVLA_ROOT}
  backend root   : ${BACKEND_ROOT}
  python         : ${PYTHON_BIN}
  checkpoint     : ${STARVLA_CHECKPOINT}
  output         : ${OUTPUT_ROOT}
  gpus / port    : ${GPU_LIST} / ${PORT_BASE}
  trials/task    : ${NUM_TRIALS_PER_TASK}
  steps wait     : ${GR00T_EVAL_INIT_OFFSET}
  max tasks      : ${MAX_TASKS}
  use bf16       : ${STARVLA_USE_BF16}
  save video     : ${SAVE_VIDEO}
  unnorm key     : ${STARVLA_UNNORM_KEY:-auto}
================================================================
EOF
}

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
}

run_parse() {
    "$PYTHON_BIN" "${QUANTVLA_ROOT}/tools/starvla/parse_starvla_eval.py" \
        --log-file "$EVAL_LOG" \
        --output-root "$OUTPUT_ROOT" \
        --task-suite-name "$TASK_SUITE" \
        --benchmark "${PROFILE}_${SUITE}" \
        --quantization "FP16/BF16" \
        --gpus "$GPU_LIST"
}

run_eval() {
    [[ -f "$STARVLA_CHECKPOINT" ]] || die "checkpoint not found: $STARVLA_CHECKPOINT"
    mkdir -p "$LOG_DIR" "$ROLLOUT_DIR"

    pushd "$BACKEND_ROOT" >/dev/null

    local server_args=(
        deployment/model_server/server_policy.py
        --ckpt_path "$STARVLA_CHECKPOINT"
        --port "$PORT_BASE"
    )
    if bool_true "$STARVLA_USE_BF16"; then
        server_args+=(--use_bf16)
    fi

    CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PYTHON_BIN" "${server_args[@]}" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    trap cleanup EXIT

    if ! wait_for_port 127.0.0.1 "$PORT_BASE"; then
        echo "[starvla] server failed to open port. Last server log lines:" >&2
        tail -n 120 "$SERVER_LOG" >&2 || true
        exit 1
    fi

    local eval_args=(
        examples/LIBERO/eval_files/eval_libero.py
        --args.pretrained-path "$STARVLA_CHECKPOINT"
        --args.host "127.0.0.1"
        --args.port "$PORT_BASE"
        --args.task-suite-name "$TASK_SUITE"
        --args.num-trials-per-task "$NUM_TRIALS_PER_TASK"
        --args.num-steps-wait "$GR00T_EVAL_INIT_OFFSET"
        --args.video-out-path "$ROLLOUT_DIR"
        --args.max-tasks "$MAX_TASKS"
        --args.seed "$SEED"
    )
    if [[ -n "$STARVLA_UNNORM_KEY" ]]; then
        eval_args+=(--args.unnorm-key "$STARVLA_UNNORM_KEY")
    fi
    if bool_true "$SAVE_VIDEO"; then
        eval_args+=(--args.save-video)
    fi

    set +e
    CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PYTHON_BIN" "${eval_args[@]}" 2>&1 | tee "$EVAL_LOG"
    local eval_status=${PIPESTATUS[0]}
    set -e

    cleanup
    trap - EXIT
    popd >/dev/null

    if [[ "$eval_status" -ne 0 ]]; then
        echo "[starvla] eval failed with exit=${eval_status}. Logs: ${LOG_DIR}" >&2
        tail -n 120 "$SERVER_LOG" >&2 || true
        exit "$eval_status"
    fi

    run_parse
}

if [[ "$ACTION" != "result" ]]; then
    print_plan
fi

if [[ "$ACTION" == "plan" ]]; then
    exit 0
elif [[ "$ACTION" == "result" ]]; then
    print_result "$OUTPUT_ROOT"
elif [[ "$ACTION" == "run" ]]; then
    run_eval
else
    die "unknown action '$ACTION'"
fi
