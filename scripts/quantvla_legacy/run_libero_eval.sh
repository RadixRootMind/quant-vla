#!/bin/bash
# Script to run Libero evaluation
# Usage: ./run_libero_eval.sh [task_suite_name] [extra args...]
# task_suite_name: libero_spatial (default), libero_goal, libero_object, libero_90, libero_10

TASK=${1:-libero_10}
shift || true
EXTRA_ARGS=("$@")

HEADLESS_FLAG="no"
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--headless" ]]; then
        HEADLESS_FLAG="yes"
        break
    fi
done

# Activate libero_test environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate libero_test

# Add QuantVLA_GR00T and LIBERO to Python path
export PYTHONPATH=/home/jz97/VLM_REPO/groot_test/QuantVLA_GR00T:/home/jz97/VLM_REPO/Isaac-GR00T/LIBERO:$PYTHONPATH

echo "=========================================="
echo "Running Libero evaluation for $TASK"
echo "Headless mode: $HEADLESS_FLAG"
echo "Port: 5556 (GR00T)"
echo "=========================================="
echo ""
echo "Make sure the inference server is running in another terminal!"
echo "Run: ./run_inference_server.sh $TASK"
echo ""
echo "Results will be saved to:"
echo "  - Log: /tmp/logs/libero_eval_${TASK}.log"
echo "  - Videos: /tmp/logs/rollout_*.mp4"
echo "=========================================="
echo ""

cd /home/jz97/VLM_REPO/groot_test/QuantVLA_GR00T/examples/Libero/eval

python run_libero_eval.py --task_suite_name "$TASK" --port 5556 "${EXTRA_ARGS[@]}"
