# StarVLA Integration Guide

This repository vendors StarVLA under `third_party/starvla` as an additional VLA baseline route. The current integration focuses on LIBERO FP16/BF16 evaluation through StarVLA's own policy-server architecture. Quantized StarVLA routes are intentionally not advertised as validated yet because StarVLA uses Qwen-VL / world-model backbones and action heads that need a separate quantization adapter.

## Supported Profile

| Profile | Purpose | Status |
| --- | --- | --- |
| `starvla_oft_fp16` | StarVLA-OFT LIBERO evaluation with BF16/FP16 inference | Integrated and validated on LIBERO Spatial, 10 trials per task, result 99.0% |
| `starvla_gr00t_fp16` | StarVLA-GR00T LIBERO evaluation | Integrated entry, requires matching checkpoint |
| `starvla_pi_fp16` | StarVLA-PI LIBERO evaluation | Integrated entry, requires matching checkpoint |
| `starvla_fast_fp16` | StarVLA-FAST LIBERO evaluation | Integrated entry, requires matching checkpoint |

## Validated Result

The integrated StarVLA-OFT route has completed a full LIBERO Spatial validation run:

| Profile | Suite | Trials | Episodes | Result |
| --- | --- | ---: | ---: | ---: |
| `starvla_oft_fp16` | LIBERO Spatial | 10 per task | 100 | 99.0% |

The `EGL_NOT_INITIALIZED` message printed during cleanup is a known robosuite/MuJoCo warning and does not invalidate the result when `merged_summary.json` is produced.

## Checkpoint Layout

The default StarVLA-OFT checkpoint path is:

```bash
$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
```

Download the StarVLA policy checkpoint:

```bash
mkdir -p "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1"
huggingface-cli download StarVLA/Qwen3-VL-OFT-LIBERO-4in1 \
  --local-dir "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1"
```

The policy checkpoint still loads a Qwen3-VL base VLM from StarVLA's upstream relative path `playground/Pretrained_models/Qwen3-VL-4B-Instruct`. Keep the large base model under `$CHECKPOINTS_ROOT` and symlink it into the expected StarVLA path:

```bash
mkdir -p "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct"
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct"

mkdir -p "$AWESOME_QVLA_ROOT/third_party/starvla/playground/Pretrained_models"
ln -sfn "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct" \
  "$AWESOME_QVLA_ROOT/third_party/starvla/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
```

## Environment

Use a separate environment from GR00T/Pi0.5 and OpenVLA/OFT:

```bash
conda create -n awesome_qvla_starvla python=3.10 -y
conda activate awesome_qvla_starvla

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r third_party/starvla/requirements.txt
python -m pip install -e third_party/starvla --no-build-isolation
if [ -f "$LIBERO_ROOT/setup.py" ] || [ -f "$LIBERO_ROOT/pyproject.toml" ]; then
  python -m pip install -e "$LIBERO_ROOT" --no-build-isolation
elif [ -f "$LIBERO_ROOT/libero/setup.py" ] || [ -f "$LIBERO_ROOT/libero/pyproject.toml" ]; then
  python -m pip install -e "$LIBERO_ROOT/libero" --no-build-isolation
else
  echo "Cannot find LIBERO setup.py or pyproject.toml under $LIBERO_ROOT"
  find "$LIBERO_ROOT" -maxdepth 3 \( -name setup.py -o -name pyproject.toml \) -print
  exit 1
fi
python -m pip install "imageio[ffmpeg]" tyro matplotlib mediapy msgpack websockets
python -m pip install --no-user \
  "robosuite==1.4.0" \
  "mujoco==2.3.7" \
  "gym==0.23.1" \
  "bddl==1.0.1" \
  "future" \
  "cloudpickle" \
  "easydict" \
  "termcolor" \
  "h5py" \
  "opencv-python-headless==4.9.0.80" \
  "numpy==1.26.4"
```

For mainland China networks, switch the pip index and set Hugging Face mirror variables before installing or downloading.

## Run

Plan first:

```bash
bash scripts/run_awesome_quant_vla.sh starvla_oft_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 1 \
  --max-tasks 1 \
  --action plan
```

Smoke test:

```bash
bash scripts/run_awesome_quant_vla.sh starvla_oft_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 1 \
  --max-tasks 1 \
  --port-base 8200 \
  --starvla-python "$STARVLA_PYTHON" \
  --starvla-checkpoint "$STARVLA_CKPT"
```

Full LIBERO Spatial evaluation:

```bash
bash scripts/run_awesome_quant_vla.sh starvla_oft_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --port-base 8200 \
  --starvla-python "$STARVLA_PYTHON" \
  --starvla-checkpoint "$STARVLA_CKPT" \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/starvla_oft_fp16_spatial_final"
```

Read result:

```bash
bash scripts/run_awesome_quant_vla.sh starvla_oft_fp16 \
  --suite spatial \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/starvla_oft_fp16_spatial_final" \
  --action result
```

## Output

The StarVLA runner writes:

```text
results/.../
  merged_summary.json
  merged_summary.md
  logs/server.log
  logs/eval_stdout.log
  rollouts/              # only populated when --save-video True
```

Videos are disabled by default for faster headless validation and to avoid ffmpeg-related failures. Enable them with `--save-video True`.
