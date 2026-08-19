# UniVLA Guide

Awesome-quant-vla includes OpenDriveLab/UniVLA as a fourth backend family.

The current integration exposes the UniVLA LIBERO FP16 evaluation route. UniVLA's source tree also contains QVLA-related scripts, but Awesome-quant-vla does not yet mark a UniVLA quantized route as verified because UniVLA evaluation requires both latent action generation and an external action decoder.

## Included Files

| Component | Path |
| --- | --- |
| UniVLA backend snapshot | `third_party/univla` |
| Unified runner | `scripts/run_univla_libero.sh` |
| Unified profile | `univla_fp16` |

## Required Checkpoint Files

The LIBERO route requires:

- A Hugging Face-format UniVLA checkpoint directory, usually one suite subdirectory from `qwbu/univla-7b-224-sft-libero`.
- An action decoder file, usually `action_decoder.pt` inside the checkpoint directory.

Suggested local layout:

```bash
$CHECKPOINTS_ROOT/
  univla-7b-224-sft-libero/
    univla-libero-spatial/
      config.json
      model.safetensors.index.json
      dataset_statistics.json
      action_decoder.pt
    univla-libero-object/
    univla-libero-goal/
    univla-libero-10/
```

Download only the spatial checkpoint:

```bash
mkdir -p "$CHECKPOINTS_ROOT"

hf download qwbu/univla-7b-224-sft-libero \
  --include "univla-libero-spatial/*" \
  --local-dir "$CHECKPOINTS_ROOT/univla-7b-224-sft-libero"
```

The full repository contains four LIBERO suite checkpoints and is much larger. Downloading the whole repository is only needed if you plan to evaluate all suites.

If the model card or upstream location changes, keep the local directory stable or pass explicit paths:

```bash
--univla-checkpoint /path/to/univla-7b-224-sft-libero/univla-libero-spatial
--univla-action-decoder /path/to/univla-7b-224-sft-libero/univla-libero-spatial/action_decoder.pt
```

## Environment

Use the OpenVLA-style environment. The validated dependency family is PyTorch 2.2.0 CUDA 12.1, transformers 4.40.1, tokenizers 0.19.1, peft 0.11.1, timm 0.9.10, TensorFlow 2.15.0, tensorflow-datasets 4.9.3, protobuf 3.20.3, numpy 1.26.4, and opencv-python-headless 4.9.0.80.

Install UniVLA-specific extras in the same environment:

```bash
python -m pip install --no-user \
  braceexpand \
  "webdataset==0.2.111" \
  "ema-pytorch==0.5.1" \
  "rotary-embedding-torch==0.8.4" \
  "hydra-core==1.3.2" \
  "omegaconf==2.3.0" \
  "piq==0.8.0" \
  "pyquaternion==0.9.9"
```

If your mirror cannot find `braceexpand`, install it from another index or conda-forge:

```bash
python -m pip install --no-user braceexpand -i https://pypi.org/simple
# or
conda install -y -c conda-forge braceexpand
```

UniVLA defaults to eager attention through:

```bash
export UNIVLA_ATTN_IMPL=eager
```

## Run

```bash
conda activate awesome_qvla_openvla
cd "$AWESOME_QVLA_ROOT"
source .env.local

bash scripts/run_awesome_quant_vla.sh univla_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --univla-python "$UNIVLA_PYTHON" \
  --univla-checkpoint "$UNIVLA_CKPT" \
  --univla-action-decoder "$UNIVLA_ACTION_DECODER" \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/univla_fp16_spatial"
```

Read the result:

```bash
bash scripts/run_awesome_quant_vla.sh univla_fp16 \
  --suite spatial \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/univla_fp16_spatial" \
  --action result
```

## Current Status

- `univla_fp16`: integrated and validated on LIBERO Spatial with 10 trials per task, result `96.0%`.
- `univla_qvla_w8`: not exposed as a verified profile yet.
