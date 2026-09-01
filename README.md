# quant-vla

quant-vla is an open-source engineering stack for Vision-Language-Action (VLA) post-training quantization and LIBERO evaluation. It brings QuantVLA, Omega-QVLA, QVLA/OpenVLA, OpenDriveLab/UniVLA, and StarVLA style routes into one repository with a shared launcher, shared asset conventions, and reproducible validation notes.

> This repository is under active development. The current focus is research reproduction, quantized evaluation, and hardware-portability preparation. Large checkpoints, quantized packs, datasets, and generated benchmark outputs are intentionally kept outside git.

Day-one workflow:

1. Environment + assets: create the conda environment, stage checkpoints and LIBERO assets outside the repository.
2. Plan: run a profile with `--action plan` to verify paths, GPU selection, ports, checkpoints, and quantization mode.
3. Evaluate: launch the selected VLA quantization route on a LIBERO suite.
4. Inspect: read `merged_summary.json`, `merged_summary.md`, logs, rollouts, calibration files, packs, or QVLA proxy artifacts before making any benchmark claim.

## Links

- [Installation](docs/installation.md)
- [Checkpoints and Quantized Packs](docs/checkpoints.md)
- [Verification Guide](docs/verification.md)
- [UniVLA Guide](docs/univla.md)
- [StarVLA Guide](docs/starvla.md)
- [Chinese Documentation](docs/zh_cn/README.md)

## Community

Join the RadixRootMind China developer WeChat group:

<p align="center">
  <img src="assets/radixrootmind-wechat-group.png" alt="RadixRootMind China developer WeChat group QR code" width="360">
</p>

## What quant-vla Provides

| Surface | Purpose | Entry point |
| --- | --- | --- |
| Unified profile launcher | One command surface for GR00T, Pi0.5/OpenPI, OpenVLA, OpenVLA-OFT, UniVLA, and StarVLA routes. | `scripts/run_awesome_quant_vla.sh` |
| GR00T evaluation | LIBERO evaluation for GR00T-N1.5 FP16, W4A8, W4A4 GPTQ, W4A4 DuQuant, and W4A4 RTN routes. | `scripts/run_groot_benchmark.sh` |
| Pi0.5/OpenPI evaluation | LIBERO evaluation through OpenPI service mode, with FP16, W4A8, W4A4 GPTQ, and W4A4 RTN profiles. | `scripts/run_pi05_libero_benchmark.sh` |
| OpenVLA/QVLA evaluation | QVLA-style calibration, Hessian proxy, bit assignment, fake-weight injection, and LIBERO evaluation. | `scripts/run_openvla_qvla.sh`, `tools/qvla/` |
| UniVLA evaluation | UniVLA LIBERO FP16 evaluation with a separate latent-action decoder. | `scripts/run_univla_libero.sh` |
| StarVLA evaluation | StarVLA LIBERO FP16/BF16 evaluation through the StarVLA websocket policy server. | `scripts/run_starvla_libero.sh` |
| Quantization utilities | Runtime quantization wrappers, GPTQ pack builders, SVD/SVD-Hadamard utilities, and pack merge helpers. | `gr00t/quantization/`, `tools/` |
| Documentation | Reproducible setup, checkpoint layout, known fixes, and frozen validation results. | `docs/` |

## Supported Routes

The matrix below summarizes the routes currently exposed through the unified launcher.

| Model family | Profiles | Notes |
| --- | --- | --- |
| GR00T-N1.5 | `groot_fp16`, `groot_w4a8`, `groot_w4a4_gptq`, `groot_w4a4_duquant`, `groot_w4a4_rtn` | Supports runtime W4A8 and W4A4 pack-based evaluation. |
| Pi0.5/OpenPI | `pi05_fp16`, `pi05_w4a8_duquant`, `pi05_w4a4_gptq`, `pi05_w4a4_rtn`, `pi05_duquant_w4a8` | Uses OpenPI for model serving and LIBERO for evaluation. |
| OpenVLA | `openvla_fp16`, `openvla_qvla_w8` | Supports FP16 and QVLA mixed-bit W8 evaluation. |
| OpenVLA-OFT | `openvla_oft_fp16`, `openvla_oft_qvla_w8` | Supports FP16 and QVLA mixed-bit W8 evaluation. |
| UniVLA | `univla_fp16` | FP16 route is validated with an external action decoder. Quantized UniVLA evaluation is not yet advertised as a verified route. |
| StarVLA | `starvla_oft_fp16`, `starvla_gr00t_fp16`, `starvla_pi_fp16`, `starvla_fast_fp16` | `starvla_oft_fp16` is validated on LIBERO Spatial. Other FP16/BF16 entries require matching StarVLA checkpoints. Quantized StarVLA routes require a future Qwen-VL/action-head adapter before being advertised as validated. |

## Validation Snapshot

These are local engineering validation results after the merge. They show that the routes run end-to-end in this repository. They are not official paper benchmark numbers.

| Profile | Model | Suite | Quantization route | Result |
| --- | --- | --- | --- | ---: |
| `groot_w4a8` | GR00T-N1.5 | LIBERO Object | Runtime DuQuant/ATM/OHB | 82.0% |
| `pi05_w4a8_duquant` | Pi0.5/OpenPI | LIBERO Object | Runtime DuQuant/ATM/OHB | 99.0% |
| `pi05_w4a4_gptq` | Pi0.5/OpenPI | LIBERO Object | W4A4 GPTQ/SVD-Hadamard pack | 98.0% |
| `openvla_fp16` | OpenVLA | LIBERO Spatial | FP16 baseline | 70.0% |
| `openvla_qvla_w8` | OpenVLA | LIBERO Spatial | QVLA mixed-bit W8 | 80.0% |
| `openvla_oft_fp16` | OpenVLA-OFT | LIBERO Spatial | FP16 baseline | 60.0% |
| `openvla_oft_qvla_w8` | OpenVLA-OFT | LIBERO Spatial | QVLA mixed-bit W8 | 26.0% |
| `univla_fp16` | UniVLA | LIBERO Spatial | FP16 with action decoder | 96.0% |
| `starvla_oft_fp16` | StarVLA-OFT | LIBERO Spatial | FP16/BF16 policy-server evaluation | 99.0% |

## Verified Environment

The validation results above were reproduced on the following local workstation configuration:

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA A100 40GB |
| CPU | Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz |
| System memory | 96 GB |
| Storage | 200 GB SSD |

## From Clone To First Run

Clone the repository and create a local environment file:

```bash
git clone https://github.com/RadixRootMind/quant-vla.git quant-vla
cd quant-vla

cp .env.example .env.local
source .env.local
```

Optional settings for mainland China networks:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

Create the main GR00T/Pi0.5 environment:

```bash
conda create -n awesome_quant_vla python=3.10 -y
conda activate awesome_quant_vla

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[base]"
python -m pip install "imageio[ffmpeg]" "huggingface_hub[cli]"
```

Prepare LIBERO:

```bash
mkdir -p "$LIBERO_ROOT/datasets" "$LIBERO_CONFIG_PATH"

cat > "$LIBERO_CONFIG_PATH/config.yaml" <<EOF
benchmark_root: $LIBERO_ROOT/libero/libero
bddl_files: $LIBERO_ROOT/libero/libero/bddl_files
init_states: $LIBERO_ROOT/libero/libero/init_files
datasets: $LIBERO_ROOT/datasets
assets: $LIBERO_ROOT/libero/libero/assets
EOF
```

Check the active runtime:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

Run a small GR00T W4A8 smoke test:

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --gpus 0 \
  --trials 1 \
  --init-offset 10
```

## Assets And Checkpoints

Checkpoints and quantized packs are not committed. The default layout is:

```bash
$CHECKPOINTS_ROOT/
  gr00t-n1.5-libero-object-posttrain/
  gr00t-n1.5-libero-spatial-posttrain/
  gr00t-n1.5-libero-goal-posttrain/
  gr00t-n1.5-libero-long-posttrain/
  pi05_libero_pytorch/
  openvla-7b-finetuned-libero-spatial/
  openvla-7b-oft-finetuned-libero-spatial/
  univla-7b-224-sft-libero/
  starvla/Qwen3-VL-OFT-LIBERO-4in1/
  starvla/Qwen3-VL-4B-Instruct/
```

W4A4 GPTQ profiles also need quantized packs:

```bash
$AWESOME_QVLA_ROOT/results/packs/
  gr00t_object/quantized.pt
  gr00t_spatial/quantized.pt
  gr00t_goal/quantized.pt
  gr00t_long/quantized.pt
  pi05_object/quantized.pt
```

Important route differences:

- Runtime W4A8 routes, such as `groot_w4a8` and `pi05_w4a8_duquant`, do not require a prebuilt `quantized.pt` pack.
- W4A4 GPTQ routes, such as `groot_w4a4_gptq` and `pi05_w4a4_gptq`, require an existing or locally built `quantized.pt` pack.
- QVLA W8 routes generate calibration JSONL, Hessian proxy, gate/bit-allocation artifacts, and evaluation outputs. The proxy file is an analysis artifact, not a standalone deployable quantized model.
- Pi0.5 quantized routes require an OpenPI PyTorch checkpoint converted from the official OpenPI JAX/Orbax checkpoint.

See [docs/checkpoints.md](docs/checkpoints.md) for exact download and conversion commands.

## Run Evaluation

Always start with a plan:

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a8_duquant \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --action plan
```

Run Pi0.5 W4A8:

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a8_duquant \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --openpi-root "$OPENPI_ROOT" \
  --openpi-py "$OPENPI_PY" \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch"
```

Run Pi0.5 W4A4 GPTQ:

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --openpi-root "$OPENPI_ROOT" \
  --openpi-py "$OPENPI_PY" \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch" \
  --pi05-gptq-pack "$AWESOME_QVLA_ROOT/results/packs/pi05_object/quantized.pt"
```

Run OpenVLA QVLA W8 from the OpenVLA-specific environment:

```bash
conda activate awesome_qvla_openvla
cd "$AWESOME_QVLA_ROOT"
source .env.local
export OPENVLA_ATTN_IMPL=eager

bash scripts/run_awesome_quant_vla.sh openvla_qvla_w8 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python "$OPENVLA_PYTHON" \
  --openvla-checkpoint "$OPENVLA_CKPT" \
  --max-samples 32
```

Run UniVLA FP16:

```bash
conda activate awesome_qvla_openvla
cd "$AWESOME_QVLA_ROOT"
source .env.local
export UNIVLA_ATTN_IMPL=eager

bash scripts/run_awesome_quant_vla.sh univla_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --univla-python "$UNIVLA_PYTHON" \
  --univla-checkpoint "$UNIVLA_CKPT" \
  --univla-action-decoder "$UNIVLA_ACTION_DECODER"
```

Run StarVLA-OFT FP16/BF16:

```bash
conda activate awesome_qvla_starvla
cd "$AWESOME_QVLA_ROOT"
source .env.local

bash scripts/run_awesome_quant_vla.sh starvla_oft_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --port-base 8200 \
  --starvla-python "$STARVLA_PYTHON" \
  --starvla-checkpoint "$STARVLA_CKPT" \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/starvla_oft_fp16_spatial_final"
```

Read a saved result:

```bash
bash scripts/run_awesome_quant_vla.sh <profile> \
  --suite <suite> \
  --output-root /path/to/result_dir \
  --action result
```

## Inspect Outputs

Each successful evaluation writes a structured result directory:

```bash
results/<run_name>/
  merged_summary.json
  merged_summary.md
  logs/
  summaries/
  rollouts/
  act_stats/
  packdir/
```

Typical files:

- `merged_summary.json`: machine-readable aggregate success rate and per-task results.
- `merged_summary.md`: human-readable result table.
- `logs/run.log`: launcher-level log.
- `logs/server_shard_*.log`: inference server log.
- `logs/eval_shard_*.stdout.log`: LIBERO evaluation stdout.
- `act_stats/`: activation statistics collected during quantized runs.
- `packdir/`: temporary per-run pack staging area.
- `proxy/proxy.pt`, `gates/`, `calib/`: QVLA analysis artifacts for OpenVLA routes.

## Environment Notes

Recommended GR00T/Pi0.5 runtime:

| Component | Recommended version |
| --- | --- |
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| CUDA runtime | 12.4 |
| PyTorch | 2.5.1 |
| torchvision | 0.20.1 |
| TensorFlow | 2.15.0 |
| transformers | 4.51.3 |
| numpy | 1.26.4 |
| opencv-python-headless | 4.9.0.80 |
| robosuite | 1.4.0 |
| mujoco | 3.3.7 |
| protobuf | 3.20.3 |

OpenVLA/QVLA and UniVLA are best kept in a separate environment:

| Component | Recommended version |
| --- | --- |
| Python | 3.10 |
| CUDA runtime | 12.1 |
| PyTorch | 2.2.0 |
| torchvision | 0.17.0 |
| transformers | 4.40.1 |
| tokenizers | 0.19.1 |
| peft | 0.11.1 |
| timm | 0.9.10 |
| TensorFlow | 2.15.0 |
| tensorflow-datasets | 4.9.3 |
| protobuf | 3.20.3 |
| numpy | 1.26.4 |
| opencv-python-headless | 4.9.0.80 |


StarVLA is best kept in a third environment because it uses a newer Qwen/Transformers stack:

| Component | Recommended version |
| --- | --- |
| Environment | `awesome_qvla_starvla` |
| Python | 3.10 |
| transformers | 4.57.1 |
| accelerate | 1.5.2 |
| torchvision | 0.21.0 |
| deepspeed | 0.16.9 |
| numpy | 1.26.4 |
| StarVLA checkpoint format | `steps_XXXXX_pytorch_model.pt` |

## Development Checks

Before pushing a route or claiming a validation result, run lightweight checks first:

```bash
bash -n scripts/run_awesome_quant_vla.sh
bash -n scripts/run_groot_benchmark.sh
bash -n scripts/run_pi05_libero_benchmark.sh
bash -n scripts/run_openvla_qvla.sh
bash -n scripts/run_univla_libero.sh
bash -n scripts/run_starvla_libero.sh

PYTHONPATH=. python -m compileall -q gr00t tools scripts examples
```

When adding or changing a model route:

1. Keep the upstream project as provenance, but route users through `scripts/run_awesome_quant_vla.sh`.
2. Declare required checkpoints, packs, runtime variables, and environment assumptions in docs.
3. Add `--action plan` output that shows the resolved paths and quantization mode.
4. Run a small smoke test before full LIBERO evaluation.
5. Record evidence in `docs/verification.md` only after the summary file is produced and the result command works.

## Repository Layout

```text
quant-vla
|-- .env.example                         # Local path and runtime variable template
|-- docs/                                # English and Chinese setup and verification notes
|-- scripts/                             # User-facing launchers
|-- tools/                               # Pack builders, QVLA utilities, parsing helpers
|-- gr00t/                               # GR00T model and quantization integration
|-- examples/Libero/                     # LIBERO evaluation entrypoints
|-- atm_alpha_beta_pi05/                 # Pi0.5 ATM/OHB calibration presets
|-- third_party/
|   |-- openvla                          # OpenVLA backend snapshot
|   |-- openvla_oft                      # OpenVLA-OFT backend snapshot
|   |-- univla                           # UniVLA backend snapshot
|   `-- starvla                          # StarVLA backend snapshot
|-- tests/                               # Lightweight regression checks
`-- results/                             # Local outputs, packs, and run artifacts; ignored by git
```

## Scope And Non-goals

quant-vla is a research and engineering integration project. It focuses on quantization workflows, route unification, and LIBERO evaluation.

It does not bundle large checkpoints or datasets. It also does not claim that every route emits a standalone deployable quantized model for real robots or non-NVIDIA accelerators. Some routes evaluate quantized behavior at runtime, some load prebuilt `quantized.pt` packs, and QVLA routes produce calibration/proxy/bit-allocation artifacts. Real robot deployment or accelerator deployment still requires model export, runtime conversion, operator support, latency validation, and hardware-specific integration.

## Roadmap

- Add stronger route-level smoke tests for every public profile.
- Improve checkpoint and pack discovery for offline environments.
- Add hardware adapter notes for non-NVIDIA accelerators such as DCU and NPU/IPU platforms.
- Promote UniVLA quantized profiles after action-decoder-aware evaluation is validated.
- Add normalized run manifests and scorecards for easier result comparison.

## Lineage And Credits

quant-vla integrates and adapts ideas and code paths from QuantVLA, Omega-QVLA, QVLA/OpenVLA, OpenDriveLab/UniVLA, StarVLA, OpenPI, and LIBERO. Please check the original repositories and licenses when using or redistributing derived components.

