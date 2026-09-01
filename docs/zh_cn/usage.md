# 使用统一入口

本文档说明依赖和 checkpoint 准备好以后，如何通过统一入口运行 GR00T、Pi0.5/OpenPI、OpenVLA/OpenVLA-OFT。

## 1. 加载环境

```bash
cd /path/to/quant-vla
source .env.local
conda activate awesome_quant_vla
```

如果不使用 `.env.local`，可以手动导出：

```bash
export AWESOME_QVLA_ROOT=/root/VLM_REPO/quant-vla
export QUANTVLA_ROOT=$AWESOME_QVLA_ROOT
export CHECKPOINTS_ROOT=/root/VLM_REPO/checkpoints
export LIBERO_ROOT=/root/VLM_REPO/LIBERO
export OPENPI_ROOT=/root/VLM_REPO/openpi
export OPENPI_PY=$OPENPI_ROOT/.venv/bin/python
export PYTHONPATH=$LIBERO_ROOT:$AWESOME_QVLA_ROOT:${PYTHONPATH:-}
```

## 2. 命令格式

```bash
bash scripts/run_awesome_quant_vla.sh PROFILE \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10
```

`--suite object --trials 10` 表示 10 个 object task，每个 task 10 次，共 100 个 episode。

先看计划：

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --action plan
```

读取结果：

```bash
bash scripts/run_awesome_quant_vla.sh PROFILE \
  --suite object \
  --output-root /path/to/result_dir \
  --action result
```

## 3. Profile 列表

| profile | 说明 |
| --- | --- |
| `groot_w4a8` | GR00T W4A8 DuQuant + ATM/OHB |
| `groot_w4a4_gptq` | GR00T W4A4 GPTQ/SVD-Hadamard pack |
| `groot_w4a4_duquant` | GR00T runtime DuQuant W4A4 |
| `groot_w4a4_rtn` | GR00T RTN W4A4 baseline |
| `groot_fp16` | GR00T FP16 baseline |
| `pi05_w4a8_duquant` | Pi0.5/OpenPI W4A8 DuQuant + ATM/OHB |
| `pi05_w4a4_gptq` | Pi0.5/OpenPI W4A4 GPTQ/SVD-Hadamard pack |
| `pi05_duquant_w4a8` | Pi0.5/OpenPI generic W4A8 DuQuant |
| `pi05_w4a4_rtn` | Pi0.5/OpenPI RTN W4A4 baseline |
| `pi05_fp16` | Pi0.5/OpenPI FP16 baseline |
| `openvla_fp16` | OpenVLA FP16 LIBERO 评测 |
| `openvla_qvla_w8` | OpenVLA QVLA mixed-bit 平均 8 bit |
| `openvla_oft_fp16` | OpenVLA-OFT FP16 LIBERO 评测 |
| `openvla_oft_qvla_w8` | OpenVLA-OFT QVLA mixed-bit 平均 8 bit |

## 4. GR00T 示例

GR00T W4A8：

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10
```

GR00T W4A4 GPTQ：

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a4_gptq \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --groot-gptq-pack "$AWESOME_QVLA_ROOT/results/packs/gr00t_object/quantized.pt"
```

## 5. Pi0.5 示例

Pi0.5 W4A8：

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a8_duquant \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch"
```

Pi0.5 W4A4 GPTQ：

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a4_gptq \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --init-offset 10 \
  --openpi-checkpoint "$CHECKPOINTS_ROOT/pi05_libero_pytorch" \
  --pi05-gptq-pack "$AWESOME_QVLA_ROOT/results/packs/pi05_object/quantized.pt"
```

## 6. OpenVLA/QVLA 示例

OpenVLA-OFT FP16：

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python /path/to/openvla-env/bin/python \
  --openvla-checkpoint "$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial"
```

OpenVLA-OFT QVLA W8 一键运行：

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 \
  --suite spatial \
  --gpus 0 \
  --trials 10 \
  --openvla-python /path/to/openvla-env/bin/python \
  --openvla-checkpoint "$CHECKPOINTS_ROOT/openvla-7b-oft-finetuned-libero-spatial" \
  --target-avg-bits 8.0
```

也可以拆成阶段执行：

```bash
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 --suite spatial --action build_calib
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 --suite spatial --action proxy
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 --suite spatial --action assign
bash scripts/run_awesome_quant_vla.sh openvla_oft_qvla_w8 --suite spatial --action eval
```