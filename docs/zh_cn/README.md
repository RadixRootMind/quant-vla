# quant-vla 中文说明

quant-vla 是一个面向 Vision-Language-Action（VLA）模型后训练量化与 LIBERO 评测的工程化仓库。当前代码把 QuantVLA、Omega-QVLA、QVLA/OpenVLA、OpenVLA-OFT 和 OpenDriveLab/UniVLA 相关路线统一到一个项目中，提供统一入口、统一路径约定、统一结果输出和可复现实验记录。

> 说明：本仓库仍处于工程整合与验证阶段。当前重点是研究复现、量化评测、路线统一和国产算力适配前的准备工作。大型 checkpoint、量化 pack、数据集和评测输出不会提交到 git。

推荐工作流：

1. 准备环境与资产：创建 conda 环境，把 checkpoint、LIBERO 数据和量化 pack 放到仓库外部路径。
2. 先看计划：用 `--action plan` 检查路径、GPU、端口、checkpoint 和量化方式是否正确。
3. 启动评测：用统一脚本运行指定模型与量化路线。
4. 查看结果：以 `merged_summary.json` 和 `merged_summary.md` 为准，同时检查日志、rollout、校准文件和 pack。

## 文档入口

- [English README](../../README.md)
- [Checkpoints 与量化包](checkpoints.md)
- [验证步骤](verification.md)
- [UniVLA 说明](univla.md)
- [英文安装说明](../installation.md)

## 社群

加入 RadixRootMind 中国区开发者微信群：

<p align="center">
  <img src="../../assets/radixrootmind-wechat-group.png" alt="RadixRootMind 中国区开发者微信群二维码" width="360">
</p>

## 当前支持能力

| 模型路线 | Profile | 说明 |
| --- | --- | --- |
| GR00T-N1.5 | `groot_fp16` | FP16 baseline。 |
| GR00T-N1.5 | `groot_w4a8` | W4A8 runtime DuQuant/ATM/OHB 量化评测，不需要预先生成 `quantized.pt`。 |
| GR00T-N1.5 | `groot_w4a4_gptq` | W4A4 GPTQ/SVD-Hadamard pack 评测，需要已有或自行构建的 `quantized.pt`。 |
| GR00T-N1.5 | `groot_w4a4_duquant` | W4A4 runtime DuQuant 评测。 |
| GR00T-N1.5 | `groot_w4a4_rtn` | W4A4 RTN 评测。 |
| Pi0.5/OpenPI | `pi05_fp16` | Pi0.5 FP16 baseline，使用 OpenPI service。 |
| Pi0.5/OpenPI | `pi05_w4a8_duquant` | Pi0.5 W4A8 runtime DuQuant/ATM/OHB 量化评测。 |
| Pi0.5/OpenPI | `pi05_w4a4_gptq` | Pi0.5 W4A4 GPTQ pack 评测，需要 `pi05_object/quantized.pt`。 |
| Pi0.5/OpenPI | `pi05_w4a4_rtn` | Pi0.5 W4A4 RTN 评测。 |
| OpenVLA | `openvla_fp16` | OpenVLA FP16 baseline。 |
| OpenVLA | `openvla_qvla_w8` | QVLA mixed-bit W8 评测，生成 calibration、proxy、gate/bit allocation 等中间产物。 |
| OpenVLA-OFT | `openvla_oft_fp16` | OpenVLA-OFT FP16 baseline。 |
| OpenVLA-OFT | `openvla_oft_qvla_w8` | OpenVLA-OFT QVLA mixed-bit W8 评测。 |
| UniVLA | `univla_fp16` | UniVLA FP16 LIBERO 评测，依赖独立 action decoder。 |

## 已验证结果

以下结果来自本仓库合并后的本地工程验证，用于证明路线能够端到端跑通；它们不是论文官方 benchmark 数字。

| Profile | 模型 | Suite | 量化路线 | 结果 |
| --- | --- | --- | --- | ---: |
| `groot_w4a8` | GR00T-N1.5 | LIBERO Object | Runtime DuQuant/ATM/OHB | 82.0% |
| `pi05_w4a8_duquant` | Pi0.5/OpenPI | LIBERO Object | Runtime DuQuant/ATM/OHB | 99.0% |
| `pi05_w4a4_gptq` | Pi0.5/OpenPI | LIBERO Object | W4A4 GPTQ/SVD-Hadamard pack | 98.0% |
| `openvla_fp16` | OpenVLA | LIBERO Spatial | FP16 baseline | 70.0% |
| `openvla_qvla_w8` | OpenVLA | LIBERO Spatial | QVLA mixed-bit W8 | 80.0% |
| `openvla_oft_fp16` | OpenVLA-OFT | LIBERO Spatial | FP16 baseline | 60.0% |
| `openvla_oft_qvla_w8` | OpenVLA-OFT | LIBERO Spatial | QVLA mixed-bit W8 | 26.0% |
| `univla_fp16` | UniVLA | LIBERO Spatial | FP16 + action decoder | 96.0% |

## 验证环境

以上验证结果来自以下本地工作站配置：

| 组件 | 配置 |
| --- | --- |
| GPU | NVIDIA A100 40GB |
| CPU | Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz |
| 系统内存 | 96 GB |
| 存储 | 200 GB SSD |

## 快速开始

克隆仓库并创建本地环境配置：

```bash
git clone https://github.com/RadixRootMind/quant-vla.git quant-vla
cd quant-vla

cp .env.example .env.local
source .env.local
```

国内网络可按需设置 Hugging Face 镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

创建 GR00T/Pi0.5 主环境：

```bash
conda create -n awesome_quant_vla python=3.10 -y
conda activate awesome_quant_vla

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[base]"
python -m pip install "imageio[ffmpeg]" "huggingface_hub[cli]"
```

准备 LIBERO 路径：

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

检查 CUDA 与 PyTorch：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

先跑一个 GR00T W4A8 小规模验证：

```bash
bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --gpus 0 \
  --trials 1 \
  --init-offset 10
```

## Checkpoint 与量化 Pack

checkpoint 和量化 pack 不进入 git。默认目录结构如下：

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
```

W4A4 GPTQ 路线还需要量化 pack：

```bash
$AWESOME_QVLA_ROOT/results/packs/
  gr00t_object/quantized.pt
  gr00t_spatial/quantized.pt
  gr00t_goal/quantized.pt
  gr00t_long/quantized.pt
  pi05_object/quantized.pt
```

路线差异：

- `groot_w4a8`、`pi05_w4a8_duquant` 属于 runtime W4A8 路线，不需要提前准备 `quantized.pt`。
- `groot_w4a4_gptq`、`pi05_w4a4_gptq` 属于 W4A4 GPTQ pack 路线，必须准备对应 `quantized.pt`。
- QVLA W8 路线会生成校准 JSONL、Hessian proxy、gate/bit allocation 和评测结果；其中 `proxy.pt` 是敏感度分析产物，不是可直接部署的量化模型。
- Pi0.5 需要先用 OpenPI 把官方 JAX/Orbax checkpoint 转成 PyTorch checkpoint。

具体下载与转换步骤见 [checkpoints.md](checkpoints.md)。

## 运行示例

建议每条路线先执行 `--action plan`：

```bash
bash scripts/run_awesome_quant_vla.sh pi05_w4a8_duquant \
  --suite object \
  --gpus 0 \
  --trials 10 \
  --action plan
```

运行 Pi0.5 W4A8：

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

运行 Pi0.5 W4A4 GPTQ：

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

运行 OpenVLA QVLA W8：

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

运行 UniVLA FP16：

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

查看结果：

```bash
bash scripts/run_awesome_quant_vla.sh <profile> \
  --suite <suite> \
  --output-root /path/to/result_dir \
  --action result
```

## 输出目录说明

一次成功评测通常会生成：

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

常见文件含义：

- `merged_summary.json`：机器可读的总成功率和每个任务结果。
- `merged_summary.md`：人工查看用的结果表。
- `logs/run.log`：launcher 主日志。
- `logs/server_shard_*.log`：推理服务日志。
- `logs/eval_shard_*.stdout.log`：LIBERO 评测 stdout。
- `act_stats/`：量化运行时采集的 activation statistics。
- `packdir/`：本次运行的临时 pack staging 目录。
- `proxy/proxy.pt`、`gates/`、`calib/`：OpenVLA/QVLA 路线的分析与分配产物。

## 范围说明

quant-vla 主要面向量化流程整合、LIBERO 评测和工程复现。当前并不是所有路线都会输出一个可直接交给真实机器人或国产加速卡运行的独立量化模型。

不同路线的产物并不完全相同：有些是在推理时动态注入量化行为，有些读取预构建的 `quantized.pt` pack，有些生成 QVLA 的校准、proxy 和 bit allocation 产物。真实机器人部署或非 NVIDIA 硬件部署仍需要模型导出、运行时转换、算子适配、时延验证和控制接口对接。

## 开发检查

提交前建议至少执行：

```bash
bash -n scripts/run_awesome_quant_vla.sh
bash -n scripts/run_groot_benchmark.sh
bash -n scripts/run_pi05_libero_benchmark.sh
bash -n scripts/run_openvla_qvla.sh
bash -n scripts/run_univla_libero.sh

PYTHONPATH=. python -m compileall -q gr00t tools scripts examples
```

新增或修改路线时：

1. 保留上游来源说明，但用户入口尽量统一到 `scripts/run_awesome_quant_vla.sh`。
2. 在文档里写清楚 checkpoint、pack、环境变量和依赖假设。
3. `--action plan` 需要展示最终解析出的路径和量化模式。
4. 先跑小规模 smoke test，再跑完整 LIBERO 评测。
5. 只有在 summary 文件生成且 `--action result` 可读后，才把结果写入验证文档。

## 目录结构

```text
quant-vla
|-- .env.example
|-- docs/
|-- scripts/
|-- tools/
|-- gr00t/
|-- examples/Libero/
|-- atm_alpha_beta_pi05/
|-- third_party/
|   |-- openvla
|   |-- openvla_oft
|   `-- univla
|-- tests/
`-- results/
```

## 来源与致谢

quant-vla 整合并适配了 QuantVLA、Omega-QVLA、QVLA/OpenVLA、OpenDriveLab/UniVLA、OpenPI 和 LIBERO 等项目中的思路与代码路径。使用或再分发相关组件时，请同时核对原项目的许可证与引用要求。
