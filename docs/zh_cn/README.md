# quant-vla 中文说明

quant-vla 是一个面向 Vision-Language-Action（VLA）模型量化、LIBERO 评测与硬件适配准备的统一研究与工程项目。

本项目把 QuantVLA、Omega-QVLA、QVLA/OpenVLA、OpenVLA-OFT、OpenDriveLab/UniVLA、StarVLA、GR00T-N1.5、Pi0.5/OpenPI 等相关路线整合到同一个仓库中，并提供统一的启动入口、checkpoint 约定、输出结构和验证说明。

> quant-vla 仍处于持续开发阶段。大型 checkpoint、量化 pack、数据集和评测输出不会提交到 git。

## 为什么要合一？

VLA 模型不同于普通 LLM 或 VLM。它把视觉感知、语言理解、机器人状态和动作生成放在同一个具身策略中，最终输出的是可执行的机器人动作。因此，低比特量化带来的一个小数值误差，可能会从视觉编码传播到语义理解、动作解码、轨迹生成、接触动力学和闭环控制，最终表现为任务失败。

这意味着 VLA 量化不能只看模型大小、重构误差或 token 预测精度。一个可用的 VLA 量化系统，还需要关注动作保真度、时间稳定性、语义动作对齐，以及 LIBERO 或真实机器人任务中的闭环成功率。

当前 VLA 量化相关工作分散在不同项目中，模型族、依赖环境、checkpoint 结构、量化产物和评测脚本都不一致。quant-vla 的目标不是简单堆叠代码，而是把这些路线整理成可复现、可比较、可扩展的统一工程入口。

## 如果不合一会有什么问题？

| 分散点 | 实际影响 |
| --- | --- |
| 每个上游项目都有自己的启动脚本 | 用户需要反复修改路径、端口、任务参数和运行逻辑。 |
| 依赖环境相互冲突 | GR00T/Pi0.5、OpenVLA、UniVLA、StarVLA 往往需要不同 Python、PyTorch 和 Transformers 版本。 |
| checkpoint 和 pack 目录不统一 | 一台机器能跑通的命令，很难在另一台机器稳定复现。 |
| 量化产物语义不清 | runtime quantization、`quantized.pt`、proxy、gates、calib、act_stats 容易被混为最终模型。 |
| benchmark 设置分散 | LIBERO suite、task id、trials、init offset、视频保存和日志位置都会影响结果。 |
| 缺少硬件适配边界 | 迁移到 DCU、NPU、IPU 等平台时，模型、算子、量化产物和 runtime 边界不清晰。 |

quant-vla 通过统一 profile、launcher、路径约定、日志、summary 和文档，降低复现和继续开发的成本。

## 设计视角

本项目按照 VLA 推理链路来理解量化：

```text
观测图像 + 语言指令
        |
        v
视觉编码器 -> LLM/VLM 主干 -> 动作头 / 动作解码器
        |              |              |
        |              |              v
        |              |        可执行机器人动作
        |              |
        v              v
校准、敏感度分析、旋转、运行时量化、GPTQ pack、混合比特分配
```

因此，本项目更关注行为保持，而不只是张量压缩。每条路线都会尽量明确模型来源、量化方式、校准数据、评测 suite 和输出产物含义。

## 项目提供什么？

| 层级 | 作用 |
| --- | --- |
| 统一路线入口 | 使用同一个脚本入口管理 GR00T、Pi0.5/OpenPI、OpenVLA、OpenVLA-OFT、UniVLA 和 StarVLA。 |
| 量化路线整合 | 覆盖 W4A8、W4A4、GPTQ、RTN、DuQuant、QVLA mixed-bit W8 及相关校准流程。 |
| LIBERO 评测 | 统一 task suite、rollout、日志和 merged summary 输出。 |
| 产物规范化 | 统一 logs、summaries、rollouts、act_stats、packs、proxy、gates、calib 等输出位置。 |
| 可复现说明 | 为主要路线保留环境、checkpoint、常见修复和验证结果。 |
| 硬件适配准备 | 明确模型 checkpoint、量化路线、runtime 依赖和评测结果之间的边界。 |

## 主路线矩阵

| 模型族 | Profile | 量化 / 评测状态 |
| --- | --- | --- |
| GR00T-N1.5 | `groot_fp16`, `groot_w4a8`, `groot_w4a4_gptq`, `groot_w4a4_duquant`, `groot_w4a4_rtn` | 支持 FP16、runtime W4A8、W4A4 GPTQ pack、W4A4 DuQuant 和 W4A4 RTN。 |
| Pi0.5/OpenPI | `pi05_fp16`, `pi05_w4a8_duquant`, `pi05_w4a4_gptq`, `pi05_w4a4_rtn` | 基于 OpenPI service 评测，支持 FP16、runtime W4A8、W4A4 GPTQ pack 和 W4A4 RTN。 |
| OpenVLA | `openvla_fp16`, `openvla_qvla_w8` | 支持 FP16 baseline 和 QVLA mixed-bit W8 评测。 |
| OpenVLA-OFT | `openvla_oft_fp16`, `openvla_oft_qvla_w8` | 支持 FP16 baseline 和 QVLA mixed-bit W8 评测。 |
| UniVLA | `univla_fp16` | 支持带外部 action decoder 的 FP16 评测；量化路线暂不作为已验证路线发布。 |
| StarVLA | `starvla_oft_fp16`, `starvla_gr00t_fp16`, `starvla_pi_fp16`, `starvla_fast_fp16` | StarVLA-OFT FP16/BF16 已验证；其它 FP16/BF16 路线需要对应 checkpoint；量化路线后续需要适配 Qwen-VL/action head。 |

## 已验证结果

以下结果来自合并后的本地工程验证，用于说明路线可以在本仓库中端到端跑通，不作为论文官方 benchmark 数字。

| Profile | 模型 | Suite | 路线 | 结果 |
| --- | --- | --- | --- | ---: |
| `groot_w4a8` | GR00T-N1.5 | LIBERO Object | Runtime DuQuant/ATM/OHB | 82.0% |
| `pi05_w4a8_duquant` | Pi0.5/OpenPI | LIBERO Object | Runtime DuQuant/ATM/OHB | 99.0% |
| `pi05_w4a4_gptq` | Pi0.5/OpenPI | LIBERO Object | W4A4 GPTQ/SVD-Hadamard pack | 98.0% |
| `openvla_fp16` | OpenVLA | LIBERO Spatial | FP16 baseline | 70.0% |
| `openvla_qvla_w8` | OpenVLA | LIBERO Spatial | QVLA mixed-bit W8 | 80.0% |
| `openvla_oft_fp16` | OpenVLA-OFT | LIBERO Spatial | FP16 baseline | 60.0% |
| `openvla_oft_qvla_w8` | OpenVLA-OFT | LIBERO Spatial | QVLA mixed-bit W8 | 26.0% |
| `univla_fp16` | UniVLA | LIBERO Spatial | FP16 + action decoder | 96.0% |
| `starvla_oft_fp16` | StarVLA-OFT | LIBERO Spatial | FP16/BF16 policy-server 评测 | 99.0% |

## 验证环境

以上验证结果来自以下本地工作站配置：

| 组件 | 配置 |
| --- | --- |
| GPU | NVIDIA A100 40GB |
| CPU | Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz |
| 系统内存 | 96 GB |
| 存储 | 200 GB SSD |

## 快速开始

详细安装、checkpoint 准备和分路线验证命令放在 `docs/` 目录中。README 只保留最小入口。

```bash
git clone https://github.com/RadixRootMind/quant-vla.git quant-vla
cd quant-vla

cp .env.example .env.local
source .env.local

bash scripts/run_awesome_quant_vla.sh groot_w4a8 \
  --suite object \
  --gpus 0 \
  --trials 1 \
  --action plan
```

推荐阅读顺序：

- [英文安装说明](../installation.md)
- [Checkpoint 与量化 Pack](checkpoints.md)
- [验证指南](verification.md)
- [UniVLA 说明](univla.md)
- [StarVLA 说明](starvla.md)
- [English README](../../README.md)

## Checkpoint 与产物

checkpoint 和量化 pack 不进入 git。通常把本地模型放到 `$CHECKPOINTS_ROOT`，把运行输出和下载的量化 pack 放到 `results/`。

主要差异：

- `groot_w4a8`、`pi05_w4a8_duquant` 属于 runtime W4A8 路线，不需要提前准备 `quantized.pt`。
- `groot_w4a4_gptq`、`pi05_w4a4_gptq` 属于 W4A4 GPTQ pack 路线，需要已有或自行构建的 `quantized.pt`。
- QVLA W8 路线会生成 calibration JSONL、Hessian proxy、gate/bit allocation 和评测输出；`proxy.pt` 是分析产物，不是可直接部署的量化模型。
- Pi0.5 路线需要先把官方 OpenPI JAX/Orbax checkpoint 转成 PyTorch checkpoint。
- UniVLA 和 StarVLA 路线需要各自对应的模型 checkpoint，部分路线还需要 action decoder 资产。

具体下载与转换命令见 [checkpoints.md](checkpoints.md)。

## 输出含义

一次成功评测通常生成：

```text
results/<run_name>/
|-- merged_summary.json
|-- merged_summary.md
|-- logs/
|-- summaries/
|-- rollouts/
|-- act_stats/
|-- packdir/
`-- proxy/ 或 gates/ 或 calib/，取决于具体路线
```

主要评测依据是 `merged_summary.json` 和 `merged_summary.md`。`act_stats/`、`packdir/`、`proxy/`、`gates/`、`calib/` 属于路线相关中间产物，不应直接等同于最终可部署模型。

## 项目结构

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
|   |-- univla
|   `-- starvla
|-- tests/
`-- results/
```

## 范围与边界

quant-vla 主要面向 VLA 模型的研究复现、后训练量化评测和工程整合。

本仓库不内置大型 checkpoint 或数据集，也不声明每条路线都会输出一个可以直接交给真实机器人或非 NVIDIA 加速卡运行的独立量化模型。部分路线是在推理时注入量化行为，部分路线读取预构建 GPTQ pack，部分路线生成 calibration、proxy 或 bit allocation 产物。

真实机器人部署仍需要模型导出、runtime 转换、硬件算子支持、时延验证和机器人控制栈对接。

## 社群

加入 RadixRootMind 中国区开发者微信群：

<p align="center">
  <img src="../../assets/radixrootmind-wechat-group.png" alt="RadixRootMind 中国区开发者微信群二维码" width="360">
</p>

## 路线图

- 统一 route-level benchmark manifest 和 scorecard。
- 为所有公开 profile 增加更强的 smoke test。
- 改进离线环境下的 checkpoint 和量化 pack 发现机制。
- 补充 DCU、NPU、IPU 等平台的硬件适配说明。
- 在验证完成后，逐步发布 UniVLA 和 StarVLA 的量化路线。

## 来源与致谢

quant-vla 整合并适配了 QuantVLA、Omega-QVLA、QVLA/OpenVLA、OpenVLA-OFT、OpenDriveLab/UniVLA、StarVLA、OpenPI、GR00T 和 LIBERO 等项目中的思路与代码路线。使用或再分发相关组件时，请同时核对原项目的许可证与引用要求。
