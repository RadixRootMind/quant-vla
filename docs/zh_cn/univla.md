# UniVLA 说明

quant-vla 已将 OpenDriveLab/UniVLA 作为第四类后端接入。

当前集成的是 UniVLA 的 LIBERO FP16 评测路线。UniVLA 源码中也包含 QVLA 相关脚本，但 quant-vla 暂时不把 UniVLA 量化路线标记为已验证功能，因为 UniVLA 评测除了 latent action 生成，还依赖独立的 action decoder。

## 已接入文件

| 组件 | 路径 |
| --- | --- |
| UniVLA 后端源码 | `third_party/univla` |
| 独立 runner | `scripts/run_univla_libero.sh` |
| 统一入口 profile | `univla_fp16` |

## 需要的 Checkpoint

LIBERO 路线需要：

- Hugging Face 格式的 UniVLA checkpoint，通常来自 `qwbu/univla-7b-224-sft-libero` 的某个 suite 子目录。
- action decoder 文件，通常是 checkpoint 子目录里的 `action_decoder.pt`。

推荐本地目录：

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

只下载 spatial checkpoint：

```bash
mkdir -p "$CHECKPOINTS_ROOT"

hf download qwbu/univla-7b-224-sft-libero \
  --include "univla-libero-spatial/*" \
  --local-dir "$CHECKPOINTS_ROOT/univla-7b-224-sft-libero"
```

如果要评测其他 suite，把 include 前缀换成：

```bash
--include "univla-libero-object/*"
--include "univla-libero-goal/*"
--include "univla-libero-10/*"
```

运行时也可以显式指定：

```bash
--univla-checkpoint /path/to/univla-7b-224-sft-libero/univla-libero-spatial
--univla-action-decoder /path/to/univla-7b-224-sft-libero/univla-libero-spatial/action_decoder.pt
```

## 环境

建议复用 OpenVLA/QVLA 环境：PyTorch 2.2.0 CUDA 12.1、transformers 4.40.1、tokenizers 0.19.1、peft 0.11.1、timm 0.9.10、TensorFlow 2.15.0、tensorflow-datasets 4.9.3、protobuf 3.20.3、numpy 1.26.4、opencv-python-headless 4.9.0.80。

在同一个环境里安装 UniVLA 额外依赖：

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

如果镜像源找不到 `braceexpand`，可以换官方 PyPI 或 conda-forge：

```bash
python -m pip install --no-user braceexpand -i https://pypi.org/simple
# 或
conda install -y -c conda-forge braceexpand
```

默认使用 eager attention：

```bash
export UNIVLA_ATTN_IMPL=eager
```

## 运行

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

读取结果：

```bash
bash scripts/run_awesome_quant_vla.sh univla_fp16 \
  --suite spatial \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/univla_fp16_spatial" \
  --action result
```

## 当前状态

- `univla_fp16`：已接入并完成 LIBERO Spatial 每任务 10 次验证，结果 `96.0%`。
- `univla_qvla_w8`：暂不作为已验证 profile 暴露。
