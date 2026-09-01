# StarVLA 接入说明

本仓库将 StarVLA 作为新的 VLA baseline 路线接入到 `third_party/starvla`。当前阶段优先支持 StarVLA 在 LIBERO 上的 FP16/BF16 评测链路，暂不把 StarVLA 的 W4A8/W4A4 量化路线标记为已验证，因为 StarVLA 使用 Qwen-VL / World Model backbone 以及多种 action head，需要单独做量化适配。

## 已接入 Profile

| Profile | 用途 | 状态 |
| --- | --- | --- |
| `starvla_oft_fp16` | StarVLA-OFT LIBERO FP16/BF16 评测 | 已接入并完成 LIBERO Spatial 每任务 10 次验证，结果 99.0% |
| `starvla_gr00t_fp16` | StarVLA-GR00T LIBERO 评测 | 已接入入口，需要对应 checkpoint |
| `starvla_pi_fp16` | StarVLA-PI LIBERO 评测 | 已接入入口，需要对应 checkpoint |
| `starvla_fast_fp16` | StarVLA-FAST LIBERO 评测 | 已接入入口，需要对应 checkpoint |

## 已验证结果

StarVLA-OFT 路线已经完成一次完整 LIBERO Spatial 验证：

| Profile | Suite | 轮次 | Episodes | 结果 |
| --- | --- | ---: | ---: | ---: |
| `starvla_oft_fp16` | LIBERO Spatial | 每任务 10 次 | 100 | 99.0% |

评测结束时出现的 `EGL_NOT_INITIALIZED` 属于 robosuite/MuJoCo 清理 EGL 上下文时的常见提示；只要 `merged_summary.json` 已生成且 `--action result` 可读，就不影响结果有效性。

## 默认 Checkpoint 路径

```bash
$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
```

下载示例：

```bash
mkdir -p "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1"
huggingface-cli download StarVLA/Qwen3-VL-OFT-LIBERO-4in1 \
  --local-dir "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-OFT-LIBERO-4in1"
```

StarVLA policy checkpoint 还会从上游默认相对路径 `playground/Pretrained_models/Qwen3-VL-4B-Instruct` 加载 Qwen3-VL 基座模型。建议把大模型本体放在 `$CHECKPOINTS_ROOT` 下，再软链接到 StarVLA 期望的位置：

```bash
mkdir -p "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct"
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
  --local-dir "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct"

mkdir -p "$AWESOME_QVLA_ROOT/third_party/starvla/playground/Pretrained_models"
ln -sfn "$CHECKPOINTS_ROOT/starvla/Qwen3-VL-4B-Instruct" \
  "$AWESOME_QVLA_ROOT/third_party/starvla/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
```

## 环境建议

StarVLA 依赖的 Qwen / Transformers / DeepSpeed 栈较新，建议使用独立环境，不要和 GR00T/Pi0.5 或 OpenVLA 环境混用：

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

国内网络可以先设置 Hugging Face 镜像和 pip 镜像。

## 运行示例

先看路径计划：

```bash
bash scripts/run_awesome_quant_vla.sh starvla_oft_fp16 \
  --suite spatial \
  --gpus 0 \
  --trials 1 \
  --max-tasks 1 \
  --action plan
```

小规模验证：

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

完整 LIBERO Spatial 评测：

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

查看结果：

```bash
bash scripts/run_awesome_quant_vla.sh starvla_oft_fp16 \
  --suite spatial \
  --output-root "$AWESOME_QVLA_ROOT/results/awesome_quant_vla/starvla_oft_fp16_spatial_final" \
  --action result
```

## 输出目录

```text
results/.../
  merged_summary.json
  merged_summary.md
  logs/server.log
  logs/eval_stdout.log
  rollouts/              # 只有 --save-video True 时才会保存视频
```

默认不保存视频，避免 headless 环境中的 ffmpeg/imageio 问题，也能提升验证速度。如需保存视频，增加 `--save-video True`。
