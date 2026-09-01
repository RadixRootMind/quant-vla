# OpenVLA/QVLA 路线说明

quant-vla 已经把 AutoLab-SAI-SJTU/QVLA 中的 OpenVLA/OpenVLA-OFT 路线作为第三个项目族合并进来。

## 已集成后端

| Profile | 后端 | 量化方式 | 启动脚本 |
| --- | --- | --- | --- |
| `openvla_fp16` | `third_party/openvla` | FP16 | `scripts/run_openvla_qvla.sh` |
| `openvla_qvla_w8` | `third_party/openvla` | QVLA mixed-bit W8 | `scripts/run_openvla_qvla.sh` |
| `openvla_oft_fp16` | `third_party/openvla_oft` | FP16 | `scripts/run_openvla_qvla.sh` |
| `openvla_oft_qvla_w8` | `third_party/openvla_oft` | QVLA mixed-bit W8 | `scripts/run_openvla_qvla.sh` |

## 实现逻辑

- `tools/qvla/build_calib_jsonl.py` 负责生成 LIBERO calibration JSONL。
- `tools/qvla/sensitivity_hessian_proxy.py` 负责计算 QVLA sensitivity proxy。
- `tools/qvla/greedy_bit_allocator.py` 负责按目标平均 bit 分配层 bit。
- `tools/qvla/run_eval.py` 先按原项目方式加载 OpenVLA/OpenVLA-OFT，再注入 QVLA fake-weight 配置并执行 LIBERO 评测。

这条路线尽量复用原 OpenVLA/OpenVLA-OFT 的模型加载和评测逻辑，只把量化控制、输出目录和统一入口接入 quant-vla。

## 当前验证结果

| 路线 | 任务集 | 结果 |
| --- | --- | ---: |
| `openvla_fp16` | `spatial` | 70.0% |
| `openvla_qvla_w8` | `spatial` | 80.0% |
| `openvla_oft_fp16` | `spatial` | 60.0% |
| `openvla_oft_qvla_w8` | `spatial` | 26.0% |

以上是本地工程验证结果，用于确认集成链路可运行。

## 关键兼容修复

- OpenVLA 默认使用 `OPENVLA_ATTN_IMPL=eager`。
- OpenVLA action prediction 在必要时补齐 prompt delimiter token 和对应 attention mask。
- QVLA proxy 路线为 OpenVLA-OFT forward 补充 labels fallback。
- 如需保存视频，需要安装 `imageio[ffmpeg]`；不保存视频时可跳过。

## 最小运行命令

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
  --output-root "$AWESOME_QVLA_ROOT/results/verify/openvla_qvla_w8_spatial" \
  --max-samples 32
```

读取结果：

```bash
bash scripts/run_awesome_quant_vla.sh openvla_qvla_w8 \
  --suite spatial \
  --output-root "$AWESOME_QVLA_ROOT/results/verify/openvla_qvla_w8_spatial" \
  --action result
```

