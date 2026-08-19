#!/bin/bash
# GR00T DuQuant W4A8 Full Quantization (LLM + DiT ALL Linear Layers)
# Quantize both LLM (Eagle VLM) and DiT (Action Head) linear layers with W4A8
#
# Model Structure:
# ┌────────────────────────────────────────────────────────────┐
# │ GR00T N1.5 Model                                           │
# │                                                            │
# │ ├── backbone (Eagle2.5 VLM)                                │
# │ │   ├── vision_tower (RADIO/SigLIP) ← NOT QUANTIZED        │
# │ │   └── language_model (Qwen2.5 LLM) ← QUANTIZE ALL        │
# │ │                                                          │
# │ └── action_head (DiT-based Flow Matching)                  │
# │     ├── DiT Attention (q/k/v/o_proj) ← QUANTIZE ALL       │
# │     └── DiT MLP (gate/up/down_proj) ← QUANTIZE ALL        │
# └────────────────────────────────────────────────────────────┘
#
# Quantization target (VERIFIED):
# - LLM: 84 layers (12 layers × 7 projections: q/k/v/o + gate/up/down)
# - DiT: 96 layers (16 blocks × 6 projections: 4 attn + 2 ffn)
# - Total: 180 layers

set -e

cd /home/jz97/VLM_REPO/groot_test/QuantVLA_GR00T

# Task configuration
TASK_SUITE="${1:-libero_10}"
if [ -n "$2" ]; then
    MODEL_PATH="$2"
else
    case "$TASK_SUITE" in
        libero_spatial)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-spatial-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        libero_goal)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-goal-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfigMeanStd"
            ;;
        libero_object)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-object-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        libero_90)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-90-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        libero_10)
            MODEL_PATH="youliangtan/gr00t-n1.5-libero-long-posttrain"
            DATA_CONFIG="examples.Libero.custom_data_config:LiberoDataConfig"
            ;;
        *)
            echo "Unknown task suite: $TASK_SUITE"
            echo "Valid options: libero_spatial, libero_goal, libero_object, libero_90, libero_10"
            exit 1
            ;;
    esac
fi
DATA_CONFIG="${DATA_CONFIG:-examples.Libero.custom_data_config:LiberoDataConfig}"

echo "========================================"
echo "GR00T DuQuant W4A8 Full Quantization"
echo "LLM + DiT ALL Linear Layers"
echo "========================================"
echo "Task suite: $TASK_SUITE"
echo "Model: $MODEL_PATH"
echo ""

# ============================================
# DuQuant W4A8 Full Configuration
# ============================================
export GR00T_DUQUANT_DEBUG=1

# SCOPE: Empty = search entire model
export GR00T_DUQUANT_SCOPE=""

# INCLUDE: Match both LLM and DiT ALL linear layers
# - LLM: backbone.eagle_model.language_model.*.(q/k/v/o/gate/up/down_proj)
# - DiT: action_head.model.transformer_blocks.*.attn1.(to_q|to_k|to_v|to_out) + ff.net.*
# export GR00T_DUQUANT_INCLUDE='.*((backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))|(action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k)|ff\.net\.\d+))).*'

# # EXCLUDE: Vision tower, embeddings, timestep encoder, state encoder, norm layers (but allow norm1.linear in DiT)
# export GR00T_DUQUANT_EXCLUDE='(?:^|\.)(vision|radio|^norm|^ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|pos_embed|vl_self_attention|vlln|future_tokens)(?:\.|$)'

export GR00T_DUQUANT_INCLUDE='.*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*'
export GR00T_DUQUANT_EXCLUDE='(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)'


# Quantization parameters (optimized for full quantization)
export GR00T_DUQUANT_WBITS_DEFAULT=4
export GR00T_DUQUANT_ABITS=8
export GR00T_DUQUANT_BLOCK=64
export GR00T_DUQUANT_PERMUTE=0           # Enable input permutation
export GR00T_DUQUANT_ROW_ROT=restore     # Output rotation with restore
export GR00T_DUQUANT_ACT_PCT=99.9
export GR00T_DUQUANT_CALIB_STEPS=32      # Conservative calibration
export GR00T_DUQUANT_LS=0.15              # Increased smoothing for stability

# Pack directory for caching quantization metadata
export GR00T_DUQUANT_PACKDIR="/home/jz97/VLM_REPO/groot_test/QuantVLA_GR00T/duquant_packed_full_llm_dit_mlp_w4a8_b64c32ls015_long_0"


# ATM configuration (optional)
# export GR00T_ATM_ALPHA_PATH=/home/jz97/VLM_REPO/Isaac-GR00T/atm_alpha_dit_mlp_permute0_goal_new.json
# if [[ -n "${GR00T_ATM_ALPHA_PATH:-}" && -z "${GR00T_ATM_ENABLE:-}" ]]; then
#     export GR00T_ATM_ENABLE=1
# fi
# export GR00T_ATM_SCOPE=${GR00T_ATM_SCOPE:-dit}


export GR00T_ATM_ALPHA_PATH=/home/jz97/VLM_REPO/groot_test/QuantVLA_GR00T/atm_alpha_beta_long.json
export GR00T_ATM_ENABLE=1
export GR00T_ATM_SCOPE=${GR00T_ATM_SCOPE:-dit}

export GR00T_OHB_ENABLE=1
export GR00T_OHB_FALLBACK=1.0      # JSON 缺层时使用
export GR00T_OHB_SCOPE=${GR00T_OHB_SCOPE:-dit}

# Disable torch.compile for compatibility
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Disable CUDA graphs to avoid memory issues
export TORCH_CUDA_GRAPH_DISABLE=1
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1

# Denoising steps for quantization (CRITICAL: increase for full quantization)
# export GR00T_DENOISING_STEPS=${GR00T_DENOISING_STEPS:-20}

echo "DuQuant Config (Full LLM + DiT W4A8):"
echo "  SCOPE: $GR00T_DUQUANT_SCOPE"
echo "  INCLUDE: $GR00T_DUQUANT_INCLUDE"
echo "  EXCLUDE: $GR00T_DUQUANT_EXCLUDE"
echo "  WBITS=$GR00T_DUQUANT_WBITS_DEFAULT"
echo "  ABITS=$GR00T_DUQUANT_ABITS"
echo "  BLOCK=$GR00T_DUQUANT_BLOCK"
echo "  PERMUTE=$GR00T_DUQUANT_PERMUTE"
echo "  ROW_ROT=$GR00T_DUQUANT_ROW_ROT"
echo "  ACT_PCT=$GR00T_DUQUANT_ACT_PCT"
echo "  CALIB_STEPS=$GR00T_DUQUANT_CALIB_STEPS"
echo "  LS=$GR00T_DUQUANT_LS"
echo "  PACKDIR=$GR00T_DUQUANT_PACKDIR"
echo "  DENOISING_STEPS=$GR00T_DENOISING_STEPS (CRITICAL for full quantization)"
echo ""
echo "⚡ QUANTIZATION TARGET:"
echo "  ✅ LLM (Eagle VLM) ALL linear layers (~84 layers)"
echo "  ✅ DiT (Action Head) ALL attention + MLP layers (~186 layers)"
echo "  ❌ Vision Tower (RADIO/SigLIP) - NOT quantized"
echo "  ❌ Embeddings & Encoders - NOT quantized"
echo "  ❌ Normalization layers - NOT quantized"
echo ""
echo "⚡ FEATURES:"
echo "  ✅ W4A8 fake quantization"
echo "  ✅ Input permutation enabled"
echo "  ✅ Row rotation with output restoration"
echo "  ✅ Increased denoising steps (${GR00T_DENOISING_STEPS}) to compensate quantization noise"
echo "  ❌ torch.compile DISABLED (for stability)"
echo ""
echo "⚠️  WARNING: Full quantization may cause accuracy drop!"
echo "    Recommended to increase GR00T_DENOISING_STEPS to 24-32 if accuracy drops"
echo ""
echo "========================================"
echo ""

# First run dry-run to show which layers will be quantized
echo "🔍 DRY RUN: Scanning layers to quantize..."
echo ""
export GR00T_DUQUANT_DRYRUN=1
export GR00T_MODEL_PATH="$MODEL_PATH"
export GR00T_DATA_CONFIG="$DATA_CONFIG"

python - <<'PY'
import os
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import load_data_config

model_path = os.environ.get("GR00T_MODEL_PATH", "youliangtan/gr00t-n1.5-libero-spatial-posttrain")
data_config_path = os.environ.get("GR00T_DATA_CONFIG", "examples.Libero.custom_data_config:LiberoDataConfig")

print("Loading model for DuQuant dry-run...")
cfg = load_data_config(data_config_path)
policy = Gr00tPolicy(
    model_path=model_path,
    modality_config=cfg.modality_config(),
    modality_transform=cfg.transform(),
    embodiment_tag="new_embodiment",
    denoising_steps=8,
)
print("\n✅ DuQuant dry-run complete!\n")
PY

echo ""
echo "========================================"
echo "Dry run complete. Review the layers above."
echo ""
echo "Expected layers:"
echo "  - LLM: ~84 layers (12 layers × 7 linear each)"
echo "  - DiT: ~186 layers (varies by model)"
echo "  - Total: ~270 layers"
echo ""
echo "Press Enter to continue with actual quantization, or Ctrl+C to cancel..."
read -r

# Clear dry-run flag
unset GR00T_DUQUANT_DRYRUN
unset GR00T_MODEL_PATH
unset GR00T_DATA_CONFIG

echo ""
echo "🚀 Starting fully quantized inference server..."
echo ""
echo "⚠️  IMPORTANT NOTES:"
echo "  1. First startup will be SLOW (~5-10 min) due to quantization preprocessing"
echo "  2. Subsequent runs will be faster using cached pack directory"
echo "  3. Monitor GPU memory - full quantization uses ~40% less memory than FP16"
echo "  4. If accuracy drops significantly, increase GR00T_DENOISING_STEPS to 24-32"
echo ""

# Start the quantized inference server
# This will apply DuQuant to the model during loading
./run_inference_server.sh "$TASK_SUITE"
