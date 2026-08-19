#!/usr/bin/env python3
"""
GR00T Memory Calculator - calculate memory usage for different quantization scenarios.
Loads model from HuggingFace to avoid needing local checkpoint.
"""

import re
from pathlib import Path
import torch


def format_bytes(bytes_val):
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def parse_layer_name(param_name):
    """
    Extract layer name from parameter name.
    Example: "backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj.weight"
    Returns: "backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj"
    """
    if param_name.endswith(".weight"):
        return param_name[:-7]
    elif param_name.endswith(".bias"):
        return param_name[:-5]
    return param_name


def match_layer(layer_name, include_regex, exclude_regex):
    """Check if layer matches the given patterns."""
    # Check include
    if include_regex:
        inc = re.compile(include_regex)
        if not inc.search(layer_name):
            return False

    # Check exclude
    if exclude_regex:
        exc = re.compile(exclude_regex)
        if exc.search(layer_name):
            return False

    return True


def calculate_duquant_memory_for_layer(out_features, in_features, has_bias=True,
                                       weight_bits=4, block_size=64,
                                       enable_permute=False, row_rot_mode="restore"):
    """Calculate DuQuant memory for a single layer."""
    weight_params = out_features * in_features
    bias_params = out_features if has_bias else 0

    memory = {}

    # 1. Quantized weights (packed)
    memory["quantized_weights"] = (weight_params * weight_bits) / 8

    # 2. Weight scales (FP16 per output channel)
    memory["weight_scales"] = out_features * 2

    # 3. Input rotation matrices R_in
    n_in_blocks = (in_features + block_size - 1) // block_size
    memory["R_in_blocks"] = n_in_blocks * block_size * block_size * 4

    # 4. Output rotation matrices R_out
    if row_rot_mode in ("restore", "propagate"):
        n_out_blocks = (out_features + block_size - 1) // block_size
        memory["R_out_blocks"] = n_out_blocks * block_size * block_size * 4
    else:
        memory["R_out_blocks"] = 0

    # 5. Permutation indices
    if enable_permute:
        memory["permutation"] = in_features * 4
    else:
        memory["permutation"] = 0

    # 6. Bias
    memory["bias"] = bias_params * 2

    memory["total"] = sum(memory.values())
    return memory


def main():
    print("=" * 100)
    print("GR00T DuQuant Memory Usage Calculator")
    print("=" * 100)
    print()

    # Load model from HuggingFace
    model_name = "youliangtan/gr00t-n1.5-libero-goal-posttrain"
    print(f"Loading model from HuggingFace: {model_name}")
    print("(This may take a few minutes on first run...)")
    print()

    try:
        import sys
        sys.path.insert(0, "/home/jz97/VLM_REPO/Isaac-GR00T")

        from gr00t.model.policy import Gr00tPolicy
        from gr00t.experiment.data_config import load_data_config
        import os

        # Disable quantization for inspection
        os.environ["GR00T_DUQUANT_ENABLE"] = "0"
        os.environ["GR00T_ATM_ENABLE"] = "0"

        data_config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")

        policy = Gr00tPolicy(
            model_path=model_name,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag="new_embodiment",
            denoising_steps=8,
        )

        print("✓ Model loaded successfully")
        print()

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print()
        print("Please ensure:")
        print("  1. You have activated the correct conda environment (libero)")
        print("  2. GR00T dependencies are installed")
        print("  3. You have internet connection for HuggingFace download")
        return

    # Extract layer shapes
    layer_shapes = {}
    for name, param in policy.model.named_parameters():
        if name.endswith(".weight") and param.dim() == 2:
            layer_name = parse_layer_name(name)
            out_features, in_features = param.shape

            # Check if bias exists
            bias_name = name.replace(".weight", ".bias")
            has_bias = any(n == bias_name for n, _ in policy.model.named_parameters())

            layer_shapes[layer_name] = {
                "out_features": out_features,
                "in_features": in_features,
                "has_bias": has_bias,
                "weight_params": out_features * in_features,
                "bias_params": out_features if has_bias else 0,
            }

    print(f"✓ Found {len(layer_shapes)} Linear layers in model")
    print()

    total_params_all = sum(info["weight_params"] + info["bias_params"] for info in layer_shapes.values())
    total_original_bytes_all = total_params_all * 2  # FP16

    # Define quantization scenarios for GR00T
    scenarios = {
        "Full Model (all linears)": {
            "include": r".*",
            "exclude": r"$^",  # match nothing
        },
        "LLM Only": {
            "include": r".*backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
            "exclude": r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head)(?:\.|$)",
        },
        "DiT MLP Only": {
            "include": r".*action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2).*",
            "exclude": r"(?:^|\.)(attn1|norm|ln|layernorm|embed)(?:\.|$)",
        },
        "LLM + DiT MLP (current config)": {
            "include": r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*",
            "exclude": r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)",
        },
        "DiT Attention Only (QKV+O)": {
            "include": r".*action_head\.model\.transformer_blocks\.\d+\.attn1\.(to_q|to_k|to_v|to_out\.0).*",
            "exclude": r"(?:^|\.)(norm|ln|layernorm|embed)(?:\.|$)",
        },
        "LLM + DiT Full (Attn+MLP)": {
            "include": r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))).*",
            "exclude": r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head)(?:\.|$)",
        },
    }

    # DuQuant configuration (matching your settings)
    duquant_config = {
        "weight_bits": 4,
        "block_size": 64,
        "enable_permute": False,  # PERMUTE=0
        "row_rot_mode": "restore",
    }

    print(f"DuQuant Configuration:")
    print(f"  Weight bits: {duquant_config['weight_bits']}")
    print(f"  Block size: {duquant_config['block_size']}")
    print(f"  Permutation: {duquant_config['enable_permute']}")
    print(f"  Row rotation: {duquant_config['row_rot_mode']}")
    print()
    print("=" * 100)
    print()

    # Calculate for each scenario
    for scenario_name, scenario_config in scenarios.items():
        print(f"📊 Scenario: {scenario_name}")
        print("-" * 100)

        # Filter layers
        matched_layers = {}
        for layer_name, info in layer_shapes.items():
            if match_layer(layer_name, scenario_config["include"], scenario_config["exclude"]):
                matched_layers[layer_name] = info

        num_layers = len(matched_layers)
        total_params = sum(info["weight_params"] + info["bias_params"] for info in matched_layers.values())

        print(f"  Matched layers: {num_layers}")
        print(f"  Total parameters: {total_params:,}")
        print()

        if num_layers == 0:
            print("  ⚠️  No layers matched!")
            print()
            print("=" * 100)
            print()
            continue

        # Calculate original memory (FP16: 2 bytes per param)
        subset_original_bytes = total_params * 2
        print(f"  Original Memory (subset FP16):")
        print(f"    Total: {format_bytes(subset_original_bytes)} ({subset_original_bytes:,} bytes)")
        print()

        # Calculate DuQuant memory
        duquant_total = {
            "quantized_weights": 0,
            "weight_scales": 0,
            "R_in_blocks": 0,
            "R_out_blocks": 0,
            "permutation": 0,
            "bias": 0,
        }

        for layer_name, info in matched_layers.items():
            layer_memory = calculate_duquant_memory_for_layer(
                info["out_features"],
                info["in_features"],
                info["has_bias"],
                **duquant_config
            )
            for key in duquant_total:
                duquant_total[key] += layer_memory[key]

        subset_quant_bytes = sum(duquant_total.values())

        print(f"  DuQuant Memory (subset W{duquant_config['weight_bits']}):")
        print(f"    Quantized weights: {format_bytes(duquant_total['quantized_weights'])}")
        print(f"    Weight scales:     {format_bytes(duquant_total['weight_scales'])}")
        print(f"    R_in blocks:       {format_bytes(duquant_total['R_in_blocks'])}")
        print(f"    R_out blocks:      {format_bytes(duquant_total['R_out_blocks'])}")
        print(f"    Permutation:       {format_bytes(duquant_total['permutation'])}")
        print(f"    Bias:              {format_bytes(duquant_total['bias'])}")
        print(f"    {'─' * 40}")
        print(f"    Total:             {format_bytes(subset_quant_bytes)} ({subset_quant_bytes:,} bytes)")
        print()

        subset_compression = subset_original_bytes / subset_quant_bytes if subset_quant_bytes > 0 else 0
        remaining_bytes = total_original_bytes_all - subset_original_bytes
        total_quant_bytes = remaining_bytes + subset_quant_bytes

        # Calculate savings relative to full model
        savings_bytes = total_original_bytes_all - total_quant_bytes
        savings_ratio = (savings_bytes / total_original_bytes_all) * 100 if total_original_bytes_all > 0 else 0
        compression_ratio = total_original_bytes_all / total_quant_bytes if total_quant_bytes > 0 else 0

        print(f"  💾 Memory Savings:")
        print(f"    Subset compression: {subset_compression:.2f}x ({format_bytes(subset_original_bytes)} -> {format_bytes(subset_quant_bytes)})")
        print(f"    Subset savings:     {format_bytes(subset_original_bytes - subset_quant_bytes)} ({((subset_original_bytes - subset_quant_bytes) / subset_original_bytes * 100) if subset_original_bytes else 0:.2f}%)")
        print()
        print(f"    Full model (FP16):        {format_bytes(total_original_bytes_all)}")
        print(f"    After quantizing subset:  {format_bytes(total_quant_bytes)}")
        print(f"    Overall savings:          {format_bytes(savings_bytes)} ({savings_ratio:.2f}%)")
        print(f"    Overall compression:      {compression_ratio:.2f}x")
        print()

        # Show some example layers
        if num_layers > 0 and num_layers <= 10:
            print(f"  Matched layers:")
            for i, (name, info) in enumerate(matched_layers.items()):
                short_name = ".".join(name.split(".")[-5:])  # Last 5 parts
                print(f"    {i+1:2d}. {short_name}")
                print(f"        Shape: [{info['out_features']}, {info['in_features']}], Params: {info['weight_params'] + info['bias_params']:,}")
        elif num_layers > 10:
            print(f"  Example layers (first 10):")
            for i, (name, info) in enumerate(list(matched_layers.items())[:10]):
                short_name = ".".join(name.split(".")[-5:])
                print(f"    {i+1:2d}. {short_name}")
                print(f"        Shape: [{info['out_features']}, {info['in_features']}], Params: {info['weight_params'] + info['bias_params']:,}")
            print(f"    ... ({num_layers - 10} more layers)")

        print()
        print("=" * 100)
        print()

    print("✓ Memory calculation complete!")


if __name__ == "__main__":
    main()
