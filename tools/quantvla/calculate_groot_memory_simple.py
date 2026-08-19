#!/usr/bin/env python3
"""
GR00T Memory Calculator - Complete model memory breakdown.

Shows: Total model size, quantized total size, per-module sizes, per-module quantized sizes.
"""

import re


def format_bytes(bytes_val):
    """Format bytes to human-readable string."""
    gb = bytes_val / (1024**3)
    mb = bytes_val / (1024**2)
    if gb >= 1:
        return f"{gb:.2f} GB ({mb:.0f} MB)"
    return f"{mb:.2f} MB"


def calculate_duquant_memory_for_layer(out_features, in_features, has_bias=True,
                                       weight_bits=4, block_size=64,
                                       enable_permute=False, row_rot_mode="restore"):
    """Calculate DuQuant memory for a single layer."""
    weight_params = out_features * in_features
    bias_params = out_features if has_bias else 0

    memory = {}
    memory["quantized_weights"] = (weight_params * weight_bits) / 8
    memory["weight_scales"] = out_features * 2

    n_in_blocks = (in_features + block_size - 1) // block_size
    memory["R_in_blocks"] = n_in_blocks * block_size * block_size * 4

    if row_rot_mode in ("restore", "propagate"):
        n_out_blocks = (out_features + block_size - 1) // block_size
        memory["R_out_blocks"] = n_out_blocks * block_size * block_size * 4
    else:
        memory["R_out_blocks"] = 0

    memory["permutation"] = in_features * 4 if enable_permute else 0
    memory["bias"] = bias_params * 2
    memory["total"] = sum(memory.values())
    return memory


def get_groot_architecture():
    """
    GR00T N1.5 architecture - verified from actual pack files.

    Total ~5B params:
    - Vision: ~2.5B (RADIO-2B + SigLIP)
    - LLM: ~2.0B (Qwen2.5-2B, 12 layers)
    - DiT: ~0.5B (16 blocks)
    """
    layers = []

    # LLM: 12 layers, hidden=2048, intermediate=6144
    # K/V use GQA with reduced dimensions (1024)
    num_llm_layers = 12

    for i in range(num_llm_layers):
        layers.extend([
            {"name": f"llm.layers.{i}.self_attn.q_proj", "out": 2048, "in": 2048, "bias": False, "cat": "LLM"},
            {"name": f"llm.layers.{i}.self_attn.k_proj", "out": 1024, "in": 2048, "bias": False, "cat": "LLM"},
            {"name": f"llm.layers.{i}.self_attn.v_proj", "out": 2048, "in": 2048, "bias": False, "cat": "LLM"},
            {"name": f"llm.layers.{i}.self_attn.o_proj", "out": 2048, "in": 2048, "bias": False, "cat": "LLM"},
            {"name": f"llm.layers.{i}.mlp.gate_proj", "out": 6144, "in": 2048, "bias": False, "cat": "LLM"},
            {"name": f"llm.layers.{i}.mlp.up_proj", "out": 6144, "in": 2048, "bias": False, "cat": "LLM"},
            {"name": f"llm.layers.{i}.mlp.down_proj", "out": 2048, "in": 6144, "bias": False, "cat": "LLM"},
        ])

    # DiT: 16 blocks, hidden=1536, intermediate=6144
    num_dit_blocks = 16

    for i in range(num_dit_blocks):
        layers.extend([
            {"name": f"dit.blocks.{i}.attn1.to_q", "out": 1536, "in": 1536, "bias": True, "cat": "DiT"},
            {"name": f"dit.blocks.{i}.attn1.to_k", "out": 1536, "in": 1536, "bias": True, "cat": "DiT"},
            {"name": f"dit.blocks.{i}.attn1.to_v", "out": 1536, "in": 1536, "bias": True, "cat": "DiT"},
            {"name": f"dit.blocks.{i}.attn1.to_out.0", "out": 1536, "in": 1536, "bias": True, "cat": "DiT"},
            {"name": f"dit.blocks.{i}.ff.net.0.proj", "out": 6144, "in": 1536, "bias": True, "cat": "DiT"},
            {"name": f"dit.blocks.{i}.ff.net.2", "out": 1536, "in": 6144, "bias": True, "cat": "DiT"},
        ])

    return layers


def match_layer(layer_name, include_regex, exclude_regex):
    if include_regex and not re.compile(include_regex).search(layer_name):
        return False
    if exclude_regex and re.compile(exclude_regex).search(layer_name):
        return False
    return True


def main():
    print("=" * 100)
    print("GR00T DuQuant Memory Calculator - Complete Model Memory Breakdown")
    print("=" * 100)
    print()

    # Model architecture
    VISION_PARAMS = 2.5e9  # RADIO-2B + SigLIP
    VISION_FP16_BYTES = VISION_PARAMS * 2

    all_layers = get_groot_architecture()

    # Calculate LLM and DiT params
    llm_layers = [l for l in all_layers if l["cat"] == "LLM"]
    dit_layers = [l for l in all_layers if l["cat"] == "DiT"]

    llm_params = sum((l["out"] * l["in"]) + (l["out"] if l["bias"] else 0) for l in llm_layers)
    dit_params = sum((l["out"] * l["in"]) + (l["out"] if l["bias"] else 0) for l in dit_layers)

    llm_fp16_bytes = llm_params * 2
    dit_fp16_bytes = dit_params * 2

    total_params = VISION_PARAMS + llm_params + dit_params
    total_fp16_bytes = VISION_FP16_BYTES + llm_fp16_bytes + dit_fp16_bytes

    print("GR00T N1.5 Architecture (5B Total):")
    print(f"  Vision (RADIO + SigLIP): {VISION_PARAMS/1e9:.2f}B params = {format_bytes(VISION_FP16_BYTES)} (NOT quantized)")
    print(f"  LLM (Qwen2.5-2B):        {llm_params/1e9:.2f}B params = {format_bytes(llm_fp16_bytes)}")
    print(f"  DiT (16 blocks):         {dit_params/1e9:.2f}B params = {format_bytes(dit_fp16_bytes)}")
    print(f"  ─────────────────────────────────────────────────────────────")
    print(f"  Total Model:             {total_params/1e9:.2f}B params = {format_bytes(total_fp16_bytes)} (FP16)")
    print()

    # Quantization scenarios
    scenarios = {
        "LLM Only": {
            "include": r".*llm\.layers.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
            "exclude": r"$^",
        },
        "DiT MLP Only": {
            "include": r".*dit\.blocks.*\.ff\.net\.(0\.proj|2).*",
            "exclude": r"$^",
        },
        "DiT Attention Only": {
            "include": r".*dit\.blocks.*\.attn1\.(to_q|to_k|to_v|to_out\.0).*",
            "exclude": r"$^",
        },
        "DiT Full (Attn + MLP)": {
            "include": r".*dit\.blocks.*\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2)).*",
            "exclude": r"$^",
        },
        "LLM + DiT MLP (current)": {
            "include": r".*(llm\.layers.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|dit\.blocks.*\.ff\.net\.(0\.proj|2)).*",
            "exclude": r"$^",
        },
        "LLM + DiT Full": {
            "include": r".*(llm\.layers.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|dit\.blocks.*\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))).*",
            "exclude": r"$^",
        },
    }

    duquant_config = {
        "weight_bits": 4,
        "block_size": 64,
        "enable_permute": False,
        "row_rot_mode": "restore",
    }

    print(f"DuQuant Configuration: W{duquant_config['weight_bits']} block={duquant_config['block_size']} row_rot={duquant_config['row_rot_mode']}")
    print()
    print("=" * 100)
    print()

    for scenario_name, scenario_config in scenarios.items():
        print(f"📊 Scenario: {scenario_name}")
        print("-" * 100)

        matched_layers = [
            layer for layer in all_layers
            if match_layer(layer["name"], scenario_config["include"], scenario_config["exclude"])
        ]

        num_layers = len(matched_layers)

        if num_layers == 0:
            print("  ⚠️  No layers matched!")
            print()
            print("=" * 100)
            print()
            continue

        # Separate matched layers by module
        matched_llm = [l for l in matched_layers if l["cat"] == "LLM"]
        matched_dit = [l for l in matched_layers if l["cat"] == "DiT"]

        llm_matched_params = sum((l["out"] * l["in"]) + (l["out"] if l["bias"] else 0) for l in matched_llm)
        dit_matched_params = sum((l["out"] * l["in"]) + (l["out"] if l["bias"] else 0) for l in matched_dit)

        llm_matched_fp16 = llm_matched_params * 2
        dit_matched_fp16 = dit_matched_params * 2

        print(f"  Quantized layers: {num_layers} ({len(matched_llm)} LLM + {len(matched_dit)} DiT)")
        print(f"  Quantized params: {(llm_matched_params + dit_matched_params)/1e9:.2f}B")
        print()

        # Calculate quantized memory for matched layers
        duquant_llm = {"quantized_weights": 0, "weight_scales": 0, "R_in_blocks": 0, "R_out_blocks": 0, "permutation": 0, "bias": 0}
        duquant_dit = {"quantized_weights": 0, "weight_scales": 0, "R_in_blocks": 0, "R_out_blocks": 0, "permutation": 0, "bias": 0}

        for layer in matched_llm:
            layer_memory = calculate_duquant_memory_for_layer(layer["out"], layer["in"], layer["bias"], **duquant_config)
            for key in duquant_llm:
                duquant_llm[key] += layer_memory[key]

        for layer in matched_dit:
            layer_memory = calculate_duquant_memory_for_layer(layer["out"], layer["in"], layer["bias"], **duquant_config)
            for key in duquant_dit:
                duquant_dit[key] += layer_memory[key]

        llm_quant_bytes = sum(duquant_llm.values())
        dit_quant_bytes = sum(duquant_dit.values())

        # Unmatched parts remain FP16
        llm_unmatched_fp16 = llm_fp16_bytes - llm_matched_fp16
        dit_unmatched_fp16 = dit_fp16_bytes - dit_matched_fp16

        # Total after quantization
        total_llm_after = llm_unmatched_fp16 + llm_quant_bytes
        total_dit_after = dit_unmatched_fp16 + dit_quant_bytes
        total_after_quant = VISION_FP16_BYTES + total_llm_after + total_dit_after

        # LLM+DiT only (without Vision)
        llm_dit_original = llm_fp16_bytes + dit_fp16_bytes
        llm_dit_after = total_llm_after + total_dit_after
        llm_dit_savings = llm_dit_original - llm_dit_after
        llm_dit_compression = llm_dit_original / llm_dit_after if llm_dit_after > 0 else 0

        print("  📦 LLM + DiT Memory (Vision NOT included):")
        print()
        print(f"  LLM (Qwen2.5-2B):")
        print(f"    Original (FP16):        {format_bytes(llm_fp16_bytes)}")
        if llm_quant_bytes > 0:
            print(f"    Quantized portion:      {format_bytes(llm_matched_fp16)} → {format_bytes(llm_quant_bytes)} ({llm_matched_fp16/llm_quant_bytes:.2f}x)")
            print(f"    Unquantized portion:    {format_bytes(llm_unmatched_fp16)}")
        else:
            print(f"    Quantized portion:      None (keeping FP16)")
        print(f"    After quantization:     {format_bytes(total_llm_after)}")
        print(f"    Savings:                {format_bytes(llm_fp16_bytes - total_llm_after)} ({(llm_fp16_bytes - total_llm_after)/llm_fp16_bytes*100:.1f}%)")
        print()

        print(f"  DiT (16 blocks):")
        print(f"    Original (FP16):        {format_bytes(dit_fp16_bytes)}")
        if dit_quant_bytes > 0:
            print(f"    Quantized portion:      {format_bytes(dit_matched_fp16)} → {format_bytes(dit_quant_bytes)} ({dit_matched_fp16/dit_quant_bytes:.2f}x)")
            print(f"    Unquantized portion:    {format_bytes(dit_unmatched_fp16)}")
        else:
            print(f"    Quantized portion:      None (keeping FP16)")
        print(f"    After quantization:     {format_bytes(total_dit_after)}")
        print(f"    Savings:                {format_bytes(dit_fp16_bytes - total_dit_after)} ({(dit_fp16_bytes - total_dit_after)/dit_fp16_bytes*100:.1f}%)")
        print()

        print("  ─────────────────────────────────────────────────────────────")
        print(f"  LLM + DiT Total:")
        print(f"    Original (FP16):        {format_bytes(llm_dit_original)}")
        print(f"    After quantization:     {format_bytes(llm_dit_after)}")
        print(f"    Total savings:          {format_bytes(llm_dit_savings)} ({llm_dit_savings/llm_dit_original*100:.1f}%)")
        print(f"    Compression ratio:      {llm_dit_compression:.2f}x")
        print()

        print("=" * 100)
        print()

    print("✓ Memory calculation complete!")


if __name__ == "__main__":
    main()
