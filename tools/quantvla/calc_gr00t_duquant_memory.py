#!/usr/bin/env python3
"""
GR00T DuQuant memory calculator (fast mode).

Reads linear-layer shapes directly from safetensors metadata, then estimates
memory usage before and after DuQuant for different layer-selection scenarios.
"""

import argparse
import re
from pathlib import Path

from safetensors import safe_open


###############################################################################
# Utilities
###############################################################################


def format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def normalize_layer_name(tensor_name: str) -> str:
    if tensor_name.endswith(".weight"):
        return tensor_name[:-7]
    if tensor_name.endswith(".bias"):
        return tensor_name[:-5]
    return tensor_name


def match_layer(layer_name: str, scope_prefix: str | None, include_regex: str, exclude_regex: str) -> bool:
    if scope_prefix and not layer_name.startswith(scope_prefix):
        return False
    if include_regex and not re.search(include_regex, layer_name):
        return False
    if exclude_regex and re.search(exclude_regex, layer_name):
        return False
    return True


def calc_duquant_memory(
    out_features: int,
    in_features: int,
    has_bias: bool,
    *,
    weight_bits: int = 4,
    block_size: int = 64,
    enable_permute: bool = True,
    row_rot_mode: str = "restore",
) -> dict[str, float]:
    weight_params = out_features * in_features
    bias_params = out_features if has_bias else 0

    mem = {}
    mem["quantized_weights"] = (weight_params * weight_bits) / 8
    mem["weight_scales"] = out_features * 2  # FP16 per output channel

    n_in_blocks = (in_features + block_size - 1) // block_size
    mem["R_in_blocks"] = n_in_blocks * block_size * block_size * 4

    if row_rot_mode in ("restore", "propagate"):
        n_out_blocks = (out_features + block_size - 1) // block_size
        mem["R_out_blocks"] = n_out_blocks * block_size * block_size * 4
    else:
        mem["R_out_blocks"] = 0

    mem["permutation"] = in_features * 4 if enable_permute else 0
    mem["bias"] = bias_params * 2  # FP16
    mem["total"] = sum(mem.values())
    return mem


###############################################################################
# Scenarios
###############################################################################

SCENARIOS = {
    "LLM only": {
        "scope": "backbone.eagle_model.language_model.",
        "include": r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
        "exclude": r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head)(?:\.|$)",
    },
    "DiT MLP only": {
        "scope": "action_head.model.transformer_blocks.",
        "include": r".*ff\.net\.(0\.proj|2).*",
        "exclude": r"(?:^|\.)(attn1|vision|radio|norm|ln|layernorm)(?:\.|$)",
    },
    "LLM + DiT MLP": {
        "scope": "",
        "include": r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.ff\.net\.(0\.proj|2)).*",
        "exclude": r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|attn1)(?:\.|$)",
    },
    "Full LLM + DiT": {
        "scope": "",
        "include": r".*(backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))).*",
        "exclude": r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head)(?:\.|$)",
    },
}


###############################################################################
# Main
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GR00T DuQuant memory calculator.")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path.home() / "VLM_REPO/Isaac-GR00T/ckpts/gr00t_n1p5/model.safetensors",
        help="Path to GR00T safetensors checkpoint.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 100)
    print("GR00T DuQuant Memory Usage Calculator")
    print("=" * 100)
    print()

    ckpt_path = args.ckpt
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return
    print(f"Checkpoint: {ckpt_path}")
    print()

    layer_shapes: dict[str, dict[str, int | bool]] = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for key in f.keys():
            if not key.endswith(".weight"):
                continue
            name = normalize_layer_name(key)
            tensor = f.get_tensor(key)
            if tensor.ndim != 2:
                continue
            out_features, in_features = tensor.shape
            has_bias = f"{name}.bias" in f.keys()
            layer_shapes[name] = {
                "out_features": out_features,
                "in_features": in_features,
                "has_bias": has_bias,
                "weight_params": out_features * in_features,
                "bias_params": out_features if has_bias else 0,
            }

    total_params_all = sum(info["weight_params"] + info["bias_params"] for info in layer_shapes.values())
    total_original_bytes_all = total_params_all * 2  # BF16

    duquant_cfg = {
        "weight_bits": 4,
        "block_size": 64,
        "enable_permute": True,
        "row_rot_mode": "restore",
    }

    print(f"DuQuant Config: {duquant_cfg}")
    print()

    for scenario, cfg in SCENARIOS.items():
        print(f"📊 Scenario: {scenario}")
        print("-" * 100)

        matched = {
            name: info
            for name, info in layer_shapes.items()
            if match_layer(name, cfg["scope"] or None, cfg["include"], cfg["exclude"])
        }
        num_layers = len(matched)
        total_params = sum(info["weight_params"] + info["bias_params"] for info in matched.values())

        print(f"  Matched layers: {num_layers}")
        print(f"  Total parameters (subset): {total_params:,}")
        if num_layers == 0:
            print("  ⚠️  No layers matched.\n")
            continue

        subset_original_bytes = total_params * 2
        print(f"  Original memory (subset, BF16): {format_bytes(subset_original_bytes)}")

        duquant_totals = {
            "quantized_weights": 0,
            "weight_scales": 0,
            "R_in_blocks": 0,
            "R_out_blocks": 0,
            "permutation": 0,
            "bias": 0,
        }

        for info in matched.values():
            mem = calc_duquant_memory(
                info["out_features"],
                info["in_features"],
                info["has_bias"],
                **duquant_cfg,
            )
            for key in duquant_totals:
                duquant_totals[key] += mem[key]

        subset_quant_bytes = sum(duquant_totals.values())
        print(f"  DuQuant memory (subset): {format_bytes(subset_quant_bytes)}")
        for key, value in duquant_totals.items():
            print(f"    {key:15s}: {format_bytes(value)}")

        subset_savings = subset_original_bytes - subset_quant_bytes
        subset_ratio = subset_original_bytes / subset_quant_bytes if subset_quant_bytes else 0
        print(f"    Savings (subset): {format_bytes(subset_savings)} ({subset_ratio:.2f}x compression)")

        total_quant_bytes = subset_quant_bytes + (total_original_bytes_all - subset_original_bytes)
        total_savings = total_original_bytes_all - total_quant_bytes
        total_ratio = total_original_bytes_all / total_quant_bytes if total_quant_bytes else 0
        print()
        print(f"  Full model (BF16):        {format_bytes(total_original_bytes_all)}")
        print(f"  After this subset:        {format_bytes(total_quant_bytes)}")
        print(f"  Absolute savings:         {format_bytes(total_savings)}")
        print(f"  Overall compression:      {total_ratio:.2f}x")
        print()
        print("=" * 100)
        print()

    print("✓ Memory calculation complete.")


if __name__ == "__main__":
    main()
