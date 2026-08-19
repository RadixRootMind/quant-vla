#!/usr/bin/env python3
"""Diagnostic script to check if ATM is properly loaded in GR00T model."""

import json
import os
import sys
from pathlib import Path

# Add repo to path
REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

def check_alpha_json(alpha_path: str):
    """Check the alpha JSON file contents."""
    print("=" * 80)
    print("1. Checking Alpha JSON File")
    print("=" * 80)

    if not os.path.exists(alpha_path):
        print(f"❌ Alpha JSON file not found: {alpha_path}")
        return None

    print(f"✅ Alpha JSON exists: {alpha_path}\n")

    with open(alpha_path, 'r') as f:
        data = json.load(f)

    print(f"Total layers in JSON: {len(data)}")
    print(f"\nFirst 5 layer names:")
    for i, key in enumerate(list(data.keys())[:5]):
        alpha_values = data[key].get('all', data[key].get('alpha', []))
        print(f"  {i+1}. {key}")
        print(f"     Num heads: {len(alpha_values)}")
        print(f"     Alpha range: [{min(alpha_values):.4f}, {max(alpha_values):.4f}]")

    # Check alpha value statistics
    all_alphas = []
    for layer_data in data.values():
        alpha_values = layer_data.get('all', layer_data.get('alpha', []))
        all_alphas.extend(alpha_values)

    import numpy as np
    alphas = np.array(all_alphas)

    print(f"\n📊 Alpha Statistics:")
    print(f"  Mean: {np.mean(alphas):.4f}")
    print(f"  Std:  {np.std(alphas):.4f}")
    print(f"  Min:  {np.min(alphas):.4f}")
    print(f"  Max:  {np.max(alphas):.4f}")
    print(f"  % in [0.95, 1.05] (neutral): {np.sum(np.abs(alphas - 1.0) <= 0.05) / len(alphas) * 100:.1f}%")
    print(f"  % at extremes (≤0.85 or ≥1.2): {np.sum((alphas <= 0.85) | (alphas >= 1.2)) / len(alphas) * 100:.1f}%")

    return data


def check_model_layers():
    """Load model and check actual layer names."""
    print("\n" + "=" * 80)
    print("2. Checking Model Layer Names")
    print("=" * 80)

    try:
        from gr00t.model.policy import Gr00tPolicy
        from gr00t.experiment.data_config import load_data_config
        import torch

        # Disable quantization and ATM for inspection
        os.environ["GR00T_DUQUANT_ENABLE"] = "0"
        os.environ["GR00T_ATM_ENABLE"] = "0"

        model_path = "youliangtan/gr00t-n1.5-libero-goal-posttrain"
        data_config = load_data_config("examples.Libero.custom_data_config:LiberoDataConfig")

        print(f"Loading model: {model_path}...")
        policy = Gr00tPolicy(
            model_path=model_path,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag="new_embodiment",
            denoising_steps=8,
        )

        print("✅ Model loaded\n")

        # Find DiT attention layers
        dit_attn_layers = []
        for name, module in policy.model.named_modules():
            if "action_head" in name and "transformer_blocks" in name and "attn1" in name:
                # Only keep the parent attention module, not submodules
                if name.endswith("attn1"):
                    dit_attn_layers.append(name)

        print(f"Found {len(dit_attn_layers)} DiT attention layers:\n")
        for i, name in enumerate(dit_attn_layers[:5]):
            print(f"  {i+1}. {name}")

        if len(dit_attn_layers) > 5:
            print(f"  ... ({len(dit_attn_layers) - 5} more)")

        return dit_attn_layers

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_name_matching(alpha_json_data, model_layer_names):
    """Check if alpha JSON layer names match model layer names."""
    print("\n" + "=" * 80)
    print("3. Checking Layer Name Matching")
    print("=" * 80)

    if alpha_json_data is None or model_layer_names is None:
        print("⚠️  Skipping due to previous errors")
        return

    json_keys = set(alpha_json_data.keys())
    model_names = set(model_layer_names)

    matched = json_keys & model_names
    json_only = json_keys - model_names
    model_only = model_names - json_keys

    print(f"\n📊 Matching Results:")
    print(f"  Layers in both JSON and model: {len(matched)}")
    print(f"  Layers only in JSON: {len(json_only)}")
    print(f"  Layers only in model: {len(model_only)}")

    if matched:
        print(f"\n✅ Matched layers (showing first 3):")
        for name in list(matched)[:3]:
            print(f"    {name}")

    if json_only:
        print(f"\n⚠️  Layers in JSON but not in model (showing first 3):")
        for name in list(json_only)[:3]:
            print(f"    {name}")

    if model_only:
        print(f"\n⚠️  Layers in model but not in JSON (showing first 3):")
        for name in list(model_only)[:3]:
            print(f"    {name}")

    # Check if there's a 'model.' prefix mismatch
    if json_only and model_only:
        json_stripped = {k.replace('model.', ''): k for k in json_only}
        model_stripped = {k.replace('model.', ''): k for k in model_only}

        potential_matches = set(json_stripped.keys()) & set(model_stripped.keys())
        if potential_matches:
            print(f"\n💡 Potential 'model.' prefix mismatch detected!")
            print(f"   Example:")
            ex = list(potential_matches)[0]
            print(f"   JSON:  {json_stripped[ex]}")
            print(f"   Model: {model_stripped[ex]}")

    return len(matched)


def main():
    print("\n" + "=" * 80)
    print("GR00T ATM Diagnostic Tool")
    print("=" * 80)

    # Check environment variables
    print("\n📋 Environment Variables:")
    atm_enable = os.getenv("GR00T_ATM_ENABLE", "0")
    atm_alpha_path = os.getenv("GR00T_ATM_ALPHA_PATH", "")

    print(f"  GR00T_ATM_ENABLE: {atm_enable}")
    print(f"  GR00T_ATM_ALPHA_PATH: {atm_alpha_path}")

    if atm_enable not in ("1", "true", "True"):
        print(f"\n⚠️  Warning: ATM is not enabled (GR00T_ATM_ENABLE={atm_enable})")

    if not atm_alpha_path:
        print(f"\n❌ Error: GR00T_ATM_ALPHA_PATH is not set!")
        alpha_path = "atm_alpha_dit_mlp_permute0_goal.json"  # Default for testing
        print(f"   Using default for testing: {alpha_path}")
    else:
        alpha_path = atm_alpha_path

    # Run diagnostics
    alpha_data = check_alpha_json(alpha_path)
    model_layers = check_model_layers()

    if alpha_data and model_layers:
        matched_count = check_name_matching(alpha_data, model_layers)

        print("\n" + "=" * 80)
        print("📊 Summary")
        print("=" * 80)

        if matched_count == len(model_layers):
            print(f"✅ Perfect match: All {matched_count} model layers have alpha values")
        elif matched_count > 0:
            print(f"⚠️  Partial match: {matched_count}/{len(model_layers)} layers matched")
            print(f"   ATM will only be applied to matched layers")
        else:
            print(f"❌ No match: ATM will NOT work!")
            print(f"   Possible causes:")
            print(f"   - Layer name format mismatch (check 'model.' prefix)")
            print(f"   - Alpha JSON from different model version")

    print("\n" + "=" * 80)
    print("✅ Diagnostic Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
