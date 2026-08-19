#!/usr/bin/env python
"""Scan all linear layers in GR00T model to understand structure for DuQuant quantization."""

import os
import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.data.embodiment_tags import EmbodimentTag

def scan_linear_layers(model: torch.nn.Module, prefix: str = ""):
    """Recursively scan and print all Linear layers in the model."""
    linear_layers = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            full_name = f"{prefix}{name}" if prefix else name
            linear_layers.append({
                'name': full_name,
                'in_features': module.in_features,
                'out_features': module.out_features,
                'has_bias': module.bias is not None,
            })

    return linear_layers

def group_by_component(layers):
    """Group layers by model component (LLM, DiT, etc.)."""
    groups = {
        'vlm': [],           # Vision-Language Model backbone
        'vision': [],        # Vision encoder
        'llm_attn': [],      # LLM attention layers
        'llm_mlp': [],       # LLM MLP layers
        'dit_attn': [],      # DiT attention layers
        'dit_mlp': [],       # DiT MLP layers
        'action_head': [],   # Action prediction head
        'other': [],
    }

    for layer in layers:
        name = layer['name']

        # Vision encoder
        if 'vision' in name.lower() or 'siglip' in name.lower() or 'radio' in name.lower():
            groups['vision'].append(layer)
        # VLM/LLM attention
        elif ('vlm' in name.lower() or 'language' in name.lower() or 'qwen' in name.lower() or 'eagle' in name.lower()) and \
             any(x in name.lower() for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv', 'attn']):
            groups['llm_attn'].append(layer)
        # VLM/LLM MLP
        elif ('vlm' in name.lower() or 'language' in name.lower() or 'qwen' in name.lower() or 'eagle' in name.lower()) and \
             any(x in name.lower() for x in ['mlp', 'fc', 'gate_proj', 'up_proj', 'down_proj']):
            groups['llm_mlp'].append(layer)
        # DiT attention
        elif ('dit' in name.lower() or 'diffusion' in name.lower()) and \
             any(x in name.lower() for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'qkv', 'attn']):
            groups['dit_attn'].append(layer)
        # DiT MLP
        elif ('dit' in name.lower() or 'diffusion' in name.lower()) and \
             any(x in name.lower() for x in ['mlp', 'fc', 'gate_proj', 'up_proj', 'down_proj']):
            groups['dit_mlp'].append(layer)
        # Action head
        elif 'action' in name.lower() or 'head' in name.lower():
            groups['action_head'].append(layer)
        # VLM backbone (catch-all for remaining VLM layers)
        elif any(x in name.lower() for x in ['vlm', 'language', 'qwen', 'eagle']):
            groups['vlm'].append(layer)
        else:
            groups['other'].append(layer)

    return groups

def print_layer_summary(groups):
    """Print a summary of layers by component."""
    print("\n" + "="*100)
    print("GR00T MODEL LINEAR LAYER STRUCTURE")
    print("="*100)

    total = 0
    for group_name, layers in groups.items():
        if not layers:
            continue

        print(f"\n{'─'*100}")
        print(f"📦 {group_name.upper().replace('_', ' ')}: {len(layers)} layers")
        print(f"{'─'*100}")

        # Show first 5 examples
        for i, layer in enumerate(layers[:5]):
            print(f"  {i+1}. {layer['name']}")
            print(f"     └─ ({layer['in_features']} → {layer['out_features']})")

        if len(layers) > 5:
            print(f"  ... and {len(layers) - 5} more layers")

        total += len(layers)

    print(f"\n{'='*100}")
    print(f"TOTAL LINEAR LAYERS: {total}")
    print(f"{'='*100}\n")

def print_quantization_recommendations(groups):
    """Print quantization recommendations based on OpenPI approach."""
    print("\n" + "="*100)
    print("RECOMMENDED DUQUANT W4A8 QUANTIZATION STRATEGY")
    print("="*100)

    print("\n✅ QUANTIZE (following OpenPI LLM + DiT MLP pattern):")
    print("   1. LLM Attention layers (Q, K, V, O projections)")
    print(f"      └─ Count: {len(groups['llm_attn'])}")
    print("   2. LLM MLP layers (gate_proj, up_proj, down_proj)")
    print(f"      └─ Count: {len(groups['llm_mlp'])}")
    print("   3. DiT MLP layers (gate_proj, up_proj, down_proj)")
    print(f"      └─ Count: {len(groups['dit_mlp'])}")
    total_quant = len(groups['llm_attn']) + len(groups['llm_mlp']) + len(groups['dit_mlp'])
    print(f"\n   📊 TOTAL TO QUANTIZE: {total_quant} layers")

    print("\n❌ DO NOT QUANTIZE:")
    print("   1. Vision encoder (preserve visual features)")
    print(f"      └─ Count: {len(groups['vision'])}")
    print("   2. DiT Attention layers (critical for action generation)")
    print(f"      └─ Count: {len(groups['dit_attn'])}")
    print("   3. Action head (final output layer)")
    print(f"      └─ Count: {len(groups['action_head'])}")
    print("   4. Embeddings and normalization layers")

    print("\n" + "="*100 + "\n")

def save_layer_list(groups, output_path: str):
    """Save layer names to file for use in quantization scripts."""
    # Save all quantization target layers
    quant_layers = []
    quant_layers.extend([l['name'] for l in groups['llm_attn']])
    quant_layers.extend([l['name'] for l in groups['llm_mlp']])
    quant_layers.extend([l['name'] for l in groups['dit_mlp']])

    with open(output_path, 'w') as f:
        f.write("# GR00T DuQuant W4A8 Target Layers\n")
        f.write(f"# Total: {len(quant_layers)} layers\n")
        f.write("# LLM Attention + LLM MLP + DiT MLP\n\n")
        for name in sorted(quant_layers):
            f.write(f"{name}\n")

    print(f"✅ Saved quantization target layers to: {output_path}")

def main():
    print("Loading GR00T model...")

    # Create a small model instance to scan structure (no need to load weights)
    try:
        model = GR00T_N1_5(
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            device='cpu',  # Use CPU to save memory during scan
        )
        print("✅ Model created successfully\n")
    except Exception as e:
        print(f"⚠️  Could not create full model: {e}")
        print("This script requires a valid model configuration.")
        return

    # Scan all linear layers
    print("Scanning linear layers...")
    all_layers = scan_linear_layers(model)
    print(f"✅ Found {len(all_layers)} total linear layers\n")

    # Group by component
    groups = group_by_component(all_layers)

    # Print summary
    print_layer_summary(groups)

    # Print quantization recommendations
    print_quantization_recommendations(groups)

    # Save layer list
    output_dir = Path(__file__).parent.parent / "duquant_configs"
    output_dir.mkdir(exist_ok=True)
    save_layer_list(groups, str(output_dir / "gr00t_quant_layers.txt"))

    print("\n🎉 Layer scan complete!")

if __name__ == "__main__":
    main()
