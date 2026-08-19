#!/usr/bin/env python3
"""Simple script to check ATM alpha JSON file without loading the model."""

import json
import numpy as np
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_atm_simple.py <alpha_json_path>")
        sys.exit(1)

    alpha_path = sys.argv[1]

    print("=" * 80)
    print(f"Analyzing Alpha JSON: {alpha_path}")
    print("=" * 80)

    with open(alpha_path, 'r') as f:
        data = json.load(f)

    print(f"\n📊 Basic Info:")
    print(f"  Total layers: {len(data)}")

    # Collect all alpha values
    all_alphas = []
    layer_stats = []

    for layer_name in sorted(data.keys()):
        layer_data = data[layer_name]
        alpha_values = layer_data.get('all', layer_data.get('alpha', []))
        all_alphas.extend(alpha_values)

        layer_stats.append({
            'name': layer_name,
            'num_heads': len(alpha_values),
            'mean': np.mean(alpha_values),
            'std': np.std(alpha_values),
            'min': np.min(alpha_values),
            'max': np.max(alpha_values),
        })

    alphas = np.array(all_alphas)

    # Overall statistics
    print(f"\n📈 Overall Alpha Statistics:")
    print(f"  Total heads: {len(alphas)}")
    print(f"  Mean:   {np.mean(alphas):.4f}")
    print(f"  Std:    {np.std(alphas):.4f}")
    print(f"  Min:    {np.min(alphas):.4f}")
    print(f"  Max:    {np.max(alphas):.4f}")
    print(f"  Median: {np.median(alphas):.4f}")

    # Effectiveness metrics
    print(f"\n💪 ATM Effectiveness Metrics:")
    neutral_5 = np.sum(np.abs(alphas - 1.0) <= 0.05) / len(alphas) * 100
    neutral_10 = np.sum(np.abs(alphas - 1.0) <= 0.10) / len(alphas) * 100
    extremes = np.sum((alphas <= 0.85) | (alphas >= 1.2)) / len(alphas) * 100
    avg_deviation = np.mean(np.abs(alphas - 1.0))

    print(f"  Avg deviation from 1.0:    {avg_deviation:.4f}")
    print(f"  % neutral (0.95-1.05):     {neutral_5:.1f}%")
    print(f"  % neutral (0.90-1.10):     {neutral_10:.1f}%")
    print(f"  % at extremes (≤0.85|≥1.2): {extremes:.1f}%")

    # Interpretation
    print(f"\n🔍 Interpretation:")
    if avg_deviation < 0.08:
        print(f"  ⚠️  WEAK ATM: Average deviation ({avg_deviation:.4f}) is very low")
        print(f"     Most alphas are close to 1.0, ATM may have limited effect")
    elif avg_deviation < 0.12:
        print(f"  ⚙️  MODERATE ATM: Average deviation ({avg_deviation:.4f}) is moderate")
        print(f"     ATM may provide some improvement")
    else:
        print(f"  ✅ STRONG ATM: Average deviation ({avg_deviation:.4f}) is high")
        print(f"     ATM is applying significant corrections")

    if extremes < 20:
        print(f"  ⚠️  FEW EXTREME VALUES: Only {extremes:.1f}% at boundaries")
        print(f"     Calibration may be too conservative or uncertain")

    # Per-layer breakdown
    print(f"\n📋 Per-Layer Breakdown:")
    print(f"{'Layer':<55} {'Heads':<7} {'Mean':<8} {'Std':<8} {'Range':<15}")
    print("-" * 100)

    for stat in layer_stats:
        range_str = f"[{stat['min']:.2f}, {stat['max']:.2f}]"
        print(f"{stat['name']:<55} {stat['num_heads']:<7} {stat['mean']:<8.4f} {stat['std']:<8.4f} {range_str:<15}")

    # Odd-even pattern check
    print(f"\n🔄 Odd-Even Layer Pattern:")
    even_alphas = []
    odd_alphas = []

    for layer_name, layer_data in data.items():
        layer_num = int(layer_name.split('.')[3])
        alpha_values = layer_data.get('all', layer_data.get('alpha', []))

        if layer_num % 2 == 0:
            even_alphas.extend(alpha_values)
        else:
            odd_alphas.extend(alpha_values)

    even_mean = np.mean(even_alphas)
    odd_mean = np.mean(odd_alphas)
    diff = even_mean - odd_mean

    print(f"  Even layers mean: {even_mean:.4f}")
    print(f"  Odd layers mean:  {odd_mean:.4f}")
    print(f"  Difference:       {diff:+.4f}")

    if abs(diff) > 0.1:
        print(f"  ✅ Strong odd-even pattern (diff={abs(diff):.4f})")
    else:
        print(f"  ⚠️  Weak odd-even pattern (diff={abs(diff):.4f})")

    print("\n" + "=" * 80)
    print("✅ Analysis Complete")
    print("=" * 80)

if __name__ == "__main__":
    main()
