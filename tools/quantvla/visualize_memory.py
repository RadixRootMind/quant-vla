#!/usr/bin/env python
"""
Visualize memory comparison between baseline and QuantVLA
for pi0.5 and GR00T models.
"""

import os
import matplotlib.pyplot as plt

# Global font sizes 14pt
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 14,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.titlesize": 14,
})

# Data from the table (GR00T on top)
models = ['GR00T N1.5', r'$\pi\,0.5$']
baseline = [2.02, 4.27]  # Baseline memory
quantvla = [0.91, 1.28]  # QuantVLA memory (corrected data)

# Calculate memory reduction ratio (baseline / quantvla)
speedup = [baseline[i] / quantvla[i] for i in range(len(models))]

# Colors
color_baseline = '#ec7776'  # Red for baseline
color_quantvla = '#f0a23a'  # Orange for QuantVLA

# Output directory
output_dir = '/home/jz97/VLM_REPO/Isaac-GR00T'
os.makedirs(output_dir, exist_ok=True)

def safe_save(fig, filename):
    """Save figure to output directory"""
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white', transparent=False)
    print(f"Saved: {path}")
    return path

# Create the horizontal grouped bar chart
y = range(len(models))
height = 0.35

fig = plt.figure(figsize=(9, 4), facecolor='white')
ax = fig.add_subplot(111)

# Create horizontal bars (QuantVLA on top position)
quant_positions = [i + height / 2 for i in y]
baseline_positions = [i - height / 2 for i in y]
bars_quant = ax.barh(quant_positions, quantvla, height=height,
                     label='QuantVLA', color=color_quantvla, alpha=1.0, edgecolor='black', linewidth=2.5)
bars_base = ax.barh(baseline_positions, baseline, height=height,
                    label='Baseline', color=color_baseline, alpha=1.0, edgecolor='black', linewidth=2.5)

# Set labels
ax.set_xlabel('Memory (GB)')
ax.set_yticks(list(y))
ax.set_yticklabels(models)
ax.set_xlim(0, max(max(baseline), max(quantvla)) * 1.15)

# Add grid with deeper alpha
ax.grid(True, axis="x", alpha=0.4, linestyle="--")
ax.set_axisbelow(True)

# Add legend - upper right, with frame border but preserve order (Baseline first)
legend = ax.legend([bars_base, bars_quant], ['Baseline', 'QuantVLA'],
                   framealpha=1.0, facecolor='white', edgecolor='black', loc='upper right')
# Remove black edges from legend color patches
for patch in legend.get_patches():
    patch.set_edgecolor('none')

# Add memory reduction annotations aligned with QuantVLA bars
for i in range(len(models)):
    quant_y = quant_positions[i]
    start_x = baseline[i]
    end_x = quantvla[i]
    ratio_text = f'{speedup[i]:.2f}×'

    ax.annotate('', xy=(start_x, quant_y), xytext=(end_x, quant_y),
                arrowprops=dict(arrowstyle='<->', color='#7fb069', linestyle='--',
                                linewidth=2.0, alpha=0.8))
    # Place text slightly above the arrow near the QuantVLA bar's left edge
    text_x = min(start_x, end_x) + 0.05 * max(baseline + quantvla)
    if i == 0:
        ax.text(text_x+.35, quant_y - height * -0.5, ratio_text, ha='center', va='bottom',
                fontsize=13, color='#7fb069', weight='bold')
    else:
        ax.text(text_x+1.3, quant_y - height * -0.5, ratio_text, ha='center', va='bottom',
                fontsize=13, color='#7fb069', weight='bold')
# Invert y-axis to put GR00T N1.5 on top
ax.invert_yaxis()

fig.tight_layout()
saved_path = safe_save(fig, 'memory_comparison.png')
plt.show()

print(f"\n✓ Chart generated successfully!")
print(f"  - GR00T N1.5: Baseline {baseline[0]:.2f}GB → QuantVLA {quantvla[0]:.2f}GB ({speedup[0]:.2f}× reduction)")
print(f"  - π0.5: Baseline {baseline[1]:.2f}GB → QuantVLA {quantvla[1]:.2f}GB ({speedup[1]:.2f}× reduction)")
