#!/usr/bin/env python
"""
Visualize inference speedup comparison between baseline and QuantVLA
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

# Data from the table
models = [r'$\pi\,0.5$', 'GR00T N1.5']
baseline = [1.0, 1.0]  # Baseline is always 1x
quantvla = [1.22, 1.07]  # Speedup values

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

# Create horizontal bars (baseline on top position)
bars1 = ax.barh([i + height/2 for i in y], baseline, height=height,
                label='Baseline', color=color_baseline, alpha=1.0, edgecolor='black', linewidth=2.5)
bars2 = ax.barh([i - height/2 for i in y], quantvla, height=height,
                label='QuantVLA', color=color_quantvla, alpha=1.0, edgecolor='black', linewidth=2.5)

# Set labels
ax.set_xlabel('Inference speedup')
ax.set_yticks(list(y))
ax.set_yticklabels(models)
ax.set_xlim(0.9, 1.3)

# Add grid with deeper alpha
ax.grid(True, axis="x", alpha=0.4, linestyle="--")
ax.set_axisbelow(True)

# Add legend - upper right, with frame border but no patch edges
legend = ax.legend(framealpha=1.0, facecolor='white', edgecolor='black', loc='upper right')
# Remove black edges from legend color patches
for patch in legend.get_patches():
    patch.set_edgecolor('none')

# Add speedup annotations with dashed arrows (on baseline bars which are now on top)
for i, v in enumerate(quantvla):
    if v > baseline[i]:  # Only show speedup if greater than baseline
        # Draw dashed line from baseline to quantvla value on baseline bar position (now at top)
        y_pos = i + height/2
        ax.annotate('', xy=(v, y_pos), xytext=(baseline[i], y_pos),
                    arrowprops=dict(arrowstyle='<->', color='#7fb069', linestyle='--',
                                  linewidth=2.0, alpha=0.8))
        # Add speedup text below the arrow line (move up to avoid border overlap)
        mid_x = (baseline[i] + v) / 2
        ax.text(mid_x, y_pos - 0.06, f'{v:.2f}×', ha='center', va='top',
                fontsize=13, color='#7fb069', weight='bold')

fig.tight_layout()
saved_path = safe_save(fig, 'inference_speedup_comparison.png')
plt.show()

print(f"\n✓ Chart generated successfully!")
print(f"  - π0.5: Baseline 1.00x → QuantVLA 1.22x (2.0x faster)")
print(f"  - GR00T N1.5: Baseline 1.00x → QuantVLA 1.07x (1.3x faster)")
