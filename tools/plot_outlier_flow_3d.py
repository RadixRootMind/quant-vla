#!/usr/bin/env python
"""3D-surface "outlier flattening pipeline" figure, DuQuant-style.

Shows |X| / |X·P| / |X·P·R_svd| / |X·P·R_svd·H| as 4 sequential 3D surfaces
with arrows between the panels and the same z-axis range so the reader sees
the outlier mountain peaks getting flatter at each stage.

Pipeline stages (4 panels):
  0. original                — raw X (no rotation, no permutation)
  1. after Permutation       — X[:, perm] using weight-energy zigzag perm
  2. after Permutation + SVD — X[:, perm] @ block_diag(SVD basis per block)
  3. after full A2-lite      — stage-2 followed by within-block Hadamard

Arrows between panels labelled "1 Permutation" / "2 Rotation (SVD)" /
"3 Rotation (Hadamard)".

x-axis = input-channel index (sampled / downsampled for display readability)
y-axis = token index (downsampled)
z-axis = |X| (linear)  — peaks are outlier locations

Colormap: coolwarm (blue→white→red) with blue = small, red = outlier peak.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers projection)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gr00t.quantization.duquant_preprocess import (
    compute_duquant_rotation_only, zigzag_permutation,
)


OUT_DIR = Path("experiment_results/visualization_for_paper")
BLOCK = 64
# Pick one strongly outlier'd layer per model (the same picks we've used
# throughout this session). Pipeline figure produced per layer.
LAYERS = [
    # (label, src, layer-tag, z_clip_max, vcenter_red)
    # gr00t: DiT.L02.o_proj. Z clipped at 12, red transition starts at 6.
    ("gr00t",
     "experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep2.pt",
     "DiT.L02.o_proj",
     12.0,    # z-axis upper clip
     6.0),    # vcenter (red transition)
    # pi0.5: paligemma.L02.gate_proj. No z clip; red transition starts at 6.
    ("pi0.5",
     "experiment_results/svd_diagnostics_gr00t/xw_dump_pi05_object_n4.pt",
     "paligemma.L02.gate_proj",
     None,    # no z clip — use row max
     6.0),
]
OUT_NAME = "outlier_flow_3d.png"

# Display sizes
N_TOKEN_BINS    = 200
N_CHANNEL_BINS  = 256
ARROW_LABELS = ["ZigZag", "SVD", "Hadamard"]


def build_pipeline_stages(W: torch.Tensor, X: torch.Tensor):
    """Return [X_orig, X_perm, X_perm_svd, X_perm_svdhad] as abs-valued
    [N_tokens, in_features] tensors."""
    W_np = W.detach().cpu().float().numpy()
    in_f = W.shape[1]
    # Permutation derived the same way compute_duquant_rotation_only does.
    energy = np.mean(W_np ** 2, axis=0)
    mean_e = float(energy.mean())
    energy = 0.85 * energy + 0.15 * mean_e
    perm = zigzag_permutation(energy)

    perm_t = torch.from_numpy(perm.astype("int64"))
    X_orig = X.float()
    X_perm = X_orig.index_select(-1, perm_t)

    # SVD-only and SVD-Hadamard rotations (each does its own perm + R; we use
    # them to obtain R applied on already-permuted X).
    R_svd, perm_check  = compute_duquant_rotation_only(W, block_size=BLOCK, enable_permute=True,
                                                       lambda_smooth=0.15, rot_mode="svd")
    R_svdh, _          = compute_duquant_rotation_only(W, block_size=BLOCK, enable_permute=True,
                                                       lambda_smooth=0.15, rot_mode="svd_hadamard")
    # build_R_perm returns R already in post-perm space; both functions return
    # the same perm so we can apply R to X_perm directly.
    X_perm_svd  = X_perm @ R_svd.float()
    X_perm_svdh = X_perm @ R_svdh.float()
    return [X_orig.abs(), X_perm.abs(), X_perm_svd.abs(), X_perm_svdh.abs()]


def downsample_2d(M: torch.Tensor, n_rows: int, n_cols: int) -> np.ndarray:
    """MAX-aggregate M to (n_rows, n_cols) bins — preserves outlier peaks
    (mean would average them away into the noise floor)."""
    H, W = M.shape
    h_bin = max(1, H // n_rows); w_bin = max(1, W // n_cols)
    Ht = (H // h_bin) * h_bin; Wt = (W // w_bin) * w_bin
    M = M[:Ht, :Wt].float()
    return M.reshape(Ht // h_bin, h_bin, Wt // w_bin, w_bin).amax(dim=(1, 3)).cpu().numpy()


def make_figure():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Compute all tiles up front so we can derive a SHARED color/z scale.
    rows = []   # list of (model_label, tiles_list_of_4, z_clip_max, vcenter)
    for model_label, src, tag, z_clip, vcenter in LAYERS:
        print(f"[load] {src}  {tag}")
        cal = torch.load(src, weights_only=False)
        rec = cal[tag]
        W = rec["W"].float()
        X = rec["X"].float()
        print(f"  W={tuple(W.shape)}  X={tuple(X.shape)}")
        stages = build_pipeline_stages(W, X)
        tiles = [downsample_2d(M, N_TOKEN_BINS, N_CHANNEL_BINS) for M in stages]
        # Resolve z_max: either explicit clip or per-row max
        z_max = float(z_clip) if z_clip is not None else max(float(t.max()) for t in tiles)
        # Keep the UNCLIPPED max value of each tile for the annotation, and
        # a CLIPPED copy for the surface plot (so visually the surface caps
        # at z_max, but the displayed "max=..." text reports the real peak).
        true_maxes = [float(t.max()) for t in tiles]
        clipped_tiles = [np.clip(t, 0.0, z_max) for t in tiles]
        rows.append((model_label, clipped_tiles, true_maxes, z_max, float(vcenter)))

    # Per-row TwoSlopeNorm: red transitions at vcenter, deepest red at z_max.
    from matplotlib.colors import TwoSlopeNorm
    row_norm = [TwoSlopeNorm(vmin=0.0, vcenter=vcenter, vmax=max(z_max, vcenter + 0.1))
                for _, _, _, z_max, vcenter in rows]

    mpl.rcParams["font.size"] = 12
    fig = plt.figure(figsize=(22, 11))
    panel_titles = ["original", "after permutation", "perm + SVD", "perm + SVD-Hadamard"]

    from matplotlib.gridspec import GridSpec
    n_models = len(rows)
    # No arrow columns — panels sit flush next to each other.
    gs = GridSpec(n_models, 4,
                  width_ratios=[1.0, 1.0, 1.0, 1.0],
                  hspace=0.02, wspace=0.0)

    for ri, (model_label, tiles, true_maxes, z_max, _) in enumerate(rows):
        norm = row_norm[ri]
        for ci, tile in enumerate(tiles):
            ax = fig.add_subplot(gs[ri, ci], projection="3d")
            ny, nx = tile.shape
            Xg, Yg = np.meshgrid(np.arange(nx), np.arange(ny))
            ax.plot_surface(Xg, Yg, tile, cmap="bwr",
                            rcount=ny, ccount=nx,
                            linewidth=0, antialiased=True,
                            norm=norm)
            ax.set_zlim(0, z_max * 1.05)
            ax.set_xlabel("Channel", fontsize=13, labelpad=3)
            ax.set_ylabel("Token",   fontsize=13, labelpad=3)
            ax.set_zlabel("|X·R|",   fontsize=13, labelpad=3)
            ax.tick_params(labelsize=11)
            ax.view_init(elev=30, azim=-62)
            # Show the TRUE (un-clipped) max in the annotation. Surface is
            # visually clipped at z_max but the printed number is the real
            # peak so the reader sees how much got chopped off.
            ax.text2D(0.02, 0.94, f"max = {true_maxes[ci]:.2f}",
                      transform=ax.transAxes, fontsize=13, color="#222")
            if ri == 0:
                ax.set_title(panel_titles[ci], fontsize=15, pad=6)

    # Model labels on the LEFT side (vertical, one per row).
    # Use the gridspec slot position directly (NOT add_subplot — that would
    # overwrite the 3D panel sitting in column 0).
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.04,
                        hspace=0.02, wspace=0.0)
    fig.canvas.draw_idle()
    for ri, (model_label, _, _, _, _) in enumerate(rows):
        bbox = gs[ri, 0].get_position(fig)
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.018, y_center, model_label,
                 ha="center", va="center", rotation=90,
                 fontsize=19, fontweight="normal", color="#111",
                 fontfamily="DejaVu Sans")

    fig.suptitle("activation outlier", fontsize=18, fontweight="bold", y=0.985)

    out_path = OUT_DIR / OUT_NAME
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot] saved → {out_path} + .pdf")
    plt.close(fig)


def main():
    make_figure()


if __name__ == "__main__":
    main()
