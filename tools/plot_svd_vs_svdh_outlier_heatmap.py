#!/usr/bin/env python
"""Outlier-magnitude heatmap: raw / DuQuant SVD / Hadamard / SVD-Hadamard (A2-lite).

Mimics the layout of svd_diagnostics_gr00t/quant_diff_a0_vs_a2lite_L02_down_proj.png
but plots the OUTLIER MAGNITUDE (not quantization error) per element:

    color = log10( |x| / scale_block )

where scale_block is the per-block MSE-optimal int4 scale for that (token, block)
or (row, block) — exactly the scale the quantizer would pick. Diverging colormap:
- log10(|x|/scale) > 0    → element uses ≥1 int4 level (red, "outlier-ish")
- log10(|x|/scale) ≈ 0    → element exactly at one-level boundary
- log10(|x|/scale) < -0.3 → element rounds to 0 (blue, "crushed")
- log10(|x|/scale) > log10(7) ≈ 0.85 → element saturates int4 ceiling
(scaling axis: -2 to +1, dashed reference lines on the colorbar at the boundaries)

Adds a fourth column — **hadamard only** (rot_mode='hadamard') — so we can see
whether Hadamard alone (no SVD) gives the same uniformization as SVD-Hadamard.

Layout (per profile):
  4 rows × 4 cols
   row 1: layer-1 X-side
   row 2: layer-1 W-side
   row 3: layer-2 X-side
   row 4: layer-2 W-side
   cols:  raw / DuQuant SVD / Hadamard only / SVD-Hadamard (A2-lite)
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gr00t.quantization.duquant_preprocess import compute_duquant_rotation_only


OUT_DIR = Path("experiment_results/visualization_for_paper")
BLOCK = 64
QMAX = 7

PROFILES = {
    "gr00t": {
        "src": "experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep2.pt",
        "layers": [
            ("LLM.L02.down_proj", "LLM.L02.down_proj  —  structural pathology  (raw amax-ratio 48000×)"),
            ("LLM.L02.gate_proj", "LLM.L02.gate_proj  —  typical MLP layer  (SVD-only barely moves amax-ratio)"),
        ],
        "out_name": "outlier_heatmap_svd_vs_svdh.png",
        "model_label": "gr00t-N1.5",
    },
    "pi05": {
        "src": "experiment_results/svd_diagnostics_gr00t/xw_dump_pi05_object_n4.pt",
        "layers": [
            ("paligemma.L14.down_proj", "paligemma.L14.down_proj  —  pathology  (X channel-skew 156×)"),
            ("paligemma.L02.gate_proj", "paligemma.L02.gate_proj  —  typical MLP layer  (SVD-only stuck at 12.6×)"),
        ],
        "out_name": "outlier_heatmap_svd_vs_svdh_pi05.png",
        "model_label": "pi0.5",
    },
}

ROT_MODES = ["raw", "svd", "hadamard", "svd_hadamard"]
ROT_TITLES = {
    "raw":           "raw (no rotation)",
    "svd":           "DuQuant SVD",
    "hadamard":      "Hadamard only",
    "svd_hadamard":  "SVD-Hadamard (A2-lite)",
}


def build_R_perm(W: torch.Tensor, rot_mode: str, block: int = BLOCK):
    R, perm = compute_duquant_rotation_only(
        W.cpu().float(), block_size=block, enable_permute=True,
        lambda_smooth=0.15, rot_mode=rot_mode,
    )
    return R.float(), perm


def apply(M: torch.Tensor, R: torch.Tensor, perm) -> torch.Tensor:
    Mc = M.float()
    if perm is not None:
        perm_t = torch.from_numpy(perm.astype("int64"))
        Mc = Mc.index_select(-1, perm_t)
    return Mc @ R


def _mse_optimal_alpha(xb: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    alphas = torch.linspace(0.5, 1.2, 12)
    best_err = None; best_alpha = None
    for alpha in alphas:
        scale = (alpha * amax / QMAX).clamp_min(1e-12)
        q = (xb / scale.unsqueeze(1)).round().clamp(-QMAX - 1, QMAX) * scale.unsqueeze(1)
        err = ((xb - q) ** 2).sum(dim=1)
        if best_err is None:
            best_err = err.clone(); best_alpha = torch.full_like(err, float(alpha))
        else:
            mask = err < best_err
            best_err = torch.where(mask, err, best_err)
            best_alpha = torch.where(mask, torch.full_like(err, float(alpha)), best_alpha)
    return best_alpha


def log_outlier_W(W_r: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """For each element of W_r, return log10(|W| / scale_per_block_per_row)."""
    O, I = W_r.shape
    n_blocks = (I + block - 1) // block
    out = torch.zeros_like(W_r)
    for b in range(n_blocks):
        s, e = b * block, min((b + 1) * block, I)
        Wb = W_r[:, s:e].float()
        amax = Wb.abs().amax(dim=1).clamp_min(1e-12)
        alpha = _mse_optimal_alpha(Wb, amax)
        scale = (alpha * amax / QMAX).clamp_min(1e-12).unsqueeze(1)
        out[:, s:e] = torch.log10((Wb.abs() / scale).clamp_min(1e-6))
    return out


def log_outlier_X(X_r: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """For each element of X_r, return log10(|X| / scale_per_token_per_block)."""
    N, I = X_r.shape
    Xf = X_r.float()
    n_blocks = (I + block - 1) // block
    out = torch.zeros_like(Xf)
    for b in range(n_blocks):
        s, e = b * block, min((b + 1) * block, I)
        Xb = Xf[:, s:e]
        amax = Xb.abs().amax(dim=1).clamp_min(1e-12)
        alpha = _mse_optimal_alpha(Xb, amax)
        scale = (alpha * amax / QMAX).clamp_min(1e-12).unsqueeze(1)
        out[:, s:e] = torch.log10((Xb.abs() / scale).clamp_min(1e-6))
    return out


def downsample_2d_max(M: torch.Tensor, target_h: int, target_w: int) -> np.ndarray:
    """Mean-aggregate M to target_h × target_w bins (preserves overall 'tone')."""
    H, W = M.shape
    h_bin = max(1, H // target_h); w_bin = max(1, W // target_w)
    Ht = (H // h_bin) * h_bin; Wt = (W // w_bin) * w_bin
    M = M[:Ht, :Wt].float()
    out = M.reshape(Ht // h_bin, h_bin, Wt // w_bin, w_bin).mean(dim=(1, 3))
    return out.cpu().numpy()


def fraction_at_threshold(log_arr: torch.Tensor) -> tuple[float, float, float]:
    """Returns (% lost, % saturated, % useful) under per-block MSE-opt scaling.
    'lost' = log10(|x|/s) < log10(0.5) ≈ -0.301 (rounds to 0)
    'sat'  = log10(|x|/s) > log10(QMAX + 0.5) ≈ log10(7.5) ≈ 0.875 (saturates ceiling)
    """
    flat = log_arr.reshape(-1)
    n = flat.numel()
    lost = float((flat < np.log10(0.5)).sum()) * 100 / n
    sat  = float((flat > np.log10(QMAX + 0.5)).sum()) * 100 / n
    useful = 100.0 - lost - sat
    return lost, sat, useful


def make_figure(profile_name: str):
    profile = PROFILES[profile_name]
    SRC = profile["src"]
    LAYER_PICKS = profile["layers"]
    out_path = OUT_DIR / profile["out_name"]
    model_label = profile["model_label"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cal = torch.load(SRC, weights_only=False)

    # diverging colormap; centered at 0.
    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-2.0, vmax=1.0)

    mpl.rcParams["font.size"] = 9.5
    n_layers = len(LAYER_PICKS)

    # Grid: per layer = [banner-row (off), X-row, W-row]; no histogram row.
    height_ratios = []
    for _ in range(n_layers):
        height_ratios += [0.32, 2.0, 2.0]
    n_rows = len(height_ratios)
    n_cols = len(ROT_MODES)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.0 * n_cols + 1, 1.7 * sum(height_ratios) + 1.5),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.18,
                                          "height_ratios": height_ratios})

    def banner_row_idx(li): return 3 * li
    def x_row_idx(li):      return 3 * li + 1
    def w_row_idx(li):      return 3 * li + 2

    for li, (tag, subtitle) in enumerate(LAYER_PICKS):
        rec = cal[tag]
        W = rec["W"].float(); X = rec["X"].float()

        # Banner row
        for ci_off in range(n_cols):
            axes[banner_row_idx(li), ci_off].set_axis_off()
        axes[banner_row_idx(li), n_cols // 2].text(
            0.5, 0.5, subtitle, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#222",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="lemonchiffon",
                      edgecolor="darkred", linewidth=0.8, alpha=0.95),
            transform=axes[banner_row_idx(li), n_cols // 2].transAxes)

        for ci, rm in enumerate(ROT_MODES):
            if rm == "raw":
                X_r, W_r = X.clone(), W.clone()
            else:
                R, perm = build_R_perm(W, rm)
                X_r = apply(X, R, perm); W_r = apply(W, R, perm)

            log_X = log_outlier_X(X_r)
            log_W = log_outlier_W(W_r)
            tile_X = downsample_2d_max(log_X, target_h=80, target_w=260)
            tile_W = downsample_2d_max(log_W, target_h=80, target_w=260)
            lost_X, sat_X, _ = fraction_at_threshold(log_X)
            lost_W, sat_W, _ = fraction_at_threshold(log_W)

            # X-side heatmap
            ax = axes[x_row_idx(li), ci]
            im = ax.imshow(tile_X, aspect="auto", cmap=cmap, norm=norm,
                           interpolation="nearest")
            ax.set_title(f"X-side  lost(=0):{lost_X:.0f}%   "
                         f"saturated:{sat_X:.1f}%",
                         fontsize=9.5, color="#333", pad=4)
            if ci == 0:
                ax.set_ylabel(f"{tag}\nX-side  log₁₀(|X|/scale)", fontsize=8.5)
            ax.set_xlabel("input-channel bin (post-perm, post-rot)", fontsize=8)
            ax.tick_params(labelsize=7)
            if ci == n_cols - 1:
                cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
                cb.set_label("log₁₀(|x| / scale)", fontsize=8)
                cb.ax.tick_params(labelsize=7)
                cb.ax.axhline(np.log10(0.5),       color="darkblue", lw=0.6, alpha=0.6)
                cb.ax.axhline(np.log10(QMAX + 0.5),color="darkred",  lw=0.6, alpha=0.6)

            # W-side heatmap
            ax = axes[w_row_idx(li), ci]
            im = ax.imshow(tile_W, aspect="auto", cmap=cmap, norm=norm,
                           interpolation="nearest")
            ax.set_title(f"W-side  lost(=0):{lost_W:.0f}%   "
                         f"saturated:{sat_W:.1f}%",
                         fontsize=9.5, color="#333", pad=4)
            if ci == 0:
                ax.set_ylabel(f"{tag}\nW-side  log₁₀(|W|/scale)", fontsize=8.5)
            ax.set_xlabel("input-channel bin (post-perm, post-rot)", fontsize=8)
            ax.tick_params(labelsize=7)
            if ci == n_cols - 1:
                cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
                cb.set_label("log₁₀(|w| / scale)", fontsize=8)
                cb.ax.tick_params(labelsize=7)
                cb.ax.axhline(np.log10(0.5),       color="darkblue", lw=0.6, alpha=0.6)
                cb.ax.axhline(np.log10(QMAX + 0.5),color="darkred",  lw=0.6, alpha=0.6)

    fig.suptitle(
        f"{model_label}  —  Outlier magnitude heatmap (log₁₀(|x|/scale) under per-block MSE-optimal int4 scaling)\n"
        "Blue = crushed to int4 level 0;  white ≈ 1 int4 level;  red = saturating ceiling (log₁₀(7.5)≈0.87)",
        fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.subplots_adjust(top=0.905)

    # Column headers
    for ci, rm in enumerate(ROT_MODES):
        bbox = axes[0, ci].get_position()
        fig.text(bbox.x0 + bbox.width / 2, 0.92, ROT_TITLES[rm],
                 ha="center", va="bottom", fontsize=12.5, fontweight="bold",
                 color="#111")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot] saved → {out_path} + .pdf")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=list(PROFILES.keys()) + ["all"], default="all")
    args = p.parse_args()
    profiles = list(PROFILES.keys()) if args.profile == "all" else [args.profile]
    for prof in profiles:
        print(f"=== {prof} ===", flush=True)
        make_figure(prof)


if __name__ == "__main__":
    main()
