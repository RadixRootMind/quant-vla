#!/usr/bin/env python
"""Int4-level utilization heatmap: DuQuant SVD vs SVD-Hadamard (A2-lite).

Mimics the style of int4_levels_a0_vs_a2lite_L02_down_proj.png — that figure
is visually the most legible because it shows the *discrete int4 level* each
value lands on after block-wise MSE-optimal scaling, so the eye can directly
read "this block is mostly under-quantized (black, level=0)" vs. "this block
is using its int4 budget well (mid-yellow, levels 2–5)" vs. "this block is
saturating (bright, level=7)".

Color = int4 level magnitude |round(x / s_block)| clipped to [0, 7].
- 0 (black)       = signal lost (value rounded to zero, block scale too big)
- 1–5 (warm hues) = well-utilized int4 grid
- 6–7 (bright)    = saturating against the int4 ceiling

This already accounts for (a) per-block MSE-optimal scale, (b) permutation,
(c) rotation — exactly what was missing from raw-amplitude heatmaps.

Layout: 4 heatmap rows × 3 cols (raw / DuQuant SVD / SVD-Hadamard) + a final
row of level-distribution histograms.

  Row 1: LLM.L02.down_proj  X-side   (pathology)
  Row 2: LLM.L02.down_proj  W-side
  Row 3: LLM.L05.o_proj     X-side   (typical attn layer, big SVD-H win)
  Row 4: LLM.L05.o_proj     W-side
  Row 5: bottom 3 panels — level histograms (red = X-side, blue = W-side) for
         row 3's layer (typical) so the user can read the level distribution
         under each rotation at a glance.

Output: experiment_results/visualization_for_paper/heatmap_svd_vs_svdh.png + .pdf
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
from matplotlib import cm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gr00t.quantization.duquant_preprocess import compute_duquant_rotation_only


OUT_DIR = Path("experiment_results/visualization_for_paper")
BLOCK = 64
QMAX = 7        # symmetric int4

# Per-model layer picks + output filename
PROFILES = {
    "gr00t": {
        "src": "experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep2.pt",
        "layers": [
            # Pathology: raw amax-ratio 48000×; even SVD-H only brings it to
            # 11400×. Rotation alone cannot fix it — empirical motivation for
            # SQ+SpQR overlay on LLM down_proj.
            ("LLM.L02.down_proj", "LLM.L02.down_proj  —  structural pathology  (raw amax-ratio 48000× ; SVD-H still 11400×)"),
            # Typical MLP layer where SVD-only doesn't help (amax-ratio
            # 59× → 55.6×) and SVD-H clearly wins (→ 14.3×). Matches the
            # 5.5× SVD-vs-SVD-H gap in the W4A4 nMSE diagnostic CSV.
            ("LLM.L02.gate_proj", "LLM.L02.gate_proj  —  typical MLP layer  (SVD-only leaves amax-ratio 56×, SVD-H drops it to 14×)"),
        ],
        "out_name": "heatmap_svd_vs_svdh.png",
        "model_label": "gr00t-N1.5",
    },
    "pi05": {
        "src": "experiment_results/svd_diagnostics_gr00t/xw_dump_pi05_object_n4.pt",
        "layers": [
            # Pathology side: rotation makes things WORSE in nMSE under SVD-only
            # (raw 0.116 → SVD 0.357), partially recovered by SVD-H (0.149).
            # Analog of gr00t LLM.L02.down_proj.
            ("paligemma.L14.down_proj", "paligemma.L14.down_proj  —  pathology  (X channel-skew 156×; pure-SVD HURTS nMSE 3×, SVD-H partial recovery)"),
            # Typical attn layer where SVD-H wins big on actual W4A4 nMSE.
            # SVD-only: amax-ratio stays 12.6×; SVD-H: drops to 5.2× (2.4× extra)
            ("paligemma.L02.gate_proj", "paligemma.L02.gate_proj  —  typical MLP layer  (SVD-only leaves amax-ratio 12.6×, SVD-H flattens to 5.2×)"),
        ],
        "out_name": "heatmap_svd_vs_svdh_pi05.png",
        "model_label": "pi0.5",
    },
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
    """Return per-row best alpha in [0.5, 1.2] that minimizes per-row MSE for
    a (xb: [O or N, B]) block with per-row amax. For 2D xb, returns [O].
    """
    alphas = torch.linspace(0.5, 1.2, 12)
    best_err = None
    best_alpha = None
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


def int4_levels_W(W_r: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Per-element int4 level magnitude (|round(w/s)| clipped to [0, 7]) under
    per-row per-block MSE-optimal scaling. Shape: same as W_r."""
    O, I = W_r.shape
    n_blocks = (I + block - 1) // block
    lvl = torch.zeros_like(W_r, dtype=torch.int8)
    for b in range(n_blocks):
        s, e = b * block, min((b + 1) * block, I)
        Wb = W_r[:, s:e].float()
        amax = Wb.abs().amax(dim=1).clamp_min(1e-12)
        alpha = _mse_optimal_alpha(Wb, amax)
        scale = (alpha * amax / QMAX).clamp_min(1e-12).unsqueeze(1)
        q = (Wb / scale).round().clamp(-QMAX - 1, QMAX)
        lvl[:, s:e] = q.abs().clamp(0, QMAX).to(torch.int8)
    return lvl


def int4_levels_X(X_r: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """Per-element int4 level magnitude under PER-TOKEN PER-BLOCK MSE-optimal
    scaling — i.e., for each (token t, input-block b), pick the MSE-optimal
    symmetric int4 scale based on max(|X_r[t, block_b]|).

    This is the scaling regime where rotation actually matters: a heavy outlier
    in one channel of a block forces the whole block's scale wider, pushing
    every other channel in that block toward int4 level 0. Hadamard mixing
    inside the block flattens that amax → fewer values lost.

    Per-channel scaling (what we used before) hides this signal because each
    channel gets its own scale independent of block neighbours. Real W4A4
    quantizers that use online per-token quant DO see this; that's why we want
    the heatmap to track it.
    """
    N, I = X_r.shape
    Xf = X_r.float()
    n_blocks = (I + block - 1) // block
    lvl = torch.zeros_like(Xf, dtype=torch.int8)
    alphas = torch.linspace(0.5, 1.2, 12)
    for b in range(n_blocks):
        s, e = b * block, min((b + 1) * block, I)
        Xb = Xf[:, s:e]                              # [N, B]
        amax = Xb.abs().amax(dim=1).clamp_min(1e-12) # [N]   (per-token)
        # MSE-optimal alpha per token within this block
        best_err = None; best_alpha = None
        for alpha in alphas:
            scale = (alpha * amax / QMAX).clamp_min(1e-12).unsqueeze(1)
            q = (Xb / scale).round().clamp(-QMAX - 1, QMAX) * scale
            err = ((Xb - q) ** 2).sum(dim=1)
            if best_err is None:
                best_err = err.clone(); best_alpha = torch.full_like(err, float(alpha))
            else:
                mask = err < best_err
                best_err = torch.where(mask, err, best_err)
                best_alpha = torch.where(mask, torch.full_like(err, float(alpha)), best_alpha)
        scale = (best_alpha * amax / QMAX).clamp_min(1e-12).unsqueeze(1)
        q = (Xb / scale).round().clamp(-QMAX - 1, QMAX).abs().clamp(0, QMAX)
        lvl[:, s:e] = q.to(torch.int8)
    return lvl


def downsample_fraction_lost(M: torch.Tensor, target_h: int, target_w: int) -> np.ndarray:
    """For each tile, return the FRACTION of values that round to int4 level 0
    (i.e., are 'lost' because the block scale dwarfed them).

    Semantics:
      0%   (bright) — every value in tile uses ≥1 int4 level → bits well-spent
      50%          — half the tile is rounding to zero
      100% (black) — entire tile is under-quantized

    This is the most directly visible signature of "rotation didn't equalize
    blocks": a heavy outlier in a few channels inflates that block's scale,
    forcing the other channels in the same block to round to zero. Hadamard
    mixing flattens block amax → fewer values lost.
    """
    H, W = M.shape
    h_bin = max(1, H // target_h); w_bin = max(1, W // target_w)
    Ht = (H // h_bin) * h_bin; Wt = (W // w_bin) * w_bin
    M = M[:Ht, :Wt]
    is_lost = (M == 0).to(torch.float32)
    out = is_lost.reshape(Ht // h_bin, h_bin, Wt // w_bin, w_bin).mean(dim=(1, 3))
    return out.cpu().numpy() * 100.0   # percent


def level_stats(lvl: torch.Tensor) -> dict:
    flat = lvl.reshape(-1).to(torch.int64)
    n = flat.numel()
    pct = {k: float((flat == k).sum()) * 100 / n for k in range(QMAX + 1)}
    return {
        "lost":   pct[0],
        "sat":    pct[QMAX],
        "useful": sum(pct[k] for k in range(1, QMAX)),
        "hist":   np.bincount(flat.cpu().numpy(), minlength=QMAX + 1) / n,
    }


def make_figure(profile_name: str):
    profile = PROFILES[profile_name]
    SRC = profile["src"]
    LAYER_PICKS = profile["layers"]
    out_path_png = OUT_DIR / profile["out_name"]
    model_label = profile["model_label"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cal = torch.load(SRC, weights_only=False)

    rot_modes = ["raw", "svd", "svd_hadamard"]
    rot_titles = {"raw": "raw (no rotation)",
                  "svd": "DuQuant SVD",
                  "svd_hadamard": "SVD-Hadamard (A2-lite)"}

    # Reversed inferno: per-tile % of values at int4 level 0 ('lost').
    # 0% lost = bright yellow (rotation balanced the block), 100% = black.
    cmap = plt.get_cmap("inferno_r")
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=0.0, vmax=100.0)

    mpl.rcParams["font.size"] = 10
    n_layers = len(LAYER_PICKS)

    # Grid layout per layer: [banner-row (off), X-row, W-row], plus final histogram row.
    # banner-row has its full row dedicated to a centered fig.text label (axes
    # turned off). This avoids text overlap with column headers and adjacent
    # row set_titles.
    height_ratios = []
    for _ in range(n_layers):
        height_ratios += [0.32, 2.0, 2.0]   # banner / X / W
    height_ratios += [1.6]                   # histogram row
    n_rows = len(height_ratios)

    fig, axes = plt.subplots(n_rows, 3, figsize=(16, 1.6 * sum(height_ratios) + 1.5),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.22,
                                          "height_ratios": height_ratios})

    def banner_row_idx(li): return 3 * li
    def x_row_idx(li):      return 3 * li + 1
    def w_row_idx(li):      return 3 * li + 2
    hist_row_idx = 3 * n_layers

    # We'll show the histogram row for the *typical* layer (L05.o_proj),
    # since that's where the rotation-method difference is sharpest.
    hist_layer_idx = 1

    last_level_stats = {}  # for histogram row

    for li, (tag, subtitle) in enumerate(LAYER_PICKS):
        rec = cal[tag]
        W = rec["W"].float()
        X = rec["X"].float()
        in_f = W.shape[1]

        # Banner row: occupy full width via the center subplot; turn the others off.
        for ci_off in (0, 1, 2):
            ax_b = axes[banner_row_idx(li), ci_off]
            ax_b.set_axis_off()
        axes[banner_row_idx(li), 1].text(
            0.5, 0.5, subtitle, ha="center", va="center",
            fontsize=12, fontweight="bold", color="#222",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="lemonchiffon",
                      edgecolor="darkred", linewidth=0.8, alpha=0.95),
            transform=axes[banner_row_idx(li), 1].transAxes)

        for ci, rm in enumerate(rot_modes):
            if rm == "raw":
                X_r, W_r = X.clone(), W.clone()
            else:
                R, perm = build_R_perm(W, rm)
                X_r = apply(X, R, perm)
                W_r = apply(W, R, perm)

            lvl_X = int4_levels_X(X_r)             # [N, in_f]
            lvl_W = int4_levels_W(W_r)             # [out, in_f]

            tile_X = downsample_fraction_lost(lvl_X, target_h=80, target_w=240)
            tile_W = downsample_fraction_lost(lvl_W, target_h=80, target_w=240)

            stats_X = level_stats(lvl_X)
            stats_W = level_stats(lvl_W)
            if li == hist_layer_idx:
                last_level_stats[("X", rm)] = stats_X
                last_level_stats[("W", rm)] = stats_W

            # ---- X-side heatmap
            ax = axes[x_row_idx(li), ci]
            im = ax.imshow(tile_X, aspect="auto", cmap=cmap, norm=norm,
                           interpolation="nearest")
            ax.set_xticks([])
            ax.tick_params(labelsize=7)
            stat_line = (f"X-side  lost(=0):{stats_X['lost']:.0f}%   "
                         f"saturated(=7):{stats_X['sat']:.1f}%   "
                         f"useful(1–6):{stats_X['useful']:.0f}%")
            ax.set_title(stat_line, fontsize=9.5, color="#333", pad=4)
            if ci == 0:
                ax.set_ylabel(f"{tag}\nX-side int4 levels", fontsize=8.5)
            if ci == 2:
                cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01,
                                  ticks=[0,25,50,75,100])
                cb.set_label("% values lost (level=0) per tile", fontsize=8)
                cb.ax.tick_params(labelsize=7)

            # ---- W-side heatmap
            ax = axes[w_row_idx(li), ci]
            im = ax.imshow(tile_W, aspect="auto", cmap=cmap, norm=norm,
                           interpolation="nearest")
            ax.tick_params(labelsize=7)
            stat_line = (f"W-side  lost(=0):{stats_W['lost']:.0f}%   "
                         f"saturated(=7):{stats_W['sat']:.1f}%   "
                         f"useful(1–6):{stats_W['useful']:.0f}%")
            ax.set_title(stat_line, fontsize=9, color="#333", pad=4)
            if ci == 0:
                ax.set_ylabel(f"{tag}\nW-side int4 levels", fontsize=8.5)
            ax.set_xlabel("input-channel bin (post-perm, post-rot)", fontsize=8)
            if ci == 2:
                cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01,
                                  ticks=[0,25,50,75,100])
                cb.set_label("% values lost (level=0) per tile", fontsize=8)
                cb.ax.tick_params(labelsize=7)

        # Layer subtitle is embedded into the X-row center-column set_title above;
        # no separate banner needed.

    # ---------- Bottom row: level histograms ----------
    for ci, rm in enumerate(rot_modes):
        ax = axes[hist_row_idx, ci]
        s_X = last_level_stats[("X", rm)]
        s_W = last_level_stats[("W", rm)]
        x = np.arange(QMAX + 1)
        w = 0.4
        ax.bar(x - w/2, s_X["hist"] * 100, width=w, color="tab:red", alpha=0.85, label="X side")
        ax.bar(x + w/2, s_W["hist"] * 100, width=w, color="tab:blue", alpha=0.85, label="W side")
        ax.set_xticks(x)
        ax.set_xlabel("int4 level magnitude", fontsize=9)
        ax.set_ylabel("% of values" if ci == 0 else "", fontsize=9)
        ax.set_title(f"{rot_titles[rm]}  —  level distribution\n"
                     f"({LAYER_PICKS[hist_layer_idx][0]})",
                     fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, axis="y", alpha=0.3)
        # Annotate "lost" and "saturated" rates
        ax.text(0.5, max(s_X["hist"][0], s_W["hist"][0]) * 100 + 1.5,
                f"lost\n(=0)", ha="center", fontsize=7, color="#333")
        ax.text(QMAX, max(s_X["hist"][QMAX], s_W["hist"][QMAX]) * 100 + 1.5,
                f"sat\n(=7)", ha="center", fontsize=7, color="#333")
        ax.set_ylim(0, max(s_X["hist"].max(), s_W["hist"].max()) * 100 * 1.25)

    fig.suptitle(f"{model_label}  —  DuQuant SVD vs SVD-Hadamard (A2-lite)  —  int4-utilization heatmap\n"
                 "Color = % of values 'lost' (rounded to int4 level 0) per tile, after per-block MSE-optimal scaling.\n"
                 "Dark stripes = outlier blocks throwing away most int4 levels; uniform yellow = bits well-spent.",
                 fontsize=12.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    # Column headers (rotation method) placed just below the suptitle, above
    # the first banner row. Compute y from the figure-coords of row 0 (banner).
    bbox0 = axes[0, 0].get_position()
    header_y = min(0.945, bbox0.y1 + 0.045)
    for ci, rm in enumerate(rot_modes):
        bbox = axes[0, ci].get_position()
        fig.text(bbox.x0 + bbox.width / 2, header_y, rot_titles[rm],
                 ha="center", va="bottom", fontsize=12.5, fontweight="bold",
                 color="#111")

    fig.savefig(out_path_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_path_png.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot] saved → {out_path_png} + .pdf")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=list(PROFILES.keys()), default="all")
    args = p.parse_args()
    profiles = list(PROFILES.keys()) if args.profile == "all" else [args.profile]
    for prof in profiles:
        print(f"=== Generating heatmap for profile: {prof} ===", flush=True)
        make_figure(prof)


if __name__ == "__main__":
    main()
