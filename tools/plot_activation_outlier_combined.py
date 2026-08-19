#!/usr/bin/env python
"""Activation outlier distribution heatmap — combined gr00t + pi0.5 figure.

Plots |X · R| directly (no quant-error / MSE / per-block-scale division) so
the reader sees the raw outlier distribution after each rotation pipeline.

Channel axis is BINNED BY block_size=64 — the actual int4 quantization block
unit. Each output column = one quant block, value = max(|X·R|) within that
block. Token axis unchanged (one row per token). This is the natural
"per-block outlier severity" view: the value in each cell is what would
drive that block's per-token int4 scale.

Color = log10(per-block max |X · R|).  Shared color scale within each
layer-row (vmin/vmax over 3 rotations) so cross-method contrast reads as the
rotation effect.

Layout (single figure, X-side only — W removed per request):
  4 layer-rows × 3 rotation-cols.
    Row 1: gr00t  LLM.L02.down_proj         (structural pathology)
    Row 2: gr00t  LLM.L02.gate_proj         (typical MLP)
    Row 3: pi0.5  paligemma.L14.down_proj   (pathology analog)
    Row 4: pi0.5  paligemma.L02.gate_proj   (typical MLP)
  Cols: raw  /  DuQuant SVD  /  SVD-Hadamard (A2-lite)
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gr00t.quantization.duquant_preprocess import compute_duquant_rotation_only


OUT_DIR = Path("experiment_results/visualization_for_paper")
BLOCK = 64

# (display_label, dump_path, layer_tag)
LAYERS = [
    ("gr00t  LLM.L02.down_proj  —  pathology  (raw channel-skew 48000×)",
     "experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep2.pt",
     "LLM.L02.down_proj"),
    ("gr00t  LLM.L02.gate_proj  —  typical MLP  (raw skew 59×)",
     "experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep2.pt",
     "LLM.L02.gate_proj"),
    ("pi0.5  paligemma.L14.down_proj  —  pathology analog  (raw skew 156×)",
     "experiment_results/svd_diagnostics_gr00t/xw_dump_pi05_object_n4.pt",
     "paligemma.L14.down_proj"),
    ("pi0.5  paligemma.L02.gate_proj  —  typical MLP  (raw skew 12.4×)",
     "experiment_results/svd_diagnostics_gr00t/xw_dump_pi05_object_n4.pt",
     "paligemma.L02.gate_proj"),
]

ROT_MODES = ["raw", "svd", "svd_hadamard"]
ROT_TITLES = {
    "raw":           "original",
    "svd":           "svd only",
    "svd_hadamard":  "svd hadamard",
}
# Model label spans the rows that correspond to that model (heatmap-row indices).
MODEL_LABELS = [("gr00t", 0, 1), ("pi0.5", 2, 3)]


def build_R_perm(W: torch.Tensor, rot_mode: str):
    R, perm = compute_duquant_rotation_only(
        W.cpu().float(), block_size=BLOCK, enable_permute=True,
        lambda_smooth=0.15, rot_mode=rot_mode,
    )
    return R.float(), perm


def apply(M: torch.Tensor, R: torch.Tensor, perm) -> torch.Tensor:
    Mc = M.float()
    if perm is not None:
        perm_t = torch.from_numpy(perm.astype("int64"))
        Mc = Mc.index_select(-1, perm_t)
    return Mc @ R


QMAX = 7


def block_nmse(M: torch.Tensor, block_size: int = BLOCK) -> np.ndarray:
    """For each (token, block), return per-block normalized MSE under int4
    symmetric quant with MSE-optimal per-(token, block) scale.

        nMSE(t, b) = || X_tb - Q4(X_tb) ||² / || X_tb ||²

    Shape: [n_tokens, n_blocks].  Returns the actual quant error each block
    pays — directly tied to the W4A4 cost we care about.
    """
    H, W = M.shape
    n_blocks = W // block_size
    M_trim = M[:, : n_blocks * block_size].float()
    Xb = M_trim.reshape(H, n_blocks, block_size)     # [tokens, blocks, B]
    amax = Xb.abs().amax(dim=2).clamp_min(1e-12)     # [tokens, blocks]
    # 12-pt MSE-optimal alpha grid per (token, block)
    alphas = torch.linspace(0.5, 1.2, 12).to(Xb.device)
    best_err = None
    for alpha in alphas:
        scale = (alpha * amax / QMAX).unsqueeze(-1)  # [tokens, blocks, 1]
        q = (Xb / scale).round().clamp(-QMAX - 1, QMAX) * scale
        err = ((Xb - q) ** 2).sum(dim=2)             # [tokens, blocks]
        best_err = err if best_err is None else torch.minimum(best_err, err)
    den = (Xb ** 2).sum(dim=2).clamp_min(1e-12)
    nmse = (best_err / den)
    return nmse.cpu().numpy()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cache loaded dumps so we don't reload the same .pt multiple times.
    dump_cache: dict[str, dict] = {}
    def get_dump(path):
        if path not in dump_cache:
            print(f"[load] {path}")
            dump_cache[path] = torch.load(path, weights_only=False)
        return dump_cache[path]

    # Compute all 16 (4 layers × 4 rotations) tiles up front so we can pick
    # a per-row shared color scale.
    print("[compute] applying rotations and downsampling...")
    tiles = {}    # (li, rot_mode) -> numpy tile
    row_vmin = [None] * len(LAYERS)
    row_vmax = [None] * len(LAYERS)
    for li, (label, src, tag) in enumerate(LAYERS):
        cal = get_dump(src)
        rec = cal[tag]
        W = rec["W"].float(); X = rec["X"].float()
        for rm in ROT_MODES:
            if rm == "raw":
                X_r = X.clone()
            else:
                R, perm = build_R_perm(W, rm)
                X_r = apply(X, R, perm)
            tile = block_nmse(X_r, block_size=BLOCK)
            tiles[(li, rm)] = tile
            v_lo = max(tile[tile > 0].min() if (tile > 0).any() else 1e-6, 1e-6)
            v_hi = tile.max()
            row_vmin[li] = v_lo if row_vmin[li] is None else min(row_vmin[li], v_lo)
            row_vmax[li] = v_hi if row_vmax[li] is None else max(row_vmax[li], v_hi)

    print("[plot] rendering...")
    mpl.rcParams["font.size"] = 9.5
    cmap = plt.get_cmap("inferno_r")

    n_rows = len(LAYERS)
    n_cols = len(ROT_MODES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols + 0.9, 2.1 * n_rows + 0.8),
                             gridspec_kw={"hspace": 0.10, "wspace": 0.08})

    for li, (_, _, tag) in enumerate(LAYERS):
        all_vals = np.concatenate([tiles[(li, rm)].ravel() for rm in ROT_MODES])
        norm = Normalize(vmin=0.0, vmax=float(np.quantile(all_vals, 0.99)))

        for ci, rm in enumerate(ROT_MODES):
            ax = axes[li, ci]
            tile = tiles[(li, rm)]
            im = ax.imshow(tile, aspect="auto", cmap=cmap, norm=norm,
                           interpolation="nearest")
            ax.tick_params(labelsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if li == 0:
                ax.set_title(ROT_TITLES[rm], fontsize=11,
                             fontweight="normal", color="#222", pad=6,
                             fontfamily="DejaVu Sans")
            if ci == n_cols - 1:
                cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.012)
                cb.ax.tick_params(labelsize=6.5)

    fig.subplots_adjust(left=0.06, right=0.96, top=0.93, bottom=0.03,
                        hspace=0.10, wspace=0.08)
    for model, r0, r1 in MODEL_LABELS:
        bbox_top    = axes[r0, 0].get_position()
        bbox_bottom = axes[r1, 0].get_position()
        y_center = (bbox_top.y1 + bbox_bottom.y0) / 2
        fig.text(0.018, y_center, model, ha="center", va="center",
                 fontsize=13, fontweight="normal", color="#222",
                 rotation=90, fontfamily="DejaVu Sans")

    out_path = OUT_DIR / "activation_outlier_combined.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot] saved → {out_path} + .pdf")
    plt.close(fig)

    # Summary numbers (channel skew per rotation, per layer)
    print()
    print(f"{'layer':<48}  {'raw skew':>10}  {'SVD':>8}  {'SVD-H':>8}")
    print("-" * 85)
    for li, (label, src, tag) in enumerate(LAYERS):
        cal = get_dump(src)
        rec = cal[tag]
        W = rec["W"].float(); X = rec["X"].float()
        skews = []
        for rm in ROT_MODES:
            if rm == "raw":
                X_r = X.clone()
            else:
                R, perm = build_R_perm(W, rm)
                X_r = apply(X, R, perm)
            amax = X_r.abs().amax(dim=0).cpu().float().numpy()
            sk = amax.max() / max(np.median(amax), 1e-12)
            skews.append(sk)
        short = tag if len(tag) < 45 else tag[:42] + "..."
        print(f"{short:<48}  {skews[0]:>10.1f}  {skews[1]:>8.1f}  {skews[2]:>8.1f}")


if __name__ == "__main__":
    main()
