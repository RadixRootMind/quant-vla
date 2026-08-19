#!/usr/bin/env python
"""Per-channel max-abs distribution after rotation — gr00t + pi0.5 combined.

For each of 4 sampled layers and 3 rotation methods, plot per-input-channel
max(|X·R|) across the 8 diffusion steps × all tokens. Two views:

  - x = channel index (in current post-perm, post-rot order), y = max(|X·R|).
    Shows the "spike pattern" of outliers: raw has tall isolated spikes at the
    outlier channels; rotation either flattens them (Hadamard mixing) or
    redistributes them (DuQuant SVD).

  - inset / second panel: sorted (descending) per-channel amax — pure
    "outlier-distribution tail". Steep tail = heavy outliers (bad).
    Flat tail = uniform amplitude (good). Easy cross-method comparison.

Layout: 4 panels (1 per layer), each with 3 rotation curves.
"""
from __future__ import annotations
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gr00t.quantization.duquant_preprocess import compute_duquant_rotation_only


OUT_DIR = Path("experiment_results/visualization_for_paper")
BLOCK = 64

LAYERS = [
    ("gr00t  LLM.L02.down_proj  —  pathology",
     "experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep2.pt",
     "LLM.L02.down_proj"),
    ("gr00t  LLM.L02.gate_proj  —  typical MLP",
     "experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep2.pt",
     "LLM.L02.gate_proj"),
    ("pi0.5  paligemma.L14.down_proj  —  pathology analog",
     "experiment_results/svd_diagnostics_gr00t/xw_dump_pi05_object_n4.pt",
     "paligemma.L14.down_proj"),
    ("pi0.5  paligemma.L02.gate_proj  —  typical MLP",
     "experiment_results/svd_diagnostics_gr00t/xw_dump_pi05_object_n4.pt",
     "paligemma.L02.gate_proj"),
]

ROT_MODES = ["raw", "svd", "svd_hadamard"]
ROT_TITLES = {
    "raw":          "raw (no rotation)",
    "svd":          "DuQuant SVD",
    "svd_hadamard": "SVD-Hadamard (A2-lite)",
}
ROT_COLORS = {
    "raw":          "#888888",
    "svd":          "#1f77b4",
    "svd_hadamard": "#d62728",
}


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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump_cache: dict = {}

    # Compute per-channel amax for each (layer, rotation).
    amax_data = {}    # (li, rm) -> np.array [in_features]
    for li, (label, src, tag) in enumerate(LAYERS):
        if src not in dump_cache:
            print(f"[load] {src}")
            dump_cache[src] = torch.load(src, weights_only=False)
        rec = dump_cache[src][tag]
        W = rec["W"].float(); X = rec["X"].float()
        for rm in ROT_MODES:
            if rm == "raw":
                X_r = X.clone()
            else:
                R, perm = build_R_perm(W, rm)
                X_r = apply(X, R, perm)
            amax = X_r.abs().amax(dim=0).cpu().float().numpy()
            amax_data[(li, rm)] = amax

    mpl.rcParams["font.size"] = 10
    n_layers = len(LAYERS)
    n_rot = len(ROT_MODES)
    BIN = 5    # group channels in bins of 5 along the x-axis; each bar = max within bin
    fig, axes = plt.subplots(n_layers, n_rot, figsize=(5.0 * n_rot, 3.0 * n_layers),
                             sharey="row")

    def bin_max(arr: np.ndarray, bin_size: int) -> np.ndarray:
        n = len(arr)
        n_full = (n // bin_size) * bin_size
        head = arr[:n_full].reshape(-1, bin_size).max(axis=1)
        tail = np.array([arr[n_full:].max()]) if n > n_full else np.empty(0)
        return np.concatenate([head, tail])

    for li, (label, _, tag) in enumerate(LAYERS):
        amax_row = [amax_data[(li, rm)] for rm in ROT_MODES]
        binned_row = [bin_max(a, BIN) for a in amax_row]
        y_max = max(b.max() for b in binned_row) * 1.1
        n_bins = len(binned_row[0])

        for ci, rm in enumerate(ROT_MODES):
            ax = axes[li, ci]
            amax = amax_data[(li, rm)]
            binned = binned_row[ci]
            x = np.arange(len(binned))
            ax.bar(x, binned, width=1.0, color=ROT_COLORS[rm], alpha=0.92,
                   edgecolor="none")
            skew = amax.max() / max(np.median(amax), 1e-12)
            stats = f"skew={skew:.1f}×   max={amax.max():.1f}   med={np.median(amax):.2f}"
            ax.set_title(f"{ROT_TITLES[rm]}\n{stats}", fontsize=10,
                         fontweight="bold" if rm == "raw" else "normal",
                         color="#222" if rm == "raw" else "#333")
            ax.set_xlim(0, len(binned))
            ax.set_ylim(0, y_max)
            ax.grid(True, axis="y", alpha=0.25)
            if ci == 0:
                ax.set_ylabel(f"{tag}\nbin max(|X·R|)  (bin = {BIN} channels)", fontsize=9)
            if li == n_layers - 1:
                ax.set_xlabel(f"channel-bin index  (post-perm, post-rot;  bin = {BIN} channels)",
                              fontsize=9)
            ax.tick_params(labelsize=8)

    fig.suptitle(f"Per-channel max(|X·R|), binned by groups of {BIN} (bar = max within bin)\n"
                 "X = input-channel-bin index in current rotation's order.  Tall spikes = outlier groups.",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout()

    out_path = OUT_DIR / "activation_outlier_bars.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot] saved → {out_path} + .pdf")

    # Also print summary
    print()
    print(f"{'layer':<48}  {'raw':>10}  {'SVD':>10}  {'SVD-H':>10}")
    print("-" * 90)
    for li, (label, _, tag) in enumerate(LAYERS):
        s = {rm: amax_data[(li, rm)].max() / max(np.median(amax_data[(li, rm)]), 1e-12)
             for rm in ROT_MODES}
        print(f"{tag:<48}  {s['raw']:>10.1f}  {s['svd']:>10.1f}  {s['svd_hadamard']:>10.1f}")


if __name__ == "__main__":
    main()
