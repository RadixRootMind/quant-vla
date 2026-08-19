#!/usr/bin/env python
"""Per-block W4 quantization difficulty: DuQuant-SVD vs A2-lite (svd-Hadamard).

Metric (block-wise, MSE-optimal int4 scale):
    diff(b) = || x_b - Q4_MSE(x_b) ||_2 / || x_b ||_2

Why this metric (vs raw |X|/|W| length):
  * Each int4 block gets its own MSE-optimal symmetric scale → raw amplitude
    cannot be compared across blocks (a big value in one block does not affect
    quantization of other blocks).
  * Both methods use the SAME zigzag permutation (lambda_smooth=0.15, weight
    energy) — so the metric must be computed AFTER permutation (i.e. on the
    rearranged columns) to be a fair comparison.
  * diff(b) directly reflects how much energy each block loses under W4 — the
    real engineering quantity that drives layer-level quant error.

Comparison axes:
  * Rotation method: DuQuant SVD (rot_mode='svd')  vs  A2-lite (rot_mode='svd_hadamard')
  * Models:           gr00t-N1.5                    vs  pi0.5
  * Modules:          gr00t LLM, gr00t DiT          vs  pi0.5 paligemma, pi0.5 expert
  * Sides:            activations X                 vs  weights W

For X (per-channel quant): block-wise MSE-optimal scale per channel within block;
err computed by treating the [N, B] block as one batch and quantizing column-wise.
For W (per-block-per-row int4 quant): block-wise MSE-optimal scale per row;
err computed row-wise then averaged.

Output: experiment_results/visualization_for_paper/block_quant_difficulty.png
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gr00t.quantization.duquant_preprocess import compute_duquant_rotation_only


SRC_GR00T = "experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep2.pt"
SRC_PI05  = "experiment_results/svd_diagnostics_gr00t/xw_dump_pi05_object_n4.pt"
OUT_DIR = Path("experiment_results/visualization_for_paper")
CACHE   = OUT_DIR / "block_quant_difficulty_results.pt"
BLOCK = 64
QMAX = 7  # symmetric int4


def build_rotation(W: torch.Tensor, rot_mode: str, block: int = BLOCK):
    """Compute (R, perm) for given rotation mode. Both modes use SAME perm."""
    W_cpu = W.detach().cpu().float()
    R, perm = compute_duquant_rotation_only(
        W_cpu, block_size=block, enable_permute=True,
        lambda_smooth=0.15, rot_mode=rot_mode,
    )
    return R.float(), perm


def apply_rot_perm(X_or_W_cols: torch.Tensor, R: torch.Tensor, perm) -> torch.Tensor:
    """X_or_W_cols has its input dim along the last axis.
    Apply permutation first (reorder columns), then rotation (block-diag).
    Result is in the rotated, permuted order.
    """
    M = X_or_W_cols.float()
    if perm is not None:
        perm_t = torch.from_numpy(perm.astype("int64"))
        M = M.index_select(-1, perm_t)
    # R is already in the permuted order (build_rotation applied perm internally
    # when picking columns). So just apply R.
    return M @ R


def block_quant_diff_X(X_r: torch.Tensor, block: int = BLOCK) -> np.ndarray:
    """For each block of `block` channels along last axis of X_r, compute
    MSE-optimal symmetric int4 (per-channel) quant rel-err.
    Returns per-block scalar = mean over channels in that block.
    """
    N, C = X_r.shape
    n_blocks = (C + block - 1) // block
    out = []
    for b in range(n_blocks):
        s, e = b * block, min((b + 1) * block, C)
        xb = X_r[:, s:e].float()        # [N, B]
        # Per-channel MSE-optimal symmetric scale → grid search on max-based scales
        amax = xb.abs().amax(dim=0).clamp_min(1e-12)   # [B]
        # MSE-optimal: try 12 candidates s_i = alpha_i * (amax / QMAX), pick min MSE
        alphas = torch.linspace(0.5, 1.2, 12)
        best_err = None
        for alpha in alphas:
            scale = (alpha * amax / QMAX).clamp_min(1e-12)        # [B]
            q = (xb / scale).round().clamp(-QMAX - 1, QMAX) * scale
            err = ((xb - q) ** 2).sum(dim=0) / (xb ** 2).sum(dim=0).clamp_min(1e-12)  # [B]
            if best_err is None:
                best_err = err.clone()
            else:
                best_err = torch.minimum(best_err, err)
        out.append(float(best_err.sqrt().mean()))   # sqrt → relative L2 err
    return np.asarray(out)


def block_quant_diff_W(W_r: torch.Tensor, block: int = BLOCK) -> np.ndarray:
    """For each input-dim block of W_r [out, in], compute per-row MSE-optimal
    symmetric int4 quant rel-err. Block scalar = mean over rows.
    """
    O, I = W_r.shape
    n_blocks = (I + block - 1) // block
    out = []
    for b in range(n_blocks):
        s, e = b * block, min((b + 1) * block, I)
        Wb = W_r[:, s:e].float()        # [O, B]
        amax = Wb.abs().amax(dim=1).clamp_min(1e-12)   # [O]
        alphas = torch.linspace(0.5, 1.2, 12)
        best_err = None
        for alpha in alphas:
            scale = (alpha * amax / QMAX).unsqueeze(1).clamp_min(1e-12)   # [O, 1]
            q = (Wb / scale).round().clamp(-QMAX - 1, QMAX) * scale
            num = ((Wb - q) ** 2).sum(dim=1)
            den = (Wb ** 2).sum(dim=1).clamp_min(1e-12)
            err = num / den   # [O]
            if best_err is None:
                best_err = err.clone()
            else:
                best_err = torch.minimum(best_err, err)
        out.append(float(best_err.sqrt().mean()))
    return np.asarray(out)


def stat_layer(rec, rot_mode: str):
    """Return dict with per-block X / W quant-diff arrays under given rot mode."""
    W = rec["W"].float()
    X = rec["X"].float()
    R, perm = build_rotation(W, rot_mode=rot_mode, block=BLOCK)
    X_r = apply_rot_perm(X, R, perm)
    W_r = apply_rot_perm(W, R, perm)
    return {
        "X_diff": block_quant_diff_X(X_r),
        "W_diff": block_quant_diff_W(W_r),
        "X_diff_raw": block_quant_diff_X(X),   # also raw (no rotation, no perm) for context
        "W_diff_raw": block_quant_diff_W(W),
    }


def collect_all(src_path: str) -> dict:
    cal = torch.load(src_path, weights_only=False)
    out = {}
    for name, rec in cal.items():
        module = name.split(".")[0]
        layer_idx = int(name.split(".L")[1].split(".")[0])
        kind = name.split(".")[-1]
        t0 = time.time()
        d_svd = stat_layer(rec, "svd")
        d_a2  = stat_layer(rec, "svd_hadamard")
        dt = time.time() - t0
        out[(module, layer_idx, kind)] = {"svd": d_svd, "svd_hadamard": d_a2}
        # Print summary numbers (median per-block diff)
        print(f"  {module}.L{layer_idx:02d}.{kind:<10}  "
              f"X svd p50={np.median(d_svd['X_diff']):.3f}  a2={np.median(d_a2['X_diff']):.3f}  "
              f"W svd p50={np.median(d_svd['W_diff']):.3f}  a2={np.median(d_a2['W_diff']):.3f}  "
              f"({dt:.1f}s)", flush=True)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        print(f"[cache] {CACHE}")
        results = torch.load(CACHE, weights_only=False)
    else:
        results = {}
        for src in [SRC_GR00T, SRC_PI05]:
            print(f"\n=== {src}")
            results.update(collect_all(src))
        torch.save(results, CACHE)
        print(f"[cache] saved {CACHE}")

    # ------------- Plot -------------
    mpl.rcParams["font.size"] = 10
    fig, axes = plt.subplots(4, 2, figsize=(17, 17))

    method_color = {"svd": "#1f77b4", "svd_hadamard": "#d62728"}
    method_label = {"svd": "DuQuant SVD", "svd_hadamard": "A2-lite (svd-Hadamard)"}
    raw_color = "#888888"

    module_titles = {
        "LLM":       "gr00t-N1.5  LLM (eagle)",
        "DiT":       "gr00t-N1.5  DiT",
        "paligemma": "pi0.5  paligemma",
        "expert":    "pi0.5  expert / action head",
    }
    row_order = ["LLM", "DiT", "paligemma", "expert"]

    for row_idx, module in enumerate(row_order):
        # Collect rows for this module, sorted by (layer, kind)
        rows = sorted([k for k in results if k[0] == module],
                      key=lambda k: (k[1], k[2]))
        if not rows:
            for col in (0, 1):
                axes[row_idx, col].axis("off")
                axes[row_idx, col].set_title(module_titles[module] + "  (no data)")
            continue
        labels = [f"L{i:02d}.{kd.replace('_proj','')}" for (_, i, kd) in rows]
        x = np.arange(len(rows))

        for col, side in enumerate(["X_diff", "W_diff"]):
            ax = axes[row_idx, col]
            # Per-layer: 3 bars (raw / DuQuant SVD / A2-lite) showing the 90th
            # percentile of per-block diff. 90th highlights worst blocks (the
            # ones that drive layer-level quant error).
            width = 0.27
            ys_raw  = [float(np.percentile(results[r]["svd"][side.replace("_diff", "_diff_raw")], 90))
                       for r in rows]
            ys_svd  = [float(np.percentile(results[r]["svd"][side], 90)) for r in rows]
            ys_a2   = [float(np.percentile(results[r]["svd_hadamard"][side], 90)) for r in rows]
            ax.bar(x - width, ys_raw, width, label="raw (no rot)",       color=raw_color, alpha=0.85,
                   edgecolor="black", linewidth=0.3)
            ax.bar(x,         ys_svd, width, label=method_label["svd"],  color=method_color["svd"], alpha=0.92,
                   edgecolor="black", linewidth=0.3)
            ax.bar(x + width, ys_a2,  width, label=method_label["svd_hadamard"], color=method_color["svd_hadamard"],
                   alpha=0.92, edgecolor="black", linewidth=0.3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_yscale("log")
            ax.set_ylabel(r"per-block W4 rel-err (p90 over blocks)")
            side_name = "Activation (X)" if side == "X_diff" else "Weight (W)"
            ax.set_title(f"{module_titles[module]}  —  {side_name}", fontweight="bold", fontsize=10.5)
            ax.grid(True, axis="y", which="both", alpha=0.25)
            ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
            # Annotate worst layer under raw
            worst_i = int(np.argmax(ys_raw))
            ax.annotate(f"worst raw:\n{labels[worst_i]}={ys_raw[worst_i]:.3f}",
                        xy=(worst_i - width, ys_raw[worst_i]),
                        xytext=(15, 25), textcoords="offset points",
                        fontsize=8, color="darkred",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lemonchiffon", alpha=0.85),
                        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.7))

    fig.suptitle("Per-block W4 quantization difficulty (p90 of MSE-optimal int4 rel-err)\n"
                 "DuQuant SVD vs A2-lite (svd-Hadamard) — both with weight-energy zigzag permutation, block=64",
                 fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout()
    out_path = OUT_DIR / "block_quant_difficulty.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\n[plot] saved → {out_path} + .pdf")

    # -------------- Numeric summary --------------
    print("\n" + "=" * 110)
    print("Summary: median rel-err across blocks per (module, kind), p90 → max difficulty")
    print("=" * 110)
    print(f"{'module.kind':<22}  "
          f"{'X raw p90':>10}  {'X svd p90':>10}  {'X a2 p90':>10}  {'a2/svd':>8}  ||  "
          f"{'W raw p90':>10}  {'W svd p90':>10}  {'W a2 p90':>10}  {'a2/svd':>8}")
    print("-" * 130)
    sums = defaultdict(list)
    for (mod, lidx, kd), d in sorted(results.items()):
        x_raw = float(np.percentile(d["svd"]["X_diff_raw"], 90))
        x_svd = float(np.percentile(d["svd"]["X_diff"], 90))
        x_a2  = float(np.percentile(d["svd_hadamard"]["X_diff"], 90))
        w_raw = float(np.percentile(d["svd"]["W_diff_raw"], 90))
        w_svd = float(np.percentile(d["svd"]["W_diff"], 90))
        w_a2  = float(np.percentile(d["svd_hadamard"]["W_diff"], 90))
        sums[(mod, kd)].append((x_raw, x_svd, x_a2, w_raw, w_svd, w_a2))
        print(f"{mod}.L{lidx:02d}.{kd:<10}  "
              f"{x_raw:>10.4f}  {x_svd:>10.4f}  {x_a2:>10.4f}  {x_a2/x_svd:>8.2f}  ||  "
              f"{w_raw:>10.4f}  {w_svd:>10.4f}  {w_a2:>10.4f}  {w_a2/w_svd:>8.2f}")


if __name__ == "__main__":
    main()
