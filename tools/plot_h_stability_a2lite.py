#!/usr/bin/env python
"""Bootstrap Hessian convergence on **A2-lite-rotated** X — the actual GPTQ input.

Pipeline we *always* run (per §1 of methodology.md):
    1. A2-lite rotation: X_r = X @ R, R = block-diag(svd_hadamard, block=64) with weight-zigzag perm
    2. Then optionally: SmoothQuant scale on X_r, SpQR bypass on H_r, GPTQ on rotated W.

GPTQ's Hessian (and SpQR's damage map) are computed on H_r = X_r^T X_r / N,
NOT on raw X. So cal-sensitivity should be measured on H_r.

Hypothesis (user, refined):
  - DiT: A2-lite alone already makes H_r cal-cheap → cal=2 ≈ cal=full
  - LLM: A2-lite enough for non-down_proj layers
         layer.2.down_proj's structural outlier survives even A2-lite + cal-scaling
         → needs SQ / SpQR on top, NOT just more cal
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


def build_a2lite_rotation(W: torch.Tensor, block: int = 64) -> torch.Tensor:
    """A2-lite rotation R ∈ ℝ^{in × in} from FP weight (no act_stats).
    Equivalent to what build_gptq_weights.py does with
    --duquant-rot-mode svd_hadamard --duquant-permute --duquant-block-size 64.
    """
    R, perm = compute_duquant_rotation_only(
        W, block_size=block, enable_permute=True,
        lambda_smooth=0.15, rot_mode="svd_hadamard",
    )
    if perm is not None:
        P = torch.zeros((W.shape[1], W.shape[1]), dtype=torch.float32)
        perm_t = torch.from_numpy(perm.astype("int64"))
        P[perm_t, torch.arange(W.shape[1])] = 1.0
        R = P @ R
    return R


def channel_skew(X: torch.Tensor) -> float:
    amax = X.abs().amax(dim=0)
    med = amax.median()
    return float(amax.max() / max(float(med), 1e-12))


def relative_h_error(X_full: torch.Tensor, k_values: list[int], n_boot: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    N, C = X_full.shape
    Xf = X_full.float()
    H_full = Xf.T @ Xf / N
    H_full_fro = float(torch.linalg.norm(H_full).item())

    means, stds = [], []
    for k in k_values:
        if k > N:
            means.append(np.nan); stds.append(np.nan); continue
        errs = []
        for _ in range(n_boot):
            idx = rng.choice(N, size=k, replace=False)
            X_k = Xf[idx]
            H_k = X_k.T @ X_k / k
            err = float(torch.linalg.norm(H_k - H_full).item()) / max(H_full_fro, 1e-12)
            errs.append(err)
        means.append(float(np.mean(errs)))
        stds.append(float(np.std(errs)))
    return k_values, means, stds


def main():
    src = Path("experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep2.pt")
    out_dir = Path("experiment_results/visualization_for_paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = torch.load(src, weights_only=False)

    dit_layers = ["DiT.L02.q_proj", "DiT.L02.o_proj", "DiT.L02.down_proj",
                  "DiT.L08.q_proj", "DiT.L08.down_proj",
                  "DiT.L14.q_proj", "DiT.L14.down_proj"]
    llm_layers = ["LLM.L02.q_proj", "LLM.L02.o_proj", "LLM.L02.down_proj",
                  "LLM.L05.q_proj", "LLM.L05.down_proj",
                  "LLM.L09.q_proj", "LLM.L09.down_proj"]

    k_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    n_boot = 32

    # Compute A2-lite rotation per layer, transform X, then bootstrap H on rotated X.
    print(f"{'layer':<22}  {'skew_raw':>10}  {'skew_a2':>10}  {'err@k=2':>10}  {'err@k=8':>10}  {'err@k=64':>10}  {'err@k=256':>10}")
    print("-" * 100)
    dit_results, llm_results = {}, {}
    for tag in dit_layers + llm_layers:
        rec = cal[tag]
        W = rec["W"].float()
        X = rec["X"].float()
        skew_raw = channel_skew(X)
        # A2-lite rotation
        R = build_a2lite_rotation(W, block=64).float()
        X_r = X @ R
        skew_a2 = channel_skew(X_r)
        _, m, s = relative_h_error(X_r, k_values, n_boot=n_boot)
        d = dict(skew_raw=skew_raw, skew_a2=skew_a2, means=m, stds=s)
        if tag.startswith("DiT"):
            dit_results[tag] = d
        else:
            llm_results[tag] = d
        print(f"{tag:<22}  {skew_raw:>10.1f}  {skew_a2:>10.1f}  {m[0]:>10.3f}  {m[2]:>10.3f}  {m[5]:>10.3f}  {m[7]:>10.3f}")

    # ----------- Plot -----------
    mpl.rcParams["font.size"] = 11
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    # ---- Panel A: H convergence after A2-lite ----
    ax = axes[0]
    for tag, res in dit_results.items():
        ax.plot(k_values, res["means"], "-o", color="C0", alpha=0.7,
                label="DiT (after A2-lite)" if tag == dit_layers[0] else None,
                markersize=4, linewidth=1.5)
    for tag, res in llm_results.items():
        if "L02.down_proj" in tag:
            ax.plot(k_values, res["means"], "-s", color="darkred", alpha=1.0,
                    label="LLM.L02.down_proj (after A2-lite, still pathological)",
                    markersize=7, linewidth=2.2, zorder=10)
        elif "down_proj" in tag:
            ax.plot(k_values, res["means"], "--o", color="C3", alpha=0.7,
                    label="LLM other down_proj (after A2-lite)" if "L05" in tag else None,
                    markersize=4, linewidth=1.5)
        else:
            ax.plot(k_values, res["means"], "-o", color="orange", alpha=0.55,
                    label="LLM q/k/v/o/gate (after A2-lite)" if tag == llm_layers[0] else None,
                    markersize=4, linewidth=1.5)
    ax.axhline(0.1, color="gray", linestyle=":", linewidth=1)
    ax.text(1.05, 0.107, "10% rel err (GPTQ usable)", color="gray", fontsize=8, va="bottom")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.text(1.05, 1.05, "100% (H≈noise)", color="gray", fontsize=8, va="bottom")
    # cal=2 region
    ax.axvspan(2, 16, alpha=0.08, color="orange")
    ax.text(3.2, 0.015, "cal=2 typical token range\n(~400 DiT / ~800 LLM tokens via\nstep-bucket / full-seq pooling)",
            fontsize=8, color="darkorange", ha="left")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("# cal tokens used (k)")
    ax.set_ylabel(r"$\|H_k - H_{\mathrm{full}}\|_F\ /\ \|H_{\mathrm{full}}\|_F$ on A2-lite rotated X")
    ax.set_title("After A2-lite rotation: H convergence vs cal size\n(A2-lite is ALWAYS the first step; this is what GPTQ actually sees)")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)

    # ---- Panel B: skew before vs after A2-lite ----
    ax = axes[1]
    # Plot raw skew (left) and post-A2 skew (right) as paired bars per layer.
    all_layers = dit_layers + llm_layers
    raw_skews = [(*[
        (dit_results[t]["skew_raw"] if t in dit_results else llm_results[t]["skew_raw"])
        for t in [tag]],) for tag in all_layers]
    a2_skews = [(dit_results[t]["skew_a2"] if t in dit_results else llm_results[t]["skew_a2"])
                 for t in all_layers]
    raw_skews = [(dit_results[t]["skew_raw"] if t in dit_results else llm_results[t]["skew_raw"])
                  for t in all_layers]
    short_labels = [t.replace("DiT.", "D.").replace("LLM.", "L.").replace("_proj", "") for t in all_layers]
    n = len(all_layers)
    x = np.arange(n)
    width = 0.38
    colors_raw = ["C0" if t.startswith("DiT") else ("darkred" if "L02.down" in t else ("C3" if "down" in t else "orange")) for t in all_layers]
    colors_a2 = colors_raw
    ax.bar(x - width/2, raw_skews, width, color=colors_raw, alpha=0.45, label="raw X")
    ax.bar(x + width/2, a2_skews, width, color=colors_a2, alpha=0.9, hatch="//", edgecolor="black", linewidth=0.4, label="A2-lite rotated X")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("channel skew  max(|X|) / median(|X|)")
    ax.set_title("Channel-skew reduction by A2-lite rotation\n(√64 ≈ 8× expected; layer.2.down_proj resists)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    # Annotation for the pathological layer
    for i, tag in enumerate(all_layers):
        if "LLM.L02.down" in tag:
            ax.annotate(f"50000× → {a2_skews[i]:.0f}×\n(still pathological)",
                        (i + width/2, a2_skews[i]),
                        xytext=(-90, 25), textcoords="offset points",
                        fontsize=8.5, color="darkred",
                        arrowprops=dict(arrowstyle="->", color="darkred"))
            break

    fig.suptitle("A2-lite is the fixed first step. Then compare GPTQ / SQ / SpQR on top.\n"
                 "(LLM.L02.down_proj's pathology survives A2-lite — needs SQ + SpQR, not more cal)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "h_stability_a2lite_dit_vs_llm.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[plot] saved → {out_path}")


if __name__ == "__main__":
    main()
