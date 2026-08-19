#!/usr/bin/env python
"""Bootstrap Hessian convergence: DiT vs LLM.

Tests the hypothesis that GPTQ is "cal-friendly" on DiT but cal-hungry on LLM.

Metric: relative Frobenius distance between H estimated from k subsampled
tokens vs full-sample H:
    H_k    = X_k^T · X_k / k          (X_k = k randomly-sampled tokens)
    H_full = X_full^T · X_full / N    (full = all available cal tokens)
    err(k) = ||H_k - H_full||_F / ||H_full||_F

Higher err(k) = H estimate is noisier at that cal size.

Hypothesis predicts:
  - DiT: err(k=2) < 0.5 → H stabilizes with very few cal tokens
  - LLM: err(k=2) > 1   → H still noisy, GPTQ compensation would mis-direct

Output: experiment_results/visualization_for_paper/h_stability_dit_vs_llm.png
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl


def channel_skew(X: torch.Tensor) -> float:
    """Returns max/median of per-channel max-abs, robust outlier metric."""
    amax = X.abs().amax(dim=0)
    med = amax.median()
    return float(amax.max() / max(float(med), 1e-12))


def relative_h_error(X_full: torch.Tensor, k_values: list[int], n_boot: int = 32, seed: int = 0):
    """For each k, bootstrap n_boot times: subsample k tokens, compute H_k,
    compare to H_full via relative Frobenius distance.

    Returns: (k_values, mean_err, std_err) — mean and std across bootstrap reps.
    """
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

    # Pick representative layers — same kind across modules for fair comparison
    dit_layers = ["DiT.L02.q_proj", "DiT.L02.o_proj", "DiT.L02.down_proj",
                  "DiT.L08.q_proj", "DiT.L08.down_proj",
                  "DiT.L14.q_proj", "DiT.L14.down_proj"]
    llm_layers = ["LLM.L02.q_proj", "LLM.L02.o_proj", "LLM.L02.down_proj",
                  "LLM.L05.q_proj", "LLM.L05.down_proj",
                  "LLM.L09.q_proj", "LLM.L09.down_proj"]

    # k values: token count to subsample for H estimation
    k_values = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    n_boot = 32

    # Run bootstrap
    print(f"Bootstrap relative-H error, n_boot={n_boot}, k in {k_values}")
    print()
    dit_results, llm_results = {}, {}
    for tag in dit_layers:
        X = cal[tag]["X"]
        skew = channel_skew(X)
        _, m, s = relative_h_error(X, k_values, n_boot=n_boot)
        dit_results[tag] = dict(skew=skew, means=m, stds=s)
        print(f"  {tag}: skew={skew:.1f}×  N={X.shape[0]}  err@k=2={m[0]:.3f}  err@k=64={m[5]:.3f}")
    print()
    for tag in llm_layers:
        X = cal[tag]["X"]
        skew = channel_skew(X)
        _, m, s = relative_h_error(X, k_values, n_boot=n_boot)
        llm_results[tag] = dict(skew=skew, means=m, stds=s)
        print(f"  {tag}: skew={skew:.1f}×  N={X.shape[0]}  err@k=2={m[0]:.3f}  err@k=64={m[5]:.3f}")
    print()

    # ----------- Plot -----------
    mpl.rcParams["font.size"] = 11
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)

    # Panel A: convergence curves
    ax = axes[0]
    for tag, res in dit_results.items():
        ax.plot(k_values, res["means"], "-o", color="C0", alpha=0.7,
                label="DiT" if tag == dit_layers[0] else None, markersize=4, linewidth=1.5)
    for tag, res in llm_results.items():
        # Highlight the pathological layer
        if "L02.down_proj" in tag:
            ax.plot(k_values, res["means"], "-s", color="darkred", alpha=1.0,
                    label="LLM.L02.down_proj (50000× outlier)", markersize=7, linewidth=2.2)
        else:
            ax.plot(k_values, res["means"], "-o", color="C3", alpha=0.5,
                    label="LLM (other)" if tag == llm_layers[0] else None,
                    markersize=4, linewidth=1.5)
    ax.axhline(0.1, color="gray", linestyle=":", linewidth=1, label="10% relative err")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="100% (H_k ≈ noise)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("# cal tokens used (k)")
    ax.set_ylabel(r"$\|H_k - H_{\mathrm{full}}\|_F\ /\ \|H_{\mathrm{full}}\|_F$")
    ax.set_title("Hessian estimate convergence vs cal size\n(bootstrap, n_boot=32, X from object cal)")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)

    # Add k=2 annotation
    ax.axvline(2, color="orange", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(2.1, 1.5, "k=2", color="orange", fontsize=9, va="bottom")

    # Panel B: scatter of channel-skew vs err@k=2 (the punchline)
    ax = axes[1]
    for tag, res in dit_results.items():
        ax.scatter(res["skew"], res["means"][0], color="C0", s=60, alpha=0.8,
                   label="DiT" if tag == dit_layers[0] else None, edgecolors="black", linewidth=0.5)
    for tag, res in llm_results.items():
        if "L02.down_proj" in tag:
            ax.scatter(res["skew"], res["means"][0], color="darkred", marker="s", s=120,
                       label="LLM.L02.down_proj", edgecolors="black", linewidth=0.8)
            ax.annotate("layer.2.down_proj\n(50000× pathology)",
                        (res["skew"], res["means"][0]),
                        xytext=(-110, -25), textcoords="offset points",
                        fontsize=9, color="darkred",
                        arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
        else:
            ax.scatter(res["skew"], res["means"][0], color="C3", s=60, alpha=0.6,
                       label="LLM (other)" if tag == llm_layers[0] else None,
                       edgecolors="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("channel skew  max(|X|) / median(|X|)")
    ax.set_ylabel(r"err($H_{k=2}$, $H_{\mathrm{full}}$) (relative Frobenius)")
    ax.set_title("Channel-skew → Hessian-estimation difficulty\n(harder outlier ↔ harder cal)")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)

    fig.suptitle("DiT vs LLM: Why cal=2 GPTQ works on DiT but not on LLM",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    out_path = out_dir / "h_stability_dit_vs_llm.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[plot] saved → {out_path}")

    # ----------- Numeric table -----------
    print()
    print("=" * 90)
    print("Numeric summary: relative-H error at key cal sizes")
    print("=" * 90)
    print(f"{'layer':<22}  {'skew':>10}  {'err@k=2':>10}  {'err@k=8':>10}  {'err@k=32':>10}  {'err@k=128':>11}")
    print("-" * 90)
    for tag, res in {**dit_results, **llm_results}.items():
        print(f"{tag:<22}  {res['skew']:>10.1f}  {res['means'][0]:>10.3f}  "
              f"{res['means'][2]:>10.3f}  {res['means'][4]:>10.3f}  {res['means'][6]:>11.3f}")


if __name__ == "__main__":
    main()
