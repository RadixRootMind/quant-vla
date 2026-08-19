#!/usr/bin/env python
"""Per-step act_scale necessity — 2-panel visualization.

Empirical companion to methodology.md §4. Built on the existing per-step
X dump `xw_dump_object_cal_perstep2.pt` (9 DiT layers × 8 diffusion steps ×
392 tokens × in_features, captured pre-quantization on FP cal forward).

Panel 1 (left) — q999 drift across the 8 diffusion steps:
  For each layer × step, compute the median across channels of the per-channel
  q999. This is the same metric driving the per-step act_scale_table (q999 /
  qmax(4) per channel per step). Shows the 1.1–1.3× monotonic drift from t=0
  (noise-dominated input) to t=7 (action-signal-dominated input).

Panel 2 (right) — per-step int4 quant MSE under two scaling schemes:
  - **per-step scale** (what E2 actually uses): for each step t, scale[t] =
    q999(X_t) / qmax. Quant MSE stays flat and low across all 8 steps.
  - **single-bucket scale** (E2-noPS / our mean-collapse experiment): scale =
    q999(mean over 8 steps) / qmax, same scale used for all t. MSE V-shapes:
    elevated at steps whose q999 deviates most from the mean.

Together: panel 1 = "the magnitude actually drifts", panel 2 = "the drift
materializes as quant MSE waste when you collapse to a single bucket".
Direct empirical motivation for the per-step act_scale_table.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl


SRC = "experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep2.pt"
SRC_FFNET0 = "experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep_ffnet0.pt"
OUT_DIR = Path("experiment_results/visualization_for_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)
QMAX = 7  # symmetric int4
NUM_STEPS = 8

# 12 sampled DiT layers grouped by kind.
# Order chosen so paired "input" vs "output" colours are adjacent in legend.
LAYER_GROUPS = [
    # === residual-stream inputs (q/k/v share input; ff.net.0 is MLP input) ===
    ("q_proj  (attn input,  q/k/v shared X)",
     ["DiT.L02.q_proj",    "DiT.L08.q_proj",    "DiT.L14.q_proj"],   "#1f77b4"),
    ("ff_net_0  (MLP input)",
     ["DiT.L02.ff_net_0",  "DiT.L08.ff_net_0",  "DiT.L14.ff_net_0"], "#9467bd"),
    # === post-normalize / post-gate outputs ===
    ("o_proj  (attn output, post-softmax)",
     ["DiT.L02.o_proj",    "DiT.L08.o_proj",    "DiT.L14.o_proj"],   "#2ca02c"),
    ("down_proj  (MLP output, post-GLU)",
     ["DiT.L02.down_proj", "DiT.L08.down_proj", "DiT.L14.down_proj"], "#d62728"),
]


def q999_per_channel(X: torch.Tensor) -> torch.Tensor:
    """Per-channel 99.9 percentile (matches the act_scale_table construction)."""
    return torch.quantile(X.abs(), 0.999, dim=0).clamp_min(1e-12)


def median_q999(X: torch.Tensor) -> float:
    """Median across channels of per-channel q999 — single scalar per layer-step."""
    return float(q999_per_channel(X).median())


def quant_mse_under_scale(X: torch.Tensor, scale: torch.Tensor) -> float:
    """Per-element symmetric int4 quant relative MSE under given per-channel scale.
       scale: [in_features].
    """
    s = scale.unsqueeze(0).clamp_min(1e-12)
    q = (X / s).round().clamp(-QMAX - 1, QMAX) * s
    num = ((X - q) ** 2).sum()
    den = (X ** 2).sum().clamp_min(1e-12)
    return float(num / den)


def quant_abs_mse_under_scale(X: torch.Tensor, scale: torch.Tensor) -> float:
    """Per-element absolute MSE (no normalization by ||X||²) under given scale."""
    s = scale.unsqueeze(0).clamp_min(1e-12)
    q = (X / s).round().clamp(-QMAX - 1, QMAX) * s
    return float(((X - q) ** 2).mean())


def main():
    print(f"[perstep-necessity] loading {SRC}")
    cal = torch.load(SRC, weights_only=False)
    if Path(SRC_FFNET0).exists():
        print(f"[perstep-necessity] merging  {SRC_FFNET0}")
        cal_extra = torch.load(SRC_FFNET0, weights_only=False)
        cal.update(cal_extra)
    else:
        print(f"[perstep-necessity][warn] {SRC_FFNET0} missing; ff_net_0 lines will be skipped")

    # ---- compute per-layer per-step quantities ----
    # results[layer_tag] = {
    #   "q999":         np.ndarray [num_steps]  median q999 per step
    #   "mse_perstep":  np.ndarray [num_steps]  quant MSE using its own per-step scale
    #   "mse_singleb":  np.ndarray [num_steps]  quant MSE using single-bucket (mean) scale
    # }
    results = {}
    for _, layer_list, _ in LAYER_GROUPS:
        for tag in layer_list:
            if tag not in cal:
                print(f"[skip] {tag} not in dump")
                continue
            rec = cal[tag]
            xps = rec["X_per_step"]
            # Per-step per-channel q999 → per-step scale tables
            scales_per_step = []
            for t in range(NUM_STEPS):
                X_t = xps[t].float()
                scales_per_step.append(q999_per_channel(X_t) / QMAX)   # [in_features]
            mean_scale = torch.stack(scales_per_step, dim=0).mean(dim=0)   # [in_features]

            q999_med = np.array([median_q999(xps[t].float()) for t in range(NUM_STEPS)])
            mse_perstep = np.array([
                quant_mse_under_scale(xps[t].float(), scales_per_step[t])
                for t in range(NUM_STEPS)
            ])
            mse_singleb = np.array([
                quant_mse_under_scale(xps[t].float(), mean_scale)
                for t in range(NUM_STEPS)
            ])
            abs_perstep = np.array([
                quant_abs_mse_under_scale(xps[t].float(), scales_per_step[t])
                for t in range(NUM_STEPS)
            ])
            abs_singleb = np.array([
                quant_abs_mse_under_scale(xps[t].float(), mean_scale)
                for t in range(NUM_STEPS)
            ])
            results[tag] = dict(
                q999=q999_med,
                mse_perstep=mse_perstep,
                mse_singleb=mse_singleb,
                abs_perstep=abs_perstep,
                abs_singleb=abs_singleb,
            )

    # ---- print summary numbers ----
    print()
    print("=" * 96)
    print(f"{'layer':<24}  {'q999 max/min':>14}  {'mse_perstep mean':>17}  "
          f"{'mse_singleb mean':>17}  {'gap (×)':>9}")
    print("-" * 96)
    for _, layer_list, _ in LAYER_GROUPS:
        for tag in layer_list:
            if tag not in results: continue
            r = results[tag]
            q_ratio = r["q999"].max() / r["q999"].min()
            ps_mean = r["mse_perstep"].mean()
            sb_mean = r["mse_singleb"].mean()
            gap = sb_mean / max(ps_mean, 1e-12)
            print(f"{tag:<24}  {q_ratio:>14.3f}  {ps_mean:>17.5f}  {sb_mean:>17.5f}  {gap:>9.2f}×")

    # ---- plot ----
    mpl.rcParams["font.size"] = 11
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    steps = np.arange(NUM_STEPS)

    # ===== Panel 1: q999 drift =====
    # 3 lines (one per kind) + shading for layer-depth range
    ax = axes[0]
    for kind, layer_list, color in LAYER_GROUPS:
        # Stack the 3 depths and normalize each to its own t=0
        avail = [t for t in layer_list if t in results]
        if not avail:
            continue
        ratios = np.stack([results[t]["q999"] / results[t]["q999"][0]
                            for t in avail], axis=0)                  # [n_depths, 8 steps]
        mean = ratios.mean(axis=0)
        lo, hi = ratios.min(axis=0), ratios.max(axis=0)
        ax.fill_between(steps, lo, hi, color=color, alpha=0.16)
        ax.plot(steps, mean, "-o", color=color, lw=2.2, ms=6, alpha=0.95,
                label=kind)
    ax.axhline(1.0, color="black", lw=0.6, alpha=0.4)
    ax.set_xlabel("diffusion step t  (0 = noise input, 7 = near action signal)", fontsize=11)
    ax.set_ylabel("median q999  (normalized to t=0)", fontsize=11)
    ax.set_title("Panel 1 — q999 drift across 8 diffusion steps\n"
                 "Monotone decreasing; early-block q_proj drops ~17–20% by t=7",
                 fontweight="bold", fontsize=12)
    ax.set_xticks(steps)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              fontsize=9, framealpha=0.95)
    ax.set_ylim(0.75, 1.04)

    # ===== Panel 2: quant MSE per step, per-step vs single-bucket =====
    # Aggregate to per-kind mean across the 3 layer-depths so we get 3
    # paired-line groups instead of 18 lines.
    ax = axes[1]
    for kind, layer_list, color in LAYER_GROUPS:
        avail = [t for t in layer_list if t in results]
        if not avail:
            continue
        ps = np.stack([results[t]["mse_perstep"] for t in avail], axis=0).mean(axis=0)
        sb = np.stack([results[t]["mse_singleb"] for t in avail], axis=0).mean(axis=0)
        # Shade the gap (waste from no-perstep)
        ax.fill_between(steps, ps, sb, color=color, alpha=0.16)
        ax.plot(steps, ps, "-o", color=color, lw=1.4, ms=4.5, alpha=0.75,
                label=f"{kind}  per-step scale")
        ax.plot(steps, sb, "-s", color=color, lw=2.4, ms=6, alpha=0.95,
                label=f"{kind}  single-bucket")
    ax.set_xlabel("diffusion step t", fontsize=11)
    ax.set_ylabel("int4 quant relative MSE  $\\|X - Q_4(X)\\|_F^2 / \\|X\\|_F^2$", fontsize=11)
    ax.set_title("Panel 2 — Per-step vs single-bucket int4 quant MSE\n"
                 "Shaded gap = waste from collapsing 8 step-scales into one bucket",
                 fontweight="bold", fontsize=12)
    ax.set_xticks(steps)
    ax.grid(True, axis="y", alpha=0.3)
    # Place the legend BELOW the axes so it doesn't sit on top of any line.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              fontsize=9, framealpha=0.95)
    ax.set_yscale("log")

    # Annotate the average gap percent at the right edge of panel 2
    all_gap_pct = []
    for _, layer_list, _ in LAYER_GROUPS:
        for tag in layer_list:
            if tag not in results: continue
            r = results[tag]
            all_gap_pct.append((r["mse_singleb"] - r["mse_perstep"]) / r["mse_perstep"] * 100)
    mean_gap = float(np.mean(np.stack(all_gap_pct)))
    print(f"\n[note] mean MSE gap (all kinds × depths × steps) = {mean_gap:.2f}%  "
          "— interpretation lives in docs/visualization_guide.md, not on the figure")

    fig.suptitle("Per-step act_scale necessity — empirical motivation (gr00t-N1.5 DiT, libero_object cal)",
                 fontsize=13.5, fontweight="bold", y=1.005)
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    out_path = OUT_DIR / "perstep_necessity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\n[plot] saved → {out_path} + .pdf")

    # ===== Alternate view: per-step int4 ABSOLUTE MSE (no /||X||²), LINEAR y =====
    # Same data as panel 2 but with unnormalized MSE so the per-kind magnitude
    # difference (driven by activation amplitude) is visible. Linear y makes
    # the per-step vs single-bucket gap obvious for the kinds with larger
    # absolute residuals (down_proj, o_proj).
    fig2, ax = plt.subplots(figsize=(9, 5.5))
    for kind, layer_list, color in LAYER_GROUPS:
        avail = [t for t in layer_list if t in results]
        if not avail: continue
        ps_abs = np.stack([results[t]["abs_perstep"] for t in avail], axis=0).mean(axis=0)
        sb_abs = np.stack([results[t]["abs_singleb"] for t in avail], axis=0).mean(axis=0)
        ax.fill_between(steps, ps_abs, sb_abs, color=color, alpha=0.16)
        ax.plot(steps, ps_abs, "-o", color=color, lw=1.4, ms=4.5, alpha=0.75,
                label=f"{kind}  per-step scale")
        ax.plot(steps, sb_abs, "-s", color=color, lw=2.4, ms=6, alpha=0.95,
                label=f"{kind}  single-bucket")
    ax.set_xlabel("diffusion step t", fontsize=11)
    ax.set_ylabel(r"absolute int4 quant MSE  mean($(X - Q_4(X))^2$)", fontsize=11)
    ax.set_title("Per-step vs single-bucket int4 quant MSE — ABSOLUTE units, LINEAR y\n"
                 "Shaded gap = waste from collapsing 8 step-scales into one bucket",
                 fontweight="bold", fontsize=11.5)
    ax.set_xticks(steps)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              fontsize=9, framealpha=0.95)
    fig2.suptitle("Per-step act_scale — absolute MSE view (gr00t-N1.5 DiT, libero_object cal)",
                  fontsize=13, fontweight="bold", y=1.005)
    fig2.tight_layout(rect=[0, 0.04, 1, 1])

    out_path2 = OUT_DIR / "perstep_necessity_abs.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    fig2.savefig(out_path2.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot] saved → {out_path2} + .pdf")

    # Print absolute-MSE summary
    print()
    print("=" * 96)
    print(f"{'layer':<24}  {'abs_perstep mean':>17}  {'abs_singleb mean':>17}  {'gap (abs)':>11}")
    print("-" * 96)
    for _, layer_list, _ in LAYER_GROUPS:
        for tag in layer_list:
            if tag not in results: continue
            r = results[tag]
            ps = r["abs_perstep"].mean(); sb = r["abs_singleb"].mean()
            print(f"{tag:<24}  {ps:>17.5e}  {sb:>17.5e}  {sb - ps:>+11.3e}")


if __name__ == "__main__":
    main()
