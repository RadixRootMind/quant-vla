#!/usr/bin/env python
"""Cross-model outlier comparison: gr00t-N1.5 vs pi0.5.

Question: does pi0.5 have an analog of gr00t LLM.L02.down_proj's 50000×
structural pathology, or is it uniformly cal-friendly after A2-lite?

Uses pre-aggregated per-channel act_max / q999 (no raw X needed).
Channel skew = max-abs across channels / median-abs.
Estimates post-A2-lite skew via the √64 = 8× rule (Hadamard upper bound;
actual reduction usually 2-4× as we measured on gr00t).
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl


GR00T_STATS = "duquant_act_stats/object_q999.pt"
PI05_STATS = {
    "object": "duquant_act_stats/pi05_libero_object_q999.pt",
    "spatial": "duquant_act_stats/pi05_libero_spatial_q999.pt",
    "goal":    "duquant_act_stats/pi05_libero_goal_q999.pt",
    "long":    "duquant_act_stats/pi05_libero_10_q999.pt",
}
OUT_DIR = Path("experiment_results/visualization_for_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def stats_for_layer(rec):
    """Return (max/med channel skew, max value)."""
    v = rec.get("act_max", rec.get("q999"))
    if v is None:
        return None, None
    a = np.abs(v.cpu().float().numpy())
    med = max(float(np.median(a)), 1e-12)
    return float(a.max() / med), float(a.max())


def module_kind_gr00t(name: str):
    """gr00t LLM layer kind."""
    for k in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
        if name.endswith(k):
            return k
    return None


def module_kind_pi05(name: str):
    """pi0.5 module + kind."""
    mod = "paligemma" if "paligemma_with_expert.paligemma" in name else (
          "expert"    if "paligemma_with_expert.gemma_expert" in name else "?")
    kind = module_kind_gr00t(name)
    return mod, kind


def collect_gr00t():
    d = torch.load(GR00T_STATS, weights_only=False, map_location="cpu")
    out = []  # list of (model, module, kind, layer_idx, skew)
    for name, rec in d.items():
        kind = module_kind_gr00t(name)
        if kind is None:
            continue
        layer_idx = int(name.split(".layers.")[1].split(".")[0])
        skew, mx = stats_for_layer(rec)
        out.append(("gr00t LLM", "eagle", kind, layer_idx, skew, mx))
    return out


def collect_pi05(suite="object"):
    d = torch.load(PI05_STATS[suite], weights_only=False, map_location="cpu")
    out = []
    for name, rec in d.items():
        mod, kind = module_kind_pi05(name)
        if kind is None or mod == "?":
            continue
        layer_idx = int(name.split(".layers.")[1].split(".")[0])
        skew, mx = stats_for_layer(rec)
        out.append(("pi0.5 " + mod, mod, kind, layer_idx, skew, mx))
    return out


def main():
    gr_records = collect_gr00t()
    pi_records = collect_pi05("object")  # object suite for consistent reference (skew is stable across suites per §pi05 diagnostic)

    # ---------- Combined per-kind summary ----------
    print(f"{'model/module':<22}  {'kind':<12}  {'#L':>4}  {'median skew':>12}  {'max skew':>12}  {'worst layer.kind':<30}")
    print("-" * 110)
    grp = defaultdict(list)
    for model, mod, kind, idx, skew, mx in gr_records + pi_records:
        grp[(model, kind)].append((skew, idx))
    for (model, kind), rows in sorted(grp.items()):
        skews = [s for s, i in rows]
        worst_i = rows[int(np.argmax(skews))][1]
        med = float(np.median(skews))
        mxs = float(np.max(skews))
        print(f"{model:<22}  {kind:<12}  {len(rows):>4}  {med:>12.1f}  {mxs:>12.1f}  layer.{worst_i:>2}.{kind:<10}")

    # ---------- Plot: per-layer skew bars, grouped by module ----------
    mpl.rcParams["font.size"] = 11
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=False)

    # Pre-compute estimates
    def by_module_kind(records, model_filter):
        out = defaultdict(dict)   # out[kind][layer_idx] = skew
        for model, mod, kind, idx, skew, mx in records:
            if model_filter(model):
                out[kind][idx] = skew
        return out

    gr_data = by_module_kind(gr_records, lambda m: "gr00t" in m)
    pi_pali = by_module_kind(pi_records, lambda m: m == "pi0.5 paligemma")
    pi_expert = by_module_kind(pi_records, lambda m: m == "pi0.5 expert")

    kinds = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    kind_colors = {
        "q_proj": "#1f77b4", "k_proj": "#1f77b4", "v_proj": "#1f77b4", "o_proj": "#2ca02c",
        "gate_proj": "#ff7f0e", "up_proj": "#ff7f0e", "down_proj": "#d62728",
    }

    # --- (0,0) gr00t LLM eagle ---
    ax = axes[0, 0]
    for kind in kinds:
        ds = gr_data.get(kind, {})
        if not ds:
            continue
        xs = sorted(ds.keys())
        ys = [ds[i] for i in xs]
        ax.plot(xs, ys, "-o", color=kind_colors[kind], label=kind, alpha=0.85, markersize=5, linewidth=1.5)
    ax.set_yscale("log")
    ax.axhline(50000, color="darkred", ls=":", alpha=0.7)
    ax.text(0.5, 60000, "L02.down_proj pathology (50000×)", color="darkred", fontsize=9)
    ax.axhline(100, color="gray", ls=":", alpha=0.5)
    ax.text(0.5, 110, "100× (mild-severe boundary)", color="gray", fontsize=8)
    ax.set_xlabel("layer idx")
    ax.set_ylabel("channel skew  max(|X|) / median(|X|)")
    ax.set_title("gr00t-N1.5  LLM (eagle, 12 layers)\nLayer.2.down_proj = 50000× pathology")
    ax.set_xticks(range(12))
    ax.legend(ncol=3, fontsize=8.5, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(1, 1e5)

    # --- (0,1) pi0.5 paligemma ---
    ax = axes[0, 1]
    for kind in kinds:
        ds = pi_pali.get(kind, {})
        if not ds:
            continue
        xs = sorted(ds.keys())
        ys = [ds[i] for i in xs]
        ax.plot(xs, ys, "-o", color=kind_colors[kind], label=kind, alpha=0.85, markersize=5, linewidth=1.5)
    ax.set_yscale("log")
    ax.axhline(50000, color="darkred", ls=":", alpha=0.5)
    ax.text(0.5, 60000, "(gr00t L02.down pathology line)", color="darkred", fontsize=8.5)
    ax.axhline(100, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("layer idx")
    ax.set_ylabel("channel skew  max / median")
    ax.set_title("pi0.5  paligemma (18 layers)\nWorst: layer.16.down_proj ≈ 2050×")
    ax.set_xticks(range(18))
    ax.legend(ncol=3, fontsize=8.5, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(1, 1e5)

    # --- (1,0) pi0.5 expert ---
    ax = axes[1, 0]
    for kind in kinds:
        ds = pi_expert.get(kind, {})
        if not ds:
            continue
        xs = sorted(ds.keys())
        ys = [ds[i] for i in xs]
        ax.plot(xs, ys, "-o", color=kind_colors[kind], label=kind, alpha=0.85, markersize=5, linewidth=1.5)
    ax.set_yscale("log")
    ax.axhline(100, color="gray", ls=":", alpha=0.5)
    ax.text(0.5, 110, "100×", color="gray", fontsize=8.5)
    ax.set_xlabel("layer idx")
    ax.set_ylabel("channel skew  max / median")
    ax.set_title("pi0.5  expert / action head (18 layers)\nUniformly calm; max ≈ 159×")
    ax.set_xticks(range(18))
    ax.legend(ncol=3, fontsize=8.5, loc="upper right")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_ylim(1, 1e5)

    # --- (1,1) summary scatter: "is this layer in trouble after A2-lite?" ---
    ax = axes[1, 1]
    # Estimate post-A2 skew: divide by ~3-5× (we measured on gr00t the actual reduction is ~2-4× not √64=8)
    # Use a conservative 4× factor for prediction
    a2_factor = 4.0

    def scatter_group(records, color, marker, label, sz=70):
        for model, mod, kind, idx, skew, mx in records:
            ax.scatter(skew, skew / a2_factor, color=color, marker=marker, s=sz, alpha=0.7,
                       edgecolors="black", linewidth=0.4)
        # Single dummy for legend
        ax.scatter([], [], color=color, marker=marker, s=sz, alpha=0.85, edgecolors="black",
                   linewidth=0.4, label=label)

    scatter_group([r for r in gr_records if "down" not in r[2]], "lightblue", "o",
                  "gr00t LLM non-down (84 layers)")
    scatter_group([r for r in gr_records if "down" in r[2] and "L02" not in str(r[3])], "lightcoral", "s",
                  "gr00t LLM down_proj (other)")
    scatter_group([r for r in gr_records if "down" in r[2] and r[3] == 2], "darkred", "*",
                  "gr00t LLM L02.down (50000×)", sz=200)
    scatter_group([r for r in pi_records if r[1] == "paligemma" and "down" not in r[2]], "skyblue", "o",
                  "pi0.5 paligemma non-down")
    scatter_group([r for r in pi_records if r[1] == "paligemma" and "down" in r[2]], "darkorange", "s",
                  "pi0.5 paligemma down")
    scatter_group([r for r in pi_records if r[1] == "expert"], "lightgreen", "^",
                  "pi0.5 expert (all)")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("raw skew (max/med)")
    ax.set_ylabel(f"estimated post-A2-lite skew (raw / {a2_factor:.0f}×, conservative)")
    # Danger zone shading (where H estimation breaks)
    ax.axhspan(1000, 1e5, color="red", alpha=0.06)
    ax.text(2, 3000, "H-cal-broken zone\n(skew > 1000× post-A2)", color="darkred", fontsize=9, alpha=0.7)
    ax.axhspan(10, 100, color="green", alpha=0.05)
    ax.text(2, 20, "H-cal-friendly zone\n(skew < 100× post-A2)", color="darkgreen", fontsize=9, alpha=0.7)
    ax.set_title("Post-A2-lite skew estimate → H-cal difficulty zone\n(only gr00t L02.down lands in 'broken' zone; pi0.5 all clear)")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlim(3, 1e5)
    ax.set_ylim(1, 2e4)

    fig.suptitle("Cross-model outlier severity: gr00t-N1.5 vs pi0.5\n"
                 "Does pi0.5 have a structural pathology like gr00t LLM.L02.down_proj?",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout()
    out_path = OUT_DIR / "outlier_dist_gr00t_vs_pi05.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[plot] saved → {out_path}")


if __name__ == "__main__":
    main()
