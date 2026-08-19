#!/usr/bin/env python
"""V3-noSQ-SVDQ ablation visualization (single-seed, existing data only).

Goal: visualize where the gain of V3-noSQ-SVDQ comes from — LLM-side SVDQ or
DiT-side or both — using all already-run eval data on gr00t-N1.5 and pi0.5.

DiT side ALWAYS uses A2-lite rotation (the fixed first step). Comparison is
about what happens AFTER A2-lite:
  - A0: LLM A2-lite + RTN  /  DiT A2-lite + GPTQ + per-step (= Method C)
  - L_only / V3-noSQ-SVDQ: LLM A2-lite + SVD-r16 + GPTQ  /  DiT Method C
  - V3-noSQ-SpQR: LLM A2-lite + SpQR(1%) + GPTQ  /  DiT Method C
  - A1: LLM A2-lite + SVD-r16 + RTN  /  DiT Method C  (drop LLM GPTQ)
  - A2: LLM A2-lite + SVD-r16 + GPTQ  /  DiT A2-lite + RTN + per-step  (drop DiT GPTQ)

Single-seed (off=10) — noise ±5-10pp per suite — read with caution.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# ---------- gr00t configs (per-suite) ----------
# Manually compiled from existing eval results + QUANT_OVERVIEW reference table.
# All numbers are W4A4 single-seed off=10, 10 trials/task, 4 suites.

GR00T = [
    # (short_label, long_label, [object, spatial, goal, long], source_note)
    ("FP16",          "FP16 (no quant)",
     [93, 86, 86, 79], "results/fp16_baseline/"),
    ("A0 = Method C", "C0: A2-lite RTN | Method C",
     [84, 87, 90, 82], "calgrid_c_cal10 (file)"),
    ("L_only V3-SVDQ","LLM A2-lite+SVD-r16+GPTQ | Method C",
     [95, 86, 93, 81], "attn_iter_LLM_svdqV3noSQ_DiT_methodC_cal10_r16 (file; long=81 not 89 from doc)"),
    ("V3-noSQ-SpQR",  "LLM A2-lite+SpQR(1%)+GPTQ | Method C",
     [93, 88, 89, 81], "0521-22 todo doc table"),
    ("D_only (full-A2 LLM)", "LLM A2-Full RTN | DiT SVDQ-custom",
     [87, 87, 88, 78], "QUANT_OVERVIEW §4.1 main table (W4A4 + SVDQ custom full A2 LLM)"),
    ("D_only (A2-lite LLM)", "LLM A2-lite RTN | DiT SVDQ-custom",
     [89, 88, 86, 83], "QUANT_OVERVIEW §4.1 (W4A4 + SVDQ custom, A2-lite LLM)"),
    ("LD_both",       "LLM SVDQ-custom | DiT SVDQ-custom",
     [86, 86, 95, 77], "QUANT_OVERVIEW §5.6"),
]

# ---------- pi0.5 configs ----------
# Per 0521-22_overnight_todo.md table + QUANT_OVERVIEW §4.3.
PI05 = [
    ("FP16",          "FP16",
     [98, 100, 97, 93], "QUANT_OVERVIEW §4.3"),
    ("A0 = Method C", "Method C (paligemma DuQuant + expert SVDQ-Method-C)",
     [98, 99, 99, 94], "0521-22 todo doc pi0.5 table"),
    # V3-noSQ-SVDQ on pi0.5 was abandoned (runtime perf bug + quality concern)
    ("L_only V3-SVDQ", "V3-noSQ-SVDQ (paligemma SVDQ + expert SVDQ)  — ABANDONED",
     [None, None, None, None], "abandoned 2026-05-22 (perf bug + partial goal=49)"),
]

SUITES = ["object", "spatial", "goal", "long"]


def fourth_avg(row):
    if any(v is None for v in row):
        return None
    return float(np.mean(row))


def build_panel(ax, data, title, ylim=None, hatch_unavailable=True):
    n_cfg = len(data)
    n_grp = len(SUITES) + 1   # 4 suites + avg
    x = np.arange(n_grp)
    width = 0.85 / n_cfg

    palette = {
        "FP16":           "#444444",
        "A0 = Method C":  "#1f77b4",
        "L_only V3-SVDQ": "#d62728",
        "V3-noSQ-SpQR":   "#9467bd",
        "D_only (full-A2 LLM)": "#ff7f0e",
        "D_only (A2-lite LLM)": "#ffbb78",
        "LD_both":        "#2ca02c",
    }
    fp16_row = None
    for i, (label, _, vals, _) in enumerate(data):
        if label == "FP16":
            fp16_row = vals + [fourth_avg(vals)]
            break

    for i, (label, _, vals, _) in enumerate(data):
        if any(v is None for v in vals):
            if hatch_unavailable:
                # plot empty hatched bars to show "missing"
                ys = [0] * n_grp
                ax.bar(x + (i - n_cfg/2 + 0.5) * width, ys, width,
                       label=label + " (abandoned)", color="lightgray",
                       hatch="xxx", edgecolor="gray", alpha=0.45)
            continue
        ys = vals + [fourth_avg(vals)]
        color = palette.get(label, "C0")
        bars = ax.bar(x + (i - n_cfg/2 + 0.5) * width, ys, width,
                      label=label, color=color, alpha=0.92,
                      edgecolor="black", linewidth=0.4)
        for bar, y in zip(bars, ys):
            ax.text(bar.get_x() + bar.get_width()/2, y + 0.6, f"{y:.0f}",
                    ha="center", va="bottom", fontsize=7.5)

    # FP16 reference horizontal lines per group
    if fp16_row is not None:
        for j, fp_v in enumerate(fp16_row):
            if fp_v is None: continue
            ax.plot([j - 0.42, j + 0.42], [fp_v, fp_v],
                    color="black", linestyle="--", linewidth=1.0, alpha=0.55,
                    zorder=20)

    ax.set_xticks(x)
    ax.set_xticklabels([s for s in SUITES] + ["**4-avg**"])
    ax.set_ylabel("success %")
    ax.set_title(title, fontweight="bold")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08),
              fontsize=8.5, framealpha=0.92, ncol=min(n_cfg, 4))


def main():
    out_dir = Path("experiment_results/visualization_for_paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams["font.size"] = 10
    fig, axes = plt.subplots(2, 1, figsize=(16, 13),
                             gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.55})

    build_panel(axes[0], GR00T,
                "gr00t-N1.5  W4A4 single-seed (off=10)  |  dashed = FP16 per suite",
                ylim=(70, 102))
    build_panel(axes[1], PI05,
                "pi0.5  W4A4 single-seed (off=10)  |  V3-noSQ-SVDQ ABANDONED (perf bug + partial goal=49%)",
                ylim=(85, 103))

    # Decomposition findings as figure-level text (no overlap with bars)
    findings = (
        "Decomposition findings (Δ vs A0 = Method C, 4-avg):\n"
        "  • L_only (LLM SVDQ only, DiT unchanged) → +3.0pp  ←  main contributor (object +11, goal +3)\n"
        "  • D_only (DiT SVDQ only, LLM unchanged) → +0.75pp / −0.75pp  ←  within single-seed noise (±5pp)\n"
        "  • LD_both (LLM SVDQ + DiT SVDQ) → +0.25pp  ←  no synergy; long regresses (−4pp vs L_only)\n"
        "Read: V3-noSQ-SVDQ's gain comes mostly from the LLM side; DiT SVDQ is neutral, and stacking both shows mild anti-synergy on long-horizon."
    )
    fig.text(0.5, 0.005, findings, fontsize=10, color="#222",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lemonchiffon",
                       edgecolor="darkred", linewidth=0.8, alpha=0.92),
             ha="center", va="bottom")

    fig.suptitle("V3-noSQ-SVDQ decomposition: where the gain comes from\n"
                 "(LLM SVDQ vs DiT SVDQ — single-seed, existing data only, noise ±5-10pp/suite)",
                 fontsize=13, fontweight="bold", y=1.00)
    fig.tight_layout(rect=[0, 0.10, 1, 0.99])
    out_path = out_dir / "v3nosq_svdq_decomposition.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[plot] saved → {out_path} + .pdf")

    # Print summary table
    print()
    print("=" * 92)
    print("gr00t single-seed decomposition table")
    print("=" * 92)
    print(f"{'config':<28}  {'obj':>5}  {'spa':>5}  {'goal':>5}  {'long':>5}  {'4-avg':>6}  {'Δ vs FP16':>10}")
    print("-" * 92)
    fp16 = None
    for label, lname, vals, _ in GR00T:
        if any(v is None for v in vals):
            row = f"{label:<28}  " + "    -      -      -      -      -        -"
            print(row); continue
        avg = fourth_avg(vals)
        if label == "FP16": fp16 = avg
        delta = (avg - fp16) if fp16 is not None and label != "FP16" else (0 if label == "FP16" else None)
        delta_str = f"{delta:+.2f}" if delta is not None else "-"
        print(f"{label:<28}  {vals[0]:>5}  {vals[1]:>5}  {vals[2]:>5}  {vals[3]:>5}  {avg:>6.2f}  {delta_str:>10}")

    print()
    print("=" * 92)
    print("Decomposition (Δ vs A0 = Method C)")
    print("=" * 92)
    a0 = next(fourth_avg(v) for l, _, v, _ in GR00T if l == "A0 = Method C")
    for label, _, vals, _ in GR00T:
        if any(v is None for v in vals) or label == "A0 = Method C" or label == "FP16":
            continue
        avg = fourth_avg(vals)
        delta = avg - a0
        marker = " ★" if abs(delta) > 4 else ""
        print(f"  {label:<28}  4-avg = {avg:.2f}   Δ vs A0 = {delta:+.2f}pp{marker}")


if __name__ == "__main__":
    main()
