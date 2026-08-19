"""Reads svd_vs_svdh_quant_mse.csv (produced by diagnose_svd_vs_svdh_quanterr.py)
and replots all bar panels in LINEAR y-axis (per user request: log compressed
the visible differences).

Three linear panels:
  1. Layer-output W4A4 nMSE (absolute, linear)
  2. A4 scale ceiling = p99 of per-row max(|X·R|) (linear)
  3. Reduction-vs-raw ratio = nMSE_rotated / nMSE_raw  (linear; <1 = better;
     anomaly L02.down_proj will exceed 1 — flagged in caption)

No model load; pure plotting. Output: svd_vs_svdh_quant_mse_linear.png next to CSV.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="experiment_results/visualization_for_paper/svd_vs_svdh_quant_mse.csv")
    ap.add_argument("--output", default="experiment_results/visualization_for_paper/svd_vs_svdh_quant_mse_linear.png")
    args = ap.parse_args()

    rows = []
    with open(args.csv) as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            for k in r:
                try: r[k] = float(r[k])
                except (TypeError, ValueError): pass
            rows.append(r)
    print(f"Read {len(rows)} layers from {args.csv}")

    labels = [f"{r['tag']} ({r['side']})" for r in rows]
    x = np.arange(len(rows))
    bw = 0.27

    nmse_raw  = np.array([r["nmse_raw"]  for r in rows])
    nmse_svd  = np.array([r["nmse_svd"]  for r in rows])
    nmse_svdh = np.array([r["nmse_svdh"] for r in rows])

    p99_raw  = np.array([r["rowmax_p99_raw"]  for r in rows])
    p99_svd  = np.array([r["rowmax_p99_svd"]  for r in rows])
    p99_svdh = np.array([r["rowmax_p99_svdh"] for r in rows])

    ratio_svd  = nmse_svd  / nmse_raw
    ratio_svdh = nmse_svdh / nmse_raw

    fig = plt.figure(figsize=(15, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    # Panel 1: absolute nMSE (linear)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - bw, nmse_raw,  width=bw, label="raw",            color="tab:gray")
    ax1.bar(x,      nmse_svd,  width=bw, label="SVD only",       color="tab:orange")
    ax1.bar(x + bw, nmse_svdh, width=bw, label="SVD-Hadamard",   color="tab:green")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=70, fontsize=8, ha="right")
    ax1.set_ylabel("W4A4 normalized MSE")
    ax1.set_title("Layer-output quant MSE")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=9)
    for xi, (a, b, c) in enumerate(zip(nmse_raw, nmse_svd, nmse_svdh)):
        ax1.text(xi - bw, a, f"{a:.3f}", ha="center", va="bottom", fontsize=6)
        ax1.text(xi,      b, f"{b:.3f}", ha="center", va="bottom", fontsize=6)
        ax1.text(xi + bw, c, f"{c:.3f}", ha="center", va="bottom", fontsize=6, color="tab:green")

    # Panel 2: A4 scale ceiling (p99 row-max)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - bw, p99_raw,  width=bw, label="raw",            color="tab:gray")
    ax2.bar(x,      p99_svd,  width=bw, label="SVD only",       color="tab:orange")
    ax2.bar(x + bw, p99_svdh, width=bw, label="SVD-Hadamard",   color="tab:green")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=70, fontsize=8, ha="right")
    ax2.set_ylabel("p99 per-row max(|X·R|)")
    ax2.set_title("A4 scale ceiling")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9)
    for xi, (a, b, c) in enumerate(zip(p99_raw, p99_svd, p99_svdh)):
        ax2.text(xi - bw, a, f"{a:.1f}", ha="center", va="bottom", fontsize=6)
        ax2.text(xi,      b, f"{b:.1f}", ha="center", va="bottom", fontsize=6)
        ax2.text(xi + bw, c, f"{c:.1f}", ha="center", va="bottom", fontsize=6, color="tab:green")

    fig.suptitle("SVD vs SVD-Hadamard rotation — GR00T-N1.5", fontsize=13)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out_path} + .pdf")


if __name__ == "__main__":
    main()
