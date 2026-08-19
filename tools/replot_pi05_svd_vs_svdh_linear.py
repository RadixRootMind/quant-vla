"""Mirror of replot_svd_vs_svdh_linear.py for pi0.5.

Reads pi05_svd_vs_svdh_quant_mse.csv (produced by diagnose_svd_vs_svdh_quanterr_pi05.py)
and replots the 3-panel LINEAR-y figure (no model load). Saves both PNG and PDF.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default="experiment_results/visualization_for_paper/pi05_svd_vs_svdh_quant_mse.csv")
    ap.add_argument("--output", default="experiment_results/visualization_for_paper/pi05_svd_vs_svdh_quant_mse_linear.png")
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

    fig = plt.figure(figsize=(22, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - bw, nmse_raw,  width=bw, label="raw (no rotation)",   color="tab:gray")
    ax1.bar(x,      nmse_svd,  width=bw, label="SVD only",             color="tab:orange")
    ax1.bar(x + bw, nmse_svdh, width=bw, label="SVD-Hadamard (A2-lite)", color="tab:green")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=70, fontsize=8, ha="right")
    ax1.set_ylabel("end-to-end W4A4 normalized MSE  (lower = better)")
    ax1.set_title("Layer-output quant MSE under W4A4 (linear y)")
    ax1.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=9)
    for xi, (a, b, c) in enumerate(zip(nmse_raw, nmse_svd, nmse_svdh)):
        ax1.text(xi - bw, a, f"{a:.3f}", ha="center", va="bottom", fontsize=6)
        ax1.text(xi,      b, f"{b:.3f}", ha="center", va="bottom", fontsize=6)
        ax1.text(xi + bw, c, f"{c:.3f}", ha="center", va="bottom", fontsize=6, color="tab:green")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(x - bw, p99_raw,  width=bw, label="raw", color="tab:gray")
    ax2.bar(x,      p99_svd,  width=bw, label="SVD only", color="tab:orange")
    ax2.bar(x + bw, p99_svdh, width=bw, label="SVD-Hadamard", color="tab:green")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=70, fontsize=8, ha="right")
    ax2.set_ylabel("p99 per-row max(|X·R|)  (drives A4 scale on heavy rows)")
    ax2.set_title("A4 scale ceiling: 99th-percentile token-max (linear y)")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9)
    for xi, (a, b, c) in enumerate(zip(p99_raw, p99_svd, p99_svdh)):
        ax2.text(xi - bw, a, f"{a:.1f}", ha="center", va="bottom", fontsize=6)
        ax2.text(xi,      b, f"{b:.1f}", ha="center", va="bottom", fontsize=6)
        ax2.text(xi + bw, c, f"{c:.1f}", ha="center", va="bottom", fontsize=6, color="tab:green")

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(x - bw/2, ratio_svd,  width=bw, label="SVD only / raw",         color="tab:orange")
    ax3.bar(x + bw/2, ratio_svdh, width=bw, label="SVD-Hadamard / raw",     color="tab:green")
    ax3.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=70, fontsize=8, ha="right")
    ax3.set_ylabel("nMSE(rotated) / nMSE(raw)   ( <1 = rotation HELPS )")
    ax3.set_title("Relative quant MSE vs raw   (linear y; dashed=raw)")
    ax3.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=9)
    for xi, (b, c) in enumerate(zip(ratio_svd, ratio_svdh)):
        ax3.text(xi - bw/2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=6)
        ax3.text(xi + bw/2, c, f"{c:.2f}", ha="center", va="bottom", fontsize=6, color="tab:green")
    ax3.set_ylim(0.0, float(max(ratio_svd.max(), ratio_svdh.max()) * 1.1))

    fig.suptitle(
        "pi0.5 — SVD vs SVD-Hadamard rotation, true W4A4 quant metrics  (LINEAR scale)   "
        "[pi05 libero_object_obs, calib=4, block=64]",
        fontsize=13,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"wrote {out_path} + .pdf")


if __name__ == "__main__":
    main()
