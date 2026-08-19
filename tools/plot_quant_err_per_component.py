#!/usr/bin/env python
"""Per-component (layer × kind) W4 quantization error: methods compared.

For each (layer, kind) in the gr00t xw_dump we compute relative output error
    err(Q) = || X W^T  -  X R Q(W R)^T ||_F  /  || X W^T ||_F
under four methods:
  M1. RTN W4 (no rotation):       Q(W) = block-MSE round to int4
  M2. A2-lite + RTN W4:           rotate (svd-hadamard, block=64, perm), then RTN
  M3. A2-lite + GPTQ W4:          rotate, then GPTQ (block=128, damp=0.05)  ←  Method C weight side
  M4. A2-lite + SVD-r16 + GPTQ:   rotate, SVD-r16 lowrank head, GPTQ on residual  ←  V3-noSQ-SVDQ

This is the *weight-only* error decomposition; act quant is excluded so the
panels isolate where on the network V3-noSQ-SVDQ structurally helps.

Output: experiment_results/visualization_for_paper/quant_err_per_component.png
"""
from __future__ import annotations
from pathlib import Path
import sys
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from gr00t.quantization.duquant_preprocess import compute_duquant_rotation_only, compute_mse_scales
from gr00t.quantization.gptq_layers import gptq_quantize_weight


SRC_GR00T = "experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep2.pt"
SRC_PI05  = "experiment_results/svd_diagnostics_gr00t/xw_dump_pi05_object_n4.pt"
OUT_DIR = Path("experiment_results/visualization_for_paper")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_a2lite_rotation(W: torch.Tensor, block: int = 64) -> torch.Tensor:
    W_cpu = W.detach().cpu().float()
    R, perm = compute_duquant_rotation_only(
        W_cpu, block_size=block, enable_permute=True,
        lambda_smooth=0.15, rot_mode="svd_hadamard",
    )
    R = R.to(torch.float32)
    if perm is not None:
        P = torch.zeros((W.shape[1], W.shape[1]), dtype=torch.float32)
        perm_t = torch.from_numpy(perm.astype("int64"))
        P[perm_t, torch.arange(W.shape[1])] = 1.0
        R = P @ R
    return R.to(device=W.device, dtype=torch.float32)


def rtn_w4(W: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Per-row MSE-scaled symmetric int4, applied once over the full input dim."""
    scales = compute_mse_scales(W, bits).to(dtype=W.dtype, device=W.device)
    qmax = (1 << (bits - 1)) - 1
    Q = (W / scales.unsqueeze(1)).round().clamp(-qmax - 1, qmax) * scales.unsqueeze(1)
    return Q.to(dtype=W.dtype)


def rel_output_err(W: torch.Tensor, X: torch.Tensor, W_q_pipeline) -> float:
    """
    rel-err = || X W^T - W_q_pipeline(X) ||_F / || X W^T ||_F
    W_q_pipeline takes X and returns the quant-emulated Y_q = X' Q(W')^T.
    """
    Y = X @ W.T
    Y_q = W_q_pipeline(X)
    num = torch.linalg.norm(Y - Y_q).item()
    den = max(torch.linalg.norm(Y).item(), 1e-12)
    return float(num / den)


def quantize_methods(W: torch.Tensor, X: torch.Tensor, *, block=64,
                     gptq_block=128, gptq_damp=0.05, svd_rank=16) -> dict[str, float]:
    """Compute rel-output-err for all 4 methods on a single (W, X)."""
    W = W.to(DEVICE).float()
    X = X.to(DEVICE).float()
    out = {}

    # M1. RTN no rotation
    Q1 = rtn_w4(W)
    out["RTN (no rot)"] = rel_output_err(W, X, lambda x: x @ Q1.T)

    # Rotation
    R = build_a2lite_rotation(W, block=block)
    W_r = W @ R
    X_r = X @ R
    N = X_r.shape[0]
    H = (X_r.T @ X_r) / float(N)

    # M2. A2-lite + RTN
    Q2 = rtn_w4(W_r)
    out["A2-lite + RTN"] = rel_output_err(W, X, lambda x: (x @ R) @ Q2.T)

    # M3. A2-lite + GPTQ
    Q3 = gptq_quantize_weight(W_r, H, bits=4, block_size=gptq_block,
                              damp_percent=gptq_damp)
    out["A2-lite + GPTQ"] = rel_output_err(W, X, lambda x: (x @ R) @ Q3.T)

    # M4. A2-lite + SVD-r16 + GPTQ on residual
    U, S_, Vh = torch.linalg.svd(W_r, full_matrices=False)
    r = min(svd_rank, S_.shape[0])
    sqrtS = torch.sqrt(S_[:r].clamp_min(0.0))
    lowA = U[:, :r] * sqrtS[None, :]
    lowB = (Vh[:r, :].T) * sqrtS[None, :]
    W_res = W_r - lowA @ lowB.T
    Q4 = gptq_quantize_weight(W_res, H, bits=4, block_size=gptq_block,
                              damp_percent=gptq_damp)
    W4_full = Q4 + lowA @ lowB.T   # lowrank head kept FP, residual int4
    out["A2-lite + SVD-r16 + GPTQ"] = rel_output_err(W, X, lambda x: (x @ R) @ W4_full.T)

    return out


def main():
    methods = ["RTN (no rot)", "A2-lite + RTN", "A2-lite + GPTQ"]
    # Main per-layer bar color + a lighter same-hue shade for the "average" bar
    # (drawn at the leftmost slot of each panel).
    colors = {"RTN (no rot)":   "#666666", "A2-lite + RTN":   "#1f77b4",
              "A2-lite + GPTQ": "#2ca02c"}
    colors_avg = {"RTN (no rot)":   "#cccccc", "A2-lite + RTN":   "#aec7e8",
                  "A2-lite + GPTQ": "#98df8a"}

    cache_path = OUT_DIR / "quant_err_per_component_results.pt"
    if cache_path.exists():
        print(f"[cache] loading from {cache_path}")
        results = torch.load(cache_path, weights_only=False)
    else:
        results = {}
        t_start = time.time()
        for src_path in [SRC_GR00T, SRC_PI05]:
            if not Path(src_path).exists():
                print(f"[warn] {src_path} missing, skipping", flush=True)
                continue
            print(f"\n=== processing {src_path} ===", flush=True)
            cal = torch.load(src_path, weights_only=False)
            for name, rec in cal.items():
                W, X = rec["W"], rec["X"]
                module = name.split(".")[0]   # LLM / DiT / paligemma / expert
                layer_idx = int(name.split(".L")[1].split(".")[0])
                kind = name.split(".")[-1]
                t0 = time.time()
                errs = quantize_methods(W, X)
                dt = time.time() - t0
                print(f"  {name:<32}  " + "  ".join(f"{m}={errs[m]:.4f}" for m in methods)
                      + f"   ({dt:.1f}s)", flush=True)
                results[(module, layer_idx, kind)] = errs
        print(f"\ntotal compute: {time.time() - t_start:.1f}s")
        torch.save(results, cache_path)
        print(f"[cache] saved → {cache_path}")

    # ---------- Plot ----------
    mpl.rcParams["font.size"] = 10
    fig, axes = plt.subplots(3, 2, figsize=(17, 14))

    def bar_panel(ax, module, title, ylim=None):
        rows = sorted([k for k in results if k[0] == module], key=lambda k: (k[1], k[2]))
        if not rows:
            ax.set_title(title + "  (no data)", fontweight="bold")
            ax.axis("off")
            return
        # "average across layers" pseudo-row, drawn at index 0 with a lighter shade.
        x_labels = ["AVG"] + [f"L{i:02d}.{kd.replace('_proj','')}" for (_, i, kd) in rows]
        n_x = len(rows) + 1
        x = np.arange(n_x)
        width = 0.85 / len(methods)
        for mi, m in enumerate(methods):
            ys_layers = [results[r][m] for r in rows]
            avg_y = float(np.mean(ys_layers))
            ys = [avg_y] + ys_layers
            # Two-tone bars: lighter shade for the AVG slot (index 0)
            bar_colors = [colors_avg[m]] + [colors[m]] * len(ys_layers)
            ax.bar(x + (mi - len(methods)/2 + 0.5) * width, ys, width,
                   label=m, color=bar_colors, alpha=0.92,
                   edgecolor="black", linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
        # Visual divider between AVG and per-layer groups
        ax.axvline(0.5, color="black", lw=0.5, alpha=0.35, linestyle="--")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_ylabel(r"rel-err $\|Y - Y_q\|_F\ /\ \|Y\|_F$")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.32),
                  ncol=3, framealpha=0.95)
        # Worst-layer annotation (shifted by +1 because index 0 is AVG)
        worst_i = int(np.argmax([results[r]["A2-lite + RTN"] for r in rows]))
        worst_row = rows[worst_i]
        worst_err = results[worst_row]["A2-lite + RTN"]
        ax.annotate(f"worst: {x_labels[worst_i + 1]}\nA2+RTN err = {worst_err:.3f}",
                    xy=(worst_i + 1, worst_err), xytext=(15, 30),
                    textcoords="offset points", fontsize=8.5, color="darkred",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lemonchiffon", alpha=0.85),
                    arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))

    bar_panel(axes[0, 0], "LLM",       "gr00t-N1.5  LLM (eagle)  —  W4 weight-only rel-output-err per component", ylim=(0, 0.22))
    bar_panel(axes[0, 1], "DiT",       "gr00t-N1.5  DiT  —  W4 weight-only rel-output-err per component",         ylim=(0, 0.22))
    bar_panel(axes[1, 0], "paligemma", "pi0.5  paligemma  —  W4 weight-only rel-output-err per component",        ylim=(0, 0.22))
    bar_panel(axes[1, 1], "expert",    "pi0.5  expert / action head  —  W4 weight-only rel-output-err per component", ylim=(0, 0.22))

    # ---- (2,0) Method delta vs A2-lite+RTN, per kind, all modules ----
    ax = axes[2, 0]
    kinds = sorted({k[2] for k in results})
    all_modules = ["LLM", "DiT", "paligemma", "expert"]
    # For each kind & module, compute mean rel-err per method
    summary = {}  # (module, kind) -> {method: mean_err}
    for module in all_modules:
        for kind in kinds:
            errs_per_method = {m: [] for m in methods}
            for (mod, _, kd), d in results.items():
                if mod != module or kd != kind: continue
                for m in methods: errs_per_method[m].append(d[m])
            if any(errs_per_method[m] for m in methods):
                summary[(module, kind)] = {m: float(np.mean(v)) if v else None
                                           for m, v in errs_per_method.items()}

    # Plot grouped bars: x = ["AVG"] + (module, kind) pairs; groups = methods.
    # AVG = mean across all (module, kind) for each method (left-most slot,
    # lighter shade of the same hue).
    pairs = sorted(summary.keys(), key=lambda p: (p[0], p[1]))
    xlabels = ["AVG"] + [f"{m}.{kd.replace('_proj','')}" for m, kd in pairs]
    n_x = len(pairs) + 1
    x = np.arange(n_x)
    width = 0.85 / len(methods)
    for mi, m in enumerate(methods):
        ys_pairs = [summary[p][m] for p in pairs]
        avg_y = float(np.mean(ys_pairs))
        ys = [avg_y] + ys_pairs
        bar_colors = [colors_avg[m]] + [colors[m]] * len(ys_pairs)
        ax.bar(x + (mi - len(methods)/2 + 0.5) * width, ys, width,
               label=m, color=bar_colors, alpha=0.92, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8.5)
    ax.axvline(0.5, color="black", lw=0.5, alpha=0.35, linestyle="--")
    ax.set_ylabel("mean rel-err  (across layers)")
    ax.set_title("Per-kind summary: mean rel-err across sampled layers (AVG = mean over all kinds)",
                 fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              ncol=3, framealpha=0.95)

    # ---- (2,1) Isolated GPTQ contribution: A2+RTN  vs  A2+GPTQ ----
    # (was SVD-r16 vs A2+GPTQ — SVD-r16 dropped per request)
    ax = axes[2, 1]
    rows_all = sorted(results.keys(), key=lambda k: (k[0], k[1], k[2]))
    module_marker = {"LLM": "s", "DiT": "o", "paligemma": "D", "expert": "^"}
    kind_color = {"q_proj": "#1f77b4", "o_proj": "#2ca02c",
                  "gate_proj": "#ff7f0e", "down_proj": "#d62728"}
    seen_legend = set()
    for (mod, lidx, kd) in rows_all:
        d = results[(mod, lidx, kd)]
        x_v = d["A2-lite + RTN"]
        y_v = d["A2-lite + GPTQ"]
        lab_key = (mod, kd)
        lab = None
        if lab_key not in seen_legend:
            lab = f"{mod} {kd}"
            seen_legend.add(lab_key)
        ax.scatter(x_v, y_v, s=85, alpha=0.85,
                   c=kind_color.get(kd, "gray"),
                   marker=module_marker.get(mod, "x"),
                   edgecolors="black", linewidth=0.5, label=lab)
        if mod == "LLM" and kd == "down_proj":
            ax.annotate(f"LLM.L{lidx:02d}.down",
                        xy=(x_v, y_v), xytext=(8, 4),
                        textcoords="offset points", fontsize=8, color="darkred")

    lo = min(min(d["A2-lite + RTN"], d["A2-lite + GPTQ"])
             for d in results.values())
    hi = max(max(d["A2-lite + RTN"], d["A2-lite + GPTQ"])
             for d in results.values())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, lw=1, label="y = x  (GPTQ = no gain)")
    ax.plot([lo, hi], [0.5 * lo, 0.5 * hi], color="gray", linestyle=":", alpha=0.5, lw=1,
            label="y = 0.5 x  (2× reduction)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rel-err  A2-lite + RTN")
    ax.set_ylabel("rel-err  A2-lite + GPTQ")
    ax.set_title("Isolated GPTQ contribution after rotation:\n"
                 "GPTQ over RTN reduces W4 weight-only rel-err ~40–50%",
                 fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              ncol=3, framealpha=0.95)
    ax.grid(True, which="both", alpha=0.25)

    # Mean reduction across all layers
    reds = [(d["A2-lite + RTN"] - d["A2-lite + GPTQ"]) / d["A2-lite + RTN"]
            for d in results.values()]
    ax.text(0.04, 0.96, f"mean reduction\n{100*float(np.mean(reds)):.1f}% ± {100*float(np.std(reds)):.1f}%",
            transform=ax.transAxes, va="top", fontsize=9, color="darkred",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lemonchiffon", alpha=0.85))

    fig.suptitle("W4 weight-quant error per component: RTN  vs  A2-lite + RTN  vs  A2-lite + GPTQ\n"
                 "(weight-only error; activation quant excluded — isolates structural effect of each step)",
                 fontsize=12.5, fontweight="bold", y=1.00)
    fig.tight_layout()
    out_path = OUT_DIR / "quant_err_per_component.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"\n[plot] saved → {out_path} + .pdf")

    # ---- Numeric summary table ----
    print("\n" + "=" * 110)
    print("Per-kind summary (mean rel-err across sampled layers)")
    print("=" * 110)
    print(f"{'module.kind':<20}  " + "  ".join(f"{m:<26}" for m in methods))
    print("-" * 130)
    for (mod, kd), d in sorted(summary.items()):
        print(f"{mod}.{kd:<14}  " + "  ".join(f"{d[m]:.4f}".ljust(26) for m in methods))


if __name__ == "__main__":
    main()
