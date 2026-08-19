"""SVD vs SVD-Hadamard rotation comparison using **true W4A4 quantization metrics**.

Unlike diagnose_svd_vs_svdh_quickcap.py (which used per-channel max/median as a
proxy), this script reports the metrics SVD-Hadamard is actually designed to
optimize:

  1. End-to-end W4A4 quant MSE at layer output:
        y       = X · Wᵀ                (FP reference)
        X_q     = fake_W4A4_per_token(X·R)
        W_q     = fake_W4_per_output(W·R)
        y_hat   = X_q · W_qᵀ
        nMSE    = ||y_hat - y||² / ||y||²
     Lower = better.
  2. Per-token A4 scale spread (row_max(|X·R|)) — directly = A4 step size.
     Reports quartiles and CDF of row-max for each rotation.
  3. (Supporting) per-channel q999 max/median — same proxy as before, kept for
     reference but de-emphasized.

For each of 10 GR00T layers (5 LLM + 5 DiT), we compare:
  raw  (R = I)   |   SVD only (R = U_b)   |   SVD-Hadamard (R = U_b @ H_b)

Run under the awesome_quant_vla conda env. Same args as diagnose_svd_vs_svdh_quickcap.py.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    ensure_libero_runtime, load_libero_samples, load_policy, seed_everything,
    get_named_module,
)
from tools.analyze_layerwise_quant_drift import normalized_input_no_inference  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402


# ------------------------------------------------------------ helpers


def hadamard_matrix(n: int, device, dtype) -> torch.Tensor:
    assert n & (n - 1) == 0, f"Hadamard requires power-of-two dim, got {n}"
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def block_svd_rotation(W: torch.Tensor, block_size: int, with_hadamard: bool) -> torch.Tensor:
    """Block-diag rotation. with_hadamard=False → R = block-diag(U_b); True → R = block-diag(U_b @ H_b)."""
    out_f, in_f = W.shape
    if in_f % block_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by block_size={block_size}")
    n_blocks = in_f // block_size
    H_blk = hadamard_matrix(block_size, W.device, torch.float32) if with_hadamard else None
    R = torch.zeros((in_f, in_f), device=W.device, dtype=torch.float32)
    Wf = W.detach().float()
    for b in range(n_blocks):
        s = b * block_size
        e = s + block_size
        U, _, _ = torch.linalg.svd(Wf[:, s:e].T, full_matrices=True)
        R[s:e, s:e] = (U @ H_blk) if with_hadamard else U
    return R


def fake_quant_sym_per_row(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-row INT quant fake-dequant. Last dim is quantization axis (per-row over dim=-1)."""
    qmax = (1 << (bits - 1)) - 1
    scale = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    return torch.round(t / scale).clamp_(-qmax, qmax) * scale


def quant_output_mse(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor | None,
                     w_bits: int = 4, a_bits: int = 4) -> dict:
    """Compute normalized W4A4 quant MSE on y = X @ Wᵀ under rotation R.

    Returns {"nmse": float, "y_norm_sq": float, "n_tokens": int}.
    """
    # FP reference computed on rotated weight too — y is invariant to R since
    # (X·R) @ (W·R)ᵀ = X·R·Rᵀ·Wᵀ = X·Wᵀ when R is orthogonal.
    y = X.float() @ W.float().T
    if R is None:
        X_r = X
        W_r = W
    else:
        X_r = X @ R
        W_r = W @ R
    X_q = fake_quant_sym_per_row(X_r.float(), a_bits)         # per-token A4
    W_q = fake_quant_sym_per_row(W_r.float(), w_bits)         # per-output-row W4
    y_hat = X_q @ W_q.T
    delta = y_hat - y
    nmse = float((delta.pow(2).mean() / y.pow(2).mean().clamp_min(1e-12)).item())
    return {
        "nmse": nmse,
        "y_norm_sq": float(y.pow(2).mean().item()),
        "n_tokens": int(X.shape[0]),
    }


def row_max_stats(X: torch.Tensor) -> dict:
    """Per-row max(|X|) distribution stats."""
    rm = X.abs().float().max(dim=-1).values
    return {
        "p50": float(rm.median().item()),
        "p90": float(torch.quantile(rm, 0.90).item()),
        "p99": float(torch.quantile(rm, 0.99).item()),
        "max": float(rm.max().item()),
        "mean": float(rm.mean().item()),
        "vals_cpu": rm.detach().cpu().numpy(),
    }


def chan_q999_maxmed(X: torch.Tensor) -> float:
    """Per-channel q999(|X|) max/median — kept as proxy reference."""
    q = torch.quantile(X.abs().float(), 0.999, dim=0).cpu().numpy()
    return float(q.max() / (np.median(q) + 1e-12))


# ------------------------------------------------------------ main


LLM_TARGETS = [
    ("L02.down_proj", "backbone.eagle_model.language_model.model.layers.2.mlp.down_proj"),
    ("L05.down_proj", "backbone.eagle_model.language_model.model.layers.5.mlp.down_proj"),
    ("L09.down_proj", "backbone.eagle_model.language_model.model.layers.9.mlp.down_proj"),
    ("L02.q_proj",    "backbone.eagle_model.language_model.model.layers.2.self_attn.q_proj"),
    ("L02.gate_proj", "backbone.eagle_model.language_model.model.layers.2.mlp.gate_proj"),
]
DIT_TARGETS = [
    ("L02.to_q",    "action_head.model.transformer_blocks.2.attn1.to_q"),
    ("L02.ff.down", "action_head.model.transformer_blocks.2.ff.net.2"),
    ("L08.to_q",    "action_head.model.transformer_blocks.8.attn1.to_q"),
    ("L08.ff.down", "action_head.model.transformer_blocks.8.ff.net.2"),
    ("L14.to_q",    "action_head.model.transformer_blocks.14.attn1.to_q"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task-suite-name", default="libero_object")
    ap.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--video-backend", default="torchvision_av")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--token-cap", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--libero-num-trials-per-task", type=int, default=1)
    ap.add_argument("--libero-num-steps-wait", type=int, default=10)
    ap.add_argument("--libero-sampling-mode", default="one_per_task",
                    choices=["sequential", "one_per_task", "one_per_trial"])
    ap.add_argument("--libero-resolution", type=int, default=256)
    ap.add_argument("--denoising-steps", type=int, default=8)
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=4)
    ap.add_argument("--output-dir", default="experiment_results/visualization_for_paper")
    args = ap.parse_args()

    ensure_libero_runtime()
    seed_everything(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[SVDH-QERR] loading policy {args.checkpoint}")
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    policy.model.eval()
    for p in policy.model.parameters():
        p.requires_grad_(False)

    targets = LLM_TARGETS + DIT_TARGETS
    cache: dict[str, list[torch.Tensor]] = {tag: [] for tag, _ in targets}
    name_to_mod: dict[str, torch.nn.Module] = {}

    def make_hook(tag: str):
        def hook(_m, inputs):
            x = inputs[0]
            if x is None:
                return
            x = x.detach().reshape(-1, x.shape[-1]).float()
            if x.shape[0] > args.token_cap:
                idx = torch.randperm(x.shape[0], device=x.device)[: args.token_cap]
                x = x[idx]
            cache[tag].append(x.cpu())
        return hook

    handles = []
    for tag, name in targets:
        try:
            mod = get_named_module(policy.model, name)
        except Exception as e:
            print(f"[skip-hook] {tag}: {e}")
            continue
        handles.append(mod.register_forward_pre_hook(make_hook(tag)))
        name_to_mod[tag] = mod

    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    print(f"[SVDH-QERR] driving {len(samples)} cal samples")
    with torch.no_grad():
        for i, s in enumerate(samples):
            seed_everything(s["seed"])
            norm = normalized_input_no_inference(policy, s["obs"])
            policy.model.get_action(norm)
            print(f"[SVDH-QERR] sample {i+1}/{len(samples)}")
    for h in handles:
        h.remove()

    rows = []
    rowmax_data: dict[str, dict[str, np.ndarray]] = {}

    for tag, _ in targets:
        if tag not in name_to_mod:
            continue
        chunks = cache[tag]
        if not chunks:
            continue
        X = torch.cat(chunks, dim=0).to(device=device)
        W = name_to_mod[tag].weight.detach().to(device=device, dtype=torch.float32)
        in_f = X.shape[1]
        if in_f % args.block_size != 0:
            print(f"[skip] {tag}: in_f={in_f} not divisible by block_size={args.block_size}")
            continue

        R_svd  = block_svd_rotation(W, args.block_size, with_hadamard=False)
        R_svdh = block_svd_rotation(W, args.block_size, with_hadamard=True)

        # End-to-end W4A4 quant MSE
        nmse_raw  = quant_output_mse(X, W, None,    w_bits=args.w_bits, a_bits=args.a_bits)
        nmse_svd  = quant_output_mse(X, W, R_svd,  w_bits=args.w_bits, a_bits=args.a_bits)
        nmse_svdh = quant_output_mse(X, W, R_svdh, w_bits=args.w_bits, a_bits=args.a_bits)

        # Per-row max stats
        rm_raw  = row_max_stats(X)
        rm_svd  = row_max_stats(X @ R_svd)
        rm_svdh = row_max_stats(X @ R_svdh)
        rowmax_data[tag] = {
            "raw":  rm_raw["vals_cpu"],
            "svd":  rm_svd["vals_cpu"],
            "svdh": rm_svdh["vals_cpu"],
        }

        # Channel max/med (proxy)
        chmm_raw  = chan_q999_maxmed(X)
        chmm_svd  = chan_q999_maxmed(X @ R_svd)
        chmm_svdh = chan_q999_maxmed(X @ R_svdh)

        rows.append({
            "tag": tag,
            "side": "LLM" if tag in {t for t,_ in LLM_TARGETS} else "DiT",
            "n_tokens": int(X.shape[0]),
            "in_f": int(in_f),
            "nmse_raw":  nmse_raw["nmse"],
            "nmse_svd":  nmse_svd["nmse"],
            "nmse_svdh": nmse_svdh["nmse"],
            "rowmax_p99_raw":  rm_raw["p99"],
            "rowmax_p99_svd":  rm_svd["p99"],
            "rowmax_p99_svdh": rm_svdh["p99"],
            "rowmax_max_raw":  rm_raw["max"],
            "rowmax_max_svd":  rm_svd["max"],
            "rowmax_max_svdh": rm_svdh["max"],
            "chan_maxmed_raw":  chmm_raw,
            "chan_maxmed_svd":  chmm_svd,
            "chan_maxmed_svdh": chmm_svdh,
        })
        print(f"[{tag}] nMSE raw={nmse_raw['nmse']:.3g}  svd={nmse_svd['nmse']:.3g}  "
              f"svdh={nmse_svdh['nmse']:.3g}   "
              f"rowmax_max raw={rm_raw['max']:.2g}  svd={rm_svd['max']:.2g}  "
              f"svdh={rm_svdh['max']:.2g}   "
              f"chan_maxmed raw={chmm_raw:.2g}  svd={chmm_svd:.2g}  svdh={chmm_svdh:.2g}")

    # ---- figure ----
    fig = plt.figure(figsize=(22, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.0])

    x = np.arange(len(rows))
    bw = 0.27
    labels = [r["tag"] + (" (LLM)" if r["side"] == "LLM" else " (DiT)") for r in rows]

    # Panel 1: End-to-end W4A4 nMSE (bottom-line metric)
    ax1 = fig.add_subplot(gs[0, 0])
    nmse_raw  = np.array([r["nmse_raw"]  for r in rows])
    nmse_svd  = np.array([r["nmse_svd"]  for r in rows])
    nmse_svdh = np.array([r["nmse_svdh"] for r in rows])
    ax1.bar(x - bw, nmse_raw,  width=bw, label="raw (no rotation)", color="tab:gray")
    ax1.bar(x,      nmse_svd,  width=bw, label="SVD only",           color="tab:orange")
    ax1.bar(x + bw, nmse_svdh, width=bw, label="SVD-Hadamard (A2-lite)", color="tab:green")
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=70, fontsize=8, ha="right")
    ax1.set_ylabel(f"end-to-end W{args.w_bits}A{args.a_bits} normalized MSE (lower = better)")
    ax1.set_title("Bottom line: layer-output quant MSE under W4A4")
    ax1.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=9)
    # annotate
    for xi, (a, b, c) in enumerate(zip(nmse_raw, nmse_svd, nmse_svdh)):
        ax1.text(xi - bw, a * 1.05, f"{a:.1e}", ha="center", fontsize=6)
        ax1.text(xi,      b * 1.05, f"{b:.1e}", ha="center", fontsize=6)
        ax1.text(xi + bw, c * 1.05, f"{c:.1e}", ha="center", fontsize=6, color="tab:green")

    # Panel 2: Per-row max p99 (drives A4 scale on heavy rows)
    ax2 = fig.add_subplot(gs[0, 1])
    p99_raw  = np.array([r["rowmax_p99_raw"]  for r in rows])
    p99_svd  = np.array([r["rowmax_p99_svd"]  for r in rows])
    p99_svdh = np.array([r["rowmax_p99_svdh"] for r in rows])
    ax2.bar(x - bw, p99_raw,  width=bw, label="raw", color="tab:gray")
    ax2.bar(x,      p99_svd,  width=bw, label="SVD only", color="tab:orange")
    ax2.bar(x + bw, p99_svdh, width=bw, label="SVD-Hadamard", color="tab:green")
    ax2.set_yscale("log")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=70, fontsize=8, ha="right")
    ax2.set_ylabel("p99 of per-row max(|X·R|)  (drives A4 scale on heavy rows)")
    ax2.set_title("A4 scale ceiling: 99th-percentile token-max")
    ax2.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9)
    for xi, (a, b, c) in enumerate(zip(p99_raw, p99_svd, p99_svdh)):
        ax2.text(xi - bw, a * 1.05, f"{a:.1f}", ha="center", fontsize=6)
        ax2.text(xi,      b * 1.05, f"{b:.1f}", ha="center", fontsize=6)
        ax2.text(xi + bw, c * 1.05, f"{c:.1f}", ha="center", fontsize=6, color="tab:green")

    # Panel 3: Row-max CDF for the worst layer
    ax3 = fig.add_subplot(gs[0, 2])
    hot_tag = max(rows, key=lambda r: r["nmse_raw"])["tag"]
    rmd = rowmax_data[hot_tag]
    def _ecdf(v):
        v = np.sort(v); n = v.size
        return v, np.arange(1, n + 1) / n
    for label, vals, col in [
        ("raw", rmd["raw"], "tab:gray"),
        ("SVD only", rmd["svd"], "tab:orange"),
        ("SVD-Hadamard", rmd["svdh"], "tab:green"),
    ]:
        xs, ys = _ecdf(vals)
        ax3.plot(xs, ys, color=col, linewidth=1.6, label=label)
    ax3.set_xscale("log")
    ax3.set_xlabel("per-row max(|X·R|)   (log scale)")
    ax3.set_ylabel("CDF over tokens")
    ax3.set_title(f"Row-max CDF — worst layer {hot_tag}\n(curve LEFT = tighter A4 scale)")
    ax3.grid(True, which="both", linestyle=":", alpha=0.4)
    ax3.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        f"SVD vs SVD-Hadamard rotation — true W{args.w_bits}A{args.a_bits} quantization metrics  "
        f"[GR00T-N1.5, suite={args.task_suite_name}, calib={args.num_samples}, block={args.block_size}]",
        fontsize=13,
    )

    out_png = out_dir / "svd_vs_svdh_quant_mse.png"
    fig.savefig(out_png, dpi=150)
    print(f"[SVDH-QERR] wrote {out_png}")

    csv_path = out_dir / "svd_vs_svdh_quant_mse.csv"
    with open(csv_path, "w") as f:
        f.write("tag,side,n_tokens,in_f,nmse_raw,nmse_svd,nmse_svdh,"
                "rowmax_p99_raw,rowmax_p99_svd,rowmax_p99_svdh,"
                "rowmax_max_raw,rowmax_max_svd,rowmax_max_svdh,"
                "chan_maxmed_raw,chan_maxmed_svd,chan_maxmed_svdh\n")
        for r in rows:
            f.write(",".join([
                r["tag"], r["side"], str(r["n_tokens"]), str(r["in_f"]),
                f"{r['nmse_raw']:.6g}", f"{r['nmse_svd']:.6g}", f"{r['nmse_svdh']:.6g}",
                f"{r['rowmax_p99_raw']:.6g}", f"{r['rowmax_p99_svd']:.6g}", f"{r['rowmax_p99_svdh']:.6g}",
                f"{r['rowmax_max_raw']:.6g}", f"{r['rowmax_max_svd']:.6g}", f"{r['rowmax_max_svdh']:.6g}",
                f"{r['chan_maxmed_raw']:.6g}", f"{r['chan_maxmed_svd']:.6g}", f"{r['chan_maxmed_svdh']:.6g}",
            ]) + "\n")
    print(f"[SVDH-QERR] wrote {csv_path}")


if __name__ == "__main__":
    main()
