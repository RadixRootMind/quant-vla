"""pi0.5 version of diagnose_svd_vs_svdh_quanterr.py.

Same metrics, same plot layout, but targets pi0.5 paligemma (LLM) + gemma_expert
(DiT analog) layers. Uses pre-recorded LIBERO obs (pi05_libero_<suite>_obs.pt).

Run under openpi venv:

  CUDA_VISIBLE_DEVICES=7 LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH \\
  $OPENPI_ROOT/.venv/bin/python -u -m tools.diagnose_svd_vs_svdh_quanterr_pi05 \\
      --checkpoint $CHECKPOINTS_ROOT/pi05_libero_pytorch \\
      --obs-path duquant_act_stats/pi05_libero_object_obs.pt \\
      --num-samples 4 --block-size 64 \\
      --output-dir experiment_results/visualization_for_paper
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


# ------------------------------------------------------------ helpers


def hadamard_matrix(n: int, device, dtype) -> torch.Tensor:
    assert n & (n - 1) == 0, f"Hadamard requires power-of-two dim, got {n}"
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def block_svd_rotation(W: torch.Tensor, block_size: int, with_hadamard: bool) -> torch.Tensor:
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
    qmax = (1 << (bits - 1)) - 1
    scale = t.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
    return torch.round(t / scale).clamp_(-qmax, qmax) * scale


def quant_output_mse(X: torch.Tensor, W: torch.Tensor, R: torch.Tensor | None,
                     w_bits: int = 4, a_bits: int = 4) -> dict:
    y = X.float() @ W.float().T
    if R is None:
        X_r = X; W_r = W
    else:
        X_r = X @ R; W_r = W @ R
    X_q = fake_quant_sym_per_row(X_r.float(), a_bits)
    W_q = fake_quant_sym_per_row(W_r.float(), w_bits)
    y_hat = X_q @ W_q.T
    delta = y_hat - y
    nmse = float((delta.pow(2).mean() / y.pow(2).mean().clamp_min(1e-12)).item())
    return {"nmse": nmse, "n_tokens": int(X.shape[0])}


def row_max_stats(X: torch.Tensor) -> dict:
    rm = X.abs().float().max(dim=-1).values
    return {
        "p50":  float(rm.median().item()),
        "p99":  float(torch.quantile(rm, 0.99).item()),
        "max":  float(rm.max().item()),
        "vals_cpu": rm.detach().cpu().numpy(),
    }


def chan_q999_maxmed(X: torch.Tensor) -> float:
    q = torch.quantile(X.abs().float(), 0.999, dim=0).cpu().numpy()
    return float(q.max() / (np.median(q) + 1e-12))


# ------------------------------------------------------------ pi0.5 layer probes


# pi0.5 has 18 layers each side, 7 projs each (q/k/v/o + gate/up/down).
# Mirror the GR00T figure's choices: 3 layers spread through depth.
LLM_TARGETS = [
    ("L02.down_proj", "paligemma_with_expert.paligemma.model.language_model.layers.2.mlp.down_proj"),
    ("L08.down_proj", "paligemma_with_expert.paligemma.model.language_model.layers.8.mlp.down_proj"),
    ("L14.down_proj", "paligemma_with_expert.paligemma.model.language_model.layers.14.mlp.down_proj"),
    ("L02.q_proj",    "paligemma_with_expert.paligemma.model.language_model.layers.2.self_attn.q_proj"),
    ("L02.gate_proj", "paligemma_with_expert.paligemma.model.language_model.layers.2.mlp.gate_proj"),
]
DIT_TARGETS = [
    ("L02.q_proj",    "paligemma_with_expert.gemma_expert.model.layers.2.self_attn.q_proj"),
    ("L02.down_proj", "paligemma_with_expert.gemma_expert.model.layers.2.mlp.down_proj"),
    ("L08.q_proj",    "paligemma_with_expert.gemma_expert.model.layers.8.self_attn.q_proj"),
    ("L08.down_proj", "paligemma_with_expert.gemma_expert.model.layers.8.mlp.down_proj"),
    ("L14.q_proj",    "paligemma_with_expert.gemma_expert.model.layers.14.self_attn.q_proj"),
]


def load_pi05_policy(checkpoint: str, data_config: str, device: str):
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    train_cfg = _config.get_config(data_config)
    policy = _policy_config.create_trained_policy(train_cfg, checkpoint, pytorch_device=device)
    if not getattr(policy, "_is_pytorch_model", False):
        raise RuntimeError("policy is not PyTorch; convert JAX→PyTorch first")
    model = policy._model
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return policy, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--obs-path", required=True)
    ap.add_argument("--data-config", default="pi05_libero")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--token-cap", type=int, default=4096)
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=4)
    ap.add_argument("--output-dir", default="experiment_results/visualization_for_paper")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"[SVDH-QERR-PI05] loading policy {args.checkpoint}")
    policy, model = load_pi05_policy(args.checkpoint, args.data_config, args.device)

    targets = LLM_TARGETS + DIT_TARGETS
    # Disambiguate tags between LLM/DiT (both have L02.down_proj etc.)
    tagged = [(("LLM-" + tag) if (tag, name) in LLM_TARGETS else ("DiT-" + tag), name)
              for tag, name in targets]

    cache: dict[str, list[torch.Tensor]] = {tag: [] for tag, _ in tagged}
    name_to_mod = dict(model.named_modules())
    found: dict[str, torch.nn.Module] = {}
    handles = []

    def make_hook(tag: str):
        def hook(_m, inputs):
            if not inputs: return
            x = inputs[0]
            if x is None: return
            x = x.detach().reshape(-1, x.shape[-1]).float()
            if x.shape[0] > args.token_cap:
                idx = torch.randperm(x.shape[0], device=x.device)[: args.token_cap]
                x = x[idx]
            cache[tag].append(x.cpu())
        return hook

    for tag, name in tagged:
        mod = name_to_mod.get(name)
        if mod is None:
            print(f"[skip-hook] {tag}: missing module {name}")
            continue
        handles.append(mod.register_forward_pre_hook(make_hook(tag)))
        found[tag] = mod

    print(f"[SVDH-QERR-PI05] {len(handles)} hooks ready")

    obs_data = torch.load(args.obs_path, weights_only=False)
    samples = obs_data["samples"] if isinstance(obs_data, dict) and "samples" in obs_data else obs_data
    samples = samples[: args.num_samples]
    print(f"[SVDH-QERR-PI05] driving {len(samples)} obs samples")
    with torch.no_grad():
        for i, obs in enumerate(samples, 1):
            print(f"[SVDH-QERR-PI05] sample {i}/{len(samples)}: {obs.get('prompt','')!r}")
            _ = policy.infer(obs)
    for h in handles:
        h.remove()

    rows = []
    rowmax_data: dict[str, dict[str, np.ndarray]] = {}

    for tag, _ in tagged:
        if tag not in found:
            continue
        chunks = cache[tag]
        if not chunks:
            print(f"[skip] {tag}: no activations")
            continue
        X = torch.cat(chunks, dim=0).to(device=device)
        W = found[tag].weight.detach().to(device=device, dtype=torch.float32)
        in_f = X.shape[1]
        if in_f % args.block_size != 0:
            print(f"[skip] {tag}: in_f={in_f} not divisible by block_size={args.block_size}")
            continue

        R_svd  = block_svd_rotation(W, args.block_size, with_hadamard=False)
        R_svdh = block_svd_rotation(W, args.block_size, with_hadamard=True)

        nmse_raw  = quant_output_mse(X, W, None,    w_bits=args.w_bits, a_bits=args.a_bits)
        nmse_svd  = quant_output_mse(X, W, R_svd,  w_bits=args.w_bits, a_bits=args.a_bits)
        nmse_svdh = quant_output_mse(X, W, R_svdh, w_bits=args.w_bits, a_bits=args.a_bits)

        rm_raw  = row_max_stats(X)
        rm_svd  = row_max_stats(X @ R_svd)
        rm_svdh = row_max_stats(X @ R_svdh)
        rowmax_data[tag] = {
            "raw":  rm_raw["vals_cpu"],
            "svd":  rm_svd["vals_cpu"],
            "svdh": rm_svdh["vals_cpu"],
        }

        chmm_raw  = chan_q999_maxmed(X)
        chmm_svd  = chan_q999_maxmed(X @ R_svd)
        chmm_svdh = chan_q999_maxmed(X @ R_svdh)

        rows.append({
            "tag": tag,
            "side": "LLM" if tag.startswith("LLM-") else "DiT",
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

    # ---- figure (linear y, matching the linear gr00t plot) ----
    fig = plt.figure(figsize=(22, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)
    x = np.arange(len(rows))
    bw = 0.27
    labels = [r["tag"].replace("LLM-", "").replace("DiT-", "")
              + (" (LLM)" if r["side"] == "LLM" else " (DiT)") for r in rows]

    nmse_raw  = np.array([r["nmse_raw"]  for r in rows])
    nmse_svd  = np.array([r["nmse_svd"]  for r in rows])
    nmse_svdh = np.array([r["nmse_svdh"] for r in rows])
    p99_raw   = np.array([r["rowmax_p99_raw"]  for r in rows])
    p99_svd   = np.array([r["rowmax_p99_svd"]  for r in rows])
    p99_svdh  = np.array([r["rowmax_p99_svdh"] for r in rows])
    ratio_svd  = nmse_svd  / np.maximum(nmse_raw, 1e-12)
    ratio_svdh = nmse_svdh / np.maximum(nmse_raw, 1e-12)

    # Panel 1: end-to-end W4A4 nMSE (linear)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(x - bw, nmse_raw,  width=bw, label="raw (no rotation)", color="tab:gray")
    ax1.bar(x,      nmse_svd,  width=bw, label="SVD only", color="tab:orange")
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

    # Panel 2: A4 scale ceiling (linear)
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

    # Panel 3: ratio vs raw (linear; cap if extreme)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(x - bw/2, ratio_svd,  width=bw, label="SVD only / raw", color="tab:orange")
    ax3.bar(x + bw/2, ratio_svdh, width=bw, label="SVD-Hadamard / raw", color="tab:green")
    ax3.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, rotation=70, fontsize=8, ha="right")
    ax3.set_ylabel("nMSE(rotated) / nMSE(raw)   ( <1 = rotation HELPS )")
    ax3.set_title("Relative quant MSE vs raw (linear y)")
    ax3.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=9)
    ratio_ymax = float(max(ratio_svd.max(), ratio_svdh.max()) * 1.1)
    ax3.set_ylim(0.0, ratio_ymax)
    for xi, (b, c) in enumerate(zip(ratio_svd, ratio_svdh)):
        ax3.text(xi - bw/2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=6)
        ax3.text(xi + bw/2, c, f"{c:.2f}", ha="center", va="bottom", fontsize=6, color="tab:green")

    suite = Path(args.obs_path).stem
    fig.suptitle(
        f"pi0.5 — SVD vs SVD-Hadamard rotation, true W4A4 quant metrics (LINEAR)   "
        f"[{suite}, calib={args.num_samples}, block={args.block_size}]",
        fontsize=13,
    )

    out_png = out_dir / "pi05_svd_vs_svdh_quant_mse_linear.png"
    fig.savefig(out_png, dpi=150)
    print(f"[SVDH-QERR-PI05] wrote {out_png}")

    csv_path = out_dir / "pi05_svd_vs_svdh_quant_mse.csv"
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
    print(f"[SVDH-QERR-PI05] wrote {csv_path}")


if __name__ == "__main__":
    main()
