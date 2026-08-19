"""Quick activation-rotation comparison: SVD vs SVD-Hadamard (A2-lite).

For a selected handful of GR00T layers (with known outlier severity), drive
N LIBERO observations through the policy, hook each layer's input X, and
compute per-input-channel q999(|X · R|) under three rotations:

  * raw            : R = I
  * svd            : R = block-diag of U_b   (left singular vectors of W_block^T)
  * svd_hadamard   : R = block-diag of (U_b @ H_b)   (A2-lite)

Then write a 3-panel figure:

  panel 1 (per-layer overlay): sorted-desc per-channel range, raw vs svd vs svdh
  panel 2 (max/median bars):  heavy-tail strength across all probed layers
  panel 3 (W4 per-row scale-spread bars): per-row max(|X·R|) / median(|X·R|)
                                          — drives quantization scale variance

Run under the awesome_quant_vla conda env (has LIBERO simulator + torch + matplotlib):

  CUDA_VISIBLE_DEVICES=0 LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH \\
  python \\
    -m tools.diagnose_svd_vs_svdh_quickcap \\
    --checkpoint $CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain \\
    --task-suite-name libero_object \\
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

from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    ensure_libero_runtime, load_libero_samples, load_policy, seed_everything,
    get_named_module,
)
from tools.analyze_layerwise_quant_drift import normalized_input_no_inference  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402


# Probe layers — pick the "interesting" ones: known LLM outlier hot layers + a
# couple of DiT layers (relatively flat, as control).
LLM_TARGETS = [
    ("L02.down_proj", "backbone.eagle_model.language_model.model.layers.2.mlp.down_proj"),
    ("L05.down_proj", "backbone.eagle_model.language_model.model.layers.5.mlp.down_proj"),
    ("L09.down_proj", "backbone.eagle_model.language_model.model.layers.9.mlp.down_proj"),
    ("L02.q_proj",    "backbone.eagle_model.language_model.model.layers.2.self_attn.q_proj"),
    ("L02.gate_proj", "backbone.eagle_model.language_model.model.layers.2.mlp.gate_proj"),
]
DIT_TARGETS = [
    ("L02.to_q", "action_head.model.transformer_blocks.2.attn1.to_q"),
    ("L02.ff.down", "action_head.model.transformer_blocks.2.ff.net.2"),
    ("L08.to_q", "action_head.model.transformer_blocks.8.attn1.to_q"),
    ("L08.ff.down", "action_head.model.transformer_blocks.8.ff.net.2"),
    ("L14.to_q", "action_head.model.transformer_blocks.14.attn1.to_q"),
]


def hadamard_matrix(n: int, device, dtype) -> torch.Tensor:
    assert n & (n - 1) == 0, f"Hadamard requires power-of-two dim, got {n}"
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def svd_only_rotation(W: torch.Tensor, block_size: int) -> torch.Tensor:
    """Block-diag rotation built from per-block SVD basis (no Hadamard)."""
    out_f, in_f = W.shape
    if in_f % block_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by block_size={block_size}")
    n_blocks = in_f // block_size
    R = torch.zeros((in_f, in_f), device=W.device, dtype=torch.float32)
    Wf = W.detach().float()
    for b in range(n_blocks):
        s = b * block_size
        e = s + block_size
        U, _, _ = torch.linalg.svd(Wf[:, s:e].T, full_matrices=True)
        R[s:e, s:e] = U
    return R


def svd_hadamard_rotation(W: torch.Tensor, block_size: int) -> torch.Tensor:
    """A2-lite rotation = block-diag of (U_b @ H_b)."""
    out_f, in_f = W.shape
    if in_f % block_size != 0:
        raise ValueError(f"in_features={in_f} not divisible by block_size={block_size}")
    n_blocks = in_f // block_size
    H_blk = hadamard_matrix(block_size, W.device, torch.float32)
    R = torch.zeros((in_f, in_f), device=W.device, dtype=torch.float32)
    Wf = W.detach().float()
    for b in range(n_blocks):
        s = b * block_size
        e = s + block_size
        U, _, _ = torch.linalg.svd(Wf[:, s:e].T, full_matrices=True)
        R[s:e, s:e] = U @ H_blk
    return R


def per_channel_q999(X: torch.Tensor) -> np.ndarray:
    """Per-input-channel q999(|X|), unsorted."""
    return torch.quantile(X.abs().float(), 0.999, dim=0).cpu().numpy()


def per_row_max(X: torch.Tensor) -> np.ndarray:
    """Per-row (token) max(|X|) — drives W4A4 per-row scale."""
    return X.abs().float().max(dim=1).values.cpu().numpy()


def ratio_max_over_med(v: np.ndarray) -> float:
    med = float(np.median(v)) + 1e-12
    return float(np.max(v) / med)


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
    ap.add_argument("--output-dir", default="experiment_results/visualization_for_paper")
    args = ap.parse_args()

    ensure_libero_runtime()
    seed_everything(args.seed)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[SVD-VS-SVDH] loading policy {args.checkpoint}")
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    policy.model.eval()
    for p in policy.model.parameters():
        p.requires_grad_(False)

    targets = LLM_TARGETS + DIT_TARGETS
    cache: dict[str, list[torch.Tensor]] = {tag: [] for tag, _ in targets}

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
    name_to_mod = {}
    for tag, name in targets:
        try:
            mod = get_named_module(policy.model, name)
        except Exception as e:
            print(f"[skip-hook] {tag} ({name}): {e}")
            continue
        handles.append(mod.register_forward_pre_hook(make_hook(tag)))
        name_to_mod[tag] = mod

    print(f"[SVD-VS-SVDH] {len(handles)} hooks; driving {args.num_samples} cal samples")
    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    with torch.no_grad():
        for i, s in enumerate(samples):
            seed_everything(s["seed"])
            norm = normalized_input_no_inference(policy, s["obs"])
            policy.model.get_action(norm)
            print(f"[SVD-VS-SVDH] sample {i+1}/{len(samples)}")
    for h in handles:
        h.remove()

    device = torch.device(args.device)
    rows = []
    per_layer_curves: dict[str, dict[str, np.ndarray]] = {}

    for tag, _ in targets:
        if tag not in name_to_mod:
            continue
        chunks = cache[tag]
        if not chunks:
            continue
        X = torch.cat(chunks, dim=0).to(device=device)
        mod = name_to_mod[tag]
        W = mod.weight.detach().to(device=device, dtype=torch.float32)
        in_f = X.shape[1]
        if in_f % args.block_size != 0:
            print(f"[skip] {tag}: in_f={in_f} not divisible by block_size={args.block_size}")
            continue

        R_svd = svd_only_rotation(W, args.block_size)
        R_svdh = svd_hadamard_rotation(W, args.block_size)

        X_raw = X
        X_svd = X @ R_svd
        X_svdh = X @ R_svdh

        # Per-channel q999, sorted desc for overlay plot.
        raw_q = np.sort(per_channel_q999(X_raw))[::-1]
        svd_q = np.sort(per_channel_q999(X_svd))[::-1]
        svdh_q = np.sort(per_channel_q999(X_svdh))[::-1]

        # Per-row max for scale spread (use sorted desc trim).
        raw_r = per_row_max(X_raw)
        svd_r = per_row_max(X_svd)
        svdh_r = per_row_max(X_svdh)

        per_layer_curves[tag] = {
            "raw_chan": raw_q,
            "svd_chan": svd_q,
            "svdh_chan": svdh_q,
        }

        row = {
            "tag": tag,
            "side": "LLM" if tag.startswith("L0") and "to_q" not in tag and "ff" not in tag
                   else ("LLM" if tag in {t for t,_ in LLM_TARGETS} else "DiT"),
            "in_f": int(in_f),
            "n_tokens": int(X.shape[0]),
            "raw_chan_maxmed":  ratio_max_over_med(per_channel_q999(X_raw)),
            "svd_chan_maxmed":  ratio_max_over_med(per_channel_q999(X_svd)),
            "svdh_chan_maxmed": ratio_max_over_med(per_channel_q999(X_svdh)),
            "raw_rowmax_maxmed":  ratio_max_over_med(raw_r),
            "svd_rowmax_maxmed":  ratio_max_over_med(svd_r),
            "svdh_rowmax_maxmed": ratio_max_over_med(svdh_r),
        }
        rows.append(row)
        print(f"[{tag}] chan max/med  raw={row['raw_chan_maxmed']:.2g}  "
              f"svd={row['svd_chan_maxmed']:.2g}  "
              f"svdh={row['svdh_chan_maxmed']:.2g}  "
              f"row max/med  raw={row['raw_rowmax_maxmed']:.2g}  "
              f"svd={row['svd_rowmax_maxmed']:.2g}  svdh={row['svdh_rowmax_maxmed']:.2g}")

    # Re-tag side correctly using the lists.
    llm_tags = {t for t, _ in LLM_TARGETS}
    for r in rows:
        r["side"] = "LLM" if r["tag"] in llm_tags else "DiT"

    # ---- figure ----
    fig = plt.figure(figsize=(20, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1.2])

    # Panel 1: overlay per-channel curves for the most extreme LLM layer
    ax1 = fig.add_subplot(gs[0, 0])
    hot_tag = max(rows, key=lambda r: r["raw_chan_maxmed"])["tag"]
    c = per_layer_curves[hot_tag]
    for label, y, col in [
        ("raw (no rotation)", c["raw_chan"], "tab:gray"),
        ("SVD only", c["svd_chan"], "tab:orange"),
        ("SVD-Hadamard (A2-lite)", c["svdh_chan"], "tab:green"),
    ]:
        x = np.arange(1, y.size + 1) / y.size
        ax1.plot(x, y, color=col, linewidth=1.6, label=label)
    ax1.set_yscale("log")
    ax1.set_xlabel("normalized channel rank (sorted desc)")
    ax1.set_ylabel("per-channel q999(|X·R|)")
    ax1.set_title(f"Worst layer overlay — {hot_tag}\n(SVD only leaves a few channels heavy; SVD-H flattens)")
    ax1.grid(True, which="both", linestyle=":", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: chan max/med bars (3-way) — log y
    ax2 = fig.add_subplot(gs[0, 1])
    labels = [r["tag"] + (" (LLM)" if r["side"] == "LLM" else " (DiT)") for r in rows]
    raw = np.array([r["raw_chan_maxmed"] for r in rows])
    svd = np.array([r["svd_chan_maxmed"] for r in rows])
    svdh = np.array([r["svdh_chan_maxmed"] for r in rows])
    x = np.arange(len(rows))
    bw = 0.27
    ax2.bar(x - bw, raw, width=bw, label="raw", color="tab:gray")
    ax2.bar(x,      svd, width=bw, label="SVD only", color="tab:orange")
    ax2.bar(x + bw, svdh, width=bw, label="SVD-Hadamard", color="tab:green")
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=70, fontsize=8, ha="right")
    ax2.set_ylabel("per-channel max / median (log)")
    ax2.set_title("Activation channel heavy-tail (lower = better quant-friendly)")
    ax2.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9)
    for xi, (a, b, c2) in enumerate(zip(raw, svd, svdh)):
        ax2.text(xi - bw, a * 1.05, f"{a:.0f}", ha="center", fontsize=6)
        ax2.text(xi,      b * 1.05, f"{b:.0f}", ha="center", fontsize=6)
        ax2.text(xi + bw, c2 * 1.05, f"{c2:.1f}", ha="center", fontsize=6, color="tab:green")

    # Panel 3: per-row max(|X·R|) max/med — drives per-row W4A4 scale spread
    ax3 = fig.add_subplot(gs[0, 2])
    raw_r = np.array([r["raw_rowmax_maxmed"] for r in rows])
    svd_r = np.array([r["svd_rowmax_maxmed"] for r in rows])
    svdh_r = np.array([r["svdh_rowmax_maxmed"] for r in rows])
    ax3.bar(x - bw, raw_r, width=bw, label="raw", color="tab:gray")
    ax3.bar(x,      svd_r, width=bw, label="SVD only", color="tab:orange")
    ax3.bar(x + bw, svdh_r, width=bw, label="SVD-Hadamard", color="tab:green")
    ax3.set_yscale("log")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=70, fontsize=8, ha="right")
    ax3.set_ylabel("per-row max / median (log)")
    ax3.set_title("Per-token row-max spread (lower = uniform A4 scale)")
    ax3.grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    ax3.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"DuQuant rotation: SVD only vs SVD-Hadamard (A2-lite)   "
        f"[GR00T-N1.5, suite={args.task_suite_name}, calib={args.num_samples}, block={args.block_size}]",
        fontsize=13,
    )

    out_png = out_dir / "svd_vs_svdh_rotation_compare.png"
    fig.savefig(out_png, dpi=150)
    print(f"[SVD-VS-SVDH] wrote {out_png}")

    # CSV
    csv_path = out_dir / "svd_vs_svdh_rotation_compare.csv"
    with open(csv_path, "w") as f:
        f.write("tag,side,in_f,n_tokens,raw_chan_maxmed,svd_chan_maxmed,svdh_chan_maxmed,"
                "raw_rowmax_maxmed,svd_rowmax_maxmed,svdh_rowmax_maxmed\n")
        for r in rows:
            f.write(f"{r['tag']},{r['side']},{r['in_f']},{r['n_tokens']},"
                    f"{r['raw_chan_maxmed']:.4g},{r['svd_chan_maxmed']:.4g},{r['svdh_chan_maxmed']:.4g},"
                    f"{r['raw_rowmax_maxmed']:.4g},{r['svd_rowmax_maxmed']:.4g},{r['svdh_rowmax_maxmed']:.4g}\n")
    print(f"[SVD-VS-SVDH] wrote {csv_path}")


if __name__ == "__main__":
    main()
