"""Build a pi0.5 SVDQuant pack (PaliGemma backbone + Gemma expert).

OpenPI/pi0.5 sibling of tools/build_dit_svdquant_weights.py — same per-layer
SVDQuant pipeline (SmoothQuant migration + low-rank FP16 head + GPTQ residual +
per-step act_scale_table) but adapted to the pi0.5 PyTorch model:

  - LLM side  = paligemma.model.language_model    → single-step bucket
  - Expert    = gemma_expert.model                 → per-step (uses pi0.5's
                                                    sample_actions loop with
                                                    set_dit_quant_step(t))

Output pack format matches GptqLinear's runtime loader (format=dit_svdquant_v1
per-layer record). No base pack needed; each layer's record stands alone.

Requires the openpi venv (Python 3.11 with JAX + PyTorch + openpi). Driven by
the obs pickle produced by tools/record_libero_obs_for_pi05.py.

Usage:
  $OPENPI_ROOT/.venv/bin/python -m tools.build_pi05_svdquant_weights \\
      --checkpoint $CHECKPOINTS_ROOT/pi05_libero_pytorch \\
      --data-config pi05_libero \\
      --obs-path duquant_act_stats/pi05_libero_object_obs.pt \\
      --output results/multisuite_packs/pi05_object_SVDQ_custom_W4A4/quantized.pt \\
      --quant-type custom --side both \\
      --w-bits 4 --a-bits 4 --svd-rank 16 \\
      --num-samples 10 --token-cap 512 --num-steps 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gr00t.quantization.dit_step_context import get_current_dit_step  # noqa: E402
from gr00t.quantization.gptq_layers import gptq_quantize_weight  # noqa: E402
from gr00t.quantization.duquant_preprocess import qmax  # noqa: E402


# Layer-name regexes for pi0.5 (PyTorch HF-style).
EXPERT_PATTERN = re.compile(
    r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\..*"
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
PALIGEMMA_PATTERN = re.compile(
    r"paligemma_with_expert\.paligemma\.model\.language_model\.layers\.\d+\..*"
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
SIDE_PATTERNS = {"expert": EXPERT_PATTERN, "paligemma": PALIGEMMA_PATTERN}

DEFAULT_INCLUDE = (
    r".*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)"
    r"\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
DEFAULT_EXCLUDE = (
    r"(?:^|\.)(vision_tower|vision_model|embeddings|embed_tokens|norm|layernorm|lm_head)(?:\.|$)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-config", default="pi05_libero")
    p.add_argument("--obs-path", required=True,
                   help="Pickled obs list from tools/record_libero_obs_for_pi05.py")
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--token-cap", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=10,
                   help="pi0.5 flow-matching denoising steps")
    p.add_argument("--device", default="cuda")

    p.add_argument("--side", choices=["expert", "paligemma", "both"], default="both")
    p.add_argument("--include-regex", default=DEFAULT_INCLUDE)
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE)

    p.add_argument("--w-bits", type=int, default=4)
    p.add_argument("--a-bits", type=int, default=4, choices=[4, 8])
    p.add_argument("--svd-rank", type=int, default=16)
    p.add_argument("--quant-type", choices=["original", "custom"], default="custom",
                   help="custom = GPTQ(W_s) then SVD residual; original = SVD W_s directly")

    p.add_argument("--sq-alpha", type=float, default=0.5)
    p.add_argument("--sq-clamp-lo", type=float, default=0.5)
    p.add_argument("--sq-clamp-hi", type=float, default=2.0)

    p.add_argument("--gptq-block-size", type=int, default=128)
    p.add_argument("--gptq-damp-percent", type=float, default=0.01)
    p.add_argument("--gptq-reg-lambda", type=float, default=0.0,
                   help="Absolute L2 reg on (W-Q): adds λI to GPTQ Hessian. "
                        "Helps when cal H is rank-deficient or noisy.")
    p.add_argument("--gptq-err-comp-gamma", type=float, default=1.0,
                   help="Error-compensation strength: 1.0 = full GPTQ, 0.0 = no "
                        "propagation (per-block-MSE RTN), in-between = scaled propagation.")

    p.add_argument("--act-percentile", type=float, default=99.9)
    p.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--max-layers", type=int, default=0)
    return p.parse_args()


def load_pi05_policy(checkpoint: str, data_config: str, device: str):
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    train_cfg = _config.get_config(data_config)
    policy = _policy_config.create_trained_policy(train_cfg, checkpoint, pytorch_device=device)
    if not getattr(policy, "_is_pytorch_model", False):
        raise RuntimeError("policy is not PyTorch; convert JAX→PyTorch first")
    model = policy._model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return policy, model


def main() -> None:
    args = parse_args()
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    a_qmax = float(qmax(args.a_bits))
    device = torch.device(args.device)

    print(f"[SVDQ-PI05] loading policy {args.checkpoint}", flush=True)
    policy, model = load_pi05_policy(args.checkpoint, args.data_config, args.device)

    # Resolve target Linear layers
    inc_re = re.compile(args.include_regex)
    exc_re = re.compile(args.exclude_regex) if args.exclude_regex else None
    target_names: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if exc_re is not None and exc_re.search(name):
            continue
        if not inc_re.match(name):
            continue
        if args.side != "both":
            if not SIDE_PATTERNS[args.side].search(name):
                continue
        target_names.append(name)
    if args.max_layers > 0:
        target_names = target_names[: args.max_layers]
    print(f"[SVDQ-PI05] target layers: {len(target_names)} (side={args.side})", flush=True)
    if not target_names:
        raise SystemExit("no target layers matched")

    # Classify each target as expert (per-step) or paligemma (single-bucket)
    is_expert = {n: bool(EXPERT_PATTERN.search(n)) for n in target_names}

    # Capture FP weights up-front (before any wrap)
    fp_weights: Dict[str, torch.Tensor] = {}
    for name in target_names:
        mod = dict(model.named_modules())[name]
        fp_weights[name] = mod.weight.detach().clone().to(torch.float32)
    print(f"[SVDQ-PI05] captured FP weights for {len(fp_weights)} layers", flush=True)

    # Hook layers; bucket by step via dit_step_context (or t=0 for paligemma)
    cache: Dict[str, Dict[int, List[torch.Tensor]]] = {n: defaultdict(list) for n in target_names}

    def make_hook(name: str):
        def hook(_m, inputs):
            if not inputs:
                return
            x = inputs[0]
            if x is None:
                return
            x = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
            if x.shape[0] > args.token_cap:
                idx = torch.randperm(x.shape[0], device=x.device)[: args.token_cap]
                x = x[idx]
            if is_expert[name]:
                t = get_current_dit_step()
                if t is None:
                    return  # only collect inside denoising loop
            else:
                t = 0  # paligemma backbone has no step concept
            cache[name][int(t)].append(x.cpu())
        return hook

    handles = []
    name_to_mod = dict(model.named_modules())
    for name in target_names:
        handles.append(name_to_mod[name].register_forward_pre_hook(make_hook(name)))

    # Drive cal forward
    samples = torch.load(args.obs_path, weights_only=False)
    if isinstance(samples, dict) and "samples" in samples:
        samples = samples["samples"]
    samples = samples[: args.max_samples]
    print(f"[SVDQ-PI05] driving policy with {len(samples)} obs samples", flush=True)
    try:
        with torch.no_grad():
            for i, obs in enumerate(samples, 1):
                lang = obs.get("prompt", "")
                print(f"[SVDQ-PI05] sample {i}/{len(samples)}: {lang!r}", flush=True)
                _ = policy.infer(obs)
    finally:
        for h in handles:
            h.remove()
    print(f"[SVDQ-PI05] activation capture done", flush=True)

    # Concatenate per-layer per-step
    X_per_step: Dict[str, Dict[int, torch.Tensor]] = {}
    X_total: Dict[str, torch.Tensor] = {}
    num_steps_table_expert = args.num_steps
    for name in target_names:
        n_buckets = num_steps_table_expert if is_expert[name] else 1
        per_step = {}
        all_chunks = []
        for t in range(n_buckets):
            chunks = cache[name].get(t, [])
            if chunks:
                X = torch.cat(chunks, dim=0)
                per_step[t] = X
                all_chunks.append(X)
        X_per_step[name] = per_step
        if all_chunks:
            X_total[name] = torch.cat(all_chunks, dim=0)
        else:
            X_total[name] = torch.empty(0, fp_weights[name].shape[1])
    cache.clear()

    # Free policy GPU memory before per-layer GPTQ
    del policy
    torch.cuda.empty_cache()

    # Per-layer SVDQ build (identical to build_dit_svdquant_weights.py inner loop)
    eps = 1e-8
    p = float(args.act_percentile) / 100.0
    new_records: Dict[str, Dict] = {}
    diag_rows: List[Dict] = []

    for li, name in enumerate(target_names):
        t0 = time.time()
        W = fp_weights[name].to(device)
        out_features, in_features = W.shape
        Xt = X_total[name]
        if Xt.shape[0] == 0:
            print(f"[SVDQ-PI05][skip] {name}: no activations captured", flush=True)
            continue
        Xt_dev = Xt.to(device)

        # SmoothQuant
        A = torch.zeros(in_features, dtype=torch.float32, device=device)
        for t_idx, X_t in X_per_step[name].items():
            q_t = torch.quantile(X_t.to(device).abs(), 0.999, dim=0)
            A = torch.maximum(A, q_t)
        A = A.clamp_min(eps)
        B = W.detach().abs().amax(dim=0).clamp_min(eps)
        raw_s = (A.pow(args.sq_alpha)) / (B.pow(1.0 - args.sq_alpha))
        med = raw_s.median().clamp_min(eps)
        s = torch.clamp(raw_s / med, min=args.sq_clamp_lo, max=args.sq_clamp_hi).to(torch.float32)
        W_s = W * s[None, :]
        Xs_total = Xt_dev / s[None, :]
        n_total = Xs_total.shape[0]
        H_s = (Xs_total.T @ Xs_total) / float(max(n_total, 1))
        H_s = H_s.to(torch.float32)

        # SVD target
        if args.quant_type == "custom":
            W4_tmp = gptq_quantize_weight(W_s, H_s, bits=int(args.w_bits),
                                          block_size=int(args.gptq_block_size),
                                          damp_percent=float(args.gptq_damp_percent),
                                          reg_lambda=float(args.gptq_reg_lambda),
                                          err_comp_gamma=float(args.gptq_err_comp_gamma))
            svd_target = W_s - W4_tmp
            E_norm_rel = float(svd_target.norm().item() / max(W_s.norm().item(), eps))
        else:
            W4_tmp = None
            svd_target = W_s
            E_norm_rel = 1.0

        # SVD
        if args.svd_rank > 0:
            try:
                U_r, S_r, V_r = torch.svd_lowrank(svd_target.to(torch.float32), q=int(args.svd_rank) + 6)
                U_r = U_r[:, : args.svd_rank]
                S_r = S_r[: args.svd_rank]
                V_r = V_r[:, : args.svd_rank]
            except Exception:
                U_full, S_full, Vh_full = torch.linalg.svd(svd_target.to(torch.float32), full_matrices=False)
                U_r = U_full[:, : args.svd_rank]
                S_r = S_full[: args.svd_rank]
                V_r = Vh_full.t()[:, : args.svd_rank]
            sqrtS = S_r.clamp_min(0).sqrt()
            lowrank_A = U_r * sqrtS[None, :]
            lowrank_B = V_r * sqrtS[None, :]
            E_recon = lowrank_A @ lowrank_B.T
            energy = float((S_r.pow(2).sum() / svd_target.pow(2).sum().clamp_min(eps)).item())
        else:
            lowrank_A = torch.zeros(out_features, 0, dtype=torch.float32, device=device)
            lowrank_B = torch.zeros(in_features, 0, dtype=torch.float32, device=device)
            E_recon = torch.zeros_like(svd_target)
            energy = 0.0

        # Residual GPTQ
        W_res = W_s - E_recon
        W_res_q = gptq_quantize_weight(W_res, H_s, bits=int(args.w_bits),
                                       block_size=int(args.gptq_block_size),
                                       damp_percent=float(args.gptq_damp_percent),
                                       reg_lambda=float(args.gptq_reg_lambda),
                                       err_comp_gamma=float(args.gptq_err_comp_gamma))
        recon = W_res_q + E_recon
        recon_err = float((W_s - recon).norm().item() / max(W_s.norm().item(), eps))

        # Per-step act_scale_table
        n_buckets = num_steps_table_expert if is_expert[name] else 1
        table = torch.zeros(n_buckets, in_features, dtype=torch.float32, device=device)
        clip_per_step = []
        n_per_step = []
        for t_idx in range(n_buckets):
            X_t = X_per_step[name].get(t_idx)
            if X_t is None or X_t.numel() == 0:
                n_per_step.append(0)
                clip_per_step.append(0.0)
                continue
            Xs_t = X_t.to(device) / s[None, :]
            q_vec = torch.quantile(Xs_t.abs(), p, dim=0).clamp_min(1e-6)
            scale = q_vec / a_qmax
            table[t_idx] = scale
            clip_frac = float(((Xs_t.abs() > q_vec[None, :]).any(dim=1).float()).mean().item())
            n_per_step.append(int(Xs_t.shape[0]))
            clip_per_step.append(clip_frac)

        for t_name, t_obj in [("smooth_scale", s), ("act_scale_table", table),
                              ("lowrank_A", lowrank_A), ("lowrank_B", lowrank_B),
                              ("weight_res_q", W_res_q)]:
            if not torch.isfinite(t_obj).all():
                raise RuntimeError(f"{name}: {t_name} has non-finite values")

        rec = {
            "format": "dit_svdquant_v1",
            "weight_res_q": W_res_q.to(save_dtype).cpu().contiguous(),
            "lowrank_A": lowrank_A.to(save_dtype).cpu().contiguous(),
            "lowrank_B": lowrank_B.to(save_dtype).cpu().contiguous(),
            "smooth_scale": s.to(save_dtype).cpu().contiguous(),
            "act_scale_table": table.to(torch.float32).cpu().contiguous(),
            "weight_bits": int(args.w_bits),
            "a_bits": int(args.a_bits),
            "rank": int(args.svd_rank),
            "sq_alpha": float(args.sq_alpha),
            "sq_clamp": [float(args.sq_clamp_lo), float(args.sq_clamp_hi)],
            "act_percentile": float(args.act_percentile),
            "in_features": int(in_features),
            "out_features": int(out_features),
            "n_calib_total": int(n_total),
            "n_calib_per_step": n_per_step,
            "diag_E_norm_rel": E_norm_rel,
            "diag_svd_energy": energy,
            "diag_recon_err": recon_err,
            "diag_clip_per_step": clip_per_step,
            "diag_quant_type": args.quant_type,
            "diag_side": "expert" if is_expert[name] else "paligemma",
        }
        new_records[name] = rec
        diag_rows.append({"name": name, "out": out_features, "in": in_features,
                          "E_norm_rel": E_norm_rel, "svd_energy": energy,
                          "recon_err": recon_err, "n_total": n_total,
                          "mean_clip": float(np.mean(clip_per_step or [0.0])),
                          "side": "expert" if is_expert[name] else "paligemma"})

        elapsed = time.time() - t0
        if li < 3 or (li + 1) % 12 == 0:
            print(f"[SVDQ-PI05] {li+1}/{len(target_names)} {name}: "
                  f"||E||_rel={E_norm_rel:.4f} svd_energy={energy:.3f} "
                  f"recon_err={recon_err:.4f} n_total={n_total} elapsed={elapsed:.1f}s",
                  flush=True)

        del W, W_s, H_s, svd_target, E_recon, W_res, W_res_q
        if W4_tmp is not None:
            del W4_tmp
        del lowrank_A, lowrank_B, table, Xt_dev, Xs_total
        torch.cuda.empty_cache()

    # Save pack
    meta = {
        "pack_format": "pi05_svdquant_v1",
        "svdquant_w_bits": int(args.w_bits),
        "svdquant_a_bits": int(args.a_bits),
        "svdquant_rank": int(args.svd_rank),
        "svdquant_act_percentile": float(args.act_percentile),
        "svdquant_alpha": float(args.sq_alpha),
        "svdquant_clamp": [float(args.sq_clamp_lo), float(args.sq_clamp_hi)],
        "svdquant_num_samples": int(len(samples)),
        "svdquant_token_cap": int(args.token_cap),
        "svdquant_num_steps": int(args.num_steps),
        "svdquant_quant_type": args.quant_type,
        "svdquant_side": args.side,
    }
    merged: Dict = {"__meta__": meta}
    merged.update(new_records)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + f".tmp.{os.getpid()}"
    torch.save(merged, tmp)
    os.replace(tmp, out_path)
    print(f"[SVDQ-PI05] wrote {out_path} ({len(new_records)} records)", flush=True)

    diag_rows.sort(key=lambda r: r["recon_err"], reverse=True)
    diag_path = out_path.with_suffix(".diag.json")
    with open(diag_path, "w") as f:
        json.dump({"layers": diag_rows, "meta": meta}, f, indent=2)
    print(f"[SVDQ-PI05] wrote {diag_path}", flush=True)


if __name__ == "__main__":
    main()
