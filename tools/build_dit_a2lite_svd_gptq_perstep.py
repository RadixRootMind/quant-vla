#!/usr/bin/env python
"""Build A2-lite + SVD-r16 + GPTQ + per-step act_scale_table pack for DiT.

Pipeline (per DiT linear, no SmoothQuant):
  1. Capture per-step activations via dit_step_context (groupsize 8 denoise steps)
  2. Compute A2-lite rotation from FP W (perm + block-diag svd_hadamard rotation)
     - rotation_R (combined input rotation) = P @ R_in_full   shape [in × in]
     - rotation_R_out (block-diag output rotation)            shape [out × out]
  3. Rotate W and per-step X:
     - W_r = rotation_R_out @ (W @ rotation_R)
     - X_r per step = X_per_step @ rotation_R
  4. SVD-r16 of W_r:
     - U, S, Vh = svd(W_r);  lowA = U[:,:r]·sqrt(S[:r]);  lowB = (V[:,:r])·sqrt(S[:r])
     - W_res = W_r - lowA @ lowB.T
  5. GPTQ on W_res with damp_percent=0.05:
     - H_r = X_r_total.T @ X_r_total / N
     - W_res_q = gptq_quantize_weight(W_res, H_r, bits=4, damp=0.05)
  6. Per-step act_scale_table on rotated X:
     - act_scale_table[t, j] = quantile(|X_r_t|, 0.999, dim=0) / qmax(a_bits)

Output record format = "dit_svdquant_v1" with:
  weight_res_q, lowrank_A, lowrank_B, duquant_rotation, duquant_rotation_out,
  act_scale_table, weight_bits, a_bits, rank, in_features, out_features

Storage cost vs the existing A2-lite+GPTQ-only pack: +(out+in)*16 FP16 ~ 1.3% of W.

Usage:
  CUDA_VISIBLE_DEVICES=0 python tools/build_dit_a2lite_svd_gptq_perstep.py \\
    --checkpoint $CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain \\
    --task-suite-name libero_object \\
    --output-path results/multisuite_packs/object_DiT_a2lite_svd_gptq_perstep_cal2/quantized.pt \\
    --num-samples 2 --gptq-damp-percent 0.05
"""
from __future__ import annotations
import argparse, os, re, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    ensure_libero_runtime, get_named_module, load_libero_samples, load_policy, seed_everything,
)
from tools.analyze_layerwise_quant_drift import normalized_input_no_inference  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.quantization.dit_step_context import get_current_dit_step  # noqa: E402
from gr00t.quantization.gptq_layers import gptq_quantize_weight  # noqa: E402
from gr00t.quantization.duquant_preprocess import qmax, pack_weight  # noqa: E402

DIT_PATTERN = re.compile(
    r"action_head\.model\.transformer_blocks\.\d+\."
    r"(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--task-suite-name", default="libero_object")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=2)
    p.add_argument("--libero-num-trials-per-task", type=int, default=2)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task",
                   choices=["sequential", "one_per_task", "one_per_trial"])
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--token-cap", type=int, default=1024)
    p.add_argument("--svd-rank", type=int, default=16)
    p.add_argument("--gptq-block-size", type=int, default=128)
    p.add_argument("--gptq-damp-percent", type=float, default=0.05)
    p.add_argument("--use-rtn", action="store_true",
                   help="Skip GPTQ error compensation (err_comp_gamma=0). Each block quantized independently.")
    p.add_argument("--duquant-block-size", type=int, default=64)
    p.add_argument("--duquant-block-out", type=int, default=64)
    p.add_argument("--a-bits", type=int, default=4)
    p.add_argument("--w-bits", type=int, default=4)
    p.add_argument("--act-percentile", type=float, default=99.9)
    p.add_argument("--save-dtype", default="float16", choices=["float16", "float32"])
    return p.parse_args()


def build_rotation(W: torch.Tensor, block: int, block_out: int, device):
    """Build A2-lite (svd_hadamard) rotation matrices: P @ R_in_full, R_out_full.

    Uses gr00t.quantization.duquant_preprocess.pack_weight with rot_mode=svd_hadamard.
    """
    pack = pack_weight(
        W,
        block_size=int(block),
        block_out_size=int(block_out),
        enable_permute=True,
        lambda_smooth=0.15,
        perm_score="weight",
        rot_mode="svd_hadamard",
    )
    in_features = int(W.shape[1])
    out_features = int(W.shape[0])
    blk = int(pack.meta.get("block_size", block))
    R_in_full = torch.zeros((in_features, in_features), dtype=torch.float32, device=device)
    for b, R in pack.R_in_blocks.items():
        rs, re_ = b * blk, min((b + 1) * blk, in_features)
        R_in_full[rs:re_, rs:re_] = torch.from_numpy(R[: (re_ - rs), : (re_ - rs)].astype("float32")).to(device)
    if pack.perm is not None:
        P = torch.zeros((in_features, in_features), dtype=torch.float32, device=device)
        perm_t = torch.from_numpy(pack.perm.astype("int64")).to(device)
        P[perm_t, torch.arange(in_features, device=device)] = 1.0
        rotation_R = P @ R_in_full
    else:
        rotation_R = R_in_full
    R_out_full = None
    if pack.R_out_blocks:
        blk_o = int(pack.meta.get("block_out_size", block_out))
        R_out_full = torch.zeros((out_features, out_features), dtype=torch.float32, device=device)
        for b, R in pack.R_out_blocks.items():
            rs, re_ = b * blk_o, min((b + 1) * blk_o, out_features)
            R_out_full[rs:re_, rs:re_] = torch.from_numpy(R[: (re_ - rs), : (re_ - rs)].astype("float32")).to(device)
    return rotation_R, R_out_full


def main():
    args = parse_args()
    ensure_libero_runtime()
    seed_everything(args.seed)
    device = torch.device(args.device)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32

    print(f"[A2-SVD-GPTQ] loading policy {args.checkpoint}", flush=True)
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    policy.model.eval()
    for p in policy.model.parameters():
        p.requires_grad_(False)

    # Find target DiT layers
    target_layers = []
    for name, mod in policy.model.named_modules():
        if DIT_PATTERN.match(name) and hasattr(mod, "weight") and mod.weight is not None and mod.weight.dim() == 2:
            target_layers.append((name, mod))
    print(f"[A2-SVD-GPTQ] {len(target_layers)} DiT linears identified", flush=True)

    # Capture per-step inputs
    cache: Dict[str, Dict[int, List[torch.Tensor]]] = {n: defaultdict(list) for n, _ in target_layers}

    def make_hook(name: str):
        def hook(_m, inputs):
            if not inputs:
                return
            x = inputs[0]
            if x is None:
                return
            t = get_current_dit_step()
            if t is None:
                return
            x = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
            if x.shape[0] > args.token_cap:
                idx = torch.randperm(x.shape[0], device=x.device)[: args.token_cap]
                x = x[idx]
            cache[name][int(t)].append(x.cpu())
        return hook

    handles = [mod.register_forward_pre_hook(make_hook(name)) for name, mod in target_layers]

    samples = load_libero_samples(args, data_config)
    if isinstance(samples, tuple):
        samples = samples[-1]
    samples = samples[: args.num_samples]
    print(f"[A2-SVD-GPTQ] driving {len(samples)} cal trajectories", flush=True)
    try:
        with torch.no_grad():
            for i, sample in enumerate(samples):
                norm = normalized_input_no_inference(policy, sample["obs"])
                policy.model.get_action(norm)
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"[A2-SVD-GPTQ] sample {i+1}/{len(samples)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    print("[A2-SVD-GPTQ] activation capture done", flush=True)

    # Per-layer processing
    records: Dict[str, dict] = {}
    a_qmax = qmax(int(args.a_bits))
    p_frac = float(args.act_percentile) / 100.0
    num_steps = int(args.num_steps)

    for li, (name, mod) in enumerate(target_layers):
        t0 = time.time()
        W_fp = mod.weight.detach().to(device=device, dtype=torch.float32)
        out_f, in_f = W_fp.shape

        # 1. Rotation
        rotation_R, R_out_full = build_rotation(
            W_fp, args.duquant_block_size, args.duquant_block_out, device
        )

        # 2. Rotate W
        W_r = W_fp @ rotation_R
        if R_out_full is not None:
            W_r = R_out_full @ W_r

        # 3. SVD-r16 of W_r
        U, S_, Vh = torch.linalg.svd(W_r.to(torch.float32), full_matrices=False)
        r = min(int(args.svd_rank), S_.shape[0])
        sqrtS = torch.sqrt(S_[:r].clamp_min(0.0))
        lowA = U[:, :r] * sqrtS[None, :]
        lowB = (Vh[:r, :].T) * sqrtS[None, :]
        W_res = W_r - lowA @ lowB.T

        # 4. Per-step rotated X buckets
        per_step_x: Dict[int, torch.Tensor] = {}
        all_x_chunks = []
        for t in range(num_steps):
            chunks = cache[name].get(t, [])
            if not chunks:
                continue
            X_t = torch.cat(chunks, dim=0).to(device)
            X_t_r = X_t @ rotation_R
            per_step_x[t] = X_t_r
            all_x_chunks.append(X_t_r)

        if not all_x_chunks:
            print(f"[A2-SVD-GPTQ][skip] {name}: no activations", flush=True)
            continue

        X_all = torch.cat(all_x_chunks, dim=0)
        N = X_all.shape[0]
        H_r = (X_all.T @ X_all) / float(max(N, 1))

        # 5. GPTQ on residual
        W_res_q = gptq_quantize_weight(
            W_res, H_r,
            bits=int(args.w_bits),
            block_size=int(args.gptq_block_size),
            damp_percent=float(args.gptq_damp_percent),
            err_comp_gamma=0.0 if args.use_rtn else 1.0,
        )

        # 6. Per-step act_scale_table on rotated X
        table = torch.zeros(num_steps, in_f, dtype=torch.float32, device=device)
        n_per_step: List[int] = []
        for t in range(num_steps):
            if t not in per_step_x:
                n_per_step.append(0)
                continue
            Xt = per_step_x[t]
            q_vec = torch.quantile(Xt.abs(), p_frac, dim=0).clamp_min(1e-6)
            table[t] = q_vec / a_qmax
            n_per_step.append(int(Xt.shape[0]))

        # Sanity checks
        for tname, tobj in [("W_res_q", W_res_q), ("lowA", lowA), ("lowB", lowB),
                            ("rotation_R", rotation_R), ("table", table)]:
            if not torch.isfinite(tobj).all():
                raise RuntimeError(f"[A2-SVD-GPTQ] non-finite in {name}/{tname}")

        record = {
            "format": "dit_svdquant_v1",
            "weight_res_q": W_res_q.to(save_dtype).cpu().contiguous(),
            "lowrank_A": lowA.to(save_dtype).cpu().contiguous(),
            "lowrank_B": lowB.to(save_dtype).cpu().contiguous(),
            "act_scale_table": table.to(torch.float32).cpu().contiguous(),
            "duquant_rotation": rotation_R.to(save_dtype).cpu().contiguous(),
            "weight_bits": int(args.w_bits),
            "a_bits": int(args.a_bits),
            "rank": int(args.svd_rank),
            "in_features": int(in_f),
            "out_features": int(out_f),
            "n_calib_total": int(N),
            "n_calib_per_step": n_per_step,
            "act_percentile": float(args.act_percentile),
            "gptq_damp_percent": float(args.gptq_damp_percent),
        }
        if R_out_full is not None:
            record["duquant_rotation_out"] = R_out_full.to(save_dtype).cpu().contiguous()
        records[name] = record

        elapsed = time.time() - t0
        if li < 3 or (li + 1) % 12 == 0:
            print(f"[A2-SVD-GPTQ] {li+1}/{len(target_layers)} {name}  "
                  f"W={tuple(W_fp.shape)}  N_cal={N}  elapsed={elapsed:.1f}s", flush=True)

    # __meta__
    records["__meta__"] = {
        "format": "dit_a2lite_svd_gptq_perstep_v1",
        "checkpoint": args.checkpoint,
        "task_suite": args.task_suite_name,
        "num_samples": int(args.num_samples),
        "svd_rank": int(args.svd_rank),
        "gptq_damp_percent": float(args.gptq_damp_percent),
        "duquant_block_size": int(args.duquant_block_size),
        "duquant_block_out": int(args.duquant_block_out),
        "rot_mode": "svd_hadamard",
        "perm": "weight-zigzag",
        "act_percentile": float(args.act_percentile),
        "a_bits": int(args.a_bits),
        "w_bits": int(args.w_bits),
        "num_steps": num_steps,
    }
    torch.save(records, out_path)
    print(f"[A2-SVD-GPTQ] wrote {out_path}  ({len(records) - 1} layers)", flush=True)


if __name__ == "__main__":
    main()
