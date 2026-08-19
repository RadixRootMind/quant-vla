#!/usr/bin/env python
"""Build DiT-only SVDQuant pack on top of an existing M3/D1 base pack.

For each DiT target Linear:
  1. Hook input activations per denoising step (1..num_steps), captured via
     the same dit_step_context mechanism used by calibrate_dit_timestep_scales.py.
  2. SmoothQuant:
       A_j = max_t quantile(|X_{l,t,j}|, 99.9)     [activation per channel]
       B_j = max_i |W_{i,j}|                       [weight per channel]
       s_j = A_j^alpha / B_j^(1-alpha)
       s   = clamp(s / median(s), [lo, hi])
       W_s = W * s[None, :]
       X_s = X / s[None, :]
  3. Hessian on smoothed activation: H_s = X_s^T X_s / n
  4. error_svd:
       W4_tmp = GPTQ(W_s, H_s, bits=4)
       E = W_s - W4_tmp
       U_r, S_r, V_r = svd_lowrank(E, rank=svd_rank)
       lowrank_A = U_r * sqrt(S_r)        [out, r]
       lowrank_B = V_r * sqrt(S_r)        [in,  r]
       W_res = W_s - lowrank_A @ lowrank_B.T
       W_res_q = GPTQ(W_res, H_s, bits=4)
  5. Per-step act table:
       act_scale_table[t, j] = quantile(|X_s_{l,t,j}|, act_percentile) / qmax(a_bits)

The record stored on disk is format="dit_svdquant_v1" with the keys listed in
the build script docstring. LLM records from the base pack are copied
unchanged; non-DiT records are also preserved untouched.

Usage:
  python tools/build_dit_svdquant_weights.py \
      --checkpoint /.../gr00t-n1.5-libero-goal-posttrain \
      --base-pack results/multisuite_packs/goal_dit_gptq_w4a8/quantized.pt \
      --output-path results/multisuite_packs/goal_R3_SVD_W4A8_DiT_r16_q999/quantized.pt \
      --w-bits 4 --a-bits 8 --svd-rank 16 \
      --act-percentile 99.9 \
      --num-samples 20 --token-cap 1024 --num-steps 8
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

from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    DEFAULT_EXCLUDE_REGEX,
    DEFAULT_INCLUDE_REGEX,
    ensure_libero_runtime,
    get_named_module,
    load_libero_samples,
    load_policy,
    seed_everything,
)
from tools.analyze_layerwise_quant_drift import normalized_input_no_inference  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.quantization.dit_step_context import get_current_dit_step  # noqa: E402
from gr00t.quantization.duquant_layers import select_targets  # noqa: E402
from gr00t.quantization.gptq_layers import gptq_quantize_weight  # noqa: E402
from gr00t.quantization.duquant_preprocess import qmax  # noqa: E402

DIT_PATTERN = re.compile(
    r"action_head\.model\.transformer_blocks\.\d+\.(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))"
)
LLM_PATTERN = re.compile(
    r"backbone\.eagle_model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
TARGET_PATTERNS = {"dit": DIT_PATTERN, "llm": LLM_PATTERN}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--base-pack", required=True,
                   help="Existing pack (e.g. M3 sq_widemse or D1 q990) for non-DiT records.")
    p.add_argument("--output-path", required=True)

    p.add_argument("--task-suite-name", default="libero_goal")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfigMeanStd")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)

    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--libero-num-trials-per-task", type=int, default=2)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task",
                   choices=["sequential", "one_per_task", "one_per_trial"])
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-steps", type=int, default=8, help="DiT denoising steps")
    p.add_argument("--token-cap", type=int, default=1024)
    p.add_argument("--act-percentile", type=float, default=99.9)
    p.add_argument(
        "--single-step-scale",
        action="store_true",
        help=(
            "DiT mode: collapse per-denoise-step activation bucketing into a "
            "single global scale (like vanilla DuQuant). Captures activations "
            "across all denoising steps but bins them into one bucket → "
            "act_scale_table shape [1, in_features]. Ablates the per-step-scale "
            "contribution while keeping SVD low-rank + GPTQ residual intact."
        ),
    )

    p.add_argument("--include-regex", default=DEFAULT_INCLUDE_REGEX)
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE_REGEX)
    p.add_argument("--scope", default="dit_only", help="(label only)")
    p.add_argument(
        "--target-side",
        choices=["dit", "llm"],
        default="dit",
        help=(
            "Which model side to build SVDQuant pack for:\n"
            "  dit (default): action_head transformer_blocks (attn1 + ff). "
            "Activations bucketed per-denoise-step via dit_step_context.\n"
            "  llm: backbone eagle_model.language_model q/k/v/o + gate/up/down_proj. "
            "Activations bucketed into a single global scale (LLM has no step concept)."
        ),
    )

    p.add_argument("--w-bits", type=int, default=4)
    p.add_argument("--a-bits", type=int, default=8, choices=[4, 8])

    p.add_argument("--svd-mode", default="error_svd", choices=["error_svd"])
    p.add_argument("--svd-rank", type=int, default=16)
    p.add_argument(
        "--quant-type-DiT",
        choices=["original", "custom"],
        default="custom",
        help=(
            "DiT quantization variant:\n"
            "  custom   (default): GPTQ(W_s) first, SVD quantization residual,"
            "                      then GPTQ the low-rank-removed weight again.\n"
            "  original          : SVD W_s directly, GPTQ residual W_s - AB^T once."
        ),
    )
    # Back-compat shim: --use-custom-svdquant maps to --quant-type-DiT
    p.add_argument(
        "--use-custom-svdquant",
        type=lambda v: str(v).lower() not in ("0", "false", "no"),
        default=None,
        help="DEPRECATED: prefer --quant-type-DiT. true→custom, false→original.",
    )

    p.add_argument("--sq-alpha", type=float, default=0.5)
    p.add_argument("--sq-clamp-lo", type=float, default=0.5)
    p.add_argument("--sq-clamp-hi", type=float, default=2.0)

    p.add_argument("--gptq-block-size", type=int, default=128)
    p.add_argument("--gptq-damp-percent", type=float, default=0.01)
    p.add_argument("--use-rtn", action="store_true",
                   help="Replace GPTQ error compensation with pure RTN "
                        "(err_comp_gamma=0). Use with --quant-type-DiT original "
                        "for paper-style 'naive SVDQuant' (SQ + SVD + RTN).")

    p.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--max-layers", type=int, default=0, help="0=all; debug only")
    p.add_argument(
        "--use-q-llm-teacher",
        default=None,
        help=(
            "Path to an LLM-W4 GPTQ pack. When set, the LLM is wrapped "
            "with these quantized weights BEFORE the calibration forward pass, "
            "so DiT activation capture sees realistic Q-LLM input distribution "
            "instead of FP-LLM. This implements the bi-level calibration scheme "
            "(P1 in HANDOFF.md) that addresses the FP-vs-Q activation mismatch "
            "responsible for the 26pp goal-suite drop on LLM-W4 + DiT-SVD-W4A4. "
            "DiT weights remain FP for residual computation; only the forward "
            "path through LLM uses Q."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device)
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    a_qmax = float(qmax(args.a_bits))
    # Resolve back-compat: --use-custom-svdquant overrides --quant-type-DiT default
    # so old call sites keep working. New call sites should pass --quant-type-DiT.
    if args.use_custom_svdquant is not None:
        args.quant_type_DiT = "custom" if args.use_custom_svdquant else "original"
    quant_type = args.quant_type_DiT
    print(
        f"[SVDQ] start: w_bits={args.w_bits} a_bits={args.a_bits} rank={args.svd_rank} "
        f"alpha={args.sq_alpha} clamp=[{args.sq_clamp_lo},{args.sq_clamp_hi}] "
        f"act_pct={args.act_percentile} num_samples={args.num_samples} "
        f"token_cap={args.token_cap} a_qmax={a_qmax} "
        f"quant_type_DiT={quant_type}",
        flush=True,
    )

    ensure_libero_runtime()

    data_config = load_data_config(args.data_config)
    print(f"[SVDQ] loading {args.num_samples} LIBERO observations", flush=True)
    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    print(f"[SVDQ] loaded {len(samples)} samples", flush=True)

    print(f"[SVDQ] loading FP policy from {args.checkpoint}", flush=True)
    policy = load_policy(args, data_config, quantized_layers=None)

    target_side = args.target_side
    target_pattern = TARGET_PATTERNS[target_side]
    if args.single_step_scale and target_side == "dit":
        num_steps_table = 1
        print("[SVDQ] single_step_scale: DiT activations collapsed into one bucket", flush=True)
    else:
        num_steps_table = args.num_steps if target_side == "dit" else 1
    side_upper = target_side.upper()

    # Resolve target layer names from include/exclude regex.
    targets = select_targets(
        policy.model,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        scope_prefix=None,
        whitelist=None,
        blacklist=None,
    )
    dit_target_names = [n for n, _ in targets if target_pattern.search(n)]
    if args.max_layers > 0:
        dit_target_names = dit_target_names[: args.max_layers]
    print(f"[SVDQ] {side_upper} target layers: {len(dit_target_names)}", flush=True)
    if len(dit_target_names) == 0:
        raise RuntimeError(f"no {side_upper} target layers matched")

    # Capture FP weights (from the live model, before any wrap).
    # IMPORTANT: this must happen BEFORE the optional LLM Q-wrap below — DiT
    # weights stay FP throughout (the wrap is LLM-only), but doing this first
    # makes the dependency explicit and crash-safe.
    fp_weights: Dict[str, torch.Tensor] = {}
    fp_biases: Dict[str, Optional[torch.Tensor]] = {}
    for name in dit_target_names:
        mod = get_named_module(policy.model, name)
        fp_weights[name] = mod.weight.detach().clone().to(torch.float32)
        fp_biases[name] = (
            mod.bias.detach().clone().to(torch.float32) if mod.bias is not None else None
        )
    print(f"[SVDQ] captured FP weights for {len(fp_weights)} layers", flush=True)

    # Bi-level (P1): optionally wrap LLM with Q weights so DiT activation
    # capture sees the deployment-time noisy distribution. Implements the
    # iterative pipeline LLM-Q → DiT-Q-recalibrated (see HANDOFF.md P1).
    if args.use_q_llm_teacher:
        from gr00t.quantization.gptq_layers import wrap_gptq, GptqConfig

        llm_include_regex = (
            r".*backbone\.eagle_model\.language_model\..*"
            r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*"
        )
        llm_exclude_regex = (
            r"(?:^|\.)(vision|radio|norm|ln|layernorm|embed|lm_head|"
            r"timestep_encoder|state_encoder|action_encoder|action_decoder|"
            r"pos_embed|vl_self_attention|vlln|future_tokens|attn1|ff)(?:\.|$)"
        )
        llm_targets = select_targets(
            policy.model,
            include_regex=llm_include_regex,
            exclude_regex=llm_exclude_regex,
            scope_prefix=None,
            whitelist=None,
            blacklist=None,
        )
        llm_layer_names = [n for n, _ in llm_targets]
        print(
            f"[SVDQ][BiLevel] wrapping {len(llm_layer_names)} LLM layers with Q "
            f"from {args.use_q_llm_teacher}",
            flush=True,
        )
        llm_cfg = GptqConfig(
            enabled=True,
            path=args.use_q_llm_teacher,
            weight_bits=4,
            act_bits=8,
            missing="fallback",
        )
        wrap_gptq(policy.model, llm_layer_names, llm_cfg)
        policy.model.eval()
        print(f"[SVDQ][BiLevel] LLM Q-wrap done; DiT inputs will be Q-LLM teacher", flush=True)

    # Hook target-side inputs.
    #   DiT mode: per denoise step via dit_step_context (same as D1 calibrator)
    #   LLM mode: single bucket (t=0); LLM forwards once per chunk, no step concept
    cache: Dict[str, Dict[int, List[torch.Tensor]]] = {n: defaultdict(list) for n in dit_target_names}

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
            if target_side == "llm":
                t = 0
            elif args.single_step_scale:
                # Still drain only when inside a DiT denoising forward (the same
                # call sites that would set get_current_dit_step), so we don't
                # double-count outside-of-denoise forwards. Then collapse to t=0.
                if get_current_dit_step() is None:
                    return
                t = 0
            else:
                t = get_current_dit_step()
                if t is None:
                    return
            cache[name][int(t)].append(x.cpu())
        return hook

    handles = []
    for name in dit_target_names:
        mod = get_named_module(policy.model, name)
        handles.append(mod.register_forward_pre_hook(make_hook(name)))
    try:
        with torch.no_grad():
            for i, sample in enumerate(samples):
                norm = normalized_input_no_inference(policy, sample["obs"])
                # Run DiT denoising loop (set_dit_quant_step is set inside policy)
                policy.model.get_action(norm)
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"[SVDQ] hook collect: {i+1}/{len(samples)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    print(f"[SVDQ] activation capture done", flush=True)

    # Concatenate per-layer per-step (or single-bucket for LLM) caches.
    # X_total[name] = concat over all steps    shape [N_total, in_features]
    # X_per_step[name][t] = concat for step t  shape [N_t, in_features]
    X_per_step: Dict[str, Dict[int, torch.Tensor]] = {}
    X_total: Dict[str, torch.Tensor] = {}
    for name in dit_target_names:
        per_step = {}
        all_chunks = []
        for t in range(num_steps_table):
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

    # Free policy GPU memory before per-layer GPTQ work.
    del policy
    torch.cuda.empty_cache()

    # Process each layer and build the SVDQuant record.
    diag_rows: List[Dict] = []
    new_records: Dict[str, Dict] = {}

    eps = 1e-8
    p = float(args.act_percentile) / 100.0

    for li, name in enumerate(dit_target_names):
        t0 = time.time()
        W = fp_weights[name].to(device)
        out_features, in_features = W.shape
        Xt = X_total[name]
        if Xt.shape[0] == 0:
            print(f"[SVDQ][skip] {name}: no activations captured", flush=True)
            continue
        Xt_dev = Xt.to(device)
        # SmoothQuant migration scale.
        # A_j = max_t quantile(|X_{l,t,j}|, 99.9)  per-channel from raw activations
        A = torch.zeros(in_features, dtype=torch.float32, device=device)
        for t, X_t in X_per_step[name].items():
            X_dev = X_t.to(device)
            q_t = torch.quantile(X_dev.abs(), 0.999, dim=0)
            A = torch.maximum(A, q_t)
        A = A.clamp_min(eps)
        B = W.detach().abs().amax(dim=0).clamp_min(eps)  # (in,)
        raw_s = (A.pow(args.sq_alpha)) / (B.pow(1.0 - args.sq_alpha))
        med = raw_s.median().clamp_min(eps)
        s_norm = raw_s / med
        s = torch.clamp(s_norm, min=args.sq_clamp_lo, max=args.sq_clamp_hi).to(torch.float32)
        # smoothed weight and activations
        W_s = W * s[None, :]
        Xs_total = Xt_dev / s[None, :]
        # Hessian on smoothed activation
        n_total = Xs_total.shape[0]
        H_s = (Xs_total.T @ Xs_total) / float(max(n_total, 1))
        H_s = H_s.to(torch.float32)

        # Step 1: pick SVD target based on quant_type
        #   custom  : SVD the GPTQ-quant residual E = W_s − W4_tmp  → captures
        #             directions GPTQ fails on
        #   original: SVD W_s directly                              → captures
        #             dominant singular structure of the smoothed weight
        if quant_type == "custom":
            W4_tmp = gptq_quantize_weight(
                W_s, H_s,
                bits=int(args.w_bits),
                block_size=int(args.gptq_block_size),
                damp_percent=float(args.gptq_damp_percent),
                err_comp_gamma=0.0 if args.use_rtn else 1.0,
            )
            svd_target = W_s - W4_tmp
            E_norm_rel = float(svd_target.norm().item() / max(W_s.norm().item(), eps))
        else:
            W4_tmp = None
            svd_target = W_s
            E_norm_rel = 1.0  # SVDing W_s directly; no quant residual to report

        # SVD on the chosen target (residual for custom, W_s for original)
        if args.svd_rank > 0:
            try:
                U_r, S_r, V_r = torch.svd_lowrank(svd_target.to(torch.float32), q=int(args.svd_rank) + 6)
                U_r = U_r[:, : args.svd_rank]
                S_r = S_r[: args.svd_rank]
                V_r = V_r[:, : args.svd_rank]
            except Exception as e:
                print(f"[SVDQ][warn] {name}: svd_lowrank failed ({e}), using full SVD", flush=True)
                U_full, S_full, Vh_full = torch.linalg.svd(svd_target.to(torch.float32), full_matrices=False)
                U_r = U_full[:, : args.svd_rank]
                S_r = S_full[: args.svd_rank]
                V_r = Vh_full.t()[:, : args.svd_rank]
            sqrtS = S_r.clamp_min(0).sqrt()
            lowrank_A = U_r * sqrtS[None, :]   # [out, r]
            lowrank_B = V_r * sqrtS[None, :]   # [in,  r]
            E_recon = lowrank_A @ lowrank_B.T
            energy = float((S_r.pow(2).sum() / svd_target.pow(2).sum().clamp_min(eps)).item())
        else:
            lowrank_A = torch.zeros(out_features, 0, dtype=torch.float32, device=device)
            lowrank_B = torch.zeros(in_features, 0, dtype=torch.float32, device=device)
            E_recon = torch.zeros_like(svd_target)
            energy = 0.0

        # Step 2: residual after low-rank, GPTQ again
        W_res = W_s - E_recon
        W_res_q = gptq_quantize_weight(
            W_res, H_s,
            bits=int(args.w_bits),
            block_size=int(args.gptq_block_size),
            damp_percent=float(args.gptq_damp_percent),
            err_comp_gamma=0.0 if args.use_rtn else 1.0,
        )
        # Reconstruction error of full W_s ≈ W_res_q + AB^T
        recon = W_res_q + E_recon
        recon_err = float((W_s - recon).norm().item() / max(W_s.norm().item(), eps))

        # Per-step act_scale_table on smoothed activation
        # (LLM mode: num_steps_table=1 → table shape [1, in_features])
        table = torch.zeros(num_steps_table, in_features, dtype=torch.float32, device=device)
        n_per_step: List[int] = []
        clip_per_step: List[float] = []
        for t in range(num_steps_table):
            X_t = X_per_step[name].get(t)
            if X_t is None or X_t.numel() == 0:
                n_per_step.append(0)
                clip_per_step.append(0.0)
                continue
            Xs_t = X_t.to(device) / s[None, :]
            q_vec = torch.quantile(Xs_t.abs(), p, dim=0).clamp_min(1e-6)
            scale = q_vec / a_qmax
            table[t] = scale
            # clipping fraction: tokens with any channel value > scale * qmax  (= q_vec)
            clip_frac = float(((Xs_t.abs() > q_vec[None, :]).any(dim=1).float()).mean().item())
            n_per_step.append(int(Xs_t.shape[0]))
            clip_per_step.append(clip_frac)

        # Sanity: any NaN/Inf?
        any_bad = False
        for t_name, t_obj in [("smooth_scale", s), ("act_scale_table", table),
                              ("lowrank_A", lowrank_A), ("lowrank_B", lowrank_B),
                              ("weight_res_q", W_res_q)]:
            if not torch.isfinite(t_obj).all():
                print(f"[SVDQ][BAD] {name}: {t_name} has non-finite values", flush=True)
                any_bad = True
        if any_bad:
            raise RuntimeError(f"non-finite values in {name}")

        # Save record
        # IMPORTANT: act_scale_table MUST be FP32 — for channels with very low
        # activation magnitude, scale = q99.9 / qmax(8)=127 can be ~8e-9 which
        # underflows FP16's subnormal range (~6e-8) and gets stored as exact 0.
        # At runtime, fake_quantize_sym then divides by 0 → NaN cascade. The
        # tensor is small (8 × in_features) so FP32 overhead is negligible.
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
            "svd_mode": str(args.svd_mode),
            "sq_alpha": float(args.sq_alpha),
            "sq_clamp": [float(args.sq_clamp_lo), float(args.sq_clamp_hi)],
            "act_percentile": float(args.act_percentile),
            "in_features": int(in_features),
            "out_features": int(out_features),
            "n_calib_total": int(n_total),
            "n_calib_per_step": n_per_step,
            # Diagnostics
            "diag_E_norm_rel": E_norm_rel,
            "diag_svd_energy": energy,
            "diag_recon_err": recon_err,
            "diag_clip_per_step": clip_per_step,
            "diag_quant_type_DiT": quant_type,
        }
        new_records[name] = rec

        elapsed = time.time() - t0
        if li < 3 or (li + 1) % 12 == 0:
            mean_clip = float(np.mean(clip_per_step))
            print(
                f"[SVDQ] {li+1}/{len(dit_target_names)} {name}: "
                f"||E||_rel={E_norm_rel:.4f} svd_energy={energy:.3f} "
                f"recon_err={recon_err:.4f} mean_clip={mean_clip:.3f} "
                f"n_total={n_total} elapsed={elapsed:.1f}s",
                flush=True,
            )
        diag_rows.append({
            "name": name, "out": out_features, "in": in_features,
            "E_norm_rel": E_norm_rel, "svd_energy": energy,
            "recon_err": recon_err, "n_total": n_total,
            "mean_clip": float(np.mean(clip_per_step)),
        })

        # Free
        del W, W_s, H_s, svd_target, E_recon, W_res, W_res_q
        if W4_tmp is not None:
            del W4_tmp
        del lowrank_A, lowrank_B, table, Xt_dev, Xs_total
        torch.cuda.empty_cache()

    # Load base pack and merge: drop entries matching this run's target_pattern
    # (will be replaced by new records), keep everything else (other side or non-quantized).
    print(f"[SVDQ] loading base pack: {args.base_pack}", flush=True)
    base = torch.load(args.base_pack, map_location="cpu", weights_only=False)
    merged: Dict = {}
    if "__meta__" in base:
        meta = dict(base["__meta__"])
    else:
        meta = {}
    meta["pack_format"] = "mixed_dit_svdquant_v1"
    meta["svdquant_w_bits"] = int(args.w_bits)
    meta["svdquant_a_bits"] = int(args.a_bits)
    meta["svdquant_rank"] = int(args.svd_rank)
    meta["svdquant_act_percentile"] = float(args.act_percentile)
    meta["svdquant_alpha"] = float(args.sq_alpha)
    meta["svdquant_clamp"] = [float(args.sq_clamp_lo), float(args.sq_clamp_hi)]
    meta["svdquant_num_samples"] = int(args.num_samples)
    meta["svdquant_token_cap"] = int(args.token_cap)
    meta["svdquant_base_pack"] = args.base_pack
    meta["svdquant_target_side"] = target_side
    # Per-side variant tag (record both sides if base already had one)
    meta[f"svdquant_quant_type_{side_upper}"] = quant_type
    # legacy fields kept for back-compat readers (still set DiT names if building DiT)
    if target_side == "dit":
        meta["svdquant_quant_type_DiT"] = quant_type
        meta["svdquant_use_custom"] = (quant_type == "custom")
    merged["__meta__"] = meta

    n_base_overridden = 0
    n_base_kept = 0
    for k, v in base.items():
        if k == "__meta__":
            continue
        if target_pattern.search(k):
            n_base_overridden += 1
            continue  # will be replaced by new SVDQ record
        merged[k] = v
        n_base_kept += 1

    n_added = 0
    for name, rec in new_records.items():
        merged[name] = rec
        n_added += 1

    print(
        f"[SVDQ] merge ({side_upper}): base others kept={n_base_kept} "
        f"base {side_upper} overridden={n_base_overridden} new {side_upper} records={n_added}",
        flush=True,
    )

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic save (tmp + rename) to avoid concurrent readers seeing partial file.
    tmp_path = str(out_path) + f".tmp.{os.getpid()}"
    torch.save(merged, tmp_path)
    os.replace(tmp_path, out_path)
    print(f"[SVDQ] wrote {out_path}", flush=True)

    # Top 10 worst layers by recon_err
    diag_rows.sort(key=lambda r: r["recon_err"], reverse=True)
    print("[SVDQ] top 10 worst layers by recon_err:")
    for r in diag_rows[:10]:
        print(
            f"  {r['name']:75s} ||E||_rel={r['E_norm_rel']:.4f} "
            f"svd_energy={r['svd_energy']:.3f} recon_err={r['recon_err']:.4f} "
            f"mean_clip={r['mean_clip']:.3f}"
        )

    # Save diagnostics JSON
    diag_path = out_path.with_suffix(".diag.json")
    with open(diag_path, "w") as f:
        json.dump({"layers": diag_rows, "meta": meta}, f, indent=2)
    print(f"[SVDQ] wrote diagnostics to {diag_path}", flush=True)


if __name__ == "__main__":
    main()
