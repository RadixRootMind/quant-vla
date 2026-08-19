#!/usr/bin/env python
"""Build pi0.5 Expert pack with A2-lite rotation + GPTQ + per-step act_scale_table.

Equivalent to gr00t's tools/build_dit_a2lite_svd_gptq_perstep.py (no SVD).
For each Expert Linear (Gemma):
  1. Capture per-step activations via dit_step_context bucketing.
  2. Compute A2-lite rotation (perm + svd_hadamard block) from FP W.
  3. Rotate W and per-step X.
  4. GPTQ on rotated W using pooled H = X_r^T X_r / N.
  5. Per-step act_scale_table[t, j] = quantile(|X_r_t|, 0.999, dim=0) / qmax.

Output format = "dit_svdquant_v1" with empty lowrank head (lowA/lowB shape [out, 0] / [in, 0]).
GptqLinear runtime auto-detects empty lowrank → INT4 path only + per-step scale dispatch.

Run with the openpi venv:
  $OPENPI_ROOT/.venv/bin/python -m tools.build_pi05_a2lite_gptq_perstep \\
      --checkpoint $CHECKPOINTS_ROOT/pi05_libero_pytorch \\
      --obs-path duquant_act_stats/pi05_libero_object_obs.pt \\
      --output results/multisuite_packs/pi05_object_a2lite_gptq_perstep_cal10/quantized.pt \\
      --max-samples 10 --num-steps 10 --gptq-damp-percent 0.05
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from gr00t.quantization.dit_step_context import get_current_dit_step  # noqa: E402
from gr00t.quantization.gptq_layers import gptq_quantize_weight  # noqa: E402
from gr00t.quantization.duquant_preprocess import qmax, pack_weight  # noqa: E402


EXPERT_PATTERN = re.compile(
    r"paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\..*"
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
DEFAULT_INCLUDE = (
    r".*paligemma_with_expert\.gemma_expert\.model\.layers\.\d+\..*"
    r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
)
DEFAULT_EXCLUDE = (
    r"(?:^|\.)(vision_tower|vision_model|embeddings|embed_tokens|norm|layernorm|lm_head)(?:\.|$)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-config", default="pi05_libero")
    p.add_argument("--obs-path", required=True, help="LIBERO obs pickle from record_libero_obs_for_pi05")
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--token-cap", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=10, help="pi0.5 denoising steps")
    p.add_argument("--device", default="cuda")
    p.add_argument("--include-regex", default=DEFAULT_INCLUDE)
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE)
    p.add_argument("--w-bits", type=int, default=4)
    p.add_argument("--a-bits", type=int, default=4)
    p.add_argument("--gptq-block-size", type=int, default=128)
    p.add_argument("--gptq-damp-percent", type=float, default=0.05)
    p.add_argument("--use-rtn", action="store_true",
                   help="Skip GPTQ error compensation (err_comp_gamma=0).")
    p.add_argument("--duquant-block-size", type=int, default=64)
    p.add_argument("--duquant-block-out", type=int, default=64)
    p.add_argument("--act-percentile", type=float, default=99.9)
    p.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--capture-prefix", action="store_true",
                   help="Bucket prefix-pass activations (no denoising step) at t=0. "
                        "Use for PaliGemma/LLM layers, which run once in the prefix and "
                        "never inside the denoising loop. Leave off for Expert/DiT layers.")
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
    for param in model.parameters():
        param.requires_grad_(False)
    return policy, model


def build_a2lite_rotation(W: torch.Tensor, block: int, block_out: int, device):
    """A2-lite (svd_hadamard) rotation. Returns:
      rotation_R:   dense [in, in]  (perm @ block_diag(R_in_b)) — legacy compat
      R_out_full:   dense [out, out] or None
      r_in_blocks:  stacked block tensor [n_in, B, B] FP32 (for fast runtime path)
      perm:         [in] int64 (the input-side perm, identity if not permuted)
      r_out_blocks: stacked [n_out, B_out, B_out] or None
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
    # Stack block matrices (zero-padded to B if last block is short).
    n_blocks_in = (in_features + blk - 1) // blk
    r_in_blocks = torch.zeros((n_blocks_in, blk, blk), dtype=torch.float32, device=device)
    R_in_full = torch.zeros((in_features, in_features), dtype=torch.float32, device=device)
    for b, R in pack.R_in_blocks.items():
        rs, re_ = b * blk, min((b + 1) * blk, in_features)
        Rt = torch.from_numpy(R.astype("float32")).to(device)  # (B, B)
        r_in_blocks[b, : (re_ - rs), : (re_ - rs)] = Rt[: (re_ - rs), : (re_ - rs)]
        R_in_full[rs:re_, rs:re_] = Rt[: (re_ - rs), : (re_ - rs)]
    if pack.perm is not None:
        perm_t = torch.from_numpy(pack.perm.astype("int64")).to(device)
        P = torch.zeros((in_features, in_features), dtype=torch.float32, device=device)
        P[perm_t, torch.arange(in_features, device=device)] = 1.0
        rotation_R = P @ R_in_full
    else:
        perm_t = torch.arange(in_features, device=device, dtype=torch.int64)
        rotation_R = R_in_full

    R_out_full = None
    r_out_blocks = None
    if pack.R_out_blocks:
        blk_o = int(pack.meta.get("block_out_size", block_out))
        n_blocks_out = (out_features + blk_o - 1) // blk_o
        r_out_blocks = torch.zeros((n_blocks_out, blk_o, blk_o), dtype=torch.float32, device=device)
        R_out_full = torch.zeros((out_features, out_features), dtype=torch.float32, device=device)
        for b, R in pack.R_out_blocks.items():
            rs, re_ = b * blk_o, min((b + 1) * blk_o, out_features)
            Rt = torch.from_numpy(R.astype("float32")).to(device)
            r_out_blocks[b, : (re_ - rs), : (re_ - rs)] = Rt[: (re_ - rs), : (re_ - rs)]
            R_out_full[rs:re_, rs:re_] = Rt[: (re_ - rs), : (re_ - rs)]
    return rotation_R, R_out_full, r_in_blocks, perm_t, r_out_blocks


def main():
    args = parse_args()
    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    a_qmax = float(qmax(args.a_bits))
    device = torch.device(args.device)

    print(f"[A2-GPTQ-PI05] loading policy {args.checkpoint}", flush=True)
    policy, model = load_pi05_policy(args.checkpoint, args.data_config, args.device)

    # Resolve target Expert Linear layers
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
        target_names.append(name)
    if args.max_layers > 0:
        target_names = target_names[: args.max_layers]
    print(f"[A2-GPTQ-PI05] {len(target_names)} Expert linears", flush=True)

    # FP weights
    fp_weights: Dict[str, torch.Tensor] = {}
    for name in target_names:
        mod = dict(model.named_modules())[name]
        fp_weights[name] = mod.weight.detach().clone().to(torch.float32)

    # Hook layers; bucket by step via dit_step_context
    cache: Dict[str, Dict[int, List[torch.Tensor]]] = {n: defaultdict(list) for n in target_names}

    def make_hook(name: str):
        def hook(_m, inputs):
            if not inputs:
                return
            x = inputs[0]
            if x is None:
                return
            t = get_current_dit_step()
            if t is None:
                if not args.capture_prefix:
                    return  # Expert/DiT: only collect inside the denoising loop
                t = 0  # PaliGemma/LLM prefix: single bucket at t=0
            x = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
            if x.shape[0] > args.token_cap:
                idx = torch.randperm(x.shape[0], device=x.device)[: args.token_cap]
                x = x[idx]
            cache[name][int(t)].append(x.cpu())
        return hook

    handles = []
    name_to_mod = dict(model.named_modules())
    for name in target_names:
        handles.append(name_to_mod[name].register_forward_pre_hook(make_hook(name)))

    # Drive cal forward via obs pickle
    samples = torch.load(args.obs_path, weights_only=False)
    if isinstance(samples, dict) and "samples" in samples:
        samples = samples["samples"]
    samples = samples[: args.max_samples]
    print(f"[A2-GPTQ-PI05] driving {len(samples)} obs samples", flush=True)
    try:
        with torch.no_grad():
            for i, obs in enumerate(samples, 1):
                lang = obs.get("prompt", "")
                if i == 1 or i % 5 == 0:
                    print(f"[A2-GPTQ-PI05] sample {i}/{len(samples)}: {lang!r}", flush=True)
                _ = policy.infer(obs)
    finally:
        for h in handles:
            h.remove()
    print(f"[A2-GPTQ-PI05] activation capture done", flush=True)

    # Concatenate per-step
    X_per_step: Dict[str, Dict[int, torch.Tensor]] = {}
    num_steps = int(args.num_steps)
    for name in target_names:
        per_step = {}
        for t in range(num_steps):
            chunks = cache[name].get(t, [])
            if chunks:
                per_step[t] = torch.cat(chunks, dim=0)
        X_per_step[name] = per_step
    cache.clear()

    del policy
    torch.cuda.empty_cache()

    p_frac = float(args.act_percentile) / 100.0
    new_records: Dict[str, Dict] = {}
    for li, name in enumerate(target_names):
        t0 = time.time()
        W = fp_weights[name].to(device)
        out_f, in_f = W.shape
        per_step = X_per_step[name]
        if not per_step:
            print(f"[A2-GPTQ-PI05][skip] {name}: no activations captured", flush=True)
            continue

        # 1. A2-lite rotation
        rotation_R, R_out_full, r_in_blocks, perm_in, r_out_blocks = build_a2lite_rotation(
            W, args.duquant_block_size, args.duquant_block_out, device
        )
        # 2. Rotate W
        W_r = W @ rotation_R
        if R_out_full is not None:
            W_r = R_out_full @ W_r

        # 3. Per-step rotated X buckets
        all_x_chunks = []
        rotated_per_step: Dict[int, torch.Tensor] = {}
        for t, X_t in per_step.items():
            X_t_r = X_t.to(device) @ rotation_R
            rotated_per_step[t] = X_t_r
            all_x_chunks.append(X_t_r)
        X_all = torch.cat(all_x_chunks, dim=0)
        N = X_all.shape[0]
        H_r = (X_all.T @ X_all) / float(max(N, 1))

        # 4. GPTQ (or RTN if --use-rtn) on rotated W (NO SVD lowrank head)
        W_r_q = gptq_quantize_weight(
            W_r, H_r,
            bits=int(args.w_bits),
            block_size=int(args.gptq_block_size),
            damp_percent=float(args.gptq_damp_percent),
            err_comp_gamma=0.0 if args.use_rtn else 1.0,
        )

        # 5. Per-step act_scale_table
        table = torch.zeros(num_steps, in_f, dtype=torch.float32, device=device)
        n_per_step: List[int] = []
        for t in range(num_steps):
            if t not in rotated_per_step:
                n_per_step.append(0)
                continue
            Xt = rotated_per_step[t]
            q_vec = torch.quantile(Xt.abs(), p_frac, dim=0).clamp_min(1e-6)
            table[t] = q_vec / a_qmax
            n_per_step.append(int(Xt.shape[0]))

        # Sanity finite
        for tname, tobj in [("W_r_q", W_r_q), ("rotation_R", rotation_R), ("table", table)]:
            if not torch.isfinite(tobj).all():
                raise RuntimeError(f"non-finite in {name}/{tname}")

        # Empty lowrank head (= dit_svdquant_v1 format with rank=0)
        empty_A = torch.zeros(out_f, 0, dtype=torch.float32, device=device)
        empty_B = torch.zeros(in_f, 0, dtype=torch.float32, device=device)
        rec = {
            "format": "dit_svdquant_v1",
            "weight_res_q": W_r_q.to(save_dtype).cpu().contiguous(),
            "lowrank_A": empty_A.to(save_dtype).cpu().contiguous(),
            "lowrank_B": empty_B.to(save_dtype).cpu().contiguous(),
            "act_scale_table": table.to(torch.float32).cpu().contiguous(),
            "duquant_rotation": rotation_R.to(save_dtype).cpu().contiguous(),
            # Fast runtime path (block-diag bmm + perm) — see gptq_layers.py
            # _has_duquant_rotation_blocks. Dense "duquant_rotation" above kept
            # for backward compat with old packs / non-fast runtimes.
            "duquant_rotation_blocks": r_in_blocks.to(save_dtype).cpu().contiguous(),
            "duquant_rotation_perm": perm_in.to(torch.int64).cpu().contiguous(),
            "weight_bits": int(args.w_bits),
            "a_bits": int(args.a_bits),
            "rank": 0,
            "in_features": int(in_f),
            "out_features": int(out_f),
            "n_calib_total": int(N),
            "n_calib_per_step": n_per_step,
            "act_percentile": float(args.act_percentile),
            "gptq_damp_percent": float(args.gptq_damp_percent),
        }
        if R_out_full is not None:
            rec["duquant_rotation_out"] = R_out_full.to(save_dtype).cpu().contiguous()
            if r_out_blocks is not None:
                rec["duquant_rotation_out_blocks"] = r_out_blocks.to(save_dtype).cpu().contiguous()
        new_records[name] = rec

        if li < 3 or (li + 1) % 12 == 0:
            print(f"[A2-GPTQ-PI05] {li+1}/{len(target_names)} {name} W={tuple(W.shape)} N={N} elapsed={time.time()-t0:.1f}s",
                  flush=True)

    new_records["__meta__"] = {
        "format": "pi05_a2lite_gptq_perstep_v1",
        "checkpoint": args.checkpoint,
        "max_samples": int(args.max_samples),
        "gptq_damp_percent": float(args.gptq_damp_percent),
        "duquant_block_size": int(args.duquant_block_size),
        "duquant_block_out": int(args.duquant_block_out),
        "rot_mode": "svd_hadamard",
        "perm": "weight-zigzag",
        "num_steps": num_steps,
        "act_percentile": float(args.act_percentile),
        "a_bits": int(args.a_bits),
        "w_bits": int(args.w_bits),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_records, out_path)
    print(f"[A2-GPTQ-PI05] wrote {out_path} ({len(new_records) - 1} layers)", flush=True)


if __name__ == "__main__":
    main()
