#!/usr/bin/env python
"""Build offline pure-GPTQ quantized weights for GR00T.

For each target Linear layer we collect an input Gram matrix H from a handful
of LIBERO observations, run the blockwise GPTQ solver, and save the dequantized
weight as ``baseline_q``. Optionally a DuQuant input/output rotation can be
baked into the pack (``--duquant-rotation``); the runtime ``GptqLinear`` applies
it before the matmul.

This is the ``pure_gptq`` baseline (no rotation by default, no per-step scale,
no SVD low-rank head). For the DiT per-step / SVD recipe use
``build_dit_a2lite_svd_gptq_perstep.py`` instead.
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

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
    normalized_input_no_inference,
    seed_everything,
)
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.model.policy import COMPUTE_DTYPE  # noqa: E402
from gr00t.quantization.gptq_layers import gptq_quantize_weight  # noqa: E402
from gr00t.quantization.duquant_layers import select_targets  # noqa: E402

DEFAULT_LIBERO_ROOT = os.environ.get("LIBERO_ROOT", os.path.join(os.path.expanduser("~"), "LIBERO"))
DEFAULT_LIBERO_DATASET = f"{DEFAULT_LIBERO_ROOT}/datasets/lerobot_libero_10"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-path", required=True, help="Where to save the GPTQ quantized weights (.pt)")
    p.add_argument("--weight-bits", type=int, default=4)
    p.add_argument("--gptq-block-size", type=int, default=128)
    p.add_argument("--gptq-damp-percent", type=float, default=0.01)
    p.add_argument("--gptq-err-comp-gamma", type=float, default=1.0,
                   help="GPTQ inter-column error-compensation strength. 1.0 = full GPTQ; "
                        "0.0 = RTN (each block quantized independently).")
    p.add_argument("--save-dtype", choices=["float16", "float32"], default="float16")

    p.add_argument("--data-source", default="libero", choices=["libero"])
    p.add_argument("--dataset-path", default=DEFAULT_LIBERO_DATASET)
    p.add_argument("--task-suite-name", default="libero_10")
    p.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--device", default="cuda")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task", choices=["sequential", "one_per_task", "one_per_trial"])
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--include-regex", default=DEFAULT_INCLUDE_REGEX)
    p.add_argument("--exclude-regex", default=DEFAULT_EXCLUDE_REGEX)
    p.add_argument("--scope", default="")
    p.add_argument("--start-layer", type=int, default=0)
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--token-cap", type=int, default=512)
    p.add_argument("--duquant-rotation", action="store_true",
                   help="Bake a DuQuant per-block input rotation into the pack as preprocessing. "
                        "Input rotated by R (block-diag of rotations from FP weight), then GPTQ on "
                        "rotated (W, H). Rotation is stored in the record and applied at runtime "
                        "before the GPTQ forward.")
    p.add_argument("--duquant-block-size", type=int, default=64)
    p.add_argument("--duquant-rot-mode", default="svd",
                   choices=["svd", "hadamard", "svd_hadamard", "random_hadamard"],
                   help="DuQuant rotation mode. 'svd' = original DuQuant SVD basis; "
                        "'svd_hadamard' = A2-lite (SVD @ Hadamard); 'hadamard' = pure Hadamard.")
    p.add_argument("--duquant-permute", action="store_true",
                   help="Enable zigzag permutation (matches GR00T_DUQUANT_PERMUTE=1 at runtime).")
    p.add_argument("--duquant-output-rotation", action="store_true",
                   help="Also apply block-diag output rotation R_out (matches runtime row_rot=restore).")
    p.add_argument("--duquant-block-out", type=int, default=0,
                   help="Block size for output rotation; 0 → same as --duquant-block-size.")
    # ---- Per-layer-type overrides (attention vs MLP) ----
    p.add_argument("--attn-block-size", type=int, default=-1)
    p.add_argument("--attn-block-out", type=int, default=-1)
    p.add_argument("--attn-rot-mode", default="",
                   choices=["", "svd", "hadamard", "svd_hadamard", "random_hadamard"])
    p.add_argument("--attn-gptq-damp", type=float, default=-1.0)
    p.add_argument("--attn-group-size", type=int, default=0,
                   help="Per-group W4 quantization on attention layers (0 = per-row).")
    p.add_argument("--mlp-block-size", type=int, default=-1)
    p.add_argument("--mlp-block-out", type=int, default=-1)
    p.add_argument("--mlp-rot-mode", default="",
                   choices=["", "svd", "hadamard", "svd_hadamard", "random_hadamard"])
    p.add_argument("--mlp-gptq-damp", type=float, default=-1.0)
    p.add_argument("--mlp-group-size", type=int, default=0)
    return p.parse_args()


def _resolve_per_kind(args, name: str) -> dict:
    """Return effective {block_size, block_out, rot_mode, gptq_damp, group_size} for a layer name."""
    is_attn = "attn1" in name
    prefix = "attn" if is_attn else "mlp"
    def _pick(per, default):
        return per if per not in (-1, -1.0, "") else default
    block_size = _pick(getattr(args, f"{prefix}_block_size"), int(args.duquant_block_size))
    block_out_default = int(args.duquant_block_out) if int(args.duquant_block_out) > 0 else int(args.duquant_block_size)
    block_out = _pick(getattr(args, f"{prefix}_block_out"), block_out_default)
    rot_mode = _pick(getattr(args, f"{prefix}_rot_mode"), args.duquant_rot_mode)
    damp = _pick(getattr(args, f"{prefix}_gptq_damp"), float(args.gptq_damp_percent))
    group_size = int(getattr(args, f"{prefix}_group_size"))
    return {
        "is_attn": is_attn,
        "block_size": int(block_size),
        "block_out": int(block_out),
        "rot_mode": rot_mode,
        "gptq_damp": float(damp),
        "group_size": group_size,
    }


def resolve_target_layers(policy, args) -> list[str]:
    targets = select_targets(
        policy.model,
        include_regex=args.include_regex,
        exclude_regex=args.exclude_regex,
        scope_prefix=args.scope or None,
        whitelist=None,
        blacklist=None,
    )
    names = [n for n, _ in targets]
    names = names[args.start_layer:]
    if args.max_layers > 0:
        names = names[: args.max_layers]
    return names


def _clear_quant_env() -> None:
    for prefix in ("GR00T_DUQUANT_", "GR00T_GPTQ"):
        for key in list(os.environ.keys()):
            if key == prefix or key.startswith(prefix):
                os.environ.pop(key, None)


def _autocast_context(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=COMPUTE_DTYPE)
    return nullcontext()


def collect_layer_gram(
    policy,
    normalized_samples,
    layer_name: str,
    token_cap: int,
    rng: np.random.Generator,
    device: str,
) -> tuple[torch.Tensor, int]:
    module = get_named_module(policy.model, layer_name)
    gram: Optional[torch.Tensor] = None
    n_tokens = 0
    gram_device = module.weight.device if hasattr(module, "weight") else torch.device(device)

    def hook(_m, inputs):
        nonlocal gram, n_tokens
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
        if token_cap > 0 and x.shape[0] > token_cap:
            idx = torch.from_numpy(rng.choice(x.shape[0], size=token_cap, replace=False)).to(x.device)
            x = x.index_select(0, idx)
        if gram is None:
            gram = torch.zeros((x.shape[1], x.shape[1]), dtype=torch.float32, device=gram_device)
        x = x.to(device=gram_device, dtype=torch.float32, non_blocking=True)
        gram += x.t() @ x
        n_tokens += int(x.shape[0])

    handle = module.register_forward_pre_hook(hook)
    try:
        with torch.no_grad(), _autocast_context(device):
            for sample in normalized_samples:
                seed_everything(sample["seed"])
                policy.model.get_action(sample["normalized"])
    finally:
        handle.remove()

    if gram is None or n_tokens == 0:
        raise RuntimeError(f"Failed to collect activations for layer '{layer_name}'")
    return gram / float(n_tokens), n_tokens


def _build_duquant_rotation(W, args, per_kind, solve_device):
    """Return (rotation_R, rotation_R_out) for the requested DuQuant mode, or (None, None)."""
    if args.duquant_permute or args.duquant_output_rotation:
        from gr00t.quantization.duquant_preprocess import pack_weight
        block_size_eff = per_kind["block_size"]
        block_out = per_kind["block_out"]
        rot_mode_eff = per_kind["rot_mode"]
        pack = pack_weight(
            W,
            block_size=int(block_size_eff),
            block_out_size=int(block_out),
            enable_permute=bool(args.duquant_permute),
            lambda_smooth=0.15,
            perm_score="weight",
            rot_mode=rot_mode_eff,
        )
        in_features = int(W.shape[1])
        out_features = int(W.shape[0])
        R_in_full = torch.zeros((in_features, in_features), dtype=torch.float32, device=solve_device)
        if pack.R_in_blocks:
            blk = int(pack.meta.get("block_size", int(block_size_eff)))
            for b, R in pack.R_in_blocks.items():
                rs, re = b * blk, min((b + 1) * blk, in_features)
                R_in_full[rs:re, rs:re] = torch.from_numpy(R[: (re - rs), : (re - rs)].astype("float32")).to(solve_device)
        if pack.perm is not None:
            P = torch.zeros((in_features, in_features), dtype=torch.float32, device=solve_device)
            perm_t = torch.from_numpy(pack.perm.astype("int64")).to(solve_device)
            P[perm_t, torch.arange(in_features, device=solve_device)] = 1.0
            rotation_R = P @ R_in_full
        else:
            rotation_R = R_in_full
        rotation_R_out = None
        if args.duquant_output_rotation and pack.R_out_blocks:
            R_out_full = torch.zeros((out_features, out_features), dtype=torch.float32, device=solve_device)
            blk_o = int(pack.meta.get("block_out_size", block_out))
            for b, R in pack.R_out_blocks.items():
                rs, re = b * blk_o, min((b + 1) * blk_o, out_features)
                R_out_full[rs:re, rs:re] = torch.from_numpy(R[: (re - rs), : (re - rs)].astype("float32")).to(solve_device)
            rotation_R_out = R_out_full
        return rotation_R, rotation_R_out
    else:
        from gr00t.quantization.duquant_preprocess import compute_duquant_rotation_only
        R_cpu, _perm = compute_duquant_rotation_only(
            W, block_size=int(per_kind["block_size"]), enable_permute=False,
            rot_mode=per_kind["rot_mode"],
        )
        return R_cpu.to(dtype=torch.float32, device=solve_device), None


def main() -> None:
    args = parse_args()
    ensure_libero_runtime()
    _clear_quant_env()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    print(f"[GPTQ] loading FP policy ... {args.checkpoint}")
    data_config = load_data_config(args.data_config)
    policy = load_policy(args, data_config, quantized_layers=None)
    policy.model.eval()

    layer_names = resolve_target_layers(policy, args)
    print(f"[GPTQ] target layers: {len(layer_names)}")

    print(f"[GPTQ] loading {args.num_samples} LIBERO observations")
    _, _, samples = load_libero_samples(args, data_config)
    samples = samples[: args.num_samples]
    normalized_samples = []
    for sample in samples:
        normalized_samples.append(
            {
                "seed": sample["seed"],
                "normalized": normalized_input_no_inference(policy, sample["obs"]),
            }
        )
    print(f"[GPTQ] prepared {len(normalized_samples)} normalized observations")

    save_dtype = torch.float16 if args.save_dtype == "float16" else torch.float32
    records: dict[str, dict] = {}

    for idx, layer_name in enumerate(layer_names, start=1):
        module = get_named_module(policy.model, layer_name)
        if not isinstance(module, torch.nn.Linear):
            continue
        print(f"[GPTQ] layer {idx}/{len(layer_names)}: {layer_name}")
        solve_device = module.weight.device
        H, n_tokens = collect_layer_gram(
            policy, normalized_samples, layer_name, args.token_cap, rng, str(args.device),
        )
        W = module.weight.detach().to(dtype=torch.float32, device=solve_device)
        H_dev = H.to(dtype=torch.float32, device=solve_device)

        per_kind = _resolve_per_kind(args, layer_name)
        rotation_R = None
        rotation_R_out = None
        if args.duquant_rotation:
            rotation_R, rotation_R_out = _build_duquant_rotation(W, args, per_kind, solve_device)
            W = W @ rotation_R
            if rotation_R_out is not None:
                W = rotation_R_out @ W
            H_dev = rotation_R.t() @ H_dev @ rotation_R

        baseline_q = gptq_quantize_weight(
            W,
            H_dev,
            bits=args.weight_bits,
            block_size=args.gptq_block_size,
            damp_percent=per_kind["gptq_damp"],
            group_size=int(per_kind["group_size"]),
            err_comp_gamma=float(args.gptq_err_comp_gamma),
        )

        rec_dict = {
            "baseline_q": baseline_q.to(dtype=save_dtype).cpu().contiguous(),
            "weight_bits": int(args.weight_bits),
            "n_calib_tokens": int(n_tokens),
            "gptq_block_size": int(args.gptq_block_size),
            "gptq_damp_percent": float(args.gptq_damp_percent),
        }
        if rotation_R is not None:
            rec_dict["duquant_rotation"] = rotation_R.to(dtype=torch.float16).cpu().contiguous()
        if rotation_R_out is not None:
            rec_dict["duquant_rotation_out"] = rotation_R_out.to(dtype=torch.float16).cpu().contiguous()
        records[layer_name] = rec_dict
        print(
            f"[GPTQ] saved {layer_name} (shape={tuple(baseline_q.shape)} calib_tokens={n_tokens})"
        )

    payload = {
        "__meta__": {
            "checkpoint": args.checkpoint,
            "weight_bits": int(args.weight_bits),
            "num_samples": int(args.num_samples),
            "token_cap": int(args.token_cap),
            "gptq_block_size": int(args.gptq_block_size),
            "gptq_damp_percent": float(args.gptq_damp_percent),
            "gptq_err_comp_gamma": float(args.gptq_err_comp_gamma),
            "duquant_rotation": bool(args.duquant_rotation),
            "duquant_rot_mode": args.duquant_rot_mode if args.duquant_rotation else None,
            "save_dtype": args.save_dtype,
        }
    }
    payload.update(records)
    tmp_path = str(out_path) + f".tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    print(f"[GPTQ] wrote {out_path}")


if __name__ == "__main__":
    main()
