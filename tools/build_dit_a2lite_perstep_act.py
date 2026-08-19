#!/usr/bin/env python
"""Augment an existing A2-lite + GPTQ DiT pack with a per-step act_scale_table.

Existing pack (from build_dit_a2lite_gptq_full.sh):
  baseline_q, duquant_rotation, duquant_rotation_out, ...
  → at runtime, GptqLinear applies x' = x @ duquant_rotation; uses a
    single PercentileCalibrator-derived _act_scale across all 8 DiT steps.

This script adds the missing piece:
  act_scale_table[t, j] = q999(|x_t @ duquant_rotation|, dim=0) / qmax(a_bits)
  shape [num_steps, in_features], baked into the pack.

At runtime, GptqLinear automatically picks _act_scale_table[t] when the
field is present and get_current_dit_step() is set — no runtime change needed.

The script:
  1. Loads the existing pack (one .pt per suite)
  2. Loads the matching gr00t-N1.5-libero-* policy
  3. Drives N LIBERO trajectories through policy.get_action(); during each
     DiT denoising loop, hooks every Linear that has 'duquant_rotation' in
     the pack to collect inputs per-step (via dit_step_context).
  4. For each layer: x_rot = x[:, ...] @ duquant_rotation; per-step quantile;
     bakes act_scale_table into the record.
  5. Saves the augmented pack to a NEW path (does not overwrite input).

Usage:
  python tools/build_dit_a2lite_perstep_act.py \
      --checkpoint $CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain \
      --input-pack results/multisuite_packs/object_DiT_a2lite_GPTQ_full/quantized.pt \
      --output-pack results/multisuite_packs/object_DiT_a2lite_GPTQ_perstep/quantized.pt \
      --task-suite-name libero_object \
      --num-samples 10 --token-cap 1024 --num-steps 8 --a-bits 4
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_layerwise_quant_drift import (  # noqa: E402
    ensure_libero_runtime,
    get_named_module,
    load_libero_samples,
    load_policy,
    seed_everything,
)
from tools.analyze_layerwise_quant_drift import normalized_input_no_inference  # noqa: E402
from gr00t.experiment.data_config import load_data_config  # noqa: E402
from gr00t.quantization.dit_step_context import get_current_dit_step  # noqa: E402
from gr00t.quantization.duquant_preprocess import qmax  # noqa: E402

DIT_PATTERN = re.compile(
    r"action_head\.model\.transformer_blocks\.\d+\."
    r"(attn1\.(to_q|to_k|to_v|to_out\.0)|ff\.net\.(0\.proj|2))"
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-pack", required=True,
                   help="Existing A2-lite+GPTQ pack (.pt) to augment.")
    p.add_argument("--output-pack", required=True,
                   help="Where to write the augmented pack. New file; input is "
                        "not modified.")

    p.add_argument("--task-suite-name", default="libero_object")
    p.add_argument("--data-config",
                   default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--video-backend", default="torchvision_av")
    p.add_argument("--device", default="cuda")

    p.add_argument("--denoising-steps", type=int, default=8,
                   help="DiT denoising steps. Must match runtime config.")
    p.add_argument("--num-steps", type=int, default=8,
                   help="Bucket count for act_scale_table (= denoising-steps).")

    p.add_argument("--num-samples", type=int, default=10,
                   help="LIBERO trajectories to drive for activation capture.")
    p.add_argument("--libero-num-trials-per-task", type=int, default=2)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task",
                   choices=["sequential", "one_per_task", "one_per_trial"])
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--token-cap", type=int, default=1024,
                   help="Per-forward token cap before pooling per layer per step.")
    p.add_argument("--a-bits", type=int, default=4)
    p.add_argument("--act-percentile", type=float, default=99.9)

    # Misc
    p.add_argument("--save-dtype", default="float32",
                   choices=["float16", "float32"],
                   help="Storage dtype for act_scale_table. FP32 recommended — "
                        "FP16 underflows on channels with q999 < ~1e-6.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    seed_everything(args.seed)
    ensure_libero_runtime()

    in_path = Path(args.input_pack)
    out_path = Path(args.output_pack)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    assert in_path.exists(), f"input pack does not exist: {in_path}"
    assert in_path != out_path, "output must differ from input"

    print(f"[A2P-PERSTEP] loading existing pack {in_path}", flush=True)
    pack: Dict = torch.load(in_path, map_location="cpu", weights_only=False)

    # Identify DiT records with duquant_rotation — these are the layers to augment.
    dit_layers: Dict[str, Dict] = {}
    for name, rec in pack.items():
        if not isinstance(rec, dict):
            continue
        if not DIT_PATTERN.match(name):
            continue
        if "duquant_rotation" not in rec:
            print(f"[A2P-PERSTEP][skip] {name}: no duquant_rotation in record", flush=True)
            continue
        dit_layers[name] = rec
    print(f"[A2P-PERSTEP] {len(dit_layers)} DiT layers to augment", flush=True)
    if not dit_layers:
        raise SystemExit("no augmentable DiT records found")

    # Load policy (we only need it to drive DiT forward; weights from disk).
    print(f"[A2P-PERSTEP] loading policy: {args.checkpoint}", flush=True)
    data_config = load_data_config(args.data_config)
    # quantized_layers=None means policy loads FP weights; we just need the
    # activation distribution, not the quantized DiT inference path.
    policy = load_policy(args, data_config, quantized_layers=None)
    policy.model.eval()
    for p in policy.model.parameters():
        p.requires_grad_(False)

    # Hook DiT layer inputs, bucketed by current_dit_step.
    cache: Dict[str, Dict[int, List[torch.Tensor]]] = {
        n: defaultdict(list) for n in dit_layers
    }

    def make_hook(name: str):
        def hook(_m, inputs):
            if not inputs:
                return
            x = inputs[0]
            if x is None:
                return
            t = get_current_dit_step()
            if t is None:
                return  # outside DiT denoising loop (e.g. LLM forwards)
            x = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
            if x.shape[0] > args.token_cap:
                idx = torch.randperm(x.shape[0], device=x.device)[: args.token_cap]
                x = x[idx]
            cache[name][int(t)].append(x.cpu())
        return hook

    handles = []
    for name in dit_layers:
        try:
            mod = get_named_module(policy.model, name)
        except Exception as e:
            print(f"[A2P-PERSTEP][skip-hook] {name}: {e}", flush=True)
            continue
        handles.append(mod.register_forward_pre_hook(make_hook(name)))
    print(f"[A2P-PERSTEP] {len(handles)} hooks registered", flush=True)

    # Drive samples through policy.
    samples = load_libero_samples(args, data_config)
    if isinstance(samples, tuple):
        samples = samples[-1]
    samples = samples[: args.num_samples]
    print(f"[A2P-PERSTEP] running {len(samples)} samples", flush=True)
    try:
        with torch.no_grad():
            for i, sample in enumerate(samples):
                norm = normalized_input_no_inference(policy, sample["obs"])
                policy.model.get_action(norm)
                if (i + 1) % 5 == 0 or i == 0:
                    print(f"[A2P-PERSTEP] sample {i+1}/{len(samples)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    print("[A2P-PERSTEP] capture done", flush=True)

    # Free policy GPU memory before per-layer work.
    del policy
    torch.cuda.empty_cache()

    # Compute per-step act_scale_table per layer.
    save_dtype = torch.float32 if args.save_dtype == "float32" else torch.float16
    a_qmax = qmax(int(args.a_bits))
    p_frac = float(args.act_percentile) / 100.0
    num_steps = int(args.num_steps)

    new_pack: Dict = dict(pack)  # shallow copy; we'll overwrite DiT records
    diag_rows: List[Dict] = []

    for li, (name, rec) in enumerate(dit_layers.items()):
        t0 = time.time()
        R = rec["duquant_rotation"]  # CPU tensor, shape [in, in]
        if isinstance(R, torch.Tensor):
            R_dev = R.to(device=device, dtype=torch.float32)
        else:
            R_dev = torch.as_tensor(R, device=device, dtype=torch.float32)
        in_features = R_dev.shape[0]

        per_step = cache[name]
        if not per_step:
            print(f"[A2P-PERSTEP][skip] {name}: no activations captured", flush=True)
            continue

        table = torch.zeros(num_steps, in_features, dtype=torch.float32, device=device)
        n_per_step: List[int] = []
        for t in range(num_steps):
            chunks = per_step.get(t, [])
            if not chunks:
                n_per_step.append(0)
                # Leave row as zeros; runtime load clamps to 1e-8 anyway.
                continue
            Xt = torch.cat(chunks, dim=0).to(device)
            X_rot = Xt @ R_dev
            q_vec = torch.quantile(X_rot.abs(), p_frac, dim=0).clamp_min(1e-6)
            table[t] = q_vec / a_qmax
            n_per_step.append(int(X_rot.shape[0]))

        # Bake into the record (clone so we don't mutate the loaded pack).
        new_rec = dict(rec)
        new_rec["act_scale_table"] = table.to(save_dtype).cpu().contiguous()
        new_rec["a_bits"] = int(args.a_bits)
        new_rec["num_steps"] = num_steps
        new_rec["act_percentile"] = float(args.act_percentile)
        new_rec["diag_a2lite_perstep_n_per_step"] = n_per_step
        new_pack[name] = new_rec

        elapsed = time.time() - t0
        row = {
            "layer": name,
            "elapsed": elapsed,
            "n_per_step": n_per_step,
            "table_min": float(table.min().item()),
            "table_max": float(table.max().item()),
            "table_mean": float(table.mean().item()),
        }
        diag_rows.append(row)
        if li < 3 or (li + 1) % 12 == 0:
            print(
                f"[A2P-PERSTEP] {li+1}/{len(dit_layers)} {name}: "
                f"n_per_step={n_per_step}  "
                f"table min/mean/max={row['table_min']:.4g}/{row['table_mean']:.4g}/{row['table_max']:.4g}  "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    # Mark __meta__ so we can tell augmented packs apart at load time.
    meta = dict(new_pack.get("__meta__", {}) if isinstance(new_pack.get("__meta__"), dict) else {})
    meta["a2lite_perstep_act"] = {
        "augmented_from": str(in_path),
        "num_steps": num_steps,
        "a_bits": int(args.a_bits),
        "act_percentile": float(args.act_percentile),
        "num_samples": int(args.num_samples),
        "token_cap": int(args.token_cap),
        "layers_augmented": len(diag_rows),
    }
    new_pack["__meta__"] = meta

    torch.save(new_pack, out_path)
    print(f"[A2P-PERSTEP] wrote {out_path}", flush=True)
    print(f"[A2P-PERSTEP] augmented {len(diag_rows)} / {len(dit_layers)} DiT layers",
          flush=True)


if __name__ == "__main__":
    main()
