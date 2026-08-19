#!/usr/bin/env python
"""Minimal pi0.5 W/X dump for plot_quant_err_per_component.

Mirrors gr00t's xw_dump format. Samples paligemma layers {2, 8, 14} and expert
layers {2, 8, 14} (3 layers per module × 4 kinds = ~24 entries). Captures one
token-cap chunk per layer (pooled across denoising steps).

Run with the openpi venv:
  $OPENPI_ROOT/.venv/bin/python tools/dump_pi05_xw_for_quanterr.py \\
      --checkpoint $CHECKPOINTS_ROOT/pi05_libero_pytorch \\
      --obs-path  duquant_act_stats/pi05_libero_object_obs_n4.pt \\
      --output    experiment_results/svd_diagnostics_gr00t/xw_dump_pi05_object_n4.pt \\
      --max-samples 4 --token-cap 1024
"""
from __future__ import annotations
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


PALIGEMMA_PATTERN = re.compile(
    r"paligemma_with_expert\.paligemma\.model\.language_model\.layers\.(\d+)\..*\.(q_proj|o_proj|gate_proj|down_proj)$"
)
EXPERT_PATTERN = re.compile(
    r"paligemma_with_expert\.gemma_expert\.model\.layers\.(\d+)\..*\.(q_proj|o_proj|gate_proj|down_proj)$"
)
SAMPLE_LAYERS = {2, 8, 14}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-config", default="pi05_libero")
    p.add_argument("--obs-path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-samples", type=int, default=4)
    p.add_argument("--token-cap", type=int, default=1024)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    train_cfg = _config.get_config(args.data_config)
    print(f"[pi05-dump] loading {args.checkpoint}", flush=True)
    policy = _policy_config.create_trained_policy(train_cfg, args.checkpoint, pytorch_device=args.device)
    model = policy._model
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Resolve targets
    targets = {}  # tag (e.g. "paligemma.L02.q_proj") -> module
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        m = PALIGEMMA_PATTERN.match(name)
        if m:
            li, kd = int(m.group(1)), m.group(2)
            if li in SAMPLE_LAYERS:
                targets[f"paligemma.L{li:02d}.{kd}"] = (name, mod)
            continue
        m = EXPERT_PATTERN.match(name)
        if m:
            li, kd = int(m.group(1)), m.group(2)
            if li in SAMPLE_LAYERS:
                targets[f"expert.L{li:02d}.{kd}"] = (name, mod)
    print(f"[pi05-dump] {len(targets)} target layers", flush=True)

    cache = defaultdict(list)

    def make_hook(tag):
        def hook(_m, inputs):
            if not inputs: return
            x = inputs[0]
            if x is None: return
            x = x.detach().to(torch.float32).reshape(-1, x.shape[-1])
            if x.shape[0] > args.token_cap:
                idx = torch.randperm(x.shape[0], device=x.device)[: args.token_cap]
                x = x[idx]
            cache[tag].append(x.cpu())
        return hook

    handles = []
    for tag, (name, mod) in targets.items():
        handles.append(mod.register_forward_pre_hook(make_hook(tag)))

    samples = torch.load(args.obs_path, weights_only=False)
    if isinstance(samples, dict) and "samples" in samples:
        samples = samples["samples"]
    samples = samples[: args.max_samples]
    print(f"[pi05-dump] driving {len(samples)} obs samples", flush=True)
    try:
        with torch.no_grad():
            for i, obs in enumerate(samples, 1):
                lang = obs.get("prompt", "") if isinstance(obs, dict) else ""
                print(f"  sample {i}/{len(samples)}: {lang!r}", flush=True)
                _ = policy.infer(obs)
    finally:
        for h in handles:
            h.remove()

    out = {}
    for tag, (name, mod) in targets.items():
        chunks = cache.get(tag, [])
        if not chunks:
            print(f"  [skip] {tag}: no activations", flush=True)
            continue
        X = torch.cat(chunks, dim=0)
        if X.shape[0] > args.token_cap * 4:
            idx = torch.randperm(X.shape[0])[: args.token_cap * 4]
            X = X[idx]
        W = mod.weight.detach().float().cpu()
        out[tag] = {"W": W, "X": X.cpu()}
        print(f"  saved {tag}: W={tuple(W.shape)} X={tuple(X.shape)}", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)
    print(f"[pi05-dump] wrote {args.output}  ({len(out)} entries)", flush=True)


if __name__ == "__main__":
    main()
