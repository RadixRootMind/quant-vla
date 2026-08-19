#!/usr/bin/env python
"""Minimal dump: DiT ff.net[0].proj per-step X for layers L02 / L08 / L14.

Mirrors the format of xw_dump_object_cal_perstep2.pt (one entry per layer
keyed as "DiT.L{idx:02d}.ff_net_0", each entry has W + X + X_per_step), so
plot_perstep_necessity.py can load it alongside the existing dump and append
a new layer-kind ("ff_net_0" = MLP input, the missing piece for "does
per-step also help MLP?").

Run with the awesome_quant_vla env (which has LIBERO + gr00t):
  CUDA_VISIBLE_DEVICES=0 LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH \\
  python tools/dump_dit_ffnet0_perstep.py \\
      --checkpoint $CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain \\
      --task-suite-name libero_object \\
      --num-samples 49 \\
      --output-pt experiment_results/svd_diagnostics_gr00t/xw_dump_object_cal_perstep_ffnet0.pt
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from tools.analyze_layerwise_quant_drift import ensure_libero_runtime, load_libero_samples
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import load_data_config
from gr00t.data.embodiment_tags import EmbodimentTag


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-config", default="examples.Libero.custom_data_config:LiberoDataConfig")
    ap.add_argument("--task-suite-name", required=True)
    ap.add_argument("--num-samples", type=int, default=49)
    ap.add_argument("--libero-resolution", type=int, default=256)
    ap.add_argument("--libero-num-trials-per-task", type=int, default=1)
    ap.add_argument("--libero-num-steps-wait", type=int, default=10)
    ap.add_argument("--libero-sampling-mode", default="one_per_task")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dit-layers", type=int, nargs="+", default=[2, 8, 14])
    ap.add_argument("--cap-per-step", type=int, default=512)
    ap.add_argument("--output-pt", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    ensure_libero_runtime()

    print(f"[ffnet0-dump] loading policy: {args.checkpoint}", flush=True)
    data_config = load_data_config(args.data_config)
    policy = Gr00tPolicy(
        model_path=args.checkpoint,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=EmbodimentTag(args.embodiment_tag),
        denoising_steps=args.denoising_steps,
        device=args.device,
    )
    model = policy.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    dit_layers = model.action_head.model.transformer_blocks
    print(f"[ffnet0-dump] DiT depth={len(dit_layers)}", flush=True)

    try:
        from gr00t.quantization.dit_step_context import get_current_dit_step
    except Exception:
        def get_current_dit_step(): return None

    cache_by_step: dict[str, dict[int, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))
    tag_to_module = {}

    def make_hook(tag):
        def hook(_m, inputs, _out):
            x = inputs[0]
            if x is None: return
            xa = x.detach().reshape(-1, x.shape[-1]).float().cpu()
            t = get_current_dit_step()
            if t is None: return
            cache_by_step[tag][int(t)].append(xa)
        return hook

    handles = []
    for li in args.dit_layers:
        if li >= len(dit_layers):
            continue
        ff_net = dit_layers[li].ff.net
        # ff.net is a ModuleList; index 0 is the GEGLU/projection-in
        mod0 = ff_net[0]
        # diffusers GEGLU wraps a .proj nn.Linear; if hooked module has .proj, use that
        target = getattr(mod0, "proj", mod0)
        if not isinstance(target, torch.nn.Linear):
            print(f"[ffnet0-dump][skip] L{li:02d} ff.net[0] is {type(target).__name__}, not Linear", flush=True)
            continue
        tag = f"DiT.L{li:02d}.ff_net_0"
        tag_to_module[tag] = target
        handles.append(target.register_forward_hook(make_hook(tag)))
        print(f"[ffnet0-dump] hooked {tag}: weight={tuple(target.weight.shape)}", flush=True)

    samples = load_libero_samples(args, data_config)
    if isinstance(samples, tuple):
        samples = samples[-1]
    samples = samples[: args.num_samples]
    print(f"[ffnet0-dump] driving {len(samples)} obs samples", flush=True)

    try:
        for i, sample in enumerate(samples, 1):
            obs = sample["obs"] if isinstance(sample, dict) and "obs" in sample else sample
            lang = sample.get("language", "?") if isinstance(sample, dict) else "?"
            if i == 1 or i % 10 == 0:
                print(f"  sample {i}/{len(samples)}: {lang!r}", flush=True)
            with torch.no_grad():
                _ = policy.get_action(obs)
    finally:
        for h in handles:
            h.remove()
    print("[ffnet0-dump] capture done", flush=True)

    # Build output dict matching xw_dump_*.pt format
    out: dict[str, dict] = {}
    for tag, mod in tag_to_module.items():
        by_step = cache_by_step.get(tag, {})
        if not by_step:
            print(f"[ffnet0-dump][skip] {tag}: no activations captured", flush=True)
            continue
        X_per_step = {}
        for t, chunks in by_step.items():
            X_t = torch.cat(chunks, dim=0)
            if X_t.shape[0] > args.cap_per_step:
                idx = torch.randperm(X_t.shape[0])[: args.cap_per_step]
                X_t = X_t[idx]
            X_per_step[int(t)] = X_t
        # Aggregate X across all steps for the .X field (compat with xw_dump format)
        X_agg = torch.cat([X_per_step[t] for t in sorted(X_per_step)], dim=0)
        W = mod.weight.detach().float().cpu()
        out[tag] = {"W": W, "X": X_agg, "X_per_step": X_per_step}
        sample_shape = next(iter(X_per_step.values())).shape
        print(f"  saved {tag}: W={tuple(W.shape)}  per-step X={tuple(sample_shape)}  total tokens={X_agg.shape[0]}", flush=True)

    Path(args.output_pt).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output_pt)
    print(f"[ffnet0-dump] wrote {args.output_pt}  ({len(out)} entries)", flush=True)


if __name__ == "__main__":
    main()
