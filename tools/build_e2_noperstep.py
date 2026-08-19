#!/usr/bin/env python
"""Build an E2-no-perstep variant: take an existing E2 pack
(LLM gptq + DiT a2lite RTN with per-step act_scale_table), and collapse
each DiT layer's act_scale_table from shape [8, in_features] to a single-bucket
equivalent (mean over the 8 steps, then expanded back to [8, in_features]).

Why this is the right "no per-step" baseline: keeping the dispatch path
identical (the runtime still indexes act_scale_table[cur]) but making all 8
rows identical isolates exactly the per-step variation as the only changed
variable. This matches §4.2 of methodology.md (1.3pp expected loss from
8-step magnitude drift collapsed into a single global scale).

Usage:
  python tools/build_e2_noperstep.py \\
      --src-pack results/multisuite_packs/object_LLM_a2liteGPTQ_DiT_RTN_perstep/quantized.pt \\
      --out-pack results/multisuite_packs/object_LLM_a2liteGPTQ_DiT_RTN_noPS/quantized.pt
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import torch


def collapse_act_scale_table(table: torch.Tensor) -> torch.Tensor:
    """Replace [num_steps, in_features] with mean(dim=0).expand(num_steps, -1)
    so every step sees the same scale."""
    if table.dim() != 2:
        raise ValueError(f"expected [num_steps, in_features] table, got shape {tuple(table.shape)}")
    mean_row = table.mean(dim=0, keepdim=True)        # [1, in_features]
    return mean_row.expand_as(table).contiguous()      # [num_steps, in_features]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-pack", required=True)
    ap.add_argument("--out-pack", required=True)
    args = ap.parse_args()

    src = Path(args.src_pack)
    out = Path(args.out_pack)
    if not src.is_file():
        raise FileNotFoundError(src)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[E2-noPS] loading {src}")
    pack = torch.load(src, weights_only=False)

    n_layers_modified = 0
    n_layers_skipped = 0
    for name, rec in pack.items():
        if name == "__meta__":
            continue
        if not isinstance(rec, dict):
            continue
        tbl = rec.get("act_scale_table", None)
        if tbl is None:
            n_layers_skipped += 1
            continue
        if not torch.is_tensor(tbl):
            n_layers_skipped += 1
            continue
        if tbl.dim() != 2 or tbl.shape[0] <= 1:
            # already single-bucket or unexpected shape
            n_layers_skipped += 1
            continue
        rec["act_scale_table"] = collapse_act_scale_table(tbl)
        n_layers_modified += 1

    # Stamp meta so downstream eval scripts can tell it apart
    meta = pack.setdefault("__meta__", {})
    meta["perstep_disabled"] = True
    meta["perstep_collapse_method"] = "mean_over_steps"
    meta["perstep_source_pack"] = str(src)

    print(f"[E2-noPS] modified {n_layers_modified} layers, skipped {n_layers_skipped} (no table or already single)")
    print(f"[E2-noPS] writing {out}")
    torch.save(pack, out)
    print(f"[E2-noPS] done ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
