#!/usr/bin/env python
"""Merge several GPTQ packs into one (records keyed by layer name).

The runtime loads a single ``GR00T_GPTQ_PATH``. When two sides are both GPTQ
(e.g. an LLM pack + a DiT pack, each covering disjoint layers), merge them into
one pack first:

    python -m tools.merge_packs --out merged/quantized.pt llm/quantized.pt dit/quantized.pt

Layer-name keys must be disjoint across inputs (an overlap is an error).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load(p):
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(p, map_location="cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output pack path (.pt)")
    ap.add_argument("packs", nargs="+", help="input pack .pt files to merge")
    args = ap.parse_args()

    merged: dict = {}
    sources = []
    for pp in args.packs:
        d = _load(pp)
        recs = {k: v for k, v in d.items() if k != "__meta__"}
        overlap = set(merged) & set(recs)
        if overlap:
            raise SystemExit(f"layer-name overlap between packs: {sorted(overlap)[:5]} ...")
        merged.update(recs)
        sources.append({"path": pp, "n_records": len(recs)})
        print(f"[merge] {pp}: {len(recs)} records")

    merged["__meta__"] = {"merged_from": sources}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out) + ".tmp"
    torch.save(merged, tmp)
    Path(tmp).replace(out)
    n = len([k for k in merged if k != "__meta__"])
    print(f"[merge] wrote {out}  ({n} records)")


if __name__ == "__main__":
    main()
