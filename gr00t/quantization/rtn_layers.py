"""Pure round-to-nearest (RTN) baseline for LLM / VLA weight + activation quant.

This is intentionally the simplest possible W4A4 baseline:
  - Weights: groupwise (default group_size=64 along input dim) asymmetric
    min-max int4. Dequantized to fp16/bf16 once at load time; matmul stays
    in original dtype (no speedup, this is a simulation baseline).
  - Activations: per-token asymmetric min-max int4 computed online per
    forward (one (scale, zp) per token row).
  - No rotation, no permutation, no calibration data.

Wiring mirrors enable_duquant_if_configured: a single env-var switch
(GR00T_RTN=1) replaces all Linear layers matching include/exclude regex
with RTNLinear.
"""
from __future__ import annotations

import os
import re
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .duquant_layers import _get_parent_module_and_attr


def _quant_dequant_weight_groupwise(W: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    """Per-group asymmetric min-max int quantization, returns dequantized tensor."""
    out_ch, in_ch = W.shape
    assert in_ch % group_size == 0, f"in_ch={in_ch} not divisible by group_size={group_size}"
    n_groups = in_ch // group_size
    Wf = W.detach().to(torch.float32).view(out_ch, n_groups, group_size)
    qmax = float((1 << bits) - 1)
    w_min = Wf.amin(dim=-1, keepdim=True)
    w_max = Wf.amax(dim=-1, keepdim=True)
    scale = (w_max - w_min).clamp_min(1e-8) / qmax
    zp = (-w_min / scale).round().clamp(0.0, qmax)
    q = (Wf / scale + zp).round().clamp(0.0, qmax)
    W_deq = (q - zp) * scale
    return W_deq.view(out_ch, in_ch)


class RTNLinear(nn.Module):
    """Drop-in replacement for nn.Linear with W4 groupwise + A4 per-token RTN.

    Pure round-to-nearest: weights are per-group asymmetric int4, activations
    are per-token asymmetric int4. No rotation, no SmoothQuant, no calibration.
    """

    def __init__(
        self,
        lin: nn.Linear,
        *,
        name: str,
        group_size: int,
        wbits: int,
        abits: int,
    ) -> None:
        super().__init__()
        self.name = name
        self.in_features = lin.in_features
        self.out_features = lin.out_features
        self.wbits = wbits
        self.abits = abits

        # Pick the largest divisor of in_features that's <= requested group_size.
        gs = group_size
        while gs > 1 and self.in_features % gs != 0:
            gs //= 2
        if gs < 1:
            gs = 1
        self.group_size = gs

        W = lin.weight.data

        W_deq = _quant_dequant_weight_groupwise(W, self.group_size, self.wbits)
        # Keep dequantized weight in the original dtype/device.
        self.register_buffer("weight", W_deq.to(dtype=W.dtype, device=W.device))

        if lin.bias is not None:
            self.register_buffer("bias", lin.bias.data.clone())
        else:
            self.bias = None

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"W{self.wbits} A{self.abits} group_size={self.group_size}"
        )

    def _quant_dequant_act(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token asymmetric min-max int{abits}, reduces over the last dim."""
        if self.abits <= 0 or self.abits >= 16:
            return x
        orig_dtype = x.dtype
        xf = x.to(torch.float32)
        qmax = float((1 << self.abits) - 1)
        x_min = xf.amin(dim=-1, keepdim=True)
        x_max = xf.amax(dim=-1, keepdim=True)
        scale = (x_max - x_min).clamp_min(1e-8) / qmax
        zp = (-x_min / scale).round().clamp(0.0, qmax)
        q = (xf / scale + zp).round().clamp(0.0, qmax)
        x_deq = (q - zp) * scale
        return x_deq.to(orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self._quant_dequant_act(x)
        return F.linear(x_q, self.weight, self.bias)


def _select_targets(
    model: nn.Module, include_regex: str, exclude_regex: str
) -> List[Tuple[str, nn.Linear]]:
    inc = re.compile(include_regex)
    exc = re.compile(exclude_regex)
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not inc.search(name) or exc.search(name):
            continue
        out.append((name, mod))
    return out


def enable_rtn_if_configured(model: nn.Module) -> None:
    """Replace matching nn.Linear with RTNLinear when GR00T_RTN* vars are set.

    Env vars:
      GR00T_RTN              master switch (any truthy value activates)
      GR00T_RTN_INCLUDE      regex of layer names to wrap
      GR00T_RTN_EXCLUDE      regex of layer names to skip
      GR00T_RTN_GROUP_SIZE   weight group size (default 64)
      GR00T_RTN_WBITS        weight bits (default 4)
      GR00T_RTN_ABITS        activation bits (default 4, set 16 to disable A quant)
      GR00T_RTN_DRYRUN       1 = list matched layers only, don't wrap
    """
    env = os.environ
    if not any(k.startswith("GR00T_RTN") for k in env):
        return

    include = env.get(
        "GR00T_RTN_INCLUDE",
        # Default = same as DuQuant on pi0.5: backbone Gemma + expert Gemma proj layers.
        r".*paligemma_with_expert\.(paligemma\.model\.language_model|gemma_expert\.model)"
        r"\.layers\.[0-9]+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
    )
    exclude = env.get(
        "GR00T_RTN_EXCLUDE",
        r"(?:^|\.)(vision_tower|vision_model|embeddings|embed_tokens|norm|layernorm|lm_head)(?:\.|$)",
    )
    group_size = int(env.get("GR00T_RTN_GROUP_SIZE", "64"))
    wbits = int(env.get("GR00T_RTN_WBITS", "4"))
    abits = int(env.get("GR00T_RTN_ABITS", "4"))
    dry_run = env.get("GR00T_RTN_DRYRUN", "0") not in ("0", "false", "False")

    targets = _select_targets(model, include, exclude)
    print(
        f"[GR00T-RTN] Matched Linear layers: {len(targets)} "
        f"(group_size={group_size}, W{wbits}A{abits}, dry_run={dry_run})"
    )

    if dry_run:
        for name, mod in targets:
            print(
                f"[GR00T-RTN][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features})"
            )
        return

    replaced = 0
    skipped = 0
    for name, mod in targets:
        if mod.in_features < group_size:
            print(
                f"[GR00T-RTN][SKIP] {name}: in_features={mod.in_features} < group_size={group_size}"
            )
            skipped += 1
            continue
        parent, attr = _get_parent_module_and_attr(model, name)
        rtn = RTNLinear(mod, name=name, group_size=group_size, wbits=wbits, abits=abits)
        setattr(parent, attr, rtn)
        print(
            f"[GR00T-RTN][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) "
            f"-> RTNLinear W{wbits} A{abits} gs={rtn.group_size}"
        )
        replaced += 1
    print(f"[GR00T-RTN] Total layers replaced: {replaced} (skipped {skipped})")
