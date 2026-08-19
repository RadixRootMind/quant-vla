"""Unified quantization entry point.

`enable_quant_if_configured` is the single hook the model calls to apply
quantization. The three building blocks are orthogonal and compose by scope —
each is self-gated by its own env vars and is a no-op when unset, and each
wrapper skips already-wrapped layers, so they can be combined freely as long as
their INCLUDE regexes target non-overlapping scopes:

  * GR00T_GPTQ      → load an offline GPTQ pack (GptqLinear); the pack may have a
                      DuQuant rotation (svd | svd_hadamard) baked in.
  * GR00T_RTN       → pure RTN (RTNLinear, no rotation).
  * GR00T_DUQUANT_* → DuQuant rotation (svd | svd_hadamard, via
                      GR00T_DUQUANT_ROT_MODE) + RTN quant, applied at runtime.

This realizes the per-side (LLM / DiT) × (rotation) × (quantizer) matrix the
benchmark exposes, e.g. LLM = DuQuant(svd_hadamard) + DiT = GPTQ pack is just
GR00T_DUQUANT_* (LLM scope) + GR00T_GPTQ_* (DiT scope) set together.

Quantization runs in a fixed order (GPTQ → RTN → DuQuant) so that, on the rare
overlapping-scope config, the outcome is deterministic.
"""

from __future__ import annotations

import os

from torch import nn


def enable_quant_if_configured(model: nn.Module) -> bool:
    """Apply configured quantization in-place. Returns True if anything wrapped."""
    applied = False

    gptq_requested = os.environ.get("GR00T_GPTQ", "0") not in ("0", "false", "False")
    try:
        from .gptq_layers import enable_gptq_if_configured

        if enable_gptq_if_configured(model):
            applied = True
    except Exception as e:
        if gptq_requested:
            raise RuntimeError(f"GPTQ requested but failed to apply: {e}") from e
        print(f"[GR00T] GPTQ not enabled or failed to apply: {e}")

    if any(k.startswith("GR00T_RTN") for k in os.environ):
        try:
            from .rtn_layers import enable_rtn_if_configured

            enable_rtn_if_configured(model)
            applied = True
            print("[GR00T] RTN replacement applied")
        except Exception as e:
            print(f"[GR00T] RTN failed to apply: {e}")

    if any(k.startswith("GR00T_DUQUANT_") and k != "GR00T_DUQUANT_PACKDIR" for k in os.environ):
        try:
            from .duquant_layers import enable_duquant_if_configured

            enable_duquant_if_configured(model)
            applied = True
            print("[GR00T] DuQuant replacement applied")
        except Exception as e:
            print(f"[GR00T] DuQuant failed to apply: {e}")

    return applied
