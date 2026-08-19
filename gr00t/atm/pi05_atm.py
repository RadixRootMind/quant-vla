"""pi0.5 ATM/OHB: per-head query and output scaling for HF Gemma attention.

GR00T's original ATM (``gr00t.atm.dit_atm``) targets diffusers ``Attention``
modules under ``action_head.model.transformer_blocks.*``. pi0.5 has no DiT;
its attention is plain HF ``GemmaAttention`` (from openpi's
``transformers_replace``) under
``paligemma_with_expert.{paligemma.model.language_model, gemma_expert.model}.layers.[N].self_attn``.

This module mirrors the dit_atm API but uses forward hooks on ``q_proj`` /
``o_proj`` for runtime α/β scaling and a monkey-patched forward for the
calibration capture path.

Env namespace is shared with dit_atm — switch backends via ``GR00T_ATM_SCOPE``:
  GR00T_ATM_SCOPE=dit  → dit_atm.py handles it (GR00T-N1.5)
  GR00T_ATM_SCOPE=pi05 → this module handles it

Runtime API:
  enable_pi05_atm_if_configured(model)

Calibration API (used by tools/calibrate_atm_pi05.py):
  install_pi05_capture(model, scope)         # monkey-patch attention.forward
  register_pi05_atm_capture(model, cb)       # per-head logits std
  register_pi05_ohb_capture(model, cb)       # per-head output RMS
  clear_pi05_atm_capture(model)
  uninstall_pi05_capture(model)
"""

from __future__ import annotations

import json
import os
import types
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

# Shared env names with dit_atm.py so the same calibration JSON / enable flag
# round-trips between GR00T-N1.5 and pi0.5 backends.
ATM_ENABLE_ENV = "GR00T_ATM_ENABLE"
ATM_ALPHA_ENV = "GR00T_ATM_ALPHA_PATH"
ATM_SCOPE_ENV = "GR00T_ATM_SCOPE"
OHB_ENABLE_ENV = "GR00T_OHB_ENABLE"
OHB_SCOPE_ENV = "GR00T_OHB_SCOPE"
OHB_FALLBACK_ENV = "GR00T_OHB_FALLBACK"

_PI05_HOOK_PATCH_FLAG = "_pi05_atm_hooked"
_PI05_CAPTURE_PATCH_FLAG = "_pi05_atm_capture_patched"
_PI05_ORIG_FORWARD_ATTR = "_pi05_atm_orig_forward"


# ============================================================
# Module matching
# ============================================================

def _is_pi05_attention(name: str, module: nn.Module, scope: str = "pi05") -> bool:
    """Match GemmaAttention modules under paligemma_with_expert.

    Detection avoids import-time dependency on openpi/transformers by checking
    class name plus presence of {q,k,v,o}_proj children.
    """
    if type(module).__name__ != "GemmaAttention":
        return False
    if not all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return False
    if scope in ("pi05", "all"):
        return True
    return scope in name


def _num_heads_and_dim(module: nn.Module) -> Tuple[int, int]:
    head_dim = int(module.head_dim)
    num_heads = int(module.q_proj.out_features) // head_dim
    return num_heads, head_dim


# ============================================================
# Runtime: forward hooks for α (on q_proj) and β (on o_proj)
# ============================================================

def _make_q_alpha_hook(attn_module: nn.Module, num_heads: int, head_dim: int):
    """Post-hook on q_proj. Multiplies per-head when ``_atm_alpha_all`` is set."""

    def hook(_q_mod, _inputs, output: torch.Tensor) -> torch.Tensor:
        alpha = getattr(attn_module, "_atm_alpha_all", None)
        if alpha is None:
            return output
        a = alpha.to(dtype=output.dtype, device=output.device)
        # q_proj output is (..., num_heads*head_dim)
        shape = output.shape
        x = output.view(*shape[:-1], num_heads, head_dim)
        # broadcast a over leading dims and head_dim
        a_view = [1] * (x.dim() - 2) + [num_heads, 1]
        x = x * a.view(*a_view)
        return x.view(*shape)

    return hook


def _make_o_beta_prehook(attn_module: nn.Module, num_heads: int, head_dim: int):
    """Pre-hook on o_proj. Multiplies per-head when ``_ohb_beta_perhead`` is set,
    else falls back to scalar ``_ohb_beta_scalar``.
    """

    def prehook(_o_mod, inputs):
        x = inputs[0]
        beta_ph = getattr(attn_module, "_ohb_beta_perhead", None)
        if beta_ph is not None:
            b = beta_ph.to(dtype=x.dtype, device=x.device)
            shape = x.shape
            t = x.view(*shape[:-1], num_heads, head_dim)
            b_view = [1] * (t.dim() - 2) + [num_heads, 1]
            t = t * b.view(*b_view)
            return (t.view(*shape),)
        beta_s = getattr(attn_module, "_ohb_beta_scalar", None)
        if beta_s is not None and float(beta_s) != 1.0:
            return (x * float(beta_s),)
        return None  # no change

    return prehook


def _ensure_pi05_atm_hooks(model: nn.Module, scope: str = "pi05") -> int:
    """Register α/β hooks on every matching attention. Idempotent."""
    count = 0
    for name, module in model.named_modules():
        if not _is_pi05_attention(name, module, scope=scope):
            continue
        if getattr(module, _PI05_HOOK_PATCH_FLAG, False):
            continue
        num_heads, head_dim = _num_heads_and_dim(module)
        module.q_proj.register_forward_hook(_make_q_alpha_hook(module, num_heads, head_dim))
        module.o_proj.register_forward_pre_hook(_make_o_beta_prehook(module, num_heads, head_dim))
        setattr(module, _PI05_HOOK_PATCH_FLAG, True)
        setattr(module, "_pi05_atm_path", name)
        count += 1
    return count


@dataclass
class _AlphaSummary:
    matched_layers: int = 0
    total_heads: int = 0


def enable_pi05_atm_if_configured(model: nn.Module) -> None:
    """Read GR00T_ATM_* / GR00T_OHB_* env vars and attach α/β scalars."""
    atm_enabled = os.environ.get(ATM_ENABLE_ENV, "0") not in ("0", "false", "False", "")
    ohb_enabled = os.environ.get(OHB_ENABLE_ENV, "0") not in ("0", "false", "False", "")
    if not atm_enabled and not ohb_enabled:
        return

    scope = os.environ.get(ATM_SCOPE_ENV, "dit")
    if scope != "pi05":
        # Let dit_atm.py handle scope=dit; nothing for us to do.
        return

    alpha_path = os.environ.get(ATM_ALPHA_ENV)
    if not alpha_path or not os.path.exists(alpha_path):
        print(
            f"[GR00T-ATM-PI05] alpha JSON not found at {alpha_path!r}; ATM/OHB skipped.",
            flush=True,
        )
        return

    with open(alpha_path, "r", encoding="utf-8") as f:
        alpha_data = json.load(f)

    hook_count = _ensure_pi05_atm_hooks(model, scope="pi05")
    if hook_count == 0:
        print("[GR00T-ATM-PI05] no GemmaAttention modules matched scope=pi05.", flush=True)
        return

    summary = _AlphaSummary()
    ohb_layers = 0
    ohb_fallback = float(os.environ.get(OHB_FALLBACK_ENV, "1.0"))

    for name, module in model.named_modules():
        if not _is_pi05_attention(name, module, scope="pi05"):
            continue
        entry = alpha_data.get(name)
        if not entry:
            # tolerate ``language_model.layers`` vs ``language_model.model.layers``
            entry = alpha_data.get(name.replace(".model.", ".", 1))
        alpha_values = beta_value = beta_perhead = None
        if entry:
            alpha_values = entry.get("all") or entry.get("alpha")
            beta_value = entry.get("beta")
            beta_perhead = entry.get("beta_perhead")

        if atm_enabled and alpha_values:
            t = torch.tensor(alpha_values, dtype=torch.float32)
            setattr(module, "_atm_alpha_all", t)
            summary.matched_layers += 1
            summary.total_heads += int(t.numel())

        if ohb_enabled:
            if beta_perhead is not None:
                setattr(module, "_ohb_beta_perhead", torch.tensor(beta_perhead, dtype=torch.float32))
                ohb_layers += 1
            elif beta_value is not None:
                setattr(module, "_ohb_beta_scalar", float(beta_value))
                ohb_layers += 1
            elif ohb_fallback != 1.0:
                setattr(module, "_ohb_beta_scalar", ohb_fallback)
                ohb_layers += 1

    if atm_enabled:
        if summary.matched_layers == 0:
            print(f"[GR00T-ATM-PI05] no layers matched alpha JSON {alpha_path}.", flush=True)
        else:
            print(
                f"[GR00T-ATM-PI05] ATM α applied to {summary.matched_layers} layers "
                f"({summary.total_heads} heads) from {alpha_path}",
                flush=True,
            )
    if ohb_enabled:
        print(f"[GR00T-ATM-PI05] OHB β applied to {ohb_layers} layers.", flush=True)


# ============================================================
# Calibration: monkey-patch attention forward to capture per-head stats
# ============================================================

def _build_capture_forward(
    attn_module: nn.Module,
    num_heads: int,
    head_dim: int,
    num_kv_groups: int,
    scaling: float,
):
    """Wrap attention forward to compute per-head logits std & output RMS.

    Replicates the GemmaAttention.forward body so we can measure logits AFTER
    RoPE but BEFORE softmax, which is the same point dit_atm.py measures.
    Only used in the calibration script — runtime never pays this cost.
    """
    from transformers.models.gemma.modeling_gemma import (
        apply_rotary_pos_emb,
        repeat_kv,
        eager_attention_forward,
        ALL_ATTENTION_FUNCTIONS,
    )

    def wrapped(hidden_states, position_embeddings=None, attention_mask=None,
                past_key_value=None, cache_position=None, use_cache=False, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, head_dim)

        # q_proj forward goes through any registered q-hook (no-op when α unset).
        q_out = attn_module.q_proj(hidden_states)
        k_out = attn_module.k_proj(hidden_states)
        v_out = attn_module.v_proj(hidden_states)

        query_states = q_out.view(hidden_shape).transpose(1, 2)
        key_states = k_out.view(hidden_shape).transpose(1, 2)
        value_states = v_out.view(hidden_shape).transpose(1, 2)

        cos = sin = None
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # --- capture per-head logits std (post-RoPE, pre-softmax) ---
        capture_cb = getattr(attn_module, "_pi05_atm_capture_callback", None)
        if capture_cb is not None:
            with torch.no_grad():
                k_rep = repeat_kv(key_states, num_kv_groups)
                logits = torch.matmul(
                    query_states.float(),
                    k_rep.float().transpose(-1, -2),
                ) * scaling
                if attention_mask is not None:
                    mask4 = attention_mask[:, :, :, : k_rep.shape[-2]]
                    valid = (mask4 >= -1e4).to(logits.dtype)
                else:
                    valid = torch.ones_like(logits)
                count = valid.sum(dim=(-1, -2)).clamp_min(1.0)
                mean = (logits * valid).sum(dim=(-1, -2)) / count
                mean = mean.unsqueeze(-1).unsqueeze(-1)
                var = ((logits - mean) ** 2 * valid).sum(dim=(-1, -2)) / count
                std = torch.sqrt(var.clamp_min(1e-12)).detach().mean(dim=0)  # (H,)
                capture_cb(attn_module, std)

        # cache handling (mirrors original)
        if past_key_value is not None:
            if use_cache:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(
                    key_states, value_states, attn_module.layer_idx, cache_kwargs
                )
            else:
                key_states = torch.cat(
                    [past_key_value[attn_module.layer_idx][0], key_states], dim=2
                )
                value_states = torch.cat(
                    [past_key_value[attn_module.layer_idx][1], value_states], dim=2
                )

        attention_interface = eager_attention_forward
        if attn_module.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_module.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            attn_module,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not attn_module.training else attn_module.attention_dropout,
            scaling=scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # --- capture per-head output RMS (pre-β, pre-o_proj) ---
        rms_cb = getattr(attn_module, "_pi05_ohb_capture_callback", None)
        if rms_cb is not None:
            with torch.no_grad():
                B = attn_output.shape[0]
                S = attn_output.shape[1] if attn_output.dim() >= 3 else 1
                t = attn_output.view(B, S, num_heads, head_dim).float()
                rms_ph = torch.sqrt(torch.mean(t ** 2, dim=(0, 1, 3)) + 1e-12)
                rms_cb(attn_module, rms_ph.detach())

        attn_output = attn_module.o_proj(attn_output)  # β applied via pre-hook if set
        return attn_output, attn_weights

    return wrapped


def install_pi05_capture(model: nn.Module, scope: str = "pi05") -> int:
    """Monkey-patch matching ``self_attn.forward`` to enable per-head capture.

    Idempotent. Also ensures the runtime α/β hooks are registered so α applied
    at the q_proj output is honored during capture-time forwards too (matters
    for calibrating quant policy where α=identity → no effect).
    """
    count = 0
    for name, module in model.named_modules():
        if not _is_pi05_attention(name, module, scope=scope):
            continue
        if getattr(module, _PI05_CAPTURE_PATCH_FLAG, False):
            continue
        if not getattr(module, _PI05_HOOK_PATCH_FLAG, False):
            num_heads, head_dim = _num_heads_and_dim(module)
            module.q_proj.register_forward_hook(_make_q_alpha_hook(module, num_heads, head_dim))
            module.o_proj.register_forward_pre_hook(_make_o_beta_prehook(module, num_heads, head_dim))
            setattr(module, _PI05_HOOK_PATCH_FLAG, True)

        num_heads, head_dim = _num_heads_and_dim(module)
        num_kv_groups = int(getattr(module, "num_key_value_groups", 1))
        scaling = float(getattr(module, "scaling", head_dim ** -0.5))
        orig_fwd = module.forward
        setattr(module, _PI05_ORIG_FORWARD_ATTR, orig_fwd)
        wrapped = _build_capture_forward(module, num_heads, head_dim, num_kv_groups, scaling)
        # bind as a real method so calls like self_attn(...) dispatch correctly
        module.forward = types.MethodType(lambda self, *args, _w=wrapped, **kw: _w(*args, **kw), module)
        setattr(module, _PI05_CAPTURE_PATCH_FLAG, True)
        setattr(module, "_pi05_atm_path", name)
        count += 1
    return count


def uninstall_pi05_capture(model: nn.Module) -> None:
    for _, module in model.named_modules():
        if not getattr(module, _PI05_CAPTURE_PATCH_FLAG, False):
            continue
        orig = getattr(module, _PI05_ORIG_FORWARD_ATTR, None)
        if orig is not None:
            module.forward = orig
            delattr(module, _PI05_ORIG_FORWARD_ATTR)
        delattr(module, _PI05_CAPTURE_PATCH_FLAG)


def register_pi05_atm_capture(
    model: nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "pi05",
) -> int:
    """Attach a per-layer logits-std capture. Returns count of wired layers."""
    install_pi05_capture(model, scope=scope)
    n = 0
    for name, module in model.named_modules():
        if _is_pi05_attention(name, module, scope=scope):
            setattr(
                module,
                "_pi05_atm_capture_callback",
                lambda attn, std, layer=name: callback(layer, std),
            )
            n += 1
    return n


def register_pi05_ohb_capture(
    model: nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "pi05",
) -> int:
    """Attach a per-layer output-RMS capture. Returns count of wired layers."""
    install_pi05_capture(model, scope=scope)
    n = 0
    for name, module in model.named_modules():
        if _is_pi05_attention(name, module, scope=scope):
            setattr(
                module,
                "_pi05_ohb_capture_callback",
                lambda attn, rms, layer=name: callback(layer, rms),
            )
            n += 1
    return n


def clear_pi05_atm_capture(model: nn.Module) -> None:
    for _, module in model.named_modules():
        for attr in ("_pi05_atm_capture_callback", "_pi05_ohb_capture_callback"):
            if hasattr(module, attr):
                delattr(module, attr)
