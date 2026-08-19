import json
import math
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch

try:
    from diffusers.models.attention import Attention
    from diffusers.models.attention_processor import AttnProcessor2_0
except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError(
        "diffusers is required for GR00T ATM support. "
        "Please ensure the gr00t environment is activated."
    ) from exc

ATM_ENABLE_ENV = "GR00T_ATM_ENABLE"
ATM_ALPHA_ENV = "GR00T_ATM_ALPHA_PATH"
ATM_SCOPE_ENV = "GR00T_ATM_SCOPE"

OHB_ENABLE_ENV = "GR00T_OHB_ENABLE"
OHB_SCOPE_ENV = "GR00T_OHB_SCOPE"
OHB_ONLY_DIT_ENV = "GR00T_OHB_ONLY_DIT"
OHB_FALLBACK_ENV = "GR00T_OHB_FALLBACK"

_ATM_PATCH_FLAG = "_gr00t_atm_processor_patched"


def _is_dit_attention(name: str, module: Attention, scope: str = "dit") -> bool:
    if not isinstance(module, Attention):
        return False
    if scope == "dit":
        return "action_head.model.transformer_blocks" in name
    if scope == "all":
        return True
    return scope in name


class _ATMProcessor(AttnProcessor2_0):
    """Attention processor with per-head statistics capture and optional scaling."""

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Capture logits statistics before ATM scaling
        capture_cb = getattr(attn, "_atm_capture_callback", None)
        logits_capture_cb = getattr(attn, "_atm_logits_capture_callback", None)
        if capture_cb is not None or logits_capture_cb is not None:
            std, logits_tensor = _compute_logits_std(
                query, key, attention_mask, head_dim, return_logits=True
            )
            if capture_cb is not None:
                capture_cb(attn, std)
            if logits_capture_cb is not None:
                logits_capture_cb(attn, logits_tensor)

        # Apply ATM scaling if provided
        alpha = getattr(attn, "_atm_alpha_all", None)
        if alpha is not None:
            alpha = alpha.to(dtype=query.dtype, device=query.device).view(1, -1, 1, 1)
            query = query * alpha

        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Capture per-head RMS for per-head OHB calibration (BEFORE reshape)
        # hidden_states shape here: (batch, heads, seq, head_dim)
        ohb_perhead_capture_cb = getattr(attn, "_atm_ohb_perhead_capture_callback", None)
        if ohb_perhead_capture_cb is not None:
            ohb_perhead_capture_cb(attn, _compute_rms_per_head(hidden_states))

        # Apply per-head OHB beta scaling (BEFORE reshape)
        beta_perhead = getattr(attn, "_ohb_beta_perhead", None)
        if beta_perhead is not None:
            # beta_perhead shape: (heads,), expand to (1, heads, 1, 1)
            beta_perhead = beta_perhead.to(dtype=hidden_states.dtype, device=hidden_states.device)
            beta_perhead = beta_perhead.view(1, -1, 1, 1)
            hidden_states = hidden_states * beta_perhead

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # Capture/output scale for OHB calibration
        ohb_capture_cb = getattr(attn, "_atm_ohb_capture_callback", None)
        if ohb_capture_cb is not None:
            ohb_capture_cb(attn, _compute_rms(hidden_states))

        beta = getattr(attn, "_ohb_beta_scalar", None)
        if beta is not None and beta != 1.0:
            hidden_states = hidden_states * beta

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def _compute_logits_std(
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    head_dim: int,
    *,
    return_logits: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    dtype = torch.float32
    scale = 1.0 / math.sqrt(max(float(head_dim), 1.0))
    logits = torch.matmul(query.to(dtype), key.to(dtype).transpose(-1, -2)) * scale

    if attention_mask is not None:
        # attention mask is additive, with large negative entries for masked positions
        valid = attention_mask >= -1e4
    else:
        valid = torch.ones_like(logits, dtype=torch.bool)

    valid = valid.to(dtype)
    count = valid.sum(dim=(-1, -2)).clamp_min(1.0)
    mean = (logits * valid).sum(dim=(-1, -2)) / count
    mean = mean.unsqueeze(-1).unsqueeze(-1)
    var = ((logits - mean) ** 2 * valid).sum(dim=(-1, -2)) / count
    std = torch.sqrt(var.clamp_min(1e-12))
    std = std.detach()
    if return_logits:
        return std, logits.detach()
    return std


def _compute_rms(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(tensor.detach().to(torch.float32) ** 2) + 1e-12)


def _compute_rms_per_head(tensor: torch.Tensor) -> torch.Tensor:
    """Compute RMS per head for per-head OHB.

    Args:
        tensor: Shape (batch, heads, seq, head_dim) - attention output BEFORE reshape.

    Returns:
        Shape (heads,) - RMS value per head averaged over batch, seq, head_dim.
    """
    # tensor shape: (batch, heads, seq, head_dim)
    t = tensor.detach().to(torch.float32)
    # Compute RMS per head: sqrt(mean(x^2)) over (batch, seq, head_dim)
    rms_per_head = torch.sqrt(torch.mean(t ** 2, dim=(0, 2, 3)) + 1e-12)  # (heads,)
    return rms_per_head


def ensure_dit_attention_patch(model: torch.nn.Module, scope: str = "dit") -> None:
    """Replace attention processors for DiT attention layers with ATM-enabled processor."""

    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            if getattr(module, _ATM_PATCH_FLAG, False):
                continue
            module.set_processor(_ATMProcessor())
            setattr(module, _ATM_PATCH_FLAG, True)


def register_atm_capture(
    model: torch.nn.Module,
    callback: Callable[[Attention, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(module, "_atm_capture_callback", lambda attn, std, layer=name: callback(layer, std))
            setattr(module, "_atm_capture_name", name)


def register_atm_logits_capture(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_logits_capture_callback",
                lambda attn, logits, layer=name: callback(layer, logits),
            )
            setattr(module, "_atm_logits_capture_name", name)


def register_ohb_capture(
    model: torch.nn.Module,
    callback: Callable[[Attention, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(module, "_atm_ohb_capture_callback", lambda attn, rms, layer=name: callback(layer, rms))
            setattr(module, "_atm_ohb_capture_name", name)


def register_ohb_perhead_capture(
    model: torch.nn.Module,
    callback: Callable[[str, torch.Tensor], None],
    scope: str = "dit",
) -> None:
    """Register per-head OHB capture callback.

    The callback receives (layer_name, rms_per_head) where rms_per_head has shape (heads,).
    """
    for name, module in model.named_modules():
        if _is_dit_attention(name, module, scope=scope):
            setattr(
                module,
                "_atm_ohb_perhead_capture_callback",
                lambda attn, rms, layer=name: callback(layer, rms),
            )
            setattr(module, "_atm_ohb_perhead_capture_name", name)


def clear_atm_capture(model: torch.nn.Module) -> None:
    for _, module in model.named_modules():
        if isinstance(module, Attention):
            if hasattr(module, "_atm_capture_callback"):
                delattr(module, "_atm_capture_callback")
            if hasattr(module, "_atm_capture_name"):
                delattr(module, "_atm_capture_name")
            if hasattr(module, "_atm_logits_capture_callback"):
                delattr(module, "_atm_logits_capture_callback")
            if hasattr(module, "_atm_logits_capture_name"):
                delattr(module, "_atm_logits_capture_name")
            if hasattr(module, "_atm_ohb_capture_callback"):
                delattr(module, "_atm_ohb_capture_callback")
            if hasattr(module, "_atm_ohb_capture_name"):
                delattr(module, "_atm_ohb_capture_name")
            if hasattr(module, "_atm_ohb_perhead_capture_callback"):
                delattr(module, "_atm_ohb_perhead_capture_callback")
            if hasattr(module, "_atm_ohb_perhead_capture_name"):
                delattr(module, "_atm_ohb_perhead_capture_name")


@dataclass
class _AlphaSummary:
    matched_layers: int = 0
    total_heads: int = 0


def enable_dit_atm_if_configured(model: torch.nn.Module) -> None:
    atm_flag = os.environ.get(ATM_ENABLE_ENV, "0")
    ohb_flag = os.environ.get(OHB_ENABLE_ENV, "0")
    atm_enabled = atm_flag not in ("0", "false", "False", "")
    ohb_enabled = ohb_flag not in ("0", "false", "False", "")
    if not atm_enabled and not ohb_enabled:
        return

    alpha_path = os.environ.get(ATM_ALPHA_ENV)
    if not alpha_path:
        print("[GR00T-ATM] Scaling requested but GR00T_ATM_ALPHA_PATH not set; skipping.")
        return

    if not os.path.exists(alpha_path):
        print(f"[GR00T-ATM] Alpha JSON not found at {alpha_path}; skipping ATM.")
        return

    with open(alpha_path, "r", encoding="utf-8") as f:
        alpha_data = json.load(f)

    scope = os.environ.get(ATM_SCOPE_ENV, "dit")
    ohb_scope = os.environ.get(OHB_SCOPE_ENV, None)
    if ohb_scope is None:
        ohb_only_dit = os.environ.get(OHB_ONLY_DIT_ENV, "1") not in ("0", "false", "False")
        ohb_scope = "dit" if ohb_only_dit else scope
    summary = _AlphaSummary()
    ohb_layers = 0
    ohb_fallback = float(os.environ.get(OHB_FALLBACK_ENV, "1.0"))

    ensure_dit_attention_patch(model, scope=scope)

    for name, module in model.named_modules():
        if not _is_dit_attention(name, module, scope=scope):
            continue
        alpha_entry = alpha_data.get(name) or alpha_data.get(name.replace("model.", "model", 1))
        if not alpha_entry:
            beta_value = None
            alpha_values = None
        else:
            alpha_values = alpha_entry.get("all") or alpha_entry.get("alpha")
            beta_value = alpha_entry.get("beta")

        if atm_enabled and alpha_values:
            alpha_tensor = torch.tensor(alpha_values, dtype=torch.float32)
            setattr(module, "_atm_alpha_all", alpha_tensor)
            summary.matched_layers += 1
            summary.total_heads += len(alpha_values)

        if ohb_enabled and _is_dit_attention(name, module, scope=ohb_scope):
            # Check for per-head beta first
            beta_perhead_values = alpha_entry.get("beta_perhead") if alpha_entry else None
            if beta_perhead_values is not None:
                # Per-head OHB
                beta_tensor = torch.tensor(beta_perhead_values, dtype=torch.float32)
                setattr(module, "_ohb_beta_perhead", beta_tensor)
                ohb_layers += 1
            else:
                # Per-layer OHB (fallback)
                beta = float(beta_value) if beta_value is not None else ohb_fallback
                setattr(module, "_ohb_beta_scalar", beta)
                ohb_layers += 1

    if summary.matched_layers == 0 and atm_enabled:
        print(f"[GR00T-ATM] No attention layers matched alpha JSON ({alpha_path}).")
    elif atm_enabled:
        print(
            f"[GR00T-ATM] ATM enabled for {summary.matched_layers} layers "
            f"({summary.total_heads} heads) using {alpha_path}"
        )

    if ohb_enabled:
        if ohb_layers == 0:
            print(f"[GR00T-ATM] OHB requested but no layers found (scope={ohb_scope}); fallback beta={ohb_fallback}")
        else:
            print(f"[GR00T-ATM] OHB enabled for {ohb_layers} layers using {alpha_path}")
