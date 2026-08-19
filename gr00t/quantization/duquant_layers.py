"""
GR00T DuQuant W4A8 Fake Quantization Layers

Adapted from OpenPI duquant implementation for GR00T model quantization.
Supports quantization of LLM (Eagle VLM) and DiT (action transformer) layers.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import numpy as np
from torch import nn

from .duquant_preprocess import (
    MSEActCalibrator,
    PackResult,
    PercentileCalibrator,
    apply_input_transform,
    apply_output_restore,
    apply_bias_row_rot,
    fake_quantize_sym,
    load_pack,
    pack_weight,
    qmax,
    sanitize_name,
    save_pack,
    transform_weight_for_forward,
)


_ACT_STATS_CACHE: Dict[str, Dict[str, Any]] = {}


def _load_act_stats_file(path: str) -> Dict[str, Any]:
    """Load offline per-layer activation statistics (q999 / act_max) once."""
    cache_key = str(Path(path).resolve())
    if cache_key in _ACT_STATS_CACHE:
        return _ACT_STATS_CACHE[cache_key]
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"act-stats file {path} did not contain a dict")
    _ACT_STATS_CACHE[cache_key] = data
    return data


def _resolve_per_layer_act_stats(
    layer_name: str, path: Optional[str]
) -> Optional[np.ndarray]:
    """Return per-input-channel q999(|X|) for ``layer_name`` from the stats file.

    Looks for key 'q999' first, then falls back to 'act_max' (SmoothQuant
    collector format) so existing SQ pt files double as a coarse act-stats
    source. Returns None when the file or layer entry is missing.
    """
    if not path:
        return None
    try:
        store = _load_act_stats_file(path)
    except Exception:
        return None
    rec = store.get(layer_name)
    if rec is None:
        return None
    if isinstance(rec, dict):
        for k in ("q999", "act_q999", "act_p999", "act_max"):
            if k in rec:
                v = rec[k]
                if isinstance(v, torch.Tensor):
                    return v.detach().to(torch.float32).cpu().numpy()
                return np.asarray(v, dtype=np.float32)
        return None
    if isinstance(rec, torch.Tensor):
        return rec.detach().to(torch.float32).cpu().numpy()
    return np.asarray(rec, dtype=np.float32)


@dataclass
class DuQuantConfig:
    """DuQuant configuration matching OpenPI parameters.

    NOTE: Default values are set to None and resolved in __post_init__ to ensure
    environment variables are read at instantiation time, not at module import time.
    """
    weight_bits: Optional[int] = None
    act_bits: Optional[int] = None
    block_size: Optional[int] = None
    lambda_smooth: Optional[float] = None
    enable_permute: Optional[bool] = None
    act_percentile: Optional[float] = None
    calib_batches: Optional[int] = None
    pack_dir: Optional[str] = None
    row_rot_mode: Optional[str] = None
    block_out_size: Optional[int] = None
    # A4-aware LLM DuQuant extensions (W4A4 experiments).
    # All default to current behavior so existing W4A8 packs stay byte-identical.
    perm_score: Optional[str] = None         # weight | act | act_weight
    rot_mode: Optional[str] = None           # svd | hadamard | svd_hadamard
    act_scale_mode: Optional[str] = None     # percentile | mse
    act_stats_path: Optional[str] = None     # offline q999(|X|) per layer (.pt)

    def __post_init__(self):
        """Read environment variables at instantiation time."""
        if self.weight_bits is None:
            self.weight_bits = int(os.environ.get("GR00T_DUQUANT_WBITS_DEFAULT", 4))
        if self.act_bits is None:
            self.act_bits = int(os.environ.get("GR00T_DUQUANT_ABITS", 8))
        if self.block_size is None:
            self.block_size = int(os.environ.get("GR00T_DUQUANT_BLOCK", 16))
        if self.lambda_smooth is None:
            self.lambda_smooth = float(os.environ.get("GR00T_DUQUANT_LS", 0.15))
        if self.enable_permute is None:
            self.enable_permute = os.environ.get("GR00T_DUQUANT_PERMUTE", "1") not in ("0", "false", "False")
        if self.act_percentile is None:
            self.act_percentile = float(os.environ.get("GR00T_DUQUANT_ACT_PCT", 99.9))
        if self.calib_batches is None:
            self.calib_batches = int(os.environ.get("GR00T_DUQUANT_CALIB_STEPS", 32))
        if self.pack_dir is None:
            self.pack_dir = os.environ.get("GR00T_DUQUANT_PACKDIR", None)
        if self.row_rot_mode is None:
            self.row_rot_mode = os.environ.get("GR00T_DUQUANT_ROW_ROT", "restore")
        if self.block_out_size is None:
            self.block_out_size = int(os.environ.get("GR00T_DUQUANT_BLOCK_OUT", os.environ.get("GR00T_DUQUANT_BLOCK", 16)))
        # A4-aware extensions
        if self.perm_score is None:
            self.perm_score = os.environ.get("GR00T_DUQUANT_PERM_SCORE", "weight").lower()
        if self.rot_mode is None:
            self.rot_mode = os.environ.get("GR00T_DUQUANT_ROT_MODE", "svd").lower()
        if self.act_scale_mode is None:
            self.act_scale_mode = os.environ.get("GR00T_DUQUANT_ACT_SCALE_MODE", "percentile").lower()
        if self.act_stats_path is None:
            self.act_stats_path = os.environ.get("GR00T_DUQUANT_ACT_STATS_PATH", None)
        if self.perm_score not in ("weight", "act", "act_weight"):
            raise ValueError(f"GR00T_DUQUANT_PERM_SCORE must be weight|act|act_weight, got {self.perm_score!r}")
        if self.rot_mode not in ("svd", "hadamard", "svd_hadamard"):
            raise ValueError(f"GR00T_DUQUANT_ROT_MODE must be svd|hadamard|svd_hadamard, got {self.rot_mode!r}")
        if self.act_scale_mode not in ("percentile", "mse"):
            raise ValueError(f"GR00T_DUQUANT_ACT_SCALE_MODE must be percentile|mse, got {self.act_scale_mode!r}")


def _parse_per_layer_wbits(env_val: Optional[str]) -> Dict[str, int]:
    """Parse per-layer weight bits from environment variable."""
    if not env_val:
        return {}
    result: Dict[str, int] = {}
    parts = [p.strip() for p in env_val.split(",") if p.strip()]
    for p in parts:
        if ":" not in p:
            continue
        k, v = p.split(":", 1)
        try:
            result[k.strip()] = int(v.strip())
        except ValueError:
            pass
    return result


class DuQuantLinear(nn.Module):
    """DuQuant quantized linear layer with W4A8 fake quantization."""

    def __init__(self, base: nn.Linear, name: str, cfg: DuQuantConfig, weight_bits: Optional[int] = None) -> None:
        super().__init__()
        self.name = name
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bias = nn.Parameter(base.bias.detach().clone()) if base.bias is not None else None
        # SmoothQuant absorption: pre-divide x by s, pre-multiply W column-wise by s
        # We modify the weight HERE before pack so DuQuant rotation is computed on the rescaled W.
        _w_for_pack = base.weight.detach().clone()
        self._sq_s_inv: Optional[torch.Tensor] = None
        sq_path = os.environ.get("GR00T_DUQUANT_SMOOTHQUANT_PATH", "")
        if sq_path and os.path.exists(sq_path):
            try:
                _sq_alpha = float(os.environ.get("GR00T_DUQUANT_SMOOTHQUANT_ALPHA", "0.5"))
                _sq_clip = float(os.environ.get("GR00T_DUQUANT_SMOOTHQUANT_CLIP", "1e3"))
                _sq_min = 1.0 / _sq_clip
                _sq_data = torch.load(sq_path, map_location="cpu", weights_only=False)
                _rec = _sq_data.get(name, None)
                if _rec is not None and "act_max" in _rec and "weight_max" in _rec:
                    a = _rec["act_max"].float().clamp(min=1e-5)
                    w_col = _rec["weight_max"].float().clamp(min=1e-5)
                    s = (a.pow(_sq_alpha) / w_col.pow(1.0 - _sq_alpha)).clamp(min=_sq_min, max=_sq_clip)
                    # scale weight columns (input dim) by s
                    s_dev = s.to(dtype=_w_for_pack.dtype)
                    _w_for_pack = _w_for_pack * s_dev[None, :]
                    self.register_buffer("_sq_s_inv", (1.0 / s_dev).clone(), persistent=False)
                    if os.environ.get("GR00T_DUQUANT_DEBUG", "0") not in ("0", "false", "False"):
                        print(f"[GR00T-SQ] {name} alpha={_sq_alpha} s_min={float(s.min()):.3e} s_max={float(s.max()):.3e} a_max={float(a.max()):.3f} w_max={float(w_col.max()):.3f}", flush=True)
            except Exception as _e:
                print(f"[GR00T-SQ][WARN] {name}: {_e}", flush=True)
        self.register_buffer("_weight", _w_for_pack)

        # Config
        self.cfg = cfg
        self.weight_bits = cfg.weight_bits if weight_bits is None else int(weight_bits)

        # Per-layer offline activation stats (q999 of |X| per input channel).
        # Used to drive A4-aware permutation (perm_score=act|act_weight).
        # Silent None when not configured; pack_weight then auto-falls-back
        # to weight-energy scoring.
        act_stats_np = _resolve_per_layer_act_stats(self.name, cfg.act_stats_path)
        self._act_stats_available = act_stats_np is not None

        def _build_pack() -> PackResult:
            return pack_weight(
                self._weight,
                block_size=cfg.block_size,
                block_out_size=cfg.block_out_size,
                enable_permute=cfg.enable_permute,
                lambda_smooth=cfg.lambda_smooth,
                perm_score=cfg.perm_score,
                rot_mode=cfg.rot_mode,
                act_stats=act_stats_np,
            )

        # Load or build pack — cooperative multi-shard via fcntl lock so
        # only one shard does the per-block SVDs while others wait for the
        # atomically-saved cache file. Falls back to direct compute when
        # no pack_dir is configured.
        if cfg.pack_dir:
            from .duquant_preprocess import build_or_load_pack
            pack = build_or_load_pack(self.name, cfg.pack_dir, _build_pack)
        else:
            pack = load_pack(self.name, cfg.pack_dir)
            if pack is None:
                pack = _build_pack()
                save_pack(self.name, pack, cfg.pack_dir)
        self.pack: PackResult = pack

        # Sanity: when loading an EXISTING cache, verify the cached pack
        # matches the requested perm_score/rot_mode. Old (pre-W4A4) packs are
        # missing these meta keys; we silently accept those as weight/svd.
        # A mismatch here means the user reused GR00T_DUQUANT_PACKDIR across
        # ablation cells — silently using the wrong pack would produce
        # plausible-but-invalid results.
        cached_perm = pack.meta.get("perm_score")
        cached_rot = pack.meta.get("rot_mode")
        if cached_perm is not None and cached_perm != cfg.perm_score:
            print(
                f"[GR00T-DUQUANT][WARN] {self.name}: cached pack perm_score={cached_perm!r} "
                f"differs from requested {cfg.perm_score!r}; use a fresh GR00T_DUQUANT_PACKDIR "
                f"for this ablation cell.",
                flush=True,
            )
        if cached_rot is not None and cached_rot != cfg.rot_mode:
            print(
                f"[GR00T-DUQUANT][WARN] {self.name}: cached pack rot_mode={cached_rot!r} "
                f"differs from requested {cfg.rot_mode!r}; use a fresh GR00T_DUQUANT_PACKDIR.",
                flush=True,
            )

        # Cache rotation matrices as torch tensors
        if pack.perm is not None:
            self.register_buffer("_perm_cache", torch.from_numpy(pack.perm).long())
        else:
            self._perm_cache = None

        # Cache input rotation matrices
        self._R_in_block_indices: List[int] = []
        if pack.R_in_blocks:
            for b, R in pack.R_in_blocks.items():
                buffer_name = f"_R_in_{b}"
                self.register_buffer(buffer_name, torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_in_block_indices.append(b)

        # Cache output rotation matrices
        self._R_out_block_indices: List[int] = []
        if pack.R_out_blocks:
            for b, R in pack.R_out_blocks.items():
                buffer_name = f"_R_out_{b}"
                self.register_buffer(buffer_name, torch.from_numpy(R).to(dtype=self._weight.dtype))
                self._R_out_block_indices.append(b)

        # Store metadata
        self._block_size = int(pack.meta.get("block_size", 16))
        self._block_out_size = int(pack.meta.get("block_out_size", self._block_size))

        # Pre-build a single stacked rotation buffer per side. The hot path of every
        # forward used to do `torch.stack([R_in_cache[b] for b in range(n_blocks)])`,
        # which dominated CPU time when there are O(1k) Linear layers per step.
        # Building it once here removes that overhead entirely.
        # PyTorch >= 2.5 rejects register_buffer if the name is already a regular
        # attribute, so we register the buffer up-front (with None) before the
        # conditional branch.
        self.register_buffer("_R_in_stack", None, persistent=False)
        if pack.R_in_blocks and self.in_features % self._block_size == 0:
            n_in_blocks = self.in_features // self._block_size
            if all(b in pack.R_in_blocks for b in range(n_in_blocks)) and all(
                pack.R_in_blocks[b].shape == (self._block_size, self._block_size)
                for b in range(n_in_blocks)
            ):
                stack_in = torch.stack(
                    [torch.from_numpy(pack.R_in_blocks[b]) for b in range(n_in_blocks)],
                    dim=0,
                ).to(dtype=self._weight.dtype).contiguous()
                self._R_in_stack = stack_in

        self.register_buffer("_R_out_stack", None, persistent=False)
        if pack.R_out_blocks and self.out_features % self._block_out_size == 0:
            n_out_blocks = self.out_features // self._block_out_size
            if all(b in pack.R_out_blocks for b in range(n_out_blocks)) and all(
                pack.R_out_blocks[b].shape == (self._block_out_size, self._block_out_size)
                for b in range(n_out_blocks)
            ):
                stack_out = torch.stack(
                    [torch.from_numpy(pack.R_out_blocks[b]) for b in range(n_out_blocks)],
                    dim=0,
                ).to(dtype=self._weight.dtype).contiguous()
                self._R_out_stack = stack_out

        # Calibrator for activation. With act_scale_mode=mse we use a
        # reservoir-buffer calibrator that picks the per-channel scale
        # minimizing fake-quant MSE — meaningfully better than the
        # percentile/qmax heuristic at A4 where qmax=7 leaves little slack.
        # At A8 we keep percentile by default (default mode).
        self._use_mse_act_scale = (
            cfg.act_scale_mode == "mse" and self.cfg.act_bits > 0
        )
        if self.cfg.act_bits <= 0:
            self.calibrator = None
        elif self._use_mse_act_scale:
            self.calibrator = MSEActCalibrator(
                bits=int(self.cfg.act_bits),
                max_batches=int(cfg.calib_batches),
                percentile=float(cfg.act_percentile),
            )
        else:
            self.calibrator = PercentileCalibrator(
                percentile=cfg.act_percentile, max_batches=cfg.calib_batches
            )
        self.register_buffer("_act_scale", None)
        self._act_scale_initialized = False
        # Pre-transform activation stats (collected once during calibration,
        # used only for diagnostics — does not affect runtime).
        self._diag_pre_q999_mean: float = 0.0
        self._diag_pre_q999_max: float = 0.0
        self._diag_logged: bool = False

        # Cache transformed weight
        self._cached_weight_key: Optional[Tuple[Any, ...]] = None
        self.register_buffer("_W_t", torch.zeros_like(self._weight))
        self.register_buffer("_w_scales", torch.ones(self.out_features, dtype=self._weight.dtype))

        # Pre-cache quantized weights
        self._precache_weight = os.environ.get("GR00T_DUQUANT_PRECACHE_WEIGHTS", "1") not in (
            "0", "false", "False",
        )
        if self._precache_weight:
            self.register_buffer("_W_t_quantized", torch.zeros_like(self._weight))
        else:
            self._W_t_quantized = None
        self._weight_quantized_cached = False

        self._bias_rot: Optional[torch.Tensor] = None
        self._debug_enabled = os.environ.get("GR00T_DUQUANT_DEBUG", "0") not in ("0", "false", "False")
        self._debug_forward_logged = False
        # Resolve hot-path env flags ONCE at construction. Reading os.environ on every
        # forward is itself a measurable cost when called O(1k * denoising_steps) times.
        self._layer_stats_enabled = os.environ.get("GR00T_DEBUG_LAYER_STATS", "0") not in (
            "0", "false", "False",
        )
        try:
            self._layer_stats_every = int(os.environ.get("GR00T_DEBUG_LAYER_STATS_EVERY", "200"))
        except ValueError:
            self._layer_stats_every = 200
    def _get_R_in_cache(self) -> Dict[int, torch.Tensor]:
        """Get R_in rotation matrices on the correct device."""
        if not hasattr(self, '_R_in_cache_dict'):
            self._R_in_cache_dict = {}
        for b in self._R_in_block_indices:
            self._R_in_cache_dict[b] = getattr(self, f"_R_in_{b}")
        return self._R_in_cache_dict

    def _get_R_out_cache(self) -> Dict[int, torch.Tensor]:
        """Get R_out rotation matrices on the correct device."""
        if not hasattr(self, '_R_out_cache_dict'):
            self._R_out_cache_dict = {}
        for b in self._R_out_block_indices:
            self._R_out_cache_dict[b] = getattr(self, f"_R_out_{b}")
        return self._R_out_cache_dict

    def _quantize_weight(self, W_t: torch.Tensor, scales: torch.Tensor, apply_row_rot: bool, *, label: str) -> torch.Tensor:
        if self.weight_bits <= 0:
            return W_t
        return fake_quantize_sym(W_t, scales[:, None], self.weight_bits, label=label)

    @property
    def weight(self) -> torch.Tensor:
        """Expose packed weight buffer for compatibility."""
        return self._weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        with torch.no_grad():
            self._weight.copy_(value)

    def _maybe_update_weight_cache(self) -> None:
        apply_row = (self.cfg.row_rot_mode != "0")
        key = (str(self._weight.device), self._weight.dtype, int(self.weight_bits), int(apply_row))
        if self._cached_weight_key == key:
            return

        from .duquant_preprocess import transform_weight_for_forward_optimized

        W_t, scales = transform_weight_for_forward_optimized(
            self._weight,
            self.pack,
            weight_bits=self.weight_bits,
            apply_row_rot=apply_row,
            perm_cache=self._perm_cache,
            R_in_cache=self._get_R_in_cache(),
            R_out_cache=self._get_R_out_cache(),
            block_size=self._block_size,
            block_out_size=self._block_out_size,
        )
        self._W_t.copy_(W_t)
        self._w_scales.copy_(scales)

        # Pre-quantize weights if enabled
        if self._precache_weight and self.weight_bits > 0:
            with torch.no_grad():
                self._W_t_quantized.copy_(self._quantize_weight(W_t, scales, apply_row, label="weight_prequant"))
            self._weight_quantized_cached = True
        else:
            self._weight_quantized_cached = False

        self._cached_weight_key = key
        if self.bias is not None:
            if self.cfg.row_rot_mode == "propagate" and self.pack.R_out_blocks is not None:
                with torch.no_grad():
                    from .duquant_preprocess import apply_bias_row_rot_optimized
                    self._bias_rot = apply_bias_row_rot_optimized(
                        self.bias.detach(), self.pack, self._get_R_out_cache(), self._block_out_size
                    )
            else:
                self._bias_rot = None
        if self._debug_enabled:
            import logging
            logging.info(
                f"[GR00T-DUQUANT][CACHE] {self.name} device={self._weight.device} dtype={self._weight.dtype} "
                f"Wbits={self.weight_bits} Abits={self.cfg.act_bits} block_in={self.cfg.block_size} "
                f"permute={self.pack.perm is not None} row_rot={self.cfg.row_rot_mode}"
            )
            if self._weight_quantized_cached:
                logging.info(f"[GR00T-DUQUANT][CACHE] {self.name} pre-quantized weights cached")

    def _dump_act_stats(self, p_vec: torch.Tensor) -> None:
        """Dump per-channel A8 percentile vector when GR00T_ACT_STATS_DIR is set."""
        out_dir = os.environ.get("GR00T_ACT_STATS_DIR")
        if not out_dir:
            return
        try:
            import json
            from pathlib import Path
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            safe_name = self.name.replace("/", "_")
            (d / f"{safe_name}.json").write_text(json.dumps({
                "layer": self.name,
                "method": "duquant",
                "act_bits": int(self.cfg.act_bits),
                "percentile": float(self.cfg.act_percentile),
                "calib_batches": int(self.cfg.calib_batches),
                "n_channels": int(p_vec.numel()),
                "p_vec": p_vec.detach().cpu().tolist(),
            }))
        except Exception:
            pass

    def _log_diagnostics(self, scale: torch.Tensor, p_vec: torch.Tensor) -> None:
        """One-shot post-calibration log per layer (A4 ablation diagnostics)."""
        if self._diag_logged:
            return
        self._diag_logged = True
        try:
            max_q = qmax(self.cfg.act_bits)
            mse_diag = getattr(self.calibrator, "diagnostics", None) if self.calibrator else None
            extra = ""
            if mse_diag:
                extra = (
                    f" mse_act={mse_diag.get('mse_act', 0.0):.3e}"
                    f" rel_mse_act={mse_diag.get('rel_mse_act', 0.0):.3e}"
                    f" clip_frac={mse_diag.get('clip_frac', 0.0):.3e}"
                )
            print(
                f"[GR00T-DUQUANT][DIAG] {self.name} "
                f"perm_score={self.cfg.perm_score}(used={str(self.pack.meta.get('perm_score_used', 'weight'))}) "
                f"rot_mode={self.cfg.rot_mode} act_scale_mode={self.cfg.act_scale_mode} "
                f"a_bits={self.cfg.act_bits} w_bits={self.weight_bits} block={self._block_size} "
                f"pre_q999_mean={self._diag_pre_q999_mean:.3f} "
                f"post_q999_mean={float(p_vec.mean()):.3f} "
                f"post_q999_max={float(p_vec.max()):.3f} "
                f"scale_mean={float(scale.mean()):.3e} "
                f"scale_min={float(scale.min()):.3e} "
                f"scale_max={float(scale.max()):.3e}"
                f" act_stats={int(self._act_stats_available)}"
                f"{extra}",
                flush=True,
            )
        except Exception:
            pass

    def _maybe_collect_pre_transform_stats(self, x_raw: torch.Tensor) -> None:
        """Record per-channel q999 of |X| BEFORE permutation/rotation, once.

        Diagnostic-only: lets us compare ``mean(q999) before`` vs ``after`` the
        DuQuant transform to verify Hadamard / SVD-Hadamard actually flattens
        the activation outlier distribution.
        """
        if self._diag_logged or self._diag_pre_q999_mean > 0.0:
            return
        try:
            with torch.no_grad():
                xa = x_raw.detach().to(torch.float32).abs()
                C = xa.shape[-1]
                x2d = xa.reshape(-1, C)
                if x2d.shape[0] < 4:
                    return
                p_pre = torch.quantile(x2d, self.cfg.act_percentile / 100.0, dim=0)
                self._diag_pre_q999_mean = float(p_pre.mean())
                self._diag_pre_q999_max = float(p_pre.max())
        except Exception:
            pass

    def _get_act_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.act_bits <= 0:
            return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)

        # Fast path: calibrator has finalized; reuse cached scale.
        if self._act_scale_initialized:
            return self._act_scale

        with torch.no_grad():
            # Always observe until the calibrator fills. (Pre-2026-05-12 the
            # cold-start branch below also set _act_scale_initialized=True,
            # which silently truncated calibration to 1 batch — fatal for
            # MSEActCalibrator, harmless for PercentileCalibrator since its
            # 1-batch finalize ≈ cold-start q999.)
            if self.calibrator is not None and not self.calibrator.is_full():
                self.calibrator.observe(x)
                if self.calibrator.is_full():
                    if isinstance(self.calibrator, MSEActCalibrator):
                        # MSE-optimal per-channel scale: already in scale units.
                        scale_vec = self.calibrator.finalize()
                        scale = torch.clamp(scale_vec, min=1e-6)
                        scale = scale.to(dtype=x.dtype, device=x.device).clone()
                        # Pseudo p_vec = scale * qmax (implied saturation) so
                        # the diagnostic log stays comparable to percentile.
                        p_vec = scale.detach().to(torch.float32).cpu() * qmax(self.cfg.act_bits)
                    else:
                        p_vec = self.calibrator.finalize()
                        max_q = qmax(self.cfg.act_bits)
                        scale = torch.clamp(p_vec / max_q, min=1e-6)
                        scale = scale.to(dtype=x.dtype, device=x.device).clone()
                    if self._act_scale is None:
                        self._act_scale = scale
                    else:
                        self._act_scale.copy_(scale)
                    self._act_scale_initialized = True
                    self._dump_act_stats(p_vec)
                    self._log_diagnostics(scale, p_vec)
                    return self._act_scale

            # Calibrator not yet full — return a fresh cold-start scale (NOT
            # cached, NOT marking initialized) so subsequent forwards continue
            # to observe through the calibrator until it fills.
            x_abs = torch.abs(x.detach().to(torch.float32))
            C = x_abs.shape[-1]
            x2d = x_abs.reshape(-1, C)
            p_vec_cs = torch.quantile(x2d, self.cfg.act_percentile / 100.0, dim=0)
            max_q = qmax(self.cfg.act_bits)
            scale_cs = torch.clamp(p_vec_cs / max_q, min=1e-6)
            return scale_cs.to(dtype=x.dtype, device=x.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SmoothQuant: pre-divide x by per-channel s (broadcast on input dim)
        if self._sq_s_inv is not None:
            x = x * self._sq_s_inv.to(dtype=x.dtype, device=x.device)

        # Diagnostic: one-shot pre-transform activation range (driven by env
        # flag only, since reading q999 once-per-layer is non-trivial).
        if (self.calibrator is not None and not self.calibrator.is_full()
                and self._diag_pre_q999_mean == 0.0):
            self._maybe_collect_pre_transform_stats(x)

        # Per-block input rotation. Fast path uses the prebuilt stacked tensor so we
        # avoid Python list-comp / torch.stack overhead per layer per step.
        if self._perm_cache is not None:
            x = x.index_select(dim=-1, index=self._perm_cache)
        if self._R_in_stack is not None:
            orig_shape = x.shape
            n_blocks = self._R_in_stack.shape[0]
            x_t = torch.einsum(
                "rnb,nbc->rnc",
                x.reshape(-1, n_blocks, self._block_size),
                self._R_in_stack,
            ).reshape(orig_shape)
        elif self._R_in_block_indices:
            from .duquant_preprocess import apply_input_transform_optimized
            x_t = apply_input_transform_optimized(
                x, self.pack, None, self._get_R_in_cache(), self._block_size
            )
        else:
            x_t = x

        # Fake-quantize activations if enabled
        if self.cfg.act_bits > 0:
            s_a = self._get_act_scale(x_t)
            x_t = fake_quantize_sym(x_t, s_a, self.cfg.act_bits, label="activation_forward")

        # Transform and fake-quantize weights
        self._maybe_update_weight_cache()

        # Use pre-quantized weights
        if self._weight_quantized_cached:
            y_lin = torch.nn.functional.linear(x_t, self._W_t_quantized, None)
        elif self.weight_bits > 0:
            y_lin = torch.nn.functional.linear(
                x_t,
                self._quantize_weight(
                    self._W_t,
                    self._w_scales,
                    self.cfg.row_rot_mode != "0",
                    label="weight_fallback",
                ),
                None
            )
        else:
            y_lin = torch.nn.functional.linear(x_t, self._W_t, None)

        # Apply row restore if requested
        if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
            if self._R_out_stack is not None:
                orig_shape = y_lin.shape
                n_out = self._R_out_stack.shape[0]
                y_lin = torch.einsum(
                    "rnb,nbc->rnc",
                    y_lin.reshape(-1, n_out, self._block_out_size),
                    self._R_out_stack,
                ).reshape(orig_shape)
            else:
                from .duquant_preprocess import apply_output_restore_optimized
                y_lin = apply_output_restore_optimized(
                    y_lin, self.pack, self._get_R_out_cache(), self._block_out_size
                )
            if self.bias is not None:
                y_lin = y_lin + self.bias
        else:
            if self.bias is not None:
                bias_to_add = (
                    self._bias_rot
                    if self.cfg.row_rot_mode == "propagate" and self._bias_rot is not None
                    else self.bias
                )
                y_lin = y_lin + bias_to_add
        if self._debug_enabled and not self._debug_forward_logged:
            import logging
            logging.info(
                f"[GR00T-DUQUANT][FORWARD] {self.name} input={tuple(x.shape)} output={tuple(y_lin.shape)} "
                f"weight_bits={self.weight_bits} act_bits={self.cfg.act_bits}"
            )
            self._debug_forward_logged = True
        # Optional per-layer stat logging (env-gated, resolved once at init)
        if self._layer_stats_enabled:
            try:
                _every = self._layer_stats_every
                _cnt = getattr(self, "_dbg_call_cnt", 0) + 1
                self._dbg_call_cnt = _cnt
                if _cnt == 1 or (_cnt % _every) == 0:
                    with torch.no_grad():
                        xa = x.detach().abs()
                        ya = y_lin.detach()
                        nan_cnt = int(torch.isnan(ya).sum().item())
                        inf_cnt = int(torch.isinf(ya).sum().item())
                        x_max = float(xa.max().item()) if xa.numel() else 0.0
                        y_amax = float(ya.abs().max().item()) if ya.numel() else 0.0
                        y_norm = float(ya.float().norm().item()) if ya.numel() else 0.0
                        # crude clip rate: fraction of |x| beyond p99.9 of own batch (proxy)
                        if xa.numel() > 1024:
                            thr = float(torch.quantile(xa.flatten()[: min(xa.numel(), 1<<20)].float(), 0.999).item())
                            clip_rate = float((xa > thr).float().mean().item())
                        else:
                            thr, clip_rate = 0.0, 0.0
                    print(
                        f"[LAYER] call#{_cnt} {self.name} "
                        f"xmax={x_max:.3f} ymax={y_amax:.3f} ynorm={y_norm:.2f} "
                        f"nan={nan_cnt} inf={inf_cnt} clip_rate@p99.9={clip_rate:.3e}",
                        flush=True,
                    )
            except Exception as _e:
                pass
        return y_lin


def _get_parent_module_and_attr(model: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def select_targets(
    model: nn.Module,
    *,
    include_regex: str = r".*(q_proj|k_proj|v_proj|out_proj|fc1|fc2|up_proj|down_proj|gate_proj).*",
    exclude_regex: str = r"(?:^|\.)(norm|ln|layernorm|emb)(?:\.|$)",
    scope_prefix: Optional[str] = None,
    whitelist: Optional[Iterable[str]] = None,
    blacklist: Optional[Iterable[str]] = None,
) -> List[Tuple[str, nn.Linear]]:
    """Select linear layers to quantize based on regex patterns."""
    inc = re.compile(include_regex)
    exc = re.compile(exclude_regex)
    wl = set(whitelist or [])
    bl = set(blacklist or [])
    results: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if scope_prefix is not None and not name.startswith(scope_prefix):
            continue
        if name in bl:
            continue
        if wl and name not in wl:
            continue
        if not wl and (not inc.search(name) or exc.search(name)):
            continue
        results.append((name, mod))
    return results


def wrap_duquant(
    model: nn.Module,
    layer_names: Iterable[str],
    cfg: DuQuantConfig,
    per_layer_wbits: Optional[Dict[str, int]] = None,
    dry_run: bool = False,
) -> None:
    """Wrap selected layers with DuQuant quantization."""
    per_layer_wbits = per_layer_wbits or {}
    replaced = 0
    listed = 0
    for name in layer_names:
        # Skip action head by default unless explicitly requested
        if os.environ.get("GR00T_DUQUANT_INCLUDE_ACTION_HEAD", "0") in ("0", "false", "False"):
            is_action_head = "action_head" in name and not name.startswith("action_head.model.")
            if (
                name.endswith("action_out_proj")
                or ".action_out_proj" in name
                or is_action_head
            ):
                continue
        parent, attr = _get_parent_module_and_attr(model, name)
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        wbits = per_layer_wbits.get(name, cfg.weight_bits)
        if dry_run:
            msg = (
                f"[GR00T-DUQUANT][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features}) "
                f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} "
                f"block_in={cfg.block_size} block_out={cfg.block_out_size} row_rot={cfg.row_rot_mode} "
                f"perm_score={cfg.perm_score} rot_mode={cfg.rot_mode} act_scale_mode={cfg.act_scale_mode}"
            )
            print(msg)
            listed += 1
            continue
        dq = DuQuantLinear(mod, name=name, cfg=cfg, weight_bits=wbits)
        setattr(parent, attr, dq)
        # Use actual block sizes from pack (not cfg defaults)
        actual_block_in = dq._block_size
        actual_block_out = dq._block_out_size
        print(
            f"[GR00T-DUQUANT][REPLACED] {name}: Linear({mod.in_features}->{mod.out_features}) -> DuQuantLinear "
            f"W{wbits} A{cfg.act_bits} perm={cfg.enable_permute} block_in={actual_block_in} "
            f"block_out={actual_block_out} row_rot={cfg.row_rot_mode} "
            f"perm_score={cfg.perm_score}(stats={int(dq._act_stats_available)}) "
            f"rot_mode={cfg.rot_mode} act_scale_mode={cfg.act_scale_mode}"
        )
        replaced += 1
    if dry_run:
        print(f"[GR00T-DUQUANT] Dry-run total layers listed: {listed}")
    else:
        print(f"[GR00T-DUQUANT] Total layers replaced: {replaced}")


def enable_duquant_if_configured(model: nn.Module) -> None:
    """
    Entry point to enable DuQuant based on environment variables.

    Activation conditions:
    - If GR00T_DUQUANT_DRYRUN is set => dry-run listing only
    - Or if any GR00T_DUQUANT_* variable (other than PACKDIR) is set => perform replacement
    - Otherwise do nothing
    """
    env = os.environ
    keys = [k for k in env.keys() if k.startswith("GR00T_DUQUANT_")]
    activate = any(k not in ("GR00T_DUQUANT_PACKDIR",) for k in keys)
    if not activate:
        return

    # Scope defaults to empty (search entire model)
    scope = env.get("GR00T_DUQUANT_SCOPE", "")
    whitelist = env.get("GR00T_DUQUANT_LAYERS")
    whitelist_list = [x.strip() for x in whitelist.split(",") if x.strip()] if whitelist else None

    # Default: quantize LLM + DiT MLP layers (matching OpenPI pattern)
    # Include LLM attention+MLP and DiT MLP projections
    inc = env.get(
        "GR00T_DUQUANT_INCLUDE",
        (
            r".*(?:"
            r"backbone\.eagle_model\.language_model\..*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
            r"|"
            r"action_head\.model\..*(?:attn1\.to_(?:q|k|v)|attn1\.to_out\.0|ff\.net\.(?:0\.proj|2))"
            r").*"
        ),
    )
    # Exclude vision encoder, embeddings, auxiliary projectors
    exc = env.get(
        "GR00T_DUQUANT_EXCLUDE",
        (
            r"(?:^|\.)"
            r"(?:vision_model|vision|radio|norm|ln|layernorm|embed|lm_head|timestep_encoder|state_encoder|action_encoder|action_decoder|future_tokens|vl_self_attention)"
            r"(?:\.|$)"
        ),
    )

    per_layer_wbits = _parse_per_layer_wbits(env.get("GR00T_DUQUANT_WBITS"))
    dry_run = env.get("GR00T_DUQUANT_DRYRUN", "0") not in ("0", "false", "False")

    cfg = DuQuantConfig()

    targets = select_targets(
        model,
        include_regex=inc,
        exclude_regex=exc,
        scope_prefix=scope if scope else None,
        whitelist=whitelist_list,
        blacklist=None,
    )
    layer_names = [n for n, _ in targets]
    print(f"[GR00T-DUQUANT] SCOPE filter: '{scope}'")
    print(f"[GR00T-DUQUANT] Matched Linear layers: {len(layer_names)}")

    if len(layer_names) == 0 and scope:
        # Debug: print some layer names to help diagnose
        all_linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        print(f"[GR00T-DUQUANT] DEBUG: Total Linear layers in model: {len(all_linears)}")
        print(f"[GR00T-DUQUANT] DEBUG: First 10 Linear layer names:")
        for name, _ in all_linears[:10]:
            print(f"[GR00T-DUQUANT] DEBUG:   {name}")
        if scope:
            matching_prefix = [n for n, _ in all_linears if n.startswith(scope.rstrip('.'))]
            print(f"[GR00T-DUQUANT] DEBUG: Layers matching prefix '{scope.rstrip('.')}': {len(matching_prefix)}")
            if matching_prefix:
                for name in matching_prefix[:5]:
                    print(f"[GR00T-DUQUANT] DEBUG:   {name}")

    if dry_run:
        wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run=True)
        return
    wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run=False)
