"""SVD-spectrum + INT4-error diagnostics for GR00T LLM (Qwen3) vs DiT.

Drives a Gr00tPolicy through real LIBERO observations from the requested
task suite, hooks q_proj inputs in the LLM and attn1.to_q inputs in the DiT,
and computes:

 * raw singular-value spectra,
 * per-channel q999(|X|) sorted desc,
 * INT4 per-channel quant error after no-op / block-Hadamard / SVD-Hadamard
   rotations,
 * SVDQuant residual quant error at ranks 16/32.

Validation A (per-task drift):  run with the matching {ckpt, suite} pair
(object-ckpt + libero_object, long-ckpt + libero_10).  Compare SVDQ residual
errors across suites to verify the "object has cleaner low-rank outliers than
long" hypothesis.

Usage (under the awesome_quant_vla conda env which has both LIBERO simulator and torch):

    CUDA_VISIBLE_DEVICES=7 LIBERO_CONFIG_PATH=$LIBERO_CONFIG_PATH \\
    python \\
      -m tools.diagnose_outlier_topology_gr00t \\
      --checkpoint $CHECKPOINTS_ROOT/gr00t-n1.5-libero-object-posttrain \\
      --task-suite-name libero_object \\
      --num-samples 8 \\
      --output-tag object \\
      --output-dir ./experiment_results/svd_diagnostics_gr00t
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from tools.analyze_layerwise_quant_drift import (
    ensure_libero_runtime,
    load_libero_samples,
    _convert_libero_observation,  # noqa: F401  (kept for reference)
)
from gr00t.model.policy import Gr00tPolicy
from gr00t.experiment.data_config import load_data_config
from gr00t.data.embodiment_tags import EmbodimentTag


# ----------------------------- math helpers -----------------------------

def hadamard_matrix(n: int, device, dtype) -> torch.Tensor:
    assert n & (n - 1) == 0, f"Hadamard requires power-of-two dim, got {n}"
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def block_hadamard_rotation(d: int, block_size: int, device, dtype) -> torch.Tensor:
    assert d % block_size == 0, f"d={d} not divisible by block_size={block_size}"
    n_blocks = d // block_size
    H_blk = hadamard_matrix(block_size, device, dtype)
    R = torch.zeros((d, d), device=device, dtype=dtype)
    for b in range(n_blocks):
        s = b * block_size
        R[s:s + block_size, s:s + block_size] = H_blk
    return R


def svd_hadamard_rotation_from_weight(W: torch.Tensor, block_size: int) -> torch.Tensor:
    """Real A2-lite svd_hadamard rotation, faithful to
    gr00t.quantization.duquant_preprocess._rotation_for_block:

      R = block-diagonal of (U_b @ H_b)
      U_b = left singular vectors of W_block.T, computed per input-feature block
      H_b = block Hadamard

    Apply: X' = X @ R (along the in_features axis).  R is independent of X.
    """
    out_features, in_features = W.shape
    if in_features % block_size != 0:
        raise ValueError(f"in_features={in_features} not divisible by block_size={block_size}")
    n_blocks = in_features // block_size
    H_blk = hadamard_matrix(block_size, W.device, torch.float32)
    R = torch.zeros((in_features, in_features), device=W.device, dtype=torch.float32)
    Wf = W.detach().float()
    for b in range(n_blocks):
        s = b * block_size
        e = s + block_size
        W_block = Wf[:, s:e]  # [out, B]
        # SVD of W_block.T shape [B, out]; U has shape [B, B].
        U, _, _ = torch.linalg.svd(W_block.T, full_matrices=True)
        R[s:e, s:e] = U @ H_blk
    return R


def spectrum(X: torch.Tensor) -> np.ndarray:
    Xs = X
    if Xs.shape[0] > 8192:
        idx = torch.randperm(Xs.shape[0], device=Xs.device)[:8192]
        Xs = Xs[idx]
    S = torch.linalg.svdvals(Xs.float())
    return S.cpu().numpy()


def stable_rank(S: np.ndarray) -> float:
    return float((S ** 2).sum() / (S[0] ** 2))


def per_channel_range(X: torch.Tensor) -> np.ndarray:
    Xa = X.detach().float().abs()
    R = torch.quantile(Xa, 0.999, dim=0)
    return np.sort(R.cpu().numpy())[::-1]


def per_row_max_abs(X: torch.Tensor) -> np.ndarray:
    """For each row (token), the worst-channel |X|.  This is what drives the
    per-row INT4 activation scale in real W4A4 deployment."""
    Xa = X.detach().float().abs()
    return Xa.max(dim=1).values.cpu().numpy()


def _gini(vals: torch.Tensor) -> float:
    """Gini coefficient of a 1-D non-negative vector.  0 = perfectly uniform,
    1 = all mass in one entry.  Higher = more concentrated."""
    v = vals.detach().float().abs().flatten()
    v, _ = torch.sort(v)
    n = v.numel()
    if n == 0:
        return float("nan")
    s = v.sum().clamp_min(1e-12)
    # G = (2 * sum_i i * v_i) / (n * sum v) - (n+1)/n
    idx = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    g = (2.0 * (idx * v).sum() / (n * s)) - (n + 1) / n
    return float(g.item())


def outlier_topology_stats(X: torch.Tensor, sigma_thresh: float = 5.0) -> dict:
    """Quantify whether outliers are channel-localized or token-localized.

    For activation X [N tokens, d channels]:
      • channel energy E_c[j] = sum_i X[i,j]^2  — distribution across channels
      • token   energy E_t[i] = sum_j X[i,j]^2  — distribution across tokens
      • Gini coefficient of each tells us which axis the outlier mass concentrates on.
      • Additional: identify "extreme" entries (|X| > sigma_thresh * std), then
        attribute their energy to channels vs tokens to see which axis dominates
        the high-magnitude tail specifically.

    Returns:
      gini_chan  Gini of E_c (high = a few channels dominate energy)
      gini_tok   Gini of E_t (high = a few tokens   dominate energy)
      chan_tok_ratio = gini_chan / gini_tok   ( >1 → channel-localized,
                                                <1 → token-localized )
      extreme_chan_mass_top1 = fraction of extreme energy in top-1 channel
      extreme_tok_mass_top1  = fraction of extreme energy in top-1 token
      extreme_chan_mass_top5 / top16  — same for top-5 and top-16 axes
      extreme_tok_mass_top5  / top16  — same for top-5 and top-16 tokens
    """
    Xf = X.detach().float()
    # Per-channel and per-token L2 energy.
    E_chan = (Xf ** 2).sum(dim=0)  # [d]
    E_tok = (Xf ** 2).sum(dim=1)   # [N]
    gini_chan = _gini(E_chan)
    gini_tok = _gini(E_tok)

    # Extreme-element analysis.
    std = Xf.float().std().clamp_min(1e-8)
    extreme_mask = (Xf.abs() > sigma_thresh * std).float()  # [N, d]
    E_extreme = (Xf ** 2) * extreme_mask
    if E_extreme.sum() < 1e-12:
        # No extremes at sigma_thresh; loosen threshold.
        sigma_thresh = 3.0
        extreme_mask = (Xf.abs() > sigma_thresh * std).float()
        E_extreme = (Xf ** 2) * extreme_mask

    total_extreme = E_extreme.sum().clamp_min(1e-12)
    chan_extreme = E_extreme.sum(dim=0)
    tok_extreme = E_extreme.sum(dim=1)
    # Sorted descending mass fractions.
    chan_sorted, _ = torch.sort(chan_extreme, descending=True)
    tok_sorted, _ = torch.sort(tok_extreme, descending=True)

    def cum_frac(sorted_vec: torch.Tensor, k: int) -> float:
        k = min(k, sorted_vec.numel())
        if k == 0:
            return 0.0
        return float((sorted_vec[:k].sum() / total_extreme).item())

    return {
        "gini_chan": gini_chan,
        "gini_tok": gini_tok,
        "chan_tok_gini_ratio": gini_chan / gini_tok if gini_tok > 1e-9 else float("nan"),
        "extreme_chan_top1": cum_frac(chan_sorted, 1),
        "extreme_chan_top5": cum_frac(chan_sorted, 5),
        "extreme_chan_top16": cum_frac(chan_sorted, 16),
        "extreme_tok_top1": cum_frac(tok_sorted, 1),
        "extreme_tok_top5": cum_frac(tok_sorted, 5),
        "extreme_tok_top16": cum_frac(tok_sorted, 16),
        "sigma_thresh_used": float(sigma_thresh),
        "extreme_fraction": float((extreme_mask.sum() / extreme_mask.numel()).item()),
    }


def int4_per_channel_quant_error(X: torch.Tensor, denom_norm_sq: float) -> float:
    Xf = X.float()
    R = torch.quantile(Xf.abs(), 0.999, dim=0).clamp(min=1e-8)
    scale = R / 7.0
    q = torch.clamp(torch.round(Xf / scale), min=-7, max=7)
    Xq = q * scale
    num = ((Xf - Xq) ** 2).sum()
    return float((num / max(denom_norm_sq, 1e-12)).item())


def svdquant_residual(X: torch.Tensor, rank: int) -> torch.Tensor:
    Xs = X
    if Xs.shape[0] > 8192:
        idx = torch.randperm(Xs.shape[0], device=Xs.device)[:8192]
        Xs = Xs[idx]
    _U, _S, Vh = torch.linalg.svd(Xs.float(), full_matrices=False)
    Vr = Vh[:rank, :].T
    X_proj = (X.float() @ Vr) @ Vr.T
    return (X.float() - X_proj).to(X.dtype)


def _quantize_per_row(M: torch.Tensor, qmax: int = 7) -> torch.Tensor:
    """Symmetric INT4 along the LAST dim with one scale per row.
    Each row's scale = q999(|row|) / qmax."""
    Mf = M.float()
    R = torch.quantile(Mf.abs(), 0.999, dim=-1, keepdim=True).clamp_min(1e-8)
    scale = R / qmax
    q = torch.clamp(torch.round(Mf / scale), min=-qmax, max=qmax)
    return q * scale


def _quantize_per_in_channel(X: torch.Tensor, qmax: int = 7,
                             percentile: float = 0.999) -> torch.Tensor:
    """Symmetric INT4 with one scale per INPUT channel — matches GR00T's
    PercentileCalibrator (one act_scale per input feature, from q999 over
    calibration tokens)."""
    Xf = X.float()
    # X: [N, in].  Scale shape [in].
    p_vec = torch.quantile(Xf.abs(), percentile, dim=0)
    scale = (p_vec / qmax).clamp_min(1e-8)
    q = torch.clamp(torch.round(Xf / scale[None, :]), min=-qmax, max=qmax)
    return q * scale[None, :]


def _quantize_per_out_channel(W: torch.Tensor, qmax: int = 7) -> torch.Tensor:
    """Symmetric INT4 with one scale per OUTPUT channel (row of W [out, in]).
    Per-channel scale = max(|row|) / qmax."""
    Wf = W.float()
    R = Wf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = R / qmax
    q = torch.clamp(torch.round(Wf / scale), min=-qmax, max=qmax)
    return q * scale


def zigzag_perm_from_energy(energy: torch.Tensor) -> torch.Tensor:
    """Reproduces gr00t.quantization.duquant_preprocess.zigzag_permutation:
    sort channels by energy descending, then interleave from both ends so
    highest-energy channels are spread across blocks."""
    order = torch.argsort(-energy)
    n = energy.shape[0]
    perm = torch.empty(n, dtype=torch.long, device=energy.device)
    left = 0
    right = n - 1
    idx = 0
    toggle = True
    while left <= right:
        if toggle:
            perm[idx] = order[left]
            left += 1
        else:
            perm[idx] = order[right]
            right -= 1
        toggle = not toggle
        idx += 1
    return perm


def vanilla_duq_transform(W: torch.Tensor, block_size: int = 16,
                          lambda_smooth: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """Vanilla DuQuant (rot_mode='svd'): zigzag perm by W energy + per-block
    SVD basis (no Hadamard).  Returns (perm, R)."""
    Wf = W.detach().float()
    out_features, in_features = Wf.shape
    e = (Wf ** 2).mean(dim=0)
    e = (1.0 - lambda_smooth) * e + lambda_smooth * e.mean()
    perm = zigzag_perm_from_energy(e)
    W_perm = Wf[:, perm]
    assert in_features % block_size == 0
    n_blocks = in_features // block_size
    R = torch.zeros((in_features, in_features), device=Wf.device, dtype=torch.float32)
    for b in range(n_blocks):
        s = b * block_size
        e_idx = s + block_size
        W_block = W_perm[:, s:e_idx]
        U, _, _ = torch.linalg.svd(W_block.T, full_matrices=True)
        R[s:e_idx, s:e_idx] = U
    return perm, R


def a2_lite_transform(W: torch.Tensor, block_size: int = 16,
                      lambda_smooth: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """A2-lite (rot_mode='svd_hadamard'): same as vanilla DuQ but R = U @ H per block."""
    Wf = W.detach().float()
    out_features, in_features = Wf.shape
    e = (Wf ** 2).mean(dim=0)
    e = (1.0 - lambda_smooth) * e + lambda_smooth * e.mean()
    perm = zigzag_perm_from_energy(e)
    W_perm = Wf[:, perm]
    assert in_features % block_size == 0
    n_blocks = in_features // block_size
    H = hadamard_matrix(block_size, Wf.device, torch.float32)
    R = torch.zeros((in_features, in_features), device=Wf.device, dtype=torch.float32)
    for b in range(n_blocks):
        s = b * block_size
        e_idx = s + block_size
        W_block = W_perm[:, s:e_idx]
        U, _, _ = torch.linalg.svd(W_block.T, full_matrices=True)
        R[s:e_idx, s:e_idx] = U @ H
    return perm, R


def a2_full_transform(X: torch.Tensor, W: torch.Tensor, block_size: int = 16,
                      lambda_smooth: float = 0.15) -> tuple[torch.Tensor, torch.Tensor]:
    """A2 full: perm_score='act_weight' + rot='svd_hadamard'.
    Permutation energy = q999(|X[:,j]|)^2 * mean(W[:,j]^2)  (matches
    duquant_preprocess._compute_perm_energy for score='act_weight')."""
    Wf = W.detach().float()
    Xf = X.float()
    out_features, in_features = Wf.shape
    act_q999 = torch.quantile(Xf.abs(), 0.999, dim=0)  # [in_features]
    weight_e = (Wf ** 2).mean(dim=0)
    e = (act_q999 ** 2) * weight_e
    e = (1.0 - lambda_smooth) * e + lambda_smooth * e.mean()
    perm = zigzag_perm_from_energy(e)
    W_perm = Wf[:, perm]
    assert in_features % block_size == 0
    n_blocks = in_features // block_size
    H = hadamard_matrix(block_size, Wf.device, torch.float32)
    R = torch.zeros((in_features, in_features), device=Wf.device, dtype=torch.float32)
    for b in range(n_blocks):
        s = b * block_size
        e_idx = s + block_size
        W_block = W_perm[:, s:e_idx]
        U, _, _ = torch.linalg.svd(W_block.T, full_matrices=True)
        R[s:e_idx, s:e_idx] = U @ H
    return perm, R


def svdq_original_decompose(X: torch.Tensor, W: torch.Tensor, rank: int):
    """SVDQuant original (NVIDIA): SmoothQuant + top-r SVD of W_s directly.
    Returns (X_s, W_res, lowA, lowB) where:
       W_s = lowA @ lowB^T + W_res
       lowA @ lowB^T = top-r SVD of W_s
    """
    Xf = X.float()
    Wf = W.float()
    s = smoothquant_scale(Xf, Wf)
    X_s = Xf / s[None, :]
    W_s = Wf * s[None, :]
    U, S_, Vh = torch.linalg.svd(W_s, full_matrices=False)
    r = min(rank, S_.shape[0])
    sqrtS = torch.sqrt(S_[:r].clamp_min(0.0))
    lowA = U[:, :r] * sqrtS[None, :]
    lowB = Vh[:r, :].T * sqrtS[None, :]
    W_res = W_s - lowA @ lowB.T
    return X_s, W_res, lowA, lowB


# Lazy-load GPTQ from the actual codebase (only when invoked).
_GPTQ_FN = None


def _gptq(W: torch.Tensor, H: torch.Tensor, bits: int = 4,
          block_size: int = 128, damp_percent: float = 0.01) -> torch.Tensor:
    """Wrapper for the codebase's gptq_quantize_weight."""
    global _GPTQ_FN
    if _GPTQ_FN is None:
        from gr00t.quantization.gptq_layers import gptq_quantize_weight as _g
        _GPTQ_FN = _g
    return _GPTQ_FN(W, H, bits=bits, block_size=block_size,
                    damp_percent=damp_percent)


def svdq_original_gptq(X: torch.Tensor, W: torch.Tensor, rank: int,
                       gptq_block_size: int = 128,
                       damp_percent: float = 0.01):
    """SVDQ original WITH GPTQ.  Matches build_dit_svdquant_weights.py's
    quant_type='original' branch:
      X_s = X / s, W_s = W * s
      SVD W_s -> lowA, lowB  (top-r of W_s itself)
      W_res = W_s - lowA @ lowB^T
      H = X_s^T X_s / n
      W_res_q = GPTQ(W_res, H, bits=4)
    Returns (X_s, W_res, lowA, lowB, W_res_q).
    """
    Xf = X.float()
    Wf = W.float()
    s = smoothquant_scale(Xf, Wf)
    X_s = Xf / s[None, :]
    W_s = Wf * s[None, :]
    U, S_, Vh = torch.linalg.svd(W_s, full_matrices=False)
    r = min(rank, S_.shape[0])
    sqrtS = torch.sqrt(S_[:r].clamp_min(0.0))
    lowA = U[:, :r] * sqrtS[None, :]
    lowB = Vh[:r, :].T * sqrtS[None, :]
    W_res = W_s - lowA @ lowB.T
    n = max(X_s.shape[0], 1)
    H = (X_s.T @ X_s) / float(n)
    W_res_q = _gptq(W_res, H, bits=4, block_size=gptq_block_size,
                    damp_percent=damp_percent)
    return X_s, W_res, lowA, lowB, W_res_q


def svdq_custom_gptq(X: torch.Tensor, W: torch.Tensor, rank: int,
                     gptq_block_size: int = 128,
                     damp_percent: float = 0.01):
    """SVDQ custom (error-SVD) WITH GPTQ.  Matches build_dit_svdquant_weights.py's
    quant_type='custom' branch:
      X_s = X / s, W_s = W * s
      H = X_s^T X_s / n
      W4_tmp = GPTQ(W_s, H, bits=4)              # first-pass
      E = W_s - W4_tmp
      SVD E -> lowA, lowB                         # top-r of error
      W_res = W_s - lowA @ lowB^T
      W_res_q = GPTQ(W_res, H, bits=4)            # second-pass
    Returns (X_s, W_res, lowA, lowB, W_res_q).
    """
    Xf = X.float()
    Wf = W.float()
    s = smoothquant_scale(Xf, Wf)
    X_s = Xf / s[None, :]
    W_s = Wf * s[None, :]
    n = max(X_s.shape[0], 1)
    H = (X_s.T @ X_s) / float(n)
    # First-pass GPTQ
    W4_tmp = _gptq(W_s, H, bits=4, block_size=gptq_block_size,
                   damp_percent=damp_percent)
    E = W_s - W4_tmp
    U, S_, Vh = torch.linalg.svd(E, full_matrices=False)
    r = min(rank, S_.shape[0])
    sqrtS = torch.sqrt(S_[:r].clamp_min(0.0))
    lowA = U[:, :r] * sqrtS[None, :]
    lowB = Vh[:r, :].T * sqrtS[None, :]
    W_res = W_s - lowA @ lowB.T
    # Second-pass GPTQ
    W_res_q = _gptq(W_res, H, bits=4, block_size=gptq_block_size,
                    damp_percent=damp_percent)
    return X_s, W_res, lowA, lowB, W_res_q


def compute_method_metrics_svdq_gptq(X: torch.Tensor, W: torch.Tensor,
                                     X_s: torch.Tensor, W_res: torch.Tensor,
                                     lowA: torch.Tensor, lowB: torch.Tensor,
                                     W_res_q: torch.Tensor) -> dict:
    """Like compute_method_metrics_svdq but uses externally-computed W_res_q
    (GPTQ-quantized residual) instead of re-quantizing with RTN."""
    Xf, Wf = X.float(), W.float()
    Y = Xf @ Wf.T
    Y_norm_sq = (Y ** 2).sum().clamp_min(1e-12)
    W_s = (W_res.float() + lowA.float() @ lowB.float().T)
    # Activation: per-input-channel INT4 (unchanged from RTN path)
    X_sq = _quantize_per_in_channel(X_s.float())
    Y_int = X_sq @ W_res_q.T
    Y_low = X_s.float() @ lowB.float() @ lowA.float().T
    Y_pred = Y_int + Y_low
    act_e = ((X_sq - X_s) ** 2).sum() / (X_s ** 2).sum().clamp_min(1e-12)
    W_recovered = lowA.float() @ lowB.float().T + W_res_q
    w_e = ((W_recovered - W_s) ** 2).sum() / (W_s ** 2).sum().clamp_min(1e-12)
    o_e = ((Y_pred - Y) ** 2).sum() / Y_norm_sq
    return {"act": float(act_e.item()), "weight": float(w_e.item()),
            "out": float(o_e.item())}


def output_quant_error(X: torch.Tensor, W: torch.Tensor) -> float:
    """Joint INT4 layer-output error: ||Q(X) Q(W)^T - X W^T||^2 / ||X W^T||^2.

    X: [N, in], per-row activation INT4.
    W: [out, in], per-out-channel weight INT4.
    """
    Xf = X.float()
    Wf = W.float()
    Y = Xf @ Wf.T
    Y_norm_sq = (Y ** 2).sum().clamp_min(1e-12)
    Xq = _quantize_per_row(Xf)
    Wq = _quantize_per_out_channel(Wf)
    Yq = Xq @ Wq.T
    err = ((Yq - Y) ** 2).sum() / Y_norm_sq
    return float(err.item())


def output_quant_error_with_rotation(X: torch.Tensor, W: torch.Tensor,
                                     R: torch.Tensor) -> float:
    """Output error after applying rotation R (in_features × in_features) to
    both X and W along the in_features axis (preserves the matmul exactly in
    fp32; rotation only changes the distribution that gets quantized)."""
    Rf = R.to(dtype=torch.float32, device=X.device)
    Xr = X.float() @ Rf
    Wr = W.float() @ Rf
    return output_quant_error(Xr, Wr)


def smoothquant_scale(X: torch.Tensor, W: torch.Tensor,
                      alpha: float = 0.5,
                      clamp_lo: float = 0.1,
                      clamp_hi: float = 10.0) -> torch.Tensor:
    """Reproduces SmoothQuant smoothing scale from
    build_dit_svdquant_weights.py:
        A_j = q999(|X_:,j|)         per-channel activation outlier
        B_j = max_i |W_i,j|         per-channel weight outlier
        s_j = A_j^alpha / B_j^(1-alpha)
        s   = clamp(s / median(s), [lo, hi])"""
    Xf = X.float()
    Wf = W.float()
    eps = 1e-8
    A = torch.quantile(Xf.abs(), 0.999, dim=0).clamp_min(eps)   # [in]
    B = Wf.abs().amax(dim=0).clamp_min(eps)                     # [in]
    s = (A ** alpha) / (B ** (1.0 - alpha))
    s = s / s.median().clamp_min(eps)
    s = s.clamp(min=clamp_lo, max=clamp_hi)
    return s  # [in_features]


def svdquant_faithful(X: torch.Tensor, W: torch.Tensor, rank: int
                      ) -> tuple[torch.Tensor, torch.Tensor,
                                 torch.Tensor, torch.Tensor]:
    """Faithful SVDQuant surrogate following build_dit_svdquant_weights.py:

      1. SmoothQuant: s = A^a / B^(1-a), W_s = W * s, X_s = X / s
      2. First-pass RTN INT4 of W_s -> W4_tmp; error E = W_s - W4_tmp
      3. Top-r SVD of E:
           lowrank_A = U_r * sqrt(S_r)  [out, r]
           lowrank_B = V_r * sqrt(S_r)  [in,  r]
         and W_res = W_s - lowrank_A @ lowrank_B^T
      4. Inference:
           INT4 path: Q(X_s) @ Q(W_res)^T
           FP16 path: X_s @ lowrank_B @ lowrank_A^T

    GPTQ in the real builder is approximated here by RTN (per-out-channel
    INT4); calibration weight Hessian H_s is omitted.  The SmoothQuant +
    error-SVD structure that drives the +pp gain is preserved.

    Returns (X_s, W_res, lowrank_A, lowrank_B) — quantization is left to
    the caller."""
    Xf = X.float()
    Wf = W.float()
    s = smoothquant_scale(Xf, Wf)
    X_s = Xf / s[None, :]
    W_s = Wf * s[None, :]

    W4_tmp = _quantize_per_out_channel(W_s)
    E = W_s - W4_tmp

    U, S_, Vh = torch.linalg.svd(E, full_matrices=False)
    r = min(rank, S_.shape[0])
    sqrtS = torch.sqrt(S_[:r].clamp_min(0.0))
    lowrank_A = U[:, :r] * sqrtS[None, :]          # [out, r]
    lowrank_B = (Vh[:r, :].T) * sqrtS[None, :]     # [in,  r]
    W_res = W_s - lowrank_A @ lowrank_B.T          # ≈ W4_tmp + residual SVD tail
    return X_s, W_res, lowrank_A, lowrank_B


def output_quant_error_svdquant(X: torch.Tensor, W: torch.Tensor,
                                rank: int) -> float:
    """SVDQuant joint output error with SmoothQuant + error-SVD."""
    Xf = X.float()
    Wf = W.float()
    Y = Xf @ Wf.T
    Y_norm_sq = (Y ** 2).sum().clamp_min(1e-12)
    X_s, W_res, lowA, lowB = svdquant_faithful(Xf, Wf, rank)
    X_sq = _quantize_per_row(X_s)
    W_resq = _quantize_per_out_channel(W_res)
    Y_int = X_sq @ W_resq.T              # INT4 path (smoothed X and residual W)
    Y_low = X_s @ lowB @ lowA.T          # FP16 low-rank path
    Y_pred = Y_int + Y_low
    err = ((Y_pred - Y) ** 2).sum() / Y_norm_sq
    return float(err.item())


def output_quant_error_smoothquant_only(X: torch.Tensor,
                                        W: torch.Tensor) -> float:
    """Ablation A: SmoothQuant only, NO low-rank decomposition.

    s = A^a / B^(1-a) (clamped),  X_s = X/s,  W_s = W*s
    Y_pred = Q(X_s) @ Q(W_s)^T  (single INT4 path, no FP16 head)
    """
    Xf = X.float()
    Wf = W.float()
    Y = Xf @ Wf.T
    Y_norm_sq = (Y ** 2).sum().clamp_min(1e-12)
    s = smoothquant_scale(Xf, Wf)
    X_s = Xf / s[None, :]
    W_s = Wf * s[None, :]
    X_sq = _quantize_per_row(X_s)
    W_sq = _quantize_per_out_channel(W_s)
    Y_pred = X_sq @ W_sq.T
    err = ((Y_pred - Y) ** 2).sum() / Y_norm_sq
    return float(err.item())


def output_quant_error_errsvd_only(X: torch.Tensor, W: torch.Tensor,
                                   rank: int) -> float:
    """Ablation B: error-SVD only, NO SmoothQuant preprocessing.

    W4_tmp = Q(W) (per-out-ch RTN)
    E = W - W4_tmp
    top-r SVD of E -> lowA, lowB
    W_res = W - lowA @ lowB^T
    Y_pred = Q(X) @ Q(W_res)^T + X @ lowB @ lowA^T
    """
    Xf = X.float()
    Wf = W.float()
    Y = Xf @ Wf.T
    Y_norm_sq = (Y ** 2).sum().clamp_min(1e-12)
    W4_tmp = _quantize_per_out_channel(Wf)
    E = Wf - W4_tmp
    U, S_, Vh = torch.linalg.svd(E, full_matrices=False)
    r = min(rank, S_.shape[0])
    sqrtS = torch.sqrt(S_[:r].clamp_min(0.0))
    lowA = U[:, :r] * sqrtS[None, :]
    lowB = Vh[:r, :].T * sqrtS[None, :]
    W_res = Wf - lowA @ lowB.T
    W_resq = _quantize_per_out_channel(W_res)
    X_q = _quantize_per_row(Xf)
    Y_int = X_q @ W_resq.T
    Y_low = Xf @ lowB @ lowA.T
    Y_pred = Y_int + Y_low
    err = ((Y_pred - Y) ** 2).sum() / Y_norm_sq
    return float(err.item())


def act_quant_error_per_row(X: torch.Tensor) -> float:
    """Per-row INT4 activation quant error: ||Q(X) - X||^2 / ||X||^2."""
    Xf = X.float()
    Xq = _quantize_per_row(Xf)
    num = ((Xq - Xf) ** 2).sum()
    den = (Xf ** 2).sum().clamp_min(1e-12)
    return float((num / den).item())


def weight_quant_error_per_out_ch(W: torch.Tensor) -> float:
    """Per-out-channel INT4 weight quant error: ||Q(W) - W||^2 / ||W||^2."""
    Wf = W.float()
    Wq = _quantize_per_out_channel(Wf)
    num = ((Wq - Wf) ** 2).sum()
    den = (Wf ** 2).sum().clamp_min(1e-12)
    return float((num / den).item())


def weight_quant_error_svdquant(W_s: torch.Tensor, W_res: torch.Tensor,
                                lowrank_A: torch.Tensor,
                                lowrank_B: torch.Tensor) -> float:
    """Faithful SVDQuant weight reconstruction error.  The lowrank head is
    rank-r filtered (only top-r singular components of E are kept in FP16);
    the residual W_res is INT4 quantized.

    Reconstruction explicitly: W_recovered = lowrank_A @ lowrank_B^T + Q(W_res)
    Error explicitly:          ||W_s - W_recovered||^2 / ||W_s||^2

    This is the TRUE total weight reconstruction error.  Algebraically it
    equals ||W_res - Q(W_res)||^2 / ||W_s||^2 (lowrank terms cancel) but we
    compute it the explicit way so the FP16 head's rank-r filtering is
    materialised in the calculation, not assumed."""
    W_sf = W_s.float()
    W_resq = _quantize_per_out_channel(W_res.float())
    W_recovered = lowrank_A.float() @ lowrank_B.float().T + W_resq
    num = ((W_sf - W_recovered) ** 2).sum()
    den = (W_sf ** 2).sum().clamp_min(1e-12)
    return float((num / den).item())


def compute_method_metrics_raw(X: torch.Tensor, W: torch.Tensor) -> dict:
    Xf, Wf = X.float(), W.float()
    Y = Xf @ Wf.T
    Y_norm_sq = (Y ** 2).sum().clamp_min(1e-12)
    Xq = _quantize_per_in_channel(Xf)
    Wq = _quantize_per_out_channel(Wf)
    Y_pred = Xq @ Wq.T
    act_e = ((Xq - Xf) ** 2).sum() / (Xf ** 2).sum().clamp_min(1e-12)
    w_e = ((Wq - Wf) ** 2).sum() / (Wf ** 2).sum().clamp_min(1e-12)
    o_e = ((Y_pred - Y) ** 2).sum() / Y_norm_sq
    return {"act": float(act_e.item()), "weight": float(w_e.item()),
            "out": float(o_e.item())}


def compute_method_metrics_rotation(X: torch.Tensor, W: torch.Tensor,
                                    perm: torch.Tensor, R: torch.Tensor) -> dict:
    """For DuQ-family methods (vanilla DuQ, A2-lite).  X' = X[:, perm] @ R,
    W' = W[:, perm] @ R, then per-channel X and per-out-channel W INT4."""
    Xf, Wf = X.float(), W.float()
    Y = Xf @ Wf.T
    Y_norm_sq = (Y ** 2).sum().clamp_min(1e-12)
    Rf = R.to(Xf.device).float()
    Xt = Xf[:, perm] @ Rf
    Wt = Wf[:, perm] @ Rf
    Xq = _quantize_per_in_channel(Xt)
    Wq = _quantize_per_out_channel(Wt)
    Y_pred = Xq @ Wq.T
    act_e = ((Xq - Xt) ** 2).sum() / (Xt ** 2).sum().clamp_min(1e-12)
    w_e = ((Wq - Wt) ** 2).sum() / (Wt ** 2).sum().clamp_min(1e-12)
    o_e = ((Y_pred - Y) ** 2).sum() / Y_norm_sq
    return {"act": float(act_e.item()), "weight": float(w_e.item()),
            "out": float(o_e.item())}


def compute_method_metrics_svdq(X: torch.Tensor, W: torch.Tensor,
                                X_s: torch.Tensor, W_res: torch.Tensor,
                                lowA: torch.Tensor, lowB: torch.Tensor) -> dict:
    """For SVDQ-family methods (original, custom).  X_s = X / s, W_s = lowA @ lowB^T + W_res.
    Quantize X_s per-channel, W_res per-out-channel.  Lowrank stays FP16."""
    Xf, Wf = X.float(), W.float()
    Y = Xf @ Wf.T
    Y_norm_sq = (Y ** 2).sum().clamp_min(1e-12)
    W_s = (W_res.float() + lowA.float() @ lowB.float().T)  # by construction
    X_sq = _quantize_per_in_channel(X_s.float())
    W_resq = _quantize_per_out_channel(W_res.float())
    Y_int = X_sq @ W_resq.T
    Y_low = X_s.float() @ lowB.float() @ lowA.float().T
    Y_pred = Y_int + Y_low
    # Activation error: Q(X_s) vs X_s (the INT4-path input).
    act_e = ((X_sq - X_s) ** 2).sum() / (X_s ** 2).sum().clamp_min(1e-12)
    # Weight error: ||W_s - W_recovered||^2 / ||W_s||^2
    # = ||lowA@lowB^T + Q(W_res) - W_s||^2 = ||Q(W_res) - W_res||^2
    W_recovered = lowA.float() @ lowB.float().T + W_resq
    w_e = ((W_recovered - W_s) ** 2).sum() / (W_s ** 2).sum().clamp_min(1e-12)
    o_e = ((Y_pred - Y) ** 2).sum() / Y_norm_sq
    return {"act": float(act_e.item()), "weight": float(w_e.item()),
            "out": float(o_e.item())}


def act_quant_error_svdquant(X_s: torch.Tensor) -> float:
    """Faithful SVDQuant activation error.  X_s is the smoothed activation.
    The FP16 lowrank path uses X_s un-quantized (zero error); the INT4 path
    uses Q(X_s).  Reporting the per-row INT4 path error normalized by ||X_s||^2,
    which is the activation quantization cost incurred only by the INT4 branch.
    (The joint out_err metric accounts for the FP16 path's mitigation.)"""
    X_sf = X_s.float()
    X_sq = _quantize_per_row(X_sf)
    num = ((X_sq - X_sf) ** 2).sum()
    den = (X_sf ** 2).sum().clamp_min(1e-12)
    return float((num / den).item())


# ----------------------------- arg parsing -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-config",
                   default="examples.Libero.custom_data_config:LiberoDataConfig")
    p.add_argument("--task-suite-name", required=True,
                   help="libero_object | libero_goal | libero_spatial | libero_10")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--libero-resolution", type=int, default=256)
    p.add_argument("--libero-num-trials-per-task", type=int, default=1)
    p.add_argument("--libero-num-steps-wait", type=int, default=10)
    p.add_argument("--libero-sampling-mode", default="one_per_task")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--embodiment-tag", default="new_embodiment")
    p.add_argument("--denoising-steps", type=int, default=8)
    p.add_argument("--device", default="cuda")

    p.add_argument("--llm-layers", type=int, nargs="+", default=[2, 5, 9],
                   help="LLM layer indices to probe (Qwen3, 12 layers).")
    p.add_argument("--dit-layers", type=int, nargs="+", default=[2, 8, 14],
                   help="DiT block indices to probe (16 blocks).")
    p.add_argument("--max-tokens-per-layer", type=int, default=4096)
    p.add_argument("--block-size", type=int, default=16,
                   help="DuQuant block_size. Default 16 matches deployment "
                        "(GR00T_DUQUANT_BLOCK).")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--output-tag", required=True,
                   help="Short tag for outputs, e.g. 'object' or 'long'.")
    p.add_argument("--save-x", action="store_true",
                   help="Also save downsampled per-layer X (1024 tokens) plus W "
                        "for cross-suite calibration-transfer analysis.")
    p.add_argument("--save-x-tokens", type=int, default=1024,
                   help="Max tokens per layer to save when --save-x is set.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_libero_runtime()

    print(f"[GR00T-DIAG] loading policy: {args.checkpoint}", flush=True)
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

    # Locate the two stacks.
    llm_layers = model.backbone.eagle_model.language_model.model.layers
    dit_layers = model.action_head.model.transformer_blocks
    print(f"[GR00T-DIAG] LLM depth={len(llm_layers)}, DiT depth={len(dit_layers)}",
          flush=True)

    captured: dict[str, list[torch.Tensor]] = {}
    captured_steps: dict[str, list[int | None]] = {}  # parallel to captured, step per call
    tag_to_module: dict[str, torch.nn.Linear] = {}

    try:
        from gr00t.quantization.dit_step_context import get_current_dit_step
    except Exception:
        def get_current_dit_step():
            return None

    def make_hook(name: str):
        is_dit = name.startswith("DiT")
        def hook(_m, inputs, _out):
            x = inputs[0]
            if x is None:
                return
            xa = x.detach().reshape(-1, x.shape[-1]).float().cpu()
            captured.setdefault(name, []).append(xa)
            step = get_current_dit_step() if is_dit else None
            captured_steps.setdefault(name, []).append(step)
        return hook

    def register(tag: str, module: torch.nn.Linear):
        tag_to_module[tag] = module
        return module.register_forward_hook(make_hook(tag))

    handles = []
    for li in args.llm_layers:
        if li >= len(llm_layers):
            continue
        attn = llm_layers[li].self_attn
        handles.append(register(f"LLM.L{li:02d}.q_proj", attn.q_proj))
        handles.append(register(f"LLM.L{li:02d}.o_proj", attn.o_proj))
        mlp = llm_layers[li].mlp
        handles.append(register(f"LLM.L{li:02d}.down_proj", mlp.down_proj))
        handles.append(register(f"LLM.L{li:02d}.gate_proj", mlp.gate_proj))
    for li in args.dit_layers:
        if li >= len(dit_layers):
            continue
        attn = dit_layers[li].attn1
        handles.append(register(f"DiT.L{li:02d}.q_proj", attn.to_q))
        handles.append(register(f"DiT.L{li:02d}.o_proj", attn.to_out[0]))
        ff_net = dit_layers[li].ff.net
        if len(ff_net) > 2:
            handles.append(register(f"DiT.L{li:02d}.down_proj", ff_net[2]))

    print(f"[GR00T-DIAG] loading {args.num_samples} obs from {args.task_suite_name}",
          flush=True)
    samples = load_libero_samples(args, data_config)
    if isinstance(samples, tuple):
        # analyze_layerwise_quant_drift.load_libero_samples returns (env_info, lang, samples)
        samples = samples[-1]
    samples = samples[: args.num_samples]
    print(f"[GR00T-DIAG] got {len(samples)} samples", flush=True)

    try:
        for i, sample in enumerate(samples, 1):
            obs = sample["obs"] if isinstance(sample, dict) and "obs" in sample else sample
            lang = sample.get("language", "?") if isinstance(sample, dict) else "?"
            print(f"[GR00T-DIAG] sample {i}/{len(samples)}: {lang!r}", flush=True)
            with torch.no_grad():
                _ = policy.get_action(obs)
    finally:
        for h in handles:
            h.remove()

    print("[GR00T-DIAG] computing spectra", flush=True)
    results: dict[str, dict] = {"_meta": {"task_suite": args.task_suite_name,
                                          "checkpoint": args.checkpoint,
                                          "num_samples": len(samples)}}
    for tag, chunks in captured.items():
        X = torch.cat(chunks, dim=0)
        if X.shape[0] > args.max_tokens_per_layer:
            idx = torch.randperm(X.shape[0])[: args.max_tokens_per_layer]
            X = X[idx]
        X = X.to(args.device)
        d = X.shape[1]
        Xnorm_sq = float((X.float() ** 2).sum().item())
        module = tag_to_module.get(tag)
        W = module.weight.to(X.device) if module is not None else None

        rec: dict = {"tag": tag, "N_tokens": int(X.shape[0]), "dim": int(d),
                     "Xnorm_sq": Xnorm_sq,
                     "out_features": int(W.shape[0]) if W is not None else None}
        S = spectrum(X)
        rec["spectrum_raw"] = S
        rec["stable_rank_raw"] = stable_rank(S)
        rec["chan_range_raw"] = per_channel_range(X)
        rec["int4_err_raw"] = int4_per_channel_quant_error(X, Xnorm_sq)
        prm_raw = per_row_max_abs(X)
        rec["per_row_max_raw"] = prm_raw

        # Outlier topology: is energy concentrated along channels or tokens?
        rec["topology"] = outlier_topology_stats(X)

        # ---- 3-metric set per method: act / weight / joint output ----
        if W is not None:
            try:
                rec["act_err_raw"] = act_quant_error_per_row(X)
                rec["weight_err_raw"] = weight_quant_error_per_out_ch(W)
                rec["out_err_raw"] = output_quant_error(X, W)
            except Exception as e:
                print(f"[GR00T-DIAG] {tag}: raw 3-metric skipped ({e})")

        try:
            R_blk = block_hadamard_rotation(d, args.block_size, X.device, X.dtype)
            X_blk = X @ R_blk
            rec["chan_range_block_hadamard"] = per_channel_range(X_blk)
            rec["int4_err_block_hadamard"] = int4_per_channel_quant_error(X_blk, Xnorm_sq)
            rec["per_row_max_block_hadamard"] = per_row_max_abs(X_blk)
            if W is not None:
                W_blk = W.float() @ R_blk.to(W.device).float()
                rec["act_err_block_hadamard"] = act_quant_error_per_row(X_blk)
                rec["weight_err_block_hadamard"] = weight_quant_error_per_out_ch(W_blk)
                rec["out_err_block_hadamard"] = output_quant_error_with_rotation(X, W, R_blk)
        except AssertionError as e:
            print(f"[GR00T-DIAG] {tag}: block-Hadamard skipped ({e})")

        try:
            if W is None:
                raise RuntimeError("no module reference for tag")
            R_svdh = svd_hadamard_rotation_from_weight(W, args.block_size)
            X_svdh = X @ R_svdh.to(X.dtype)
            rec["chan_range_svd_hadamard"] = per_channel_range(X_svdh)
            rec["int4_err_svd_hadamard"] = int4_per_channel_quant_error(X_svdh, Xnorm_sq)
            rec["per_row_max_svd_hadamard"] = per_row_max_abs(X_svdh)
            W_svdh = W.float() @ R_svdh.to(W.device).float()
            rec["act_err_svd_hadamard"] = act_quant_error_per_row(X_svdh)
            rec["weight_err_svd_hadamard"] = weight_quant_error_per_out_ch(W_svdh)
            rec["out_err_svd_hadamard"] = output_quant_error_with_rotation(X, W, R_svdh)
        except Exception as e:
            print(f"[GR00T-DIAG] {tag}: svd_hadamard skipped ({e})")

        # ============================================================
        # Unified ablation table — 4 paper methods + raw baseline.
        # Uses per-input-channel X INT4 + per-out-channel W INT4 to match
        # GR00T DuQ deployment.
        # ============================================================
        if W is not None:
            try:
                method_table = {}
                method_table["raw"] = compute_method_metrics_raw(X, W)
                perm_vd, R_vd = vanilla_duq_transform(W, args.block_size)
                method_table["vanilla_duq"] = compute_method_metrics_rotation(
                    X, W, perm_vd, R_vd)
                perm_a2, R_a2 = a2_lite_transform(W, args.block_size)
                method_table["a2_lite"] = compute_method_metrics_rotation(
                    X, W, perm_a2, R_a2)
                perm_a2f, R_a2f = a2_full_transform(X, W, args.block_size)
                method_table["a2_full"] = compute_method_metrics_rotation(
                    X, W, perm_a2f, R_a2f)
                for r in (16, 32):
                    # RTN-quantized variants (kept for comparison)
                    Xs_o, Wres_o, lowA_o, lowB_o = svdq_original_decompose(X, W, r)
                    method_table[f"svdq_original_rtn_r{r}"] = compute_method_metrics_svdq(
                        X, W, Xs_o, Wres_o, lowA_o, lowB_o)
                    Xs_c, Wres_c, lowA_c, lowB_c = svdquant_faithful(X, W, r)
                    method_table[f"svdq_custom_rtn_r{r}"] = compute_method_metrics_svdq(
                        X, W, Xs_c, Wres_c, lowA_c, lowB_c)
                    # GPTQ variants (match real deployment)
                    try:
                        Xs_og, Wres_og, lowA_og, lowB_og, Wresq_og = svdq_original_gptq(
                            X, W, r)
                        method_table[f"svdq_original_gptq_r{r}"] = \
                            compute_method_metrics_svdq_gptq(
                                X, W, Xs_og, Wres_og, lowA_og, lowB_og, Wresq_og)
                    except Exception as e:
                        print(f"[GR00T-DIAG] {tag}: svdq_original_gptq_r{r} skipped ({e})")
                    try:
                        Xs_cg, Wres_cg, lowA_cg, lowB_cg, Wresq_cg = svdq_custom_gptq(
                            X, W, r)
                        method_table[f"svdq_custom_gptq_r{r}"] = \
                            compute_method_metrics_svdq_gptq(
                                X, W, Xs_cg, Wres_cg, lowA_cg, lowB_cg, Wresq_cg)
                    except Exception as e:
                        print(f"[GR00T-DIAG] {tag}: svdq_custom_gptq_r{r} skipped ({e})")
                rec["method_table"] = method_table
            except Exception as e:
                print(f"[GR00T-DIAG] {tag}: method_table skipped ({e})")

        if W is not None:
            try:
                rec["out_err_smoothquant"] = output_quant_error_smoothquant_only(X, W)
            except Exception as e:
                print(f"[GR00T-DIAG] {tag}: smoothquant_only skipped ({e})")

        for r in (16, 32):
            try:
                Xres = svdquant_residual(X, r)
                rec[f"chan_range_svdq_r{r}"] = per_channel_range(Xres)
                rec[f"int4_err_svdq_r{r}"] = int4_per_channel_quant_error(Xres, Xnorm_sq)
                rec[f"per_row_max_svdq_r{r}"] = per_row_max_abs(Xres)
                if W is not None:
                    X_s, W_res, lowA, lowB = svdquant_faithful(X, W, r)
                    s = smoothquant_scale(X.float(), W.float())
                    W_s = W.float() * s[None, :]
                    rec[f"act_err_svdq_r{r}"] = act_quant_error_svdquant(X_s)
                    rec[f"weight_err_svdq_r{r}"] = weight_quant_error_svdquant(
                        W_s, W_res, lowA, lowB)
                    rec[f"out_err_svdq_r{r}"] = output_quant_error_svdquant(X, W, r)
                    # error-SVD-only ablation (no SmoothQuant).
                    rec[f"out_err_errsvd_r{r}"] = output_quant_error_errsvd_only(X, W, r)
            except Exception as e:
                print(f"[GR00T-DIAG] {tag}: svdq_r{r} skipped ({e})")

        results[tag] = rec
        msg = f"  {tag}: d={d}, N={X.shape[0]}"
        for key in ("out_err_raw", "out_err_block_hadamard", "out_err_svd_hadamard",
                    "out_err_svdq_r16", "out_err_svdq_r32"):
            if key in rec:
                msg += f", {key.replace('out_err_','')}={rec[key]:.4f}"
        print(msg)

    out_pt = out_dir / f"outlier_topology_{args.output_tag}.pt"
    torch.save(results, out_pt)
    print(f"[GR00T-DIAG] wrote {out_pt}", flush=True)

    if args.save_x:
        # Save downsampled X + W per layer.  For DiT layers we also save per-step
        # buckets so the cal/eval-split surrogate can model SVDQuant's per-step
        # act_scale_table.
        xw_dump = {}
        for tag, chunks in captured.items():
            X_all = torch.cat(chunks, dim=0).float()
            module = tag_to_module.get(tag)
            W = module.weight.detach().float().cpu() if module is not None else None
            rec = {"W": W}
            if tag.startswith("DiT"):
                # Group chunks by step (None tolerated for non-step contexts).
                steps_per_chunk = captured_steps.get(tag, [None] * len(chunks))
                by_step: dict[int | None, list[torch.Tensor]] = {}
                for c, s in zip(chunks, steps_per_chunk):
                    by_step.setdefault(s, []).append(c)
                X_per_step = {}
                # Cap EACH STEP at save_x_tokens (not split across steps) so
                # per-step calibration has enough samples for stable q999.
                cap_per_step = args.save_x_tokens
                for s, cs in by_step.items():
                    Xs = torch.cat(cs, dim=0).float()
                    if Xs.shape[0] > cap_per_step:
                        idx = torch.randperm(Xs.shape[0])[:cap_per_step]
                        Xs = Xs[idx]
                    X_per_step[s] = Xs.cpu()
                # Aggregate X (used by methods that don't care about steps).
                X = torch.cat(list(X_per_step.values()), dim=0)
                rec["X"] = X
                rec["X_per_step"] = X_per_step
            else:
                X = X_all
                if X.shape[0] > args.save_x_tokens:
                    idx = torch.randperm(X.shape[0])[: args.save_x_tokens]
                    X = X[idx]
                rec["X"] = X.cpu()
            xw_dump[tag] = rec
        xw_path = out_dir / f"xw_dump_{args.output_tag}.pt"
        torch.save(xw_dump, xw_path)
        print(f"[GR00T-DIAG] wrote {xw_path} (X+W for {len(xw_dump)} layers, "
              f"DiT also per-step)", flush=True)


if __name__ == "__main__":
    main()
