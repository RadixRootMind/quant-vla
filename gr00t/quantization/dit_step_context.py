"""Current DiT denoising step (contextvars-based for async/threaded safety).

The DiT action head runs ``num_inference_timesteps`` (default 8) denoising
iterations per observation. Each iteration produces a different activation
distribution at every DiT linear layer; a single static A8 scale averages
over these distributions and underuses the available 8-bit headroom.

This module exposes a per-context "current DiT step" via ``contextvars``
(thread- AND async-safe; child tasks inherit the value, mutations don't
leak out).  The denoising loop in ``flow_matching_action_head.py`` enters
``set_dit_quant_step(t)`` per iteration; quantized linears
(``GptqLinear``) and calibration hooks read it via
``get_current_dit_step()`` to dispatch per-step activation scales / collect
per-step calibration stats.

When no step is active (LLM forward, non-DiT contexts), the getter returns
``None`` and consumers fall back to their step-agnostic behavior.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

# ContextVar (vs threading.local) is the right tool here:
#   * thread-safe by construction
#   * async-safe: each asyncio task has its own context
#   * multiprocessing-irrelevant (each process has its own module state)
_step_var: ContextVar[Optional[int]] = ContextVar("gr00t_dit_step", default=None)
_total_var: ContextVar[Optional[int]] = ContextVar("gr00t_dit_total_steps", default=None)


def get_current_dit_step() -> Optional[int]:
    """Return the current DiT denoising step index (0-based), or None."""
    return _step_var.get()


def get_total_dit_steps() -> Optional[int]:
    """Return total denoising steps for the active loop, or None."""
    return _total_var.get()


@contextmanager
def set_dit_quant_step(step: Optional[int], total: Optional[int] = None) -> Iterator[None]:
    """Context manager for the current DiT denoising step.

    Usage in the denoising loop:
        for t in range(num_steps):
            with set_dit_quant_step(t, total=num_steps):
                output = self.model(...)

    The context restores prior state on exit (re-entrant safe).
    """
    step_token = _step_var.set(step)
    total_token = _total_var.set(total) if total is not None else None
    try:
        yield
    finally:
        _step_var.reset(step_token)
        if total_token is not None:
            _total_var.reset(total_token)
