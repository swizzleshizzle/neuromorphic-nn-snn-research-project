"""Gating primitives — apply a router's control lines to a pathway.

The Thalamic Router emits **gate-closed** signals (1 = channel tonically inhibited
/ closed, 0 = disinhibited / open). ``apply_gate`` releases a content pathway
through those gates: open channels pass, closed channels are blocked. Reused for
the gated pathways 3 (store), 4 (recall), and 5 (action-enable).

``apply_gain`` generalises this to a **continuous multiplicative gain** ``g`` — the
biological thalamic gain-modulation view (spec §2.5, designed L11). Because
``g·(Wx) = (gW)x``, scaling the signal is equivalent to scaling the projection
weights: ``g=0`` off · ``0<g<1`` suppress · ``g=1`` pass · ``g>1`` amplify. The
binary gate is the special case ``g = 1 − gate_closed``, so the router can now
**amplify** a pathway, not only veto it (L10 burst = transient high gain).
"""

from __future__ import annotations

import torch


def apply_gate(signal: torch.Tensor, gate_closed: torch.Tensor) -> torch.Tensor:
    """Release ``signal`` through a router gate.

    Args:
        signal: pathway content, any shape (e.g. ``[T, B, N]``).
        gate_closed: same shape; 1 = channel closed (blocked), 0 = open (passes).

    Returns:
        ``signal * (1 - gate_closed)`` — open channels pass, closed channels zero.
    """
    if signal.shape != gate_closed.shape:
        raise ValueError(
            f"apply_gate shape mismatch: signal {tuple(signal.shape)} vs "
            f"gate {tuple(gate_closed.shape)}"
        )
    return signal * (1.0 - gate_closed)


def apply_gain(signal: torch.Tensor, gain: torch.Tensor | float) -> torch.Tensor:
    """Scale ``signal`` by a continuous per-pathway gain ``g``.

    Args:
        signal: pathway content, any shape (e.g. ``[T, B, N]``).
        gain: a scalar (applied uniformly) or a tensor matching ``signal``'s shape
            for per-channel control. ``g ∈ [0, g_max]``: ``0`` off, ``<1`` suppress,
            ``1`` pass, ``>1`` amplify. The binary gate is ``g = 1 − gate_closed``.

    Returns:
        ``gain * signal``.
    """
    if isinstance(gain, torch.Tensor) and gain.shape != signal.shape:
        raise ValueError(
            f"apply_gain shape mismatch: signal {tuple(signal.shape)} vs "
            f"gain {tuple(gain.shape)}"
        )
    return gain * signal
