"""Gating primitive — apply a router's inhibitory control lines to a pathway.

The Thalamic Router emits **gate-closed** signals (1 = channel tonically inhibited
/ closed, 0 = disinhibited / open). ``apply_gate`` releases a content pathway
through those gates: open channels pass, closed channels are blocked. Reused for
the gated pathways 3 (store), 4 (recall), and 5 (action-enable).
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
