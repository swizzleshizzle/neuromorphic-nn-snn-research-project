"""``NeuromodBus`` — the global neuromodulatory bus (spec §1).

Not a region: a one-to-all broadcast of two scalars read by any region as global
parameters.

- **dopamine** — reward / learning-enable. Gates plasticity (``learning_enabled``).
  No plasticity exists yet, so today it is a broadcast signal + hook for future STDP.
- **ACh** — gain / precision. Sharpens competition; e.g. the Motor WTA reads it to
  tighten winner-take-all (higher ACh → crisper selection).

There is no addressing — every region sees the same values.
"""

from __future__ import annotations


class NeuromodBus:
    """Broadcast holder for dopamine and ACh.

    Args:
        dopamine: reward / learning-enable level (default 0.0).
        ach: acetylcholine gain / precision level (default 1.0 = neutral).
        learning_threshold: dopamine level at/above which learning is enabled.
    """

    def __init__(self, dopamine: float = 0.0, ach: float = 1.0, learning_threshold: float = 0.5):
        self.dopamine = dopamine
        self.ach = ach
        self.learning_threshold = learning_threshold

    def set(self, dopamine: float | None = None, ach: float | None = None) -> None:
        """Update one or both modulators (partial update; unset stays put)."""
        if dopamine is not None:
            self.dopamine = dopamine
        if ach is not None:
            self.ach = ach

    def reset(self) -> None:
        """Restore neutral levels (dopamine 0.0, ACh 1.0)."""
        self.dopamine = 0.0
        self.ach = 1.0

    @property
    def learning_enabled(self) -> bool:
        """Whether dopamine is high enough to enable plasticity."""
        return self.dopamine >= self.learning_threshold
