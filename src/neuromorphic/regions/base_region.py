"""``BrainRegion`` — the standard interface every brain region implements.

A region maps **input spikes ``[T, B, N_in]`` → output spikes ``[T, B, N_out]``**
over the inference window ``T``. Subclasses are ``nn.Module``s (so parameters and
device placement work) and must override :meth:`forward`, :meth:`reset`, and
:meth:`get_state`.

Built-in logging hooks feed the viz toolkit: a subclass calls ``self._record(key,
value)`` once per time step inside its forward loop (passing a ``[B, N]`` tensor),
and :meth:`get_recording` stacks those into the canonical ``[T, B, N]`` contract
consumed by ``spike_raster`` / ``population_rate`` / ``psth``. Recording is
**opt-in** (off by default) so the hot loop carries no overhead in production.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BrainRegion(nn.Module, ABC):
    """Abstract base class for a spiking brain region.

    Args:
        name: human-readable region name (used in logs / viz titles).
        n_neurons: total neuron count for the region.
    """

    def __init__(self, name: str, n_neurons: int):
        super().__init__()
        self.name = name
        self.n_neurons = n_neurons
        self._recording_enabled = False
        self._records: dict[str, list[torch.Tensor]] = {}

    # ------------------------------------------------------------------ #
    # Contract (must be implemented by every region)
    # ------------------------------------------------------------------ #
    @abstractmethod
    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """Process ``input_spikes`` ``[T, B, N_in]`` → output ``[T, B, N_out]``."""

    @abstractmethod
    def reset(
        self,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """(Re)initialize the region's neuron states (membranes, recurrent spikes)."""

    @abstractmethod
    def get_state(self) -> dict[str, torch.Tensor]:
        """Return the region's current dynamic state as a dict of tensors."""

    # ------------------------------------------------------------------ #
    # Logging hooks for the viz toolkit (opt-in)
    # ------------------------------------------------------------------ #
    def enable_recording(self, flag: bool = True) -> None:
        """Turn per-step recording on/off. Disabling also clears the buffer."""
        self._recording_enabled = flag
        if not flag:
            self.clear_recording()

    def clear_recording(self) -> None:
        """Drop all recorded per-step tensors."""
        self._records = {}

    def _record(self, key: str, value: torch.Tensor) -> None:
        """Append a ``[B, N]`` tensor for ``key`` at the current time step.

        No-op unless recording is enabled. Stored detached so recordings never
        hold the autograd graph alive.
        """
        if not self._recording_enabled:
            return
        self._records.setdefault(key, []).append(value.detach().clone())

    def get_recording(
        self, key: str | None = None
    ) -> torch.Tensor | dict[str, torch.Tensor] | None:
        """Return recorded signals stacked to ``[T, B, N]``.

        Args:
            key: a single signal name, or ``None`` for all signals.

        Returns:
            For a ``key``: the ``[T, B, N]`` tensor, or ``None`` if nothing was
            recorded under it. For ``None``: a dict mapping every recorded key to
            its ``[T, B, N]`` tensor (empty dict if nothing was recorded).
        """
        if key is not None:
            steps = self._records.get(key)
            if not steps:
                return None
            return torch.stack(steps, dim=0)
        return {k: torch.stack(v, dim=0) for k, v in self._records.items() if v}
