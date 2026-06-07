"""``Prefrontal`` — planning / integration region (spec §2.3).

Integrates the sensory concept code into a vector of action utilities. Structure:
a **recurrent state-hold** population (``snn.RLeaky``, holds context across the
window) feeding a **feedforward transform** (``snn.Leaky``) and a small utility
read-out (``snn.Leaky``, ``N_actions``).

Per the agreed composition model, the region **owns its afferent** as a
:class:`~neuromorphic.connections.projection.Projection`: it takes the sensory
concept *spikes* (uniform ``BrainRegion`` contract) and the Projection turns them
into the delayed afferent current that drives the state-hold layer (pathway 2,
``Δ=1``). For the open loop there is no hippocampal recall input and no gating yet.
"""

from __future__ import annotations

import math

import snntorch as snn
import torch
import torch.nn as nn

from neuromorphic.connections.projection import Projection
from neuromorphic.regions.base_region import BrainRegion


class Prefrontal(BrainRegion):
    """Recurrent planning region: concept code → action utilities.

    Args:
        concept_dim: sensory concept input width.
        n_state: recurrent state-hold population size (RLeaky).
        n_transform: feedforward transform population size (Leaky).
        n_actions: action-utility output width.
        beta, threshold, reset_mechanism: neuron params (week-7 locked config).
        num_steps: inference window ``T``.
        sparsity: afferent Projection density (pathway 2 is dense → 1.0).
        delay: afferent transmission delay ``Δ`` (pathway 2 → 1).
        weight_gain: excitability knob scaling the ``1/sqrt(fan_in)`` weight inits.
            Kept moderate (≈2.0) so the untrained utility read-out stays in the
            *responsive* regime — too high and the favoured action saturates
            (fires every step) and washes out all upstream concept selectivity.
        seed: RNG seed for reproducible weights (afferent, recurrent, transform).
    """

    def __init__(
        self,
        concept_dim: int = 64,
        n_state: int = 100,
        n_transform: int = 50,
        n_actions: int = 4,
        beta: float = 0.9,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 32,
        sparsity: float = 1.0,
        delay: int = 1,
        weight_gain: float = 2.0,
        seed: int | None = None,
    ):
        super().__init__(name="prefrontal", n_neurons=n_state + n_transform)
        self.num_steps = num_steps
        self.concept_dim = concept_dim
        self.n_state = n_state

        # Seed the global RNG so snnTorch's internal (RLeaky recurrent, Linear
        # default) inits are reproducible alongside our explicit inits.
        if seed is not None:
            torch.manual_seed(seed)

        # Afferent (pathway 2): concept spikes -> delayed state-hold current.
        self.afferent = Projection(
            n_source=concept_dim,
            n_target=n_state,
            sparsity=sparsity,
            delay=delay,
            weight_scale=weight_gain / math.sqrt(concept_dim),
            seed=seed,
        )

        self.lif_state = snn.RLeaky(
            beta=beta,
            all_to_all=True,
            linear_features=n_state,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        )
        self.fc_transform = nn.Linear(n_state, n_transform)
        self.lif_transform = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
        self.fc_utility = nn.Linear(n_transform, n_actions)
        self.lif_utility = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)

        self._init_weights(weight_gain)

        self.spk_state: torch.Tensor | None = None
        self.mem_state: torch.Tensor | None = None
        self.mem_transform: torch.Tensor | None = None
        self.mem_utility: torch.Tensor | None = None

    def _init_weights(self, gain: float) -> None:
        with torch.no_grad():
            for layer in (self.fc_transform, self.fc_utility):
                fan_in = layer.weight.shape[1]
                layer.weight.copy_(torch.randn(layer.weight.shape) * (gain / math.sqrt(fan_in)))
                layer.bias.zero_()

    def reset(self, batch_size: int | None = None, device: torch.device | None = None) -> None:
        self.spk_state, self.mem_state = self.lif_state.init_rleaky()
        self.mem_transform = self.lif_transform.init_leaky()
        self.mem_utility = self.lif_utility.init_leaky()

    def get_state(self) -> dict[str, torch.Tensor]:
        names = ("spk_state", "mem_state", "mem_transform", "mem_utility")
        return {n: getattr(self, n) for n in names if getattr(self, n) is not None}

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """``input_spikes`` concept code ``[T, B, concept_dim]`` → utilities ``[T, B, N_actions]``."""
        if input_spikes.ndim != 3:
            raise ValueError(
                f"Prefrontal expects [T, B, concept_dim], got {tuple(input_spikes.shape)}"
            )
        if input_spikes.shape[-1] != self.concept_dim:
            raise ValueError(
                f"expected concept_dim={self.concept_dim}, got {input_spikes.shape[-1]}"
            )

        afferent_current = self.afferent(input_spikes)  # [T, B, n_state], delayed
        self.reset()
        utilities = []
        for t in range(input_spikes.shape[0]):
            self.spk_state, self.mem_state = self.lif_state(
                afferent_current[t], self.spk_state, self.mem_state
            )
            self._record("state", self.spk_state)

            spk_t, self.mem_transform = self.lif_transform(
                self.fc_transform(self.spk_state), self.mem_transform
            )
            self._record("transform", spk_t)

            spk_u, self.mem_utility = self.lif_utility(
                self.fc_utility(spk_t), self.mem_utility
            )
            self._record("utility", spk_u)
            utilities.append(spk_u)

        return torch.stack(utilities, dim=0)
