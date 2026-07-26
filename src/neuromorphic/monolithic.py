"""``MonolithicBrain`` - the unregionalized control for the Phase-3 regionalization question.

One flat spiking stack, neuron-matched to the five-region ``Brain`` and frozen at random
init, exposing the same duck-typed surface ``reinforce.py`` consumes. Only the topology
differs: no hippocampus, no prefrontal, no router, no motor. If the regionalized arm beats
this at matched neuron count, matched head, matched seeds and matched protocol, the gap is
attributable to regionalization.

There is deliberately no motor readout: in v1 the action comes from the policy head, not
from the brain (ADR-0001 Amendment 1), so a motor region would be dead weight here.
"""

from __future__ import annotations

import numpy as np
import torch

from neuromorphic.encoders import grid_encoder
from neuromorphic.regions.sensory_cortex import SensoryCortex


class MonolithicBrain:
    """A single feedforward spiking stack sized to a whole ``Brain``'s neuron budget.

    Args:
        n_obs: encoder output width (144 for a 2x2 cube).
        n_actions: action-space width, for head sizing.
        total_neurons: the matching budget, normally ``Brain.n_neurons``.
        content: concept width. Kept equal to the Brain's so the policy head is
            parameter-identical across arms.
        num_steps: inference window ``T``.
        seed: RNG seed for reproducible weight init.
        obs_width: raw observation width (24 for a 2x2 cube, 4 for the grid).
        encoder: picklable encoder callable; defaults to the grid encoder.
        weight_gain: excitability knob. 5.0 is correct for both the grid's 2-hot code
            and the cube's 24-hot code (measured); lowering it collapses the code.
    """

    def __init__(
        self,
        n_obs: int,
        n_actions: int,
        total_neurons: int,
        *,
        content: int = 64,
        num_steps: int = 32,
        seed: int = 0,
        obs_width: int = 4,
        encoder=None,
        weight_gain: float = 5.0,
    ):
        if total_neurons <= content:
            raise ValueError(
                f"total_neurons ({total_neurons}) must exceed content ({content}); "
                "there would be no hidden layer left"
            )
        self.content = content
        self.n_actions = n_actions
        self.T = num_steps
        self.n_obs = n_obs
        self.obs_width = obs_width
        self.n_neurons = total_neurons
        self._encoder = encoder if encoder is not None else grid_encoder(5)

        # One flat stack spending the entire budget: hidden + concept == total_neurons.
        self.stack = SensoryCortex(
            n_obs=n_obs,
            hidden=total_neurons - content,
            concept=content,
            num_steps=num_steps,
            weight_gain=weight_gain,
            seed=seed,
        )

    def _to_obs_tensor(self, obs) -> torch.Tensor:
        """Coerce an observation to a ``[B, obs_width]`` int tensor."""
        arr = np.asarray(obs)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != self.obs_width:
            raise ValueError(
                f"obs must be [{self.obs_width}] or [B, {self.obs_width}]; got {arr.shape}"
            )
        return torch.as_tensor(arr, dtype=torch.long)

    def step(
        self,
        obs,
        *,
        store: bool = False,
        recall: bool = False,
        record: bool = False,
        generator: torch.Generator | None = None,
    ) -> dict:
        """One decision window: obs -> concept.

        ``store``, ``recall`` and ``record`` are accepted for interface parity with
        ``Brain`` and ignored: there is no hippocampus to store into and no separate
        regions to record. The action comes from the policy head, as in v1.
        """
        obs_t = self._to_obs_tensor(obs)
        obs_spikes = self._encoder(obs_t, T=self.T, generator=generator)
        concept = self.stack(obs_spikes)  # [T, B, content]
        return {"concept": concept, "obs_spikes": obs_spikes}
