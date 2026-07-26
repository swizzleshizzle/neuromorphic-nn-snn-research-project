"""``SensoryCortex`` — the first concrete brain region (spec §2.1).

Encodes a grid-world observation into a compressed *concept code*. Two-stage
feedforward Leaky compression ``N_obs → hidden → concept`` (Spaun's visual
hierarchy collapsed to two stages at our scale). Pure feedforward — no recurrence.

``encode_gridworld`` is kept **separate** from the region so the region honours the
uniform ``BrainRegion`` contract (spikes in → spikes out): the encoder turns an
agent/goal observation into rate-coded (Poisson) input spikes, which are then fed
to the region.
"""

from __future__ import annotations

import math

import snntorch as snn
import torch
import torch.nn as nn

from neuromorphic.regions.base_region import BrainRegion


def encode_gridworld(
    obs: torch.Tensor,
    grid_n: int,
    T: int = 32,
    max_rate: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Rate-encode an agent+goal grid observation into Poisson spikes.

    Args:
        obs: ``[B, 4]`` integer tensor ``(agent_x, agent_y, goal_x, goal_y)``,
            each coordinate in ``[0, grid_n)``.
        grid_n: grid side length ``N`` (so each one-hot is ``N*N`` wide).
        T: number of time steps in the inference window.
        max_rate: per-step spike probability for an active (one-hot) cell.
        generator: RNG for reproducible Poisson sampling (``None`` = global RNG).

    Returns:
        ``[T, B, N_obs]`` binary spikes, ``N_obs = 2 * grid_n * grid_n``
        (agent one-hot ⊕ goal one-hot).
    """
    if obs.ndim != 2 or obs.shape[1] != 4:
        raise ValueError(f"encode_gridworld expects obs [B, 4], got {tuple(obs.shape)}")
    if obs.min() < 0 or obs.max() >= grid_n:
        raise ValueError(f"obs coordinates must be in [0, {grid_n}), got {obs.tolist()}")

    B = obs.shape[0]
    cells = grid_n * grid_n
    n_obs = 2 * cells

    ax, ay, gx, gy = obs[:, 0], obs[:, 1], obs[:, 2], obs[:, 3]
    agent_idx = ay * grid_n + ax              # flatten (x, y) row-major on y
    goal_idx = cells + (gy * grid_n + gx)     # goal lives in the second half

    rate = torch.zeros(B, n_obs)
    rows = torch.arange(B)
    rate[rows, agent_idx] = max_rate
    rate[rows, goal_idx] = max_rate

    probs = rate.unsqueeze(0).expand(T, B, n_obs).contiguous()
    return torch.bernoulli(probs, generator=generator)


def encode_cube(
    obs: torch.Tensor,
    cube_n: int = 2,
    n_colors: int = 6,
    T: int = 32,
    max_rate: float = 0.5,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Rate-encode a cube facelet observation into Poisson spikes (mirrors encode_gridworld).

    Args:
        obs: ``[B, 6*cube_n**2]`` integer facelet colors, each in ``[0, n_colors)``.
        cube_n: cube side length (2 for the Pocket Cube; size-generic for 3+).
        n_colors: number of sticker colors (6).
        T: number of time steps in the inference window.
        max_rate: per-step spike probability for an active one-hot cell.
        generator: RNG for reproducible Poisson sampling (``None`` = global RNG).

    Returns:
        ``[T, B, 6*cube_n**2*n_colors]`` binary spikes (one-hot facelet-color, Poisson-rated).
    """
    n_facelets = 6 * cube_n * cube_n
    if obs.ndim != 2 or obs.shape[1] != n_facelets:
        raise ValueError(f"encode_cube expects obs [B, {n_facelets}], got {tuple(obs.shape)}")
    if int(obs.min()) < 0 or int(obs.max()) >= n_colors:
        raise ValueError(f"facelet colors must be in [0, {n_colors})")

    B = obs.shape[0]
    n_in = n_facelets * n_colors

    # one-hot each facelet's color: input index = facelet * n_colors + color
    idx = torch.arange(n_facelets).unsqueeze(0) * n_colors + obs.long()
    rate = torch.zeros(B, n_in)
    rate.scatter_(1, idx, max_rate)

    probs = rate.unsqueeze(0).expand(T, B, n_in).contiguous()
    return torch.bernoulli(probs, generator=generator)


class SensoryCortex(BrainRegion):
    """Feedforward spiking sensory region: ``N_obs → hidden → concept``.

    Args:
        n_obs: input dimension (encoder output width).
        hidden: first compression layer size.
        concept: concept-code (output) size.
        beta, threshold, reset_mechanism: Leaky neuron params (week-7 locked config).
        num_steps: inference window ``T``.
        weight_gain: excitability knob — scales the ``1/sqrt(fan_in)`` weight init
            so the sparse 2-hot grid code reliably drives the hierarchy (bring-up
            tuning, spec §4).
        seed: RNG seed for reproducible weight init (``None`` = default init).
    """

    def __init__(
        self,
        n_obs: int,
        hidden: int = 128,
        concept: int = 64,
        beta: float = 0.9,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 32,
        weight_gain: float = 5.0,
        seed: int | None = None,
    ):
        super().__init__(name="sensory_cortex", n_neurons=hidden + concept)
        self.num_steps = num_steps
        self.concept = concept

        self.fc1 = nn.Linear(n_obs, hidden)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
        self.fc2 = nn.Linear(hidden, concept)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)

        self._init_weights(weight_gain, seed)

        self.mem1: torch.Tensor | None = None
        self.mem2: torch.Tensor | None = None

    def _init_weights(self, gain: float, seed: int | None) -> None:
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        with torch.no_grad():
            for layer in (self.fc1, self.fc2):
                fan_in = layer.weight.shape[1]
                std = gain / math.sqrt(fan_in)
                layer.weight.copy_(torch.randn(layer.weight.shape, generator=gen) * std)
                layer.bias.zero_()

    def reset(self, batch_size: int | None = None, device: torch.device | None = None) -> None:
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

    def get_state(self) -> dict[str, torch.Tensor]:
        state = {}
        if self.mem1 is not None:
            state["mem1"] = self.mem1
        if self.mem2 is not None:
            state["mem2"] = self.mem2
        return state

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """``input_spikes`` ``[T, B, N_obs]`` → concept code ``[T, B, concept]``."""
        if input_spikes.ndim != 3:
            raise ValueError(
                f"SensoryCortex expects [T, B, N_obs], got {tuple(input_spikes.shape)}"
            )
        if input_spikes.shape[-1] != self.fc1.in_features:
            raise ValueError(
                f"expected N_obs={self.fc1.in_features}, got {input_spikes.shape[-1]}"
            )

        self.reset()
        T = input_spikes.shape[0]
        concept_out = []
        for t in range(T):
            cur1 = self.fc1(input_spikes[t])
            spk1, self.mem1 = self.lif1(cur1, self.mem1)
            self._record("hidden", spk1)

            cur2 = self.fc2(spk1)
            spk2, self.mem2 = self.lif2(cur2, self.mem2)
            self._record("concept", spk2)
            concept_out.append(spk2)

        return torch.stack(concept_out, dim=0)
