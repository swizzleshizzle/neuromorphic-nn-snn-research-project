"""``Hippocampus`` — recurrent attractor memory (spec §2.2).

A single recurrent population that **stores** a content pattern, **holds** it across
a delay with no input (attractor persistence), and supports **recall** read-out.
Encoding scheme (resolved from spec §5.1): a recurrent attractor with a **one-shot
Hebbian imprint** — at store time an outer-product (Hopfield-style) rule writes the
sparse stored pattern into the recurrent weights as a stable fixed point, so the
pattern self-sustains once input is removed.

Gating: in the full loop the store pathway (3, PFC→Hippo) and recall pathway
(4, Hippo→PFC) are opened by the Thalamic Router via
:func:`neuromorphic.connections.gating.apply_gate`. The region itself runs the
attractor dynamics; the bring-up drives content only during the (gated-open) store
window and zero during the delay.
"""

from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn

from neuromorphic.regions.base_region import BrainRegion


class Hippocampus(BrainRegion):
    """Attractor memory: store → hold → recall.

    Args:
        content_dim: PFC content code width (in and out).
        n_neurons: recurrent attractor population size.
        sparsity: fraction of neurons active in a stored pattern.
        beta, threshold, reset_mechanism: Leaky neuron params.
        num_steps: inference window ``T``.
        input_gain: scale of the store-time content drive.
        recurrent_gain: strength of the imprinted attractor recurrence (hold).
        seed: RNG seed for reproducible afferent/read-out weights.
    """

    def __init__(
        self,
        content_dim: int = 64,
        n_neurons: int = 150,
        sparsity: float = 0.2,
        beta: float = 0.9,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 32,
        input_gain: float = 3.0,
        recurrent_gain: float = 2.0,
        seed: int | None = None,
    ):
        super().__init__(name="hippocampus", n_neurons=n_neurons)
        self.content_dim = content_dim
        self.sparsity = sparsity
        self.input_gain = input_gain
        self.recurrent_gain = recurrent_gain
        self.num_steps = num_steps

        if seed is not None:
            torch.manual_seed(seed)
        self.fc_in = nn.Linear(content_dim, n_neurons)    # store afferent (pathway 3)
        self.fc_out = nn.Linear(n_neurons, content_dim)   # recall read-out (pathway 4)
        self.lif = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
        self.lif_out = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)

        self.register_buffer("W_rec", torch.zeros(n_neurons, n_neurons))
        self._stored_pattern: torch.Tensor | None = None
        self._stored_patterns: list[torch.Tensor] = []
        self.mem: torch.Tensor | None = None
        self.mem_out: torch.Tensor | None = None
        self.spk_prev: torch.Tensor | None = None

    def store(self, content: torch.Tensor) -> torch.Tensor:
        """One-shot Hebbian imprint of a content code as an attractor fixed point.

        Args:
            content: ``[B, content_dim]`` content code (batch averaged → one pattern).

        Returns:
            the stored sparse binary pattern ``p`` of shape ``[n_neurons]``.
        """
        with torch.no_grad():
            drive = self.fc_in(content).mean(dim=0)  # [n_neurons]
            k = max(1, int(self.sparsity * self.n_neurons))
            p = torch.zeros(self.n_neurons)
            p[torch.topk(drive, k).indices] = 1.0
            s = 2.0 * p - 1.0  # bipolar {-1, +1}
            w = torch.outer(s, s) / self.n_neurons
            w.fill_diagonal_(0.0)
            # Accumulate, do not assign. Assigning kept only the most recent pattern,
            # which collapsed recall to a near-constant code (measured cosine 0.998).
            self.W_rec = self.W_rec + self.recurrent_gain * w
            self._stored_pattern = p
            self._stored_patterns.append(p)
        return p

    @property
    def n_stored(self) -> int:
        """How many patterns are currently imprinted."""
        return len(self._stored_patterns)

    def clear(self) -> None:
        """Forget everything. Required for episodic memory: without it, imprints
        persist across episodes and accumulate into an uninterpretable mixture."""
        with torch.no_grad():
            self.W_rec = torch.zeros_like(self.W_rec)
            self._stored_pattern = None
            self._stored_patterns = []

    def familiarity(self, content: torch.Tensor) -> torch.Tensor:
        """``[B, content_dim]`` -> ``[B]`` Hopfield field alignment, one scalar per item.

        High when the content's sparse pattern sits near a stored attractor, so it
        answers "have I been here?". Reuses ``W_rec`` rather than a lookup table, so
        familiarity stays a property of the attractor. Zeros when nothing is stored.
        """
        with torch.no_grad():
            drive = self.fc_in(content)                       # [B, n_neurons]
            k = max(1, int(self.sparsity * self.n_neurons))
            idx = torch.topk(drive, k, dim=1).indices
            p = torch.zeros_like(drive)
            p.scatter_(1, idx, 1.0)
            s = 2.0 * p - 1.0
            return ((s @ self.W_rec) * s).sum(dim=1) / self.n_neurons

    def reset(self, batch_size: int | None = None, device: torch.device | None = None) -> None:
        self.mem = self.lif.init_leaky()
        self.mem_out = self.lif_out.init_leaky()
        self.spk_prev = (
            torch.zeros(batch_size, self.n_neurons) if batch_size is not None else None
        )

    def get_state(self) -> dict[str, torch.Tensor]:
        names = ("mem", "mem_out", "spk_prev")
        return {n: getattr(self, n) for n in names if getattr(self, n) is not None}

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """``input_spikes`` content ``[T, B, content_dim]`` → recall code ``[T, B, content_dim]``.

        Content is present during the store window and zero during the delay; the
        imprinted recurrence holds the pattern across the gap.
        """
        if input_spikes.ndim != 3:
            raise ValueError(
                f"Hippocampus expects [T, B, content_dim], got {tuple(input_spikes.shape)}"
            )
        if input_spikes.shape[-1] != self.content_dim:
            raise ValueError(
                f"expected content_dim={self.content_dim}, got {input_spikes.shape[-1]}"
            )

        B = input_spikes.shape[1]
        self.reset(batch_size=B)
        readouts = []
        for t in range(input_spikes.shape[0]):
            cur_in = self.input_gain * self.fc_in(input_spikes[t])
            cur_rec = (2.0 * self.spk_prev - 1.0) @ self.W_rec.t()  # Hopfield field
            spk, self.mem = self.lif(cur_in + cur_rec, self.mem)
            self.spk_prev = spk
            self._record("population", spk)

            spk_out, self.mem_out = self.lif_out(self.fc_out(spk), self.mem_out)
            self._record("recall", spk_out)
            readouts.append(spk_out)

        return torch.stack(readouts, dim=0)
