"""``MotorCortex`` — winner-take-all action selection (spec §2.4).

Turns a vector of candidate action utilities into a near one-hot action via
**lateral inhibition**: a single output layer of ``N_actions`` Leaky neurons where
every neuron inhibits the others, so the strongest action suppresses its
competitors and one winner emerges. The winning action is read out by spike count
over the ``T``-window.

This is the WTA-focused bring-up (build-order step 2): the input drive is
action-aligned (identity-initialised) so the highest-utility action wins. The
spec's learned decompression stack is deferred until the closed loop can train it.
An ``ach_gain`` knob scales the inhibition — the hook for the future ACh
neuromodulatory bus (build-order step 6), not yet wired.
"""

from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn

from neuromorphic.regions.base_region import BrainRegion


class MotorCortex(BrainRegion):
    """WTA action-selection region.

    Args:
        n_actions: number of discrete actions (=4 for grid-world).
        beta, threshold, reset_mechanism: Leaky neuron params (week-7 locked config).
        num_steps: inference window ``T``.
        input_gain: scale of the action-aligned (identity) input drive.
        inhibition: strength of the lateral (off-diagonal) inhibition.
        ach_gain: multiplies the inhibition — ACh-precision hook (default 1.0).
        bus: optional :class:`~neuromorphic.neuromod.NeuromodBus`; its ``ach`` level
            further scales the lateral inhibition (higher ACh → sharper WTA).
    """

    def __init__(
        self,
        n_actions: int = 4,
        beta: float = 0.9,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 32,
        input_gain: float = 2.0,
        inhibition: float = 3.0,
        ach_gain: float = 1.0,
        bus=None,
    ):
        super().__init__(name="motor_cortex", n_neurons=n_actions)
        self.n_actions = n_actions
        self.num_steps = num_steps
        self.ach_gain = ach_gain
        self.bus = bus  # optional NeuromodBus; its ACh scales the WTA precision

        # Action-aligned input drive: utility i drives output neuron i.
        self.fc_in = nn.Linear(n_actions, n_actions)
        with torch.no_grad():
            self.fc_in.weight.copy_(torch.eye(n_actions) * input_gain)
            self.fc_in.bias.zero_()

        # Lateral inhibition: negative off-diagonal, zero self-connection.
        w_inh = -inhibition * (1.0 - torch.eye(n_actions))
        self.register_buffer("w_inh", w_inh)

        self.lif_out = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
        self.mem_out: torch.Tensor | None = None
        self.spk_prev: torch.Tensor | None = None

    @staticmethod
    def winner(spk_out: torch.Tensor) -> torch.Tensor:
        """Winning action per batch element: argmax of spike count over time. → ``[B]``."""
        return spk_out.sum(dim=0).argmax(dim=-1)

    def reset(self, batch_size: int | None = None, device: torch.device | None = None) -> None:
        self.mem_out = self.lif_out.init_leaky()
        self.spk_prev = (
            torch.zeros(batch_size, self.n_actions) if batch_size is not None else None
        )

    def get_state(self) -> dict[str, torch.Tensor]:
        state = {}
        if self.mem_out is not None:
            state["mem_out"] = self.mem_out
        if self.spk_prev is not None:
            state["spk_prev"] = self.spk_prev
        return state

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """``input_spikes`` ``[T, B, N_actions]`` → near one-hot action ``[T, B, N_actions]``."""
        if input_spikes.ndim != 3:
            raise ValueError(
                f"MotorCortex expects [T, B, N_actions], got {tuple(input_spikes.shape)}"
            )
        if input_spikes.shape[-1] != self.n_actions:
            raise ValueError(
                f"expected N_actions={self.n_actions}, got {input_spikes.shape[-1]}"
            )

        B = input_spikes.shape[1]
        self.reset(batch_size=B)
        ach = self.bus.ach if self.bus is not None else 1.0  # ACh sharpens the WTA
        out = []
        for t in range(input_spikes.shape[0]):
            drive = self.fc_in(input_spikes[t])               # [B, N_actions]
            inhib = self.ach_gain * ach * (self.spk_prev @ self.w_inh.t())
            spk, self.mem_out = self.lif_out(drive + inhib, self.mem_out)
            self.spk_prev = spk
            self._record("action", spk)
            out.append(spk)
        return torch.stack(out, dim=0)
