"""``ThalamicRouter`` — selection + gating control region (spec §2.5).

A small, hand-derived control region (no training) with two internal stages:

- **Stage A — Selection (basal-ganglia-like):** a lateral-inhibition WTA over the
  incoming utilities picks the winning channel. The input drive is sub-threshold
  for a single spike, so a channel is selected only when its utility is
  *sustained* — this is the **do-nothing floor** (weak/ambiguous utilities select
  nothing).
- **Stage B — Gating (thalamus-like):** one gate neuron per channel, **tonically
  firing = channel closed**. Stage A's winner inhibits its gate neuron, silencing
  it and **disinhibiting** (opening) that channel only. ``mode="off"`` removes the
  tonic drive so every channel is open.

The output is the **gate-closed control line** ``[T, B, N_channels]`` (1 = closed,
inhibitory). It carries no content — apply it to a pathway with
:func:`neuromorphic.connections.gating.apply_gate`.
"""

from __future__ import annotations

import snntorch as snn
import torch

from neuromorphic.regions.base_region import BrainRegion


class ThalamicRouter(BrainRegion):
    """Two-stage selection + gating router.

    Args:
        n_actions: number of channels (action/route options).
        beta, threshold, reset_mechanism: Leaky neuron params.
        num_steps: inference window ``T``.
        input_gain: Stage-A drive scale per input spike.
        select_bias: constant inhibitory drive on Stage A — the **do-nothing
            floor**. A channel selects only when its mean utility drive exceeds
            this bias, so weak/ambiguous utilities accumulate nothing and the
            router withholds. Floor rate ≈ ``select_bias / input_gain``.
        inhibition: Stage-A lateral (WTA) inhibition strength.
        tonic_drive: Stage-B baseline drive that keeps gates closed.
        gate_inhibition: how strongly a selection silences its gate (disinhibition).
        mode: ``"tonic"`` (default; channels closed unless selected) or ``"off"``
            (no tonic drive; all channels open).
    """

    def __init__(
        self,
        n_actions: int = 4,
        beta: float = 0.9,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 32,
        input_gain: float = 0.7,
        select_bias: float = 0.15,
        inhibition: float = 3.0,
        tonic_drive: float = 1.5,
        gate_inhibition: float = 3.0,
        mode: str = "tonic",
    ):
        super().__init__(name="thalamic_router", n_neurons=2 * n_actions)
        if mode not in ("tonic", "off"):
            raise ValueError(f"mode must be 'tonic' or 'off', got {mode!r}")
        self.n_actions = n_actions
        self.num_steps = num_steps
        self.input_gain = input_gain
        self.select_bias = select_bias
        self.gate_inhibition = gate_inhibition
        self.mode = mode

        # Stage A: lateral inhibition (negative off-diagonal, zero self).
        self.register_buffer("w_inh", -inhibition * (1.0 - torch.eye(n_actions)))
        self.lif_select = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)
        # Stage B: gate neurons.
        self.lif_gate = snn.Leaky(beta=beta, threshold=threshold, reset_mechanism=reset_mechanism)

        self.mem_select: torch.Tensor | None = None
        self.sel_prev: torch.Tensor | None = None
        self.mem_gate: torch.Tensor | None = None

    @staticmethod
    def open_mask(gate_closed: torch.Tensor) -> torch.Tensor:
        """Convert gate-closed control lines to an open mask (1 = open)."""
        return 1.0 - gate_closed

    def reset(self, batch_size: int | None = None, device: torch.device | None = None) -> None:
        self.mem_select = self.lif_select.init_leaky()
        self.mem_gate = self.lif_gate.init_leaky()
        self.sel_prev = (
            torch.zeros(batch_size, self.n_actions) if batch_size is not None else None
        )

    def get_state(self) -> dict[str, torch.Tensor]:
        names = ("mem_select", "sel_prev", "mem_gate")
        return {n: getattr(self, n) for n in names if getattr(self, n) is not None}

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """``input_spikes`` utilities ``[T, B, N_actions]`` → gate-closed ``[T, B, N_actions]``."""
        if input_spikes.ndim != 3:
            raise ValueError(
                f"ThalamicRouter expects [T, B, N_actions], got {tuple(input_spikes.shape)}"
            )
        if input_spikes.shape[-1] != self.n_actions:
            raise ValueError(
                f"expected N_actions={self.n_actions}, got {input_spikes.shape[-1]}"
            )

        B = input_spikes.shape[1]
        self.reset(batch_size=B)
        tonic_drive = 1.5 if self.mode == "tonic" else 0.0
        gates = []
        for t in range(input_spikes.shape[0]):
            # Stage A — selection (WTA) with a constant do-nothing-floor bias.
            drive = (
                self.input_gain * input_spikes[t]
                - self.select_bias
                + (self.sel_prev @ self.w_inh.t())
            )
            spk_sel, self.mem_select = self.lif_select(drive, self.mem_select)
            self.sel_prev = spk_sel
            self._record("select", spk_sel)

            # Stage B — gating: tonic drive (closed) minus disinhibition from selection.
            cur_gate = tonic_drive - self.gate_inhibition * spk_sel
            spk_gate, self.mem_gate = self.lif_gate(cur_gate, self.mem_gate)
            self._record("gate", spk_gate)
            gates.append(spk_gate)

        return torch.stack(gates, dim=0)
