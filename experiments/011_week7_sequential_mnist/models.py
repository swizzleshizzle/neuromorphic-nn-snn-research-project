"""Models for sequential MNIST (exp 011).

Contains:
- ``row_encode``: pure helper that reshapes an MNIST batch into a
  ``[T=28, B, 28]`` sequence, one row per timestep.
- ``SequentialSNN``: added in Task 3.
"""

from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn


def row_encode(images: torch.Tensor) -> torch.Tensor:
    """Reshape an MNIST batch into a row-major time sequence.

    Args:
        images: ``[B, 1, 28, 28]`` or ``[B, 28, 28]`` tensor of pixel values.

    Returns:
        ``[28, B, 28]`` -- row index becomes time index. Pure function: no
        spike sampling, no normalization, no allocation beyond reshape.
    """
    if images.ndim == 4:
        # [B, 1, 28, 28] -> [B, 28, 28]
        images = images.squeeze(1)
    if images.ndim != 3 or images.shape[-2:] != (28, 28):
        raise ValueError(
            f"row_encode expects [B, 1, 28, 28] or [B, 28, 28], got {tuple(images.shape)}"
        )
    # [B, 28, 28] -> [28, B, 28]
    return images.permute(1, 0, 2).contiguous()


class SequentialSNN(nn.Module):
    """Sequential-MNIST SNN with a `recurrent: bool` switch.

    Architecture (both variants):
        x_t: [B, 28]                                       # one image row at time t
        cur1 = fc1(x_t)                                    # Linear(28, 128) -> [B, 128]
        spk1, mem1 = hidden(cur1, [spk1,] mem1)            # RLeaky or Leaky
        cur2 = fc2(spk1)                                   # Linear(128, 10) -> [B, 10]
        spk2, mem2 = lif_out(cur2, mem2)                   # output Leaky
        record spk2

    Returns ``spk_out`` of shape ``[28, B, 10]``. Loss is computed
    externally as ``CE(spk_out[-readout_window:].sum(dim=0), labels)``.
    """

    def __init__(
        self,
        num_inputs: int = 28,
        hidden_size: int = 128,
        num_outputs: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 28,
        readout_window: int = 4,
        recurrent: bool = False,
    ):
        super().__init__()
        if num_steps != 28:
            raise ValueError(f"SequentialSNN expects num_steps=28, got {num_steps}")
        self.num_steps = num_steps
        self.readout_window = readout_window
        self.recurrent = recurrent

        self.fc1 = nn.Linear(num_inputs, hidden_size)
        if recurrent:
            self.lif1 = snn.RLeaky(
                beta=beta,
                all_to_all=True,
                linear_features=hidden_size,
                threshold=threshold,
                reset_mechanism=reset_mechanism,
            )
        else:
            self.lif1 = snn.Leaky(
                beta=beta,
                threshold=threshold,
                reset_mechanism=reset_mechanism,
            )

        self.fc2 = nn.Linear(hidden_size, num_outputs)
        self.lif_out = snn.Leaky(
            beta=beta,
            threshold=threshold,
            reset_mechanism=reset_mechanism,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``images``: ``[B, 1, 28, 28]`` or ``[B, 28, 28]``.

        Returns ``spk_out`` of shape ``[28, B, num_outputs]``.
        """
        sequence = row_encode(images)  # [28, B, 28]

        if self.recurrent:
            spk1, mem1 = self.lif1.init_rleaky()
        else:
            mem1 = self.lif1.init_leaky()
        mem_out = self.lif_out.init_leaky()

        spk_out_rec = []
        for t in range(self.num_steps):
            cur1 = self.fc1(sequence[t])              # [B, hidden_size]
            if self.recurrent:
                spk1, mem1 = self.lif1(cur1, spk1, mem1)
            else:
                spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)                     # [B, num_outputs]
            spk_out, mem_out = self.lif_out(cur2, mem_out)
            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec, dim=0)        # [28, B, num_outputs]

    def readout(self, spk_out: torch.Tensor) -> torch.Tensor:
        """Sum the last ``readout_window`` timesteps. Returns ``[B, num_outputs]``."""
        return spk_out[-self.readout_window:].sum(dim=0)
