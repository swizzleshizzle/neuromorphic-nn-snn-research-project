"""Models for week-6 MNIST optimization (experiment 009).

Contains:
- ``encode_rate``: shape-generic rate encoder, works for both ``[B, 784]``
  (MLP input) and ``[B, 1, 28, 28]`` (CNN input).
- ``FeedforwardSNN``: lifted verbatim from week-5 ``run2.py`` so this
  experiment is self-contained.
- ``SpikingCNN``: added in Task 2.
"""

from __future__ import annotations

import snntorch as snn
import torch
import torch.nn as nn


def encode_rate(data: torch.Tensor, num_steps: int, gain: float = 1.0) -> torch.Tensor:
    """Rate-code an input tensor as Bernoulli spike trains.

    Generalized over input rank: returns ``[num_steps, *data.shape]`` with
    values in ``{0.0, 1.0}``. Works for ``[B, 784]`` and ``[B, 1, 28, 28]``
    alike (the week-5 version's ``.expand(num_steps, -1, -1)`` only handled
    2D input).

    Args:
        data: tensor of any rank, values clipped to ``[0, 1]`` after gain.
        num_steps: time steps to simulate.
        gain: multiplier on values before clipping (probability scaling).

    Returns:
        ``[num_steps, *data.shape]`` float tensor with 0/1 entries.
    """
    probs = torch.clamp(data * gain, min=0.0, max=1.0)
    probs_expanded = probs.unsqueeze(0).expand(num_steps, *([-1] * probs.ndim))
    spikes = (torch.rand_like(probs_expanded) < probs_expanded).float()
    return spikes


class FeedforwardSNN(nn.Module):
    """Three-layer feedforward spiking NN. Lifted verbatim from week 5.

    Architecture: ``num_inputs -> hidden_dims[0] -> hidden_dims[1] -> num_outputs``,
    each Linear layer followed by ``snn.Leaky``.
    """

    def __init__(
        self,
        num_inputs: int = 784,
        hidden_dims: tuple[int, int] = (1000, 1000),
        num_outputs: int = 10,
        beta: float = 0.95,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 25,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(num_inputs, hidden_dims[0])
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)
        self.fc3 = nn.Linear(hidden_dims[1], num_outputs)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

    def forward(self, spk_in: torch.Tensor):
        """``spk_in``: ``[num_steps, B, num_inputs]``.
        Returns ``(spk_out, mem_out)`` each ``[num_steps, B, num_outputs]``."""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk_out_rec, mem_out_rec = [], []
        for step in range(self.num_steps):
            cur1 = self.fc1(spk_in[step]); spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1);          spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc3(spk2);          spk3, mem3 = self.lif3(cur3, mem3)
            spk_out_rec.append(spk3)
            mem_out_rec.append(mem3)
        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)


class SpikingCNN(nn.Module):
    """Three-layer spiking CNN for MNIST.

    Topology (hardcoded — see design doc §3):
        Conv2d(1, 16, k=5, padding=2)  -> MaxPool2d(2) -> snn.Leaky    # 28x28 -> 14x14
        Conv2d(16, 32, k=5, padding=2) -> MaxPool2d(2) -> snn.Leaky    # 14x14 -> 7x7
        Flatten                                                          # -> 32*7*7 = 1568
        Linear(1568, num_outputs)                  -> snn.Leaky          # output
    """

    def __init__(
        self,
        num_outputs: int = 10,
        beta: float = 0.95,
        threshold: float = 1.0,
        reset_mechanism: str = "subtract",
        num_steps: int = 25,
    ):
        super().__init__()
        self.num_steps = num_steps

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

        self.fc = nn.Linear(32 * 7 * 7, num_outputs)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold,
                              reset_mechanism=reset_mechanism)

    def forward(self, spk_in: torch.Tensor):
        """``spk_in``: ``[num_steps, B, 1, 28, 28]``.
        Returns ``(spk_out, mem_out)`` each ``[num_steps, B, num_outputs]``."""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk_out_rec, mem_out_rec = [], []
        for step in range(self.num_steps):
            cur1 = self.pool1(self.conv1(spk_in[step]))     # [B, 16, 14, 14]
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.pool2(self.conv2(spk1))             # [B, 32, 7, 7]
            spk2, mem2 = self.lif2(cur2, mem2)
            cur3 = self.fc(spk2.flatten(start_dim=1))       # [B, num_outputs]
            spk3, mem3 = self.lif3(cur3, mem3)
            spk_out_rec.append(spk3)
            mem_out_rec.append(mem3)
        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
