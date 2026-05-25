"""STDP-WTA layer for EXP-012.

A single feedforward layer (12 -> 4) with:
  - LIF dynamics on the 4 output neurons (beta decay + threshold)
  - Hard winner-take-all: at most 1 output spikes per timestep
  - Manual pair-based STDP plasticity applied under torch.no_grad()

NOT a generic STDP library. Demo-only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class STDPState:
    mem: torch.Tensor       # [n_post]  membrane potential
    x_pre: torch.Tensor     # [n_pre]   pre-synaptic trace
    y_post: torch.Tensor    # [n_post]  post-synaptic trace


class STDPLayer(nn.Module):
    """Feedforward layer with LIF + hard WTA + manual pair-STDP."""

    def __init__(
        self,
        n_pre: int = 12,
        n_post: int = 4,
        beta: float = 0.9,
        threshold: float = 1.0,
        a_plus: float = 0.005,
        a_minus: float = -0.0055,
        tau_pre_ms: float = 20.0,
        tau_post_ms: float = 20.0,
        dt_ms: float = 2.5,
        w_max: float = 1.0,
        w_init_low: float = 0.0,
        w_init_high: float = 1.0,
        seed: int = 0,
    ):
        super().__init__()
        self.n_pre = n_pre
        self.n_post = n_post
        self.beta = beta
        self.threshold = threshold
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max
        self.dt_ms = dt_ms
        self.alpha_pre = math.exp(-dt_ms / tau_pre_ms)
        self.alpha_post = math.exp(-dt_ms / tau_post_ms)

        gen = torch.Generator().manual_seed(seed)
        W = torch.empty(n_pre, n_post).uniform_(w_init_low, w_init_high, generator=gen)
        self.register_buffer("W", W)

    def init_state(self, device: torch.device | str = "cpu") -> STDPState:
        return STDPState(
            mem=torch.zeros(self.n_post, device=device),
            x_pre=torch.zeros(self.n_pre, device=device),
            y_post=torch.zeros(self.n_post, device=device),
        )

    @torch.no_grad()
    def forward_step(
        self,
        spk_pre: torch.Tensor,
        state: STDPState,
        learn: bool = True,
    ) -> tuple[torch.Tensor, STDPState]:
        cur = spk_pre @ self.W
        mem = self.beta * state.mem + cur

        crossed = mem >= self.threshold
        spk_post = torch.zeros_like(mem)
        if crossed.any():
            mem_for_pick = mem.masked_fill(~crossed, float("-inf"))
            winner = int(mem_for_pick.argmax().item())
            spk_post[winner] = 1.0
            mem_new = torch.where(
                torch.arange(self.n_post, device=mem.device) == winner,
                mem - self.threshold,
                torch.zeros_like(mem),
            )
        else:
            mem_new = mem

        x_pre = state.x_pre * self.alpha_pre
        y_post = state.y_post * self.alpha_post

        if learn:
            if spk_pre.any():
                dW_ltd = self.a_minus * torch.outer(spk_pre, y_post)
                self.W += dW_ltd
            if spk_post.any():
                dW_ltp = self.a_plus * torch.outer(x_pre, spk_post)
                self.W += dW_ltp
            self.W.clamp_(0.0, self.w_max)

        x_pre = x_pre + spk_pre
        y_post = y_post + spk_post

        return spk_post, STDPState(mem=mem_new, x_pre=x_pre, y_post=y_post)
