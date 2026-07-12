"""EXP-028: degrade the sensory concept before the policy head reads it.

Two operators: additive Gaussian noise (continuous fidelity dose) and structural
unit-drop (zero a fraction of concept units; random or most-important-first).
Wrapping the head means both training and eval flow through the ablated channel
with no change to reinforce.py (mirrors EXP-027's MaskedHead).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AblationSpec:
    """A single dose of concept corruption. JSON-safe (scalars only)."""

    kind: str          # "gaussian" | "unitdrop"
    dose: float        # gaussian: sigma; unitdrop: fraction of units to zero
    mode: str = "random"  # unitdrop only: "random" | "top"
    seed: int = 0      # reproducible noise / random mask


class AblatedConcept(nn.Module):
    """Wrap a policy ``head``; perturb the concept vector before ``head`` reads it."""

    def __init__(self, head, spec: AblationSpec | None, *, width: int, order=None):
        super().__init__()
        self.head = head
        self.spec = spec
        self.width = width
        self._gen = None
        self._mask = None
        if spec is not None and spec.dose > 0 and spec.kind == "unitdrop":
            k = round(spec.dose * width)
            mask = torch.ones(width)
            if spec.mode == "top":
                if order is None:
                    raise ValueError("unitdrop mode='top' requires an importance order")
                idx = torch.tensor(list(order)[:k], dtype=torch.long)
            else:
                g = torch.Generator().manual_seed(spec.seed)
                idx = torch.randperm(width, generator=g)[:k]
            mask[idx] = 0.0
            self._mask = mask
        if spec is not None and spec.dose > 0 and spec.kind == "gaussian":
            self._gen = torch.Generator().manual_seed(spec.seed)

    def forward(self, x):
        spec = self.spec
        if spec is None or spec.dose == 0:
            return self.head(x)
        if spec.kind == "gaussian":
            x = x + spec.dose * torch.randn(x.shape, generator=self._gen)
        elif spec.kind == "unitdrop":
            x = x * self._mask.to(x.dtype)
        else:
            raise ValueError(f"unknown ablation kind {spec.kind!r}")
        return self.head(x)
