"""Inverse-model pretraining for the sensory encoder (vault Stage 2, EXP-039).

Everything the cube line has achieved runs on a **frozen randomly-initialised brain** and a
`Linear(64 -> 6)` head - 390 trainable parameters. EXP-033 measured the cost: a linear probe
for "which moves reduce distance-to-solved" reads **0.459** off the shipped concept@64 at
depth 4, against **0.766** off the raw facelets. The representation throws information away,
and EXP-033 Finding 1 refuted width as the way to get it back (concept@512 reaches only 0.638
and the doublings are saturating).

So the encoder has to be **trained**. This module is that machinery.

THE OBJECTIVE, per the vault: predict the move from a state pair. Self-supervised - the move
is known because we applied it - so no oracle labels are involved and distance-to-solved stays
strictly an instrument.

    loss = CrossEntropy(inverse_head([rate(concept(s)), rate(concept(s'))]), a)

WHY THIS LIVES IN src/ RATHER THAN THE EXPERIMENT DIRECTORY: the Stage 2 follow-on trains a
policy on a pretrained encoder and needs the same forward path. An experiment-local copy would
fork on first contact.

WHAT IS DELIBERATELY NOT HERE: `SensoryCortex` is not modified. It is already differentiable
through snntorch's surrogate gradients, and `BrainRegion._record` is a no-op when recording is
off, so BPTT works as-is. Changing it would make every frozen-arm number in EXP-033
non-comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from neuromorphic.encoders import cube_encoder
from neuromorphic.envs.cube import MOVES, apply_move
from neuromorphic.regions.sensory_cortex import SensoryCortex

# Never a literal: a 3x3 cube is 12 or 18 moves. (CLAUDE.md architecture invariant.)
N_ACTIONS = len(MOVES)

# The SHIPPED cube encoder configuration, mirroring `Brain.__init__` exactly. If these drift
# apart, the frozen arm stops being the encoder the policy actually uses and every comparison
# in EXP-039 silently changes meaning.
CUBE_N_OBS = 144        # 24 facelets x 6 colours
CUBE_OBS_WIDTH = 24
DEFAULT_CONTENT = 64
DEFAULT_T = 32


def make_sensory(seed: int, *, content: int = DEFAULT_CONTENT, num_steps: int = DEFAULT_T
                 ) -> SensoryCortex:
    """A sensory cortex identical to the one `Brain` builds for the cube.

    `Brain` calls `SensoryCortex(n_obs=self.n_obs, concept=content, num_steps=num_steps,
    seed=seed)`, leaving hidden=128, weight_gain=5.0, beta=0.9, threshold=1.0 at their
    defaults. Reproduced here rather than constructed through `Brain` so pretraining does not
    have to build four unused regions per seed.
    """
    return SensoryCortex(n_obs=CUBE_N_OBS, concept=content, num_steps=num_steps, seed=seed)


def states_to_obs(states) -> torch.Tensor:
    """`[N, 24]` long tensor of facelet colours, the layout `encode_cube` expects."""
    return torch.tensor([list(s) for s in states], dtype=torch.long)


def concept_rates(
    sensory: SensoryCortex,
    obs: torch.Tensor,
    *,
    num_steps: int = DEFAULT_T,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mean concept firing rate for a BATCH of observations. `[B, 24]` -> `[B, content]`.

    This is the batched twin of `reinforce.concept_rate`, which is
    `out["concept"].mean(dim=0)[0]` for a single agent. Same quantity, same window mean, but
    B states at once instead of one.

    > It is NOT bit-identical to looping `brain.step` over the same states. `encode_cube` draws
    > Poisson spikes, and drawing `[T, B, 144]` in one call consumes the generator differently
    > from B separate `[T, 1, 144]` draws. The two agree in DISTRIBUTION, not per-sample. This
    > is why EXP-039 re-measures its own frozen arm instead of comparing against EXP-033's
    > published 0.459 across pipelines.
    """
    spikes = cube_encoder()(obs, T=num_steps, generator=generator)   # [T, B, 144]
    concept = sensory(spikes)                                        # [T, B, content]
    return concept.mean(dim=0)                                       # [B, content]


class InverseModel(nn.Module):
    """Sensory encoder plus a linear head that names the move between two states.

    The head is deliberately LINEAR. A nonlinear head could learn the move from two frozen
    concepts without the encoder improving at all, which would train the head instead of the
    thing under test and make the probe result unreadable.
    """

    def __init__(self, sensory: SensoryCortex, *, content: int = DEFAULT_CONTENT,
                 n_actions: int = N_ACTIONS):
        super().__init__()
        self.sensory = sensory
        self.head = nn.Linear(2 * content, n_actions)

    def forward(self, obs_s: torch.Tensor, obs_next: torch.Tensor, *,
                num_steps: int = DEFAULT_T,
                generator: torch.Generator | None = None) -> torch.Tensor:
        """`[B, 24]`, `[B, 24]` -> move logits `[B, n_actions]`.

        Both states go through the encoder in ONE batch so they see identically-distributed
        Poisson noise; encoding them separately would give `s'` a different noise realisation
        than `s` for no reason.
        """
        b = obs_s.shape[0]
        both = torch.cat([obs_s, obs_next], dim=0)                  # [2B, 24]
        rates = concept_rates(self.sensory, both, num_steps=num_steps, generator=generator)
        return self.head(torch.cat([rates[:b], rates[b:]], dim=1))  # [B, 2*content]


def build_pairs(states, forbidden=None) -> list[tuple[tuple, int, tuple]]:
    """Every `(s, a, apply_move(s, a))` for `s` in `states`, dropping pairs that touch
    `forbidden` at EITHER endpoint.

    `forbidden` is the contamination control: pass the probe's held-out states. If the encoder
    pretrains on the states the probe scores, "held-out" probe accuracy measures an encoder
    that has already seen them. The inverse model never sees distance labels so it cannot
    memorise optimality directly, but it can memorise state-specific structure, and that is
    enough to inflate the number.

    BOTH endpoints are checked, not just `s`: `s'` is pushed through the same encoder and its
    facelets are seen just as directly.

    > EXCLUSION, NOT INCLUSION, AND THE DIFFERENCE IS LARGE. An earlier version required the
    > successor to be inside an ALLOWED set drawn from the probed depths. Because every cube
    > move changes distance-to-solved by exactly +-1, that silently deleted every outward move
    > from the deepest probed shell - and the deepest shell is most of the data. Measured
    > 2026-08-08 on depths 1-6: 16,032 surviving pairs out of 71,472, i.e. 22%, with depth-6
    > states contributing almost nothing. Excluding only what the probe scores keeps the
    > successor free to be a depth-0 or depth-7 state, which needs no label because pretraining
    > is self-supervised.
    """
    forbidden_set = set() if forbidden is None else set(forbidden)
    pairs = []
    for s in states:
        if s in forbidden_set:
            continue
        for a in range(N_ACTIONS):
            nxt = apply_move(s, a)
            if nxt in forbidden_set:
                continue
            pairs.append((s, a, nxt))
    return pairs


@dataclass
class PretrainConfig:
    seed: int = 0
    content: int = DEFAULT_CONTENT
    num_steps: int = DEFAULT_T
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    log_every: int = 5


@dataclass
class PretrainResult:
    sensory: SensoryCortex
    model: InverseModel
    history: list[dict] = field(default_factory=list)

    @property
    def final_accuracy(self) -> float:
        return self.history[-1]["accuracy"] if self.history else 0.0


def train_inverse_model(pairs, cfg: PretrainConfig, *, sensory: SensoryCortex | None = None,
                        progress=None) -> PretrainResult:
    """Train the encoder to name the move between two states.

    `pairs` is the output of `build_pairs`. Returns the trained encoder plus a per-epoch
    history of loss and training accuracy, so a smoke run can calibrate epochs from
    measurement instead of a guess.

    Training accuracy is on the pretraining task, NOT the probe. It says whether the objective
    is being learned at all - a run that never gets above the 1/6 move-naming floor has not
    trained, and its probe result would be measuring a random encoder under another name.
    """
    torch.manual_seed(cfg.seed)
    sensory = sensory if sensory is not None else make_sensory(cfg.seed, content=cfg.content,
                                                              num_steps=cfg.num_steps)
    model = InverseModel(sensory, content=cfg.content)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(cfg.seed)

    obs_s = states_to_obs([p[0] for p in pairs])
    obs_n = states_to_obs([p[2] for p in pairs])
    actions = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    n = len(pairs)

    history: list[dict] = []
    for epoch in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen)
        total_loss, correct, seen = 0.0, 0, 0
        for start in range(0, n, cfg.batch_size):
            idx = perm[start:start + cfg.batch_size]
            logits = model(obs_s[idx], obs_n[idx], num_steps=cfg.num_steps, generator=gen)
            loss = loss_fn(logits, actions[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach()) * len(idx)
            correct += int((logits.argmax(dim=1) == actions[idx]).sum())
            seen += len(idx)
        row = {"epoch": epoch, "loss": total_loss / seen, "accuracy": correct / seen}
        history.append(row)
        if progress is not None and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            progress(row)

    return PretrainResult(sensory=sensory, model=model, history=history)
