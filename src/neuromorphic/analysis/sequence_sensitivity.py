"""EXP-054: does the concept still distinguish shells after pretraining?

The encoder is pretrained on an inverse model - predict the move from a state pair - which is
**purely single-step**. Four experiments have leaned on the idea that over-training it yields a
code good at "which move just happened" and bad at "how far along a sequence we are", and none
measured it. This module is that measurement.

THE STATISTIC TRAINS NOTHING. For shells `d = 1..6` of exact BFS distance, encode states, take
the concept rate, centre, and measure how cosine similarity decays with shell separation
`|d1 - d2|`. `S` is the negated slope: higher means the code separates shells more sharply.

No classifier, no probe, no fitted parameters beyond that slope. Every instrument this project
has retired was a trained linear probe or a pretext score, and a statistic with no capacity
cannot overfit its way into a story.

Distance-to-solved appears here ONLY as an offline analysis label, the same way EXP-033 used
depth labels. Nothing about it reaches an encoder.
"""

from __future__ import annotations

import random

import torch

from neuromorphic.training.encoder_pretrain import (
    DEFAULT_CONTENT,
    DEFAULT_T,
    concept_rates,
    states_to_obs,
)

DEPTHS = (1, 2, 3, 4, 5, 6)
N_PER_SHELL = 60


def sample_shells(provider, depths=DEPTHS, n_per_shell=N_PER_SHELL, rng=None):
    """Up to `n_per_shell` states from each exact-distance shell, deterministically.

    The shallow shells are SMALL and that is a real limit, not a sampling choice: depth 1 has
    6 states and depth 2 has 27, against 8,969 at depth 6. Their means are noisy and no claim
    may rest on a single shell pair.
    """
    rng = rng or random.Random(0)
    out = {}
    for d in depths:
        states = provider.states_at_distance(d)
        out[d] = list(states) if len(states) <= n_per_shell else rng.sample(list(states), n_per_shell)
    return out


def shell_concepts(sensory, shells, *, generator=None, num_steps=DEFAULT_T):
    """`{depth: [n_d, content]}` concept rates, one batched forward per shell.

    `concept_rates` is batched, which is what makes this experiment minutes rather than hours.

    Wrapped in `torch.no_grad()` because no gradient is ever taken here - this is a read-only
    measurement, not training. Without it, `SensoryCortex.forward` retains the full LIF/BPTT
    autograd graph for `num_steps=32` timesteps per shell, which at driver scale (60 encoders
    x 6 shells) is pure waste: retained backward-pass activations for a backward pass that
    never happens.
    """
    with torch.no_grad():
        return {
            d: concept_rates(sensory, states_to_obs(states), num_steps=num_steps, generator=generator)
            for d, states in shells.items()
        }


def similarity_matrix(concepts, *, centre=True):
    """Mean cosine similarity between every pair of shells, keyed `(d1, d2)` with `d1 <= d2`.

    CENTRING IS LOAD-BEARING, NOT COSMETIC. Concept vectors are firing rates and therefore
    non-negative, so raw cosine between any two of them is compressed near 1 and is dominated
    by overall activity rather than by structure. The uncentred statistic would report "every
    state resembles every other" for every encoder and would look like a clean null. The grand
    mean is subtracted once, over the whole sampled set, before any similarity is computed.

    Self-pairs are excluded: a vector's cosine with itself is always 1.0 and would inflate the
    within-shell term. A shell with a single vector therefore yields NO `(d, d)` entry.
    """
    depths = sorted(concepts)
    mats = {d: concepts[d].float() for d in depths}
    if centre:
        grand = torch.cat([mats[d] for d in depths]).mean(dim=0, keepdim=True)
        mats = {d: mats[d] - grand for d in depths}
    unit = {d: torch.nn.functional.normalize(mats[d], dim=1) for d in depths}

    sim = {}
    for i, d1 in enumerate(depths):
        for d2 in depths[i:]:
            block = unit[d1] @ unit[d2].T
            if d1 == d2:
                n = block.shape[0]
                if n < 2:
                    continue          # no off-diagonal pairs exist
                total = block.sum() - block.diagonal().sum()
                sim[(d1, d2)] = float(total / (n * (n - 1)))
            else:
                sim[(d1, d2)] = float(block.mean())
    return sim


def sensitivity_from_similarity(sim, *, min_separation=0):
    """`S` = the negated least-squares slope of similarity against `|d1 - d2|`.

    Positive `S` means similarity falls as shells get further apart, i.e. the code carries
    distance structure. Zero means it does not. Raises `ValueError` when fewer than two
    distinct separations are present, because a slope is undefined there and a silent 0.0
    would be indistinguishable from a genuine "no decay" measurement - those are different
    states of the world. Unreachable in practice for real runs (depths 1..6 with at least 6
    states per shell always produce multiple separations); it exists so a misconfiguration
    is loud instead of quiet.

    `min_separation` restricts the fit to pairs with `|d1 - d2| >= min_separation`. The
    default, 0, includes the `|dd| = 0` within-shell term and is the pre-registered `S`
    exactly as it has always been computed - this default must never change. Passing 1 drops
    that term and gives `S_cross`: measured on this repo's own `_shell_structured` test
    fixture, whose shell centres are independent random directions with NO distance ordering
    at all, `S` comes out to 0.26708 while `S_cross` on the same fixture is 0.02318. So on a
    fixture that only clusters and is not graded by distance, `S` is about 91% driven by the
    `|dd| = 0` term alone. `S_cross` is the part of `S` that cannot be explained by within-shell
    tightness, and is reported beside it for exactly that reason (see the EXP-054 spec
    amendment).
    """
    pairs = [(d1, d2) for (d1, d2) in sim if abs(d2 - d1) >= min_separation]
    xs = [float(abs(d2 - d1)) for (d1, d2) in pairs]
    ys = [sim[k] for k in pairs]
    if len(set(xs)) < 2:
        raise ValueError(
            f"fewer than two distinct shell separations in sim ({sorted(set(xs))}) at "
            f"min_separation={min_separation}; a slope is undefined"
        )
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0.0:
        raise ValueError(
            "zero variance in shell separations despite multiple distinct values; "
            "a slope is undefined"
        )
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return -slope


def sequence_sensitivity(sensory, provider, *, depths=DEPTHS, n_per_shell=N_PER_SHELL, seed=0,
                         num_steps=DEFAULT_T, content=DEFAULT_CONTENT):
    """The whole measurement for one encoder. Deterministic given `seed`.

    Encoding is stochastic (Poisson spiking), so the generator is seeded here rather than left
    to the global RNG: re-running an encoder must reproduce its `S` exactly.
    """
    shells = sample_shells(provider, depths, n_per_shell, rng=random.Random(seed))
    generator = torch.Generator().manual_seed(seed)
    concepts = shell_concepts(sensory, shells, generator=generator, num_steps=num_steps)
    sim = similarity_matrix(concepts)
    return {
        "S": sensitivity_from_similarity(sim),
        "sim": {f"{d1}_{d2}": v for (d1, d2), v in sim.items()},
        "n_by_shell": {d: len(s) for d, s in shells.items()},
    }


def sim_from_record(sim: dict) -> dict:
    """Parse a stored record's `{"d1_d2": value}` sim dict back into the `{(d1, d2): value}`
    form `sensitivity_from_similarity` expects. The inverse of the serialization above, so an
    aggregator reading records off disk can recompute `S_cross` (or any other separation-
    filtered variant) without re-running anything.
    """
    out = {}
    for key, value in sim.items():
        d1, d2 = key.split("_")
        out[(int(d1), int(d2))] = value
    return out
