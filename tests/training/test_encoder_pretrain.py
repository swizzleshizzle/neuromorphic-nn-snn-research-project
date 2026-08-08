"""Tests for inverse-model encoder pretraining (EXP-039, vault Stage 2).

Each test here is written to fail against a specific plausible defect. The two that matter
most are the contamination control (`build_pairs` must exclude a pair when EITHER endpoint is
outside the allowed set) and the claim that gradients actually reach the ENCODER rather than
only the head - if either silently broke, EXP-039 would still produce numbers, and they would
mean something other than what the spec says.
"""

from __future__ import annotations

import torch

from neuromorphic.envs.cube import MOVES, apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.cube_baseline import CubeConfig, make_agent
from neuromorphic.training.encoder_pretrain import (
    N_ACTIONS,
    InverseModel,
    PretrainConfig,
    build_pairs,
    concept_rates,
    make_sensory,
    states_to_obs,
    train_inverse_model,
)
from neuromorphic.training.reinforce import concept_rate


def _states(depth: int, n: int | None = None):
    prov = ExactBFSDistance(max_depth=depth + 1)
    s = prov.states_at_distance(depth)
    return s if n is None else s[:n]


def _states_upto(max_depth: int):
    """Every state within `max_depth` of solved, as one contiguous universe.

    A DEPTH RANGE, not a single shell, and that is structural rather than convenience: every
    cube move changes distance-to-solved by exactly +-1, so a state at depth d has neighbours
    only at d-1 and d+1. An allowed set drawn from one shell therefore yields ZERO pairs.
    Pretraining data must span contiguous depths for any pair to survive `build_pairs`.
    """
    prov = ExactBFSDistance(max_depth=max_depth + 1)
    out = []
    for d in range(0, max_depth + 1):
        out.extend(prov.states_at_distance(d))
    return out


def test_make_sensory_matches_the_shipped_brain_encoder():
    """The frozen arm must BE the encoder the policy uses, not a lookalike.

    Fails if `Brain`'s SensoryCortex construction drifts from `make_sensory` - different
    hidden width, weight gain, or seeding would all make EXP-039's frozen arm incomparable to
    EXP-033 while still producing plausible numbers.
    """
    brain = make_agent(CubeConfig(seed=3, content=64))
    mine = make_sensory(3, content=64)
    for a, b in ((brain.sensory.fc1, mine.fc1), (brain.sensory.fc2, mine.fc2)):
        assert a.weight.shape == b.weight.shape
        assert torch.equal(a.weight, b.weight), "same seed must give the same weights"
        assert torch.equal(a.bias, b.bias)


def test_concept_rates_agrees_with_brain_step_in_distribution_only():
    """The instrument check from spec section 4c.

    `encode_cube` draws Poisson spikes, so drawing [T, B, 144] once consumes the generator
    differently from B separate [T, 1, 144] draws. The two pipelines agree in DISTRIBUTION,
    never per-sample.

    Asserting exact equality would be a test that can never pass; asserting nothing would be a
    test that can never fail. So this asserts BOTH directions: the means converge, and the
    draws are not identical. Measured 2026-08-08: mean per-unit |diff| falls 0.0202 -> 0.0054
    as N goes 12 -> 240, consistent with 1/sqrt(N).
    """
    state = _states(3)[0]
    brain = make_agent(CubeConfig(seed=0, content=64))
    obs = states_to_obs([state])

    N = 120
    stepwise, batched = [], []
    for r in range(N):
        g1 = torch.Generator().manual_seed(10_000 + r)
        with torch.no_grad():
            out = brain.step(state_obs := list(state), store=False, recall=False,
                             record=False, generator=g1)
        assert state_obs is not None
        stepwise.append(concept_rate(out))
        g2 = torch.Generator().manual_seed(900_000 + r)
        with torch.no_grad():
            batched.append(concept_rates(brain.sensory, obs, generator=g2)[0])
    a, b = torch.stack(stepwise), torch.stack(batched)

    assert not torch.equal(a, b), "different RNG consumption must give different draws"
    # Aggregate rate is the quantity the probe consumes; it must agree tightly.
    assert abs(float(a.mean()) - float(b.mean())) < 0.02, (
        f"batched mean rate {float(b.mean()):.4f} vs stepwise {float(a.mean()):.4f}")
    # Per-unit agreement, loose enough for Monte-Carlo error at N=120 but far tighter than a
    # systematic pipeline difference would produce.
    assert float((a.mean(0) - b.mean(0)).abs().mean()) < 0.03


def test_build_pairs_excludes_a_pair_when_EITHER_endpoint_is_disallowed():
    """The contamination control, and the reason it checks both endpoints.

    `s'` is pushed through the same encoder as `s`, so pretraining on a pair whose SUCCESSOR is
    a probe-held-out state shows the encoder that state's facelets just as directly.

    This fails against the natural wrong implementation that filters on `s` only.
    """
    states = _states_upto(3)
    # Hold out every fourth state, spread across depths, mimicking the probe's split.
    forbidden = {s for i, s in enumerate(states) if i % 4 == 0}
    kept = [s for s in states if s not in forbidden]
    pairs = build_pairs(states, forbidden=forbidden)

    assert pairs, "sanity: the split must still yield some pairs"
    for s, a, nxt in pairs:
        assert s not in forbidden
        assert nxt not in forbidden, "a held-out state leaked in as a successor"

    # The control must actually BIND: with this split there really are pairs whose source is
    # kept but whose successor is held out. Without this, the loop above could pass vacuously.
    leaky = [
        (s, a) for s in kept for a in range(N_ACTIONS)
        if apply_move(s, a) in forbidden
    ]
    assert leaky, "test is vacuous unless some pair would leak under an s-only filter"
    # An s-only filter would have produced exactly these extra pairs. Naming the count makes
    # the regression obvious rather than a silent shift in dataset size.
    assert len(pairs) + len(leaky) == len(kept) * N_ACTIONS


def test_successors_outside_the_probed_depths_are_KEPT():
    """The fix for a defect the EXP-039 pilot exposed, pinned so it cannot come back.

    An earlier version required the successor to be inside an ALLOWED set drawn from the probed
    depths. Because every move changes distance by exactly +-1, that silently deleted every
    outward move from the DEEPEST probed shell - and the deepest shell is most of the data.
    Measured on depths 1-6: 16,032 pairs survived out of 71,472 (22%), starving the encoder
    exactly where Wall 1 bites hardest.

    Pretraining is self-supervised, so a successor needs no label and may lie outside the
    probed range entirely. With NOTHING forbidden, every state must contribute all six moves.
    """
    deepest = 3
    states = _states_upto(deepest)
    pairs = build_pairs(states)
    assert len(pairs) == len(states) * N_ACTIONS, "every state must keep all six moves"

    # And the depletion is real: some successors genuinely fall outside the probed universe,
    # so an inclusion-based filter would have dropped them.
    universe = set(states)
    escaping = [(s, a) for s in states for a in range(N_ACTIONS)
                if apply_move(s, a) not in universe]
    assert escaping, "test is vacuous unless some successor escapes the probed depths"
    kept_under_inclusion_filter = len(states) * N_ACTIONS - len(escaping)
    assert kept_under_inclusion_filter < len(pairs), (
        "an inclusion filter must lose pairs that the exclusion filter keeps")


def test_every_move_changes_depth_by_exactly_one():
    """The structural fact behind both pair-construction rules, pinned because it is unobvious.

    Every cube move changes distance-to-solved by exactly +-1 - never 0. That is why an
    inclusion filter over a single shell yields ZERO pairs, and why an inclusion filter over a
    depth RANGE still silently starves the deepest shell.
    """
    prov = ExactBFSDistance(max_depth=4)
    for s in prov.states_at_distance(2)[:30]:
        for a in range(N_ACTIONS):
            nd = prov.distance(apply_move(s, a))
            assert nd in (1, 3), f"a move from depth 2 reached depth {nd}, not 1 or 3"


def test_build_pairs_unrestricted_covers_every_move():
    states = _states(2, 5)
    pairs = build_pairs(states)
    assert len(pairs) == len(states) * len(MOVES)
    assert {a for _, a, _ in pairs} == set(range(len(MOVES)))


def test_training_updates_the_ENCODER_not_just_the_head():
    """The load-bearing claim of the whole experiment.

    If surrogate gradients did not reach `SensoryCortex` - a frozen module, a detach, a
    `no_grad` in the wrong place - the head alone would train, the loss would still fall, and
    the probe would be measuring a RANDOM encoder under the name of a trained one. That is
    exactly the kind of result that survives review.

    So this checks the encoder weights moved, not that the loss went down.
    """
    states = _states(2)
    pairs = build_pairs(states)
    sensory = make_sensory(0)
    before = {"fc1": sensory.fc1.weight.detach().clone(),
              "fc2": sensory.fc2.weight.detach().clone()}

    res = train_inverse_model(pairs, PretrainConfig(seed=0, epochs=2, batch_size=64))

    for name, w0 in before.items():
        w1 = getattr(res.sensory, name).weight
        assert not torch.equal(w0, w1), f"{name} did not change: the encoder is not training"
        # Not merely different by numerical dust.
        assert float((w1 - w0).detach().abs().max()) > 1e-5, f"{name} moved only negligibly"

    assert len(res.history) == 2
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in res.history)


def test_inverse_model_forward_shape_and_action_width():
    """Action width comes from MOVES, never a literal (CLAUDE.md invariant)."""
    sensory = make_sensory(0)
    model = InverseModel(sensory)
    obs = states_to_obs(_states(2, 3))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        logits = model(obs, obs, generator=g)
    assert logits.shape == (3, len(MOVES))
    assert model.head.in_features == 2 * 64
