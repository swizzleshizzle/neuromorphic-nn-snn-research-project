"""EXP-063's learned readouts: `memory_attn` and its capacity-matched control.

EXP-059 found memory HURTS and EXP-061 found the recall block indistinguishable from
matched-magnitude noise. Both measured ONE readout - a raw hippocampal read concatenated onto
the concept. Neither can say whether the episodic information is unusable or merely unused, and
that is the question the checkpoint has to answer.

`memory_attn` attends over a PERFECT cache of prior concepts, so it is a CEILING instrument: if
a learned readout with perfect episodic memory cannot beat a control with nothing real to
attend to, no readout over the lossy attractor read can. `memory_attn_noise` holds the
parameter count, the feature width and the read block's MAGNITUDE fixed, so the pair differs in
the content of the attended set and in nothing else.

Every assertion below is chosen to fail against a specific wrong implementation, named in the
test. Three matter most:

  * `test_attention_gradient_reaches_the_parameters` - EXP-047 shipped a "make X trainable"
    change where a stray `no_grad` meant nothing trained, `fc1.weight` moved by exactly 0.0,
    and the run looked perfectly ordinary. Asserting a switch is set does not catch that.
  * `test_attn_noise_read_is_exactly_the_noise_cache` - the leak detector. EXP-061's cosine
    detector compared two PRE-substitution values, read ~0 by construction, and was blind to a
    50% leak. This recomputes the returned vector independently and demands exact equality, so
    a leak of any size fails it.
  * `test_read_at_the_first_step_is_zero` - if the current state is appended BEFORE the read,
    attention can put all its mass on the state the agent is already looking at and the arm
    silently becomes a wider amnesic arm.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from neuromorphic.training.cube_baseline import (
    ATTN_DK,
    CubeConfig,
    MemoryReadout,
    feature_width,
    make_agent,
    run_cube_baseline,
)
from neuromorphic.training.reinforce import concept_rate, policy_parameters, train_episode
from neuromorphic.envs.cube import CubeEnv

CONTENT = 64
HEAD_PARAMS = (CONTENT * 2 + 1) * 6 + 6      # 780: the frozen policy head at memory width
ATTN_PARAMS = 2 * CONTENT * ATTN_DK          # 2048: W_q and W_k, no biases


def _brain(seed: int = 0):
    cfg = CubeConfig(arm="regionalized", readout="memory_attn", tag="t", depth=5, seed=seed,
                     sigma=0.0, content=CONTENT)
    return make_agent(cfg), cfg


def _one_out(brain, obs_seed: int = 0):
    rng = random.Random(obs_seed)
    obs = torch.tensor([rng.randrange(6) for _ in range(24)], dtype=torch.float32)
    with torch.no_grad():
        return brain.step(obs, store=True, recall=True)


def _readout(mode: str, brain, seed: int = 7):
    ro = MemoryReadout(mode, random.Random(seed), brain)
    ro.reset()
    return ro


# --------------------------------------------------------------------------- shape and width

@pytest.mark.parametrize("mode", ["memory_attn", "memory_attn_noise"])
def test_feature_width_matches_the_fixed_memory_modes(mode):
    """The learned read REPLACES the recall block in place, so the head width must be
    unchanged. Breaks any width change, which would confound the contrast against EXP-059's
    arms with head width - the confound `memory_amnesic` exists to remove. Also breaks
    forgetting the mode in `feature_width`, which raises."""
    base = CubeConfig(arm="regionalized", readout="memory", tag="t", depth=5, seed=0,
                      sigma=0.0, content=CONTENT)
    new = CubeConfig(arm="regionalized", readout=mode, tag="t", depth=5, seed=0,
                     sigma=0.0, content=CONTENT)
    assert feature_width(new) == feature_width(base) == CONTENT * 2 + 1


@pytest.mark.parametrize("mode", ["memory_attn", "memory_attn_noise"])
def test_mode_is_accepted_by_reset(mode):
    """Breaks forgetting the mode in `reset`'s whitelist, which raises mid-episode."""
    brain, _ = _brain()
    MemoryReadout(mode, random.Random(0), brain).reset()


def test_unknown_readout_is_still_rejected():
    """The whitelist must stay a whitelist. Breaks widening it to accept anything."""
    brain, _ = _brain()
    with pytest.raises(ValueError):
        MemoryReadout("memory_attention", random.Random(0), brain).reset()


# ------------------------------------------------------- the read is of STRICTLY PRIOR states

def test_read_at_the_first_step_is_zero():
    """Nothing has been visited yet, so there is nothing to attend over.

    Breaks appending the current snapshot BEFORE the read: attention would then have one
    entry - the current state - and the read block would be a copy of the concept, turning the
    arm into a wider amnesic arm while every other number looked normal.
    """
    brain, _ = _brain(seed=1)
    ro = _readout("memory_attn", brain)
    feat = ro(_one_out(brain, obs_seed=2))
    read = feat[CONTENT:2 * CONTENT]
    assert torch.equal(read, torch.zeros(CONTENT))
    assert ro.empty_cache_steps == 1


def test_second_step_reads_exactly_the_single_prior_concept():
    """With one prior state the softmax is 1.0 on it, so the read IS that concept.

    Breaks including the current state (the read would be a two-way mixture, not equal to the
    first concept), and breaks attending over the wrong cache.
    """
    brain, _ = _brain(seed=1)
    ro = _readout("memory_attn", brain)
    out1 = _one_out(brain, obs_seed=2)
    first_concept = concept_rate(out1).clone()
    ro(out1)
    feat = ro(_one_out(brain, obs_seed=3))
    assert torch.allclose(feat[CONTENT:2 * CONTENT], first_concept, atol=0, rtol=0)


def test_read_is_a_convex_combination_of_prior_concepts():
    """Recomputes the attention independently and demands exact equality.

    Breaks a value projection sneaking in, breaks a missing softmax normalisation, and breaks
    scaling the read by anything.
    """
    brain, _ = _brain(seed=2)
    ro = _readout("memory_attn", brain)
    for s in (2, 3, 4):
        ro(_one_out(brain, obs_seed=s))
    prior = torch.stack(list(ro._attn_cache))
    out = _one_out(brain, obs_seed=5)
    concept = concept_rate(out)
    feat = ro(out)
    with torch.no_grad():
        logits = (ro.attn.W_k(prior) @ ro.attn.W_q(concept)) / math.sqrt(ATTN_DK)
        expected = torch.softmax(logits, dim=0) @ prior
    assert torch.equal(feat[CONTENT:2 * CONTENT].detach(), expected)


def test_concept_and_familiarity_blocks_match_the_fixed_memory_readout():
    """Only the middle block may differ from `memory`. Breaks substituting the wrong slice and
    breaks rebuilding the whole feature vector."""
    brain, _ = _brain(seed=3)
    out = _one_out(brain, obs_seed=4)
    ref = _readout("memory", brain)(out)
    got = _readout("memory_attn", brain)(out)
    assert torch.equal(got[:CONTENT], ref[:CONTENT])
    assert torch.equal(got[-1:], ref[-1:])


# ------------------------------------------------------------------- the gradient, EXP-047's trap

def test_attention_gradient_reaches_the_parameters():
    """THE EXP-047 TEST. The parameters must MOVE, not merely be registered.

    Breaks wrapping `_attention_features` in `no_grad` (the exact defect that made EXP-047's
    first fine-tuning implementation train nothing while producing an ordinary success rate),
    and breaks detaching the read before it reaches the head.
    """
    brain, _ = _brain(seed=3)
    torch.manual_seed(3)
    ro = _readout("memory_attn", brain)
    head = torch.nn.Linear(CONTENT * 2 + 1, 6)
    opt = torch.optim.Adam(list(policy_parameters(head)), lr=0.05)
    opt.add_param_group({"params": list(ro.attn.parameters()), "lr": 0.05})
    before = {n: p.detach().clone() for n, p in ro.attn.named_parameters()}
    env = CubeEnv(scramble_depth=2, max_steps=6, scramble_seed=1)
    gen = torch.Generator().manual_seed(3)
    for _ in range(6):
        ro.reset()
        brain.hippo.clear()
        train_episode(brain, head, env, opt, generator=gen, max_steps=6,
                      store=True, recall=True, feature_fn=ro)
    for name, p in ro.attn.named_parameters():
        assert p.grad is not None, f"{name} received no gradient"
        assert float((p.detach() - before[name]).abs().max()) > 1e-6, f"{name} did not move"


def test_the_encoder_stays_frozen_under_the_learned_readout():
    """The gradient must reach the attention and NOTHING else.

    The brain output here is built WITH a graph, deliberately. Mutation testing showed that
    building it under `no_grad` - the convenient thing, and what `_one_out` does - makes this
    test unable to fail: the concept arrives already detached, so deleting the readout's own
    `no_grad` changes nothing and the mutation survives. Feeding a live graph in is the only
    way the assertion tests the wrapper it names.
    """
    brain, _ = _brain(seed=3)
    ro = _readout("memory_attn", brain)
    head = torch.nn.Linear(CONTENT * 2 + 1, 6)
    rng = random.Random(2)
    live = [brain.step(torch.tensor([rng.randrange(6) for _ in range(24)], dtype=torch.float32),
                       store=True, recall=True) for _ in range(2)]
    assert live[0]["concept"].requires_grad, "the fixture itself must carry a graph"
    ro(live[0])
    head(ro(live[1])).sum().backward()
    assert all(p.grad is None for p in brain.sensory.parameters())


def test_trainable_surface_is_the_head_plus_the_attention(tmp_path):
    """Counted from the optimizer, so it reports what is ACTUALLY trained.

    Breaks forgetting the `add_param_group` - which would leave the attention at its random
    init for the whole run, producing a perfectly plausible null.
    """
    cfg = CubeConfig(arm="regionalized", readout="memory_attn", tag="t_surface", depth=1,
                     seed=3, sigma=0.0, episodes=2, entropy_beta=0.0, max_depth=1,
                     out_dir=tmp_path)
    assert run_cube_baseline(cfg)["trainable_params"] == HEAD_PARAMS + ATTN_PARAMS


# ------------------------------------------------------------ the control arm, and leak detection

def test_attn_noise_read_is_exactly_the_noise_cache():
    """THE LEAK DETECTOR, and the reason it is not a cosine.

    EXP-061's cosine compared two values taken BEFORE the substitution, so it read ~0 by
    construction and a `read = 0.5 * read + 0.5 * real_read` mutation passed every assertion.
    Here the returned vector is recomputed from the noise cache alone and compared exactly, so
    ANY admixture of the real read fails.
    """
    brain, _ = _brain(seed=4)
    ro = _readout("memory_attn_noise", brain)
    for s in (2, 3, 4):
        ro(_one_out(brain, obs_seed=s))
    noise_prior = torch.stack(list(ro._noise_cache))
    real_prior = torch.stack(list(ro._attn_cache))
    out = _one_out(brain, obs_seed=5)
    concept = concept_rate(out)
    feat = ro(out)
    with torch.no_grad():
        def read_over(keys):
            logits = (ro.attn.W_k(keys) @ ro.attn.W_q(concept)) / math.sqrt(ATTN_DK)
            return torch.softmax(logits, dim=0) @ keys
        noise_read = read_over(noise_prior)
        expected = noise_read * (read_over(real_prior).norm() / noise_read.norm())
    assert torch.equal(feat[CONTENT:2 * CONTENT].detach(), expected)


def test_noise_read_magnitude_matches_the_real_read():
    """The control differs in CONTENT only. Breaks removing the rescale, which would leave the
    control with a systematically different block magnitude and turn an information control
    into a magnitude control - EXP-061's central design point."""
    brain, _ = _brain(seed=4)
    ro = _readout("memory_attn_noise", brain)
    for s in (2, 3, 4):
        ro(_one_out(brain, obs_seed=s))
    real_prior = torch.stack(list(ro._attn_cache))
    out = _one_out(brain, obs_seed=6)
    concept = concept_rate(out)
    feat = ro(out)
    with torch.no_grad():
        logits = (ro.attn.W_k(real_prior) @ ro.attn.W_q(concept)) / math.sqrt(ATTN_DK)
        real_read = torch.softmax(logits, dim=0) @ real_prior
    got = float(feat[CONTENT:2 * CONTENT].detach().norm())
    assert got == pytest.approx(float(real_read.norm()), rel=1e-5)


def test_noise_entries_are_non_negative_and_norm_matched():
    """Concept codes are spike RATES and live in the positive orthant. Signed noise would
    cancel under a convex combination and shrink the read by ~1/sqrt(n).

    Breaks dropping the `.abs()`, and breaks dropping the per-entry norm match.
    """
    brain, _ = _brain(seed=5)
    ro = _readout("memory_attn_noise", brain)
    for s in (2, 3, 4):
        ro(_one_out(brain, obs_seed=s))
    assert len(ro._noise_cache) == len(ro._attn_cache) == 3
    for noise, real in zip(ro._noise_cache, ro._attn_cache):
        assert float(noise.min()) >= 0.0
        assert float(noise.norm()) == pytest.approx(float(real.norm()), rel=1e-5)
        assert float(torch.dot(noise, real) / (noise.norm() * real.norm())) < 0.95


def test_noise_cache_is_stable_within_an_episode():
    """A stored memory does not change identity between steps. Breaks regenerating the noise
    on every read, which would make the control non-stationary as well as uninformative."""
    brain, _ = _brain(seed=5)
    ro = _readout("memory_attn_noise", brain)
    ro(_one_out(brain, obs_seed=2))
    first = ro._noise_cache[0].clone()
    for s in (3, 4):
        ro(_one_out(brain, obs_seed=s))
    assert torch.equal(ro._noise_cache[0], first)


@pytest.mark.parametrize("mode", ["memory_attn", "memory_attn_noise"])
def test_reset_clears_both_caches(mode):
    """Episodic memory is per-episode. Breaks forgetting either cache in `reset`, which would
    let one episode attend over another's states."""
    brain, _ = _brain(seed=5)
    ro = _readout(mode, brain)
    for s in (2, 3):
        ro(_one_out(brain, obs_seed=s))
    assert ro._attn_cache
    ro.reset()
    assert ro._attn_cache == [] and ro._noise_cache == []


# ------------------------------------------------------------------- the mechanism instruments

def test_attention_entropy_discriminates_uniform_from_concentrated():
    """An instrument that cannot detect the thing it exists for is the gate-that-cannot-fail
    trap in disguise. Forcing the two extremes must produce two different readings.

    Breaks a normalisation error and breaks accumulating a constant.
    """
    brain, _ = _brain(seed=6)

    def entropy_with(scale: float) -> float:
        ro = _readout("memory_attn", brain)
        with torch.no_grad():
            ro.attn.W_q.weight.mul_(scale)
            ro.attn.W_k.weight.mul_(scale)
        for s in (2, 3, 4, 5):
            ro(_one_out(brain, obs_seed=s))
        return ro.attn_entropy_sum / ro.attn_entropy_n

    assert entropy_with(0.0) == pytest.approx(1.0, abs=1e-6)     # uniform
    assert entropy_with(60.0) < 0.5                              # concentrated


def test_attention_instruments_ignore_steps_with_no_choice():
    """With one prior state the normalised entropy is 0/0 and the recency mass is trivially
    1.0, so counting those steps would report the cache size rather than the attention.

    Breaks removing the `n < 2` guard (which also makes the entropy a division by log(1)).
    """
    brain, _ = _brain(seed=6)
    ro = _readout("memory_attn", brain)
    ro(_one_out(brain, obs_seed=2))     # empty cache
    ro(_one_out(brain, obs_seed=3))     # one prior: no choice
    assert ro.attn_entropy_n == 0
    ro(_one_out(brain, obs_seed=4))     # two priors: a choice
    assert ro.attn_entropy_n == 1


# ------------------------------------------------------------------------------ inertness

# Measured on the PRE-EXP-063 code (HEAD before this change) in a clean worktree, depth 1,
# seed 3, 12 episodes. `success_rate` is 0.3333 for all three modes and therefore discriminates
# nothing on its own, which is exactly why the other three quantities are here.
PRE_EXP063 = {
    "memory": (0.3333333333333333, 0.19230769230769232, 1.4800713956356049, 0.2317596409516458),
    "memory_amnesic": (0.3333333333333333, 0.14814814814814814, 1.6277640859285991,
                       0.22298428306863838),
    "memory_noise": (0.3333333333333333, 0.19642857142857142, 1.5988838175932567,
                     0.23902492237259376),
}


@pytest.mark.parametrize("mode", sorted(PRE_EXP063))
def test_preexisting_modes_are_numerically_inert(mode, tmp_path):
    """EXP-063 reuses EXP-059's and EXP-061's arms at all 24 seeds rather than re-running them.

    That is legitimate ONLY if this change left them bit-identical. Breaks anything that
    perturbs a pre-existing mode's RNG stream - in particular constructing the attention module
    or the noise generator outside the `mode in ATTN_MODES` guard, which would shift the head
    init of every memory arm and silently invalidate the reuse.
    """
    cfg = CubeConfig(arm="regionalized", readout=mode, tag=f"inert_{mode}", depth=1, seed=3,
                     sigma=0.0, episodes=12, entropy_beta=0.0, normalize_advantages=False,
                     max_depth=1, out_dir=tmp_path)
    r = run_cube_baseline(cfg)
    got = (r["success_rate"], r["revisit_rate"], r["mean_train_entropy"],
           r["recall_concept_norm_ratio"])
    assert got == PRE_EXP063[mode]
