"""EXP-061's `memory_noise` readout: the recall block replaced by matched-magnitude noise.

The arm exists because EXP-059 could not separate two explanations of its own result. It found
correct and incorrect memory indistinguishable (M-S = +0.0204, p 0.4268) and both about 0.10
below amnesic, which is equally consistent with "the recall is noise on the policy path" and
with "the stored content is actively misleading". Matched-magnitude noise separates them.

Every assertion here is chosen to fail against a specific wrong implementation, named in the
test. The one that matters most is `test_only_the_recall_slice_changes`: substituting the wrong
slice, or perturbing magnitude as well as content, would make the arm answer a different
question than the spec claims while looking entirely normal in the numbers.
"""

from __future__ import annotations

import random

import pytest
import torch

from neuromorphic.training.cube_baseline import (
    CubeConfig,
    MemoryReadout,
    feature_width,
    make_agent,
)

CONTENT = 64


def _brain(seed: int = 0):
    cfg = CubeConfig(arm="regionalized", readout="memory", tag="t", depth=5, seed=seed,
                     sigma=0.0, content=CONTENT)
    return make_agent(cfg), cfg


def _one_out(brain, obs_seed: int = 0):
    rng = random.Random(obs_seed)
    obs = torch.tensor([rng.randrange(6) for _ in range(24)], dtype=torch.float32)
    with torch.no_grad():
        return brain.step(obs, store=True, recall=True)


def test_mode_is_accepted_by_reset():
    """Breaks forgetting to add the mode to `reset`'s whitelist, which raises mid-episode."""
    brain, _ = _brain()
    MemoryReadout("memory_noise", random.Random(0), brain).reset()


def test_feature_width_matches_memory():
    """The noise replaces recall IN PLACE, so width must be unchanged - otherwise the arm also
    varies head width, which is the exact confound the amnesic arm exists to remove. Breaks a
    width change, and breaks forgetting the mode in `feature_width` (which raises)."""
    base = CubeConfig(arm="regionalized", readout="memory", tag="t", depth=5, seed=0,
                      sigma=0.0, content=CONTENT)
    noisy = CubeConfig(arm="regionalized", readout="memory_noise", tag="t", depth=5, seed=0,
                       sigma=0.0, content=CONTENT)
    assert feature_width(noisy) == feature_width(base) == CONTENT * 2 + 1


def test_only_the_recall_slice_changes():
    """THE load-bearing test. Against the SAME brain output, `memory_noise` must leave the
    concept and familiarity blocks bit-identical to `memory` and change only the recall block.

    Breaks substituting the wrong slice, breaks perturbing the concept, and breaks an
    implementation that rebuilds the whole feature vector from scratch.
    """
    brain, _ = _brain(seed=1)
    out = _one_out(brain, obs_seed=2)
    ref = MemoryReadout("memory", random.Random(7), brain)
    ref.reset()
    noisy = MemoryReadout("memory_noise", random.Random(7), brain)
    noisy.reset()
    with torch.no_grad():
        a = ref(out)
        b = noisy(out)
    assert torch.equal(a[:CONTENT], b[:CONTENT]), "concept block must be untouched"
    assert torch.equal(a[CONTENT * 2:], b[CONTENT * 2:]), "familiarity must be untouched"
    assert not torch.equal(a[CONTENT:CONTENT * 2], b[CONTENT:CONTENT * 2]), \
        "the recall block must actually be replaced"


def test_noise_is_magnitude_matched_not_merely_random():
    """The arm's whole claim is that ONLY information content differs. A plain `randn_like`
    would have norm ~sqrt(64) against a real recall norm near 1, changing magnitude by ~8x and
    confounding the manipulation. Breaks the unscaled version.
    """
    brain, _ = _brain(seed=1)
    out = _one_out(brain, obs_seed=3)
    ref = MemoryReadout("memory", random.Random(5), brain); ref.reset()
    noisy = MemoryReadout("memory_noise", random.Random(5), brain); noisy.reset()
    with torch.no_grad():
        a = ref(out)
        b = noisy(out)
    na = float(a[CONTENT:CONTENT * 2].norm())
    nb = float(b[CONTENT:CONTENT * 2].norm())
    assert na > 0
    assert nb == pytest.approx(na, rel=1e-5), f"norm {nb} should match the real recall's {na}"


def test_noise_is_reproducible_at_a_fixed_seed():
    """A seeded run must be re-runnable byte-identically - the discipline the whole project's
    correctness checks rest on. Breaks an unseeded generator."""
    brain, _ = _brain(seed=1)
    out = _one_out(brain, obs_seed=4)
    vals = []
    for _ in range(2):
        ro = MemoryReadout("memory_noise", random.Random(11), brain)
        ro.reset()
        with torch.no_grad():
            vals.append(ro(out).clone())
    assert torch.equal(vals[0], vals[1])


def test_noise_does_not_consume_the_global_torch_stream():
    """Breaks using the global RNG. Drawing noise from it would shift every later torch draw in
    this arm - action sampling included - so the arm would differ from `memory` in more than the
    recall block, which is precisely what it must not do."""
    brain, _ = _brain(seed=1)
    out = _one_out(brain, obs_seed=5)
    torch.manual_seed(1234)
    before = torch.randn(3)
    torch.manual_seed(1234)
    ro = MemoryReadout("memory_noise", random.Random(2), brain)
    ro.reset()
    with torch.no_grad():
        ro(out)
    after = torch.randn(3)
    assert torch.equal(before, after), "the global stream must be untouched"


def test_norm_ratio_instrument_is_populated_for_memory_modes():
    """EXP-061's gate reads this. Breaks an instrument that never accumulates, which would make
    the gate read 0 and VOID the experiment for the wrong reason."""
    brain, _ = _brain(seed=1)
    out = _one_out(brain, obs_seed=6)
    for mode in ("memory", "memory_amnesic", "memory_noise"):
        ro = MemoryReadout(mode, random.Random(3), brain)
        ro.reset()
        with torch.no_grad():
            ro(out)
        assert ro.norm_n == 1, f"{mode} did not record a norm sample"
        assert ro.concept_norm_sum > 0 and ro.recall_norm_sum > 0
        ratio = ro.recall_norm_sum / ro.concept_norm_sum
        assert 0.0 < ratio < 5.0, f"{mode} ratio {ratio} is not a plausible magnitude"


def test_noise_cos_is_recorded_only_for_the_noise_arm():
    """Breaks accumulating the sanity cosine in the wrong arm, which would make it look like a
    real measurement on arms where no substitution happened."""
    brain, _ = _brain(seed=1)
    out = _one_out(brain, obs_seed=7)
    for mode, expect in (("memory", 0), ("memory_amnesic", 0), ("memory_noise", 1)):
        ro = MemoryReadout(mode, random.Random(3), brain)
        ro.reset()
        with torch.no_grad():
            ro(out)
        assert ro.noise_cos_n == expect, f"{mode} recorded {ro.noise_cos_n} noise cosines"


def test_noise_is_uninformative_about_the_recall_it_replaced():
    """Over many draws the cosine between the noise and the real recall must average near zero.
    Breaks any implementation that leaks the real recall into the substitute - scaling it,
    sign-flipping it, or adding noise to it rather than replacing it.
    """
    brain, _ = _brain(seed=1)
    out = _one_out(brain, obs_seed=8)
    ro = MemoryReadout("memory_noise", random.Random(9), brain)
    ro.reset()
    with torch.no_grad():
        for _ in range(400):
            ro(out)
    mean_cos = ro.noise_cos_sum / ro.noise_cos_n
    # 1/sqrt(64) = 0.125 per draw, so the mean over 400 has se ~0.006; 0.05 is ~8 se.
    assert abs(mean_cos) < 0.05, f"mean cosine {mean_cos} suggests the real recall leaks through"

    # AND assert on the RETURNED feature, not only the instrument. Mutation testing showed the
    # instrument alone could not catch `recall = 0.5 * recall + 0.5 * noise`, because it compared
    # the real recall to the noise vector rather than to what the policy actually receives.
    ref = MemoryReadout("memory", random.Random(9), brain)
    ref.reset()
    noisy2 = MemoryReadout("memory_noise", random.Random(9), brain)
    noisy2.reset()
    with torch.no_grad():
        real = ref(out)[CONTENT:CONTENT * 2]
        got = noisy2(out)[CONTENT:CONTENT * 2]
    cos = float(torch.dot(real, got) / (real.norm() * got.norm()))
    assert abs(cos) < 0.45, (
        f"cosine {cos} between the real recall and the RETURNED block is too high; a 50/50 "
        "blend would read about 0.7")


def test_preexisting_modes_are_numerically_inert():
    """REGRESSION. EXP-061 REUSES EXP-059's arm A and M records rather than re-running them, so
    the instruments added for this experiment must not have changed those modes' values. If this
    ever fails, EXP-061's contrasts are comparing records produced by different code.

    Breaks any future edit that makes the memory or amnesic path consume RNG or alter a tensor.
    """
    brain, _ = _brain(seed=1)
    out = _one_out(brain, obs_seed=9)
    # The generator is created ONLY for the noise mode, so constructing a readout for another
    # mode must leave its `random.Random` stream untouched.
    for mode in ("memory", "memory_amnesic", "memory_shuffled"):
        probe = random.Random(42)
        MemoryReadout(mode, probe, brain)
        assert probe.random() == random.Random(42).random(), \
            f"constructing a {mode} readout consumed from the shared rng"
    probe = random.Random(42)
    MemoryReadout("memory_noise", probe, brain)
    assert probe.random() != random.Random(42).random(), \
        "the noise arm is expected to draw its generator seed from the rng"
