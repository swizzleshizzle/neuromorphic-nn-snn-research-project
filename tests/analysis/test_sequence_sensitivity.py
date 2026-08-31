"""EXP-054: the sequence-sensitivity statistic, and the proof it measures structure.

The metric trains NOTHING - no classifier, no fitted parameters beyond a slope over
already-computed similarities. That is deliberate: every instrument this project has retired
(the EXP-033 decodability probe, pretraining move-accuracy, the entropy trace) was a trained
linear probe or a pretext score, and a statistic with no capacity cannot overfit into a story.

The tests below build encoders with KNOWN structure so the statistic can be checked against
an answer rather than against itself.
"""

from __future__ import annotations

import pytest
import torch

from neuromorphic.analysis.sequence_sensitivity import (
    sensitivity_from_similarity,
    similarity_matrix,
)


def _shell_structured(depths, dim=64, spread=1.0, seed=0):
    """Concepts whose shell mean is a distinct random direction per shell.

    A code like this SEPARATES shells, so similarity must fall as |d1-d2| grows.
    """
    g = torch.Generator().manual_seed(seed)
    centres = {d: torch.randn(dim, generator=g) for d in depths}
    return {
        d: centres[d].unsqueeze(0) + spread * 0.05 * torch.randn(12, dim, generator=g)
        for d in depths
    }


def _shell_blind(depths, dim=64, seed=0):
    """Concepts drawn from ONE distribution regardless of shell - no shell information."""
    g = torch.Generator().manual_seed(seed)
    return {d: torch.randn(12, dim, generator=g) for d in depths}


def test_structured_code_scores_high():
    """A code with a distinct direction per shell must score well above zero.

    Threshold is measured, not qualitative. Prototype this before setting the bar and record
    the observed value in a comment here.
    """
    depths = (1, 2, 3, 4, 5, 6)
    sim = similarity_matrix(_shell_structured(depths))
    s = sensitivity_from_similarity(sim)
    assert s > 0.05, f"structured code scored {s:.4f}, expected a clear positive decay"


def test_blind_code_scores_about_zero():
    """A code carrying no shell information must score near zero.

    THIS IS THE TEST THAT FAILS IF CENTRING IS DROPPED. Concept vectors are firing rates and
    therefore non-negative, so uncentred cosine between any two of them is compressed near 1
    and every encoder would look identical. These synthetic vectors are already zero-mean, so
    the guard here is on the statistic; `test_centring_changes_the_answer` covers the
    non-negative case directly.
    """
    depths = (1, 2, 3, 4, 5, 6)
    sim = similarity_matrix(_shell_blind(depths))
    s = sensitivity_from_similarity(sim)
    assert abs(s) < 0.02, f"blind code scored {s:.4f}, expected about zero"


def test_centring_changes_the_answer_on_non_negative_vectors():
    """Firing rates are non-negative. Without centring the statistic is crushed.

    Builds shell-structured concepts and adds a large positive offset to every vector, which
    is what a rate code looks like. The centred statistic must still detect the structure; an
    uncentred one would not, which is why centring is not optional.
    """
    depths = (1, 2, 3, 4, 5, 6)
    concepts = {d: v + 5.0 for d, v in _shell_structured(depths).items()}
    s_centred = sensitivity_from_similarity(similarity_matrix(concepts, centre=True))
    s_raw = sensitivity_from_similarity(similarity_matrix(concepts, centre=False))
    assert s_centred > 0.05, f"centred statistic lost the structure: {s_centred:.4f}"
    assert s_raw < s_centred / 2, (
        f"uncentred {s_raw:.4f} is not much smaller than centred {s_centred:.4f}; the offset "
        "should have crushed it, so centring may not be doing anything"
    )


def test_shuffled_labels_collapse_the_statistic():
    """Permuting which shell each vector belongs to must destroy the signal.

    Guards against the statistic measuring a sampling artefact - unequal shell sizes, say -
    rather than real structure.
    """
    depths = (1, 2, 3, 4, 5, 6)
    concepts = _shell_structured(depths)
    intact = sensitivity_from_similarity(similarity_matrix(concepts))

    pooled = torch.cat([concepts[d] for d in depths])
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(pooled.shape[0], generator=g)
    pooled = pooled[perm]
    sizes = [concepts[d].shape[0] for d in depths]
    shuffled, start = {}, 0
    for d, n in zip(depths, sizes):
        shuffled[d] = pooled[start:start + n]
        start += n

    scrambled = sensitivity_from_similarity(similarity_matrix(shuffled))
    assert scrambled < intact / 3, (
        f"shuffling labels left {scrambled:.4f} against an intact {intact:.4f}; the statistic "
        "is not measuring shell structure"
    )


def test_similarity_excludes_self_pairs():
    """A vector's similarity with itself is always 1.0 and would inflate the |dd|=0 term.

    With a single vector per shell the within-shell entry is undefined, so it must be absent
    rather than silently 1.0.
    """
    g = torch.Generator().manual_seed(0)
    concepts = {1: torch.randn(1, 8, generator=g), 2: torch.randn(1, 8, generator=g)}
    sim = similarity_matrix(concepts)
    assert (1, 1) not in sim, "a one-vector shell produced a self-similarity entry"
    assert (1, 2) in sim


def test_real_encoder_is_deterministic(tmp_path):
    """Poisson spiking makes encoding stochastic. The same seed must reproduce S exactly.

    Uses a bounded BFS build (depth 3) and a small sample so the test stays fast.
    """
    from neuromorphic.analysis.sequence_sensitivity import sequence_sensitivity
    from neuromorphic.envs.cube_distance import ExactBFSDistance
    from neuromorphic.training.encoder_pretrain import make_sensory

    provider = ExactBFSDistance(max_depth=3)
    sensory = make_sensory(0)
    a = sequence_sensitivity(sensory, provider, depths=(1, 2, 3), n_per_shell=12, seed=5)
    b = sequence_sensitivity(sensory, provider, depths=(1, 2, 3), n_per_shell=12, seed=5)
    assert a["S"] == b["S"], f"non-deterministic: {a['S']} != {b['S']}"
    assert a["n_by_shell"] == {1: 6, 2: 12, 3: 12}, (
        f"shell sampling changed: {a['n_by_shell']}. Depth 1 has only 6 states in total."
    )


def test_shell_concepts_do_not_retain_an_autograd_graph():
    """60 encoders x 6 shells, each retaining 32 timesteps of LIF activations for a
    gradient nobody takes, is pure waste. It is also a warning generator downstream.
    """
    from neuromorphic.analysis.sequence_sensitivity import shell_concepts, sample_shells
    from neuromorphic.envs.cube_distance import ExactBFSDistance
    from neuromorphic.training.encoder_pretrain import make_sensory
    import random, torch

    provider = ExactBFSDistance(max_depth=2)
    shells = sample_shells(provider, (1, 2), 6, rng=random.Random(0))
    concepts = shell_concepts(make_sensory(0), shells,
                              generator=torch.Generator().manual_seed(0))
    for d, m in concepts.items():
        assert not m.requires_grad, f"shell {d} concepts still carry a graph"
        assert m.grad_fn is None, f"shell {d} concepts still have grad_fn {m.grad_fn}"


def test_s_cross_matches_hand_arithmetic():
    """Pins `sensitivity_from_similarity(sim, min_separation=1)` - S_cross - against a slope
    computed by hand on a 3-shell toy `sim` dict.

    sim = {(1,1): 1.0, (2,2): 1.0, (3,3): 1.0, (1,2): 0.8, (2,3): 0.4, (1,3): -1.0}

    Full S (all six pairs, including the three |dd|=0 within-shell entries):
      xs = [0, 0, 0, 1, 1, 2], ys = [1, 1, 1, 0.8, 0.4, -1.0]
      slope = -0.88  ->  S = 0.88

    S_cross (dropping the |dd|=0 entries, leaving only (1,2), (2,3), (1,3)):
      xs = [1, 1, 2], ys = [0.8, 0.4, -1.0]
      slope = -1.6  ->  S_cross = 1.6

    Both were checked independently in Python before being pinned here; this test exists so a
    future change to the fit or the filter has an exact number to break against.
    """
    sim = {(1, 1): 1.0, (2, 2): 1.0, (3, 3): 1.0, (1, 2): 0.8, (2, 3): 0.4, (1, 3): -1.0}
    s = sensitivity_from_similarity(sim)
    s_cross = sensitivity_from_similarity(sim, min_separation=1)
    assert s == pytest.approx(0.88)
    assert s_cross == pytest.approx(1.6)


def test_s_cross_differs_from_s_on_a_fixture_that_clusters_without_ordering():
    """THE WHOLE POINT of S_cross. `_shell_structured`'s shell centres are INDEPENDENT RANDOM
    DIRECTIONS - shells are tight clusters, but nothing orders them by distance. On exactly
    this fixture, `S` measures 0.26708 while `S_cross` measures 0.02318: `S` is about 91%
    driven by the |dd|=0 within-shell term alone, and a code that merely clusters (rather than
    being gradiently ordered by distance) scores near zero once that term is excluded.

    This test must FAIL if S_cross accidentally still includes the |dd|=0 term - in that case
    it would equal S, not differ sharply from it.
    """
    depths = (1, 2, 3, 4, 5, 6)
    sim = similarity_matrix(_shell_structured(depths))
    s = sensitivity_from_similarity(sim)
    s_cross = sensitivity_from_similarity(sim, min_separation=1)
    assert s == pytest.approx(0.26708, abs=1e-4)
    assert s_cross == pytest.approx(0.02318, abs=1e-4)
    assert s_cross < s / 5, (
        f"S_cross ({s_cross:.4f}) is not much smaller than S ({s:.4f}); it may still include "
        "the |dd|=0 within-shell term"
    )


def test_a_degenerate_similarity_map_raises_rather_than_reporting_zero():
    """A single separation cannot define a slope. Returning 0.0 there would be
    indistinguishable from a real 'no decay' measurement, and those are different
    states of the world.
    """
    import pytest
    from neuromorphic.analysis.sequence_sensitivity import sensitivity_from_similarity
    with pytest.raises(ValueError):
        sensitivity_from_similarity({(1, 1): 0.9})
