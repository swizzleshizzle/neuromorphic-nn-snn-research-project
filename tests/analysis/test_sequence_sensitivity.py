"""EXP-054: the sequence-sensitivity statistic, and the proof it measures structure.

The metric trains NOTHING - no classifier, no fitted parameters beyond a slope over
already-computed similarities. That is deliberate: every instrument this project has retired
(the EXP-033 decodability probe, pretraining move-accuracy, the entropy trace) was a trained
linear probe or a pretext score, and a statistic with no capacity cannot overfit into a story.

The tests below build encoders with KNOWN structure so the statistic can be checked against
an answer rather than against itself.
"""

from __future__ import annotations

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
