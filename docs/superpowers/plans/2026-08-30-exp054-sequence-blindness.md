# EXP-054 Sequence-Blindness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the sensory concept loses the ability to distinguish states at different distances from solved as inverse-model pretraining continues, using a metric that trains nothing.

**Architecture:** One library module computes the statistic as pure functions over a `SensoryCortex` and a BFS distance provider. One driver iterates 60 encoders already on disk and writes a JSON record each. One aggregator applies four pre-registered claims, the fourth of which is a hard disqualifier implemented as a tested function. No training, no policy runs, no laptop.

**Tech Stack:** Python 3.10, PyTorch + snnTorch, pytest. No scipy: exact permutation tests over `2**12` sign flips.

**Spec:** `docs/superpowers/specs/2026-08-29-exp054-sequence-blindness-design.md` (pre-registered at `fb885ec`). Read section 3 before touching the aggregator: a random encoder is expected to score HIGHEST, and that is the result rather than a failure of it.

## Global Constraints

- **Always run python via `.venv/bin/python`.** Never a bare `python`.
- **Always pass an explicit timeout on Bash calls. The tool default is 120 s** and anything over it is auto-backgrounded by the harness, which stalls agents. Pass 600000.
- **Never run the full test suite** (about 13 min, exceeds any single call). Run only the files named in each task.
- **No em-dashes** anywhere in code, comments, docstrings or commit messages. Use " - ".
- **Plain commit messages. No `Co-Authored-By`. No "Generated with".**
- **Never `git add -A`.** Stage explicit paths.
- **Never write an assertion that cannot fail.** Every test must fail against pre-change or deliberately-broken code. Prefer a measured numeric threshold to a qualitative check: prototype first, then set the bar with margin.
- **Never weaken a passing threshold** to make something else pass. If a change would require that, stop and report it.
- **Distance-to-solved is an instrument, never a model input.** It is used here only as an offline analysis label, exactly as EXP-033 used depth labels. Nothing about it reaches an encoder.
- **`N_ACTIONS` comes from `len(MOVES)`, never a literal.**
- The laptop is busy with EXP-053. **This experiment must not touch it.**

---

### Task 1: The statistic

**Files:**
- Create: `src/neuromorphic/analysis/sequence_sensitivity.py`
- Test: `tests/analysis/test_sequence_sensitivity.py`

**Interfaces:**
- Consumes, all already committed and working:
  - `neuromorphic.envs.cube_distance.ExactBFSDistance(max_depth=int)` with `.states_at_distance(d) -> list[tuple[int, ...]]`. A bounded build is near free (depth 6, 11,913 states, about 0.04 s).
  - `neuromorphic.training.encoder_pretrain.states_to_obs(states) -> torch.Tensor` giving `[N, 24]` long.
  - `neuromorphic.training.encoder_pretrain.concept_rates(sensory, obs, num_steps=32, generator=None) -> torch.Tensor` giving `[N, content]`. This is BATCHED, which is why this experiment is cheap.
  - `neuromorphic.training.encoder_pretrain.make_sensory(seed, content=64, num_steps=32) -> SensoryCortex` for a fresh random encoder.
  - `neuromorphic.training.encoder_pretrain.load_encoder(path, seed=0, content=64, num_steps=32) -> SensoryCortex` for a saved one.
- Produces:
  - `sample_shells(provider, depths, n_per_shell, rng) -> dict[int, list[tuple[int, ...]]]`
  - `shell_concepts(sensory, shells, generator) -> dict[int, torch.Tensor]`
  - `similarity_matrix(concepts) -> dict[tuple[int, int], float]` keyed by `(d1, d2)` with `d1 <= d2`
  - `sensitivity_from_similarity(sim) -> float`
  - `sequence_sensitivity(sensory, provider, *, depths=(1,2,3,4,5,6), n_per_shell=60, seed=0) -> dict` returning `{"S": float, "sim": dict, "n_by_shell": dict}`

- [ ] **Step 1: Write the failing tests**

Create `tests/analysis/test_sequence_sensitivity.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/analysis/test_sequence_sensitivity.py -v
```

Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `sequence_sensitivity`.

If `tests/analysis/` does not exist, create it with an `__init__.py` only if its sibling
`tests/training/` has one; match whatever the repo already does.

- [ ] **Step 3: Write the module**

Create `src/neuromorphic/analysis/sequence_sensitivity.py`:

```python
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
    """
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


def sensitivity_from_similarity(sim):
    """`S` = the negated least-squares slope of similarity against `|d1 - d2|`.

    Positive `S` means similarity falls as shells get further apart, i.e. the code carries
    distance structure. Zero means it does not. Returns 0.0 when fewer than two distinct
    separations are present, because a slope is undefined there.
    """
    xs = [float(abs(d2 - d1)) for (d1, d2) in sim]
    ys = [sim[k] for k in sim]
    if len(set(xs)) < 2:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
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
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/analysis/test_sequence_sensitivity.py -v
```

Expected: 5 passed. Record the observed `S` values from `test_structured_code_scores_high` and
`test_blind_code_scores_about_zero` in your report; if the structured value is below 0.05 or the
blind value above 0.02, report it rather than adjusting the bar.

- [ ] **Step 5: Prove determinism on a real encoder**

Add to the same test file:

```python
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
```

Run it:

```bash
.venv/bin/python -m pytest tests/analysis/test_sequence_sensitivity.py -v
```

Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/neuromorphic/analysis/sequence_sensitivity.py tests/analysis/test_sequence_sensitivity.py
git commit -m "EXP-054: the sequence-sensitivity statistic

Measures how concept similarity decays with exact-BFS shell separation. Trains nothing -
no classifier, no probe, no fitted parameters beyond a slope over already-computed
similarities. Every instrument this project retired was a trained linear probe or a
pretext score, and a statistic with no capacity cannot overfit into a story.

Centring is load-bearing: concept vectors are firing rates and therefore non-negative, so
uncentred cosine is compressed near 1 and every encoder would look identical. One test
adds a large positive offset and asserts the uncentred statistic collapses while the
centred one survives.

Self-pairs are excluded from the within-shell term, and a shuffled-label null asserts the
statistic collapses when shell membership is destroyed."
```

---

### Task 2: The driver

**Files:**
- Create: `experiments/054_sequence_blindness/run.py`
- Test: `tests/analysis/test_exp054_arms.py`

**Interfaces:**
- Consumes `sequence_sensitivity` from Task 1, plus `make_sensory` and `load_encoder` from `neuromorphic.training.encoder_pretrain`.
- Produces:
  - `ARMS: dict[str, dict]` mapping arm name to `{"epochs": int, "path_fn": callable | None, "policy": float}`
  - `encoder_for(arm: str, seed: int) -> SensoryCortex`
  - `record_filename(arm: str, seed: int) -> str` giving `exp054_{arm}_s{seed}.json`
  - `POLICY_SOURCES: dict[str, tuple[Path, str]]` mapping arm to (records dir, tag)
  - `policy_by_seed(arm: str) -> dict[int, float]`
  - records with keys `arm`, `epochs`, `seed`, `S`, `sim`, `n_by_shell`, `policy_success`,
    where `policy_success` is that SEED's held-out success, not the arm mean

- [ ] **Step 1: Write the failing test**

Create `tests/analysis/test_exp054_arms.py`:

```python
"""EXP-054: the five arms must point at the right encoders, and E0 must be reconstructed.

A wrong path here would silently measure the wrong pretraining level, and the whole point of
the experiment is the epoch series. A config-level test costs seconds; discovering it after a
run costs the run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUN_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "054_sequence_blindness" / "run.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp054_run", RUN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_five_arms_with_the_pre_registered_epoch_levels():
    arms = _module().ARMS
    assert sorted(a["epochs"] for a in arms.values()) == [0, 10, 20, 40, 80]


def test_e0_is_reconstructed_not_loaded():
    """A random init is exactly reproducible from its seed and no file was ever saved, so E0
    must build rather than load. A path here would be a file that does not exist."""
    assert _module().ARMS["E0"]["path_fn"] is None


def test_each_trained_arm_points_at_its_own_experiment():
    arms = _module().ARMS
    expected = {
        "E10": "052_pretraining_optimum",
        "E20": "052_pretraining_optimum",
        "E40": "040_pretrained_encoder_policy",
        "E80": "050_objective_vs_gradient",
    }
    for name, frag in expected.items():
        path = str(arms[name]["path_fn"](0))
        assert frag in path, f"{name} resolves to {path}, which is not {frag}"


def test_e10_and_e20_are_distinguishable_paths():
    """Both live in the same directory and differ only by the epoch tag. A copy-paste error
    would make them the same file and the 10-vs-20 contrast would be exactly zero."""
    arms = _module().ARMS
    assert str(arms["E10"]["path_fn"](3)) != str(arms["E20"]["path_fn"](3))


def test_policy_values_match_the_pre_registered_series():
    arms = _module().ARMS
    assert arms["E0"]["policy"] == pytest.approx(0.0000)
    assert arms["E10"]["policy"] == pytest.approx(0.2012)
    assert arms["E20"]["policy"] == pytest.approx(0.1850)
    assert arms["E40"]["policy"] == pytest.approx(0.1800)
    assert arms["E80"]["policy"] == pytest.approx(0.0887)


def test_record_filenames_do_not_collide():
    m = _module()
    names = [m.record_filename(a, s) for a in m.ARMS for s in range(12)]
    assert len(set(names)) == len(names)


def test_per_seed_policy_has_within_arm_variance():
    """Claim 4 is a HARD disqualifier and it correlates S against WITHIN-arm policy.

    If policy were constant across an arm's seeds there would be no variance to correlate
    against, the rule could never fire, and a decorative hard rule is worse than none. This
    asserts the lookup returns 12 genuinely different values per trained arm.
    """
    m = _module()
    for arm in ("E10", "E20", "E40", "E80"):
        pol = m.policy_by_seed(arm)
        assert len(pol) >= 12, f"{arm} returned {len(pol)} seeds, expected at least 12"
        vals = [pol[s] for s in range(12)]
        assert len(set(vals)) > 1, (
            f"{arm} policy is constant across seeds ({vals[0]}); the tag lookup is wrong and "
            "Claim 4 would be uncomputable"
        )


def test_e0_policy_is_all_zero():
    """EXP-036 measured every seed at exactly 0.0000. That is why the spec excludes E0 from
    Claim 4 while keeping it for the Claim 3 floor."""
    pol = _module().policy_by_seed("E0")
    assert set(pol.values()) == {0.0}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/analysis/test_exp054_arms.py -v
```

Expected: FAIL, `run.py` does not exist.

- [ ] **Step 3: Write the driver**

Create `experiments/054_sequence_blindness/run.py`:

```python
"""EXP-054: is the concept sequence-blind, and does that explain the pretraining collapse?

Five arms of 12 seeds. Every encoder ALREADY EXISTS on disk (E0 is rebuilt from its seed).
Nothing is trained, no policy is run, and this must not touch the laptop - EXP-053's arms are
using it.

    epochs   move-accuracy   depth-6 policy
      0            -            0.0000     random frozen encoder (EXP-036)
     10          0.383          0.2012
     20          0.414          0.1850
     40          0.437          0.1800     the inherited value (EXP-043)
     80          0.452          0.0887     (EXP-050)

THE PARADOX: the pretext metric climbs monotonically while the policy halves. EXP-052
established that and could not say why.

PRE-REGISTERED CONTRACT: docs/superpowers/specs/2026-08-29-exp054-sequence-blindness-design.md

  1. PRIMARY: does S fall with epochs? Confirmed if it decreases in at least 2 of the 3
     adjacent contrasts (10-20, 20-40, 40-80) at p <= 0.05.
  2. THE TRADEOFF: report S beside move-accuracy and policy.
  3. THE FLOOR: E0. A random encoder may score HIGHEST - random projections preserve geometry -
     and that is the result, not a failure of it. See spec section 3.
  4. THE DISQUALIFIER: if S correlates with policy in OPPOSITE directions within and between
     arms, S is retired on the spot. That is exactly how the entropy trace behaved.

Run (repo root):
    .venv/bin/python -u experiments/054_sequence_blindness/run.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from neuromorphic.analysis.sequence_sensitivity import DEPTHS, N_PER_SHELL, sequence_sensitivity
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.encoder_pretrain import load_encoder, make_sensory

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
E052 = Path("experiments/052_pretraining_optimum/outputs")
E040 = Path("experiments/040_pretrained_encoder_policy/outputs")
E050 = Path("experiments/050_objective_vs_gradient/outputs")

SEEDS = tuple(range(12))

ARMS = {
    # E0 is REBUILT, not loaded: a random init is exactly reproducible from its seed and no
    # file was ever saved for it.
    "E0":  {"epochs": 0,  "path_fn": None,                                        "policy": 0.0000},
    "E10": {"epochs": 10, "path_fn": lambda s: E052 / f"exp052_encoder_e10_s{s}.pt", "policy": 0.2012},
    "E20": {"epochs": 20, "path_fn": lambda s: E052 / f"exp052_encoder_e20_s{s}.pt", "policy": 0.1850},
    "E40": {"epochs": 40, "path_fn": lambda s: E040 / f"exp040_encoder_s{s}.pt",     "policy": 0.1800},
    "E80": {"epochs": 80, "path_fn": lambda s: E050 / f"exp050_encoder_plus_s{s}.pt", "policy": 0.0887},
}

# PER-SEED policy, which Claim 4 requires. The arm means above are for the printed table only.
# Claim 4 correlates S against policy WITHIN an arm, and an arm-constant policy has no variance
# to correlate against - the disqualifier could then never fire and the hard rule would be
# decorative. Verified 2026-08-30 that all four tags hold 12 depth-6 records each.
E043 = Path("experiments/043_cap_at_depth_5_6/outputs")
POLICY_SOURCES = {
    "E10": (E052, "exp052_e10_d6"),
    "E20": (E052, "exp052_e20_d6"),
    "E40": (E043, "exp043_capped_d6"),
    "E80": (E050, "exp050_pre2_d6"),
    # E0 has no records: EXP-036 measured every seed at exactly 0.0000. It is excluded from
    # Claim 4 for exactly that reason and carries the constant here.
}


def policy_by_seed(arm: str) -> dict:
    """That arm's per-seed held-out success, read from the experiment that measured it."""
    if arm == "E0":
        return {s: 0.0 for s in SEEDS}
    directory, tag = POLICY_SOURCES[arm]
    out = {}
    for p in Path(directory).glob("*.json"):
        r = json.loads(p.read_text())
        if isinstance(r, dict) and r.get("tag") == tag and r.get("depth") == 6:
            out[int(r["seed"])] = float(r["success_rate"])
    return out


def record_filename(arm: str, seed: int) -> str:
    return f"exp054_{arm}_s{seed}.json"


def encoder_for(arm: str, seed: int):
    """The arm's encoder for one seed. E0 is reconstructed; every other arm is loaded."""
    path_fn = ARMS[arm]["path_fn"]
    if path_fn is None:
        return make_sensory(seed)
    return load_encoder(path_fn(seed), seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    for arm in args.arms:
        fn = ARMS[arm]["path_fn"]
        if fn is None:
            continue
        missing += [str(fn(s)) for s in args.seeds if not fn(s).exists()]
    if missing:
        raise SystemExit(f"missing encoder files: {missing[:5]} ({len(missing)} total)")

    cells = [(a, s) for a in args.arms for s in args.seeds]
    print(f"EXP-054: {len(cells)} encoders, shells {DEPTHS}, up to {N_PER_SHELL} states each")
    print("  the statistic trains NOTHING. No policy runs. Does not touch the laptop.")
    print("  E0 may score HIGHEST (random projections preserve geometry) - see spec section 3.\n",
          flush=True)
    if args.dry_run:
        print(f"  --dry-run: {len(cells)} cell(s) NOT measured.")
        return

    # One bounded build, reused for every encoder. Depth 6 is 11,913 states, about 0.04s.
    provider = ExactBFSDistance(max_depth=max(DEPTHS))

    policies = {}
    for arm in args.arms:
        policies[arm] = policy_by_seed(arm)
        absent = [s for s in args.seeds if s not in policies[arm]]
        if absent:
            raise SystemExit(
                f"arm {arm} is missing per-seed policy for seeds {absent}. Claim 4 correlates S "
                "against WITHIN-arm policy variance and cannot be computed without it."
            )

    for i, (arm, seed) in enumerate(cells, 1):
        out_path = args.out_dir / record_filename(arm, seed)
        if args.skip_existing and out_path.exists():
            continue
        result = sequence_sensitivity(encoder_for(arm, seed), provider, seed=seed)
        record = {
            "arm": arm,
            "epochs": ARMS[arm]["epochs"],
            "seed": seed,
            "policy_success": policies[arm][seed],
            "arm_mean_policy": ARMS[arm]["policy"],
            **result,
        }
        out_path.write_text(json.dumps(record), encoding="utf-8")
        print(f"  {i}/{len(cells)}  {arm} s{seed}  S={result['S']:+.4f}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/analysis/test_exp054_arms.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Dry-run, then measure one arm to check the cost**

```bash
.venv/bin/python experiments/054_sequence_blindness/run.py --dry-run
```

Expected: prints 60 cells, measures nothing, and confirms every encoder file exists.

Then time a single arm so the full cost is known before committing to it:

```bash
.venv/bin/python -u experiments/054_sequence_blindness/run.py --arms E0 --seeds 0 1
```

Expected: two `S` values printed. Report the wall clock. If two cells take more than 60 s, say
so rather than launching all 60.

- [ ] **Step 6: Commit**

```bash
git add experiments/054_sequence_blindness/run.py tests/analysis/test_exp054_arms.py
git commit -m "EXP-054: driver over the five pretraining levels

Five arms of 12 seeds, every encoder already on disk. E0 is rebuilt from its seed rather
than loaded, because a random init is exactly reproducible and no file was ever saved.

A config-level test pins each arm to its own experiment directory and asserts E10 and E20
resolve to different files - they live in one directory and differ only by an epoch tag,
so a copy-paste error would make the 10-vs-20 contrast exactly zero while looking fine."
```

---

### Task 3: The aggregator, with Claim 4 as a hard disqualifier

**Files:**
- Create: `experiments/054_sequence_blindness/aggregate.py`
- Test: `tests/analysis/test_exp054_aggregate.py`

**Interfaces:**
- Consumes records written by Task 2.
- Produces:
  - `permutation_p(diffs) -> float`
  - `spearman(x, y) -> float`
  - `claim4_verdict(within: dict[str, float], between: float) -> str`
  - a printed report covering Claims 1 to 4

- [ ] **Step 1: Write the failing test**

Create `tests/analysis/test_exp054_aggregate.py`:

```python
"""EXP-054: the aggregator's decision rules, tested before any number exists.

Claim 4 is a HARD DISQUALIFIER, not a caveat, and the test below is what keeps it hard. This
project has retired three instruments that each moved opposite to policy quality, and each was
reported with caveats that did not stop later experiments from building inferences on them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

AGG_PATH = (Path(__file__).resolve().parents[2]
            / "experiments" / "054_sequence_blindness" / "aggregate.py")


def _module():
    spec = importlib.util.spec_from_file_location("exp054_agg", AGG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_permutation_p_on_a_known_case():
    """Twelve all-positive differences is the most extreme two-sided outcome: 2/4096."""
    assert _module().permutation_p([0.1] * 12) == pytest.approx(2 / 4096)


def test_spearman_on_a_monotone_case():
    m = _module()
    assert m.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert m.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_opposite_signs_retire_the_metric():
    """THE DISQUALIFIER. Within-arm positive, between-arm negative - exactly how the entropy
    trace behaved - must retire S on the spot, in the headline and not in a caveat."""
    verdict = _module().claim4_verdict(
        within={"E10": 0.7, "E20": 0.6, "E40": 0.8, "E80": 0.5}, between=-0.9)
    assert "RETIRED" in verdict
    assert "fifth inverted instrument" in verdict


def test_agreeing_signs_keep_the_metric():
    verdict = _module().claim4_verdict(
        within={"E10": 0.7, "E20": 0.6, "E40": 0.8, "E80": 0.5}, between=0.9)
    assert "RETIRED" not in verdict


def test_a_mixed_within_arm_picture_does_not_silently_pass():
    """If the within-arm correlations disagree with EACH OTHER, there is no coherent within
    sign to compare against, and the aggregator must say so rather than pick one."""
    verdict = _module().claim4_verdict(
        within={"E10": 0.7, "E20": -0.6, "E40": 0.8, "E80": -0.5}, between=0.9)
    assert "INCONCLUSIVE" in verdict
    assert "RETIRED" not in verdict
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
.venv/bin/python -m pytest tests/analysis/test_exp054_aggregate.py -v
```

Expected: FAIL, `aggregate.py` does not exist.

- [ ] **Step 3: Write the aggregator**

Create `experiments/054_sequence_blindness/aggregate.py`:

```python
"""EXP-054 aggregator: apply the pre-registered rules to the records on disk.

Thresholds committed in the spec before any number existed.

> A RANDOM ENCODER MAY SCORE HIGHEST and that is the RESULT, not a failure of it. Random
> projections preserve geometry, so E0 separating shells best while scoring 0.0000 policy would
> say the 10-epoch optimum is a tradeoff: pretraining buys move-structure and spends
> sequence-structure from the first epoch. This is written down in the spec, in advance.

Usage:
    .venv/bin/python experiments/054_sequence_blindness/aggregate.py
"""

from __future__ import annotations

import itertools
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent

ALPHA = 0.05
ADJACENT = (("E10", "E20"), ("E20", "E40"), ("E40", "E80"))
MOVE_ACCURACY = {"E0": None, "E10": 0.383, "E20": 0.414, "E40": 0.437, "E80": 0.452}
ORDER = ("E0", "E10", "E20", "E40", "E80")


def permutation_p(diffs) -> float:
    """Exact two-sided paired permutation over all 2**n sign flips. No scipy in the venv."""
    n, obs = len(diffs), abs(sum(diffs))
    return sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12) / 2 ** n


def spearman(x, y) -> float:
    """Rank correlation. n is 4 or 12 here, so an exact library is not needed."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        out = [0] * len(v)
        for pos, i in enumerate(order):
            out[i] = pos + 1
        return out

    rx, ry = rank(list(x)), rank(list(y))
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def claim4_verdict(within: dict, between: float) -> str:
    """The pre-registered disqualifier, as a function rather than a paragraph.

    Encoding it removes the step where a human re-derives the rule while looking at the
    numbers. EXP-050's Claim 4 was satisfied and its inference was still wrong; EXP-052's
    aggregator named a shape from indistinguishable means. This is the response to both.
    """
    signs = {k: (1 if v > 0 else -1 if v < 0 else 0) for k, v in within.items()}
    nonzero = [s for s in signs.values() if s != 0]
    if not nonzero or len(set(nonzero)) > 1:
        return ("CLAIM 4 INCONCLUSIVE. The within-arm correlations disagree with each other "
                f"({within}), so there is no coherent within-arm sign to compare against the "
                f"between-arm {between:+.3f}. S is neither cleared nor retired; report the "
                "correlations and do not use S as a diagnostic until this resolves.")
    within_sign = nonzero[0]
    between_sign = 1 if between > 0 else -1 if between < 0 else 0
    if between_sign != 0 and within_sign != between_sign:
        return ("CLAIM 4 TRIPPED. S IS RETIRED. The within-arm correlations "
                f"({within}) and the between-arm correlation ({between:+.3f}) carry OPPOSITE "
                "SIGNS. S is a fifth inverted instrument, alongside the EXP-033 probe, "
                "pretraining move-accuracy and the entropy trace. It may not be used as a "
                "diagnostic and may not appear in a later spec. Report this in the headline, "
                "not in a caveat.")
    return ("CLAIM 4 PASSED. Within-arm and between-arm correlations agree in sign "
            f"({within}, between {between:+.3f}). S is not disqualified. This is NOT the same "
            "as S being a good predictor of policy - see Claim 2 for the tradeoff reading.")


def load(out_dir: Path) -> dict:
    by_arm = defaultdict(dict)
    for p in Path(out_dir).glob("exp054_*.json"):
        r = json.loads(p.read_text())
        by_arm[r["arm"]][r["seed"]] = r
    return by_arm


def main() -> None:
    by_arm = load(HERE / "outputs")
    if not by_arm:
        print("no records in outputs/; run run.py first")
        return

    print("EXP-054: is the concept sequence-blind?")
    print("The statistic trains nothing. A random encoder may score highest - see spec 3.\n")

    print(f"{'arm':>5} {'epochs':>7} {'S mean':>9} {'S sd':>8} {'move-acc':>9} {'policy':>8}")
    for arm in ORDER:
        if arm not in by_arm:
            continue
        vals = [by_arm[arm][s]["S"] for s in sorted(by_arm[arm])]
        acc = MOVE_ACCURACY[arm]
        pol = next(iter(by_arm[arm].values()))["policy_success"]
        acc_s = "-" if acc is None else f"{acc:.3f}"
        print(f"{arm:>5} {next(iter(by_arm[arm].values()))['epochs']:>7} "
              f"{st.mean(vals):>9.4f} {st.stdev(vals) if len(vals) > 1 else 0.0:>8.4f} "
              f"{acc_s:>9} {pol:>8.4f}")

    print("\nCLAIM 1 PRIMARY - does S fall with pretraining? "
          "CONFIRMED if it decreases in >= 2 of 3 at p <= 0.05.")
    decreases = 0
    for a, b in ADJACENT:
        if a not in by_arm or b not in by_arm:
            continue
        seeds = sorted(set(by_arm[a]) & set(by_arm[b]))
        diffs = [by_arm[b][s]["S"] - by_arm[a][s]["S"] for s in seeds]
        p = permutation_p(diffs)
        fell = st.mean(diffs) < 0 and p <= ALPHA
        decreases += int(fell)
        print(f"  {a} -> {b}   delta {st.mean(diffs):+.4f}   p {p:.4f}   "
              f"{'DECREASE' if fell else 'not significant'}")
    print(f"  => {decreases}/3 significant decreases -> "
          f"{'CONFIRMED' if decreases >= 2 else 'NOT CONFIRMED'}")

    print("\nCLAIM 3 FLOOR - E0 against E10, paired.")
    if "E0" in by_arm and "E10" in by_arm:
        seeds = sorted(set(by_arm["E0"]) & set(by_arm["E10"]))
        diffs = [by_arm["E0"][s]["S"] - by_arm["E10"][s]["S"] for s in seeds]
        print(f"  E0 minus E10   delta {st.mean(diffs):+.4f}   p {permutation_p(diffs):.4f}")
        if st.mean(diffs) > 0:
            print("  E0 is HIGHER: pretraining degrades sequence structure from the first "
                  "epoch, and no amount of it is protective.")

    print("\nCLAIM 4 DISQUALIFIER - does S invert between within-arm and between-arm?")
    within = {}
    for arm in ORDER:
        if arm == "E0" or arm not in by_arm:
            continue   # E0 has zero policy variance across seeds, so no correlation exists
        seeds = sorted(by_arm[arm])
        s_vals = [by_arm[arm][s]["S"] for s in seeds]
        pol = [by_arm[arm][s]["policy_success"] for s in seeds]
        if len(set(pol)) < 2:
            continue   # arm-level policy is a constant here; see the note below
        within[arm] = spearman(s_vals, pol)
    arms_present = [a for a in ORDER if a in by_arm and a != "E0"]
    between = spearman(
        [st.mean([by_arm[a][s]["S"] for s in by_arm[a]]) for a in arms_present],
        [st.mean([by_arm[a][s]["policy_success"] for s in by_arm[a]]) for a in arms_present],
    )
    if not within:
        print("  NO within-arm correlation could be computed: every arm's policy is constant "
              "across seeds. That should not happen now that records carry per-seed policy, so "
              "treat it as a bug in the policy lookup rather than as a result.")
    else:
        print("  " + claim4_verdict(within, between))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
.venv/bin/python -m pytest tests/analysis/test_exp054_aggregate.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Run the aggregator against an empty directory**

```bash
.venv/bin/python experiments/054_sequence_blindness/aggregate.py
```

Expected: prints "no records in outputs/; run run.py first" and does not raise.

- [ ] **Step 6: Verify Claim 4 has real within-arm variance to work with**

Claim 4 is a hard disqualifier and it correlates `S` against policy WITHIN an arm. If the
records carried one policy value per arm, that correlation would have no variance and the rule
could never fire - a decorative hard rule is worse than none.

Confirm the records carry genuine per-seed variance:

```bash
.venv/bin/python -c "
import json, glob, statistics as st, collections
by = collections.defaultdict(list)
for f in glob.glob('experiments/054_sequence_blindness/outputs/exp054_*.json'):
    r = json.load(open(f)); by[r['arm']].append(r['policy_success'])
for a in sorted(by):
    v = by[a]
    print(f'{a}: n={len(v)} mean={st.mean(v):.4f} sd={st.stdev(v) if len(v)>1 else 0:.4f}')
"
```

Expected: every arm except `E0` has a non-zero sd. `E0` must be exactly 0.0000 with sd 0, which
is why the spec excludes it from Claim 4. If any trained arm reports sd 0, stop and report it -
the policy lookup is matching the wrong tag.

- [ ] **Step 7: Commit**

```bash
git add experiments/054_sequence_blindness/aggregate.py tests/analysis/test_exp054_aggregate.py
git commit -m "EXP-054: aggregator, with Claim 4's disqualifier as a tested function

Claim 4 retires S on the spot if it correlates with policy in opposite directions within
and between arms. It is code rather than prose because this project has retired three
instruments that each behaved that way, and each was reported with caveats that did not
stop later experiments from building inferences on them.

The verdict function also refuses to rule when the within-arm correlations disagree with
each other, rather than picking one sign and proceeding."
```
