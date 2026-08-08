"""EXP-038's decision rules, measured on cases whose answer is already known.

Three synthetic worlds with verdicts that are not in doubt. If `aggregate.py` cannot separate
them here, it cannot separate them on the real data either, and a 21-hour run would be scored
by a rule that does not work. This is the check that caught EXP-036's break bar before it cost
anything.

The comparator arms are synthetic too, deliberately: EXP-036's real records are gitignored, so
a test that needed them would silently skip on a fresh checkout, which is the sort of
can-never-fail assertion this repo has been bitten by four times.

The load-bearing case is `randomization`. EXP-032 Finding 3 established that the entropy bonus
lowers modal fraction by INJECTING RANDOMNESS rather than by teaching the policy to read its
input, and at depth 6 the random floor (0.0008) sits ABOVE the trained result (0.0000). So a
cell can clear a naive success bar purely by going uniform. That world MUST NOT be reported as
a lever, and this test fails if it ever is.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AGG = REPO / "experiments" / "038_depth6_collapse" / "aggregate.py"
SEEDS = list(range(12))

# Measured from EXP-036's random arms. NOT 0.354, which is the 9-step figure; depth 6 runs a
# 15-step budget and depth 5 a 13-step one, and modal fraction falls with budget length.
UNIFORM_MODAL = {5: 0.321, 6: 0.309}
ENTROPY_SATURATED = 1.613   # 90% of log 6


def _rec(depth, seed, beta, success, modal, entropy, arm="regionalized"):
    return {
        "depth": depth, "seed": seed, "arm": arm, "tag": f"t_b{beta}_d{depth}",
        "readout": "concept", "success_rate": success,
        "greedy_modal_action_frac": modal, "mean_train_entropy": entropy,
        "config": {"entropy_beta": beta, "normalize_advantages": True,
                   "depth": depth, "seed": seed},
    }


def _write(d: Path, records, prefix):
    d.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(records):
        (d / f"{prefix}_{i:04d}.json").write_text(json.dumps(r))


def _comparators():
    """Stand-ins for EXP-036, matching its measured values.

    depth 6 random 0.0008 ABOVE depth 6 trained 0.0000 - the inversion that makes the naive
    'did it beat the baseline?' bar unreadable. depth 5 trained 0.0396, its floor 0.0000.
    """
    out = []
    for s in SEEDS:
        out.append(_rec(6, s, 0.0, 0.001 if s < 2 else 0.0, UNIFORM_MODAL[6], 0.0, arm="random"))
        out.append(_rec(6, s, 0.0, 0.0, 0.975, 0.204))
        out.append(_rec(5, s, 0.0, 0.0, UNIFORM_MODAL[5], 0.0, arm="random"))
        out.append(_rec(5, s, 0.0, 0.0396 + (s % 3 - 1) * 0.005, 0.779, 0.236))
    return out


# (beta, success, modal, entropy)
WORLDS = {
    # A real lever: the middle beta succeeds WHILE still selecting (modal 0.60), and the top
    # beta saturates entropy, which is what the dose axis has to demonstrate.
    "lever": [(0.05, 0.001, 0.90, 0.60), (0.2, 0.055, 0.60, 1.36), (0.8, 0.0008, 0.68, 1.70)],
    # The trap EXP-032 Finding 3 describes: the ONLY cell that "succeeds" has fallen to the
    # uniform anchor, so its success cannot be learning.
    "randomization": [(0.05, 0.001, 0.90, 0.60), (0.2, 0.002, 0.70, 1.36),
                      (0.8, 0.055, 0.31, 1.70)],
    # Nothing moves and entropy never saturates: bounded too low, exactly as EXP-032 was.
    "null": [(0.05, 0.0, 0.95, 0.50), (0.2, 0.0, 0.92, 0.80), (0.8, 0.0, 0.88, 1.10)],
}

EXPECT = {
    "lever": ["LEVER ESTABLISHED", "DOSE AXIS SATURATED"],
    "randomization": ["REFUTED BY CLAIM 2", "INSTRUMENT BROKEN"],
    "null": ["REFUTED at this budget", "SPAN STILL TOO LOW"],
}

FORBIDDEN = {
    "randomization": ["LEVER ESTABLISHED"],
    "null": ["LEVER ESTABLISHED", "DOSE AXIS SATURATED"],
    "lever": ["REFUTED BY CLAIM 2", "SPAN STILL TOO LOW", "INSTRUMENT BROKEN"],
}


def _run_world(tmp_path: Path, world: str) -> str:
    exp038 = []
    for beta, succ, modal, entropy in WORLDS[world]:
        for s in SEEDS:
            # Per-seed jitter so the permutation test is not fed 12 identical values.
            exp038.append(_rec(6, s, beta, max(0.0, succ + (s % 3 - 1) * succ * 0.15),
                               modal, entropy))
    for s in SEEDS:
        # depth-5 coherence arm, deliberately flat against the 0.0396 comparator
        exp038.append(_rec(5, s, 0.2, 0.040 + (s % 3 - 1) * 0.004, 0.75, 0.5))

    _write(tmp_path / "e038", exp038, "e038")
    _write(tmp_path / "e036", _comparators(), "e036")

    proc = subprocess.run(
        [sys.executable, str(AGG),
         "--out-dir", str(tmp_path / "e038"),
         "--exp036-dir", str(tmp_path / "e036")],
        capture_output=True, text=True)
    assert proc.returncode == 0, f"aggregate.py exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout + proc.stderr


@pytest.mark.parametrize("world", sorted(WORLDS))
def test_verdict_matches_known_world(tmp_path, world):
    out = _run_world(tmp_path, world)
    for want in EXPECT[world]:
        assert want in out, f"[{world}] missing expected verdict {want!r}\n{out}"


@pytest.mark.parametrize("world", sorted(FORBIDDEN))
def test_forbidden_verdicts_never_emitted(tmp_path, world):
    out = _run_world(tmp_path, world)
    for bad in FORBIDDEN[world]:
        assert bad not in out, f"[{world}] emitted forbidden verdict {bad!r}\n{out}"


def test_randomization_world_would_pass_a_naive_bar(tmp_path):
    """The reason Claim 2 exists at all.

    This asserts the DEFECT is real: in the randomization world the winning cell clears the
    0.02 success bar and beats the trained 0.0000 baseline outright, so a rule written only on
    success would call it a lever. Only its modal fraction at the uniform floor gives it away.
    Without this, Claim 2 could be deleted and the suite would stay green.
    """
    beta, succ, modal, entropy = WORLDS["randomization"][-1]
    assert succ >= 0.02, "the trap cell must clear the naive success bar"
    assert modal <= UNIFORM_MODAL[6] + 0.01, "the trap cell must be at the uniform anchor"

    out = _run_world(tmp_path, "randomization")
    assert "REFUTED BY CLAIM 2" in out, "the modal-fraction rule is what catches it"


def test_entropy_saturation_is_what_the_instrument_check_measures(tmp_path):
    """The correction the pilot forced, pinned so it cannot silently regress.

    The first draft checked that the top beta drove modal fraction to the uniform anchor. The
    pilot measured entropy at 95% of ceiling with modal stalled at 0.675, because entropy
    describes the SAMPLED training policy while modal describes the GREEDY argmax at eval.
    That check could never have fired. The saturation criterion is entropy, and these two
    worlds differ ONLY in the top cell's entropy.
    """
    assert WORLDS["lever"][-1][3] >= ENTROPY_SATURATED
    assert WORLDS["null"][-1][3] < ENTROPY_SATURATED
    # ...and their top-cell modal fractions are both far above the anchor, so a modal-based
    # rule could not tell them apart at all.
    assert WORLDS["lever"][-1][2] > UNIFORM_MODAL[6] + 0.3
    assert WORLDS["null"][-1][2] > UNIFORM_MODAL[6] + 0.3

    assert "DOSE AXIS SATURATED" in _run_world(tmp_path, "lever")
    assert "SPAN STILL TOO LOW" in _run_world(tmp_path, "null")
