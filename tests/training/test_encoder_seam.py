"""The EXP-040 encoder seam, and the proof it changed nothing when unused.

`CubeConfig.encoder_state_path` lets EXP-040 inject a pretrained `SensoryCortex` into the
frozen brain. Every cube number since EXP-029 was produced without that field, so the default
path must reproduce them **exactly** or none of them stay comparable. EXP-036 set this
precedent when it added head serialisation.

The neutrality test compares against a baseline captured **before** the field existed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from neuromorphic.training.cube_baseline import (
    CubeConfig,
    make_agent,
    max_steps_for,
    run_cube_baseline,
)
from neuromorphic.training.encoder_pretrain import load_encoder, make_sensory, save_encoder

# Captured 2026-08-09 from the commit immediately before `encoder_state_path` was added, by
# running the exact configs reproduced in `test_default_path_is_byte_identical`.
BASELINE = {
    0: {"success_rate": 0.0, "greedy_modal_action_frac": 0.859259,
        "mean_train_entropy": 1.600457},
    1: {"success_rate": 0.033333, "greedy_modal_action_frac": 0.797037,
        "mean_train_entropy": 1.256079},
}


def _cfg(seed: int, out_dir: Path, **kw) -> CubeConfig:
    base = dict(arm="regionalized", readout="concept", tag="neutral", depth=3,
                seed=seed, sigma=0.0, episodes=60, curriculum=(1, 2, 3),
                max_depth=4, out_dir=out_dir)
    base.update(kw)
    return CubeConfig(**base)


@pytest.mark.slow
@pytest.mark.parametrize("seed", sorted(BASELINE))
def test_default_path_is_byte_identical_to_the_pre_change_baseline(tmp_path, seed):
    """The neutrality check. Marked slow: it runs the real trainer.

    If this fails, `encoder_state_path=None` is no longer a no-op and EVERY cube record from
    EXP-029 onward has silently stopped being comparable.
    """
    rec = run_cube_baseline(_cfg(seed, tmp_path))
    for field, expected in BASELINE[seed].items():
        assert rec[field] == pytest.approx(expected, abs=1e-6), (
            f"seed {seed} field {field}: {rec[field]} != pre-change {expected}. "
            "The encoder seam is NOT neutral."
        )


def test_make_agent_weights_unchanged_when_no_path_given():
    """Cheap half of the neutrality argument, and it runs every suite.

    The encoder weights `make_agent` produces at a fixed seed must not move. This catches a
    drift in the construction path without paying for a training run.
    """
    brain = make_agent(CubeConfig(seed=7, content=64))
    ref = make_sensory(7, content=64)
    assert torch.equal(brain.sensory.fc1.weight, ref.fc1.weight)
    assert torch.equal(brain.sensory.fc2.weight, ref.fc2.weight)


def test_encoder_state_path_actually_replaces_the_weights(tmp_path):
    """The seam must DO something, or EXP-040 would silently measure EXP-036 again.

    Fails against an implementation that accepts the path and ignores it - which is the defect
    that would produce a perfectly plausible null result.
    """
    donor = make_sensory(123, content=64)
    path = tmp_path / "enc.pt"
    save_encoder(donor, path)

    plain = make_agent(CubeConfig(seed=7, content=64))
    loaded = make_agent(CubeConfig(seed=7, content=64, encoder_state_path=str(path)))

    assert not torch.equal(plain.sensory.fc1.weight, loaded.sensory.fc1.weight), (
        "encoder_state_path did not change the weights")
    assert torch.equal(loaded.sensory.fc1.weight, donor.fc1.weight)
    assert torch.equal(loaded.sensory.fc2.weight, donor.fc2.weight)


def test_partial_or_wrong_state_dict_is_refused(tmp_path):
    """`strict=True` is load-bearing.

    A silently partial load would leave half a random encoder in place, and the resulting
    numbers would describe an architecture nobody chose.
    """
    donor = make_sensory(5, content=64)
    state = donor.state_dict()
    state.pop(next(iter(state)))
    path = tmp_path / "partial.pt"
    torch.save(state, str(path))

    with pytest.raises(RuntimeError):
        make_agent(CubeConfig(seed=7, content=64, encoder_state_path=str(path)))


def test_monolithic_arm_refuses_a_pretrained_encoder(tmp_path):
    """Refusing beats loading nothing and reporting a 'pretrained' number that is random."""
    path = tmp_path / "enc.pt"
    save_encoder(make_sensory(1, content=64), path)
    with pytest.raises(ValueError, match="regionalized"):
        make_agent(CubeConfig(arm="monolithic", seed=7, content=64,
                              encoder_state_path=str(path)))


def test_round_trip_save_load_preserves_weights(tmp_path):
    donor = make_sensory(11, content=64)
    path = tmp_path / "rt.pt"
    save_encoder(donor, path)
    back = load_encoder(path, seed=999, content=64)
    assert torch.equal(donor.fc1.weight, back.fc1.weight)
    assert torch.equal(donor.fc2.weight, back.fc2.weight)


def test_encoder_state_path_appears_in_the_record_config(tmp_path):
    """Provenance: a record must say which encoder produced it."""
    path = tmp_path / "enc.pt"
    save_encoder(make_sensory(3, content=64), path)
    # Depth 1, no curriculum: the cheapest real run. A 6-state shell with a 5-step budget.
    cfg = _cfg(0, tmp_path, encoder_state_path=str(path), depth=1, curriculum=(),
               episodes=4, max_depth=2)
    rec = run_cube_baseline(cfg)
    assert rec["config"]["encoder_state_path"] == str(path)
    written = json.loads((tmp_path / [p.name for p in tmp_path.glob("*.json")][0]).read_text())
    assert written["config"]["encoder_state_path"] == str(path)


# --------------------------------------------------------------------------- EXP-042 seam

def test_max_steps_override_is_empty_by_default():
    """Empty must mean the shipped `2d+3`, or every prior cube record stops being comparable."""
    assert CubeConfig().max_steps_by_depth == ()


def test_depth1_cap_removes_the_constant_action_exploit():
    """The whole point of the EXP-042 seam, verified by enumeration rather than argued.

    EXP-041: a face move has order 4, so from a one-move scramble any repeated move either
    inverts it (1 step) or cycles back to solved (3 steps). With the shipped budget of 5 a
    CONSTANT-ACTION policy scores 1/3 at depth 1 - above a random policy's 0.221 - and
    curriculum stage 1 therefore selects for the worst possible policy.

    Capping depth 1 at 2 steps admits the inverse and not the cycle. This fails if the cap is
    ever raised to 3, which is exactly the value that would silently restore the exploit.
    """
    from neuromorphic.envs.cube import MOVES, apply_move
    from neuromorphic.envs.cube_distance import ExactBFSDistance

    prov = ExactBFSDistance(max_depth=2)
    shell = prov.states_at_distance(1)

    def const_policy_rate(budget: int) -> float:
        solved = 0
        for a in range(len(MOVES)):
            for s in shell:
                cur = s
                for _ in range(budget):
                    cur = apply_move(cur, a)
                    if prov.distance(cur) == 0:
                        solved += 1
                        break
        return solved / (len(MOVES) * len(shell))

    shipped = const_policy_rate(max_steps_for(1))          # budget 5
    capped = const_policy_rate(2)                          # the EXP-042 arm

    assert shipped == pytest.approx(1 / 3, abs=1e-6), "the trap must still be present at 2d+3"
    assert capped == pytest.approx(1 / 6, abs=1e-6), "a 2-step cap must halve the exploit"
    # The bar that matters: below a uniform-random policy's measured 0.2208.
    assert capped < 0.2208, "capping must make degeneracy WORSE than exploring"
    assert shipped > 0.2208, "and the shipped budget must make it better - that is the bug"


def test_override_changes_training_budget_but_not_evaluation(tmp_path):
    """An override must never change how an arm is SCORED.

    Evaluation runs at `max_steps_for(cfg.depth)` regardless. Overriding a depth the run does
    not evaluate at must leave the held-out result reachable by the same yardstick; this fails
    against an implementation that threads the override into `evaluate_states`.
    """
    from neuromorphic.training.cube_baseline import run_cube_baseline

    base = _cfg(0, tmp_path, depth=2, curriculum=(1, 2), episodes=8, max_depth=3)
    over = _cfg(0, tmp_path / "b", depth=2, curriculum=(1, 2), episodes=8, max_depth=3,
                max_steps_by_depth=((1, 2),))
    (tmp_path / "b").mkdir(exist_ok=True)

    r_base = run_cube_baseline(base)
    r_over = run_cube_baseline(over)

    # Same evaluated depth, so the same eval budget and the same held-out set size.
    assert r_base["n"] == r_over["n"]
    # In-memory the record keeps tuples (`asdict`); JSON turns them into lists on the way out.
    assert r_base["config"]["max_steps_by_depth"] == ()
    assert r_over["config"]["max_steps_by_depth"] == ((1, 2),)

    written = json.loads(next((tmp_path / "b").glob("*.json")).read_text())
    assert written["config"]["max_steps_by_depth"] == [[1, 2]], "provenance must survive to disk"
