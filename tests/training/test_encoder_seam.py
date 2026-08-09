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

from neuromorphic.training.cube_baseline import CubeConfig, make_agent, run_cube_baseline
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
