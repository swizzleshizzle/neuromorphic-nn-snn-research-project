"""The EXP-047 fine-tuning seam, and the proof it changed nothing when unused.

`CubeConfig.encoder_lr` makes the sensory encoder trainable during RL. Every cube number from
EXP-029 onward was produced with it frozen, so the default path must reproduce them EXACTLY or
none of them stay comparable. Same precedent as `encoder_state_path` (EXP-040) and head
serialisation (EXP-036); `test_encoder_seam.py` is the sibling file.

The baseline values are shared with `test_encoder_seam.py` deliberately rather than
re-captured. They were measured from the commit before `encoder_state_path` existed, and if
they still hold here then BOTH seams are neutral against the same fixed point.

Two distinct things are asserted, and the second is the one that matters for interpreting the
fine-tuned arm:

  1. `encoder_lr=None` reproduces the pre-change baseline.  (the field is inert when unset)
  2. `encoder_lr=0.0` reproduces it too.                    (the GRAD PATH ITSELF is inert)

(2) is stronger. It runs the whole `grad_brain=True` code path - no `no_grad`, a live autograd
graph over the spiking unroll, `backward()` reaching `fc1`/`fc2`, a second Adam parameter group
- and takes a zero-sized step. Any divergence therefore proves that enabling gradients changed
the FORWARD, which would mean the fine-tuned arm differs from its control in two ways at once
and the whole comparison is unreadable. That is the failure this file exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neuromorphic.training.cube_baseline import (
    CubeConfig,
    encoder_filename,
    run_cube_baseline,
)

# Imported, not re-typed: one copy of the fixed point, so the two seam files cannot drift.
# `tests/training` is a package (it has an `__init__.py`) but `tests` is not, so the sibling
# module is addressed relatively rather than as `tests.training.test_encoder_seam`.
from .test_encoder_seam import BASELINE

FROZEN_TRAINABLE = 390        # Linear(64 -> 6)
FINETUNE_TRAINABLE = 27_206   # + fc1 (144*128 + 128) + fc2 (128*64 + 64)


def _cfg(seed: int, out_dir: Path, **kw) -> CubeConfig:
    """The exact config `test_encoder_seam.py` measured the baseline with."""
    base = dict(arm="regionalized", readout="concept", tag="ft_neutral", depth=3,
                seed=seed, sigma=0.0, episodes=60, curriculum=(1, 2, 3),
                max_depth=4, out_dir=out_dir)
    base.update(kw)
    return CubeConfig(**base)


@pytest.mark.slow
@pytest.mark.parametrize("encoder_lr", [None, 0.0],
                         ids=["unset_is_inert", "zero_lr_grad_path_is_inert"])
@pytest.mark.parametrize("seed", sorted(BASELINE))
def test_finetune_seam_is_neutral(tmp_path, seed, encoder_lr):
    """Both inert settings must land on the pre-change baseline, to 1e-6.

    If the `None` case fails, `encoder_lr` is no longer a no-op and EVERY cube record from
    EXP-029 onward has silently stopped being comparable.

    If only the `0.0` case fails, enabling autograd changed the forward - so EXP-047's arm and
    EXP-043's control differ by more than the one variable the spec claims, and the paired
    delta measures something nobody chose.
    """
    rec = run_cube_baseline(_cfg(seed, tmp_path, encoder_lr=encoder_lr))
    for field, expected in BASELINE[seed].items():
        assert rec[field] == pytest.approx(expected, abs=1e-6), (
            f"seed {seed}, encoder_lr={encoder_lr!r}, field {field}: "
            f"{rec[field]} != pre-change {expected}. The fine-tuning seam is NOT neutral."
        )


@pytest.mark.slow
def test_encoder_actually_moves_and_is_serialised(tmp_path):
    """The complement: at a real lr the encoder MUST change, and must be written to disk.

    Without this, every assertion above is satisfied by a fine-tuning switch that does nothing
    - which is exactly the class of defect `CLAUDE.md`'s test-strength rule was written for.
    The `no_grad` in `MemoryReadout` is a live example of how this can silently happen.

    The threshold is a MEASURED one, not a qualitative "did it change": a 1e-2 lr over 60
    episodes moves `fc1.weight` far more than 1e-3, while a detached encoder moves it exactly
    0.0. Prototyped 2026-08-20 before the bar was set.
    """
    cfg = _cfg(0, tmp_path, encoder_lr=1e-2, tag="ft_moves")
    before = torch.load(
        # The starting weights are reproducible from `encoder_seed`, so re-building the same
        # brain is a fair "before" without needing a checkpoint of it.
        _reference_sensory_path(tmp_path), map_location="cpu"
    )
    rec = run_cube_baseline(cfg)

    saved = tmp_path / encoder_filename(cfg)
    assert saved.exists(), (
        f"a fine-tuned run must serialise its encoder ({saved.name} missing). Claim 2 re-probes "
        "these weights and cannot be asked afterwards if they were never written."
    )
    after = torch.load(saved, map_location="cpu")

    drift = (after["fc1.weight"] - before["fc1.weight"]).abs().max().item()
    assert drift > 1e-3, (
        f"fc1.weight moved by only {drift:.3e}. The encoder is not actually training - most "
        "likely the concept is being detached somewhere upstream of the head."
    )
    assert rec["trainable_params"] == FINETUNE_TRAINABLE, (
        f"trainable_params {rec['trainable_params']} != {FINETUNE_TRAINABLE}"
    )


def _reference_sensory_path(tmp_path: Path) -> Path:
    """Serialise a freshly-built seed-0 sensory region: the 'before' weights."""
    from neuromorphic.training.cube_baseline import CUBE_N_OBS
    from neuromorphic.regions.sensory_cortex import SensoryCortex

    sensory = SensoryCortex(n_obs=CUBE_N_OBS, concept=64, num_steps=32, seed=0)
    path = tmp_path / "reference_sensory.pt"
    torch.save(sensory.state_dict(), path)
    return path


@pytest.mark.slow
def test_frozen_run_records_390_trainable_params(tmp_path):
    """The number the whole depth series rests on, asserted rather than assumed."""
    rec = run_cube_baseline(_cfg(0, tmp_path, tag="ft_count"))
    assert rec["trainable_params"] == FROZEN_TRAINABLE


@pytest.mark.parametrize(
    "kw, expected",
    [
        (dict(arm="monolithic"), "arm='regionalized'"),
        (dict(readout="memory"), "readout='concept'"),
    ],
)
def test_finetune_refuses_configurations_it_cannot_honour(tmp_path, kw, expected):
    """Refusing beats running: both of these would train the encoder on nothing.

    `monolithic` has no `sensory` region at all. `readout="memory"` routes features through
    `MemoryReadout`, whose `__call__` wraps its body in `torch.no_grad()` - so the concept
    arrives at the head detached, the encoder receives zero gradient, and the run would report
    a perfectly ordinary-looking "fine-tuned" number produced by a frozen encoder.
    """
    cfg = _cfg(0, tmp_path, encoder_lr=1e-4, **kw)
    with pytest.raises(ValueError, match=expected):
        run_cube_baseline(cfg)
