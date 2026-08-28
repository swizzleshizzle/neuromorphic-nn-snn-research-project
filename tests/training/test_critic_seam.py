"""The EXP-053 critic seam, and the proof it changed nothing when unused.

Same discipline as `test_encoder_seam.py` (EXP-040) and `test_encoder_finetune_seam.py`
(EXP-047): every cube number from EXP-029 onward was produced with a scalar EMA baseline, so
the default path must reproduce them EXACTLY or none of them stay comparable.

The BASELINE values are imported, not re-typed, so the three seam files cannot drift.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from neuromorphic.training.cube_baseline import (
    CubeConfig,
    critic_filename,
    run_cube_baseline,
)

from .test_encoder_seam import BASELINE

FROZEN_TRAINABLE = 390        # Linear(64 -> 6)
CRITIC_TRAINABLE = 455        # + Linear(64 -> 1)


def _cfg(seed: int, out_dir: Path, **kw) -> CubeConfig:
    """The exact config `test_encoder_seam.py` measured the baseline with."""
    base = dict(arm="regionalized", readout="concept", tag="critic_neutral", depth=3,
                seed=seed, sigma=0.0, episodes=60, curriculum=(1, 2, 3),
                max_depth=4, out_dir=out_dir)
    base.update(kw)
    return CubeConfig(**base)


@pytest.mark.slow
@pytest.mark.parametrize("seed", sorted(BASELINE))
def test_critic_lr_unset_is_inert(tmp_path, seed):
    """If this fails, EVERY cube record from EXP-029 onward has stopped being comparable."""
    rec = run_cube_baseline(_cfg(seed, tmp_path))
    for field, expected in BASELINE[seed].items():
        assert rec[field] == pytest.approx(expected, abs=1e-6), (
            f"seed {seed} field {field}: {rec[field]} != pre-change {expected}. "
            "The critic seam is NOT neutral."
        )
    assert rec["trainable_params"] == FROZEN_TRAINABLE
    assert "critic_ev" not in rec


@pytest.mark.slow
def test_critic_run_differs_and_is_counted_and_serialised(tmp_path):
    """The complement: at a real lr the critic must CHANGE the run, be counted, and be saved.

    Without this, every assertion above is satisfied by a critic switch that does nothing -
    the exact class of defect `CLAUDE.md`'s test-strength rule exists for.
    """
    seed = sorted(BASELINE)[0]
    cfg = _cfg(seed, tmp_path, critic_lr=1e-2, tag="critic_live")
    rec = run_cube_baseline(cfg)

    assert rec["trainable_params"] == CRITIC_TRAINABLE, (
        f"trainable_params is {rec['trainable_params']}, so the critic is not in any "
        "optimizer and is not being trained."
    )
    assert rec["mean_train_entropy"] != pytest.approx(
        BASELINE[seed]["mean_train_entropy"], abs=1e-6), (
        "a live critic produced a run identical to the EMA-baseline one. The advantage is "
        "not actually using V(s), so this arm and its control differ in nothing."
    )
    assert -5.0 < rec["critic_ev"] <= 1.0, (
        f"critic_ev {rec['critic_ev']} is outside any sane range for an explained-variance "
        "figure; the accumulation is probably wrong."
    )
    saved = tmp_path / critic_filename(cfg)
    assert saved.exists(), (
        f"a critic run must serialise its critic ({saved.name} missing); it is a result, "
        "not a byproduct."
    )


def test_critic_lr_refuses_a_memory_readout(tmp_path):
    """`MemoryReadout` detaches the concept, so a critic on it would train on nothing.

    Refused rather than ignored, exactly as `encoder_lr` is.
    """
    with pytest.raises(ValueError, match="critic_lr requires readout='concept'"):
        run_cube_baseline(_cfg(0, tmp_path, critic_lr=1e-2, readout="memory"))
