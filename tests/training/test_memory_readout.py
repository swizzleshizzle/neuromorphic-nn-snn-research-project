import random

import pytest
import torch

from neuromorphic.brain import Brain
from neuromorphic.encoders import cube_encoder
from neuromorphic.training.cube_baseline import (
    CubeConfig,
    MemoryReadout,
    feature_width,
    run_cube_baseline,
)


def _out(brain, i=0):
    """One brain.step over a fixed cube state.

    ``i`` seeds the Poisson encoder so successive calls in a loop draw distinct
    spike trains (as a single advancing generator naturally would in real training,
    reinforce.py's ``generator`` is shared and advances step to step). Re-seeding
    to the SAME constant every call would make every "visited state" bit-identical,
    which would make it impossible for ``memory_shuffled`` to ever differ from
    ``memory`` no matter the implementation.
    """
    import numpy as np
    return brain.step(
        np.zeros(24, dtype=np.int64), store=True, recall=True,
        generator=torch.Generator().manual_seed(i),
    )


def test_feature_widths_per_mode():
    assert feature_width(CubeConfig(readout="concept")) == 64
    assert feature_width(CubeConfig(readout="memory")) == 129
    assert feature_width(CubeConfig(readout="memory_shuffled")) == 129


def _brain():
    return Brain(encoder=cube_encoder(), n_obs=144, obs_width=24, n_actions=6, seed=0)


def test_concept_mode_returns_only_the_concept():
    b = _brain()
    r = MemoryReadout("concept", random.Random(0), b)
    r.reset()
    assert r(_out(b)).shape == (64,)


def test_memory_mode_appends_recall_and_familiarity():
    b = _brain()
    r = MemoryReadout("memory", random.Random(0), b)
    r.reset()
    assert r(_out(b)).shape == (129,)


def test_shuffled_mode_matches_width_but_differs_in_content():
    """The shuffle-null must hold width fixed while destroying correspondence."""
    b = _brain()
    real = MemoryReadout("memory", random.Random(0), b)
    shuf = MemoryReadout("memory_shuffled", random.Random(0), b)
    real.reset()
    shuf.reset()
    outs = [_out(b, i) for i in range(4)]
    f_real = torch.stack([real(o) for o in outs])
    f_shuf = torch.stack([shuf(o) for o in outs])
    assert f_real.shape == f_shuf.shape == (4, 129)
    # concept half identical, memory half differs once more than one state is cached
    assert torch.allclose(f_real[:, :64], f_shuf[:, :64])
    assert not torch.allclose(f_real[-1, 64:], f_shuf[-1, 64:])


def test_reset_clears_the_cache():
    b = _brain()
    r = MemoryReadout("memory_shuffled", random.Random(0), b)
    r.reset()
    r(_out(b))
    assert len(r._cache) == 1
    r.reset()
    assert len(r._cache) == 0


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown readout"):
        MemoryReadout("nonsense", random.Random(0), _brain()).reset()


def test_run_records_readout_and_revisit_rate(tmp_path):
    cfg = CubeConfig(arm="regionalized", readout="memory", depth=1, seed=0,
                     episodes=3, max_depth=1, sigma=0.0, out_dir=tmp_path)
    rec = run_cube_baseline(cfg)
    assert rec["readout"] == "memory"
    assert 0.0 <= rec["revisit_rate"] <= 1.0
    assert rec["mean_n_stored"] >= 1.0
