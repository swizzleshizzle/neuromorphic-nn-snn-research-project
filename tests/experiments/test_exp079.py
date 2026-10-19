"""EXP-079: judge S, EXP-075's X recipe trained on walks of length 1 to 40."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.encoder_pretrain import save_encoder

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "079_judge_trained_deep"
E75 = REPO / "experiments" / "075_wide_region"
E76 = REPO / "experiments" / "076_deeper_tree"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp079_cells", EXP / "cells.py")


def _fake_x_pilot(d, seed, perturb=0.0):
    """EXP-075's pilot layout for one seed under `d`: encoder, X judge whose region has moved,
    and a training record whose encoder_drift is the true drift (plus `perturb`)."""
    enc = cells.c75.fresh_sensory("X", seed)
    path = cells.c75.encoder_path("X", seed, d)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder(enc, path)
    judge = cells.c75.build_judge(seed, "X", cells.c75.fresh_sensory("X", seed))
    with torch.no_grad():
        judge.sensory.fc1.weight.add_(0.01)
    ck = cells.c75.ckpt_dir("X", seed, d)
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(judge.state_dict(), ck / "judge.pt")
    e0 = cells.c75.load_pretrained("X", seed, d).state_dict()
    true = cells.drift(judge.sensory.state_dict(), e0)
    (d / cells.c75.record_name("X", seed)).write_text(
        json.dumps({"encoder_drift": true + perturb}))
    return true


def test_constants_are_the_spec_values():
    """Catches the walk length, update count, seeds or source folders drifting from spec
    sections 3 and 6."""
    assert cells.MAX_LEN == 40 and cells.N_UPDATES == 4000
    assert cells.ARM == "S" and cells.BASE_ARM == "X"
    assert cells.EVAL_SEEDS == tuple(range(12)) and cells.PILOT_SEEDS == (12, 13)
    assert cells.E75_OUT.name == "outputs" and cells.E75_OUT.parent.name == "075_wide_region"
    assert cells.E76_OUT.parent.name == "076_deeper_tree"
    assert cells.encoder_root(False) == cells.E75_OUT
    assert cells.encoder_root(True) == cells.PILOT_OUT
    assert cells.record_name(3) == "exp079_train_S_s3.json"
    assert cells.ckpt_dir(3, Path("x")) == Path("x") / "judge_S_s3"


def test_make_judge_is_x_built_from_the_committed_encoder():
    """Catches S built at another width, with another readout, from another encoder, or with a
    head not seeded like X's (spec section 3: the only difference is max_len)."""
    s = cells.make_judge(0, cells.E75_OUT)
    x = cells.c75.make_judge(0, "X", cells.E75_OUT)
    assert s.readout == "wide" and s.sensory.fc1.out_features == 512
    sd_s, sd_x = s.state_dict(), x.state_dict()
    assert sd_s.keys() == sd_x.keys()
    for k in sd_s:
        assert torch.equal(sd_s[k], sd_x[k]), k


def test_load_trained_refuses_a_partly_trained_or_missing_judge(tmp_path):
    """Catches an S judge evaluated before training finished (judge.pt exists from update 0)."""
    with pytest.raises(SystemExit, match="judge_S_s0"):
        cells.load_trained(0, tmp_path)
    ck = cells.ckpt_dir(0, tmp_path)
    ck.mkdir(parents=True)
    judge = cells.c75.build_judge(0, "X", cells.c75.fresh_sensory("X", 0))
    torch.save(judge.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": 3999}))
    with pytest.raises(SystemExit, match="3999"):
        cells.load_trained(0, tmp_path)
    (ck / "state.json").write_text(json.dumps({"update": 4000}))
    loaded = cells.load_trained(0, tmp_path)
    assert torch.equal(loaded.head[0].weight, judge.head[0].weight)
    assert cells.judge_updates(0, tmp_path) == 4000


def test_drift_is_exp075s_l2_over_the_encoder_tensors():
    """Catches a drift that is not EXP-075's (so the pilot proof would compare different
    quantities): one tensor moved by 0.01 everywhere gives 0.01 * sqrt(numel)."""
    a = {"w": torch.zeros(4, 25), "b": torch.zeros(3)}
    b = {"w": torch.full((4, 25), 0.01), "b": torch.zeros(3)}
    assert cells.drift(b, a) == pytest.approx(0.01 * 10.0, rel=1e-6)


def test_pilot_encoder_is_proved_only_when_the_drift_reproduces(tmp_path):
    """Catches a pilot encoder accepted without checking it is the one EXP-075's pilot record
    describes (spec section 6), and a missing file read as a match."""
    _fake_x_pilot(tmp_path, 12)
    ok, why = cells.pilot_encoder_matches(12, tmp_path)
    assert ok, why
    _fake_x_pilot(tmp_path, 13, perturb=1e-3)
    ok, why = cells.pilot_encoder_matches(13, tmp_path)
    assert not ok and "seed 13" in why
    ok, why = cells.pilot_encoder_matches(12, tmp_path / "nowhere")
    assert not ok and "missing" in why


def test_pilot_encoders_copy_all_or_nothing(tmp_path):
    """REVIEW FOCUS 3. Catches one proved encoder copied while the other failed, which would mix
    a copied encoder with a re-pretrained one in the same pilot."""
    src, dest = tmp_path / "e75", tmp_path / "e79"
    _fake_x_pilot(src, 12)
    _fake_x_pilot(src, 13, perturb=1e-3)
    with pytest.raises(SystemExit, match="pilot-pretrain"):
        cells.prepare_pilot_encoders(src, dest)
    assert not cells.c75.encoder_path("X", 12, dest).exists()
    _fake_x_pilot(src, 13)
    lines = cells.prepare_pilot_encoders(src, dest)
    assert len(lines) == 2
    for s in (12, 13):
        a = cells.c75.encoder_path("X", s, src).read_bytes()
        assert cells.c75.encoder_path("X", s, dest).read_bytes() == a


train = _load("exp079_train", EXP / "train.py")


def _tiny_judge(seed=0):
    return cells.c75.build_judge(seed, "X", cells.c75.fresh_sensory("X", seed))


def _tiny_probe():
    s1 = apply_move(SOLVED, 0)
    return [(SOLVED, 0), (s1, 1), (apply_move(s1, 2), 2)]


def _core_kwargs(tmp, max_len, n_updates=2):
    return dict(max_len=max_len, distance=lambda s: 0, n_updates=n_updates, batch=3,
                exclude=set(), probe=_tiny_probe(), seed=0, sync_every=1, probe_every=1,
                draws=1, ckpt_dir=tmp, log=lambda m: None)


def test_recorder_returns_the_walks_unchanged_and_keeps_the_last_interval():
    """Catches a recorder that alters what training sees, or a histogram over every update
    instead of the last probe interval (spec section 4)."""
    batches = iter([[0, 0], [1, 2], [2, 2]])
    rec = train.WalkRecorder(lambda n, m, rng, a, ex: next(batches), lambda s: s, keep=2)
    assert rec(2, 40, None, 6, set()) == [0, 0]
    assert rec(2, 40, None, 6, set()) == [1, 2]
    assert rec(2, 17, None, 6, set()) == [2, 2]
    out = rec.summary()
    assert out["walk_max_lens"] == [17, 40]
    assert out["train_hist"] == {"1": 0.25, "2": 0.75}
    assert out["train_hist_updates"] == 2 and rec.calls == 3


def test_recorder_reports_how_many_updates_it_covers():
    """REVIEW FOCUS 1. Catches a resumed session's short histogram passed off as a full probe
    interval: one call under keep=250 reports 1, not 250."""
    rec = train.WalkRecorder(lambda n, m, rng, a, ex: [5], lambda s: s, keep=250)
    rec(1, 40, None, 6, set())
    assert rec.summary()["train_hist_updates"] == 1


@pytest.mark.parametrize("max_len", [40, 17])
def test_max_len_reaches_random_walk_states(tmp_path, max_len):
    """THE SPEC SECTION 4 TEST. Catches a driver whose walk length never reaches the library:
    two different values must each arrive, so a hardcoded 14 (or 40) fails one case."""
    _, walks = train.train_core(_tiny_judge(), **_core_kwargs(tmp_path, max_len))
    assert walks["walk_max_lens"] == [max_len]
    assert walks["train_hist_updates"] == 1


def test_recorder_does_not_change_training(tmp_path):
    """Catches the recorder perturbing the RNG or the walks: a judge trained through train_core
    equals one trained by train_judge directly, tensor for tensor."""
    a, b = _tiny_judge(), _tiny_judge()
    train.train_core(a, **_core_kwargs(tmp_path / "a", 40))
    kw = _core_kwargs(tmp_path / "b", 40)
    kw.pop("distance")
    vi.train_judge(b, train.VI_ARM, n_actions=N_ACTIONS, **kw)
    sa, sb = a.state_dict(), b.state_dict()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_recording_restores_the_library_function_on_error():
    """Catches a patched random_walk_states left behind after a failed run, which would record
    (and slow) every later call in the process."""
    real = vi.random_walk_states
    with pytest.raises(RuntimeError):
        with train.recording(lambda s: 0, 5):
            assert vi.random_walk_states is not real
            raise RuntimeError("boom")
    assert vi.random_walk_states is real


def test_parser_requires_max_len_and_keeps_exp075s_settings():
    """Catches a driver that falls back to a default walk length, and settings drifting from
    EXP-075's X (spec section 3)."""
    with pytest.raises(SystemExit):
        train.build_parser().parse_args(["--n-updates", "1"])
    a = train.build_parser().parse_args(["--n-updates", "1", "--max-len", "40"])
    assert (a.batch, a.sync_every, a.probe_every, a.draws) == (1000, 100, 250, 1)
    assert a.max_len == 40 and a.seeds == list(range(12)) and train.VI_ARM == "W"


def test_pilot_defaults_to_the_pilot_directories():
    """REVIEW FOCUS 5. Catches pilot records written among the evaluation records, and the
    pilot reading EXP-075's evaluation encoders (seeds 12 and 13 have none there)."""
    a = train.build_parser().parse_args(["--n-updates", "1", "--max-len", "40", "--pilot",
                                         "--seeds", "12", "13"])
    assert train.resolve_dirs(a) == (cells.PILOT_OUT, cells.PILOT_OUT)
    b = train.build_parser().parse_args(["--n-updates", "1", "--max-len", "40"])
    assert train.resolve_dirs(b) == (cells.OUT, cells.E75_OUT)


def test_train_refuses_a_missing_encoder_before_building_anything(tmp_path):
    """Catches training on a random region when the encoder is absent, and the check running
    only after the 65 s BFS build."""
    import time
    t0 = time.time()
    with pytest.raises(SystemExit, match="enc_X_s12.pt"):
        train.run(12, 1, 8, 1, 1, 1, 40, tmp_path / "out", tmp_path)
    assert time.time() - t0 < 20


def test_train_relaunch_skips_runs_whose_record_exists(tmp_path):
    """Catches a relaunch that resubmits finished runs."""
    (tmp_path / cells.record_name(13)).write_text("{}")
    assert train.pending_jobs([12, 13], tmp_path) == [12]


@pytest.mark.slow
def test_run_writes_the_record_with_its_walk_evidence(tmp_path):
    """Catches a record missing the walk length evidence, the histogram, Gate L or the drift
    Gate E reads. Slow: builds the full BFS table, as the real driver does."""
    path = cells.c75.encoder_path("X", 12, tmp_path)
    path.parent.mkdir(parents=True)
    save_encoder(cells.c75.fresh_sensory("X", 12), path)
    rec = train.run(12, 2, 8, 1, 1, 1, 40, tmp_path / "out", tmp_path)
    on_disk = json.loads((tmp_path / "out" / cells.record_name(12)).read_text())
    assert on_disk["arm"] == "S" and on_disk["max_len"] == 40
    assert on_disk["walk_max_lens"] == [40] and on_disk["train_hist_updates"] == 1
    assert sum(on_disk["train_hist"].values()) == pytest.approx(1.0)
    assert on_disk["readout"] == "wide" and on_disk["hidden"] == 512
    assert on_disk["gate_l"]["n"] == 250 and on_disk["encoder_drift"] > 0.0
    assert (tmp_path / "out" / "judge_S_s12" / "judge.pt").exists()
    assert rec["gate_l"] == on_disk["gate_l"]
