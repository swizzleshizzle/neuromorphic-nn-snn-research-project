"""EXP-085: X16k continued from 16,000 to 24,000 updates on outward walks (arm O) against X's walks
(arm C), on the device. Each docstring names the bug it catches."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "085_outward_walks"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp085_cells", EXP / "cells.py")
gatec = _load("exp085_gatec", EXP / "gatec.py")
t75tests = _load("test_exp075_helpers", REPO / "tests" / "experiments" / "test_exp075.py")
vi = gatec.t75.vi

SEED = 12
START, BATCH = 2, 4


def test_constants_are_the_spec_values():
    """Catches a setting drifting from EXP-082's X16k or from spec section 2's schedule."""
    assert cells.ARMS == ("C", "O") and cells.SRC_ARM == "X"
    assert (cells.START, cells.RAMP_END, cells.END) == (16000, 20000, 24000)
    assert cells.LEGS == (20000, 23751, 24000) and cells.RECORD_LEGS == (20000, 24000)
    assert cells.SNAP_UPDATE == 23751 and (cells.MAX_LEN_C, cells.MAX_LEN_O) == (14, 30)
    assert (cells.BATCH, cells.SYNC_EVERY, cells.PROBE_EVERY, cells.DRAWS) == (1000, 100, 250, 1)
    assert cells.DEPTHS == (9, 11) and list(cells.SEEDS) == list(range(12))
    assert cells.H_SEED == 85
    c = cells.max_len_schedule("C")
    assert [c(u) for u in (16000, 19999, 23999)] == [14, 14, 14]
    o = cells.max_len_schedule("O")
    assert [o(u) for u in (16000, 18000, 19999, 20000, 23999)] == [14, 22, 29, 30, 30]
    with pytest.raises(ValueError):
        cells.max_len_schedule("X")


def test_names_and_layout():
    """The arm folders are EXP-075's layout, so EXP-075's load_judge reads an arm's judge; every
    record name carries the experiment and the arm."""
    assert cells.ckpt_dir("O", 3) == cells.OUT / "arm_O" / "judge_X_s3"
    assert cells.ckpt_dir("C", 3) == cells.c75.ckpt_dir("X", 3, cells.arm_root("C"))
    assert cells.start_dir(3) == cells.OUT / "start" / "judge_X_s3"
    assert cells.snap_dir("C", 3) == cells.ckpt_dir("C", 3) / "snap_u23751"
    assert cells.gatec_name(3) == "exp085_gatec_s3.json"
    assert cells.rebase_name("O", 3) == "exp085_rebase_O_s3.json"
    assert cells.record_name("O", 20000, 3) == "exp085_O_u20000_s3.json"
    assert cells.search_name("C", 11, 3) == "exp085_J4V-C_d11_s3.json"
    assert cells.h_search_name("C", 9, 3) == "exp085_H_J4V-C_d9_s3.json"
    assert cells.h_train_name("O", 3) == "exp085_Hb_O_s3.json"
    assert cells.PREFLIGHT_NAME == "exp085_preflight.json"
    assert cells.E82_OUT == REPO / "experiments" / "082_x_long_budget" / "outputs"


def test_exclusion_includes_every_probe_state(monkeypatch):
    """Catches the probe left open to the walks (spec section 2): arm O would train on about half
    the distance-11 probe states and Claim 2 would partly measure memorisation."""
    held = (apply_move(SOLVED, 0),)
    probe = [(apply_move(SOLVED, 1), 1), (apply_move(apply_move(SOLVED, 2), 2), 2)]

    class FakeBFS:
        def __init__(self, max_depth=None):
            self.distance = lambda s: 0

    monkeypatch.setattr(gatec, "ExactBFSDistance", FakeBFS)
    monkeypatch.setattr(gatec.c75, "exclusion_set", lambda seed, p: set(held))
    monkeypatch.setattr(gatec.c75, "probe_set", lambda seed, p: list(probe))
    exclude, got_probe, _ = gatec.sets(3)
    assert got_probe == probe
    assert exclude == set(held) | {s for s, _ in probe}


def _probe():
    return [(apply_move(SOLVED, 0), 1), (apply_move(apply_move(SOLVED, 1), 3), 2)]


def fake_measure(judge, probe, distance, seed):
    """Stands in for Gate L (which needs the full BFS table): a number every weight moves."""
    return {"margin": sum(float(p.detach().double().sum()) for p in judge.parameters())}


def _train_straight(root, n):
    judge = gatec.c75.make_judge(SEED, "X", root)
    vi.train_judge(judge, gatec.t75.VI_ARM, n_updates=n, batch=BATCH, max_len=14,
                   n_actions=N_ACTIONS, exclude=set(), probe=_probe(), seed=SEED,
                   sync_every=1, probe_every=1, draws=1,
                   ckpt_dir=gatec.c75.ckpt_dir("X", SEED, root), log=lambda m: None)
    return judge


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fake laptop copy of X16k at update START, a fake EXP-082 outputs folder holding its
    record, and an empty EXP-085 out dir."""
    monkeypatch.setattr(gatec.cells, "START", START)
    monkeypatch.setattr(gatec, "sets", lambda seed: (set(), _probe(), None))
    monkeypatch.setattr(gatec, "measure", fake_measure)
    src = tmp_path / "laptop"
    t75tests._fake_encoder(src, "X", SEED)
    judge = _train_straight(src, START)
    e82 = tmp_path / "e82"
    e82.mkdir()
    (e82 / f"exp082_X16k_u{START}_s{SEED}.json").write_text(json.dumps(
        {"gate_l": fake_measure(judge, None, None, SEED)}))
    return src, e82, tmp_path / "out"


def _tracked_under(e82):
    """git's blob only for a path under the EXP-082 reference folder."""
    return lambda p: "b" if Path(p).is_relative_to(e82) else None


def _gc(world, tracked=None, blob=lambda p: "b"):
    src, e82, out = world
    return gatec.gate_c(SEED, src, out, e82_root=e82,
                        tracked=_tracked_under(e82) if tracked is None else tracked, blob=blob)


def _src_ck(world):
    return gatec.c75.ckpt_dir("X", SEED, world[0])


def test_gate_c_passes_and_seeds_the_start_dir(world):
    """Catches a copy missing a file (the optimizer, so Adam restarts) and identity read against
    the laptop copy's own path instead of EXP-082's tracked one."""
    rec = _gc(world)
    assert rec["pass"] is True and rec["start"] == "seeded"
    dst = gatec.cells.start_dir(SEED, world[2])
    for f in ("judge.pt", "jt.pt", "opt.pt", "state.json"):
        assert (dst / f).read_bytes() == (_src_ck(world) / f).read_bytes(), f
    assert rec["continued_from"] == {"experiment": "EXP-082", "update": START}
    assert rec["opt_bad"] == [] and rec["start_gate_l_equal"] is True
    assert gatec.passed(SEED, world[2])


def _fails(world, **kw):
    with pytest.raises(SystemExit, match="GATE C FAILED|start update"):
        _gc(world, **kw)
    rec = json.loads((world[2] / gatec.cells.gatec_name(SEED)).read_text())
    assert rec["pass"] is False
    assert not (gatec.cells.start_dir(SEED, world[2]) / "state.json").exists()
    return rec


def test_gate_c_fails_on_a_judge_that_is_not_the_tracked_file(world):
    rec = _fails(world, blob=lambda p: "other")
    assert any("differs" in b for b in rec["identity_bad"])


def test_gate_c_fails_on_an_untracked_reference(world):
    rec = _fails(world, tracked=lambda p: None)
    assert any("untracked" in b for b in rec["identity_bad"])


def test_gate_c_fails_on_a_checkpoint_at_another_update(world, monkeypatch):
    monkeypatch.setattr(gatec.cells, "START", START + 1)
    _fails(world)


def test_gate_c_fails_on_an_optimizer_at_the_wrong_step(world):
    p = _src_ck(world) / "opt.pt"
    sd = torch.load(p)
    for st in sd["state"].values():
        st["step"] = torch.tensor(float(START - 1))
    torch.save(sd, p)
    rec = _fails(world)
    assert any("step" in b for b in rec["opt_bad"])


def test_gate_c_fails_on_a_start_off_in_the_last_digit(world):
    """Catches a Gate L comparison that is not exact."""
    p = world[1] / f"exp082_X16k_u{START}_s{SEED}.json"
    r = json.loads(p.read_text())
    r["gate_l"]["margin"] += 1e-12
    p.write_text(json.dumps(r))
    rec = _fails(world)
    assert rec["start_gate_l_equal"] is False


def test_gate_c_does_not_gate_jt(world):
    """Spec section 6: jt.pt is copied, not gated (train_judge re-syncs it at update 16,000)."""
    (_src_ck(world) / "jt.pt").write_bytes(b"not a checkpoint")
    assert _gc(world)["pass"] is True


def test_gate_c_never_overwrites_a_seeded_start(world):
    _gc(world)
    marker = gatec.cells.start_dir(SEED, world[2]) / "judge.pt"
    marker.write_bytes(b"seeded earlier")
    (world[2] / gatec.cells.gatec_name(SEED)).unlink()
    assert _gc(world)["start"] == "kept" and marker.read_bytes() == b"seeded earlier"


def test_gate_c_never_writes_the_source_or_exp082(world):
    def snap():
        return {str(f): f.read_bytes() for d in (world[0], world[1]) for f in d.rglob("*")
                if f.is_file()}
    before = snap()
    _gc(world)
    assert snap() == before


def test_identity_compares_against_exp082s_tracked_files():
    """Catches identity checked against a path git does not track (it would pass on anything)."""
    f = cells.E82_OUT / "judge_X_s0" / "judge.pt"
    assert gatec.tracked_blob(f) == gatec.blob_of(f) and len(gatec.blob_of(f)) == 40
