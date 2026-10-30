"""EXP-083: X16k against X at depths 9 and 10 on states neither judge was trained on."""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

import pytest

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.training import value_iteration as vi

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "083_unseen_depth9"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp083_cells", EXP / "cells.py")
sets = _load("exp083_sets", EXP / "sets.py")
t75tests = _load("test_exp075_helpers", REPO / "tests" / "experiments" / "test_exp075.py")

SEED = 12


def test_constants_are_the_spec_values():
    assert cells.DEPTHS == (9, 10) and cells.CAP == 200 and cells.SHUFFLE_BASE == 83_000
    assert (cells.X_END, cells.X16K_END) == (4000, 16000)
    assert cells.ARMS == ("J4V-X16k", "J4V-X")
    assert list(cells.EVAL_SEEDS) == list(range(12))
    assert cells.OUT == EXP / "outputs"
    assert cells.E75_OUT == REPO / "experiments" / "075_wide_region" / "outputs"
    assert cells.E82_OUT == REPO / "experiments" / "082_x_long_budget" / "outputs"
    assert (cells.BATCH, cells.MAX_LEN) == (1000, 14)


def test_names():
    assert cells.sets_name(3) == "exp083_sets_s3.json"
    assert cells.search_name("J4V-X", 10, 3) == "exp083_J4V-X_d10_s3.json"
    assert cells.GATE_R_NAME == "exp083_gateR_s0.json"
    assert cells.floor_name(9, 3) == "exp083_floor_d9_s3.json"


def _probe():
    return [(apply_move(SOLVED, 0), 1), (apply_move(apply_move(SOLVED, 1), 3), 2)]


def _train(root, n, batch=4):
    """A miniature EXP-075 X run: fake pretrained encoder, n updates of value iteration."""
    t75tests._fake_encoder(root, "X", SEED)
    judge = cells.c75.make_judge(SEED, "X", root)
    ck = cells.c75.ckpt_dir("X", SEED, root)
    vi.train_judge(judge, "W", n_updates=n, batch=batch, max_len=14, n_actions=N_ACTIONS,
                   exclude=set(), probe=_probe(), seed=SEED, sync_every=1, probe_every=1,
                   draws=1, ckpt_dir=ck, log=lambda m: None)
    return ck


def test_replay_matches_a_real_training_run(tmp_path):
    """Catches a replay that drifts from training: if anything else drew from the walk rng, or
    the batch shape differed, the random state at the end would not match state.json."""
    ck = _train(tmp_path / "e75", 3)
    _, at = sets.replay(SEED, set(), (1, 3), batch=4)
    assert at[3] == json.loads((ck / "state.json").read_text())["random_state"]
    assert at[1] != at[3]


def test_replay_seen_is_exactly_the_batches():
    seen, _ = sets.replay(SEED, set(), (5,), batch=7)
    rng, want = random.Random(SEED), set()
    for _ in range(5):
        want.update(vi.random_walk_states(7, 14, rng, N_ACTIONS, set()))
    assert seen == want


def test_pick_drops_seen_and_excluded_and_caps():
    shell = list(range(500))
    got = sets.pick(shell, set(range(100)), set(range(100, 150)), 9, 0)
    assert len(got) == 200 == len(set(got))
    assert all(x >= 150 for x in got)


def test_pick_is_independent_of_shell_order_and_varies_by_seed_and_depth():
    shell = list(range(500))
    a = sets.pick(shell, set(), set(), 9, 0)
    assert sets.pick(list(reversed(shell)), set(), set(), 9, 0) == a
    assert sets.pick(shell, set(), set(), 9, 1) != a
    assert sets.pick(shell, set(), set(), 10, 0) != a


def test_pick_returns_what_there_is_when_short():
    assert sorted(sets.pick(list(range(50)), set(range(10)), set(), 9, 0)) == list(range(10, 50))


def _sets_world(tmp_path, n=3):
    """Fake EXP-075 (at update 1) and EXP-082 (at update n) checkpoints of one seed's stream."""
    e75 = tmp_path / "e75"
    _train(e75, 1)
    e82 = tmp_path / "e82"
    _train(e82, n)
    shells = {9: list(range(1000, 1600)), 10: list(range(2000, 2600))}
    kw = dict(exclude_fn=lambda seed, prov: {1000, 1001}, shells_fn=lambda prov: shells,
              provider_fn=lambda: None, e75_root=e75, e82_root=e82, ends=(1, n), batch=4)
    return tmp_path / "out", kw


def test_build_writes_sets_when_gate_v_passes(tmp_path):
    out, kw = _sets_world(tmp_path)
    rec = sets.build(SEED, out, **kw)
    assert rec["gate_v"] == {"u4000": True, "u16000": True} and rec["pass"] is True
    assert {k: len(v) for k, v in rec["sets"].items()} == {"9": 200, "10": 200}
    assert 1000 not in rec["sets"]["9"] and 1001 not in rec["sets"]["9"]
    assert rec["candidates"] == {"9": 598, "10": 600}
    assert json.loads((out / cells.sets_name(SEED)).read_text()) == rec


def test_build_refuses_sets_when_the_stream_does_not_match(tmp_path):
    out, kw = _sets_world(tmp_path)
    p = cells.c75.ckpt_dir("X", SEED, kw["e82_root"]) / "state.json"
    s = json.loads(p.read_text())
    s["random_state"][1][0] += 1
    p.write_text(json.dumps(s))
    rec = sets.build(SEED, out, **kw)
    assert rec["gate_v"] == {"u4000": True, "u16000": False} and rec["pass"] is False
    assert rec["sets"] == {"9": [], "10": []}


def test_build_fails_gate_v_when_a_checkpoint_is_at_another_update(tmp_path):
    out, kw = _sets_world(tmp_path)
    rec = sets.build(SEED, out, **{**kw, "ends": (2, 3)})
    assert rec["gate_v"]["u4000"] is False and rec["pass"] is False


def test_sets_pending_skips_written_files(tmp_path):
    (tmp_path / cells.sets_name(2)).write_text("{}")
    assert sets.pending([1, 2, 3], tmp_path) == [1, 3]
