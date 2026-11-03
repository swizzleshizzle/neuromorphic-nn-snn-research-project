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


rebase = _load("exp085_rebase", EXP / "rebase.py")


@pytest.fixture
def rworld(world, monkeypatch):
    """`world` with Gate C passed and the start seeded, seen through rebase's own modules."""
    monkeypatch.setattr(rebase.cells, "START", START)
    monkeypatch.setattr(rebase.gatec, "sets", lambda seed: (set(), _probe(), None))
    monkeypatch.setattr(rebase.gatec, "measure", fake_measure)
    src, e82, out = world
    rebase.gatec.gate_c(SEED, src, out, e82_root=e82, tracked=_tracked_under(e82),
                        blob=lambda p: "b")
    return out


def test_rebase_carries_the_walk_state_and_resets_the_encoding_stream(rworld):
    """Catches a rebase that restarts the walk stream (the arms would no longer continue X16k's
    walks) or keeps the CPU generator state (train_judge refuses it on the device)."""
    rec = rebase.rebase("O", SEED, "cpu", rworld)
    s0 = json.loads((rebase.cells.start_dir(SEED, rworld) / "state.json").read_text())
    s1 = json.loads((rebase.cells.ckpt_dir("O", SEED, rworld) / "state.json").read_text())
    assert s1["random_state"] == s0["random_state"] and s1["update"] == START
    assert s1["probes"] == s0["probes"]
    want = torch.Generator().manual_seed(rebase.rebase_seed(SEED)).get_state().tolist()
    assert s1["generator_state"] == want != s0["generator_state"]
    for f in ("judge.pt", "jt.pt", "opt.pt"):
        assert ((rebase.cells.ckpt_dir("O", SEED, rworld) / f).read_bytes()
                == (rebase.cells.start_dir(SEED, rworld) / f).read_bytes())
    assert rec["rebase_seed"] == rebase.rebase_seed(SEED) and rec["device"] == "cpu"
    assert (rworld / rebase.cells.rebase_name("O", SEED)).exists()


def test_rebase_seed_differs_by_seed_and_matches_the_formula():
    assert rebase.rebase_seed(0) == 85_000_000 + 16000
    assert rebase.rebase_seed(3) == 85_000_000 + 3 * 100_003 + 16000


def test_rebase_refuses_without_gate_c(world, monkeypatch):
    monkeypatch.setattr(rebase.cells, "START", START)
    with pytest.raises(SystemExit, match="Gate C"):
        rebase.rebase("C", SEED, "cpu", world[2])


def test_rebase_refuses_a_started_arm(rworld):
    """REVIEW FOCUS 2. A second rebase must never reset an arm that training has advanced."""
    rebase.rebase("C", SEED, "cpu", rworld)
    p = rebase.cells.ckpt_dir("C", SEED, rworld) / "state.json"
    st = json.loads(p.read_text())
    st["update"] = START + 7
    p.write_text(json.dumps(st))
    before = p.read_bytes()
    with pytest.raises(SystemExit, match="already"):
        rebase.rebase("C", SEED, "cpu", rworld)
    assert p.read_bytes() == before


def test_rebased_checkpoint_resumes_in_train_judge(rworld):
    rebase.rebase("C", SEED, "cpu", rworld)
    judge = rebase.cells.c75.make_judge(SEED, "X", rworld.parent / "laptop")
    vi.train_judge(judge, rebase.gatec.t75.VI_ARM, n_updates=START + 1, batch=BATCH,
                   max_len=rebase.cells.max_len_schedule("C"), n_actions=N_ACTIONS,
                   exclude=set(), probe=_probe(), seed=SEED, sync_every=1, probe_every=1,
                   draws=1, ckpt_dir=rebase.cells.ckpt_dir("C", SEED, rworld),
                   log=lambda m: None)
    st = json.loads((rebase.cells.ckpt_dir("C", SEED, rworld) / "state.json").read_text())
    assert st["update"] == START + 1


preflight = _load("exp085_preflight", EXP / "preflight.py")


def _sd(v):
    return {"w": torch.tensor([1.0, v])}


@pytest.mark.parametrize("b,want", [(2.0, True), (2.5, False)])
def test_preflight_gpu_reports_determinism(rworld, b, want):
    """Catches a determinism check that compares a run with itself, or that cannot see a change."""
    runs = iter([_sd(2.0), _sd(b)])
    rec = preflight.gpu("cpu", rworld, seed=SEED, train_fn=lambda ck, seed, device, n: next(runs))
    assert rec["deterministic"] is want and rec["n_updates"] == preflight.N_PREFLIGHT
    assert json.loads((rworld / rebase.cells.PREFLIGHT_NAME).read_text())["deterministic"] is want


@pytest.mark.slow
def test_preflight_walks_matches_the_spec_table():
    """Spec section 10: arm O's walks at update 20,000 have the measured distance shares within
    0.02 of spec section 2's table, and arm C's those of max length 14."""
    rec = preflight.walks(exclude=set())
    rows = {(r["arm"], r["update"]): r["shares"] for r in rec["rows"]}
    for got, want in ((rows[("O", 20000)], (0.524, 0.108, 0.149, 0.141, 0.079)),
                      (rows[("C", 16000)], (0.884, 0.051, 0.045, 0.013, 0.007))):
        assert all(abs(g - w) <= 0.02 for g, w in zip(got, want)), (got, want)


train = _load("exp085_train", EXP / "train.py")


@pytest.fixture
def tworld(world, monkeypatch):
    """`world` seen through train's own modules: Gate C passed and both arms rebased, with legs
    shrunk to (3 recorded, 4 snapshot, 5 recorded) and the ramp to end at update 6."""
    c = train.cells
    for k, v in (("START", START), ("RAMP_END", 6), ("LEGS", (3, 4, 5)),
                 ("RECORD_LEGS", (3, 5)), ("SNAP_UPDATE", 4)):
        monkeypatch.setattr(c, k, v)
    monkeypatch.setattr(train.gatec, "sets", lambda seed: (set(), _probe(), None))
    monkeypatch.setattr(train.gatec, "measure", fake_measure)
    src, e82, out = world
    train.gatec.gate_c(SEED, src, out, e82_root=e82, tracked=_tracked_under(e82),
                       blob=lambda p: "b")
    for a in c.ARMS:
        train.rebase.rebase(a, SEED, "cpu", out)
    return src, out


def _trun(tworld, arm="C"):
    src, out = tworld
    return train.run(arm, SEED, "cpu", out, batch=BATCH, sync_every=1, probe_every=1,
                     src_root=src)


def test_legs_record_and_snapshot(tworld):
    """Catches a snapshot leg that writes a record (the aggregator would read it as a leg), a
    missing snapshot (Gate H (b) has nothing to replay from), or legs at the wrong updates."""
    recs = _trun(tworld)
    out = tworld[1]
    assert [r["n_updates"] for r in recs] == [3, 5]
    for n in (3, 5):
        assert (out / train.cells.record_name("C", n, SEED)).exists()
    assert not (out / train.cells.record_name("C", 4, SEED)).exists()
    snap = train.cells.snap_dir("C", SEED, out)
    assert json.loads((snap / "state.json").read_text())["update"] == 4
    assert all((snap / f).exists() for f in ("judge.pt", "jt.pt", "opt.pt"))
    r = recs[-1]
    assert r["arm"] == "C" and r["max_len_schedule"] == "14" and r["device"] == "cpu"
    assert r["continued_from"] == {"experiment": "EXP-082", "update": START}
    for k in ("s_per_update", "n_exclude", "n_probe", "encoder_drift", "gate_l", "git_commit",
              "torch_version", "device_name", "resumed_from"):
        assert k in r, k


def test_a_rerun_skips_finished_legs(tworld):
    """REVIEW FOCUS 1. A relaunched chain must not retrain or re-record a finished seed."""
    _trun(tworld)
    p = tworld[1] / train.cells.record_name("C", 5, SEED)
    before = p.read_bytes()
    assert _trun(tworld) == [] and p.read_bytes() == before


def test_a_checkpoint_past_a_leg_end_is_never_recorded_as_it(tworld, monkeypatch):
    _trun(tworld)
    (tworld[1] / train.cells.record_name("C", 3, SEED)).unlink()
    with pytest.raises(SystemExit, match="past leg end"):
        _trun(tworld)


def test_arm_o_resumed_mid_ramp_uses_the_absolute_update(tworld, monkeypatch):
    """REVIEW FOCUS 3. Catches a resumed arm O restarting its ramp from 14."""
    seen = []
    real = vi.random_walk_states
    monkeypatch.setattr(train.vi, "random_walk_states",
                        lambda n, ml, *a, **k: (seen.append(ml), real(n, ml, *a, **k))[1])
    monkeypatch.setattr(train.cells, "LEGS", (3,))
    _trun(tworld, "O")
    monkeypatch.setattr(train.cells, "LEGS", (3, 4, 5))
    _trun(tworld, "O")
    sched = train.cells.max_len_schedule("O")
    assert seen == [sched(u) for u in (2, 3, 4)] == [14, 18, 22]


def test_both_arms_exclude_the_probe(tworld, monkeypatch):
    """Catches training handed EXP-075's exclusion set without the probe states."""
    probe_states = {s for s, _ in _probe()}
    monkeypatch.setattr(train.gatec, "sets",
                        lambda seed: (set(probe_states), _probe(), None))
    seen = []
    real = train.vi.train_judge
    monkeypatch.setattr(train.vi, "train_judge",
                        lambda *a, **k: (seen.append(k["exclude"]), real(*a, **k))[1])
    _trun(tworld, "O")
    assert seen and all(probe_states <= e for e in seen)


def test_run_refuses_an_arm_not_rebased(world, monkeypatch):
    monkeypatch.setattr(train.cells, "START", START)
    with pytest.raises(SystemExit, match="not rebased"):
        train.run("C", SEED, "cpu", world[2])


def test_pending_is_seed_major_c_before_o_and_skips_finished(tmp_path):
    (tmp_path / train.cells.record_name("C", train.cells.END, 0)).write_text("{}")
    got = train.pending(tmp_path)
    assert got[:3] == [(0, "O"), (1, "C"), (1, "O")] and len(got) == 23


evaluate = _load("exp085_evaluate", EXP / "evaluate.py")
gateh = _load("exp085_gateh", EXP / "gateh.py")


def test_search_cell_is_exp082s_j4v(tmp_path, monkeypatch):
    """Catches the cell drifting from EXP-082's J4V (mode, k, goal check, no-revisit, EXP-075's
    held-out states and streams), or the judge loaded from the wrong arm or update."""
    seen = {}
    monkeypatch.setattr(evaluate.e75, "load_judge",
                        lambda arm, seed, root, required_updates, device: seen.update(
                            arm=arm, seed=seed, root=root, req=required_updates, dev=device) or "J")
    monkeypatch.setattr(evaluate.e75, "JudgeCritic", lambda j: ("critic", j))
    monkeypatch.setattr(evaluate.e75, "load_cell",
                        lambda d, s: ("agent", "head", ["s1", "s2", "s3"], 77))

    def fake_eval(agent, head, states, **k):
        seen.update(states=states, **k)
        return {"solved": 1, "n": len(states), "success_rate": 0.5, "first_fire_len": {"4": 1}}
    monkeypatch.setattr(evaluate, "evaluate_lookahead", fake_eval)
    ck = evaluate.cells.ckpt_dir("O", 3, tmp_path)
    ck.mkdir(parents=True)
    (ck / "state.json").write_text(json.dumps({"update": 24000}))
    rec = evaluate.run_cell("O", 11, 3, "cpu", tmp_path)
    assert seen["arm"] == "X" and seen["root"] == evaluate.cells.arm_root("O", tmp_path)
    assert seen["req"] == 24000 and seen["dev"] == "cpu" and seen["states"] == ["s1", "s2", "s3"]
    assert (seen["depth"], seen["mode"], seen["k"], seen["k_goal"]) == (11, "C", 4, 4)
    assert seen["no_revisit"] is True and seen["rng_seed"] == seen["imag_seed"] == 77
    assert seen["critic"] == ("critic", "J")
    assert rec["arm"] == "J4V-O" and rec["judge"] == "O" and rec["judge_updates"] == 24000
    assert rec["device"] == "cpu" and rec["limit_states"] is None
    assert (tmp_path / evaluate.cells.search_name("O", 11, 3)).exists()


def test_search_refuses_a_judge_short_of_24000(tmp_path):
    ck = evaluate.cells.ckpt_dir("C", 3, tmp_path)
    ck.mkdir(parents=True)
    (ck / "judge.pt").write_bytes(b"x")
    (ck / "state.json").write_text(json.dumps({"update": 20000}))
    with pytest.raises(SystemExit, match="24000"):
        evaluate.run_cell("C", 9, 3, "cpu", tmp_path)


def test_search_refuses_an_unregistered_depth(tmp_path):
    with pytest.raises(SystemExit, match="registered depths"):
        evaluate.run_cell("C", 10, 3, "cpu", tmp_path)


def test_search_pending_runs_depth_11_first_and_skips_finished(tmp_path):
    (tmp_path / evaluate.cells.search_name("C", 11, 0)).write_text("{}")
    got = evaluate.pending(tmp_path)
    assert got[0] == ("O", 11, 0) and len(got) == 47
    assert all(d == 11 for _, d, _ in got[:23]) and all(d == 9 for _, d, _ in got[23:])


def test_picks_are_fixed_and_distinct():
    a, b = gateh.search_picks(), gateh.search_picks()
    assert a == b and len(set(a)) == 4
    assert sorted(d for _, d, _ in a) == [9, 9, 11, 11]
    t = gateh.train_picks()
    assert t == gateh.train_picks() and len(set(t)) == 4
    assert sorted(x for x, _ in t) == ["C", "C", "O", "O"]


def test_gate_h_a_compares_everything_but_wall_and_commit():
    """Catches a replay check that compares only `solved` (a corrupted search can keep the count
    and change the paths) or that trips on wall time."""
    r = {"solved": 5, "first_fire_len": {"4": 2}, "wall_s": 1.0, "git_commit": "a", "n": 200}
    assert gateh.same(r, {**r, "wall_s": 9.0, "git_commit": "b"})
    assert not gateh.same(r, {**r, "solved": 6})
    assert not gateh.same(r, {**r, "first_fire_len": {"4": 3}})


def test_gate_h_a_replays_into_its_own_records(tmp_path, monkeypatch):
    calls = []
    for arm, d, s in gateh.search_picks():
        (tmp_path / gateh.cells.search_name(arm, d, s)).write_text("{}")
    monkeypatch.setattr(gateh.evaluate, "run_cell",
                        lambda arm, d, s, dev, out, name: calls.append((arm, d, s, name)) or {})
    gateh.run_a("cpu", tmp_path)
    assert [(a, d, s) for a, d, s, _ in calls] == gateh.search_picks()
    assert all(n == gateh.cells.h_search_name(a, d, s) for a, d, s, n in calls)


def test_gate_h_b_is_unavailable_when_preflight_says_not_deterministic(tmp_path, monkeypatch):
    (tmp_path / gateh.cells.PREFLIGHT_NAME).write_text(json.dumps({"deterministic": False}))
    monkeypatch.setattr(gateh.train, "train_to", lambda *a, **k: pytest.fail("replayed"))
    gateh.run_b("cpu", tmp_path)
    for arm, s in gateh.train_picks():
        r = json.loads((tmp_path / gateh.cells.h_train_name(arm, s)).read_text())
        assert r["available"] is False and "equal" not in r


class _J:
    def __init__(self, sd):
        self._sd = sd

    def state_dict(self):
        return self._sd


@pytest.mark.parametrize("v,want", [(1.0, True), (1.5, False)])
def test_gate_h_b_compares_the_replayed_judge_tensor_by_tensor(tmp_path, monkeypatch, v, want):
    (tmp_path / gateh.cells.PREFLIGHT_NAME).write_text(json.dumps({"deterministic": True}))
    monkeypatch.setattr(gateh.gatec, "sets", lambda seed: (set(), [], None))
    for arm, s in gateh.train_picks():
        snap = gateh.cells.snap_dir(arm, s, tmp_path)
        snap.mkdir(parents=True)
        for f in ("judge.pt", "jt.pt", "opt.pt", "state.json"):
            (snap / f).write_text("{}")
        torch.save({"w": torch.tensor([1.0])}, gateh.cells.ckpt_dir(arm, s, tmp_path) / "judge.pt")
    monkeypatch.setattr(gateh.train, "train_to",
                        lambda ck, arm, seed, dev, n, **k: (_J({"w": torch.tensor([v])}), {}))
    gateh.run_b("cpu", tmp_path)
    for arm, s in gateh.train_picks():
        r = json.loads((tmp_path / gateh.cells.h_train_name(arm, s)).read_text())
        assert r["available"] is True and r["equal"] is want
