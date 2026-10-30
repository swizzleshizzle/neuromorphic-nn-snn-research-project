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
    # Each shell mixes 30 states the replay trains on with unreached ones, so a build that
    # forgot `seen` would pick trained-on states (final review, Important 1).
    seen = sorted(sets.replay(SEED, {1000, 1001}, (n,), batch=4)[0])
    shells = {9: list(range(1000, 1600)) + seen[:30], 10: list(range(2000, 2600)) + seen[30:60]}
    kw = dict(exclude_fn=lambda seed, prov: {1000, 1001}, shells_fn=lambda prov: shells,
              provider_fn=lambda: None, e75_root=e75, e82_root=e82, ends=(1, n), batch=4)
    return tmp_path / "out", kw


def test_build_writes_sets_when_gate_v_passes(tmp_path):
    out, kw = _sets_world(tmp_path)
    rec = sets.build(SEED, out, **kw)
    assert rec["gate_v"] == {"u4000": True, "u16000": True} and rec["pass"] is True
    assert {k: len(v) for k, v in rec["sets"].items()} == {"9": 200, "10": 200}
    assert 1000 not in rec["sets"]["9"] and 1001 not in rec["sets"]["9"]
    seen = sets.replay(SEED, {1000, 1001}, (3,), batch=4)[0]
    for d in ("9", "10"):
        assert not any(isinstance(x, list) and tuple(x) in seen for x in rec["sets"][d])
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


evaluate = _load("exp083_evaluate", EXP / "evaluate.py")


def _write_sets(out, seed, ok=True, n=3):
    states = []
    s = SOLVED
    for m in (0, 2, 4, 1, 3, 5, 0, 2, 4, 1, 3, 5):
        s = apply_move(s, m)
        states.append(list(s))
    rec = {"seed": seed, "pass": ok, "gate_v": {"u4000": ok, "u16000": ok},
           "sets": {"9": states[:n] if ok else [], "10": states[-n:] if ok else []}}
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / cells.sets_name(seed)).write_text(json.dumps(rec))


def test_cell_refuses_a_missing_or_failed_sets_file(tmp_path):
    """REVIEW FOCUS 2. A seed whose replay did not match must never be searched."""
    with pytest.raises(SystemExit, match="no sets file"):
        evaluate.load_states(0, 9, tmp_path)
    _write_sets(tmp_path, 0, ok=False)
    with pytest.raises(SystemExit, match="Gate V"):
        evaluate.load_states(0, 9, tmp_path)
    _write_sets(tmp_path, 0)
    got = evaluate.load_states(0, 10, tmp_path)
    assert len(got) == 3 and all(isinstance(s, tuple) for s in got)


def test_judge_for_reads_each_arm_from_its_own_experiment(monkeypatch):
    calls = []
    monkeypatch.setattr(evaluate.e75, "load_judge",
                        lambda arm, seed, root, required_updates: calls.append(
                            (arm, seed, root, required_updates)))
    evaluate.judge_for("J4V-X16k", 3)
    evaluate.judge_for("J4V-X", 3)
    assert calls == [("X", 3, cells.E82_OUT, 16000), ("X", 3, cells.E75_OUT, 4000)]
    with pytest.raises(SystemExit, match="arm"):
        evaluate.judge_for("J4V-W", 3)


def test_a_real_cell_runs_at_depth_10_on_the_sets_file(tmp_path):
    """Catches a record missing a field the aggregator checks, on the real depth-9 cell's
    agent and head at depth 10."""
    e82 = tmp_path / "e82"
    t75tests._fake_judge_ckpt(e82, "X", 0, update=16000)
    out = tmp_path / "out"
    _write_sets(out, 0)
    r = evaluate.run_cell("J4V-X16k", 10, 0, out, limit_states=2, e82_root=e82)
    assert (r["arm"], r["judge"], r["depth"], r["seed"], r["judge_seed"]) == (
        "J4V-X16k", "X16k", 10, 0, 0)
    assert (r["k"], r["k_goal"], r["mode"], r["no_revisit"]) == (4, 4, "C", True)
    assert r["judge_updates"] == 16000 and r["n"] == 2 and r["limit_states"] == 2
    assert r["states_from"] == cells.sets_name(0)
    assert json.loads((out / cells.search_name("J4V-X16k", 10, 0)).read_text())["depth"] == 10


def test_a_cell_refuses_a_judge_at_the_wrong_update(tmp_path):
    e75 = tmp_path / "e75"
    t75tests._fake_judge_ckpt(e75, "X", 0, update=3000)
    _write_sets(tmp_path / "out", 0)
    with pytest.raises(SystemExit, match="4000"):
        evaluate.run_cell("J4V-X", 9, 0, tmp_path / "out", limit_states=1, e75_root=e75)


def _fake_search(solved, n=200):
    def f(judge, depth, seed, states):
        return {"solved": solved, "n": n, "success_rate": solved / n, "first_fire_len": {}}
    return f


def test_gate_r_writes_only_into_exp083(tmp_path, monkeypatch):
    """REVIEW FOCUS 1. EXP-082's run_cell writes where it reads its judge; Gate R must not."""
    e82 = tmp_path / "e82"
    e82.mkdir()
    (e82 / "exp082_J4V-X16k_d9_s0.json").write_text(json.dumps(
        {"solved": 127, "success_rate": 0.635}))
    before = sorted(p.name for p in e82.rglob("*"))
    monkeypatch.setattr(evaluate, "judge_for", lambda arm, seed, **kw: None)
    monkeypatch.setattr(evaluate.e75, "load_cell", lambda d, s: (None, None, [1, 2], 7))
    monkeypatch.setattr(evaluate, "_search", _fake_search(127))
    r = evaluate.gate_r(tmp_path / "out", e82_root=e82)
    assert r["pass"] is True and (tmp_path / "out" / cells.GATE_R_NAME).exists()
    assert sorted(p.name for p in e82.rglob("*")) == before
    monkeypatch.setattr(evaluate, "_search", _fake_search(126))
    assert evaluate.gate_r(tmp_path / "out", e82_root=e82)["pass"] is False


def test_gate_r_compares_against_exp082s_committed_record():
    r = json.loads((cells.E82_OUT / "exp082_J4V-X16k_d9_s0.json").read_text())
    assert (r["solved"], r["success_rate"]) == (127, 0.635)


def test_search_pending_order_and_skip(tmp_path):
    (tmp_path / cells.search_name("J4V-X", 10, 1)).write_text("{}")
    jobs = evaluate.pending([1, 2], tmp_path)
    assert jobs[0] == ("gate_r", 9, 0)
    assert ("J4V-X", 10, 1) not in jobs and len(jobs) == 1 + 2 * 2 * 2 - 1
    assert [d for _, d, _ in jobs[1:]] == sorted([d for _, d, _ in jobs[1:]], reverse=True)
    (tmp_path / cells.GATE_R_NAME).write_text("{}")
    assert evaluate.pending([1], tmp_path)[0] != ("gate_r", 9, 0)


floor = _load("exp083_floor", EXP / "floor.py")


def test_rollout_solves_solved_at_once_and_a_one_move_state_sometimes():
    assert floor.random_rollout(SOLVED, 1, random.Random(0)) is True
    one = apply_move(SOLVED, 0)
    rate = sum(floor.random_rollout(one, 1, random.Random(i)) for i in range(400)) / 400
    assert 0.15 < rate < 0.6


def test_rollout_budget_is_2d_plus_3(monkeypatch):
    calls = []
    real = floor.apply_move
    monkeypatch.setattr(floor, "apply_move", lambda s, a: calls.append(a) or real(s, a))
    far = SOLVED
    for m in (0, 2, 4, 1, 3, 5, 0, 2, 4, 1):
        far = apply_move(far, m)
    lens = []
    for i in range(20):
        calls.clear()
        floor.random_rollout(far, 10, random.Random(i))
        lens.append(len(calls))
    assert max(lens) == 23


def test_floor_cell_reads_the_sets_file(tmp_path, monkeypatch):
    _write_sets(tmp_path, 0)
    monkeypatch.setattr(floor, "train_seed", lambda seed: 5)
    r = floor.floor_cell(10, 0, tmp_path)
    assert (r["depth"], r["seed"], r["n"], r["rng_seed"]) == (10, 0, 3, 5)
    assert r["success_rate"] == r["solved"] / 3
    assert json.loads((tmp_path / cells.floor_name(10, 0)).read_text()) == r
    assert floor.pending([0, 1], tmp_path) == [(9, 0), (9, 1), (10, 1)]


agg = _load("exp083_aggregate", EXP / "aggregate.py")
SEEDS = list(range(12))


def _srec(arm, d, s, rate, **over):
    r = {"arm": arm, "judge": "X16k" if arm == "J4V-X16k" else "X", "depth": d, "seed": s,
         "judge_seed": s, "k": 4, "k_goal": 4, "mode": "C", "no_revisit": True, "n": 200,
         "limit_states": None, "judge_updates": 16000 if arm == "J4V-X16k" else 4000,
         "success_rate": rate, "solved": round(rate * 200), "optimality": 0.9,
         "mean_steps": 10.0, "goal_fired_frac": 0.1, "fallback_frac": 0.0, "git_commit": "c"}
    r.update(over)
    return r


def _agg_world(tmp, *, gain9=0.20, gain10=0.08, x9=0.40, x10=0.30, floor=0.0, v_ok=True,
               r_ok=True, short=False, jitter=True):
    out, e76, e82 = tmp / "out", tmp / "e76", tmp / "e82"
    for p in (out, e76, e82):
        p.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        j = (s % 3 - 1) * 0.01 if jitter else 0.0
        n = 150 if (short and s == 0) else 200
        (out / cells.sets_name(s)).write_text(json.dumps(
            {"seed": s, "pass": v_ok or s != 3, "gate_v": {"u4000": True, "u16000": v_ok or s != 3},
             "n_seen": 1000, "candidates": {"9": 5000, "10": 9000},
             "sets": {"9": [[0]] * n, "10": [[0]] * 200}}))
        for d, x, g in ((9, x9, gain9), (10, x10, gain10)):
            (out / cells.search_name("J4V-X", d, s)).write_text(json.dumps(
                _srec("J4V-X", d, s, x)))
            (out / cells.search_name("J4V-X16k", d, s)).write_text(json.dumps(
                _srec("J4V-X16k", d, s, x + g + j)))
            (out / cells.floor_name(d, s)).write_text(json.dumps(
                {"depth": d, "seed": s, "n": 200, "solved": round(floor * 200),
                 "success_rate": floor}))
        (e76 / f"exp076_J4V-X_d9_s{s}.json").write_text(json.dumps({"success_rate": 0.40}))
        (e82 / f"exp082_J4V-X16k_d9_s{s}.json").write_text(json.dumps({"success_rate": 0.62}))
    (out / cells.GATE_R_NAME).write_text(json.dumps(
        {"pass": r_ok, "solved": 127 if r_ok else 120, "expected_solved": 127,
         "git_commit": "c"}))
    return dict(out=out, e76_root=e76, e82_root=e82)


def _verdict(tmp, **kw):
    return agg.run_all(**_agg_world(tmp, **kw))["verdict"]


def test_bars_are_the_spec_values():
    assert (agg.C1_MIN, agg.C2_MIN, agg.FLOOR_MARGIN, agg.CLAIM_ALPHA) == (0.10, 0.05, 0.05, 0.025)


def test_verdict_has_a_line_for_every_row_of_spec_section_7():
    assert agg.verdict(False, True, True, True, "CONFIRMED", "CONFIRMED") == "VOID"
    assert agg.verdict(True, False, True, True, "CONFIRMED", "CONFIRMED") == "VOID"
    assert agg.verdict(True, True, False, True, "CONFIRMED", "CONFIRMED") == "UNRESOLVED"
    assert agg.verdict(True, True, True, False, "UNRESOLVED", "CONFIRMED") == "UNRESOLVED"
    assert agg.verdict(True, True, True, True, "CONFIRMED", "CONFIRMED") == \
        "DEPTH-9 GAIN CONFIRMED, REACHES DEPTH 10"
    for c2 in ("CONFIRMED (SMALL)", "NOT CONFIRMED", "UNRESOLVED"):
        assert agg.verdict(True, True, True, True, "CONFIRMED", c2) == \
            "DEPTH-9 GAIN CONFIRMED, STOPS SHORT OF DEPTH 10"
    assert agg.verdict(True, True, True, True, "CONFIRMED (SMALL)", "CONFIRMED") == \
        "DEPTH-9 GAIN SMALLER THAN REPORTED"
    assert agg.verdict(True, True, True, True, "NOT CONFIRMED", "CONFIRMED") == \
        "DEPTH-9 GAIN DOES NOT HOLD ON UNSEEN STATES"


def test_a_large_gain_at_both_depths_reaches_depth_10(tmp_path):
    assert _verdict(tmp_path) == "DEPTH-9 GAIN CONFIRMED, REACHES DEPTH 10"


def test_a_small_depth_10_gain_stops_short(tmp_path):
    assert _verdict(tmp_path, gain10=0.02) == "DEPTH-9 GAIN CONFIRMED, STOPS SHORT OF DEPTH 10"


def test_small_significant_depth9_gain_reads_smaller_than_reported(tmp_path):
    """The EXP-068 rule: the band between significance and the bar is its own verdict."""
    assert _verdict(tmp_path, gain9=0.05) == "DEPTH-9 GAIN SMALLER THAN REPORTED"


def test_no_depth9_gain_does_not_hold(tmp_path):
    assert _verdict(tmp_path, gain9=0.0) == "DEPTH-9 GAIN DOES NOT HOLD ON UNSEEN STATES"


def test_gate_f_failing_at_depth_10_only_reads_stops_short(tmp_path):
    """REVIEW FOCUS 4."""
    res = agg.run_all(**_agg_world(tmp_path, x10=0.0, gain10=0.03, floor=0.0, x9=0.40))
    assert res["c2"][0] == "UNRESOLVED"
    assert res["verdict"] == "DEPTH-9 GAIN CONFIRMED, STOPS SHORT OF DEPTH 10"


def test_gate_f_failing_at_depth_9_is_unresolved(tmp_path):
    assert _verdict(tmp_path, x9=0.30, gain9=0.20, floor=0.48) == "UNRESOLVED"


def test_a_failed_gate_v_or_gate_r_is_void(tmp_path):
    assert _verdict(tmp_path / "a", v_ok=False) == "VOID"
    assert _verdict(tmp_path / "b", r_ok=False) == "VOID"


def test_a_short_set_is_unresolved(tmp_path):
    """Final review: a short set's real search records hold n < 200, which read_search
    refuses; the aggregate must still print the Gate S row, not exit 1."""
    w = _agg_world(tmp_path, short=True)
    for a in cells.ARMS:
        p = w["out"] / cells.search_name(a, 9, 0)
        p.write_text(json.dumps({**json.loads(p.read_text()), "n": 150}))
    assert agg.run_all(**w)["verdict"] == "UNRESOLVED"


@pytest.mark.parametrize("arm,over,msg", [
    ("J4V-X", {"judge_updates": 16000}, "judge_updates"),
    ("J4V-X16k", {"judge": "X"}, "judge"),
    ("J4V-X", {"limit_states": 2}, "limit_states"),
    ("J4V-X16k", {"n": 199}, "n"),
    ("J4V-X", {"depth": 11}, "depth"),
])
def test_a_record_that_is_not_the_registered_cell_is_refused(tmp_path, arm, over, msg):
    """REVIEW FOCUS 3."""
    w = _agg_world(tmp_path)
    p = w["out"] / cells.search_name(arm, 9, 4)
    p.write_text(json.dumps(_srec(arm, 9, 4, 0.4, **over)))
    with pytest.raises(SystemExit, match=msg):
        agg.run_all(**w)


def test_mixed_checkouts_are_refused(tmp_path):
    w = _agg_world(tmp_path)
    p = w["out"] / cells.search_name("J4V-X", 10, 2)
    p.write_text(json.dumps(_srec("J4V-X", 10, 2, 0.3, git_commit="other")))
    with pytest.raises(SystemExit, match="checkout"):
        agg.run_all(**w)


def test_the_report_prints_one_verdict_line_and_the_reported_block(tmp_path):
    lines = agg.run_all(**_agg_world(tmp_path))["lines"]
    assert sum(ln.startswith("VERDICT: ") for ln in lines) == 1
    text = "\n".join(lines)
    for key in ("CLAIM 1", "CLAIM 2", "GATE V", "GATE R", "GATE S", "GATE F", "old held-out",
                "optimality", "floor", "n_seen"):
        assert key in text, key


def test_sets_stage_lines(tmp_path):
    assert agg.stage_sets_lines(_agg_world(tmp_path / "a")["out"])[-1] == "SETS: OK"
    assert agg.stage_sets_lines(_agg_world(tmp_path / "b", v_ok=False)["out"])[-1] == \
        "VERDICT: VOID"
    assert agg.stage_sets_lines(_agg_world(tmp_path / "c", short=True)["out"])[-1] == \
        "VERDICT: UNRESOLVED"


import re  # noqa: E402

WORKERS = '@{ "sets" = 4; "search" = 6; "floor" = 4 }'


def test_drivers_use_the_spec_worker_counts():
    """Spec section 8: sets 4 (one BFS table each), search 6 (12 was slower per cell in
    EXP-082 for no throughput), floor 4."""
    assert WORKERS in (EXP / "chain083.ps1").read_text()
    assert WORKERS in (EXP / "launch083.ps1").read_text()


def test_drivers_have_no_em_dashes_or_backticks_in_strings():
    for f in ("chain083.ps1", "launch083.ps1"):
        text = (EXP / f).read_text()
        assert "\u2014" not in text, f
        assert not re.search(r'"[^"\n]*`[^"\n]*"', text), f


def test_chain_reads_the_lines_the_aggregator_prints(tmp_path):
    """Catches the chain and the aggregator drifting apart on the sets stage, and a chain that
    would search after a failed Gate V."""
    chain = (EXP / "chain083.ps1").read_text()
    assert '-match "SETS: OK"' in chain and '-match "VERDICT: (VOID|UNRESOLVED)"' in chain
    assert "STOP: the sets stage exited" in chain
    assert re.search(r'verdictFile\s*=\s*"C:\\Users\\mlgbr\\verdict083\.txt"', chain)


def test_a_failed_gate_v_reaches_the_verdict_file_as_void(tmp_path):
    import os
    import subprocess
    import sys
    out = _agg_world(tmp_path, v_ok=False)["out"]
    r = subprocess.run([sys.executable, str(EXP / "aggregate.py"), "--stage", "sets",
                        "--out-dir", str(out)], capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "VERDICT: VOID", r.stderr
