"""EXP-076: a deeper look-ahead tree over EXP-075's committed X judges."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from neuromorphic.envs.cube_distance import ExactBFSDistance

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "076_deeper_tree"
E75 = REPO / "experiments" / "075_wide_region"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


evaluate = _load("exp076_evaluate", EXP / "evaluate.py")


def test_arms_are_the_spec_table():
    """Catches an arm wired to the wrong tree or goal depth (spec section 3), or J3V-X given an
    explicit goal depth instead of the default that Gate 0(b) checks."""
    assert evaluate.ARMS == {"J3V-X": (3, None), "J3VG4-X": (3, 4), "J4V-X": (4, 4)}
    assert evaluate.DEPTHS == (9, 11) and evaluate.RANK_K == 4


def test_j3v_x_here_equals_exp075s_harness_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH. Catches this harness drifting from EXP-075's: J3V-X through it
    equals EXP-075's own run_cell on the same states, committed judge, in every outcome field."""
    mine = evaluate.run_cell("J3V-X", 9, 0, tmp_path / "a", limit_states=2)
    ref = evaluate.e75.run_cell("J3V-X", 9, 0, tmp_path / "b", limit_states=2)
    for f in ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
              "greedy_modal_action_frac", "goal_fired_frac", "fallback_frac"):
        assert mine[f] == ref[f], f
    assert mine["k"] == 3 and mine["k_goal"] == 3


def test_new_arms_write_their_tree_and_goal_depths_and_fired4(tmp_path):
    """Catches a record that does not say which tree it ran (so the aggregator could not refuse
    a mislabelled one), and fired4 not taken from the first-fire histogram."""
    for arm, (k, kg) in (("J3VG4-X", (3, 4)), ("J4V-X", (4, 4))):
        rec = evaluate.run_cell(arm, 7, 0, tmp_path, limit_states=1)
        assert rec["arm"] == arm and rec["k"] == k and rec["k_goal"] == kg
        assert rec["mode"] == "C" and rec["no_revisit"] is True
        assert rec["fired4"] == rec["first_fire_len"].get("4", 0)
        assert rec["limit_states"] == 1 and rec["judge_updates"] == 4000
        on_disk = json.loads((tmp_path / f"exp076_{arm}_d7_s0.json").read_text())
        assert on_disk["k_goal"] == kg


def test_goal_check_arm_never_solves_less_than_j3v_on_a_slice(tmp_path):
    """THE GATE D INVARIANT ON THE REAL JUDGE. Catches the goal-check arm using a different
    stream from J3V-X: on the same states it must solve at least as many."""
    a = evaluate.run_cell("J3V-X", 9, 0, tmp_path, limit_states=6)
    b = evaluate.run_cell("J3VG4-X", 9, 0, tmp_path, limit_states=6)
    assert b["solved"] >= a["solved"]


def test_unknown_arm_and_missing_judge_exit_cleanly(tmp_path):
    """Catches a typo'd arm running some default, and a missing judge evaluating an untrained
    one."""
    with pytest.raises(SystemExit, match="unknown arm"):
        evaluate.run_cell("J5V-X", 9, 0, tmp_path, limit_states=1)
    with pytest.raises(SystemExit, match="judge.pt"):
        evaluate.run_cell("J4V-X", 9, 0, tmp_path, limit_states=1, judge_dir=tmp_path / "no")


def test_rank_cell_ranks_four_move_leaves(tmp_path, monkeypatch):
    """Catches Gate R measured on EXP-075's 3-move leaves instead of 4, and a rank record
    without its limit (so a smoke row could enter Gate R)."""
    seen = []
    real = evaluate.tree_levels

    def spy(state, k, n):
        seen.append(k)
        return real(state, k, n)

    monkeypatch.setattr(evaluate, "tree_levels", spy)
    rec = evaluate.rank_cell(7, 0, tmp_path, limit_states=2,
                             provider=ExactBFSDistance(max_depth=11))
    assert set(seen) == {4}
    assert rec["kind"] == "J-X4" and rec["k"] == 4 and rec["n"] == 2
    assert rec["limit_states"] == 2 and rec["judge_updates"] == 4000
    assert rec["wall_s"] >= 0
    assert 0.0 < rec["chance"] < 1.0 and rec["hit"] in (0.0, 0.5, 1.0)
    assert (tmp_path / "exp076_rank_J-X4_d7_s0.json").exists()


agg = _load("exp076_aggregate", EXP / "aggregate.py")


def _rec(arm, depth, seed, solved, **over):
    k, kg = evaluate.ARMS[arm]
    r = {"arm": arm, "depth": depth, "seed": seed, "solved": solved, "n": 200,
         "success_rate": solved / 200, "k": k, "k_goal": kg if kg is not None else k,
         "mode": "C", "no_revisit": True, "judge_updates": 4000, "limit_states": None,
         "fired4": 3, "first_fire_len": {"4": 3}, "mean_steps": 10.0, "optimality": 0.9}
    r.update(over)
    return r


def _recs(j3, g4, j4, seeds=range(12)):
    return {(arm, d): {s: _rec(arm, d, s, v) for s in seeds}
            for arm, v in (("J3V-X", j3), ("J3VG4-X", g4), ("J4V-X", j4)) for d in (9, 11)}


def test_alpha_small_and_the_small_verdict():
    """Catches the claims read at another alpha, and a confirmed gain under 0.01 printed as a
    plain CONFIRMED (spec section 8's band must be a verdict, not prose)."""
    assert agg.ALPHA == pytest.approx(0.05 / 3) and agg.SMALL == 0.01
    assert agg.claim_verdict([0.05] * 12, 0.30, 0.25, True)[0] == "CONFIRMED"
    assert agg.claim_verdict([0.005] * 12, 0.255, 0.25, True)[0] == "CONFIRMED (SMALL)"
    assert agg.claim_verdict([0.05] * 12, 0.30, 0.25, False)[0] == "VOID"
    assert agg.claim_verdict([-0.01] * 12, 0.24, 0.25, True)[0] == "REFUTED"
    assert agg.claim_verdict([0.0] * 8 + [0.01] * 4, 0.255, 0.25, True)[0] == "NOT SIGNIFICANT"
    assert agg.claim_verdict([0.005] * 12, 0.015, 0.01, True)[0] == "UNRESOLVED"


def test_claims_compare_the_registered_arms_and_depths():
    """Catches a claim wired to the wrong arm or depth: every arm and depth gets a distinct rate,
    so each claim's two means identify exactly which cells it read."""
    rates = {("J3V-X", 9): 0.25, ("J3VG4-X", 9): 0.26, ("J4V-X", 9): 0.30,
             ("J3V-X", 11): 0.12, ("J3VG4-X", 11): 0.13, ("J4V-X", 11): 0.17}
    rates = {k: {s: v for s in range(12)} for k, v in rates.items()}
    v = agg.primary_verdicts(rates, True, range(12))
    assert v["claim1"][3:] == (pytest.approx(0.17), pytest.approx(0.12))
    assert v["claim2"][3:] == (pytest.approx(0.30), pytest.approx(0.25))
    assert v["claim3"][3:] == (pytest.approx(0.17), pytest.approx(0.13))


def test_sensitivity_drops_exactly_seeds_0_and_3():
    """Catches the 10-seed line dropping the wrong seeds or none."""
    rates = {("J4V-X", 11): {s: float(s) for s in range(12)}}
    assert sorted(agg.sensitivity_rates(rates)[("J4V-X", 11)]) == [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]


def test_gate_d_names_the_violating_cell():
    """Catches Gate D comparing a different pairing (another seed or depth) or passing with one
    violation: exactly the broken cell must be reported."""
    recs = _recs(30, 32, 40)
    assert agg.gate_d(recs) == []
    recs[("J3VG4-X", 11)][7] = _rec("J3VG4-X", 11, 7, 29)
    assert agg.gate_d(recs) == [(11, 7)]
    # the pairing is per seed: a J3V-X seed that scores higher than seed 0 must be compared to itself
    recs = _recs(30, 32, 40)
    recs[("J3V-X", 9)][4] = _rec("J3V-X", 9, 4, 35)
    assert agg.gate_d(recs) == [(9, 4)]


def test_gate_g_is_unresolved_only_when_the_deeper_check_never_fired():
    """Catches Gate G reading fired4 from the wrong arm, or summing over both depths."""
    recs = _recs(30, 32, 40)
    assert agg.gate_g(recs) == {9: "RESOLVED", 11: "RESOLVED"}
    for s in range(12):
        recs[("J3VG4-X", 9)][s] = _rec("J3VG4-X", 9, s, 32, fired4=0)
        recs[("J4V-X", 9)][s] = _rec("J4V-X", 9, s, 40, fired4=9)
    assert agg.gate_g(recs) == {9: "UNRESOLVED", 11: "RESOLVED"}


def test_smoke_and_wrong_k_records_are_refused(tmp_path):
    """REVIEW FOCUS 1. Catches a smoke record or a record from another tree entering a verdict."""
    p = tmp_path / "exp076_J4V-X_d9_s0.json"
    p.write_text(json.dumps(_rec("J4V-X", 9, 0, 40, limit_states=20)))
    with pytest.raises(SystemExit, match="smoke"):
        agg._read(p)
    with pytest.raises(SystemExit, match="k_goal"):
        agg.check_eval_record(_rec("J4V-X", 9, 0, 40, k_goal=3), p)
    with pytest.raises(SystemExit, match="k is"):
        agg.check_eval_record(_rec("J4V-X", 9, 0, 40, k=3), p)
    with pytest.raises(SystemExit, match="arm"):
        agg.check_eval_record({**_rec("J4V-X", 9, 0, 40), "arm": "J3V-W"}, p)


def test_partly_trained_records_are_refused(tmp_path):
    """REVIEW FOCUS 2. Catches a row evaluated from a judge short of 4000 updates entering a
    claim, including the EXP-075 J3V-X rows this experiment reuses."""
    p = tmp_path / "x.json"
    with pytest.raises(SystemExit, match="judge_updates"):
        agg.check_eval_record(_rec("J3V-X", 11, 0, 30, judge_updates=3750), p)


def test_load_records_reads_j3v_from_exp075_and_new_arms_from_out(tmp_path):
    """Catches J3V-X read from this experiment's outputs (only Gate 0(b)'s seed 0 is re-run
    here) and refuses a missing new-arm record by name."""
    for arm in ("J3VG4-X", "J4V-X"):
        for d in (9, 11):
            for s in range(12):
                (tmp_path / evaluate.record_name(arm, d, s)).write_text(
                    json.dumps(_rec(arm, d, s, 50)))
    recs = agg.load_records(tmp_path)
    committed = json.loads((evaluate.E75_OUT / "exp075_J3V-X_d9_s0.json").read_text())
    assert recs[("J3V-X", 9)][0]["solved"] == committed["solved"]
    assert recs[("J4V-X", 11)][5]["solved"] == 50
    (tmp_path / evaluate.record_name("J4V-X", 11, 5)).unlink()
    with pytest.raises(SystemExit, match="exp076_J4V-X_d11_s5"):
        agg.load_records(tmp_path)


def test_stage_cont_compares_against_exp075s_committed_records(tmp_path):
    """THE GATE 0(b) VERDICT. Catches the continuity check comparing a record with itself, or
    against anything but EXP-075's committed J3V-X seed-0 records."""
    for d in (9, 11):
        ref = json.loads((evaluate.E75_OUT / f"exp075_J3V-X_d{d}_s0.json").read_text())
        (tmp_path / evaluate.record_name("J3V-X", d, 0)).write_text(json.dumps(ref))
    assert agg.stage_cont(tmp_path) == "PASS"
    bad = json.loads((tmp_path / evaluate.record_name("J3V-X", 11, 0)).read_text())
    bad["solved"] += 1
    (tmp_path / evaluate.record_name("J3V-X", 11, 0)).write_text(json.dumps(bad))
    assert agg.stage_cont(tmp_path) == "FAIL"


def test_pilot_report_projects_cell_hours_from_seconds_per_state(tmp_path):
    """Catches the amendment's schedule computed from wall_s without dividing by n, or from a
    pilot at another size."""
    for arm, wall in (("J3VG4-X", 200.0), ("J4V-X", 600.0)):
        (tmp_path / evaluate.record_name(arm, 11, 0)).write_text(json.dumps(
            _rec(arm, 11, 0, 3, n=20, limit_states=20, wall_s=wall)))
    (tmp_path / evaluate.rank_record_name(11, 0)).write_text(json.dumps(
        {"kind": "J-X4", "k": 4, "n": 20, "limit_states": 20, "hit": 0.6, "chance": 0.5,
         "wall_s": 40.0}))
    r = agg.pilot_report(tmp_path)
    assert r["s_per_state"]["J4V-X"] == pytest.approx(30.0)
    assert r["d11_cell_hours"]["J4V-X"] == pytest.approx(30.0 * 200 / 3600)
    bad = _rec("J4V-X", 11, 0, 3, n=5, limit_states=5, wall_s=1.0)
    (tmp_path / evaluate.record_name("J4V-X", 11, 0)).write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="limit_states"):
        agg.pilot_report(tmp_path)
