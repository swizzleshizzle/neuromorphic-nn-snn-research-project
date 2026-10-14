"""EXP-077: driver, aggregator and launcher. Each docstring names the bug it catches."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "experiments" / "077_beam_search"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("exp077_evaluate", HERE / "evaluate.py")


def test_arms_are_the_spec_table():
    """Catches an arm whose mode, depth, goal check or width drifted from spec section 3."""
    assert ev.ARMS == {"J4V-X": ("C", 4, 4, None), "J4VG5-X": ("C", 4, 5, None),
                       "B5-X": ("B", 5, 5, 64)}
    assert ev.DEPTHS == (9, 11)


def test_j4v_x_here_equals_exp076s_record_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH. Catches mode B's seam changing mode C: J4V-X here, on the first
    3 depth-9 states of seed 0, equals EXP-076's driver on the same slice in every outcome
    field and in fallback_frac."""
    e76 = _load("exp076_evaluate_t", REPO / "experiments" / "076_deeper_tree" / "evaluate.py")
    mine = ev.run_cell("J4V-X", 9, 0, tmp_path / "a", limit_states=3)
    ref = e76.run_cell("J4V-X", 9, 0, tmp_path / "b", limit_states=3)
    for f in ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
              "goal_fired_frac", "first_fire_len", "fallback_frac"):
        assert mine[f] == ref[f], f


def test_new_arms_record_mode_depths_width_and_fired5(tmp_path):
    """Catches a record that does not say what ran: B5-X carries mode B, k 5, k_goal 5,
    beam_width 64 and fired5; J4VG5-X carries mode C, k 4, k_goal 5."""
    b = ev.run_cell("B5-X", 9, 0, tmp_path, limit_states=2)
    assert (b["mode"], b["k"], b["k_goal"], b["beam_width"]) == ("B", 5, 5, 64)
    assert b["fired5"] == b["first_fire_len"].get("5", 0)
    assert 0 < b["judge_evals"] <= 1026
    g = ev.run_cell("J4VG5-X", 9, 0, tmp_path, limit_states=2)
    assert (g["mode"], g["k"], g["k_goal"]) == ("C", 4, 5) and "beam_width" not in g
    assert g["judge_evals"] == 1296
    assert json.loads((tmp_path / ev.record_name("B5-X", 9, 0)).read_text())["arm"] == "B5-X"


def test_unknown_arm_exits_cleanly(tmp_path):
    """Catches an arm typo running something unregistered."""
    with pytest.raises(SystemExit):
        ev.run_cell("B6-X", 9, 0, tmp_path, limit_states=1)


ag = _load("exp077_aggregate", HERE / "aggregate.py")


def _rec(arm, d, s, solved=50, **kw):
    mode, k, kg, w = ev.ARMS[arm]
    r = {"arm": arm, "depth": d, "seed": s, "mode": mode, "k": k, "k_goal": kg,
         "no_revisit": True, "judge_updates": 4000, "limit_states": None, "n": 200,
         "solved": solved, "success_rate": solved / 200, "first_fire_len": {"5": solved},
         "fired5": solved, "judge_evals": 1296 if mode == "C" else 900, "fallback_frac": 0.0}
    if arm == "J4V-X":
        r["first_fire_len"] = {"4": solved}
    if w is not None:
        r["beam_width"] = w
    r.update(kw)
    return r


def _recs(over=None):
    recs = {}
    for d in ev.DEPTHS:
        for arm in ag.ALL_ARMS:
            recs[(arm, d)] = {s: _rec(arm, d, s) for s in ag.SEEDS}
    for (arm, d, s), r in (over or {}).items():
        recs[(arm, d)][s] = r
    return recs


def test_constants_and_arms():
    """Catches a wrong alpha, a drifted eval bound, or a changed arm list."""
    assert ag.ALPHA == pytest.approx(0.05 / 3) and ag.SMALL == 0.01 and ag.MAX_EVALS == 1026
    assert ag.ALL_ARMS == ("J4V-X", "J4VG5-X", "B5-X")


def test_claims_compare_the_registered_arms_and_depths():
    """Catches a claim wired to the wrong pair or depth (spec section 8)."""
    rates = {(a, d): {s: 0.1 for s in ag.SEEDS} for a in ag.ALL_ARMS for d in ev.DEPTHS}
    for s in ag.SEEDS:
        rates[("B5-X", 11)][s] = 0.2 + 0.001 * s
        rates[("B5-X", 9)][s] = 0.3 + 0.001 * s
    v = ag.primary_verdicts(rates, True, ag.SEEDS)
    assert v["claim1"][0] == "CONFIRMED" and v["claim1"][2] == pytest.approx(0.1055, abs=1e-4)
    assert v["claim2"][2] == pytest.approx(0.2055, abs=1e-4)
    assert v["claim3"][0] == "CONFIRMED"
    rates[("J4VG5-X", 11)] = dict(rates[("B5-X", 11)])
    assert ag.primary_verdicts(rates, True, ag.SEEDS)["claim3"][0] == "REFUTED"


def test_gate_d_clauses_each_name_their_cell():
    """Catches any of the three Gate D clauses missing: dominance, the first-fire key, and
    fired5 below J4V-X's solves, each on its own cell."""
    assert ag.gate_d(_recs()) == []
    bad = _recs({("J4VG5-X", 9, 2): _rec("J4VG5-X", 9, 2, solved=40, fired5=60,
                                             first_fire_len={"5": 60})})
    assert ag.gate_d(bad) == [(9, 2)]
    bad = _recs({("B5-X", 11, 5): _rec("B5-X", 11, 5, first_fire_len={"4": 1, "5": 49})})
    assert ag.gate_d(bad) == [(11, 5)]
    bad = _recs({("J4VG5-X", 11, 7): _rec("J4VG5-X", 11, 7, fired5=10)})
    assert ag.gate_d(bad) == [(11, 7)]


def test_gate_b_voids_above_the_eval_bound():
    """Catches the beam instrument check being reported but never enforced."""
    assert ag.gate_b(_recs())["ok"] is True
    assert ag.gate_b(_recs({("B5-X", 9, 0): _rec("B5-X", 9, 0, judge_evals=1100)}))["ok"] is False


def test_records_are_refused_unless_they_are_the_registered_cell(tmp_path):
    """Catches a smoke record, a wrong width, a partly trained judge or a renamed file entering
    a verdict."""
    p = tmp_path / ev.record_name("B5-X", 9, 0)
    for bad in (_rec("B5-X", 9, 0, limit_states=20), _rec("B5-X", 9, 0, beam_width=32),
                _rec("B5-X", 9, 0, judge_updates=3000), _rec("B5-X", 9, 1)):
        p.write_text(json.dumps(bad))
        with pytest.raises(SystemExit):
            ag._read_cell(p, "B5-X", 9, 0)


def test_load_records_reads_j4v_from_exp076(tmp_path):
    """Catches J4V-X read from this experiment's outputs (re-run data) instead of EXP-076's
    committed records, which Gate 0(b) licenses."""
    for d in ev.DEPTHS:
        for arm in ag.NEW_ARMS:
            for s in ag.SEEDS:
                (tmp_path / ev.record_name(arm, d, s)).write_text(json.dumps(_rec(arm, d, s)))
    recs = ag.load_records(tmp_path)
    ref = json.loads((ev.E76_OUT / "exp076_J4V-X_d11_s0.json").read_text())
    assert recs[("J4V-X", 11)][0]["solved"] == ref["solved"] == 36


def test_report_runs_on_real_shaped_records_and_prints_seconds_per_state():
    """Catches main() raising KeyError on EXP-076's J4V-X records (no judge_evals, no fired5),
    and the full-run seconds-per-state secondary being absent."""
    recs = _recs()
    for key, by_seed in recs.items():
        for r in by_seed.values():
            r.update(mean_steps=7.0, optimality=0.5, wall_s=100.0)
    for by_seed in (recs[("J4V-X", 9)], recs[("J4V-X", 11)]):
        for r in by_seed.values():
            r.pop("judge_evals"), r.pop("fired5")
    for s in ag.SEEDS:
        recs[("B5-X", 11)][s]["wall_s"] = 400.0
    text = "\n".join(ag.report_lines(recs, "PASS", True))
    assert "CLAIM 1 (primary) d11 B5-X - J4V-X" in text
    assert "J4V-X 0.2500 (7.00, 0.500, n/a)" in text
    assert "B5-X 2.000" in text and "J4V-X 0.500" in text
