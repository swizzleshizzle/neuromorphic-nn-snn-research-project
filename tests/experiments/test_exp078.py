"""EXP-078: driver, aggregator and launcher. Each docstring names the bug it catches."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = REPO / "experiments" / "078_full_five_move_tree"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev = _load("exp078_evaluate", HERE / "evaluate.py")


def test_arms_and_constants_are_the_spec():
    """Catches a drifted arm table, interval or forced-stop point (spec sections 3, 4, 7)."""
    assert ev.ARMS == {"J5V-X": ("C", 5, 5, None), "J4V-X": ("C", 4, 4, None)}
    assert ev.DEPTHS == (9, 11) and ev.CKPT_EVERY == 10 and ev.FORCED_STOP == 50
    assert ev.RANK_K == 5


def test_j4v_x_with_checkpointing_equals_exp076_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH. Catches checkpointing changing an outcome: J4V-X through this
    driver (checkpoint on) equals EXP-076's driver on the first 3 depth-9 states of seed 0."""
    e76 = _load("exp076_evaluate_t", REPO / "experiments" / "076_deeper_tree" / "evaluate.py")
    mine = ev.run_cell("J4V-X", 9, 0, tmp_path / "a", limit_states=3)
    ref = e76.run_cell("J4V-X", 9, 0, tmp_path / "b", limit_states=3)
    for f in ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
              "goal_fired_frac", "first_fire_len", "fallback_frac"):
        assert mine[f] == ref[f], f
    assert mine["resumes"] == 0


def test_stopped_cell_writes_no_record_and_resumes_to_the_same_record(tmp_path):
    """Catches a stopped cell leaving a record a launcher would skip as done, a deleted
    checkpoint before the record exists, and a resume that differs from a straight run."""
    straight = ev.run_cell("J4V-X", 9, 0, tmp_path / "s", limit_states=3)
    out = tmp_path / "r"
    assert ev.run_cell("J4V-X", 9, 0, out, limit_states=3, stop_after=2) is None
    assert not (out / ev.record_name("J4V-X", 9, 0)).exists()
    assert ev.ckpt_path(out, "J4V-X", 9, 0).exists()
    resumed = ev.run_cell("J4V-X", 9, 0, out, limit_states=3)
    assert (out / ev.record_name("J4V-X", 9, 0)).exists()
    assert not ev.ckpt_path(out, "J4V-X", 9, 0).exists()
    for f in ("solved", "success_rate", "mean_steps", "first_fire_len", "judge_evals"):
        assert resumed[f] == straight[f], f
    assert resumed["resumes"] == 1


def test_j5v_x_records_its_depths_and_fired5(tmp_path):
    """Catches a J5V-X record that does not say what ran (k 5, goal check 5, 7776 leaves)."""
    r = ev.run_cell("J5V-X", 9, 0, tmp_path, limit_states=1)
    assert (r["mode"], r["k"], r["k_goal"]) == ("C", 5, 5)
    assert r["judge_evals"] == 7776 and r["fired5"] == r["first_fire_len"].get("5", 0)


def test_unknown_arm_exits_cleanly(tmp_path):
    """Catches an arm typo running something unregistered."""
    with pytest.raises(SystemExit):
        ev.run_cell("J6V-X", 9, 0, tmp_path, limit_states=1)


def test_smoke_record_is_never_overwritten_or_treated_as_done(tmp_path):
    """Catches a --limit-states smoke record sitting under the real cell's name: run_cell must
    refuse a call with a different limit_states instead of overwriting or skipping it."""
    (tmp_path / ev.record_name("J4V-X", 9, 0)).write_text('{"limit_states": 3}')
    with pytest.raises(SystemExit):
        ev.run_cell("J4V-X", 9, 0, tmp_path, limit_states=None)


def test_stale_checkpoint_of_a_different_size_is_refused(tmp_path):
    """Catches a smoke checkpoint being resumed by a full run (or silently discarded)."""
    assert ev.run_cell("J4V-X", 9, 0, tmp_path, limit_states=2, stop_after=1) is None
    with pytest.raises(ValueError, match="n_states"):
        ev.run_cell("J4V-X", 9, 0, tmp_path, limit_states=3)
    assert ev.ckpt_path(tmp_path, "J4V-X", 9, 0).exists()


ag = _load("exp078_aggregate", HERE / "aggregate.py")


def _rates(**over):
    rates = {(a, d): {s: 0.1 for s in range(12)}
             for a in ("J5V-X", "J4V-X", "J4VG5-X", "B5-X") for d in ev.DEPTHS}
    for key, val in over.items():
        arm, d = key.rsplit("_d", 1)
        rates[(arm.replace("_", "-"), int(d))] = {s: val + 0.001 * s for s in range(12)}
    return rates


def test_alpha_is_four_way():
    """Catches the three-claim alpha carried over from EXP-077."""
    assert ag.ALPHA == pytest.approx(0.0125) and ag.SMALL == 0.01


def test_four_claims_compare_the_registered_pairs():
    """Catches a claim wired to the wrong arm or depth (spec section 8)."""
    v = ag.primary_verdicts(_rates(J5V_X_d11=0.3, J5V_X_d9=0.5), True, True, range(12))
    assert [v[c][0] for c in ("claim1", "claim2", "claim3", "claim4")] == ["CONFIRMED"] * 4
    v = ag.primary_verdicts(_rates(J5V_X_d11=0.3, B5_X_d11=0.3), True, True, range(12))
    assert v["claim4"][0] == "REFUTED" and v["claim1"][0] == "CONFIRMED"


def test_exp077_void_voids_only_claims_3_and_4():
    """Catches EXP-077's validity being ignored, or voiding claims that do not read its arms."""
    v = ag.primary_verdicts(_rates(J5V_X_d11=0.3, J5V_X_d9=0.5), True, False, range(12))
    assert v["claim1"][0] == "CONFIRMED" and v["claim2"][0] == "CONFIRMED"
    assert v["claim3"][0] == "VOID" and v["claim4"][0] == "VOID"


def test_gate0a_ignores_only_the_session_fields(tmp_path):
    """Catches Gate 0(a) failing on a resumed det cell (resumes, elapsed_s differ by design) or
    passing a real difference."""
    rec = {"solved": 80, "wall_s": 1.0, "git_commit": "a", "resumes": 0, "elapsed_s": 1.0}
    (tmp_path / "o").mkdir(); (tmp_path / "d").mkdir()
    name = ev.record_name("J5V-X", 9, 0)
    (tmp_path / "o" / name).write_text(json.dumps(rec))
    (tmp_path / "d" / name).write_text(json.dumps({**rec, "wall_s": 9.0, "git_commit": "b",
                                                   "resumes": 1, "elapsed_s": 9.0}))
    assert ag.gate0a(tmp_path / "o", tmp_path / "d") == "PASS"
    (tmp_path / "d" / name).write_text(json.dumps({**rec, "solved": 79}))
    assert ag.gate0a(tmp_path / "o", tmp_path / "d") == "FAIL"


def test_gate_d_is_the_first_fire_clause():
    """Catches the first-fire instrument check missing for J5V-X."""
    recs = {("J5V-X", d): {s: {"first_fire_len": {"5": 3}} for s in range(12)} for d in ev.DEPTHS}
    assert ag.gate_d(recs) == []
    recs[("J5V-X", 11)][4] = {"first_fire_len": {"4": 1, "5": 2}}
    assert ag.gate_d(recs) == [(11, 4)]


def test_exp077_valid_reports_why_when_its_run_is_missing(tmp_path, monkeypatch):
    """Catches Claims 3 and 4 reading EXP-077 arms that never finished: with EXP-077's outputs
    pointed at an empty directory, exp077_valid is (False, reason)."""
    monkeypatch.setattr(ag.ev, "E77_OUT", tmp_path)
    ok, why = ag.exp077_valid()
    assert ok is False and why


def _synthetic_j5(tmp_path):
    """J5V-X records beside the REAL EXP-076 J4V-X ones: 0.05 above each seed's J4V-X rate."""
    for d in ev.DEPTHS:
        for s in range(12):
            base = json.loads((ev.E76_OUT / f"exp076_J4V-X_d{d}_s{s}.json").read_text())
            rec = {"arm": "J5V-X", "depth": d, "seed": s, "k": 5, "k_goal": 5, "mode": "C",
                   "no_revisit": True, "beam_width": None, "judge_updates": 4000,
                   "limit_states": None, "success_rate": base["success_rate"] + 0.05,
                   "mean_steps": 5.0, "optimality": 0.5, "judge_evals": 100, "wall_s": 10.0,
                   "n": 200, "first_fire_len": {"5": 1}}
            (tmp_path / ev.record_name("J5V-X", d, s)).write_text(json.dumps(rec))


def test_main_path_voids_claims_3_and_4_but_reports_1_and_2_when_exp077_is_missing(
        tmp_path, monkeypatch):
    """Catches the aggregator exiting on EXP-077's missing records before Claims 1 and 2, which
    stand when EXP-077 is VOID, are printed."""
    empty = tmp_path / "e77"
    empty.mkdir()
    monkeypatch.setattr(ag.ev, "E77_OUT", empty)
    _synthetic_j5(tmp_path)
    e77 = ag.exp077_valid()
    assert e77[0] is False
    recs = ag.load_records(tmp_path, include_e77=e77[0])
    lines = ag.report_lines(recs, "PASS", "PASS", e77, tmp_path)
    claim = {n: next(ln for ln in lines if ln.startswith(label))
             for n, label in ag.CLAIM_LABELS}
    for c in ("claim1", "claim2"):
        assert "VOID" not in claim[c] and "->" in claim[c]
    for c in ("claim3", "claim4"):
        assert "VOID" in claim[c] and "EXP-077" in claim[c]
    assert any("d9 J5V-X - B5-X: EXP-077 invalid" in ln for ln in lines)


def test_gate0a_fail_prints_fail_and_voids_every_claim(tmp_path, monkeypatch):
    """Catches a Gate 0(a) FAIL printing NOT CHECKED, or a FAIL not voiding the claims."""
    monkeypatch.setattr(ag.ev, "E77_OUT", tmp_path / "none")
    _synthetic_j5(tmp_path)
    lines = ag.report_lines(ag.load_records(tmp_path, include_e77=False), "PASS", "FAIL",
                            (False, "x"), tmp_path)
    assert "GATE 0(a) determinism: FAIL" in lines
    claim1 = next(ln for ln in lines if ln.startswith(ag.CLAIM_LABELS[0][1]))
    assert claim1.endswith("-> VOID")
