"""EXP-082: EXP-075's X judge continued by value iteration from 4000 to 16,000 updates."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "082_x_long_budget"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp082_cells", EXP / "cells.py")
c75 = cells.c75
t75 = _load("exp075_train", REPO / "experiments" / "075_wide_region" / "train.py")


def test_constants_are_the_spec_values():
    """Catches a setting drifting from EXP-075's X: a continuation on other settings is not the
    same run as a straight 16,000-update X."""
    a = t75.build_parser().parse_args(["--n-updates", "1"])
    assert (cells.BATCH, cells.SYNC_EVERY, cells.PROBE_EVERY, cells.DRAWS) == (
        a.batch, a.sync_every, a.probe_every, a.draws) == (1000, 100, 250, 1)
    assert t75.MAX_LEN == 14 and t75.VI_ARM == "W"
    assert cells.START == c75.N_UPDATES == 4000 and cells.LEGS == (8000, 16000)
    assert cells.DEPTHS == (9, 11) and cells.ARM == "X16k" and cells.SRC_ARM == "X"
    assert list(cells.EVAL_SEEDS) == list(range(12))


def test_names_and_layout():
    """The checkpoint folder is EXP-075's layout under EXP-082's outputs, so EXP-075's
    load_judge reads an X16k judge unchanged; every record name carries the experiment."""
    assert cells.ckpt_dir(3) == c75.ckpt_dir("X", 3, cells.OUT)
    assert cells.ckpt_dir(3).parent == EXP / "outputs"
    assert cells.record_name(8000, 3) == "exp082_X16k_u8000_s3.json"
    assert cells.gatec_name(3) == "exp082_gatec_s3.json"
    assert cells.snapshot_name(8000) == "judge_u8000.pt"
    assert cells.search_name(11, 3) == "exp082_J4V-X16k_d11_s3.json"
    assert cells.E75_OUT != cells.OUT


gatec = _load("exp082_gatec", EXP / "gatec.py")
t75tests = _load("test_exp075_helpers", REPO / "tests" / "experiments" / "test_exp075.py")
vi = gatec.t75.vi

SEED = 12
START, LEGS, BATCH = 2, (3, 4), 4


def _probe():
    return [(apply_move(SOLVED, 0), 1), (apply_move(apply_move(SOLVED, 1), 3), 2)]


def fake_measure(judge, probe, distance, seed):
    """Stands in for Gate L (which needs the full BFS table): a number every weight moves."""
    return {"margin": sum(float(p.detach().double().sum()) for p in judge.parameters())}


def _train_straight(root, n, sync_every=1):
    """EXP-075's X in miniature: fake pretrained encoder, n updates of value iteration."""
    judge = gatec.c75.make_judge(SEED, "X", root)
    vi.train_judge(judge, gatec.t75.VI_ARM, n_updates=n, batch=BATCH, max_len=gatec.t75.MAX_LEN,
                   n_actions=N_ACTIONS, exclude=set(), probe=_probe(), seed=SEED,
                   sync_every=sync_every, probe_every=1, draws=1,
                   ckpt_dir=gatec.c75.ckpt_dir("X", SEED, root), log=lambda m: None)
    return judge


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fake EXP-075 X at update START with its record, and an empty EXP-082 out dir."""
    monkeypatch.setattr(gatec.cells, "START", START)
    monkeypatch.setattr(gatec, "sets", lambda seed: (set(), _probe(), None))
    monkeypatch.setattr(gatec, "measure", fake_measure)
    src = tmp_path / "e75"
    t75tests._fake_encoder(src, "X", SEED)
    judge = _train_straight(src, START)
    (src / gatec.c75.record_name("X", SEED)).write_text(json.dumps(
        {"gate_l": fake_measure(judge, None, None, SEED), "n_probe": 2}))
    return src, tmp_path / "out"


def _gc(src, out, tracked=lambda p: "b", blob=lambda p: "b"):
    return gatec.gate_c(SEED, out, src_root=src, tracked=tracked, blob=blob)


def _src_ck(src):
    return gatec.c75.ckpt_dir("X", SEED, src)


def test_gate_c_passes_and_copies_all_four_files(world):
    """Catches a copy missing the optimizer or the target net (the continuation would crash or,
    worse, restart Adam), and a Gate C record without the provenance training reads."""
    src, out = world
    rec = _gc(src, out)
    assert rec["pass"] is True and rec["continuation"] == "seeded"
    dst = gatec.cells.ckpt_dir(SEED, out)
    for f in ("judge.pt", "jt.pt", "opt.pt", "state.json"):
        assert (dst / f).read_bytes() == (_src_ck(src) / f).read_bytes(), f
    assert rec["continued_from"] == {"experiment": "EXP-075", "update": START}
    assert rec["start_update"] == START and rec["opt_bad"] == [] and rec["jt_ok"] is True
    assert gatec.passed(SEED, out)


@pytest.mark.parametrize("tracked,blob,why", [
    (lambda p: "b", lambda p: "other", "differs"),
    (lambda p: None, lambda p: "b", "untracked"),
])
def test_gate_c_fails_on_a_source_that_is_not_the_tracked_file(world, tracked, blob, why):
    src, out = world
    with pytest.raises(SystemExit, match="GATE C FAILED"):
        _gc(src, out, tracked, blob)
    rec = json.loads((out / gatec.cells.gatec_name(SEED)).read_text())
    assert rec["pass"] is False and any(why in b for b in rec["identity_bad"])
    assert not gatec.cells.ckpt_dir(SEED, out).exists()


def test_gate_c_fails_on_a_start_that_is_not_exp075s(world):
    """Catches a Gate L comparison that is not exact, or that compares the wrong judge."""
    src, out = world
    p = src / gatec.c75.record_name("X", SEED)
    r = json.loads(p.read_text())
    r["gate_l"]["margin"] += 1e-9
    p.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="GATE C FAILED"):
        _gc(src, out)
    assert not gatec.passed(SEED, out)


def test_gate_c_fails_when_exp075_is_not_at_the_start_update(world, monkeypatch):
    src, out = world
    monkeypatch.setattr(gatec.cells, "START", START + 1)
    with pytest.raises(SystemExit, match="start update"):
        _gc(src, out)


def test_gate_c_fails_on_an_optimizer_at_the_wrong_step(world):
    """Catches a stale opt.pt (the laptop's opt.pt predates its judge.pt by two days)."""
    src, out = world
    p = _src_ck(src) / "opt.pt"
    sd = torch.load(p)
    for st in sd["state"].values():
        st["step"] = torch.tensor(float(START - 1))
    torch.save(sd, p)
    with pytest.raises(SystemExit, match="GATE C FAILED"):
        _gc(src, out)
    rec = json.loads((out / gatec.cells.gatec_name(SEED)).read_text())
    assert any("step" in b for b in rec["opt_bad"])


def test_gate_c_fails_on_a_target_net_of_another_architecture(world):
    src, out = world
    other = gatec.c75.build_judge(SEED, "Y", gatec.c75.fresh_sensory("Y", SEED))
    torch.save(other.state_dict(), _src_ck(src) / "jt.pt")
    with pytest.raises(SystemExit, match="GATE C FAILED"):
        _gc(src, out)
    assert json.loads((out / gatec.cells.gatec_name(SEED)).read_text())["jt_ok"] is False


def test_gate_c_never_overwrites_a_started_continuation(world):
    """REVIEW FOCUS 2. A re-run after training began must keep the continuation, or training
    silently restarts from 4000."""
    src, out = world
    _gc(src, out)
    marker = gatec.cells.ckpt_dir(SEED, out) / "judge.pt"
    marker.write_bytes(b"started")
    (out / gatec.cells.gatec_name(SEED)).unlink()
    rec = _gc(src, out)
    assert rec["continuation"] == "kept" and marker.read_bytes() == b"started"


def test_gate_c_never_writes_exp075s_folder(world):
    src, out = world
    before = {f.name: f.read_bytes() for f in _src_ck(src).iterdir()}
    _gc(src, out)
    assert {f.name: f.read_bytes() for f in _src_ck(src).iterdir()} == before


def test_measure_is_exp075s_gate_l_call(monkeypatch):
    """Catches Gate C re-measuring with other arguments than EXP-075's train.run did."""
    seen = {}
    monkeypatch.setattr(gatec.gate_l, "leaf_rank_margin",
                        lambda *a, **k: seen.update(a=a, k=k) or {"m": 1})
    assert gatec.measure("J", "P", "D", 5) == {"m": 1}
    assert seen == {"a": ("J", "P", "D", 5, N_ACTIONS), "k": {}}


def test_identity_reads_git_for_a_tracked_file_and_none_for_an_untracked_one(tmp_path):
    f = gatec.cells.E75_OUT / "judge_X_s0" / "state.json"
    assert len(gatec.blob_of(f)) == 40 and gatec.tracked_blob(f) == gatec.blob_of(f)
    p = tmp_path / "x.json"
    p.write_text("{}")
    assert gatec.tracked_blob(p) is None


def test_pending_rechecks_failed_seeds(tmp_path):
    (tmp_path / gatec.cells.gatec_name(1)).write_text(json.dumps({"pass": True}))
    (tmp_path / gatec.cells.gatec_name(2)).write_text(json.dumps({"pass": False}))
    assert gatec.pending([1, 2, 3], tmp_path) == [2, 3]


train = _load("exp082_train", EXP / "train.py")


def _run(out, src, legs=LEGS):
    return train.run(SEED, out, legs=legs, batch=BATCH, sync_every=1, probe_every=1, src_root=src)


@pytest.fixture
def tworld(world, monkeypatch):
    """`world`, with train's own gatec module patched the same way."""
    monkeypatch.setattr(train.gatec.cells, "START", START)
    monkeypatch.setattr(train.gatec, "sets", lambda seed: (set(), _probe(), None))
    monkeypatch.setattr(train.gatec, "measure", fake_measure)
    src, out = world
    train.gatec.gate_c(SEED, out, src_root=src, tracked=lambda p: "b", blob=lambda p: "b")
    return src, out


def test_continuation_equals_a_straight_run(tworld, tmp_path):
    """The spec's section 2 premise: catches a continuation that restarts the optimizer, skips
    the target net, or trains on other settings. Compared weight for weight."""
    src, out = tworld
    _run(out, src)
    straight = tmp_path / "straight"
    shutil.copytree(src / "encoders", straight / "encoders")
    want = _train_straight(straight, LEGS[-1]).state_dict()
    got = torch.load(train.cells.ckpt_dir(SEED, out) / "judge.pt")
    assert got.keys() == want.keys()
    assert all(torch.equal(got[k], want[k]) for k in want)


def test_leg_records_carry_what_the_aggregator_checks(tworld):
    src, out = tworld
    r3, r4 = _run(out, src)
    assert (r3["n_updates"], r4["n_updates"]) == LEGS
    assert (r3["resumed_from"], r4["resumed_from"]) == (START, LEGS[0])
    assert r4["continued_from"] == {"experiment": "EXP-075", "update": START}
    assert (r4["arm"], r4["hidden"], r4["readout"], r4["max_len"]) == ("X16k", 512, "wide", 14)
    assert (r4["sync_every"], r4["batch"], r4["n_probe"]) == (1, BATCH, 2)
    assert r4["encoder_drift"] > 0.0
    assert len(r4["probes"]) > len(r3["probes"]) > START
    on_disk = json.loads((out / train.cells.record_name(LEGS[1], SEED)).read_text())
    assert on_disk["gate_l"] == json.loads(json.dumps(r4["gate_l"]))
    snap = torch.load(train.cells.ckpt_dir(SEED, out) / train.cells.snapshot_name(LEGS[0]))
    assert any(not torch.equal(snap[k], v) for k, v in
               torch.load(train.cells.ckpt_dir(SEED, out) / "judge.pt").items())


def test_a_rerun_skips_finished_legs_and_resumes_leg_2(tworld):
    """REVIEW FOCUS 1. The 03:31 restart re-runs the chain: leg 1's record must be untouched."""
    src, out = tworld
    _run(out, src)
    p1 = out / train.cells.record_name(LEGS[0], SEED)
    before = p1.read_text()
    (out / train.cells.record_name(LEGS[1], SEED)).unlink()
    recs = _run(out, src)
    assert [r["n_updates"] for r in recs] == [LEGS[1]]
    assert p1.read_text() == before


def test_a_checkpoint_past_leg_1_is_never_recorded_as_leg_1(tworld):
    """REVIEW FOCUS 1. A checkpoint at 16,000 with leg 1's record lost must not become the 8000
    point."""
    src, out = tworld
    _run(out, src)
    (out / train.cells.record_name(LEGS[0], SEED)).unlink()
    with pytest.raises(SystemExit, match=f"past leg end {LEGS[0]}"):
        _run(out, src)


def test_training_refuses_a_seed_without_a_passing_gate_c(world, monkeypatch):
    src, out = world
    with pytest.raises(SystemExit, match="Gate C"):
        _run(out, src)


def test_pending_is_seeds_without_the_last_leg(tmp_path):
    (tmp_path / train.cells.record_name(16000, 1)).write_text("{}")
    (tmp_path / train.cells.record_name(8000, 2)).write_text("{}")
    assert train.pending([1, 2, 3], tmp_path) == [2, 3]


evaluate = _load("exp082_evaluate", EXP / "evaluate.py")


def test_search_cell_is_exp076s_j4v():
    e76 = _load("exp076_evaluate", REPO / "experiments" / "076_deeper_tree" / "evaluate.py")
    assert e76.ARMS["J4V-X"] == (evaluate.K, evaluate.K) == (4, 4)
    assert evaluate.ARM == "J4V-X16k"
    assert evaluate.record_name(9, 2) == cells.search_name(9, 2)


def test_search_cell_refuses_a_judge_short_of_16000(tmp_path):
    """REVIEW FOCUS 3. Catches a search run by hand on a judge still training."""
    t75tests._fake_judge_ckpt(tmp_path, "X", 0, update=8000)
    with pytest.raises(SystemExit, match="evaluation needs 16000"):
        evaluate.run_cell(11, 0, tmp_path, limit_states=1)
    assert not (tmp_path / evaluate.record_name(11, 0)).exists()


def test_search_cell_refuses_an_unregistered_depth(tmp_path):
    with pytest.raises(SystemExit, match="depth 7"):
        evaluate.run_cell(7, 0, tmp_path, limit_states=1)


def test_a_16k_judge_drives_a_real_j4v_cell_at_both_depths(tmp_path):
    """Catches a record missing a field the aggregator checks, on EXP-075's real cells."""
    t75tests._fake_judge_ckpt(tmp_path, "X", 0, update=16000)
    for d in (9, 11):
        r = evaluate.run_cell(d, 0, tmp_path, limit_states=1)
        assert (r["arm"], r["judge"], r["depth"], r["seed"], r["judge_seed"]) == (
            "J4V-X16k", "X16k", d, 0, 0)
        assert (r["k"], r["k_goal"], r["mode"], r["no_revisit"]) == (4, 4, "C", True)
        assert r["judge_updates"] == 16000 and r["limit_states"] == 1 and r["n"] == 1
        assert json.loads((tmp_path / evaluate.record_name(d, 0)).read_text())["depth"] == d


def test_search_pending_runs_depth_11_first_and_skips_finished(tmp_path):
    (tmp_path / evaluate.record_name(11, 1)).write_text("{}")
    assert evaluate.pending([9, 11], [1, 2], tmp_path) == [(11, 2), (9, 1), (9, 2)]


agg = _load("exp082_aggregate", EXP / "aggregate.py")
acells = agg.cells


def _gl(far, overall=0.3):
    by = {str(d): {"hit": 0.2 + (far if d >= 9 else 0.5), "chance": 0.2, "n": 50}
          for d in range(7, 12)}
    return {"by_distance": by, "margin": overall, "chance": 0.2, "hit": 0.2 + overall, "n": 250}


def _final_probe():
    return {"spearman_7_11": 0.3, "mean_j_by_distance": {str(d): float(d) for d in range(7, 12)}}


def _x_rec(seed, far):
    return {"arm": "X", "seed": seed, "n_probe": 483, "gate_l": _gl(far),
            "final_probe": _final_probe(), "loss_last": 0.2}


def _leg(seed, u, far, overall=0.3, **over):
    r = {"arm": "X16k", "seed": seed, "n_updates": u, "batch": 1000, "sync_every": 100,
         "probe_every": 250, "draws": 1, "max_len": 14, "readout": "wide", "hidden": 512,
         "resumed_from": 4000 if u == 8000 else 8000,
         "continued_from": {"experiment": "EXP-075", "update": 4000}, "encoder_drift": 13.0,
         "n_probe": 483, "gate_l": _gl(far, overall), "final_probe": _final_probe(),
         "loss_last": 0.2, "wall_s": 100.0, "git_commit": "abc"}
    r.update(over)
    return r


def _search(seed, depth, rate, arm="J4V-X16k", **over):
    r = {"arm": arm, "judge": "X16k", "depth": depth, "seed": seed, "judge_seed": seed, "k": 4,
         "k_goal": 4, "mode": "C", "no_revisit": True, "judge_updates": 16000,
         "limit_states": None, "success_rate": rate, "optimality": 0.8, "mean_steps": 10.0,
         "goal_fired_frac": 0.1, "n": 200}
    r.update(over)
    return r


def _jit(s):
    """Per-seed noise of mean -0.00033 and mixed sign: a zero gain must not confirm."""
    return ((s * 7) % 5 - 2) * 0.002


def _agg_world(tmp, *, far16=0.12, rate11=0.195, rate9=0.40, overall16=0.30, gc=True,
               **leg_over):
    roots = [tmp / n for n in ("out", "e75", "e76", "e81")]
    out, e75, e76, e81 = roots
    for d in roots:
        d.mkdir(parents=True)
    for s in range(12):
        j, lvl = _jit(s), 0.01 * (s % 4)
        (out / acells.gatec_name(s)).write_text(json.dumps({"seed": s, "pass": gc or s != 3}))
        (e75 / acells.c75.record_name("X", s)).write_text(json.dumps(_x_rec(s, 0.12 + lvl)))
        (out / acells.record_name(8000, s)).write_text(json.dumps(_leg(s, 8000, 0.12 + lvl)))
        (out / acells.record_name(16000, s)).write_text(json.dumps(
            _leg(s, 16000, far16 + lvl + j, overall16, **leg_over)))
        for d, rate, base in ((9, rate9, 0.40), (11, rate11, 0.195)):
            lv = 0.01 * (s % 3)
            (e76 / f"exp076_J4V-X_d{d}_s{s}.json").write_text(json.dumps(
                _search(s, d, base + lv, arm="J4V-X")))
            (out / acells.search_name(d, s)).write_text(json.dumps(_search(s, d, rate + lv + j)))
        (e81 / f"exp081_J4V-O-Xt16k_d11_s{s}.json").write_text(json.dumps(
            _search(s, 11, 0.41, arm="J4V-O-Xt16k")))
    return roots


def test_verdict_has_a_line_for_every_row_of_spec_section_8():
    v = agg.verdict
    assert v(False, True, "CONFIRMED", "CONFIRMED") == "VOID"
    assert v(True, False, "CONFIRMED", "CONFIRMED") == "UNRESOLVED (X16k UNTRAINED)"
    assert v(True, True, "CONFIRMED", "REFUTED") == "BUDGET-LIMITED, PAYS"
    assert v(True, True, "CONFIRMED (SMALL)", "CONFIRMED") == \
        "ORDERING IMPROVES, SEARCH DOES NOT PAY"
    assert v(True, True, "REFUTED", "CONFIRMED (SMALL)") == \
        "VALUE ITERATION SATURATED AT 4X BUDGET"


def test_bars_are_the_spec_values():
    assert agg.CLAIM_ALPHA == 0.025 and agg.C1_MIN == 0.03
    assert agg.a80.CLAIM_MIN == 0.05
    assert agg.GATE_T_THRESHOLD == pytest.approx(0.13564, abs=1e-5)


def test_a_large_search_gain_pays(tmp_path):
    res = agg.run_all(*_agg_world(tmp_path, rate11=0.295))
    assert res["verdict"] == "BUDGET-LIMITED, PAYS"


def test_a_small_search_gain_with_a_far_band_gain_reads_ordering_improves(tmp_path):
    """Catches Claim 1's 0.03 minimum missing: +0.02 on every seed is CONFIRMED (SMALL) and
    falls through to Claim 2."""
    res = agg.run_all(*_agg_world(tmp_path, rate11=0.215, far16=0.22))
    assert any("CLAIM 1" in ln and "CONFIRMED (SMALL)" in ln for ln in res["lines"])
    assert res["verdict"] == "ORDERING IMPROVES, SEARCH DOES NOT PAY"


def test_nothing_moving_reads_saturated(tmp_path):
    res = agg.run_all(*_agg_world(tmp_path))
    assert res["verdict"] == "VALUE ITERATION SATURATED AT 4X BUDGET"


def test_claim_1_reads_depth_11_not_depth_9(tmp_path):
    """Catches the depths swapped: a depth-9 gain alone is reported, never the primary."""
    res = agg.run_all(*_agg_world(tmp_path, rate9=0.60))
    assert res["verdict"] == "VALUE ITERATION SATURATED AT 4X BUDGET"
    assert any(ln.startswith("depth 9") and "+0.19" in ln for ln in res["reported"])


def test_untrained_16k_is_unresolved_whatever_the_search(tmp_path):
    res = agg.run_all(*_agg_world(tmp_path, rate11=0.295, overall16=0.10))
    assert res["verdict"] == "UNRESOLVED (X16k UNTRAINED)"


def test_a_failed_gate_c_is_void(tmp_path):
    res = agg.run_all(*_agg_world(tmp_path, rate11=0.295, gc=False))
    assert res["verdict"] == "VOID"
    assert "GATE C (continuity): FAIL on seeds [3]" in res["lines"]


@pytest.mark.parametrize("over,msg", [
    ({"resumed_from": 0}, "fresh run"),
    ({"continued_from": {"experiment": "EXP-080", "update": 4000}}, "continued_from"),
    ({"encoder_drift": 0.0}, "did not move"),
    ({"n_probe": 250}, "n_probe"),
    ({"sync_every": 1}, "sync_every"),
    ({"limit_states": 2}, "smoke record"),
])
def test_a_record_that_is_not_the_continuation_is_refused(tmp_path, over, msg):
    """REVIEW FOCUS 4."""
    with pytest.raises(SystemExit, match=msg):
        agg.run_all(*_agg_world(tmp_path, **over))


def test_mixed_checkouts_are_refused(tmp_path):
    """REVIEW FOCUS 4. Spec section 6: one git_commit across the 16,000-update records."""
    roots = _agg_world(tmp_path)
    p = roots[0] / acells.record_name(16000, 5)
    r = json.loads(p.read_text())
    r["git_commit"] = "def"
    p.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="checkouts"):
        agg.run_all(*roots)


@pytest.mark.parametrize("over,msg", [
    ({"judge_updates": 8000}, "judge_updates"),
    ({"limit_states": 1}, "smoke record"),
    ({"k": 3}, "k is 3"),
])
def test_a_search_record_that_is_not_the_registered_cell_is_refused(tmp_path, over, msg):
    """REVIEW FOCUS 4."""
    roots = _agg_world(tmp_path)
    p = roots[0] / acells.search_name(11, 4)
    p.write_text(json.dumps(_search(4, 11, 0.2, **over)))
    with pytest.raises(SystemExit, match=msg):
        agg.run_all(*roots)


def test_gates_stage_prints_exactly_one_search_line(tmp_path):
    ok = agg.format_stage_gates(agg.stage_gates(*_agg_world(tmp_path / "a")[:2]))
    bad = agg.format_stage_gates(agg.stage_gates(*_agg_world(tmp_path / "b", gc=False)[:2]))
    assert sum(ln.startswith("SEARCH: RUN") for ln in ok) == 1
    assert not any(ln.startswith("SEARCH: NOT RUN") for ln in ok)
    assert sum(ln.startswith("SEARCH: NOT RUN") for ln in bad) == 1


def test_chain_reads_the_search_lines_the_aggregator_prints(tmp_path):
    """REVIEW FOCUS 5. Catches the chain and the aggregator drifting apart, and a chain that
    reads a crashed gates stage as 'search not run'."""
    chain = (EXP / "chain082.ps1").read_text()
    assert '-match "SEARCH: NOT RUN"' in chain and '-notmatch "SEARCH: RUN"' in chain
    assert "STOP: the gates stage exited" in chain
    yes = "\n".join(agg.format_stage_gates(agg.stage_gates(*_agg_world(tmp_path / "a")[:2])))
    assert re.search("SEARCH: RUN", yes) and not re.search("SEARCH: NOT RUN", yes)
    no = "\n".join(agg.format_stage_gates(
        agg.stage_gates(*_agg_world(tmp_path / "b", gc=False)[:2])))
    assert re.search("SEARCH: NOT RUN", no)


def test_drivers_use_the_spec_worker_counts():
    """REVIEW FOCUS 5. Spec sections 5 and 10: Gate C 4, train 4 (X commits 3.6 GB per worker
    and throughput does not rise above 4), search 12."""
    spec = '@{ "gatec" = 4; "train" = 4; "search" = 12 }'
    assert spec in (EXP / "chain082.ps1").read_text()
    assert spec in (EXP / "launch082.ps1").read_text()


def test_drivers_have_no_em_dashes_or_backticks_in_strings():
    for f in ("chain082.ps1", "launch082.ps1"):
        text = (EXP / f).read_text()
        assert "—" not in text, f
        assert not re.search(r'"[^"\n]*`[^"\n]*"', text), f


def test_a_gate_c_failure_reaches_the_verdict_file_as_void(tmp_path):
    """Final review, Important 1. gatec.py exits 1 on a failed seed, which stops the chain
    before the gates stage; without a VOID path the verdict file is never written and the
    watcher re-runs a dead experiment forever. The chain must aggregate on that failure and
    write the verdict when the aggregator prints VOID."""
    import subprocess
    import sys
    chain = (EXP / "chain082.ps1").read_text()
    assert '-match "VERDICT: VOID"' in chain
    out = _agg_world(tmp_path, gc=False)[0]
    r = subprocess.run([sys.executable, str(EXP / "aggregate.py"), "--out-dir", str(out)],
                       capture_output=True, text=True, timeout=120,
                       env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0 and re.search("VERDICT: VOID", r.stdout), r.stderr
