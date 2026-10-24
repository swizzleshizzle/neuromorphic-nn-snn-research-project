"""EXP-081: EXP-080's O-Xt continued from 4000 to 16,000 updates on an extended mint."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "081_oxt_long_budget"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t80tests = _load("test_exp080_helpers", REPO / "tests" / "experiments" / "test_exp080.py")
mint = _load("exp081_mint", EXP / "mint.py")
train = _load("exp081_train", EXP / "train.py")
gatec = train.gatec
cells = train.cells
m80 = gatec.m80
t80 = train.t80
c80 = cells.c80

SEED = 12
OLD_BLOCKS, NEW_BLOCKS, BLOCK = 2, 4, 8


def test_constants_are_the_spec_values():
    assert (cells.N_BLOCKS, cells.BATCH, cells.MAX_LEN, cells.PROBE_EVERY) == (16000, 1000, 40,
                                                                               250)
    assert cells.START == 4000 and cells.LEGS == (8000, 16000)
    assert cells.N_BLOCKS * cells.BATCH >= cells.LEGS[-1] * cells.BATCH
    assert cells.ckpt_dir(3) == c80.ckpt_dir("O-Xt", 3, cells.OUT)


def test_a_longer_mint_starts_with_the_shorter_one():
    """Catches a draw that is not one ordered stream (a per-block reseed, or n_blocks feeding
    the generator), which would make the continuation train on blocks EXP-080 never saw."""
    fd = t80tests.FakeDistance()
    s2, l2 = m80.draw_blocks(OLD_BLOCKS, BLOCK, 6, SEED, set(), fd.distance)
    s4, l4 = m80.draw_blocks(NEW_BLOCKS, BLOCK, 6, SEED, set(), fd.distance)
    n = OLD_BLOCKS * BLOCK
    assert (s4[:n] == s2).all() and (l4[:n] == l2).all()
    assert not (s4[n:2 * n] == s2).all()


def test_mint_seed_asks_for_16000_blocks_of_1000(monkeypatch):
    seen = {}
    monkeypatch.setattr(mint.m80, "mint_seed", lambda seed, out, **kw: seen.update(kw) or {})
    mint.mint_seed(0, "x")
    assert seen == {"n_blocks": 16000, "block": 1000, "max_len": 40}


def _world(tmp_path, monkeypatch, start=OLD_BLOCKS):
    """EXP-080 in miniature: fake X, a 2-block mint, O-Xt trained `start` updates on it; and
    EXP-081's 4-block mint of the same seed. START is patched to `start`."""
    monkeypatch.setattr(cells, "START", start)
    x = tmp_path / "x"
    t80tests._fake_x(x, SEED)
    e80 = tmp_path / "e80"
    t80tests._fake_mint(e80, SEED, n_blocks=OLD_BLOCKS, block=BLOCK)
    t80.run("O-Xt", SEED, e80, n_updates=start, batch=BLOCK, probe_every=1, mint_root=e80,
            x_root=x, enc_root=x)
    out = tmp_path / "out"
    t80tests._fake_mint(out, SEED, n_blocks=NEW_BLOCKS, block=BLOCK)
    return x, e80, out


def test_gate_c_passes_and_seeds_the_continuation_on_the_new_mint(tmp_path, monkeypatch):
    x, e80, out = _world(tmp_path, monkeypatch)
    rec = gatec.gate_c(SEED, out, e80_root=e80)
    new = m80.read_mint(out, SEED)
    old = m80.read_mint(e80, SEED)
    assert rec["pass"] and rec["continuation"] == "seeded" and rec["prefix_bad"] == []
    assert rec["opt_bad"] == [] and rec["start_gate_l_equal"]
    st = json.loads((cells.ckpt_dir(SEED, out) / "state.json").read_text())
    assert st["mint_sha"] == new.sha and new.sha != old.sha
    assert st["continued_from"] == {"experiment": "EXP-080", "update": OLD_BLOCKS,
                                    "mint_sha": old.sha}
    for f in ("judge.pt", "opt.pt"):
        assert (cells.ckpt_dir(SEED, out) / f).read_bytes() == (
            c80.ckpt_dir("O-Xt", SEED, e80) / f).read_bytes()
    assert gatec.passed(SEED, out) and gatec.pending([SEED], out) == []


@pytest.mark.parametrize("part", ["states", "labels", "probe", "leaves", "length"])
def test_prefix_check_catches_one_changed_element_in_each_part(tmp_path, part):
    """Catches a prefix check that skips an array: any one of them differing means the
    continuation's start is not EXP-080's run."""
    t80tests._fake_mint(tmp_path / "o", SEED, n_blocks=OLD_BLOCKS, block=BLOCK)
    t80tests._fake_mint(tmp_path / "n", SEED, n_blocks=NEW_BLOCKS, block=BLOCK)
    old = m80.read_mint(tmp_path / "o", SEED)
    new = m80.read_mint(tmp_path / "n", SEED)
    assert gatec.check_prefix(new, old) == []
    if part == "states":
        s = new.states.copy()
        s[5, 3] ^= 1
        new = new._replace(states=s)
    elif part == "labels":
        lab = new.labels.copy()
        lab[OLD_BLOCKS * BLOCK - 1] += 1
        new = new._replace(labels=lab)
    elif part == "probe":
        new = new._replace(probe=[(new.probe[0][0], new.probe[0][1] + 1)] + new.probe[1:])
    elif part == "leaves":
        k = next(iter(new.leaf_dist))
        new = new._replace(leaf_dist={**new.leaf_dist, k: new.leaf_dist[k] + 1})
    else:
        new = old
    assert gatec.check_prefix(new, old) == [part]


def test_gate_c_fails_on_a_start_that_is_not_exp080s(tmp_path, monkeypatch):
    """Catches a start check that does not compare the re-measured gate_l with EXP-080's
    committed one: the record is written, says FAIL, and no continuation is seeded."""
    x, e80, out = _world(tmp_path, monkeypatch)
    p = e80 / c80.record_name("O-Xt", SEED)
    r = json.loads(p.read_text())
    r["gate_l"]["margin"] += 1e-9
    p.write_text(json.dumps(r))
    with pytest.raises(SystemExit, match="GATE C FAILED"):
        gatec.gate_c(SEED, out, e80_root=e80)
    rec = json.loads((out / cells.gatec_name(SEED)).read_text())
    assert rec["pass"] is False and rec["start_gate_l_equal"] is False
    assert not (cells.ckpt_dir(SEED, out) / "state.json").exists()
    assert gatec.pending([SEED], out) == [SEED]


def test_gate_c_fails_when_exp080_is_not_at_the_start_update(tmp_path, monkeypatch):
    x, e80, out = _world(tmp_path, monkeypatch)
    monkeypatch.setattr(cells, "START", OLD_BLOCKS + 1)
    with pytest.raises(SystemExit, match="start update 2"):
        gatec.gate_c(SEED, out, e80_root=e80)


def test_opt_check_catches_a_wrong_step_or_shape(tmp_path, monkeypatch):
    """Catches an optimizer file from another update or another network beside the judge."""
    x, e80, out = _world(tmp_path, monkeypatch)
    ck = c80.ckpt_dir("O-Xt", SEED, e80)
    judge = c80.load_trained("O-Xt", SEED, e80, required=OLD_BLOCKS)
    sd = torch.load(ck / "opt.pt")
    assert gatec.check_opt(sd, judge) == []
    assert gatec.check_opt(sd, judge, step=OLD_BLOCKS + 1) != []
    sd["state"][1]["exp_avg"] = torch.zeros(3, 3)
    assert any("param 1 exp_avg shape" in b for b in gatec.check_opt(sd, judge))
    del sd["state"][0]
    assert any("state keys" in b for b in gatec.check_opt(sd, judge))


def test_gate_c_never_overwrites_a_started_continuation(tmp_path, monkeypatch):
    x, e80, out = _world(tmp_path, monkeypatch)
    gatec.gate_c(SEED, out, e80_root=e80)
    sp = cells.ckpt_dir(SEED, out) / "state.json"
    st = json.loads(sp.read_text())
    st["update"] = 3
    sp.write_text(json.dumps(st))
    (out / cells.gatec_name(SEED)).unlink()
    assert gatec.gate_c(SEED, out, e80_root=e80)["continuation"] == "kept"
    assert json.loads(sp.read_text())["update"] == 3
    st["mint_sha"] = "other"
    sp.write_text(json.dumps(st))
    (out / cells.gatec_name(SEED)).unlink()
    with pytest.raises(SystemExit, match="continuation is on mint other"):
        gatec.gate_c(SEED, out, e80_root=e80)


def _continue(tmp_path, monkeypatch, legs=(3, 4)):
    x, e80, out = _world(tmp_path, monkeypatch)
    gatec.gate_c(SEED, out, e80_root=e80)
    recs = train.run(SEED, out, legs=legs, batch=BLOCK, probe_every=1, x_root=x, enc_root=x)
    return x, e80, out, recs


def _straight(tmp_path, x, mint_root, n):
    d = tmp_path / f"straight{n}"
    t80.run("O-Xt", SEED, d, n_updates=n, batch=BLOCK, probe_every=1, mint_root=mint_root,
            x_root=x, enc_root=x)
    return torch.load(c80.ckpt_dir("O-Xt", SEED, d) / "judge.pt")


def _same(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def test_continuation_equals_a_straight_run_on_the_new_mint(tmp_path, monkeypatch):
    """Catches a continuation that is not the longer run it stands for: a restarted optimizer,
    generator or data index, or a leg-1 snapshot taken at the wrong point."""
    x, e80, out, recs = _continue(tmp_path, monkeypatch)
    ck = cells.ckpt_dir(SEED, out)
    assert _same(torch.load(ck / "judge.pt"), _straight(tmp_path, x, out, 4))
    assert _same(torch.load(ck / cells.snapshot_name(3)), _straight(tmp_path, x, out, 3))
    assert [(r["n_updates"], r["resumed_from"]) for r in recs] == [(3, 2), (4, 3)]
    for r in recs:
        assert r["encoder_drift"] > 0.0 and r["mint_meta"]["n_blocks"] == NEW_BLOCKS
        assert r["continued_from"]["experiment"] == "EXP-080"
        assert r["mint_sha"] == m80.read_mint(out, SEED).sha


def test_leg_1_record_survives_a_rerun_of_leg_2(tmp_path, monkeypatch):
    x, e80, out, _ = _continue(tmp_path, monkeypatch)
    p1, p2 = out / cells.record_name(3, SEED), out / cells.record_name(4, SEED)
    before = p1.read_text()
    p2.unlink()
    recs = train.run(SEED, out, legs=(3, 4), batch=BLOCK, probe_every=1, x_root=x, enc_root=x)
    assert p1.read_text() == before and [r["n_updates"] for r in recs] == [4]
    assert recs[0]["resumed_from"] == 4
    # A fresh process after a restart: state.json no longer carries continued_from.
    assert "continued_from" not in json.loads(
        (cells.ckpt_dir(SEED, out) / "state.json").read_text())
    assert recs[0]["continued_from"] == {"experiment": "EXP-080", "update": OLD_BLOCKS,
                                         "mint_sha": m80.read_mint(e80, SEED).sha}


def test_a_checkpoint_past_leg_1_is_never_recorded_as_leg_1(tmp_path, monkeypatch):
    """Catches the kill-in-leg-2 case: leg 1's record missing with the checkpoint at 4 must be
    refused, not trained to 3 (a no-op) and written as the 3-update point."""
    x, e80, out, _ = _continue(tmp_path, monkeypatch)
    (out / cells.record_name(3, SEED)).unlink()
    with pytest.raises(SystemExit, match="past leg end 3"):
        train.run(SEED, out, legs=(3, 4), batch=BLOCK, probe_every=1, x_root=x, enc_root=x)


def test_training_refuses_a_seed_without_a_passing_gate_c(tmp_path, monkeypatch):
    x, e80, out = _world(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="no passing Gate C"):
        train.run(SEED, out, legs=(3, 4), batch=BLOCK, probe_every=1, x_root=x, enc_root=x)


ev = _load("exp081_evaluate", EXP / "evaluate.py")
agg = _load("exp081_aggregate", EXP / "aggregate.py")
FAR, NEAR = (9, 10, 11), (7, 8)


def _by(far_delta, near_delta=0.0):
    return {**{d: far_delta for d in FAR}, **{d: near_delta for d in NEAR}}


def _leg_rec(u, s, gl, **over):
    rec = {"arm": "O-Xt", "seed": s, "gate_l": gl, "n_updates": u, "batch": 1000,
           "probe_every": 250, "mint_sha": f"new{s}",
           "mint_meta": {"n_blocks": 16000, "block": 1000, "max_len": 40},
           "resumed_from": u // 2, "continued_from": {"experiment": "EXP-080"},
           "encoder_drift": 13.0, "loss_last": 2.5, "wall_s": 100.0, "limit_states": None,
           "updates": u, "final_probe": {"spearman_7_11": 0.3, "mean_j_by_distance":
                                         {"9": 9.0, "10": 10.0, "11": 11.0}}}
    rec.update(over)
    return rec


def _agg_world(tmp_path, d4, d16, d8=None, gc_fail=(), **over16):
    """EXP-080's O-Xt records at X's committed gate_l shifted by `d4`; EXP-081's legs at `d8`
    (default `d4`) and `d16`, all per distance; and the 12 Gate C records."""
    e80, out = tmp_path / "e80", tmp_path / "out"
    e80.mkdir(parents=True)
    out.mkdir()
    for s in range(12):
        gl = t80tests._x75(s)["gate_l"]
        rec4 = t80tests._train_rec("O-Xt", s, t80tests._shift_by(gl, d4))
        (e80 / c80.record_name("O-Xt", s)).write_text(json.dumps(rec4))
        for u, d, o in ((8000, d4 if d8 is None else d8, {}), (16000, d16, over16)):
            (out / cells.record_name(u, s)).write_text(
                json.dumps(_leg_rec(u, s, t80tests._shift_by(gl, d), **o)))
        (out / cells.gatec_name(s)).write_text(
            json.dumps({"seed": s, "pass": s not in gc_fail, "new_mint_sha": f"new{s}"}))
    return e80, out


def test_verdict_has_a_line_for_every_row_of_spec_section_8():
    v = agg.verdict
    assert v(False, True, "CONFIRMED", "CONFIRMED", "PAYS") == "VOID"
    assert v(True, False, "VOID", "VOID", None) == "UNRESOLVED (O-Xt16k UNTRAINED)"
    assert v(True, True, "CONFIRMED", "REFUTED", "PAYS") == "BUDGET-LIMITED, PAYS"
    assert v(True, True, "CONFIRMED", "CONFIRMED", "DOES NOT PAY") == (
        "BUDGET-LIMITED, DOES NOT PAY AT DEPTH 11")
    assert v(True, True, "CONFIRMED", "CONFIRMED", "UNRESOLVED") == (
        "BUDGET-LIMITED, SEARCH UNRESOLVED")
    assert v(True, True, "CONFIRMED (SMALL)", "CONFIRMED", None) == "BUDGET HELPS BUT FALLS SHORT"
    assert v(True, True, "REFUTED", "CONFIRMED", None) == "BUDGET HELPS BUT FALLS SHORT"
    assert v(True, True, "CONFIRMED (SMALL)", "REFUTED", None) == "ENCODER-LIMITED AT 4X BUDGET"
    assert v(True, True, "REFUTED", "REFUTED", None) == "ENCODER-LIMITED AT 4X BUDGET"
    with pytest.raises(SystemExit, match="run the search"):
        v(True, True, "CONFIRMED", "REFUTED", None)


def test_bars_are_exp080s():
    assert agg.CLAIM_ALPHA == 0.025 and agg.GATE_T_THRESHOLD == t80tests.agg.GATE_T_THRESHOLD
    assert agg.a80.CLAIM_MIN == 0.05


def test_a_large_gain_confirms_claim_1_and_asks_for_the_search(tmp_path):
    e80, out = _agg_world(tmp_path, _by(0.02), _by(0.10))
    res = agg.stage_gates(out, e80)
    assert res["c1"][0] == "CONFIRMED" and res["c2"][0] == "CONFIRMED" and res["run_search"]
    assert abs(res["c1"][2] - 0.10) < 1e-9 and abs(res["c2"][2] - 0.08) < 1e-9
    assert "SEARCH: RUN" in agg.format_stage_gates(res)[-1]


def test_a_small_gain_over_exp080_reads_budget_helps_but_falls_short(tmp_path):
    e80, out = _agg_world(tmp_path, _by(0.02), _by(0.03))
    res = agg.stage_gates(out, e80)
    assert res["c1"][0] == "CONFIRMED (SMALL)" and res["c2"][0] == "CONFIRMED"
    assert not res["run_search"]
    assert agg.verdict(True, res["t"], res["c1"][0], res["c2"][0], None) == (
        "BUDGET HELPS BUT FALLS SHORT")


def test_claims_read_the_far_band_not_the_near_goal(tmp_path):
    """Catches claims computed on the overall margin: a near-goal gain with the far band
    unchanged since EXP-080 is ENCODER-LIMITED AT 4X BUDGET."""
    e80, out = _agg_world(tmp_path, _by(0.0), _by(0.0, near_delta=0.3))
    res = agg.stage_gates(out, e80)
    assert res["c2"][0] == "REFUTED" and res["c1"][0] == "REFUTED"
    assert agg.verdict(True, res["t"], res["c1"][0], res["c2"][0], None) == (
        "ENCODER-LIMITED AT 4X BUDGET")


def test_untrained_16k_voids_both_claims_and_shows_both_points(tmp_path):
    """Catches a Gate T line that hides whether overall margin fell because far band rose (a
    trade) or because the judge is untrained (spec section 6)."""
    e80, out = _agg_world(tmp_path, _by(0.02), _by(-0.25, near_delta=-0.25))
    res = agg.stage_gates(out, e80)
    assert res["t"] is False and res["c1"][0] == "VOID" and res["c2"][0] == "VOID"
    line = agg.format_stage_gates(res)[1]
    assert re.search(r"GATE T .*FAIL.*u4000 far [-0-9.]+ overall [-0-9.]+  u16000 far "
                     r"[-0-9.]+ overall [-0-9.]+", line), line


def test_a_failed_gate_c_is_void(tmp_path):
    e80, out = _agg_world(tmp_path, _by(0.02), _by(0.10), gc_fail=(4,))
    res = agg.stage_gates(out, e80)
    assert res == {"gate_c": False, "gate_c_fail": [4]}
    assert agg.format_stage_gates(res)[0] == "GATE C (continuity): FAIL on seeds [4]"
    (out / cells.gatec_name(5)).unlink()
    with pytest.raises(SystemExit, match="missing exp081_gatec_s5"):
        agg.stage_gates(out, e80)


@pytest.mark.parametrize("over, msg", [
    ({"resumed_from": 0}, "fresh run"),
    ({"mint_meta": {"n_blocks": 4000, "block": 1000, "max_len": 40}}, "mint n_blocks"),
    ({"mint_sha": "other"}, "Gate C certified"),
    ({"encoder_drift": 0.0}, "did not move"),
    ({"continued_from": None}, "continued_from"),
    ({"n_updates": 12000}, "n_updates"),
    ({"limit_states": 20}, "smoke record"),
])
def test_a_record_that_is_not_the_continuation_is_refused(tmp_path, over, msg):
    e80, out = _agg_world(tmp_path, _by(0.02), _by(0.10), **over)
    with pytest.raises(SystemExit, match=msg):
        agg.stage_gates(out, e80)


def test_search_cell_refuses_a_judge_short_of_16000(tmp_path, monkeypatch):
    x, e80, out = _world(tmp_path, monkeypatch)
    gatec.gate_c(SEED, out, e80_root=e80)
    with pytest.raises(SystemExit, match="needs 16000"):
        ev.run_cell(SEED, out)


def test_chain_reads_the_search_lines_the_aggregator_prints(tmp_path):
    """Catches the chain and the aggregator drifting apart: the chain must find SEARCH: NOT RUN
    on an unconfirmed Claim 1, SEARCH: RUN on a confirmed one, and neither on a VOID."""
    chain = (EXP / "chain081.ps1").read_text()
    assert '-match "SEARCH: NOT RUN"' in chain and '-notmatch "SEARCH: RUN"' in chain
    e80, out = _agg_world(tmp_path, _by(0.02), _by(0.10))
    yes = "\n".join(agg.format_stage_gates(agg.stage_gates(out, e80)))
    assert re.search("SEARCH: RUN", yes) and not re.search("SEARCH: NOT RUN", yes)
    e80b, outb = _agg_world(tmp_path / "b", _by(0.02), _by(0.02))
    no = "\n".join(agg.format_stage_gates(agg.stage_gates(outb, e80b)))
    assert re.search("SEARCH: NOT RUN", no)
