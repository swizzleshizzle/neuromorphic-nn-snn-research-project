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


evaluate = _load("exp079_evaluate", EXP / "evaluate.py")
e76 = _load("exp076_evaluate_for_079_tests", E76 / "evaluate.py")
OUTCOME = ("solved", "n", "success_rate", "mean_steps", "optimality", "eval_revisit_rate",
           "greedy_modal_action_frac", "goal_fired_frac", "fallback_frac", "first_fire_len",
           "judge_evals")


def _fake_s_ckpt(jdir, seed, update=4000):
    """An untrained S-shaped judge saved as a finished checkpoint, with its head moved so it
    differs from X's committed judge."""
    ck = cells.ckpt_dir(seed, jdir)
    ck.mkdir(parents=True)
    judge = cells.c75.build_judge(seed, "X", cells.c75.fresh_sensory("X", seed))
    with torch.no_grad():
        judge.head[0].weight.mul_(-1.0)
    torch.save(judge.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": update}))
    return judge


def test_arms_and_constants_are_the_spec():
    """Catches an arm list, tree depth or depth list drifting from spec section 3."""
    assert evaluate.ARMS == ("J4V-S", "J4V-X") and evaluate.K == 4
    assert evaluate.DEPTHS == (9, 11)
    assert evaluate.record_name("J4V-S", 11, 3) == "exp079_J4V-S_d11_s3.json"
    assert evaluate.rank_record_name(9, 3) == "exp079_rank_J-S4_d9_s3.json"


def test_j4v_x_here_equals_exp076s_harness_on_a_slice(tmp_path):
    """THE GATE 0(b) CODE PATH. Catches this harness drifting from EXP-076's: J4V-X through it
    equals EXP-076's own run_cell on the same states, committed judge, in every outcome field."""
    mine = evaluate.run_cell("J4V-X", 9, 0, tmp_path / "a", limit_states=2)
    ref = e76.run_cell("J4V-X", 9, 0, tmp_path / "b", limit_states=2)
    for f in OUTCOME:
        assert mine[f] == ref[f], f
    assert mine["k"] == 4 and mine["k_goal"] == 4 and mine["judge"] == "X"


def test_each_arm_loads_its_own_judge(tmp_path):
    """Catches J4V-S silently running X's judge (the arms would then be one arm measured twice)
    and J4V-X reading anything but EXP-075's committed judge."""
    saved = _fake_s_ckpt(tmp_path, 0)
    s = evaluate.load_judge("J4V-S", 0, tmp_path)
    assert torch.equal(s.head[0].weight, saved.head[0].weight)
    x = evaluate.load_judge("J4V-X", 0)
    ref = torch.load(cells.E75_OUT / "judge_X_s0" / "judge.pt", map_location="cpu")
    assert torch.equal(x.head[0].weight, ref["head.0.weight"])
    assert not torch.equal(s.head[0].weight, x.head[0].weight)


def test_j4v_s_cell_writes_its_record(tmp_path):
    """Catches an S record that does not say which judge and tree it ran, so the aggregator
    could not refuse a mislabelled one."""
    _fake_s_ckpt(tmp_path / "j", 0)
    rec = evaluate.run_cell("J4V-S", 7, 0, tmp_path / "o", limit_states=1,
                            judge_dir=tmp_path / "j")
    assert rec["arm"] == "J4V-S" and rec["judge"] == "S"
    assert rec["k"] == 4 and rec["k_goal"] == 4 and rec["mode"] == "C"
    assert rec["no_revisit"] is True and rec["judge_updates"] == 4000
    assert rec["limit_states"] == 1 and rec["fired4"] == rec["first_fire_len"].get("4", 0)
    on_disk = json.loads((tmp_path / "o" / "exp079_J4V-S_d7_s0.json").read_text())
    assert on_disk["judge"] == "S"


def test_partly_trained_s_judge_is_refused_by_evaluation(tmp_path):
    """Catches a cell evaluated from an S judge that stopped short of 4000 updates."""
    _fake_s_ckpt(tmp_path / "j", 0, update=3750)
    with pytest.raises(SystemExit, match="3750"):
        evaluate.run_cell("J4V-S", 7, 0, tmp_path / "o", limit_states=1,
                          judge_dir=tmp_path / "j")


def test_unknown_arm_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="unknown arm"):
        evaluate.run_cell("J3V-S", 7, 0, tmp_path)


def test_rank_cell_ranks_four_move_leaves_with_s(tmp_path, monkeypatch):
    """Catches Gate R measured on 3-move leaves, with X's judge, or without its limit."""
    _fake_s_ckpt(tmp_path / "j", 0)
    seen, judges = [], []
    real_levels, real_load = evaluate.tree_levels, evaluate.load_judge

    def spy_levels(state, k, n):
        seen.append(k)
        return real_levels(state, k, n)

    def spy_load(arm, seed, judge_dir=None):
        judges.append(arm)
        return real_load(arm, seed, judge_dir)

    monkeypatch.setattr(evaluate, "tree_levels", spy_levels)
    monkeypatch.setattr(evaluate, "load_judge", spy_load)
    rec = evaluate.rank_cell(7, 0, tmp_path / "o", judge_dir=tmp_path / "j", limit_states=2,
                             provider=ExactBFSDistance(max_depth=11))
    assert set(seen) == {4} and judges == ["J4V-S"]
    assert rec["kind"] == "J-S4" and rec["k"] == 4 and rec["n"] == 2
    assert rec["limit_states"] == 2 and rec["judge_updates"] == 4000
    assert 0.0 < rec["chance"] < 1.0 and rec["hit"] in (0.0, 0.5, 1.0)
    assert (tmp_path / "o" / "exp079_rank_J-S4_d7_s0.json").exists()


@pytest.mark.slow
def test_gatel_x_cell_reproduces_the_committed_gate_l(tmp_path):
    """GATE M CONTINUITY. Catches today's Gate L code (tree_levels, the probe set, the judge
    load) measuring X differently from EXP-075, which would make Gate M compare S with a number
    the current code cannot produce. Measured 2026-10-19 on the VPS: equal, about 100 s. Slow:
    builds the full BFS table."""
    rec = evaluate.gatel_x_cell(0, tmp_path)
    ref = json.loads((cells.E75_OUT / "exp075_train_X_s0.json").read_text())["gate_l"]
    assert rec["gate_l"] == ref
    assert rec["kind"] == "gatel-X" and rec["judge_updates"] == 4000
    assert (tmp_path / "exp079_gatel_X_s0.json").exists()


agg = _load("exp079_aggregate", EXP / "aggregate.py")


def _x_train(seed, pilot=False):
    d = cells.E75_PILOT if pilot else cells.E75_OUT
    return json.loads((d / f"exp075_train_X_s{seed}.json").read_text())


def _as_s(rec, shift=0.0, dists=("9", "10", "11")):
    """An S training record built from a real X record: same probe and per-distance n, hits at
    `dists` moved by `shift`, overall margin recomputed, S's walk evidence added."""
    r = copy.deepcopy(rec)
    bd = r["gate_l"]["by_distance"]
    for d in dists:
        bd[d]["hit"] += shift
    n = sum(v["n"] for v in bd.values())
    hit = sum(v["hit"] * v["n"] for v in bd.values()) / n
    chance = sum(v["chance"] * v["n"] for v in bd.values()) / n
    r["gate_l"].update(hit=hit, chance=chance, margin=hit - chance)
    r.update(arm="S", max_len=40, walk_max_lens=[40], train_hist_updates=250,
             train_hist={"10": 0.5, "11": 0.5})
    return r


def _write_s_train(out, shift=0.0, seeds=range(12), pilot=False, **over):
    out.mkdir(parents=True, exist_ok=True)
    for s in seeds:
        r = _as_s(_x_train(s, pilot), shift)
        r.update(over)
        (out / cells.record_name(s)).write_text(json.dumps(r))


def _ev(arm, depth, seed, solved, **over):
    r = {"arm": arm, "depth": depth, "seed": seed, "solved": solved, "n": 200,
         "success_rate": solved / 200, "k": 4, "k_goal": 4, "mode": "C", "no_revisit": True,
         "judge_updates": 4000, "limit_states": None, "mean_steps": 10.0, "optimality": 0.9}
    if arm == "J4V-S":
        r["judge"] = "S"
    r.update(over)
    return r


def test_constants_are_the_spec():
    """Catches alpha, the stop bar, the Gate M band or the training settings drifting from
    spec sections 6 to 8, and an amendment committed before the pilot."""
    assert agg.ALPHA == 0.025 and agg.SMALL == 0.01 and agg.GATE_M_ALPHA == 0.05
    assert agg.X_PILOT_GATE_L == 0.12983333333333333 and agg.M_BAND == (9, 11)
    assert agg.TRAIN_SETTINGS == {"n_updates": 4000, "batch": 1000, "sync_every": 100,
                                  "probe_every": 250, "draws": 1, "max_len": 40}
    assert agg.DET_IGNORED == ("wall_s", "git_commit")
    assert agg.AMENDMENT is None and agg.GATE_L_THRESHOLD is None


def test_claim_verdicts_use_alpha_0025():
    """Catches a wrong alpha in either direction: p 0.0193 must confirm (it would not at
    0.05 / 3) and p 0.0327 must not (it would at 0.05)."""
    v, p = agg.claim_verdict([0.05] * 10 + [-0.05] * 2, 0.3, 0.2, True)
    assert v == "CONFIRMED" and p == pytest.approx(0.019287109375)
    v, p = agg.claim_verdict([0.02] * 9 + [-0.02] * 2 + [0.0], 0.3, 0.2, True)
    assert v == "NOT SIGNIFICANT" and p == pytest.approx(0.03271484375)
    assert agg.claim_verdict([0.005] * 12, 0.3, 0.2, True)[0] == "CONFIRMED (SMALL)"
    assert agg.claim_verdict([-0.01] * 12, 0.3, 0.2, True)[0] == "REFUTED"
    assert agg.claim_verdict([0.05] * 12, 0.3, 0.2, False)[0] == "VOID"
    assert agg.claim_verdict([0.005] * 12, 0.01, 0.005, True)[0] == "UNRESOLVED"


def test_margin_band_pools_over_distances_9_to_11():
    """Catches Gate M read over all probe distances, or averaged per distance instead of
    pooled by n."""
    gl = {"by_distance": {
        "7": {"hit": 0.9, "chance": 0.1, "n": 50},
        "9": {"hit": 0.5, "chance": 0.2, "n": 50},
        "10": {"hit": 0.4, "chance": 0.3, "n": 25},
        "11": {"hit": 0.7, "chance": 0.5, "n": 25}}}
    want = (0.5 * 50 + 0.4 * 25 + 0.7 * 25) / 100 - (0.2 * 50 + 0.3 * 25 + 0.5 * 25) / 100
    assert agg.margin_band(gl) == pytest.approx(want)


def test_gate_m_is_s_minus_x_by_sign_flip_and_refuses_a_different_probe():
    """Catches Gate M computed as S alone, at alpha other than 0.05, or across probes that
    differ (per-distance n must match X's, the cheap proof of the same probe)."""
    x = {s: _x_train(s) for s in range(12)}
    shifts = [0.02] * 9 + [-0.02] * 2 + [0.0]
    s_recs = {s: _as_s(x[s], shifts[s]) for s in range(12)}
    m = agg.gate_m(s_recs, x)
    assert m["p"] == pytest.approx(0.03271484375) and m["above"] is True
    assert m["mean"] == pytest.approx(sum(shifts) / 12)
    worse = {s: _as_s(x[s], v) for s, v in enumerate([0.05] * 9 + [-0.05] * 3)}
    assert agg.gate_m(worse, x)["above"] is False
    s_recs[4]["gate_l"]["by_distance"]["10"]["n"] += 1
    with pytest.raises(SystemExit, match="seed 4"):
        agg.gate_m(s_recs, x)


def test_train_record_must_prove_its_walk_length():
    """REVIEW FOCUS 2. Catches an S record trained on 14-move walks, one whose observed walk
    lengths disagree with its stated max_len, and one that observed no walk at all (a relaunch
    after the record write was lost), which cannot prove its walk length."""
    good = _as_s(_x_train(0))
    agg.check_train_record(good, "r.json")
    for over, msg in (({"max_len": 14}, "max_len"), ({"walk_max_lens": [14]}, "walk"),
                      ({"walk_max_lens": [14, 40]}, "walk"), ({"walk_max_lens": []}, "walk"),
                      ({"arm": "X"}, "arm"), ({"readout": "concept"}, "readout"),
                      ({"batch": 500}, "batch"), ({"hidden": 128}, "hidden")):
        bad = {**good, **over}
        with pytest.raises(SystemExit, match=msg):
            agg.check_train_record(bad, "r.json")


def test_pilot_amendment_states_the_stop_rule_and_the_threshold(tmp_path):
    """Catches the stop rule read against anything but X's 0.1298, and Gate L's threshold not
    half S's mean pilot margin (spec sections 6 and 7)."""
    _write_s_train(tmp_path, 0.0, seeds=(12, 13), pilot=True)
    a = agg.pilot_amendment(tmp_path)
    xs = [_x_train(s, pilot=True)["gate_l"]["margin"] for s in (12, 13)]
    assert a["mean_margin"] == pytest.approx(sum(xs) / 2)
    assert a["stop"] is False
    assert a["gate_l_threshold"] == pytest.approx(a["mean_margin"] / 2)
    assert set(a["by_distance_margin"][12]) == {"S", "X"}
    low = tmp_path / "low"
    low.mkdir()
    for s in (12, 13):
        r = _as_s(_x_train(s, pilot=True), -0.2, dists=("7", "8", "9", "10", "11"))
        (low / cells.record_name(s)).write_text(json.dumps(r))
    b = agg.pilot_amendment(low)
    assert b["mean_margin"] < agg.X_PILOT_GATE_L and b["stop"] is True
    # Between the bar and half of it: only the full X bar stops this one.
    mid = tmp_path / "mid"
    mid.mkdir()
    x_mean = sum(xs) / 2
    for s in (12, 13):
        r = _as_s(_x_train(s, pilot=True), 0.10 - x_mean, dists=("7", "8", "9", "10", "11"))
        (mid / cells.record_name(s)).write_text(json.dumps(r))
    c = agg.pilot_amendment(mid)
    assert agg.X_PILOT_GATE_L / 2 < c["mean_margin"] < agg.X_PILOT_GATE_L
    assert c["mean_margin"] == pytest.approx(0.10) and c["stop"] is True


def _write_gatel_x(out, **gl_over):
    ref = _x_train(0)["gate_l"]
    (out / evaluate.gatel_x_record_name(0)).write_text(
        json.dumps({"kind": "gatel-X", "seed": 0, "gate_l": {**ref, **gl_over}}))


def test_gate_m_continuity_compares_with_the_committed_record(tmp_path):
    """Catches Gate M read against X's published margins without checking they reproduce, and
    a continuity check blind to a changed per-distance hit."""
    with pytest.raises(SystemExit, match="exp079_gatel_X_s0"):
        agg.gate_m_continuity(tmp_path)
    _write_gatel_x(tmp_path)
    assert agg.gate_m_continuity(tmp_path) == "PASS"
    bd = copy.deepcopy(_x_train(0)["gate_l"]["by_distance"])
    bd["10"]["hit"] += 0.02
    _write_gatel_x(tmp_path, by_distance=bd)
    assert agg.gate_m_continuity(tmp_path) == "FAIL"


def test_stage_train_needs_the_amendment_and_voids_on_l_or_e(tmp_path, monkeypatch):
    """Catches Gates L and E read before the amendment, Gate E not checked on every seed, and
    Gate L passing with a mean below its threshold."""
    _write_s_train(tmp_path, 0.0)
    _write_gatel_x(tmp_path)
    with pytest.raises(SystemExit, match="amendment"):
        agg.stage_train(tmp_path)
    monkeypatch.setattr(agg, "AMENDMENT", {"date": "2026-10-20"})
    monkeypatch.setattr(agg, "GATE_L_THRESHOLD", 0.1)
    res = agg.stage_train(tmp_path)
    assert res["l"]["pass"] is True and res["e"] is True and res["m"]["above"] is False
    assert res["m_cont"] == "PASS"
    monkeypatch.setattr(agg, "GATE_L_THRESHOLD", 0.9)
    assert agg.stage_train(tmp_path)["l"]["pass"] is False
    monkeypatch.setattr(agg, "GATE_L_THRESHOLD", 0.1)
    r = json.loads((tmp_path / cells.record_name(5)).read_text())
    r["encoder_drift"] = 0.0
    (tmp_path / cells.record_name(5)).write_text(json.dumps(r))
    assert agg.stage_train(tmp_path)["e"] is False


def test_load_records_reads_x_from_exp076_and_s_from_out(tmp_path):
    """REVIEW FOCUS 4. Catches J4V-X read from anywhere but EXP-076's committed records, which
    carry no judge field, and a J4V-S record labelled with another judge."""
    for d in (9, 11):
        for s in range(12):
            (tmp_path / evaluate.record_name("J4V-S", d, s)).write_text(
                json.dumps(_ev("J4V-S", d, s, 50)))
    recs = agg.load_records(tmp_path)
    ref = json.loads((cells.E76_OUT / "exp076_J4V-X_d11_s3.json").read_text())
    assert recs[("J4V-X", 11)][3]["solved"] == ref["solved"] and "judge" not in ref
    assert recs[("J4V-S", 9)][7]["solved"] == 50
    (tmp_path / evaluate.record_name("J4V-S", 9, 7)).write_text(
        json.dumps(_ev("J4V-S", 9, 7, 50, judge="X")))
    with pytest.raises(SystemExit, match="judge"):
        agg.load_records(tmp_path)


def test_eval_records_at_other_settings_or_smoke_are_refused(tmp_path):
    """Catches a smoke cell, a 3-move cell, a partly trained judge, or a record whose content
    is not the cell its filename names."""
    for over, msg in (({"limit_states": 20}, "smoke"), ({"k": 3}, "k is"),
                      ({"judge_updates": 3750}, "judge_updates"), ({"seed": 4}, "seed")):
        for d in (9, 11):
            for s in range(12):
                (tmp_path / evaluate.record_name("J4V-S", d, s)).write_text(
                    json.dumps(_ev("J4V-S", d, s, 50)))
        rec = _ev("J4V-S", 9, 2, 50)
        rec.update(over)
        (tmp_path / evaluate.record_name("J4V-S", 9, 2)).write_text(json.dumps(rec))
        with pytest.raises(SystemExit, match=msg):
            agg.load_records(tmp_path)


def test_stage_cont_compares_against_exp076s_committed_records(tmp_path):
    """Catches Gate 0(b) checked against the wrong records, or blind to a changed outcome."""
    for d in (9, 11):
        ref = json.loads((cells.E76_OUT / f"exp076_J4V-X_d{d}_s0.json").read_text())
        (tmp_path / evaluate.record_name("J4V-X", d, 0)).write_text(json.dumps(ref))
    assert agg.stage_cont(tmp_path) == "PASS"
    p = tmp_path / evaluate.record_name("J4V-X", 11, 0)
    r = json.loads(p.read_text())
    r["solved"] += 1
    p.write_text(json.dumps(r))
    assert agg.stage_cont(tmp_path) == "FAIL"


def test_gate0a_ignores_only_the_session_fields(tmp_path):
    """Catches a determinism check that ignores an outcome field (judge_evals here) or that
    fails on wall_s and git_commit, which differ by design."""
    full, det = tmp_path / "o", tmp_path / "d"
    for d in (full, det):
        d.mkdir()
    base = _ev("J4V-S", 9, 0, 77, judge_evals=1296.0)
    (full / evaluate.record_name("J4V-S", 9, 0)).write_text(json.dumps(base))
    (det / evaluate.record_name("J4V-S", 9, 0)).write_text(
        json.dumps({**base, "wall_s": 1.0, "git_commit": "abc"}))
    assert agg.gate0a(full, det) == "PASS"
    (det / evaluate.record_name("J4V-S", 9, 0)).write_text(
        json.dumps({**base, "judge_evals": 1295.0}))
    assert agg.gate0a(full, det) == "FAIL"


def test_claims_compare_s_with_x_at_the_registered_depths():
    """Catches Claim 1 read at depth 9 or Claim 2 at depth 11, or the arms reversed."""
    rates = {("J4V-S", 11): {s: 0.30 for s in range(12)},
             ("J4V-X", 11): {s: 0.20 for s in range(12)},
             ("J4V-S", 9): {s: 0.40 for s in range(12)},
             ("J4V-X", 9): {s: 0.45 for s in range(12)}}
    full = agg.primary_verdicts(rates, True)
    assert full["claim1"][0] == "CONFIRMED" and full["claim1"][2] == pytest.approx(0.10)
    assert full["claim2"][0] == "REFUTED" and full["claim2"][2] == pytest.approx(-0.05)


def test_readings_lead_with_gate_m_and_name_a_trade():
    """Catches the spec's fixed readings missing from the output: a confirmed Claim 1 with
    Gate M not above 0 must lead RESULTS.md, a confirmed Claim 1 with a refuted Claim 2 is a
    trade, and a null Claim 1 is about this lever only."""
    c = ("CONFIRMED", 0.001, 0.05, 0.3, 0.25)
    r = ("REFUTED", 0.9, -0.02, 0.4, 0.42)
    ns = ("NOT SIGNIFICANT", 0.2, 0.01, 0.2, 0.19)
    lines = agg.readings({"claim1": c, "claim2": r}, gate_m_above=False)
    assert any("GATE M" in x and "lead" in x for x in lines)
    assert any("trade" in x for x in lines)
    assert agg.readings({"claim1": c, "claim2": c}, gate_m_above=True) == []
    assert any("BY THIS LEVER" in x for x in
               agg.readings({"claim1": ns, "claim2": c}, gate_m_above=True))


# The regexes chain079.ps1 applies to the aggregator's stage output (PowerShell -match is
# case-insensitive). Kept here verbatim so a change to either side fails a test.
CHAIN_GATE0B = "continuity.*: PASS"
CHAIN_GATE_L = r"GATE L \(VI worked\): PASS"
CHAIN_GATE_E = r"GATE E \(encoders trained\): PASS"
CHAIN_GATE_MC = r"GATE M continuity \(X seed 0 Gate L re-measured here\): PASS"


def test_launcher_pins_the_walk_length():
    """Catches a launcher that can start S training at a walk length other than 40, or that
    does not pass it to the driver (spec section 4)."""
    text = (EXP / "launch079.ps1").read_text()
    assert "[int]$MaxLen = 40" in text
    assert "if ($MaxLen -ne 40)" in text
    assert text.count('"--max-len", $MaxLen') == 2  # pilot-train and train


def test_launcher_carries_the_fixes_and_the_registered_cells():
    """Catches a launcher without EXP-074's .Handle fix, without the amendment guard on both
    fields, with an eval list missing an arm or depth, or with the worker cap lifted."""
    text = (EXP / "launch079.ps1").read_text()
    assert "$null = $proc.Handle" in text
    assert "a.AMENDMENT" in text and "a.GATE_L_THRESHOLD" in text
    assert 'foreach ($d in 11, 9)' in text and '"J4V-S"' in text and '"J4V-X"' in text
    assert "-gt 6" in text and "outputs_det" in text
    assert "prepare-pilot-encoders" in text and '"--gatel-x"' in text
    assert "075_wide_region\\pretrain.py" in text and '"--arms", "X"' in text
    assert '$target = Join-Path $phaseDir $j[1]' in text


def test_launcher_samples_memory_in_driver_phases_too():
    """Catches training phases whose peak committed memory is never measured: the driver runs
    its own pool, so the launcher must sample every python process, not only its own child."""
    text = (EXP / "launch079.ps1").read_text()
    assert "RedirectStandardOutput" in text
    assert text.count("PrivateMemorySize64") >= 2
    assert "max total commit MB" in text and "peak commit MB" in text


def test_chain_order_and_gate_regexes_match_the_aggregator():
    """Catches a chain that skips a gate, runs eval before Gates L and E, or whose regex no
    longer matches what the aggregator prints, so a PASS would read as a STOP or a FAIL as a
    PASS."""
    text = (EXP / "chain079.ps1").read_text()
    assert '@("cont","gate0b","train","gatele","eval","det","rank")' in text
    for rx in (CHAIN_GATE0B, CHAIN_GATE_L, CHAIN_GATE_E, CHAIN_GATE_MC):
        assert rx in text
    assert "--stage cont" in text and "--stage train" in text
    assert "train_workers" in text and "eval_workers" in text
    assert re.search(CHAIN_GATE0B, agg.format_stage_cont("PASS"), re.I)
    assert not re.search(CHAIN_GATE0B, agg.format_stage_cont("FAIL"), re.I)
    m = {"mean": 0.01, "p": 0.2, "wins": 7, "diffs": [0.0] * 12, "above": False}
    ok = agg.format_stage_train({"l": {"pass": True, "mean": 0.2, "threshold": 0.1, "p": 0.001},
                                 "e": True, "m": m, "m_cont": "PASS"})
    bad = agg.format_stage_train({"l": {"pass": False, "mean": 0.05, "threshold": 0.1,
                                        "p": 0.3}, "e": False, "m": m, "m_cont": "FAIL"})
    assert re.search(CHAIN_GATE_L, "\n".join(ok), re.I)
    assert re.search(CHAIN_GATE_E, "\n".join(ok), re.I)
    assert not re.search(CHAIN_GATE_L, "\n".join(bad), re.I)
    assert not re.search(CHAIN_GATE_E, "\n".join(bad), re.I)
    assert re.search(CHAIN_GATE_MC, "\n".join(ok), re.I)
    assert not re.search(CHAIN_GATE_MC, "\n".join(bad), re.I)


def test_gate_m_binds_only_when_its_continuity_passed():
    """Catches Gate M counting as above 0 while X's side failed to reproduce, and the printed
    verdict line naming ABOVE 0 in that case."""
    above = {"above": True}
    assert agg.gate_m_readable_above(above, "FAIL") is False
    assert agg.gate_m_readable_above(above, "PASS") is True
    assert agg.gate_m_readable_above({"above": False}, "PASS") is False
    c = ("CONFIRMED", 0.001, 0.05, 0.3, 0.25)
    lines = agg.readings({"claim1": c, "claim2": c},
                         agg.gate_m_readable_above(above, "FAIL"))
    assert any("GATE M" in x and "lead" in x for x in lines)
    m = {"mean": 0.01, "p": 0.01, "wins": 10, "diffs": [0.0] * 12, "above": True}
    res = {"l": {"pass": True, "mean": 0.2, "threshold": 0.1, "p": 0.001}, "e": True, "m": m}
    bad = "\n".join(agg.format_stage_train({**res, "m_cont": "FAIL"}))
    assert "GATE M CANNOT BE READ (continuity FAIL)" in bad
    assert "ABOVE 0" not in bad
    assert "-> ABOVE 0" in "\n".join(agg.format_stage_train({**res, "m_cont": "PASS"}))


def test_claims_gate_ok_needs_all_four_gates_and_voids_the_claims():
    """Catches a VOID composition that ignores any one of Gate 0(a), 0(b), L or E."""
    assert agg.claims_gate_ok("PASS", "PASS", True, True) is True
    flips = [("FAIL", "PASS", True, True), ("PASS", "FAIL", True, True),
             ("PASS", "PASS", False, True), ("PASS", "PASS", True, False)]
    rates = {("J4V-S", 11): {s: 0.30 for s in range(12)},
             ("J4V-X", 11): {s: 0.20 for s in range(12)},
             ("J4V-S", 9): {s: 0.40 for s in range(12)},
             ("J4V-X", 9): {s: 0.35 for s in range(12)}}
    for args in flips:
        ok = agg.claims_gate_ok(*args)
        assert ok is False
        full = agg.primary_verdicts(rates, ok)
        assert full["claim1"][0] == "VOID" and full["claim2"][0] == "VOID"


def test_readings_lead_with_gate_m_for_a_small_confirmation_too():
    """Catches the Gate M lead line keyed on exactly CONFIRMED, missing CONFIRMED (SMALL)."""
    small = ("CONFIRMED (SMALL)", 0.001, 0.005, 0.3, 0.295)
    ok = ("CONFIRMED", 0.001, 0.05, 0.3, 0.25)
    lines = agg.readings({"claim1": small, "claim2": ok}, gate_m_above=False)
    assert any("GATE M" in x and "lead" in x for x in lines)
    assert agg.readings({"claim1": small, "claim2": ok}, gate_m_above=True) == []


def test_chain_refuses_workers_outside_1_to_6_before_any_phase():
    """Catches a chain that reads the amendment's worker counts and only finds out hours later
    (at the train phase) that the launcher refuses more than 6."""
    text = (EXP / "chain079.ps1").read_text()
    check = text.index("-gt 6")
    assert "-lt 1" in text
    assert text.index("$trainW, $evalW = $w.Split") < check < text.index('Say "workers:')
    assert check < text.index("foreach ($p in $order)")
    assert check < text.index("-File $l")
    assert "exit 1" in text[check:check + 300]


def test_stage_train_refuses_a_record_whose_seed_is_not_its_file_names(tmp_path, monkeypatch):
    """Catches Gate M pairing S and X by file name while a record inside says another seed."""
    monkeypatch.setattr(agg, "AMENDMENT", {"date": "2026-10-20"})
    monkeypatch.setattr(agg, "GATE_L_THRESHOLD", 0.1)
    _write_s_train(tmp_path, 0.0)
    _write_gatel_x(tmp_path)
    r = json.loads((tmp_path / cells.record_name(3)).read_text())
    r["seed"] = 4
    (tmp_path / cells.record_name(3)).write_text(json.dumps(r))
    with pytest.raises(SystemExit, match=cells.record_name(3)):
        agg.stage_train(tmp_path)


def test_rank_records_are_checked_for_kind_k_updates_and_smoke(tmp_path):
    """Catches Gate R reading a record that is not a 4-move J-S4 rank record of the fully
    trained judge."""
    good = {"kind": "J-S4", "k": 4, "judge_updates": 4000, "limit_states": None}
    agg.check_rank_record(good, tmp_path / "rank.json")
    for field, bad in (("kind", "gatel-X"), ("k", 5), ("judge_updates", 3000),
                       ("limit_states", 10)):
        with pytest.raises(SystemExit, match=f"rank.json: {field}"):
            agg.check_rank_record({**good, field: bad}, tmp_path / "rank.json")
    (tmp_path / "r.json").write_text(json.dumps({**good, "kind": "gatel-X"}))
    with pytest.raises(SystemExit, match="kind"):
        agg._read_rank(tmp_path / "r.json")
