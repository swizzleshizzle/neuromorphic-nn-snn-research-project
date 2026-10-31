"""EXP-084: the GPU port's gates (C, P, E, M, T), aggregator and drivers. The device code itself is
tested in tests/training/test_device_port.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "084_gpu_port"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp084_cells", EXP / "cells.py")
gatec = _load("exp084_gatec", EXP / "gatec.py")


def test_names():
    assert cells.GATEC_NAME == "exp084_gatec.json"
    assert cells.p_name("J4V-X16k", 3) == "exp084_P_J4V-X16k_s3.json"
    assert cells.e_name("J4V-X", 11) == "exp084_E_J4V-X_d9_s11.json"
    assert cells.E_GATER_NAME == "exp084_E_gateR_s0.json"
    assert cells.M_NAME == "exp084_M_s0.json"
    assert cells.train_name(5) == "exp084_train_X_s5.json"
    assert cells.ckpt_dir(5) == cells.OUT / "judge_X_s5"
    assert (cells.TOL, cells.P_LIMIT, cells.N_UPDATES) == (2, 9, 4000)
    assert cells.ARMS == ("J4V-X16k", "J4V-X") and list(cells.SEEDS) == list(range(12))


def _fp():
    return {"encode_sha": "a", "judge_vals": ["0.5", "0.25"], "step_loss": "0.7",
            "step_param_sha": "b", "probe": {"spearman_7_11": 0.1, "mean_j_by_distance": {"7": 1.0}},
            "search": {"solved": 12, "optimality": 0.9}}


def test_compare_equal_is_empty():
    assert gatec.compare(_fp(), _fp()) == []


def test_compare_flags_one_float():
    b = _fp()
    b["probe"]["mean_j_by_distance"]["7"] = 1.0000000000000002
    assert gatec.compare(_fp(), b) == ["probe"]


def test_files_under_root(tmp_path):
    inside = (tmp_path / "experiments" / "x.py").as_posix()
    assert gatec.files_under({"module_files": [inside]}, tmp_path)
    assert not gatec.files_under({"module_files": [inside, "/elsewhere/y.py"]}, tmp_path)


decisions = _load("exp084_decisions", EXP / "decisions.py")


def test_stop_boundary():
    assert decisions.stop(9) is False
    assert decisions.stop(10) is True


@pytest.fixture(scope="module")
def p_inputs():
    import copy

    import torch
    judge = decisions.e75.load_judge("X", 0, cells.E82_OUT, 16000, device="cpu")
    bumped = copy.deepcopy(judge)
    with torch.no_grad():
        bumped.head[0].weight.mul_(1.01)
    states = decisions.e83.load_states(0, 9, cells.E83_OUT)[:3]
    return judge, bumped, states


def test_compare_cpu_vs_cpu_agrees(p_inputs):
    judge, _bumped, states = p_inputs
    c = decisions.e75.JudgeCritic(judge)
    res = decisions.compare_critics(c, c, states, imag_seed=123)
    assert res["n"] == 3 and res["agree"] == 3 and res["disagree"] == 0
    assert res["max_abs_dj"] == 0.0


def test_compare_detects_perturbed_judge(p_inputs):
    judge, bumped, states = p_inputs
    res = decisions.compare_critics(decisions.e75.JudgeCritic(judge),
                                    decisions.e75.JudgeCritic(bumped), states, imag_seed=123)
    assert res["max_abs_dj"] > 0.0


def test_first_move_matches_the_real_search(p_inputs):
    """Gate P claims to compute the cell's own t = 1 move: check it against the search's trace."""
    import torch

    from neuromorphic.training.lookahead import evaluate_lookahead
    judge, _bumped, states = p_inputs
    agent, head, _s, ts = decisions.e75.load_cell(9, 0)
    critic = decisions.e75.JudgeCritic(judge)
    trace = []
    evaluate_lookahead(agent, head, states, depth=9, mode="C", k=4, k_goal=4,
                       generator=torch.Generator().manual_seed(ts), rng_seed=ts, imag_seed=ts,
                       critic=critic, no_revisit=True, trace=trace)
    got = [decisions.first_move(critic, s, ts, i)[0] for i, s in enumerate(states)]
    assert got == [t[0][0] for t in trace]


reeval = _load("exp084_reeval", EXP / "reeval.py")
train = _load("exp084_train", EXP / "train.py")


def test_published_reads_records():
    assert reeval.published("J4V-X16k", 0) == 124
    assert reeval.published("J4V-X", 3) == 52
    assert reeval.published_gate_r() == 127


def test_published_refuses_limited_record(tmp_path):
    src = cells.E83_OUT / "exp083_J4V-X16k_d9_s0.json"
    rec = json.loads(src.read_text())
    rec["limit_states"] = 3
    (tmp_path / src.name).write_text(json.dumps(rec))
    with pytest.raises(SystemExit):
        reeval.published("J4V-X16k", 0, root=tmp_path)
    rec["limit_states"], rec["n"] = None, 199
    (tmp_path / src.name).write_text(json.dumps(rec))
    with pytest.raises(SystemExit):
        reeval.published("J4V-X16k", 0, root=tmp_path)


E_FIELDS = {"solved", "published", "diff", "n", "wall_s", "device", "device_name",
            "torch_version", "limit_states", "git_commit"}


def test_e_cell_record_fields(tmp_path):
    rec = reeval.e_cell("J4V-X16k", 0, "cpu", limit=1, out=tmp_path)
    assert E_FIELDS <= set(rec)
    on_disk = json.loads((tmp_path / cells.e_name("J4V-X16k", 0)).read_text())
    assert on_disk["published"] == 124 and on_disk["n"] == 1 and on_disk["limit_states"] == 1


def test_train_writes_only_under_084(tmp_path, monkeypatch):
    from neuromorphic.envs.cube import SOLVED, apply_move
    guarded = cells.E75_OUT / "judge_X_s0" / "state.json"
    before = guarded.read_bytes()
    walk = [SOLVED]
    for a in [0, 2, 4, 1, 3, 5, 0, 2]:
        walk.append(apply_move(walk[-1], a))
    probe = [(s, 7 + i % 5) for i, s in enumerate(walk[1:])]
    monkeypatch.setattr(train, "_sets", lambda seed: (set(), probe, lambda s: 5))
    rec = train.run(0, "cpu", n_updates=2, batch=20, out=tmp_path)
    assert (tmp_path / "judge_X_s0" / "state.json").exists()
    assert (tmp_path / cells.train_name(0)).exists()
    assert guarded.read_bytes() == before
    assert {"gate_l", "wall_s", "s_per_update", "loss_last", "probes", "device", "device_name",
            "torch_version", "git_commit"} <= set(rec)


agg = _load("exp084_aggregate", EXP / "aggregate.py")


def test_within_boundary():
    assert agg.within(129, 127) and agg.within(125, 127)
    assert not agg.within(130, 127) and not agg.within(124, 127)


@pytest.mark.parametrize("flags,want", [
    ((False, True, True, True), "VOID"),
    ((False, False, False, False), "VOID"),
    ((True, False, False, True), "DESKTOP DIFFERS FROM LAPTOP ON CPU TOO"),
    ((True, False, True, True), "GPU PORT DOES NOT REPRODUCE"),
    ((True, True, True, False), "GPU SEARCH REPRODUCES, TRAINING DOES NOT LEARN"),
    ((True, True, False, False), "GPU SEARCH REPRODUCES, TRAINING DOES NOT LEARN"),
    ((True, True, True, True), "GPU PORT REPRODUCES"),
])
def test_verdict_rows(flags, want):
    assert agg.verdict(*flags) == want


def _w(path, rec):
    path.write_text(json.dumps(rec))


def _p_records(out, disagree_each):
    for i, (a, s) in enumerate((a, s) for s in cells.SEEDS for a in cells.ARMS):
        d = disagree_each[i] if isinstance(disagree_each, list) else disagree_each
        _w(out / cells.p_name(a, s), {"n": 200, "agree": 200 - d, "disagree": d,
                                      "near_ties": 0, "max_abs_dj": 1e-6, "goal_fired": 30,
                                      "limit_states": None, "wall_s": 60.0})


def test_run_p_halts(tmp_path):
    _p_records(tmp_path, [1] * 10 + [0] * 14)
    res = agg.run("p", tmp_path)
    assert res["halt"] is True
    assert any("GATE P: HALT" in line for line in res["lines"])
    _p_records(tmp_path, [1] * 9 + [0] * 15)
    assert agg.run("p", tmp_path)["halt"] is False


def test_run_all_refuses_missing_cell(tmp_path):
    _w(tmp_path / cells.GATEC_NAME, {"pass": True, "differences": []})
    _p_records(tmp_path, 0)
    pairs = [(a, s) for s in cells.SEEDS for a in cells.ARMS]
    for a, s in pairs[:-1]:
        _w(tmp_path / cells.e_name(a, s), {"solved": 100, "published": 100, "n": 200,
                                           "limit_states": None, "wall_s": 100.0,
                                           "device": "cuda"})
    with pytest.raises(SystemExit, match=cells.e_name(*pairs[-1])):
        agg.run("all", tmp_path)


def _all_records(out, e_off=0):
    _w(out / cells.GATEC_NAME, {"pass": True, "differences": []})
    _p_records(out, 0)
    for i, (a, s) in enumerate((a, s) for s in cells.SEEDS for a in cells.ARMS):
        _w(out / cells.e_name(a, s), {"solved": 100 + (e_off if i == 5 else 0), "published": 100,
                                      "n": 200, "limit_states": None, "wall_s": 100.0,
                                      "device": "cuda"})
    _w(out / cells.E_GATER_NAME, {"solved": 127, "published": 127, "n": 200,
                                  "limit_states": None, "wall_s": 90.0, "device": "cuda"})
    _w(out / cells.M_NAME, {"solved": 126, "published": 127, "n": 200, "limit_states": None,
                            "wall_s": 3000.0, "device": "cpu"})
    for s in cells.SEEDS:
        cpu = json.loads((cells.E75_OUT / f"exp075_train_X_s{s}.json").read_text())
        _w(out / cells.train_name(s), {"gate_l": cpu["gate_l"], "n_updates": 4000,
                                       "s_per_update": 0.11, "device": "cuda"})


def test_run_all_reaches_a_verdict(tmp_path):
    _all_records(tmp_path)
    res = agg.run("all", tmp_path)
    assert res["verdict"] == "GPU PORT REPRODUCES"
    assert any(line.startswith("  device X minus EXP-075 CPU X, overall margin")
               for line in res["lines"])


def test_run_all_three_off_fails_gate_e(tmp_path):
    _all_records(tmp_path, e_off=3)
    assert agg.run("all", tmp_path)["verdict"] == "GPU PORT DOES NOT REPRODUCE"
    _all_records(tmp_path, e_off=-2)
    assert agg.run("all", tmp_path)["verdict"] == "GPU PORT REPRODUCES"


@pytest.mark.parametrize("name,dev", [
    (cells.e_name("J4V-X", 4), "cpu"),
    (cells.E_GATER_NAME, "cpu"),
    (cells.M_NAME, "cuda"),
    (cells.train_name(7), "cpu"),
])
def test_run_all_refuses_a_record_from_the_wrong_device(tmp_path, name, dev):
    """A manual CPU rerun of a GPU cell must not count as a GPU pass, nor the reverse for Gate M."""
    _all_records(tmp_path)
    rec = json.loads((tmp_path / name).read_text())
    rec["device"] = dev
    _w(tmp_path / name, rec)
    with pytest.raises(SystemExit, match=name):
        agg.run("all", tmp_path)
