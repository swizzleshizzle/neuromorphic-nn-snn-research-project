"""EXP-080: oracle judges trained on the true distance, as instruments for X's far ordering."""

from __future__ import annotations

import importlib.util
import json
import random
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from neuromorphic.envs.cube import N_ACTIONS, SOLVED, apply_move
from neuromorphic.training import value_iteration as vi
from neuromorphic.training.encoder_pretrain import save_encoder
from neuromorphic.training.lookahead import tree_levels

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments" / "080_oracle_ordering"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp080_cells", EXP / "cells.py")
c75 = cells.c75


def _fake_x(d, seed):
    """EXP-075's layout for one seed under `d`: the pretrained encoder, and an X judge at 4000
    updates whose region has moved away from it (so 'X's trained region' and 'the pretrained
    encoder' are different tensors)."""
    enc = c75.fresh_sensory("X", seed)
    path = c75.encoder_path("X", seed, d)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder(enc, path)
    judge = c75.build_judge(seed, "X", c75.fresh_sensory("X", seed))
    with torch.no_grad():
        judge.sensory.fc1.weight.add_(0.01)
        judge.head[0].weight.add_(0.5)
    ck = c75.ckpt_dir("X", seed, d)
    ck.mkdir(parents=True, exist_ok=True)
    torch.save(judge.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": 4000}))
    return judge


def test_constants_are_the_spec_values():
    """Catches an arm, budget, walk length or seed set drifting from spec section 3."""
    assert cells.ARMS == ("O-raw", "O-Xf", "O-Xt")
    assert cells.TRAIN_ORDER == ("O-Xt", "O-Xf", "O-raw")
    assert (cells.MAX_LEN, cells.N_UPDATES, cells.BATCH, cells.PROBE_EVERY) == (40, 4000, 1000, 250)
    assert (cells.RAW_STATES, cells.RAW_EPOCHS, cells.RAW_BATCH, cells.RAW_HIDDEN) == (
        600_000, 12, 512, 512)
    assert cells.EVAL_SEEDS == tuple(range(12)) and cells.PILOT_SEEDS == (12, 13)
    assert cells.x_judge_root(False) == cells.E75_OUT and cells.x_judge_root(True) == cells.E75_PILOT
    assert cells.encoder_root(False) == cells.E75_OUT
    assert cells.record_name("O-Xt", 3) == "exp080_train_O-Xt_s3.json"
    assert cells.ckpt_dir("O-Xf", 3, Path("/x")) == Path("/x/judge_O-Xf_s3")


def test_leaf_exclusion_is_every_3_move_leaf_of_probe_states_7_to_11():
    """Catches leaves taken at the wrong depth, from probe states outside 7 to 11, or not at all
    (the label leak spec section 3 closes)."""
    a = apply_move(SOLVED, 0)
    b = apply_move(SOLVED, 1)
    c = apply_move(SOLVED, 2)
    probe = [(a, 7), (b, 11), (c, 6)]
    got = cells.leaf_exclusion(probe)
    want = set(tree_levels(a, 3, N_ACTIONS)[3]) | set(tree_levels(b, 3, N_ACTIONS)[3])
    assert got == want
    assert not set(tree_levels(c, 3, N_ACTIONS)[3]) <= got  # distance 6 contributes nothing


def test_oracle_exclusion_is_heldout_plus_probe_plus_leaves(monkeypatch):
    """Catches an exclusion that drops the held-out states, the probe states or their leaves."""
    a = apply_move(SOLVED, 0)
    held = {apply_move(SOLVED, 5)}
    monkeypatch.setattr(cells, "exclusion_set", lambda seed, provider: set(held))
    monkeypatch.setattr(cells, "probe_set", lambda seed, provider: [(a, 9)])
    got = cells.oracle_exclusion(0, None)
    assert held <= got and a in got
    assert set(tree_levels(a, 3, N_ACTIONS)[3]) <= got
    assert got == held | {a} | set(tree_levels(a, 3, N_ACTIONS)[3])


def test_raw_judge_shape_seed_and_generator():
    """Catches an O-raw net of the wrong shape, an init not fixed by the seed, or a forward that
    cannot be called the way leaf_rank_margin and probe_judge call a judge."""
    j0, j0b, j1 = cells.make_oraw(0), cells.make_oraw(0), cells.make_oraw(1)
    shapes = [tuple(p.shape) for p in j0.parameters()]
    assert shapes == [(512, 144), (512,), (512, 512), (512,), (1, 512), (1,)]
    assert all(torch.equal(p, q) for p, q in zip(j0.parameters(), j0b.parameters()))
    assert not torch.equal(j0.net[0].weight, j1.net[0].weight)
    states = [SOLVED, apply_move(SOLVED, 3)]
    out = j0(states, torch.Generator().manual_seed(0))
    assert out.shape == (2,)
    assert torch.equal(out, j0(np.array(states, dtype=np.uint8)))
    oh = cells.RawJudge.one_hot([SOLVED])
    assert oh.shape == (1, 144) and float(oh.sum()) == 24.0


def test_oxf_is_xs_trained_region_under_oxts_fresh_head(tmp_path):
    """Catches O-Xf built on X's trained head, on the pretrained encoder instead of X's trained
    region, or with a region that is not frozen; and O-Xt not built from the encoder."""
    x = _fake_x(tmp_path, 12)
    oxf = cells.make_oxf(12, tmp_path)
    oxt = cells.make_oxt(12, tmp_path)
    for k, v in oxf.head.state_dict().items():
        assert torch.equal(v, oxt.head.state_dict()[k])
    assert not torch.equal(oxf.head[0].weight, x.head[0].weight)
    assert torch.equal(oxf.sensory.fc1.weight, x.sensory.fc1.weight)
    assert not torch.equal(oxt.sensory.fc1.weight, x.sensory.fc1.weight)
    enc = c75.load_pretrained("X", 12, tmp_path)
    assert torch.equal(oxt.sensory.fc1.weight, enc.fc1.weight)
    assert all(not p.requires_grad for p in oxf.sensory.parameters())
    assert all(p.requires_grad for p in oxt.sensory.parameters())


def test_optimizers_are_the_spec_ones(tmp_path):
    """Catches O-Xf's optimizer touching the region, O-Xt's missing it, or a wrong lr."""
    _fake_x(tmp_path, 12)
    oxf, oxt, raw = (cells.make_judge(a, 12, tmp_path, tmp_path) for a in ("O-Xf", "O-Xt", "O-raw"))
    f, t, r = (cells.optimizer_for(a, j) for a, j in (("O-Xf", oxf), ("O-Xt", oxt), ("O-raw", raw)))
    f_params = {id(p) for g in f.param_groups for p in g["params"]}
    assert f_params == {id(p) for p in oxf.head.parameters()}
    assert [g["lr"] for g in t.param_groups] == [1e-3, 1e-4]
    assert {id(p) for p in t.param_groups[1]["params"]} == {id(p) for p in oxt.sensory.parameters()}
    assert [g["lr"] for g in r.param_groups] == [1e-3]
    with pytest.raises(ValueError):
        cells.optimizer_for("S", raw)


def test_reference_sensory_is_xs_region_for_oxf_and_the_encoder_for_oxt(tmp_path):
    """Catches a drift measured from the wrong starting point, which would make O-Xf's 0.0 or
    O-Xt's > 0 meaningless."""
    x = _fake_x(tmp_path, 12)
    rf = cells.reference_sensory("O-Xf", 12, tmp_path, tmp_path)
    rt = cells.reference_sensory("O-Xt", 12, tmp_path, tmp_path)
    assert torch.equal(rf["fc1.weight"], x.sensory.fc1.weight)
    assert torch.equal(rt["fc1.weight"], c75.load_pretrained("X", 12, tmp_path).fc1.weight)
    assert cells.drift(cells.make_oxf(12, tmp_path).sensory.state_dict(), rf) == 0.0
    assert cells.drift(cells.make_oxt(12, tmp_path).sensory.state_dict(), rt) == 0.0


def test_load_trained_refuses_a_missing_or_unfinished_judge(tmp_path):
    """Catches evaluating a judge that is absent or short of its budget (judge.pt exists from the
    first checkpoint, so its existence proves nothing)."""
    with pytest.raises(SystemExit, match="missing"):
        cells.load_trained("O-Xt", 12, tmp_path)
    j = cells.make_oraw(12)
    ck = cells.ckpt_dir("O-raw", 12, tmp_path)
    ck.mkdir(parents=True)
    torch.save(j.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"epochs": 3}))
    with pytest.raises(SystemExit, match="epochs 3"):
        cells.load_trained("O-raw", 12, tmp_path)
    got = cells.load_trained("O-raw", 12, tmp_path, required=3)
    assert torch.equal(got.net[0].weight, j.net[0].weight) and not got.training
    x = _fake_x(tmp_path / "x", 12)
    ck = cells.ckpt_dir("O-Xt", 12, tmp_path)
    ck.mkdir(parents=True)
    torch.save(x.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": 250}))
    with pytest.raises(SystemExit, match="update 250"):
        cells.load_trained("O-Xt", 12, tmp_path)
    (ck / "state.json").write_text(json.dumps({"update": 4000}))
    got = cells.load_trained("O-Xt", 12, tmp_path)
    assert torch.equal(got.sensory.fc1.weight, x.sensory.fc1.weight)


mint = _load("exp080_mint", EXP / "mint.py")


class FakeDistance:
    """A deterministic stand-in for the BFS table. Mint plumbing does not care whether a label
    is physically right, only that it is the provider's answer for that state."""

    def distance(self, s):
        return sum(i * int(v) for i, v in enumerate(s)) % 15


def _probe(n=5, seed=99):
    states = vi.random_walk_states(n, 6, random.Random(seed), N_ACTIONS, set())
    return [(s, 7 + i % 5) for i, s in enumerate(states)]


def _fake_mint(root, seed, n_blocks=4, block=8, max_len=6, probe=None):
    fd = FakeDistance()
    probe = _probe() if probe is None else probe
    states, labels = mint.draw_blocks(n_blocks, block, max_len, seed, set(), fd.distance)
    ls, ld = mint.leaf_table(probe, fd.distance)
    mint.write_mint(root, seed, states, labels, probe, ls, ld,
                    {"seed": seed, "n_blocks": n_blocks, "block": block, "max_len": max_len})
    return mint.read_mint(root, seed)


def test_draw_blocks_is_random_walk_states_called_once_per_block():
    """Catches draws that are not the library's walks in update order (one call per block from
    one random.Random(seed)), or labels that are not the provider's."""
    fd = FakeDistance()
    states, labels = mint.draw_blocks(3, 5, 7, 4, set(), fd.distance)
    rng = random.Random(4)
    want = [s for _ in range(3) for s in vi.random_walk_states(5, 7, rng, N_ACTIONS, set())]
    assert states.dtype == np.uint8 and states.shape == (15, 24)
    assert [tuple(int(x) for x in r) for r in states] == want
    assert labels.dtype == np.uint8 and labels.tolist() == [fd.distance(s) for s in want]


def test_draw_blocks_never_returns_an_excluded_state():
    """Catches an exclusion that is ignored. The control draw proves the excluded states are ones
    the walks actually reach, so the assertion can fail."""
    fd = FakeDistance()
    probe = [(apply_move(SOLVED, a), 7) for a in range(N_ACTIONS)]
    leaves = cells.leaf_exclusion(probe)
    control, _ = mint.draw_blocks(20, 50, 4, 0, set(), fd.distance)
    assert sum(tuple(int(x) for x in r) in leaves for r in control) > 50
    states, _ = mint.draw_blocks(20, 50, 4, 0, leaves, fd.distance)
    assert not any(tuple(int(x) for x in r) in leaves for r in states)


def test_draw_blocks_refuses_an_unlabelled_state():
    """Catches a bounded provider silently minting None (stored as 0) for a far state."""
    with pytest.raises(SystemExit, match="no distance"):
        mint.draw_blocks(1, 4, 5, 0, set(), lambda s: None)


def test_mint_round_trips_probe_and_leaf_distances(tmp_path):
    """Catches a probe whose order or labels change on disk (leaf_rank_margin consumes its
    generator in probe order), or leaf keys that do not match what tree_levels produces."""
    probe = _probe()
    m = _fake_mint(tmp_path, 12, probe=probe)
    assert m.probe == probe
    assert all(type(x) is int for x in m.probe[0][0]) and type(m.probe[0][1]) is int
    leaf = tree_levels(probe[0][0], 3, N_ACTIONS)[3][17]
    assert m.leaf_dist[leaf] == FakeDistance().distance(leaf)
    assert len(m.leaf_dist) == len(cells.leaf_exclusion(probe))
    assert m.meta["sha256"] == m.sha and m.seed == 12


def test_read_mint_refuses_a_tampered_file(tmp_path):
    """Catches training on a mint whose arrays are not the ones its sidecar describes (a partial
    write, or a file swapped after a re-mint)."""
    _fake_mint(tmp_path, 12)
    npz = mint.mint_dir(tmp_path) / mint.npz_name(12)
    z = dict(np.load(npz))
    z["labels"] = z["labels"].copy()
    z["labels"][0] = (int(z["labels"][0]) + 1) % 15
    np.savez(npz, **z)
    with pytest.raises(SystemExit, match="sha256"):
        mint.read_mint(tmp_path, 12)


def test_read_mint_refuses_a_missing_mint_or_a_wrong_seed(tmp_path):
    """Catches training a seed on another seed's labels, or on nothing."""
    with pytest.raises(SystemExit, match="mint.py"):
        mint.read_mint(tmp_path, 12)
    _fake_mint(tmp_path, 12)
    d = mint.mint_dir(tmp_path)
    (d / mint.npz_name(13)).write_bytes((d / mint.npz_name(12)).read_bytes())
    (d / mint.meta_name(13)).write_text((d / mint.meta_name(12)).read_text())
    with pytest.raises(SystemExit, match="seed 12"):
        mint.read_mint(tmp_path, 13)


def test_mint_is_pending_until_its_sidecar_exists(tmp_path):
    """Catches a relaunch that skips a seed whose mint was killed after the arrays and before
    the sidecar (the 03:31 restart)."""
    _fake_mint(tmp_path, 12)
    d = mint.mint_dir(tmp_path)
    (d / mint.npz_name(13)).write_bytes(b"partial")
    assert mint.pending_seeds([12, 13], tmp_path) == [13]
    assert not list(d.glob("*.tmp*"))


def test_mint_seed_excludes_probe_leaves_and_records_its_settings(tmp_path, monkeypatch):
    """Catches the glue dropping the probe leaves from the exclusion, or a sidecar that does not
    state the draw it holds."""
    probe = [(apply_move(SOLVED, a), 7) for a in range(N_ACTIONS)]
    leaves = cells.leaf_exclusion(probe)
    # mint.py loads its own copy of cells.py, so patch that copy.
    monkeypatch.setattr(mint.cells, "probe_set", lambda seed, provider: list(probe))
    monkeypatch.setattr(mint.cells, "exclusion_set", lambda seed, provider: set())
    meta = mint.mint_seed(12, tmp_path, n_blocks=20, block=50, max_len=4, provider=FakeDistance())
    m = mint.read_mint(tmp_path, 12)
    assert not any(tuple(int(x) for x in r) in leaves for r in m.states)
    assert m.probe == probe and set(m.leaf_dist) == leaves
    for k, v in {"seed": 12, "n_blocks": 20, "block": 50, "max_len": 4,
                 "rng": "random.Random(seed)", "n_probe": 6, "n_leaves": len(leaves)}.items():
        assert meta[k] == v and m.meta[k] == v
    assert meta["n_exclude"] == len(leaves | {s for s, _ in probe})
    assert sum(meta["label_hist"].values()) == 1000


train = _load("exp080_train", EXP / "train.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")


def _small(arm, tmp_path, n_updates=2, mint_root=None, **kw):
    """One tiny training run on seed 12: fake X layout, fake mint of 4 blocks of 8."""
    if not (tmp_path / "x").exists():
        _fake_x(tmp_path / "x", 12)
    if mint_root is None:
        mint_root = tmp_path / "m"
        if not (mint.mint_dir(mint_root) / mint.meta_name(12)).exists():
            _fake_mint(mint_root, 12)
    args = dict(n_updates=n_updates, batch=8, probe_every=1, raw_states=32, raw_epochs=2,
                raw_batch=8, mint_root=mint_root, x_root=tmp_path / "x", enc_root=tmp_path / "x")
    args.update(kw)
    return train.run(arm, 12, tmp_path / "out", **args)


def test_oxt_trains_its_region_and_oxf_does_not(tmp_path):
    """Catches a freeze that does not hold (O-Xf's region moves) and a gradient that never
    arrives (O-Xt's region does not), CLAUDE.md's MemoryReadout lesson; and a head that never
    trains."""
    rf = _small("O-Xf", tmp_path)
    rt = _small("O-Xt", tmp_path)
    assert rf["encoder_drift"] == 0.0
    assert rt["encoder_drift"] > 0.0
    f = cells.load_trained("O-Xf", 12, tmp_path / "out", required=2)
    t = cells.load_trained("O-Xt", 12, tmp_path / "out", required=2)
    fresh = cells.make_oxt(12, tmp_path / "x")
    assert not torch.equal(f.head[0].weight, fresh.head[0].weight)
    assert not torch.equal(t.head[0].weight, fresh.head[0].weight)
    x = c75.ckpt_dir("X", 12, tmp_path / "x") / "judge.pt"
    assert torch.equal(f.sensory.fc1.weight, torch.load(x)["sensory.fc1.weight"])


def test_record_carries_the_mint_and_a_gate_l_from_minted_distances(tmp_path):
    """Catches a record that cannot be paired across arms (no mint sha), or a Gate L that is not
    leaf_rank_margin on the minted probe and leaf distances."""
    rec = _small("O-Xt", tmp_path)
    m = mint.read_mint(tmp_path / "m", 12)
    on_disk = json.loads((tmp_path / "out" / cells.record_name("O-Xt", 12)).read_text())
    assert on_disk["mint_sha"] == m.sha and on_disk["mint_meta"]["n_blocks"] == 4
    assert (on_disk["arm"], on_disk["seed"], on_disk["n_updates"], on_disk["batch"]) == (
        "O-Xt", 12, 2, 8)
    j = cells.load_trained("O-Xt", 12, tmp_path / "out", required=2)
    want = gate_l.leaf_rank_margin(j, m.probe, m.leaf_dist.__getitem__, 12, N_ACTIONS)
    assert on_disk["gate_l"] == json.loads(json.dumps(want))
    assert on_disk["gate_l"]["n"] == 5 and rec["gate_l"] == want


def test_training_uses_block_u_at_update_u(tmp_path, monkeypatch):
    """Catches O-Xf and O-Xt reading different states at the same update (they must be paired on
    identical draws), or blocks read out of order."""
    seen = []
    real = train._block

    def spy(m, u, batch):
        xs, y = real(m, u, batch)
        seen.append((u, xs[0].tolist()))
        return xs, y

    monkeypatch.setattr(train, "_block", spy)
    _small("O-Xf", tmp_path, n_updates=3)
    m = mint.read_mint(tmp_path / "m", 12)
    assert seen == [(u, m.states[u * 8].tolist()) for u in range(3)]


def test_resume_equals_a_straight_run(tmp_path):
    """Catches a resume that restarts the data, the optimizer or the encoding generator, so a
    run interrupted by the 03:31 restart is not the run it would have been."""
    a = tmp_path / "a"
    b = tmp_path / "b"
    for d in (a, b):
        _fake_x(d / "x", 12)
        _fake_mint(d / "m", 12)
    _small("O-Xt", a, n_updates=4, probe_every=2)
    _small("O-Xt", b, n_updates=2, probe_every=2)
    rb = _small("O-Xt", b, n_updates=4, probe_every=2)
    assert rb["resumed_from"] == 2
    ja = torch.load(cells.ckpt_dir("O-Xt", 12, a / "out") / "judge.pt")
    jb = torch.load(cells.ckpt_dir("O-Xt", 12, b / "out") / "judge.pt")
    assert all(torch.equal(ja[k], jb[k]) for k in ja)


def test_resume_refuses_a_checkpoint_from_a_different_mint(tmp_path):
    """Catches splicing two training sets into one judge after a seed is re-minted."""
    _small("O-Xt", tmp_path, n_updates=1)
    (tmp_path / "out" / cells.record_name("O-Xt", 12)).unlink()
    other = tmp_path / "m2"
    fd = FakeDistance()
    states, labels = mint.draw_blocks(4, 8, 6, 777, set(), fd.distance)
    probe = _probe()
    ls, ld = mint.leaf_table(probe, fd.distance)
    mint.write_mint(other, 12, states, labels, probe, ls, ld,
                    {"seed": 12, "n_blocks": 4, "block": 8, "max_len": 6})
    with pytest.raises(SystemExit, match="different mint"):
        _small("O-Xt", tmp_path, n_updates=2, mint_root=other)


def test_short_mint_is_refused_before_training(tmp_path):
    """Catches a run that would index past its mint and train on a truncated last block."""
    with pytest.raises(SystemExit, match="needs 40"):
        _small("O-Xt", tmp_path, n_updates=5)
    with pytest.raises(SystemExit, match="needs 64"):
        _small("O-raw", tmp_path, raw_states=64)


def test_oraw_trains_its_epochs_and_records_them(tmp_path):
    """Catches O-raw training a different number of states or epochs than its record says, or a
    record that a margin reader would mistake for a spiking arm."""
    rec = _small("O-raw", tmp_path)
    on_disk = json.loads((tmp_path / "out" / cells.record_name("O-raw", 12)).read_text())
    assert (on_disk["raw_states"], on_disk["raw_epochs"], on_disk["raw_batch"],
            on_disk["raw_hidden"]) == (32, 2, 8, 512)
    assert on_disk["updates"] == 8 and len(on_disk["losses"]) == 2
    assert "encoder_drift" not in on_disk and "n_updates" not in on_disk
    j = cells.load_trained("O-raw", 12, tmp_path / "out", required=2)
    assert not torch.equal(j.net[0].weight, cells.make_oraw(12).net[0].weight)
    assert rec["gate_l"]["n"] == 5


def test_training_never_builds_a_bfs_table(tmp_path, monkeypatch):
    """Catches a training worker that loads the 1.9 GB table (spec section 3: the worker count is
    set by the encoder alone)."""
    from neuromorphic.envs import cube_distance

    def boom(self, *a, **k):
        raise AssertionError("training built a BFS table")

    monkeypatch.setattr(cube_distance.ExactBFSDistance, "__init__", boom)
    _small("O-Xt", tmp_path)
    _small("O-raw", tmp_path)


def test_pending_jobs_skips_finished_runs_and_keeps_train_order(tmp_path):
    """Catches a relaunch that retrains finished runs or puts the long arm last."""
    (tmp_path / cells.record_name("O-Xf", 12)).write_text("{}")
    jobs = [(a, s) for a in cells.TRAIN_ORDER for s in (12, 13)]
    assert train.pending_jobs(jobs, tmp_path) == [
        ("O-Xt", 12), ("O-Xt", 13), ("O-Xf", 13), ("O-raw", 12), ("O-raw", 13)]
    args = train.build_parser().parse_args(["--n-updates", "4000"])
    assert args.arms == list(cells.TRAIN_ORDER)


ev = _load("exp080_evaluate", EXP / "evaluate.py")


def test_gate0_cell_is_xs_gate_l_through_the_mint(tmp_path):
    """Catches a Gate 0 that measures X some other way than the arms are measured (other probe,
    other leaf distances, other generator), so equality with EXP-075 would not license the X
    side of the claims."""
    _fake_x(tmp_path / "x", 12)
    m = _fake_mint(tmp_path / "m", 12)
    rec = ev.gate0_cell(12, tmp_path / "m", tmp_path / "out", x_root=tmp_path / "x")
    x = cells.e75.load_judge("X", 12, tmp_path / "x")
    want = gate_l.leaf_rank_margin(x, m.probe, m.leaf_dist.__getitem__, 12, N_ACTIONS)
    on_disk = json.loads((tmp_path / "out" / ev.gate0_record_name(12)).read_text())
    assert on_disk["gate_l"] == json.loads(json.dumps(want))
    assert (on_disk["kind"], on_disk["seed"], on_disk["mint_sha"], on_disk["judge_updates"]) == (
        "gate0-X", 12, m.sha, 4000)
    assert rec["gate_l"] == want


def _spy_search(monkeypatch, n_states=7):
    calls = {}

    def fake_cell(depth, seed):
        calls["cell"] = (depth, seed)
        return "agent", "head", [SOLVED] * n_states, 5 + seed

    def fake_eval(agent, head, states, **kw):
        calls["eval"] = dict(kw, n=len(states), agent=agent, head=head)
        return {"success_rate": 0.5, "solved": 1, "n": len(states), "mode": kw["mode"],
                "k": kw["k"], "k_goal": kw["k_goal"], "no_revisit": kw["no_revisit"],
                "first_fire_len": {"4": 2}, "mean_steps": 3.0, "optimality": 0.9}

    monkeypatch.setattr(ev.e75, "load_cell", fake_cell)
    monkeypatch.setattr(ev, "evaluate_lookahead", fake_eval)
    return calls


def _trained_oxt(tmp_path, seed=12, update=4000):
    x = _fake_x(tmp_path / "x", seed)
    ck = cells.ckpt_dir("O-Xt", seed, tmp_path / "j")
    ck.mkdir(parents=True)
    torch.save(x.state_dict(), ck / "judge.pt")
    (ck / "state.json").write_text(json.dumps({"update": update}))
    return x


def test_search_cell_is_j4v_with_the_oxt_judge(tmp_path, monkeypatch):
    """Catches a search arm that is not EXP-076's J4V (mode C, k 4, goal check at 4,
    no-revisit, depth 11), runs another judge, or seeds its streams differently."""
    x = _trained_oxt(tmp_path)
    calls = _spy_search(monkeypatch)
    rec = ev.run_cell(12, tmp_path / "out", judge_dir=tmp_path / "j")
    kw = calls["eval"]
    assert calls["cell"] == (11, 12)
    assert (kw["depth"], kw["mode"], kw["k"], kw["k_goal"], kw["no_revisit"]) == (
        11, "C", 4, 4, True)
    assert (kw["rng_seed"], kw["imag_seed"], kw["n"]) == (17, 17, 7)
    assert torch.equal(kw["critic"].judge.sensory.fc1.weight, x.sensory.fc1.weight)
    on_disk = json.loads((tmp_path / "out" / ev.record_name(12)).read_text())
    assert (on_disk["arm"], on_disk["judge"], on_disk["depth"], on_disk["seed"],
            on_disk["judge_seed"], on_disk["judge_updates"], on_disk["fired4"],
            on_disk["limit_states"]) == ("J4V-O-Xt", "O-Xt", 11, 12, 12, 4000, 2, None)
    assert rec["success_rate"] == 0.5


def test_timing_cell_runs_a_pilot_judge_on_another_seeds_cell_and_is_marked(tmp_path, monkeypatch):
    """Catches a timing record that could pass for a claim record: it must name both seeds and
    carry limit_states, which the aggregator refuses."""
    _trained_oxt(tmp_path, seed=12)
    calls = _spy_search(monkeypatch, n_states=30)
    rec = ev.run_cell(0, tmp_path / "out", judge_dir=tmp_path / "j", judge_seed=12,
                      limit_states=20, name=ev.time_record_name(12, 0))
    assert calls["cell"] == (11, 0) and calls["eval"]["n"] == 20
    assert (rec["seed"], rec["judge_seed"], rec["limit_states"]) == (0, 12, 20)
    assert (tmp_path / "out" / "exp080_time_J4V-O-Xt_j12_c0.json").exists()
    assert not (tmp_path / "out" / ev.record_name(0)).exists()


def test_search_refuses_an_unfinished_oxt_judge(tmp_path, monkeypatch):
    """Catches a search cell evaluated from a partly trained judge."""
    _trained_oxt(tmp_path, update=250)
    _spy_search(monkeypatch)
    with pytest.raises(SystemExit, match="update 250"):
        ev.run_cell(12, tmp_path / "out", judge_dir=tmp_path / "j")


def test_pending_search_skips_finished_cells(tmp_path):
    """Catches a relaunch that re-runs finished search cells."""
    (tmp_path / ev.record_name(3)).write_text("{}")
    assert ev.pending_search([2, 3, 4], tmp_path) == [2, 4]
