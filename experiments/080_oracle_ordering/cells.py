"""EXP-080 cells: three judges trained on the TRUE distance, as instruments for whether X's
encoder can carry far ordering. None is a candidate system; distance-to-solved is never a model
input in the method, and these arms exist to price it.

O-raw is a plain MLP on one-hot facelets (the protocol check). O-Xf is EXP-075's committed X
region, frozen, under a fresh wide head. O-Xt is EXP-079's S recipe (X's pretrained encoder, a
fresh wide head, arm W's optimizer) with the true distance as its target.

Spec: docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md, section 3.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c75 = _load("exp075_cells", REPO / "experiments" / "075_wide_region" / "cells.py")
e75 = _load("exp075_evaluate", REPO / "experiments" / "075_wide_region" / "evaluate.py")
c79 = _load("exp079_cells", REPO / "experiments" / "079_judge_trained_deep" / "cells.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402
from neuromorphic.training.lookahead import tree_levels  # noqa: E402

ARMS = ("O-raw", "O-Xf", "O-Xt")
# Longest first, so a pool's last wave is the cheap arms.
TRAIN_ORDER = ("O-Xt", "O-Xf", "O-raw")
EVAL_SEEDS = tuple(c75.EVAL_SEEDS)
PILOT_SEEDS = tuple(c75.PILOT_SEEDS)
MAX_LEN = 40
N_UPDATES = 4000
BATCH = 1000
PROBE_EVERY = 250
RAW_STATES = 600_000
RAW_EPOCHS = 12
RAW_BATCH = 512
RAW_HIDDEN = 512
N_COLOURS = 6
N_FACELETS = 24
# Gate L's tree: the leaves leaf_rank_margin ranks, which training must never see.
LEAF_K, LEAF_LO, LEAF_HI = 3, 7, 11

E75_OUT = REPO / "experiments" / "075_wide_region" / "outputs"
E75_PILOT = REPO / "experiments" / "075_wide_region" / "outputs_pilot"
E76_OUT = REPO / "experiments" / "076_deeper_tree" / "outputs"
E79_OUT = REPO / "experiments" / "079_judge_trained_deep" / "outputs"
OUT = HERE / "outputs"
PILOT_OUT = HERE / "outputs_pilot"

check_seeds = c75.check_seeds
exclusion_set = c75.exclusion_set
probe_set = c75.probe_set
drift = c79.drift


def x_judge_root(pilot: bool) -> Path:
    """Where EXP-075's trained X judges live: committed outputs, or the laptop's pilot folder."""
    return E75_PILOT if pilot else E75_OUT


def encoder_root(pilot: bool) -> Path:
    """Where X's pretrained encoders live, exactly as EXP-079 read them (its pilot copies were
    proved to be EXP-075's pilot encoders)."""
    return c79.encoder_root(pilot)


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def ckpt_dir(arm: str, seed: int, out_dir=None) -> Path:
    _check_arm(arm)
    return (Path(out_dir) if out_dir is not None else OUT) / f"judge_{arm}_s{seed}"


def record_name(arm: str, seed: int) -> str:
    _check_arm(arm)
    return f"exp080_train_{arm}_s{seed}.json"


def leaf_exclusion(probe, n_actions=N_ACTIONS, k=LEAF_K, lo=LEAF_LO, hi=LEAF_HI) -> set:
    """Every k-move leaf of every probe state at true distance lo..hi: the states
    leaf_rank_margin ranks. With true labels, a judge trained on one would have seen the answer."""
    out: set = set()
    for s, d in probe:
        if lo <= d <= hi:
            out.update(tree_levels(s, k, n_actions)[k])
    return out


def oracle_exclusion(seed: int, provider, probe=None) -> set:
    """Held-out states, probe states, and the probe leaves (spec section 3)."""
    probe = probe_set(seed, provider) if probe is None else probe
    return exclusion_set(seed, provider) | {s for s, _ in probe} | leaf_exclusion(probe)


class RawJudge(nn.Module):
    """O-raw: an MLP on one-hot facelets. `forward` takes and ignores a generator, so
    leaf_rank_margin and probe_judge call it as they call a spiking judge."""

    def __init__(self, hidden: int = RAW_HIDDEN):
        super().__init__()
        n_in = N_FACELETS * N_COLOURS
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 1))

    @staticmethod
    def one_hot(states) -> torch.Tensor:
        idx = torch.as_tensor(np.asarray(states), dtype=torch.long)
        return F.one_hot(idx, N_COLOURS).flatten(1).float()

    def forward(self, states, generator=None) -> torch.Tensor:
        return self.net(self.one_hot(states)).squeeze(-1)


def make_oraw(seed: int) -> RawJudge:
    torch.manual_seed(seed)
    return RawJudge()


def make_oxf(seed: int, x_root) -> vi.Judge:
    """X's committed trained region (refused short of 4000 updates), frozen, under the fresh wide
    head `c75.build_judge` seeds: the same head O-Xt and S start from."""
    x = e75.load_judge("X", seed, x_root)
    judge = c75.build_judge(seed, "X", c75.fresh_sensory("X", seed))
    judge.sensory.load_state_dict(x.sensory.state_dict())
    for p in judge.sensory.parameters():
        p.requires_grad_(False)
    return judge


def make_oxt(seed: int, enc_root) -> vi.Judge:
    """EXP-079's S before training: X's pretrained encoder under a fresh wide head."""
    return c79.make_judge(seed, enc_root)


def make_judge(arm: str, seed: int, x_root, enc_root):
    _check_arm(arm)
    if arm == "O-raw":
        return make_oraw(seed)
    if arm == "O-Xf":
        return make_oxf(seed, x_root)
    return make_oxt(seed, enc_root)


def optimizer_for(arm: str, judge):
    if arm == "O-raw":
        return torch.optim.Adam(judge.parameters(), lr=1e-3)
    if arm == "O-Xf":
        return vi.make_optimizer(judge, "B")
    if arm == "O-Xt":
        return vi.make_optimizer(judge, "W")
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def reference_sensory(arm: str, seed: int, x_root, enc_root) -> dict:
    """The region state an arm starts from, which its encoder_drift is measured against."""
    if arm == "O-Xf":
        return e75.load_judge("X", seed, x_root).sensory.state_dict()
    if arm == "O-Xt":
        return c75.load_pretrained("X", seed, enc_root).state_dict()
    raise ValueError(f"{arm!r} has no region")


def load_trained(arm: str, seed: int, judge_dir=None, required=None):
    """A trained judge from its checkpoint, refused unless training finished: O-raw at
    `required` epochs (default RAW_EPOCHS), O-Xf and O-Xt at `required` updates (default
    N_UPDATES)."""
    ck = ckpt_dir(arm, seed, judge_dir)
    path, state_path = ck / "judge.pt", ck / "state.json"
    for p in (path, state_path):
        if not p.exists():
            raise SystemExit(f"missing {p} for {arm} seed {seed}; train it with train.py first")
    state = json.loads(state_path.read_text())
    if arm == "O-raw":
        want = RAW_EPOCHS if required is None else required
        if state.get("epochs") != want:
            raise SystemExit(f"{state_path}: {arm} seed {seed} is at epochs {state.get('epochs')}, "
                             f"needs {want}; finish training first")
        judge = RawJudge()
    else:
        want = N_UPDATES if required is None else required
        if state.get("update") != want:
            raise SystemExit(f"{state_path}: {arm} seed {seed} is at update {state.get('update')}, "
                             f"needs {want}; finish training first")
        judge = c75.build_judge(seed, "X", c75.fresh_sensory("X", seed))
    judge.load_state_dict(torch.load(path, map_location="cpu"))
    judge.eval()
    return judge
