"""EXP-079 cells: judge S is EXP-075's X recipe trained on walks of length 1 to 40.

S starts each evaluation seed from EXP-075's committed pretrained X encoder and is built exactly
as X was (wide readout, hidden 512, head seeded by the seed). Only the walk length differs.
Pilot seeds 12 and 13 read their encoders from this experiment's outputs_pilot/encoders/, filled
either by copying EXP-075's laptop pilot encoders once each is proved to be the one its pilot
record describes, or by re-pretraining with EXP-075's pretrain.py.

Spec: docs/superpowers/specs/2026-10-18-exp079-judge-trained-where-scrambles-live-design.md,
sections 3, 4 and 6.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python experiments/079_judge_trained_deep/cells.py prepare-pilot-encoders
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


c75 = _load("exp075_cells", REPO / "experiments" / "075_wide_region" / "cells.py")

from neuromorphic.training import value_iteration as vi  # noqa: E402

ARM = "S"
# S is built at X's width, from X's pretrained encoder, by X's construction.
BASE_ARM = "X"
# Spec section 3: the only lever. EXP-075's X trained on walks of length 1 to 14.
MAX_LEN = 40
N_UPDATES = 4000
EVAL_SEEDS = tuple(c75.EVAL_SEEDS)
PILOT_SEEDS = tuple(c75.PILOT_SEEDS)
E75_OUT = REPO / "experiments" / "075_wide_region" / "outputs"
E75_PILOT = REPO / "experiments" / "075_wide_region" / "outputs_pilot"
E76_OUT = REPO / "experiments" / "076_deeper_tree" / "outputs"
E78_OUT = REPO / "experiments" / "078_full_five_move_tree" / "outputs"
OUT = HERE / "outputs"
PILOT_OUT = HERE / "outputs_pilot"
# Same machine, same float code path: a genuine match reproduces the drift to rounding.
DRIFT_TOL = 1e-6

check_seeds = c75.check_seeds
exclusion_set = c75.exclusion_set
probe_set = c75.probe_set


def encoder_root(pilot: bool) -> Path:
    """Where a seed's pretrained X encoder lives (as `<root>/encoders/enc_X_s<seed>.pt`):
    EXP-075's committed outputs for evaluation seeds, this experiment's outputs_pilot for the
    pilot seeds."""
    return PILOT_OUT if pilot else E75_OUT


def ckpt_dir(seed, out_dir=None) -> Path:
    return (Path(out_dir) if out_dir is not None else OUT) / f"judge_S_s{seed}"


def record_name(seed) -> str:
    return f"exp079_train_S_s{seed}.json"


def make_judge(seed, enc_root) -> vi.Judge:
    """S before training: X's pretrained region for `seed` under a fresh wide head, seeded as
    X's was. A missing encoder raises SystemExit naming the file."""
    return c75.build_judge(seed, BASE_ARM, c75.load_pretrained(BASE_ARM, seed, enc_root))


def judge_updates(seed, judge_dir=None) -> int:
    path = ckpt_dir(seed, judge_dir) / "state.json"
    if not path.exists():
        raise SystemExit(f"missing {path} for S seed {seed}: cannot tell how far the judge "
                         f"trained; train it with train.py first")
    return int(json.loads(path.read_text())["update"])


def load_trained(seed, judge_dir=None, required_updates=N_UPDATES) -> vi.Judge:
    """A trained S judge from its checkpoint, refused unless training reached
    `required_updates` (judge.pt is written at update 0, so its existence proves nothing)."""
    path = ckpt_dir(seed, judge_dir) / "judge.pt"
    if not path.exists():
        raise SystemExit(f"missing judge checkpoint {path} for S seed {seed}; train it with "
                         f"train.py first")
    done = judge_updates(seed, judge_dir)
    if done != required_updates:
        raise SystemExit(f"{path.parent / 'state.json'}: S seed {seed} is at update {done}, "
                         f"evaluation needs {required_updates}; finish training first")
    judge = c75.build_judge(seed, BASE_ARM, c75.fresh_sensory(BASE_ARM, seed))
    judge.load_state_dict(torch.load(path, map_location="cpu"))
    judge.eval()
    return judge


def drift(sensory_state: dict, e0_state: dict) -> float:
    """EXP-075's encoder_drift: the L2 distance over the pretrained encoder's tensors."""
    return sum(float(((sensory_state[k] - e0_state[k]) ** 2).sum()) for k in e0_state) ** 0.5


def pilot_encoder_matches(seed, e75_pilot_dir=E75_PILOT) -> tuple[bool, str]:
    """Whether EXP-075's pilot X encoder for `seed` is the one its pilot record describes:
    recomputing the drift between the X pilot judge's region and that encoder must reproduce
    the record's encoder_drift within DRIFT_TOL."""
    d = Path(e75_pilot_dir)
    enc = c75.encoder_path(BASE_ARM, seed, d)
    jpt = c75.ckpt_dir(BASE_ARM, seed, d) / "judge.pt"
    rec = d / c75.record_name(BASE_ARM, seed)
    for p in (enc, jpt, rec):
        if not p.exists():
            return False, f"seed {seed}: missing {p}"
    e0 = c75.load_pretrained(BASE_ARM, seed, d).state_dict()
    judge = c75.build_judge(seed, BASE_ARM, c75.fresh_sensory(BASE_ARM, seed))
    judge.load_state_dict(torch.load(jpt, map_location="cpu"))
    got = drift(judge.sensory.state_dict(), e0)
    want = json.loads(rec.read_text())["encoder_drift"]
    if abs(got - want) > DRIFT_TOL:
        return False, f"seed {seed}: recomputed drift {got!r} against the record's {want!r}"
    return True, f"seed {seed}: drift {got!r} matches the record"


def prepare_pilot_encoders(e75_pilot_dir=E75_PILOT, dest_root=PILOT_OUT) -> list[str]:
    """Copy EXP-075's pilot X encoders to `dest_root/encoders/`, all or nothing: every seed is
    proved first, and one failure copies none (SystemExit naming each failure)."""
    proofs = [pilot_encoder_matches(s, e75_pilot_dir) for s in PILOT_SEEDS]
    bad = [why for ok, why in proofs if not ok]
    if bad:
        raise SystemExit("pilot encoders not proved: " + "; ".join(bad)
                         + ". Nothing was copied. Run the launcher's pilot-pretrain phase.")
    lines = []
    for s, (_, why) in zip(PILOT_SEEDS, proofs):
        dest = c75.encoder_path(BASE_ARM, s, dest_root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(c75.encoder_path(BASE_ARM, s, e75_pilot_dir), dest)
        lines.append(f"{why}; copied to {dest}")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("prepare-pilot-encoders",))
    ap.add_argument("--e75-pilot-dir", type=Path, default=E75_PILOT)
    ap.add_argument("--dest-root", type=Path, default=PILOT_OUT)
    args = ap.parse_args()
    for line in prepare_pilot_encoders(args.e75_pilot_dir, args.dest_root):
        print(line)


if __name__ == "__main__":
    main()
