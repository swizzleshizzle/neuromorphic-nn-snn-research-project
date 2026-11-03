"""EXP-085 pre-flight (spec section 10).

--part gpu (desktop): rebase seed 0 arm C twice into temporary folders, train each 100 updates on
the device, and record whether the two judges are tensor-equal (it decides whether Gate H (b) is
available) and the cost per update.

--part walks (VPS): the share of each arm's training states by true distance at updates 16,000,
18,000 and 20,000, from 20,000 walks per row through the real walk generator. Instrument use
only: true distance never reaches training.

Usage (repo root, PYTHONPATH=src):
    python -u experiments/085_outward_walks/preflight.py --part gpu --device cuda
    .venv/bin/python -u experiments/085_outward_walks/preflight.py --part walks
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import tempfile
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rebase = _load("exp085_rebase", HERE / "rebase.py")
gatec = rebase.gatec
cells = rebase.cells
g81 = rebase.g81

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402
from neuromorphic.training.device import setup_device  # noqa: E402

N_PREFLIGHT = 100
N_WALKS = 20000
WALK_UPDATES = (16000, 18000, 20000)
BANDS = ((0, 8), (9, 9), (10, 10), (11, 11), (12, 14))


def _default_train(ck: Path, seed: int, device, n: int) -> dict:
    train = _load("exp085_train", HERE / "train.py")
    exclude, probe, _ = gatec.sets(seed)
    judge, _ = train.train_to(ck, "C", seed, device, cells.START + n, exclude=exclude,
                              probe=probe, log=lambda m: None)
    return {k: v.detach().cpu() for k, v in judge.state_dict().items()}


def gpu(device, out=cells.OUT, *, seed=0, train_fn=None) -> dict:
    out = Path(out)
    train_fn = train_fn or _default_train
    sds, walls = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for k in ("a", "b"):
            ck = Path(tmp) / k
            rebase.write_rebased(cells.start_dir(seed, out), ck, seed, device)
            t0 = time.time()
            sds.append(train_fn(ck, seed, device, N_PREFLIGHT))
            walls.append(time.time() - t0)
    a, b = sds
    equal = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    rec = {"part": "gpu", "device": str(device), "seed": seed, "arm": "C",
           "n_updates": N_PREFLIGHT, "deterministic": bool(equal),
           "s_per_update": min(walls) / N_PREFLIGHT, "git_commit": g81._git_commit()}
    g81._write_json(out / cells.PREFLIGHT_NAME, rec)
    return rec


def walks(out=None, *, exclude=None, seed=0) -> dict:
    provider = ExactBFSDistance(max_depth=None)
    if exclude is None:
        exclude = gatec.sets(seed)[0]
    rows = []
    for arm in cells.ARMS:
        sched = cells.max_len_schedule(arm)
        for u in WALK_UPDATES:
            states = vi.random_walk_states(N_WALKS, sched(u), random.Random(cells.H_SEED),
                                           N_ACTIONS, exclude)
            ds = [provider.distance(s) for s in states]
            shares = [round(sum(lo <= d <= hi for d in ds) / len(ds), 4) for lo, hi in BANDS]
            rows.append({"arm": arm, "update": u, "max_len": sched(u), "shares": shares})
    rec = {"part": "walks", "bands": [list(b) for b in BANDS], "n_walks": N_WALKS,
           "seed": seed, "rows": rows, "git_commit": g81._git_commit()}
    if out is not None:
        g81._write_json(Path(out) / cells.PREFLIGHT_WALKS_NAME, rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("gpu", "walks"), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    if args.part == "gpu":
        if (args.out_dir / cells.PREFLIGHT_NAME).exists():
            print("pre-flight gpu: already recorded", flush=True)
            return
        setup_device(args.device)
        r = gpu(args.device, args.out_dir)
        print(f"pre-flight gpu: deterministic {r['deterministic']}, "
              f"{r['s_per_update']:.3f} s/update", flush=True)
    else:
        r = walks(args.out_dir)
        for row in r["rows"]:
            print(f"  {row['arm']} u{row['update']} max_len {row['max_len']} {row['shares']}")
    print("done.")


if __name__ == "__main__":
    main()
