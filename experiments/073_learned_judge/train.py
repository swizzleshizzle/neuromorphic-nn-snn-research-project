"""EXP-073: the pilot and full training driver for the learned judge.

Builds the full BFS table once per process (about 65 s), the seed's exclusion set and probe set,
then runs `value_iteration.train_judge` and writes one record per (arm, seed). Seeds 12 and 13
are pilot-only; seeds 0 to 11 are evaluation-only, and `--pilot` is the switch between the two
allowed seed sets, never the seed value itself.

Spec: docs/superpowers/specs/2026-10-07-exp073-learned-judge-design.md, sections 3 and 4.

Usage (from the repo root, PYTHONPATH=src set):
    .venv/bin/python experiments/073_learned_judge/train.py --pilot --arms A --seeds 12 \
        --n-updates 4 --batch 32 --sync-every 2 --probe-every 2 --workers 1 \
        --out-dir /root/scratch/exp073-smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp073_cells", HERE / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

torch.set_num_threads(1)

# 2x2 quarter-turn diameter (God's number): the longest a random walk from solved needs to be to
# reach any reachable state. Fixed, not a CLI knob: the spec names 14 explicitly (section 3.1).
MAX_LEN = 14

OUT_DIR_DEFAULT = HERE / "outputs"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def check_seeds(seeds, pilot: bool) -> None:
    """Refuse a pilot run on an evaluation seed, or an evaluation run on a pilot seed."""
    allowed = cells.PILOT_SEEDS if pilot else cells.EVAL_SEEDS
    bad = [s for s in seeds if s not in allowed]
    if bad:
        raise SystemExit(
            f"seeds {bad} not allowed with pilot={pilot}; allowed seeds are {allowed}")


def _encoder_drift(judge, seed: int) -> float:
    """L2 norm of `judge.sensory`'s parameters minus the E1 state dict it was loaded from.

    Zero for arm B (the encoder never moves); nonzero for arm A if and only if a gradient
    actually reached the encoder (Gate E, EXP-073's version of the EXP-047 trap).
    """
    e1 = torch.load(cells.c70.published_config(7, seed).encoder_state_path, map_location="cpu")
    cur = judge.sensory.state_dict()
    total_sq = sum(float(((cur[k] - e1[k]) ** 2).sum()) for k in e1)
    return total_sq ** 0.5


def run(arm, seed, n_updates, batch, sync_every, probe_every, draws, out_dir) -> dict:
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    provider = ExactBFSDistance(max_depth=None)
    excl = cells.exclusion_set(seed, provider)
    probe = cells.probe_set(seed, provider)
    judge = cells.make_judge(seed)
    ckpt = out_dir / f"judge_{arm}_s{seed}"

    result = vi.train_judge(
        judge, arm, n_updates=n_updates, batch=batch, max_len=MAX_LEN, n_actions=N_ACTIONS,
        exclude=excl, probe=probe, seed=seed, sync_every=sync_every, probe_every=probe_every,
        draws=draws, ckpt_dir=ckpt,
    )

    rec = {
        **result,
        "arm": arm, "seed": seed,
        "n_updates": n_updates, "batch": batch, "max_len": MAX_LEN, "n_actions": N_ACTIONS,
        "sync_every": sync_every, "probe_every": probe_every, "draws": draws,
        "n_exclude": len(excl), "n_probe": len(probe),
        "encoder_drift": _encoder_drift(judge, seed),
        "git_commit": _git_commit(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cells.record_name(arm, seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(cells.ARMS_TRAIN))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--n-updates", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--sync-every", type=int, default=500)
    ap.add_argument("--probe-every", type=int, default=500)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    args = ap.parse_args()

    bad_arms = [a for a in args.arms if a not in cells.ARMS_TRAIN]
    if bad_arms:
        raise SystemExit(f"arms {bad_arms} outside {cells.ARMS_TRAIN}")
    check_seeds(args.seeds, args.pilot)

    jobs = [(arm, seed) for arm in args.arms for seed in args.seeds]
    print(f"EXP-073: {len(jobs)} training runs, {args.workers} workers, "
          f"pilot={args.pilot}, n_updates={args.n_updates}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(run, arm, seed, args.n_updates, args.batch, args.sync_every,
                       args.probe_every, args.draws, args.out_dir): (arm, seed)
            for arm, seed in jobs
        }
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  arm {r['arm']} seed {r['seed']}  "
                  f"updates {r['updates']}  loss_last {r['loss_last']:.6f}  "
                  f"wall_s {r['wall_s']:.1f}  encoder_drift {r['encoder_drift']:.6f}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
