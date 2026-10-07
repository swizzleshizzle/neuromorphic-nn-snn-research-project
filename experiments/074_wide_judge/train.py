"""EXP-074 training driver: arms W, A and B by value iteration, then Gate L on the final judge.

Builds the full BFS table once per process (about 65 s) for the exclusion and probe sets and as
Gate L's yardstick. Training itself never reads it.

Spec: docs/superpowers/specs/2026-10-07-exp074-wide-judge-design.md, sections 4 to 6.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/074_wide_judge/train.py --pilot --arms W A B --seeds 12 13 \
        --n-updates 4000 --out-dir experiments/074_wide_judge/outputs_pilot --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp074_cells", "cells.py")
gate_l = _load("exp074_gate_l", "gate_l.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

torch.set_num_threads(1)

MAX_LEN = 14  # 2x2 quarter-turn diameter (spec EXP-073 section 3.1)
OUT_DIR_DEFAULT = HERE / "outputs"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def check_seeds(seeds, pilot: bool) -> None:
    allowed = cells.PILOT_SEEDS if pilot else cells.EVAL_SEEDS
    bad = [s for s in seeds if s not in allowed]
    if bad:
        raise SystemExit(f"seeds {bad} not allowed with pilot={pilot}; allowed seeds are {allowed}")


def _encoder_drift(judge, seed: int) -> float:
    e1 = torch.load(cells.c70.published_config(7, seed).encoder_state_path, map_location="cpu")
    cur = judge.sensory.state_dict()
    return sum(float(((cur[k] - e1[k]) ** 2).sum()) for k in e1) ** 0.5


def run(arm, seed, n_updates, batch, sync_every, probe_every, draws, out_dir) -> dict:
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    provider = ExactBFSDistance(max_depth=None)
    excl = cells.exclusion_set(seed, provider)
    probe = cells.probe_set(seed, provider)
    judge = cells.make_judge(seed, arm)
    result = vi.train_judge(
        judge, arm, n_updates=n_updates, batch=batch, max_len=MAX_LEN, n_actions=N_ACTIONS,
        exclude=excl, probe=probe, seed=seed, sync_every=sync_every, probe_every=probe_every,
        draws=draws, ckpt_dir=cells.ckpt_dir(arm, seed, out_dir),
    )
    judge.eval()
    rec = {
        **result,
        "arm": arm, "seed": seed, "readout": judge.readout,
        "n_updates": n_updates, "batch": batch, "max_len": MAX_LEN, "n_actions": N_ACTIONS,
        "sync_every": sync_every, "probe_every": probe_every, "draws": draws,
        "n_exclude": len(excl), "n_probe": len(probe),
        "encoder_drift": _encoder_drift(judge, seed),
        "gate_l": gate_l.leaf_rank_margin(judge, probe, provider.distance, seed, N_ACTIONS),
        "git_commit": _git_commit(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / cells.record_name(arm, seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(cells.ARMS_TRAIN))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--n-updates", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--sync-every", type=int, default=100)
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    bad_arms = [a for a in args.arms if a not in cells.ARMS_TRAIN]
    if bad_arms:
        raise SystemExit(f"arms {bad_arms} outside {cells.ARMS_TRAIN}")
    check_seeds(args.seeds, args.pilot)
    jobs = [(arm, seed) for arm in args.arms for seed in args.seeds]
    print(f"EXP-074: {len(jobs)} training runs, {args.workers} workers, "
          f"pilot={args.pilot}, n_updates={args.n_updates}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, arm, seed, args.n_updates, args.batch, args.sync_every,
                            args.probe_every, args.draws, args.out_dir): (arm, seed)
                for arm, seed in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  arm {r['arm']} seed {r['seed']}  updates {r['updates']}  "
                  f"loss_last {r['loss_last']:.6f}  wall_s {r['wall_s']:.1f}  "
                  f"encoder_drift {r['encoder_drift']:.6f}  "
                  f"gate_l margin {r['gate_l']['margin']:.4f}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
