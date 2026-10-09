"""EXP-075 pretraining driver: EXP-039's inverse-model recipe at each arm's width.

Unchanged from EXP-039 except (spec section 4): the region's hidden width, and a forbidden set
that adds every state in the seed's evaluation held-out sets (`cells.exclusion_set`) to EXP-039's
probe held-out states. Self-supervised: the move is known because it was applied; no distance
label enters. The BFS table only builds the state and forbidden sets.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/075_wide_region/pretrain.py --pilot --seeds 12 13 \
        --out-dir experiments/075_wide_region/outputs_pilot --workers 4
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
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp075_cells", HERE / "cells.py")
e39 = _load("exp039_run", REPO / "experiments" / "039_encoder_pretraining" / "run.py")

from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training.encoder_pretrain import (  # noqa: E402
    PretrainConfig, build_pairs, save_encoder, train_inverse_model,
)

torch.set_num_threads(1)

EPOCHS = 40
BATCH = 256
LR = 3e-3
DEPTHS = e39.DEPTHS
FRAC_HELDOUT = e39.FRAC_HELDOUT
OUT_DIR_DEFAULT = HERE / "outputs"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def pretrain_states(provider7):
    """Every state at EXP-039's depths with its depth, in `e39.build_dataset`'s order (without
    the optimal-move masks, which only EXP-039's probe needs)."""
    states, depths = [], []
    for d in DEPTHS:
        for s in provider7.states_at_distance(d):
            states.append(s)
            depths.append(d)
    return states, depths


def probe_heldout(states, depths, seed) -> set:
    """EXP-039's probe held-out states for `seed`, stratified by depth exactly as its run_one."""
    held = set()
    for d in DEPTHS:
        pool = [i for i, dd in enumerate(depths) if dd == d]
        _train, he = e39.probe.split_states(pool, FRAC_HELDOUT, seed)
        held.update(states[i] for i in he)
    return held


def pretrain_pairs(seed, states, depths, excl):
    """(pairs, forbidden): EXP-039's pairs with `excl` also forbidden at both endpoints."""
    forbidden = probe_heldout(states, depths, seed) | set(excl)
    pairs = build_pairs(states, forbidden=forbidden)
    assert not (forbidden & {p[0] for p in pairs}), "forbidden state leaked in as a source"
    assert not (forbidden & {p[2] for p in pairs}), "forbidden state leaked in as a successor"
    return pairs, forbidden


def run(arm, seed, out_dir, epochs=EPOCHS) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    out_dir = Path(out_dir)
    # One depth beyond the deepest source state, as EXP-039: successors of depth-6 states are
    # depth 7. The full table builds the evaluation exclusion set (depths 7, 8, 9, 11).
    prov7 = ExactBFSDistance(max_depth=max(DEPTHS) + 1)
    excl = cells.exclusion_set(seed, ExactBFSDistance(max_depth=None))
    states, depths = pretrain_states(prov7)
    pairs, forbidden = pretrain_pairs(seed, states, depths, excl)
    n39 = len(build_pairs(states, forbidden=probe_heldout(states, depths, seed)))
    cfg = PretrainConfig(seed=seed, epochs=epochs, batch_size=BATCH, lr=LR,
                         hidden=cells.HIDDEN[arm])
    res = train_inverse_model(pairs, cfg)
    path = cells.encoder_path(arm, seed, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_encoder(res.sensory, path)
    rec = {
        "arm": arm, "seed": seed, "hidden": cells.HIDDEN[arm], "epochs": epochs,
        "batch_size": BATCH, "lr": LR, "n_pairs": len(pairs), "n_pairs_exp039_recipe": n39,
        "n_forbidden": len(forbidden), "history": res.history,
        "final_move_accuracy": res.final_accuracy, "wall_s": round(time.time() - t0, 1),
        "git_commit": _git_commit(),
    }
    (out_dir / cells.pretrain_record_name(arm, seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def pending_jobs(arms, seeds, out_dir) -> list[tuple[str, int]]:
    """(arm, seed) pairs, arms then seeds, whose pretraining record does not exist yet."""
    return [(a, s) for a in arms for s in seeds
            if not (Path(out_dir) / cells.pretrain_record_name(a, s)).exists()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(cells.ARMS_TRAIN))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    bad = [a for a in args.arms if a not in cells.ARMS_TRAIN]
    if bad:
        raise SystemExit(f"arms {bad} outside {cells.ARMS_TRAIN}")
    cells.check_seeds(args.seeds, args.pilot)
    jobs = pending_jobs(args.arms, args.seeds, args.out_dir)
    print(f"EXP-075 pretraining: {len(jobs)} runs, {args.workers} workers, pilot={args.pilot}, "
          f"epochs={args.epochs}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, a, s, args.out_dir, args.epochs): (a, s) for a, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  arm {r['arm']} seed {r['seed']}  hidden {r['hidden']}  "
                  f"pairs {r['n_pairs']}/{r['n_pairs_exp039_recipe']}  "
                  f"move-acc {r['final_move_accuracy']:.4f}  wall_s {r['wall_s']:.0f}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
