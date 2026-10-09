"""EXP-075 training driver: arms X and Y by EXP-074's value iteration, then Gate L.

Each run starts from its arm's pretrained encoder in `out_dir/encoders/` (pretrain.py). Builds
the full BFS table once per process for the exclusion and probe sets and as Gate L's yardstick;
training never reads it.

Spec: docs/superpowers/specs/2026-10-09-exp075-wide-sensory-region-design.md, sections 5 to 7.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/075_wide_region/train.py --pilot --seeds 12 13 \
        --n-updates 4000 --out-dir experiments/075_wide_region/outputs_pilot --workers 4
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
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp075_cells", HERE / "cells.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

torch.set_num_threads(1)

MAX_LEN = 14  # 2x2 quarter-turn diameter
# `vi.make_optimizer` names arms by how their encoder is treated. X and Y train the encoder
# exactly as EXP-074's W did (encoder lr 1e-4, head lr 1e-3), so they use W's optimizer.
VI_ARM = "W"
OUT_DIR_DEFAULT = HERE / "outputs"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _encoder_drift(judge, arm, seed, out_dir) -> float:
    e0 = cells.load_pretrained(arm, seed, out_dir).state_dict()
    cur = judge.sensory.state_dict()
    return sum(float(((cur[k] - e0[k]) ** 2).sum()) for k in e0) ** 0.5


def _resumed_from(ckpt_dir) -> int:
    """Updates already done when this session starts (0 for a fresh run). train_judge's wall_s
    covers only the session that wrote the record, so s/update needs this."""
    path = Path(ckpt_dir) / "state.json"
    return int(json.loads(path.read_text())["update"]) if path.exists() else 0


def run(arm, seed, n_updates, batch, sync_every, probe_every, draws, out_dir) -> dict:
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    judge = cells.make_judge(seed, arm, out_dir)  # first: a missing encoder fails before the BFS
    provider = ExactBFSDistance(max_depth=None)
    excl = cells.exclusion_set(seed, provider)
    probe = cells.probe_set(seed, provider)
    resumed_from = _resumed_from(cells.ckpt_dir(arm, seed, out_dir))
    result = vi.train_judge(
        judge, VI_ARM, n_updates=n_updates, batch=batch, max_len=MAX_LEN, n_actions=N_ACTIONS,
        exclude=excl, probe=probe, seed=seed, sync_every=sync_every, probe_every=probe_every,
        draws=draws, ckpt_dir=cells.ckpt_dir(arm, seed, out_dir),
    )
    judge.eval()
    rec = {
        **result,
        "arm": arm, "seed": seed, "hidden": cells.HIDDEN[arm], "readout": judge.readout,
        "n_updates": n_updates, "resumed_from": resumed_from, "batch": batch, "max_len": MAX_LEN, "n_actions": N_ACTIONS,
        "sync_every": sync_every, "probe_every": probe_every, "draws": draws,
        "n_exclude": len(excl), "n_probe": len(probe),
        "encoder_drift": _encoder_drift(judge, arm, seed, out_dir),
        "gate_l": gate_l.leaf_rank_margin(judge, probe, provider.distance, seed, N_ACTIONS),
        "git_commit": _git_commit(),
    }
    (out_dir / cells.record_name(arm, seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def pending_jobs(arms, seeds, out_dir) -> list[tuple[str, int]]:
    """(arm, seed) pairs, SEEDS then arms, whose record does not exist yet. Seed-major so the
    first wave mixes X (heavy) and Y (light) workers instead of starting with every X run."""
    return [(a, s) for s in seeds for a in arms
            if not (Path(out_dir) / cells.record_name(a, s)).exists()]


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
    bad = [a for a in args.arms if a not in cells.ARMS_TRAIN]
    if bad:
        raise SystemExit(f"arms {bad} outside {cells.ARMS_TRAIN}")
    cells.check_seeds(args.seeds, args.pilot)
    jobs = pending_jobs(args.arms, args.seeds, args.out_dir)
    print(f"EXP-075: {len(jobs)} training runs, {args.workers} workers, pilot={args.pilot}, "
          f"n_updates={args.n_updates}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, a, s, args.n_updates, args.batch, args.sync_every,
                            args.probe_every, args.draws, args.out_dir): (a, s) for a, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            loss = "None" if r["loss_last"] is None else format(r["loss_last"], ".6f")
            print(f"  {i}/{len(jobs)}  arm {r['arm']} seed {r['seed']}  updates {r['updates']}  "
                  f"loss_last {loss}  wall_s {r['wall_s']:.1f}  "
                  f"encoder_drift {r['encoder_drift']:.6f}  "
                  f"gate_l margin {r['gate_l']['margin']:.4f}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
