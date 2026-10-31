"""EXP-084 Gate T: retrain X from scratch on the device to EXP-075's 4000 updates, with EXP-075's
arm, batch, walk length, exclusion set, sync and probe schedule, starting from EXP-075's
pretrained encoder. Checkpoints go ONLY under this experiment's outputs; EXP-075's tracked judges
are read for nothing and never written.

Spec: docs/superpowers/specs/2026-10-31-exp084-gpu-port-design.md, section 5.

Usage (desktop, repo root, PYTHONPATH=src):
    python -u experiments/084_gpu_port/train.py --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp084_cells", HERE / "cells.py")
t75 = _load("exp075_train", cells.E75 / "train.py")
c75 = t75.cells
gate_l = t75.gate_l

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402
from neuromorphic.training.device import setup_device  # noqa: E402

SYNC_EVERY = 100
PROBE_EVERY = 250
DRAWS = 1


def _sets(seed):
    """(exclusion set, probe set, distance function), from one full BFS table."""
    provider = ExactBFSDistance(max_depth=None)
    return c75.exclusion_set(seed, provider), c75.probe_set(seed, provider), provider.distance


def run(seed, device, *, n_updates=cells.N_UPDATES, batch=1000, out=cells.OUT) -> dict:
    torch.set_num_threads(1)
    out = Path(out)
    judge = c75.build_judge(seed, "X", c75.load_pretrained("X", seed, cells.E75_OUT))
    excl, probe, distance = _sets(seed)
    ckpt = cells.ckpt_dir(seed, out)
    resumed_from = t75._resumed_from(ckpt)
    result = vi.train_judge(judge, t75.VI_ARM, n_updates=n_updates, batch=batch,
                            max_len=t75.MAX_LEN, n_actions=N_ACTIONS, exclude=excl, probe=probe,
                            seed=seed, sync_every=SYNC_EVERY, probe_every=PROBE_EVERY,
                            draws=DRAWS, ckpt_dir=ckpt, device=device)
    judge.eval()
    dev = torch.device(device)
    rec = {**result, "arm": "X", "seed": seed, "n_updates": n_updates, "batch": batch,
           "resumed_from": resumed_from,
           "s_per_update": result["wall_s"] / max(1, n_updates - resumed_from),
           "device": str(device),
           "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "torch_version": torch.__version__,
           "gate_l": gate_l.leaf_rank_margin(judge, probe, distance, seed, N_ACTIONS),
           "git_commit": t75._git_commit()}
    out.mkdir(parents=True, exist_ok=True)
    (out / cells.train_name(seed)).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.SEEDS))
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    setup_device(args.device)
    jobs = [s for s in args.seeds if not (args.out_dir / cells.train_name(s)).exists()]
    print(f"EXP-084 Gate T: {len(jobs)} seeds on {args.device}", flush=True)
    for i, s in enumerate(jobs, 1):
        r = run(s, args.device, out=args.out_dir)
        print(f"  {i}/{len(jobs)}  s{s} margin {r['gate_l']['margin']:.4f} "
              f"{r['s_per_update']:.3f} s/update", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
