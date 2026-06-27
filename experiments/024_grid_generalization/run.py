"""EXP-024 — grid-world generalization study (random goals, held-out eval).

Tests whether the v1 policy (frozen brain + linear head) learned to navigate or
memorized the fixed goal. Trains on a train subset of goal cells, evaluates greedily
on held-out cells, and reports the generalization gap. Brain stays frozen (ADR-0001).

Run (repo root, venv active):
    .venv/Scripts/python.exe experiments/024_grid_generalization/run.py --tag shaped
    .venv/Scripts/python.exe experiments/024_grid_generalization/run.py --no-shaping --tag sparse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch  # noqa: E402

from neuromorphic.training.generalization import GenConfig, run_generalization

torch.set_num_threads(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-024 grid-world generalization study")
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-heldout", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--tag", type=str, default="shaped")
    p.add_argument("--shaping", action=argparse.BooleanOptionalAction, default=True,
                   help="potential-based distance-to-goal shaping (default on)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = GenConfig(
        seed=args.seed, episodes=args.episodes, lr=args.lr, shaping=args.shaping,
        n_heldout=args.n_heldout, max_steps=args.max_steps, tag=args.tag,
        out_dir=Path("outputs"),
    )
    print(f"EXP-024 · {cfg.episodes} eps · shaping {cfg.shaping} · tag {cfg.tag}", flush=True)
    summary = run_generalization(cfg)
    et = summary["eval"]["train"]
    eh = summary["eval"]["heldout"]
    print(f"train  goals: success {et['success_rate']:.0%} · opt {et['optimality']:.2f}", flush=True)
    print(f"heldout goals: success {eh['success_rate']:.0%} · opt {eh['optimality']:.2f}", flush=True)
    print(f"generalization gap: {summary['generalization_gap']:+.2f} "
          f"(train success minus heldout success)", flush=True)


if __name__ == "__main__":
    main()
