"""EXP-071 step 0: can the critic rank a state's children? (spec section 4)

The BFS table is the YARDSTICK here and nowhere in the procedure, which is why this lives in the
experiment folder and not in lookahead.py.

Usage: .venv/bin/python -u experiments/071_critic_and_no_revisit/rank.py --workers 12
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics as st
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from neuromorphic.envs.cube import apply_move
from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.lookahead import (
    imag_seed_for, imagined_logp, imagined_values, tree_levels,
)

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp071_cells", HERE / "cells.py")
cells = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cells)

torch.set_num_threads(1)


def rank_state(agent, head, critic, state, n_actions, provider, generator) -> dict:
    children = [apply_move(state, a) for a in range(n_actions)]
    leaves = tree_levels(state, 3, n_actions)[3]
    logp = imagined_logp(agent, head, [state], generator=generator)[0]
    values = imagined_values(agent, critic, children, generator=generator)
    leaf_values = imagined_values(agent, critic, leaves, generator=generator)
    d = provider.distance(state)
    improving = [provider.distance(c) == d - 1 for c in children]
    leaf_d = [provider.distance(l) for l in leaves]
    top = int(leaf_values.argmax())
    return {
        "critic_hit": int(improving[int(values.argmax())]),
        "policy_hit": int(improving[int(logp.argmax())]),
        "chance": sum(improving) / n_actions,
        "leaf_closer_hit": int(leaf_d[top] < d),
        "leaf_closer_chance": sum(x < d for x in leaf_d) / len(leaf_d),
        "leaf_d3_hit": int(leaf_d[top] == d - 3),
        "leaf_d3_chance": sum(x == d - 3 for x in leaf_d) / len(leaf_d),
    }


def rank_seed(seed: int, out_dir: Path) -> dict:
    torch.set_num_threads(1)
    agent, head, states, train_seed = cells.load_cell(seed)
    critic = cells.load_critic(seed)
    provider = ExactBFSDistance(max_depth=cells.DEPTH + 3)
    n_actions = head.head.out_features
    rows = [rank_state(agent, head, critic, s, n_actions, provider,
                       torch.Generator().manual_seed(imag_seed_for(train_seed, i, 0)))
            for i, s in enumerate(states)]
    rec = {"seed": seed, "n": len(rows)}
    for key in rows[0]:
        rec[key] = st.mean(r[key] for r in rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"exp071_rank_d{cells.DEPTH}_s{seed}.json").write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(rank_seed, args.seeds, [args.out_dir] * len(args.seeds)):
            print(f"  s{r['seed']}: child critic {r['critic_hit']:.4f} policy {r['policy_hit']:.4f} "
                  f"chance {r['chance']:.4f} | leaf-closer {r['leaf_closer_hit']:.4f} "
                  f"chance {r['leaf_closer_chance']:.4f}", flush=True)


if __name__ == "__main__":
    main()
