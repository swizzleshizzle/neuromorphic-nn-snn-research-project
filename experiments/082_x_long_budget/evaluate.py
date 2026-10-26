"""EXP-082 search cells: EXP-081's J4V cell (mode C, k 4, goal check at 4, no-revisit, EXP-075's
held-out sets and streams), unchanged except for the judge, which is X16k, at depths 9 and 11.
The search runs on every seed whatever the probe says (spec section 8). A judge not at exactly
16,000 updates is refused.

Spec: docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md, sections 7 and 9.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/082_x_long_budget/evaluate.py --workers 12
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gatec = _load("exp082_gatec", HERE / "gatec.py")
cells = gatec.cells
e75 = gatec.e75

from neuromorphic.training.lookahead import evaluate_lookahead  # noqa: E402

torch.set_num_threads(1)

ARM = f"J4V-{cells.ARM}"
K = 4


def record_name(depth: int, seed: int) -> str:
    return cells.search_name(depth, seed)


def run_cell(depth, seed, out=cells.OUT, limit_states=None) -> dict:
    torch.set_num_threads(1)
    if depth not in cells.DEPTHS:
        raise SystemExit(f"depth {depth} is not one of the registered depths {cells.DEPTHS}")
    t0 = time.time()
    out = Path(out)
    judge = e75.load_judge(cells.SRC_ARM, seed, out, required_updates=cells.LEGS[-1])
    critic = e75.JudgeCritic(judge)
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=K, k_goal=K,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    state = json.loads((cells.ckpt_dir(seed, out) / "state.json").read_text())
    rec = {**res, "depth": depth, "seed": seed, "judge_seed": seed, "arm": ARM,
           "judge": cells.ARM, "fired4": res["first_fire_len"].get("4", 0),
           "wall_s": round(time.time() - t0, 1), "git_commit": gatec.g81._git_commit(),
           "limit_states": limit_states, "judge_updates": int(state["update"])}
    gatec.g81._write_json(out / record_name(depth, seed), rec)
    return rec


def pending(depths, seeds, out=cells.OUT) -> list[tuple[int, int]]:
    """Depth 11 first: its cells are the longest, so they start in the first wave."""
    return [(d, s) for d in sorted(depths, reverse=True) for s in seeds
            if not (Path(out) / record_name(d, s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--depths", type=int, nargs="+", default=list(cells.DEPTHS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = pending(args.depths, args.seeds, args.out_dir)
    print(f"EXP-082 search: {len(jobs)} cells, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_cell, d, s, args.out_dir): (d, s) for d, s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  {ARM} d{r['depth']} s{r['seed']} solved {r['solved']}/"
                  f"{r['n']} wall {r['wall_s']}s", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
