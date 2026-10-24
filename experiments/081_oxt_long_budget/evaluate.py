"""EXP-081 search cell, run only when Claim 1 is CONFIRMED: EXP-080's J4V cell (mode C, k 4,
goal check at 4, no-revisit, depth 11, EXP-075's held-out sets and streams) with the 16,000-update
O-Xt judge. A judge short of 16,000 updates is refused.

Spec: docs/superpowers/specs/2026-10-23-exp081-oxt-long-budget-design.md, section 9.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/081_oxt_long_budget/evaluate.py --workers 6
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


gatec = _load("exp081_gatec", HERE / "gatec.py")
cells = gatec.cells
c80 = cells.c80
ev80 = _load("exp080_evaluate", cells.E80 / "evaluate.py")
e75 = c80.e75

from neuromorphic.training.lookahead import evaluate_lookahead  # noqa: E402

torch.set_num_threads(1)

ARM = "J4V-O-Xt16k"
K = ev80.K
DEPTH = ev80.DEPTH


def record_name(seed: int) -> str:
    return f"exp081_{ARM}_d{DEPTH}_s{seed}.json"


def run_cell(seed, out=cells.OUT, limit_states=None) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    out = Path(out)
    judge = c80.load_trained(cells.ARM, seed, out, required=cells.LEGS[-1])
    critic = e75.JudgeCritic(judge)
    agent, head, states, train_seed = e75.load_cell(DEPTH, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=DEPTH, mode="C", k=K, k_goal=K,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=critic, no_revisit=True)
    state = json.loads((cells.ckpt_dir(seed, out) / "state.json").read_text())
    rec = {**res, "depth": DEPTH, "seed": seed, "judge_seed": seed, "arm": ARM,
           "judge": cells.ARM, "fired4": res["first_fire_len"].get("4", 0),
           "wall_s": round(time.time() - t0, 1), "git_commit": gatec._git_commit(),
           "limit_states": limit_states, "judge_updates": int(state["update"])}
    gatec._write_json(out / record_name(seed), rec)
    return rec


def pending(seeds, out=cells.OUT) -> list[int]:
    return [s for s in seeds if not (Path(out) / record_name(s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = pending(args.seeds, args.out_dir)
    print(f"EXP-081 search: {len(jobs)} cells, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_cell, s, args.out_dir): s for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  {ARM} d{DEPTH} s{r['seed']} solved {r['solved']}/{r['n']} "
                  f"wall {r['wall_s']}s", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
