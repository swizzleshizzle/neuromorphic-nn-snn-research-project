"""EXP-083 search cells: EXP-082's J4V cell (mode C, k 4, goal check at 4, no-revisit) for
J4V-X16k and J4V-X over each seed's replayed sets at depths 9 and 10, plus Gate R. Both depths
use the seed's depth-9 policy agent, head and train seed (EXP-074's depth-11 convention). A judge
at the wrong update count, or a sets file whose Gate V failed, is refused.

Gate R reruns J4V-X16k depth 9 seed 0 on EXP-082's own held-out states and writes ONLY into
EXP-083's outputs (EXP-082's run_cell writes where it reads its judge, so it is not called).

Spec: docs/superpowers/specs/2026-10-30-exp083-unseen-depth9-design.md, sections 4 and 5.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/083_unseen_depth9/evaluate.py --workers 6
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


cells = _load("exp083_cells", HERE / "cells.py")
sets = _load("exp083_sets", HERE / "sets.py")
e75 = _load("exp075_evaluate", cells.E75 / "evaluate.py")

from neuromorphic.training.lookahead import evaluate_lookahead  # noqa: E402

torch.set_num_threads(1)

K = 4
JUDGES = {"J4V-X16k": ("X16k", cells.X16K_END), "J4V-X": ("X", cells.X_END)}
POLICY_DEPTH = 9
GATE_R_RECORD = "exp082_J4V-X16k_d9_s0.json"


def load_states(seed, depth, out=cells.OUT) -> list[tuple]:
    p = Path(out) / cells.sets_name(seed)
    if not p.exists():
        raise SystemExit(f"no sets file {p}; run sets.py first")
    rec = json.loads(p.read_text())
    if rec.get("pass") is not True:
        raise SystemExit(f"{p.name}: Gate V failed for seed {seed}; its states are not searched")
    return [tuple(s) for s in rec["sets"][str(depth)]]


def judge_for(arm, seed, *, e75_root=cells.E75_OUT, e82_root=cells.E82_OUT):
    if arm not in JUDGES:
        raise SystemExit(f"unknown arm {arm!r}; valid arms are {list(JUDGES)}")
    root = e82_root if arm == "J4V-X16k" else e75_root
    return e75.load_judge("X", seed, root, required_updates=JUDGES[arm][1])


def _search(judge, depth, seed, states) -> dict:
    agent, head, _states, train_seed = e75.load_cell(POLICY_DEPTH, seed)
    return evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=K, k_goal=K,
                              generator=torch.Generator().manual_seed(train_seed),
                              rng_seed=train_seed, imag_seed=train_seed,
                              critic=e75.JudgeCritic(judge), no_revisit=True)


def _write(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)


def run_cell(arm, depth, seed, out=cells.OUT, limit_states=None, *, e75_root=cells.E75_OUT,
             e82_root=cells.E82_OUT) -> dict:
    torch.set_num_threads(1)
    if depth not in cells.DEPTHS:
        raise SystemExit(f"depth {depth} is not one of the registered depths {cells.DEPTHS}")
    t0 = time.time()
    out = Path(out)
    states = load_states(seed, depth, out)
    if limit_states is not None:
        states = states[:limit_states]
    judge = judge_for(arm, seed, e75_root=e75_root, e82_root=e82_root)
    root = e82_root if arm == "J4V-X16k" else e75_root
    res = _search(judge, depth, seed, states)
    rec = {**res, "arm": arm, "judge": JUDGES[arm][0], "depth": depth, "seed": seed,
           "judge_seed": seed, "fired4": res["first_fire_len"].get("4", 0),
           "judge_updates": e75.judge_updates("X", seed, root),
           "states_from": cells.sets_name(seed), "limit_states": limit_states,
           "wall_s": round(time.time() - t0, 1), "git_commit": sets._git_commit()}
    _write(out / cells.search_name(arm, depth, seed), rec)
    return rec


def gate_r(out=cells.OUT, *, e82_root=cells.E82_OUT) -> dict:
    """J4V-X16k, depth 9, seed 0, on EXP-082's own held-out states: must equal EXP-082's record."""
    torch.set_num_threads(1)
    t0 = time.time()
    want = json.loads((Path(e82_root) / GATE_R_RECORD).read_text())
    judge = judge_for("J4V-X16k", 0, e82_root=e82_root)
    _agent, _head, states, _ts = e75.load_cell(POLICY_DEPTH, 0)
    res = _search(judge, POLICY_DEPTH, 0, states)
    rec = {"solved": res["solved"], "success_rate": res["success_rate"], "n": res["n"],
           "expected_solved": want["solved"], "expected_success_rate": want["success_rate"],
           "pass": (res["solved"], res["success_rate"]) == (want["solved"],
                                                            want["success_rate"]),
           "wall_s": round(time.time() - t0, 1), "git_commit": sets._git_commit()}
    _write(Path(out) / cells.GATE_R_NAME, rec)
    return rec


def pending(seeds, out=cells.OUT) -> list[tuple]:
    """Gate R first, then depth 10 (the longer cells), then depth 9."""
    out = Path(out)
    jobs = [] if (out / cells.GATE_R_NAME).exists() else [("gate_r", 9, 0)]
    jobs += [(a, d, s) for d in sorted(cells.DEPTHS, reverse=True) for s in seeds
             for a in cells.ARMS if not (out / cells.search_name(a, d, s)).exists()]
    return jobs


def _job(arm, depth, seed, out):
    if arm == "gate_r":
        r = gate_r(out)
        return f"gate R solved {r['solved']} want {r['expected_solved']} pass {r['pass']}"
    r = run_cell(arm, depth, seed, out)
    return f"{arm} d{depth} s{seed} solved {r['solved']}/{r['n']} wall {r['wall_s']}s"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = pending(args.seeds, args.out_dir)
    print(f"EXP-083 search: {len(jobs)} cells, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_job, a, d, s, args.out_dir) for a, d, s in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            print(f"  {i}/{len(jobs)}  {f.result()}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
