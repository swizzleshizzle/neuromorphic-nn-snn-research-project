"""EXP-084 Gate P: does the GPU make the same first move as the CPU? For every state of the 24
EXP-083 depth-9 cells, the search's move at t = 1 (mode C, k 4, goal check at 4, no-revisit, the
cell's own imagination seed) is computed with the judge on the CPU and on the device. Gate P is
Gate E's mechanism and calibration, and its stop rule halts the chain before Gate E when more than
9 of the 4,800 first moves disagree.

Spec: docs/superpowers/specs/2026-10-31-exp084-gpu-port-design.md, section 5.

Usage (desktop, repo root, PYTHONPATH=src):
    python -u experiments/084_gpu_port/decisions.py --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp084_cells", HERE / "cells.py")
e83 = _load("exp083_evaluate", cells.E83 / "evaluate.py")
e75 = e83.e75

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.training.device import setup_device  # noqa: E402
from neuromorphic.training.lookahead import (  # noqa: E402
    imag_seed_for,
    imagined_values,
    sequence_eligibility,
    solving_prefix,
    tree_levels,
)

K = e83.K


def stop(disagree_total: int) -> bool:
    return disagree_total > cells.P_LIMIT


def first_move(critic, state, imag_seed: int, i: int):
    """(action, masked scores) exactly as `choose_move` mode C picks the move at t = 1, or
    (action, None) when the goal check fires (the judge is not consulted then)."""
    prefix = solving_prefix(state, K, N_ACTIONS)
    if prefix is not None:
        return prefix[0], None
    levels = tree_levels(state, K, N_ACTIONS)
    imag = torch.Generator().manual_seed(imag_seed_for(imag_seed, i, 1))
    scores = imagined_values(None, critic, levels[K], generator=imag)
    mask = sequence_eligibility(levels, K, N_ACTIONS, {state})
    if bool(mask.any()):
        scores = torch.where(mask, scores, torch.full_like(scores, float("-inf")))
    return int(scores.argmax()) // N_ACTIONS ** (K - 1), scores


def compare_critics(ref, dev, states, imag_seed: int) -> dict:
    agree = goal = near = 0
    max_dj = 0.0
    dis = []
    for i, s in enumerate(states):
        a_ref, s_ref = first_move(ref, s, imag_seed, i)
        a_dev, s_dev = first_move(dev, s, imag_seed, i)
        if s_ref is None:
            goal += 1
            dj = 0.0
        else:
            finite = torch.isfinite(s_ref)
            dj = float((s_ref[finite] - s_dev[finite]).abs().max())
        max_dj = max(max_dj, dj)
        if a_ref == a_dev:
            agree += 1
            continue
        top2 = torch.topk(s_ref[torch.isfinite(s_ref)], 2).values
        tie = float(top2[0] - top2[1]) <= dj
        near += int(tie)
        dis.append({"index": i, "ref_action": a_ref, "dev_action": a_dev, "near_tie": tie,
                    "max_abs_dj": dj})
    return {"n": len(states), "agree": agree, "disagree": len(dis), "goal_fired": goal,
            "max_abs_dj": max_dj, "near_ties": near, "disagreements": dis}


def cell(arm, seed, device, *, ref_device="cpu", limit=None, out=cells.OUT) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    states = e83.load_states(seed, cells.DEPTH, cells.E83_OUT)
    if limit is not None:
        states = states[:limit]
    root = cells.E82_OUT if arm == "J4V-X16k" else cells.E75_OUT
    need = cells.REQUIRED_UPDATES[arm]
    ref = e75.JudgeCritic(e75.load_judge("X", seed, root, need, device=ref_device))
    dev = e75.JudgeCritic(e75.load_judge("X", seed, root, need, device=device))
    train_seed = e75.load_cell(cells.DEPTH, seed)[3]
    res = compare_critics(ref, dev, states, train_seed)
    dname = torch.cuda.get_device_name(0) if torch.device(device).type == "cuda" else "cpu"
    rec = {**res, "arm": arm, "seed": seed, "device": str(device), "ref_device": str(ref_device),
           "device_name": dname, "torch_version": torch.__version__, "limit_states": limit,
           "imag_seed": train_seed, "wall_s": round(time.time() - t0, 1),
           "git_commit": e83.sets._git_commit()}
    e83._write(Path(out) / cells.p_name(arm, seed), rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.SEEDS))
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    setup_device(args.device)
    jobs = [(a, s) for s in args.seeds for a in cells.ARMS
            if not (args.out_dir / cells.p_name(a, s)).exists()]
    print(f"EXP-084 Gate P: {len(jobs)} cells on {args.device}", flush=True)
    for i, (a, s) in enumerate(jobs, 1):
        r = cell(a, s, args.device, out=args.out_dir)
        print(f"  {i}/{len(jobs)}  {a} s{s} agree {r['agree']}/{r['n']} near-ties "
              f"{r['near_ties']} max |dJ| {r['max_abs_dj']:.3g} wall {r['wall_s']}s", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
