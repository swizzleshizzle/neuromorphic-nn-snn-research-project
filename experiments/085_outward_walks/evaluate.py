"""EXP-085 search cells: EXP-082's J4V cell (mode C, k 4, goal check at 4, no-revisit, EXP-075's
held-out sets and streams, 200 scrambles), unchanged except for the judge, which is arm C or O at
24,000 updates, run on the device. The search runs on every seed whatever the probe says (spec
section 8). EXP-084's Gate E showed this search gives the CPU's numbers exactly on the GPU.

Spec: docs/superpowers/specs/2026-11-03-exp085-outward-walks-design.md, section 7.

Usage (desktop, repo root, PYTHONPATH=src):
    python -u experiments/085_outward_walks/evaluate.py --device cuda
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


gatec = _load("exp085_gatec", HERE / "gatec.py")
cells = gatec.cells
e75, g81 = gatec.e75, gatec.g81

from neuromorphic.training.device import setup_device  # noqa: E402
from neuromorphic.training.lookahead import evaluate_lookahead  # noqa: E402

K = 4


def run_cell(arm, depth, seed, device, out=cells.OUT, name=None, limit_states=None) -> dict:
    torch.set_num_threads(1)
    if depth not in cells.DEPTHS:
        raise SystemExit(f"depth {depth} is not one of the registered depths {cells.DEPTHS}")
    if arm not in cells.ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    t0 = time.time()
    out = Path(out)
    judge = e75.load_judge(cells.SRC_ARM, seed, cells.arm_root(arm, out),
                           required_updates=cells.END, device=device)
    agent, head, states, train_seed = e75.load_cell(depth, seed)
    if limit_states is not None:
        states = states[:limit_states]
    res = evaluate_lookahead(agent, head, states, depth=depth, mode="C", k=K, k_goal=K,
                             generator=torch.Generator().manual_seed(train_seed),
                             rng_seed=train_seed, imag_seed=train_seed,
                             critic=e75.JudgeCritic(judge), no_revisit=True)
    state = json.loads((cells.ckpt_dir(arm, seed, out) / "state.json").read_text())
    dev = torch.device(device)
    rec = {**res, "depth": depth, "seed": seed, "judge_seed": seed, "arm": f"J4V-{arm}",
           "judge": arm, "k": K, "k_goal": K, "mode": "C", "no_revisit": True,
           "fired4": res["first_fire_len"].get("4", 0), "judge_updates": int(state["update"]),
           "device": str(device),
           "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "torch_version": torch.__version__, "limit_states": limit_states,
           "wall_s": round(time.time() - t0, 1), "git_commit": g81._git_commit()}
    g81._write_json(out / (name or cells.search_name(arm, depth, seed)), rec)
    return rec


def pending(out=cells.OUT) -> list[tuple[str, int, int]]:
    """(arm, depth, seed) without a record; depth 11 first, its cells are the longest."""
    return [(a, d, s) for d in sorted(cells.DEPTHS, reverse=True) for s in cells.SEEDS
            for a in cells.ARMS if not (Path(out) / cells.search_name(a, d, s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    setup_device(args.device)
    jobs = pending(args.out_dir)
    print(f"EXP-085 search: {len(jobs)} cells on {args.device}", flush=True)
    for i, (a, d, s) in enumerate(jobs, 1):
        r = run_cell(a, d, s, args.device, args.out_dir)
        print(f"  {i}/{len(jobs)}  J4V-{a} d{d} s{s} solved {r['solved']}/{r['n']} "
              f"wall {r['wall_s']}s", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
