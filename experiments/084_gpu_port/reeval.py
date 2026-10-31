"""EXP-084 Gates E and M: re-evaluate committed search cells and compare with their published
solved counts. Gate E runs on the device: EXP-083's 24 depth-9 cells (J4V-X16k and J4V-X, seeds 0
to 11) and EXP-082's J4V-X16k depth-9 seed-0 cell on EXP-075's held-out set. Gate M runs that last
cell on the desktop's CPU, to separate "the GPU" from "a different machine and torch build". The
search is EXP-083's `_search`, unchanged; only the judge's device differs. Expected counts are read
from the tracked records, never typed in.

Spec: docs/superpowers/specs/2026-10-31-exp084-gpu-port-design.md, section 5.

Usage (desktop, repo root, PYTHONPATH=src):
    python -u experiments/084_gpu_port/reeval.py --gate e --device cuda
    python -u experiments/084_gpu_port/reeval.py --gate m
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

from neuromorphic.training.device import setup_device  # noqa: E402

GATE_R_RECORD = "exp082_J4V-X16k_d9_s0.json"


def _published(path: Path) -> int:
    rec = json.loads(path.read_text())
    if rec.get("limit_states") is not None or rec.get("n") != 200:
        raise SystemExit(f"{path}: not a full published cell (n {rec.get('n')}, limit_states "
                         f"{rec.get('limit_states')}); refusing it as a Gate E reference")
    return int(rec["solved"])


def published(arm: str, seed: int, root=cells.E83_OUT) -> int:
    return _published(Path(root) / f"exp083_{arm}_d{cells.DEPTH}_s{seed}.json")


def published_gate_r(root=cells.E82_OUT) -> int:
    return _published(Path(root) / GATE_R_RECORD)


def _record(res, want, device, limit, t0) -> dict:
    dev = torch.device(device)
    return {"solved": res["solved"], "published": want, "diff": res["solved"] - want,
            "n": res["n"], "success_rate": res["success_rate"], "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
            "torch_version": torch.__version__, "limit_states": limit,
            "wall_s": round(time.time() - t0, 1), "git_commit": e83.sets._git_commit()}


def e_cell(arm, seed, device, *, limit=None, out=cells.OUT) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    want = published(arm, seed)
    states = e83.load_states(seed, cells.DEPTH, cells.E83_OUT)
    if limit is not None:
        states = states[:limit]
    root = cells.E82_OUT if arm == "J4V-X16k" else cells.E75_OUT
    judge = e75.load_judge("X", seed, root, cells.REQUIRED_UPDATES[arm], device=device)
    res = e83._search(judge, cells.DEPTH, seed, states)
    rec = {**_record(res, want, device, limit, t0), "arm": arm, "seed": seed}
    e83._write(Path(out) / cells.e_name(arm, seed), rec)
    return rec


def gate_r_cell(device, name, *, limit=None, out=cells.OUT) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    want = published_gate_r()
    judge = e75.load_judge("X", 0, cells.E82_OUT, 16000, device=device)
    states = e75.load_cell(cells.DEPTH, 0)[2]
    if limit is not None:
        states = states[:limit]
    res = e83._search(judge, cells.DEPTH, 0, states)
    rec = {**_record(res, want, device, limit, t0), "arm": "J4V-X16k", "seed": 0,
           "states_from": "EXP-075 held-out (EXP-082's cell)"}
    e83._write(Path(out) / name, rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=("e", "m"), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    if args.gate == "m":
        if not (args.out_dir / cells.M_NAME).exists():
            r = gate_r_cell("cpu", cells.M_NAME, out=args.out_dir)
            print(f"Gate M (cpu): solved {r['solved']} published {r['published']} "
                  f"wall {r['wall_s']}s", flush=True)
        print("done.")
        return
    setup_device(args.device)
    jobs = [(a, s) for s in cells.SEEDS for a in cells.ARMS
            if not (args.out_dir / cells.e_name(a, s)).exists()]
    print(f"EXP-084 Gate E: {len(jobs)} EXP-083 cells on {args.device}", flush=True)
    for i, (a, s) in enumerate(jobs, 1):
        r = e_cell(a, s, args.device, out=args.out_dir)
        print(f"  {i}/{len(jobs)}  {a} s{s} solved {r['solved']} published {r['published']} "
              f"wall {r['wall_s']}s", flush=True)
    if not (args.out_dir / cells.E_GATER_NAME).exists():
        r = gate_r_cell(args.device, cells.E_GATER_NAME, out=args.out_dir)
        print(f"  gate R cell solved {r['solved']} published {r['published']} "
              f"wall {r['wall_s']}s", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
