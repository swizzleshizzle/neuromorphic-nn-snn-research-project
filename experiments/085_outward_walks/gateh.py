"""EXP-085 Gate H, hardware (voiding). The desktop has logged memory faults, so two replays:

(a) search: 2 cells per depth, picked by random.Random(85) before the run, are run again into
    their own records and must match the originals in every field but wall time and commit;
(b) training: only when the pre-flight found GPU training deterministic. 2 seeds per arm replay
    from the update-23,751 snapshot to 24,000 and the judge must be tensor-equal to the arm's.
    Otherwise each pick is recorded as unavailable.

Spec: docs/superpowers/specs/2026-11-03-exp085-outward-walks-design.md, section 6.

Usage (desktop, repo root, PYTHONPATH=src):
    python -u experiments/085_outward_walks/gateh.py --part a --device cuda
    python -u experiments/085_outward_walks/gateh.py --part b --device cuda
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load("exp085_train", HERE / "train.py")
evaluate = _load("exp085_evaluate", HERE / "evaluate.py")
gatec = train.gatec
cells = train.cells
g81 = gatec.g81

from neuromorphic.training.device import setup_device  # noqa: E402

IGNORED = ("wall_s", "git_commit")


def search_picks() -> list[tuple[str, int, int]]:
    rng = random.Random(cells.H_SEED)
    pairs = [(a, s) for a in cells.ARMS for s in cells.SEEDS]
    return [(a, d, s) for d in cells.DEPTHS for a, s in rng.sample(pairs, 2)]


def train_picks() -> list[tuple[str, int]]:
    rng = random.Random(cells.H_SEED + 1)
    return [(a, s) for a in cells.ARMS for s in rng.sample(list(cells.SEEDS), 2)]


def same(a: dict, b: dict) -> bool:
    strip = lambda r: {k: v for k, v in r.items() if k not in IGNORED}  # noqa: E731
    return strip(a) == strip(b)


def run_a(device, out=cells.OUT) -> None:
    out = Path(out)
    for arm, d, s in search_picks():
        if not (out / cells.search_name(arm, d, s)).exists():
            raise SystemExit(f"Gate H (a): {cells.search_name(arm, d, s)} missing; run the search")
        if not (out / cells.h_search_name(arm, d, s)).exists():
            evaluate.run_cell(arm, d, s, device, out, name=cells.h_search_name(arm, d, s))


def run_b(device, out=cells.OUT) -> None:
    out = Path(out)
    pre = json.loads((out / cells.PREFLIGHT_NAME).read_text())
    for arm, s in train_picks():
        path = out / cells.h_train_name(arm, s)
        if path.exists():
            continue
        rec = {"arm": arm, "seed": s, "device": str(device), "available": False,
               "git_commit": g81._git_commit()}
        if pre.get("deterministic") is True:
            exclude, probe, _ = gatec.sets(s)
            with tempfile.TemporaryDirectory() as tmp:
                ck = Path(tmp) / "replay"
                shutil.copytree(cells.snap_dir(arm, s, out), ck)
                judge, _ = train.train_to(ck, arm, s, device, cells.END, exclude=exclude,
                                          probe=probe, log=lambda m: None)
            got = {k: v.detach().cpu() for k, v in judge.state_dict().items()}
            want = torch.load(cells.ckpt_dir(arm, s, out) / "judge.pt", map_location="cpu")
            bad = sorted(k for k in set(got) | set(want)
                         if k not in got or k not in want or not torch.equal(got[k], want[k]))
            rec.update(available=True, equal=not bad, mismatched=bad)
        g81._write_json(path, rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("a", "b"), required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    setup_device(args.device)
    (run_a if args.part == "a" else run_b)(args.device, args.out_dir)
    print("done.")


if __name__ == "__main__":
    main()
