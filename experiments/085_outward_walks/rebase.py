"""EXP-085 rebase: move X16k's CPU checkpoint onto the device for one arm. Runs on the desktop.

X16k was trained on the laptop CPU, and train_judge refuses to resume a checkpoint whose
generator_device differs from the device asked for. So each arm gets a copy of the Gate C start
whose state.json keeps the walk RNG state, the update count and the probes, and whose encoding
generator restarts on the device from `rebase_seed(seed)`. Every tensor file is copied unchanged.
An arm that already has a state.json is never touched again.

Spec: docs/superpowers/specs/2026-11-03-exp085-outward-walks-design.md, section 5.

Usage (desktop, repo root, PYTHONPATH=src):
    python -u experiments/085_outward_walks/rebase.py --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
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


gatec = _load("exp085_gatec", HERE / "gatec.py")
cells = gatec.cells  # one module object, so a patched constant reaches both
g81 = gatec.g81

from neuromorphic.training.device import setup_device  # noqa: E402

TENSOR_FILES = ("judge.pt", "jt.pt", "opt.pt")
FORMULA = "85_000_000 + seed * 100_003 + 16000"


def rebase_seed(seed: int) -> int:
    return 85_000_000 + seed * 100_003 + 16000


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rebased(src: Path, dst: Path, seed: int, device) -> dict:
    """Copy `src`'s checkpoint into `dst` with the encoding stream restarted on `device`."""
    device = torch.device(device)
    state = json.loads((src / "state.json").read_text())
    gen = (torch.Generator() if device.type == "cpu" else torch.Generator(device=device))
    state["generator_state"] = gen.manual_seed(rebase_seed(seed)).get_state().tolist()
    state.pop("generator_device", None)
    if device.type != "cpu":
        state["generator_device"] = device.type
    dst.mkdir(parents=True, exist_ok=True)
    for name in TENSOR_FILES:
        g81._copy_atomic(src / name, dst / name)
    tmp = dst / "state.json.tmp"
    tmp.write_text(json.dumps(state))
    tmp.replace(dst / "state.json")  # last: its presence marks a rebased arm
    return state


def rebase(arm: str, seed: int, device, out=cells.OUT) -> dict:
    out = Path(out)
    if arm not in cells.ARMS:
        raise SystemExit(f"unknown arm {arm!r}")
    if not gatec.passed(seed, out):
        raise SystemExit(f"seed {seed} has no passing Gate C record in {out}")
    src = cells.start_dir(seed, out)
    dst = cells.ckpt_dir(arm, seed, out)
    if (dst / "state.json").exists():
        at = json.loads((dst / "state.json").read_text()).get("update")
        raise SystemExit(f"{dst}: already rebased (at update {at}); refusing to reset it")
    start = json.loads((src / "state.json").read_text())
    if start.get("update") != cells.START:
        raise SystemExit(f"{src}: start is at update {start.get('update')}, not {cells.START}")
    write_rebased(src, dst, seed, device)
    rec = {"arm": arm, "seed": seed, "device": str(device), "rebase_seed": rebase_seed(seed),
           "formula": FORMULA, "start_update": cells.START,
           "source_sha256": {n: _sha(src / n) for n in (*TENSOR_FILES, "state.json")},
           "git_commit": g81._git_commit()}
    g81._write_json(out / cells.rebase_name(arm, seed), rec)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.SEEDS))
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    setup_device(args.device)
    n = 0
    for s in args.seeds:
        for a in cells.ARMS:
            if not (cells.ckpt_dir(a, s, args.out_dir) / "state.json").exists():
                rebase(a, s, args.device, args.out_dir)
                n += 1
    print(f"EXP-085 rebase: {n} arm checkpoints rebased onto {args.device}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
