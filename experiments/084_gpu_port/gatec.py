"""EXP-084 Gate C (voiding): the CPU path is unchanged. golden.compute runs against the pre-port
tree (a detached worktree at EXP-083's commit) and against the merged tree, each in its own
process with that tree's `src` first on PYTHONPATH, with the search on 20 states. Every quantity
must be equal. The module paths each run loaded are checked to lie under its own root, so a base
run that silently imported the head's code cannot pass.

Spec: docs/superpowers/specs/2026-10-31-exp084-gpu-port-design.md, section 5.

Usage (VPS, repo root, after `git worktree add --detach /root/projects/.wt/base084 9ffe730`):
    .venv/bin/python -u experiments/084_gpu_port/gatec.py --base-root /root/projects/.wt/base084
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp084_cells", HERE / "cells.py")
GOLDEN = HERE / "golden.py"
KEYS = _load("exp084_golden", GOLDEN).KEYS
LIMIT = 20


def fingerprint(root: Path, limit: int = LIMIT) -> dict:
    root = Path(root).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "fp.json"
        env = {**os.environ, "PYTHONPATH": str(root / "src")}
        subprocess.run([sys.executable, "-u", str(GOLDEN), "--root", str(root), "--out", str(out),
                        "--limit-states", str(limit)], env=env, check=True)
        return json.loads(out.read_text())


def compare(a: dict, b: dict) -> list[str]:
    return [k for k in KEYS if a.get(k) != b.get(k)]


def files_under(fp: dict, root) -> bool:
    root = Path(root).resolve().as_posix().rstrip("/") + "/"
    return all(f.startswith(root) for f in fp.get("module_files", []))


def _commit(root) -> str:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-root", type=Path, required=True)
    ap.add_argument("--head-root", type=Path, default=cells.REPO)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    base_commit = _commit(args.base_root)
    if not base_commit.startswith(cells.BASE_COMMIT):
        raise SystemExit(f"{args.base_root} is at {base_commit}, Gate C needs {cells.BASE_COMMIT}")
    base, head = fingerprint(args.base_root), fingerprint(args.head_root)
    diffs = compare(base, head)
    roots_ok = files_under(base, args.base_root) and files_under(head, args.head_root)
    rec = {"base_commit": base_commit, "head_commit": _commit(args.head_root), "limit": LIMIT,
           "base": base, "head": head, "differences": diffs, "roots_ok": roots_ok,
           "pass": not diffs and roots_ok}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / cells.GATEC_NAME).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    print(f"GATE C: {'PASS' if rec['pass'] else 'FAIL'} differences {diffs} roots_ok {roots_ok}")


if __name__ == "__main__":
    main()
