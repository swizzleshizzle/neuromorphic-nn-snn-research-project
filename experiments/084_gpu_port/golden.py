"""EXP-084 CPU fingerprint: six quantities from the judge and search path, all on the CPU at fixed
seeds. Run at the pre-port commit, it produced `tests/fixtures/exp084_cpu_golden.json`, which the
test suite compares against forever (never regenerate it from modified code). Gate C runs the same
`compute` against two trees and compares them.

`compute(root)` loads the experiment modules from `root`, and `neuromorphic` resolves from
whatever `PYTHONPATH` the process was started with, so a caller fingerprinting another tree runs
this file in a subprocess with `PYTHONPATH=<root>/src`.

Usage (repo root):
    PYTHONPATH=src .venv/bin/python -u experiments/084_gpu_port/golden.py [--root R] [--out F]
                                                                          [--limit-states N]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "exp084_cpu_golden.json"
SETS_S0 = REPO / "experiments" / "083_unseen_depth9" / "outputs" / "exp083_sets_s0.json"
KEYS = ("encode_sha", "judge_vals", "step_loss", "step_param_sha", "probe", "search")
N_STATES = 50
GOLDEN_LIMIT = 3


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().numpy().tobytes()).hexdigest()


def _state_sha(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        h.update(sd[k].detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def compute(root=REPO, limit_states: int = GOLDEN_LIMIT) -> dict:
    root = Path(root)
    torch.set_num_threads(1)
    c75 = _load("g84_c75", root / "experiments" / "075_wide_region" / "cells.py")
    e83 = _load("g84_e83", root / "experiments" / "083_unseen_depth9" / "evaluate.py")
    from neuromorphic.encoders import cube_encoder
    from neuromorphic.envs.cube import N_ACTIONS
    from neuromorphic.envs.cube_distance import ExactBFSDistance
    from neuromorphic.training import value_iteration as vi

    states = vi.random_walk_states(N_STATES, 14, random.Random(84), N_ACTIONS, set())
    obs = torch.as_tensor(states, dtype=torch.long)
    spikes = cube_encoder()(obs, T=32, generator=torch.Generator().manual_seed(7))

    judge = c75.build_judge(0, "X", c75.load_pretrained("X", 0))
    with torch.no_grad():
        vals = judge(states, torch.Generator().manual_seed(3))

    jt = vi.sync_target(judge)
    opt = vi.make_optimizer(judge, "W")
    loss = vi.train_step(judge, jt, opt, states, N_ACTIONS, torch.Generator().manual_seed(5))

    provider = ExactBFSDistance(max_depth=None)
    probe = c75.probe_set(0, provider)[::9]
    probe_res = vi.probe_judge(judge, probe, torch.Generator().manual_seed(0))

    with tempfile.TemporaryDirectory() as tmp:
        # The sets are data, not code: always this repo's committed file, whichever tree runs.
        shutil.copy(SETS_S0, Path(tmp) / SETS_S0.name)
        rec = e83.run_cell("J4V-X16k", 9, 0, out=tmp, limit_states=limit_states)
    return {
        "encode_sha": _sha(spikes),
        "judge_vals": [repr(float(v)) for v in vals],
        "step_loss": repr(loss),
        "step_param_sha": _state_sha(judge.state_dict()),
        "probe": json.loads(json.dumps(probe_res)),
        "search": {k: v for k, v in rec.items() if k not in ("wall_s", "git_commit")},
        "module_files": sorted({Path(c75.__file__).as_posix(), Path(e83.__file__).as_posix(),
                                Path(vi.__file__).as_posix()}),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--out", type=Path, default=FIXTURE)
    ap.add_argument("--limit-states", type=int, default=GOLDEN_LIMIT)
    args = ap.parse_args()
    res = compute(args.root, args.limit_states)
    if args.out == FIXTURE:
        res.pop("module_files")  # machine-specific paths; Gate C uses them, the fixture does not
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
