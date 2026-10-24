"""EXP-081 Gate C, continuity (voiding), and the continuation it seeds.

Per seed: (a) the 16,000-block mint's first 4000 blocks, probe and leaf distances are EXP-080's
byte for byte; (b) EXP-080's O-Xt checkpoint is at update 4000, its judge re-measured on the new
mint's probe equals EXP-080's committed gate_l exactly, and its opt.pt holds a step-4000 Adam
state shaped like the judge. Only then is the checkpoint copied into EXP-081's directory with
the new mint's sha: EXP-080's resume path rightly refuses a checkpoint from another mint, and the
re-stamp is legitimate only because (a) proved the blocks it trained on unchanged.

Spec: docs/superpowers/specs/2026-10-23-exp081-oxt-long-budget-design.md, section 4.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/081_oxt_long_budget/gatec.py --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp081_cells", HERE / "cells.py")
c80 = cells.c80
m80 = _load("exp080_mint", cells.E80 / "mint.py")
gate_l = _load("exp074_gate_l", cells.REPO / "experiments" / "074_wide_judge" / "gate_l.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402

torch.set_num_threads(1)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def check_prefix(new, old) -> list[str]:
    """The names of the parts where `new` does not extend `old`; empty when it does."""
    n = len(old.labels)
    bad = []
    if len(new.labels) <= n:
        bad.append("length")
    if not np.array_equal(new.states[:n], old.states):
        bad.append("states")
    if not np.array_equal(new.labels[:n], old.labels):
        bad.append("labels")
    if new.probe != old.probe:
        bad.append("probe")
    if new.leaf_dist != old.leaf_dist:
        bad.append("leaves")
    return bad


def check_start(seed, judge, new_mint, e80_record) -> tuple[bool, dict]:
    gl = gate_l.leaf_rank_margin(judge, new_mint.probe, new_mint.leaf_dist.__getitem__, seed,
                                 N_ACTIONS)
    gl = json.loads(json.dumps(gl))
    return gl == e80_record["gate_l"], gl


def check_opt(opt_sd, judge, step=None) -> list[str]:
    """Problems with an arm-W Adam state for `judge` at `step`; empty when there are none.
    Parameter order is the optimizer's: the head's group, then the region's."""
    step = cells.START if step is None else step
    params = list(judge.head.parameters()) + list(judge.sensory.parameters())
    state = opt_sd.get("state", {})
    bad = []
    if sorted(state) != list(range(len(params))):
        bad.append(f"state keys {sorted(state)} for {len(params)} parameters")
        return bad
    for i, p in enumerate(params):
        st = state[i]
        if float(st["step"]) != step:
            bad.append(f"param {i} step {float(st['step'])}")
        for k in ("exp_avg", "exp_avg_sq"):
            if tuple(st[k].shape) != tuple(p.shape):
                bad.append(f"param {i} {k} shape {tuple(st[k].shape)} vs {tuple(p.shape)}")
    return bad


def seed_continuation(seed, new_mint, e80_root, out) -> str:
    """Copy EXP-080's checkpoint into EXP-081's directory under the new mint's sha. Never
    overwrites: a continuation that has started is left alone, and refused if it is on another
    mint."""
    dst = cells.ckpt_dir(seed, out)
    state_path = dst / "state.json"
    if state_path.exists():
        got = json.loads(state_path.read_text()).get("mint_sha")
        if got != new_mint.sha:
            raise SystemExit(f"{state_path}: continuation is on mint {got}, not seed {seed}'s "
                             f"16,000-block mint {new_mint.sha}; delete {dst} and re-run Gate C")
        return "kept"
    src = c80.ckpt_dir(cells.ARM, seed, e80_root)
    dst.mkdir(parents=True, exist_ok=True)
    _copy_atomic(src / "judge.pt", dst / "judge.pt")
    _copy_atomic(src / "opt.pt", dst / "opt.pt")
    state = json.loads((src / "state.json").read_text())
    state["continued_from"] = {"experiment": "EXP-080", "update": state["update"],
                               "mint_sha": state["mint_sha"]}
    state["mint_sha"] = new_mint.sha
    _write_json(state_path, state)  # last: its presence marks the continuation as seeded
    return "seeded"


def gate_c(seed, out=cells.OUT, e80_root=cells.E80_OUT, new_root=None) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    out = Path(out)
    new = m80.read_mint(out if new_root is None else new_root, seed)
    old = m80.read_mint(e80_root, seed)
    rec80 = json.loads((Path(e80_root) / c80.record_name(cells.ARM, seed)).read_text())
    src = c80.ckpt_dir(cells.ARM, seed, e80_root)
    update = json.loads((src / "state.json").read_text()).get("update")
    prefix_bad = check_prefix(new, old)
    start_ok, gl = False, None
    opt_bad = ["not checked"]
    if update == cells.START:
        judge = c80.load_trained(cells.ARM, seed, e80_root, required=cells.START)
        start_ok, gl = check_start(seed, judge, new, rec80)
        opt_bad = check_opt(torch.load(src / "opt.pt"), judge)
    ok = not prefix_bad and start_ok and not opt_bad and update == cells.START
    rec = {"seed": seed, "pass": ok, "prefix_bad": prefix_bad, "start_update": update,
           "start_gate_l_equal": start_ok, "start_margin": None if gl is None else gl["margin"],
           "opt_bad": opt_bad, "new_mint_sha": new.sha, "new_n_states": int(len(new.labels)),
           "e80_mint_sha": old.sha, "e80_n_states": int(len(old.labels)),
           "continued_from": {"experiment": "EXP-080", "update": update, "mint_sha": old.sha},
           "git_commit": _git_commit()}
    if ok:
        rec["continuation"] = seed_continuation(seed, new, e80_root, out)
    rec["wall_s"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / cells.gatec_name(seed), rec)
    if not ok:
        raise SystemExit(f"GATE C FAILED for seed {seed}: prefix {prefix_bad}, start update "
                         f"{update}, gate_l equal {start_ok}, opt {opt_bad}")
    return rec


def passed(seed, out=cells.OUT) -> bool:
    p = Path(out) / cells.gatec_name(seed)
    return p.exists() and json.loads(p.read_text()).get("pass") is True


def pending(seeds, out=cells.OUT) -> list[int]:
    """Seeds without a passing Gate C record; a failed record is re-checked."""
    return [s for s in seeds if not passed(s, out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = pending(args.seeds, args.out_dir)
    print(f"EXP-081 Gate C: {len(jobs)} seeds, {args.workers} workers", flush=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(gate_c, s, args.out_dir): s for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
                print(f"  {i}/{len(futs)}  seed {r['seed']}  PASS  margin {r['start_margin']!r}  "
                      f"{r['continuation']}  wall {r['wall_s']}s", flush=True)
            except SystemExit as e:
                failed += 1
                print(f"  {i}/{len(futs)}  seed {futs[f]}  {e}", flush=True)
    if failed:
        raise SystemExit(f"Gate C failed on {failed} seed(s): the experiment is VOID")
    print("done.")


if __name__ == "__main__":
    main()
