"""EXP-082 Gate C, continuity (voiding), and the continuation it seeds.

Per seed: (a) EXP-075's X judge.pt and state.json are the tracked files and state.json says
update 4000; (b) opt.pt holds a step-4000 arm-W Adam state shaped like the judge and jt.pt loads
into the judge's architecture; (c) the judge, re-measured by EXP-075's Gate L call on EXP-075's
probe, equals EXP-075's committed gate_l exactly. Only then are the four checkpoint files copied
into EXP-082's directory. EXP-075's directory is only ever read.

Spec: docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md, section 4.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/082_x_long_budget/gatec.py --workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
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


cells = _load("exp082_cells", HERE / "cells.py")
c75 = cells.c75
t75 = _load("exp075_train", cells.E75 / "train.py")
e75 = _load("exp075_evaluate", cells.E75 / "evaluate.py")
g81 = _load("exp081_gatec", cells.REPO / "experiments" / "081_oxt_long_budget" / "gatec.py")
gate_l = t75.gate_l

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402

torch.set_num_threads(1)

# state.json last: its presence marks a seeded continuation.
CKPT_FILES = ("judge.pt", "jt.pt", "opt.pt", "state.json")
TRACKED = ("judge.pt", "state.json")


def sets(seed):
    """EXP-075's exclusion and probe sets and the true distance, from the full BFS table (about
    65 s). The exclusion set shapes the walks, so a continuation must use EXP-075's exactly."""
    provider = ExactBFSDistance(max_depth=None)
    return c75.exclusion_set(seed, provider), c75.probe_set(seed, provider), provider.distance


def measure(judge, probe, distance, seed) -> dict:
    """EXP-075's Gate L call (its train.run), unchanged."""
    return gate_l.leaf_rank_margin(judge, probe, distance, seed, N_ACTIONS)


def blob_of(path) -> str:
    return subprocess.run(["git", "hash-object", str(path)], cwd=cells.REPO, capture_output=True,
                          text=True, check=True).stdout.strip()


def tracked_blob(path):
    """The blob id git records for `path` at HEAD, or None when it is not tracked there."""
    try:
        rel = Path(path).resolve().relative_to(cells.REPO).as_posix()
    except ValueError:
        return None
    r = subprocess.run(["git", "rev-parse", f"HEAD:{rel}"], cwd=cells.REPO, capture_output=True,
                       text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def check_identity(src: Path, tracked=tracked_blob, blob=blob_of) -> tuple[list[str], dict]:
    bad, ids = [], {}
    for name in TRACKED:
        want = tracked(src / name)
        got = blob(src / name) if (src / name).exists() else None
        ids[name] = got
        if want is None:
            bad.append(f"{name} untracked")
        elif got != want:
            bad.append(f"{name} differs from the tracked file")
    return bad, ids


def check_target(seed, src: Path) -> bool:
    """jt.pt loads, strictly, into the judge's architecture."""
    jt = c75.build_judge(seed, cells.SRC_ARM, c75.fresh_sensory(cells.SRC_ARM, seed))
    try:
        jt.load_state_dict(torch.load(src / "jt.pt", map_location="cpu"))
    except (RuntimeError, FileNotFoundError):
        return False
    return True


def seed_continuation(seed, src_root, out) -> str:
    """Copy EXP-075's checkpoint into EXP-082's directory. Never overwrites: a continuation that
    has started is left alone."""
    dst = cells.ckpt_dir(seed, out)
    if (dst / "state.json").exists():
        return "kept"
    src = c75.ckpt_dir(cells.SRC_ARM, seed, src_root)
    dst.mkdir(parents=True, exist_ok=True)
    for name in CKPT_FILES:
        g81._copy_atomic(src / name, dst / name)
    return "seeded"


def gate_c(seed, out=cells.OUT, src_root=cells.E75_OUT, tracked=tracked_blob,
           blob=blob_of) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    out, src_root = Path(out), Path(src_root)
    src = c75.ckpt_dir(cells.SRC_ARM, seed, src_root)
    rec75 = json.loads((src_root / c75.record_name(cells.SRC_ARM, seed)).read_text())
    update = json.loads((src / "state.json").read_text()).get("update")
    identity_bad, ids = check_identity(src, tracked, blob)
    opt_bad, jt_ok, start_ok, gl = ["not checked"], False, False, None
    if update == cells.START:
        judge = e75.load_judge(cells.SRC_ARM, seed, src_root, required_updates=cells.START)
        opt_bad = g81.check_opt(torch.load(src / "opt.pt", map_location="cpu"), judge,
                                step=cells.START)
        jt_ok = check_target(seed, src)
        _, probe, distance = sets(seed)
        gl = json.loads(json.dumps(measure(judge, probe, distance, seed)))
        start_ok = gl == rec75["gate_l"]
    ok = update == cells.START and not identity_bad and not opt_bad and jt_ok and start_ok
    rec = {"seed": seed, "pass": ok, "start_update": update, "identity_bad": identity_bad,
           "blobs": ids, "opt_bad": opt_bad, "jt_ok": jt_ok, "start_gate_l_equal": start_ok,
           "start_margin": None if gl is None else gl["margin"],
           "continued_from": {"experiment": "EXP-075", "update": update},
           "git_commit": g81._git_commit()}
    if ok:
        rec["continuation"] = seed_continuation(seed, src_root, out)
    rec["wall_s"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    g81._write_json(out / cells.gatec_name(seed), rec)
    if not ok:
        raise SystemExit(f"GATE C FAILED for seed {seed}: start update {update}, identity "
                         f"{identity_bad}, opt {opt_bad}, jt loads {jt_ok}, gate_l equal "
                         f"{start_ok}")
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
    print(f"EXP-082 Gate C: {len(jobs)} seeds, {args.workers} workers", flush=True)
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
