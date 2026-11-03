"""EXP-085 Gate C, continuity (voiding), and the start directory it seeds. Runs on the VPS.

Per seed, on the X16k checkpoint copied from the laptop into `--src-root`: (a) judge.pt and
state.json hash equal to EXP-082's tracked files and state.json says update 16,000; (b) every Adam
state in opt.pt is at step 16,000 and shaped like the judge (the arms' pairing rests on the shared
optimizer state); (c) the judge, re-measured on the CPU by EXP-075's Gate L call, equals EXP-082's
committed gate_l at 16,000 exactly. jt.pt is copied but not gated: train_judge re-syncs the target
on every update divisible by 100, so 16,000 overwrites it. Only then are the four files copied into
`outputs/start/`. EXP-082's directory is only ever read.

Spec: docs/superpowers/specs/2026-11-03-exp085-outward-walks-design.md, sections 3 and 6.

Usage (VPS, repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/085_outward_walks/gatec.py --src-root PATH
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


cells = _load("exp085_cells", HERE / "cells.py")
g82 = _load("exp082_gatec", cells.REPO / "experiments" / "082_x_long_budget" / "gatec.py")
c75 = cells.c75
t75 = g82.t75
e75 = g82.e75
g81 = g82.g81
gate_l = g82.gate_l
blob_of = g82.blob_of
tracked_blob = g82.tracked_blob

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402

torch.set_num_threads(1)

# state.json last: its presence marks a seeded start.
CKPT_FILES = ("judge.pt", "jt.pt", "opt.pt", "state.json")
TRACKED = ("judge.pt", "state.json")


def sets(seed):
    """EXP-075's exclusion set UNION the seed's probe states (spec section 2: the probe is
    excluded from training in both arms), the probe, and the true distance, from the full BFS
    table (about 65 s)."""
    provider = ExactBFSDistance(max_depth=None)
    exclusion = c75.exclusion_set(seed, provider)
    probe = c75.probe_set(seed, provider)
    return exclusion | {s for s, _ in probe}, probe, provider.distance


def measure(judge, probe, distance, seed) -> dict:
    """EXP-075's Gate L call, unchanged."""
    return gate_l.leaf_rank_margin(judge, probe, distance, seed, N_ACTIONS)


def x16k_record(seed, e82_root=cells.E82_OUT) -> dict:
    return json.loads((Path(e82_root) / f"exp082_X16k_u{cells.START}_s{seed}.json").read_text())


def check_identity(src: Path, ref: Path, tracked=tracked_blob, blob=blob_of):
    """`src`'s files against the blobs git tracks for the same names under `ref` (EXP-082)."""
    bad, ids = [], {}
    for name in TRACKED:
        want = tracked(ref / name)
        got = blob(src / name) if (src / name).exists() else None
        ids[name] = got
        if want is None:
            bad.append(f"{name} untracked")
        elif got != want:
            bad.append(f"{name} differs from the tracked file")
    return bad, ids


def seed_start(seed, src: Path, out) -> str:
    """Copy the checked files into outputs/start/. Never overwrites a seeded start."""
    dst = cells.start_dir(seed, out)
    if (dst / "state.json").exists():
        return "kept"
    dst.mkdir(parents=True, exist_ok=True)
    for name in CKPT_FILES:
        g81._copy_atomic(src / name, dst / name)
    return "seeded"


def gate_c(seed, src_root, out=cells.OUT, *, e82_root=cells.E82_OUT, tracked=tracked_blob,
           blob=blob_of) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    out, src_root, e82_root = Path(out), Path(src_root), Path(e82_root)
    src = c75.ckpt_dir(cells.SRC_ARM, seed, src_root)
    ref = c75.ckpt_dir(cells.SRC_ARM, seed, e82_root)
    update = json.loads((src / "state.json").read_text()).get("update")
    identity_bad, ids = check_identity(src, ref, tracked, blob)
    opt_bad, start_ok, gl = ["not checked"], False, None
    if update == cells.START:
        judge = e75.load_judge(cells.SRC_ARM, seed, src_root, required_updates=cells.START)
        opt_bad = g81.check_opt(torch.load(src / "opt.pt", map_location="cpu"), judge,
                                step=cells.START)
        _, probe, distance = sets(seed)
        gl = json.loads(json.dumps(measure(judge, probe, distance, seed)))
        start_ok = gl == x16k_record(seed, e82_root)["gate_l"]
    ok = update == cells.START and not identity_bad and not opt_bad and start_ok
    rec = {"seed": seed, "pass": ok, "start_update": update, "identity_bad": identity_bad,
           "blobs": ids, "opt_bad": opt_bad, "start_gate_l_equal": start_ok,
           "start_margin": None if gl is None else gl["margin"],
           "continued_from": {"experiment": "EXP-082", "update": update},
           "git_commit": g81._git_commit()}
    if ok:
        rec["start"] = seed_start(seed, src, out)
    rec["wall_s"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    g81._write_json(out / cells.gatec_name(seed), rec)
    if not ok:
        raise SystemExit(f"GATE C FAILED for seed {seed}: start update {update}, identity "
                         f"{identity_bad}, opt {opt_bad}, gate_l equal {start_ok}")
    return rec


def passed(seed, out=cells.OUT) -> bool:
    p = Path(out) / cells.gatec_name(seed)
    return p.exists() and json.loads(p.read_text()).get("pass") is True


def pending(seeds, out=cells.OUT) -> list[int]:
    """Seeds without a passing Gate C record; a failed record is re-checked."""
    return [s for s in seeds if not passed(s, out)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.SEEDS))
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = pending(args.seeds, args.out_dir)
    print(f"EXP-085 Gate C: {len(jobs)} seeds", flush=True)
    failed = 0
    for i, s in enumerate(jobs, 1):
        try:
            r = gate_c(s, args.src_root, args.out_dir)
            print(f"  {i}/{len(jobs)}  seed {s}  PASS  margin {r['start_margin']!r}  "
                  f"{r['start']}  wall {r['wall_s']}s", flush=True)
        except SystemExit as e:
            failed += 1
            print(f"  {i}/{len(jobs)}  seed {s}  {e}", flush=True)
    if failed:
        raise SystemExit(f"Gate C failed on {failed} seed(s): the experiment is VOID")
    print("done.")


if __name__ == "__main__":
    main()
