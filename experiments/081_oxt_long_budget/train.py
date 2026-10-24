"""EXP-081 training: EXP-080's train_x_arm, unchanged, continuing each seed's Gate C-seeded
checkpoint in two legs, to update 8000 (recorded, judge snapshotted) and then to 16,000.

A leg runs only when its record is missing, and refuses a checkpoint already past its end, so a
seed killed in leg 2 resumes in leg 2 and its checkpoint is never recorded as the 8000 point.

Spec: docs/superpowers/specs/2026-10-23-exp081-oxt-long-budget-design.md, section 5.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/081_oxt_long_budget/train.py --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gatec = _load("exp081_gatec", HERE / "gatec.py")
cells = gatec.cells  # one module object, so a patched constant reaches both
c80 = cells.c80
m80 = gatec.m80
t80 = _load("exp080_train", cells.E80 / "train.py")
gate_l = gatec.gate_l

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402

torch.set_num_threads(1)


def run(seed, out=cells.OUT, *, legs=cells.LEGS, batch=cells.BATCH,
        probe_every=cells.PROBE_EVERY, mint_root=None, x_root=None, enc_root=None) -> list:
    torch.set_num_threads(1)
    out = Path(out)
    if not gatec.passed(seed, out):
        raise SystemExit(f"seed {seed} has no passing Gate C record in {out}; run gatec.py first")
    x_root = c80.x_judge_root(False) if x_root is None else x_root
    enc_root = c80.encoder_root(False) if enc_root is None else enc_root
    m = m80.read_mint(out if mint_root is None else mint_root, seed)
    ck = cells.ckpt_dir(seed, out)
    state_path = ck / "state.json"
    if not state_path.exists():
        raise SystemExit(f"{state_path} missing: Gate C passed but seeded no continuation")
    continued_from = json.loads(state_path.read_text()).get("continued_from")
    written = []
    for n in legs:
        path = out / cells.record_name(n, seed)
        if path.exists():
            continue
        at = json.loads(state_path.read_text())["update"]
        if at > n:
            raise SystemExit(f"{state_path}: checkpoint is at update {at}, past leg end {n}, "
                             f"and {path.name} is missing; it cannot be recorded as the {n} point")
        judge = c80.make_judge(cells.ARM, seed, x_root, enc_root)
        res = t80.train_x_arm(cells.ARM, judge, m, n_updates=n, batch=batch,
                              probe_every=probe_every, seed=seed, ckpt_dir=ck)
        if n != legs[-1]:
            gatec._copy_atomic(ck / "judge.pt", ck / cells.snapshot_name(n))
        judge.eval()
        rec = {**res, "arm": cells.ARM, "seed": seed, "n_updates": n, "batch": batch,
               "probe_every": probe_every, "continued_from": continued_from,
               "encoder_drift": c80.drift(judge.sensory.state_dict(),
                                          c80.reference_sensory(cells.ARM, seed, x_root,
                                                                enc_root)),
               "mint_sha": m.sha, "mint_meta": m.meta, "n_actions": N_ACTIONS,
               "gate_l": gate_l.leaf_rank_margin(judge, m.probe, m.leaf_dist.__getitem__, seed,
                                                 N_ACTIONS),
               "git_commit": gatec._git_commit()}
        gatec._write_json(path, rec)
        written.append(rec)
    return written


def pending(seeds, out=cells.OUT, legs=cells.LEGS) -> list[int]:
    return [s for s in seeds if not (Path(out) / cells.record_name(legs[-1], s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = pending(args.seeds, args.out_dir)
    print(f"EXP-081 train: {len(jobs)} seeds, {args.workers} workers, legs {cells.LEGS}, "
          f"out {args.out_dir}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, s, args.out_dir): s for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            for r in f.result():
                loss = "None" if r["loss_last"] is None else format(r["loss_last"], ".6f")
                print(f"  {i}/{len(jobs)}  seed {r['seed']}  update {r['n_updates']}  resumed "
                      f"{r['resumed_from']}  loss_last {loss}  wall_s {r['wall_s']:.1f}  drift "
                      f"{r['encoder_drift']:.3f}  gate_l margin {r['gate_l']['margin']:.4f}",
                      flush=True)
    print("done.")


if __name__ == "__main__":
    main()
