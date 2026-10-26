"""EXP-082 training: EXP-075's value iteration for arm X, its settings unchanged, continuing
each seed's Gate C-seeded checkpoint in two legs, to update 8000 (recorded, judge snapshotted)
and then to 16,000. The body is EXP-075's train.run with only the checkpoint and record paths
changed; the pretrained encoder (the reference for encoder drift) is EXP-075's.

A leg runs only when its record is missing, and refuses a checkpoint already past its end, so a
seed killed in leg 2 resumes in leg 2 and its checkpoint is never recorded as the 8000 point.

Spec: docs/superpowers/specs/2026-10-25-exp082-x-long-budget-design.md, section 5.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/082_x_long_budget/train.py --workers 4
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


gatec = _load("exp082_gatec", HERE / "gatec.py")
cells = gatec.cells  # one module object, so a patched constant reaches both
c75 = gatec.c75
t75 = gatec.t75
g81 = gatec.g81

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

torch.set_num_threads(1)


def run(seed, out=cells.OUT, *, legs=cells.LEGS, batch=cells.BATCH,
        sync_every=cells.SYNC_EVERY, probe_every=cells.PROBE_EVERY, draws=cells.DRAWS,
        src_root=cells.E75_OUT) -> list:
    torch.set_num_threads(1)
    out = Path(out)
    if not gatec.passed(seed, out):
        raise SystemExit(f"seed {seed} has no passing Gate C record in {out}; run gatec.py first")
    ck = cells.ckpt_dir(seed, out)
    state_path = ck / "state.json"
    if not state_path.exists():
        raise SystemExit(f"{state_path} missing: Gate C passed but seeded no continuation")
    # From the Gate C record, never state.json: train_judge rewrites state.json at every
    # checkpoint with its own keys.
    continued_from = json.loads((out / cells.gatec_name(seed)).read_text())["continued_from"]
    exclude, probe, distance = gatec.sets(seed)
    written = []
    for n in legs:
        path = out / cells.record_name(n, seed)
        if path.exists():
            continue
        at = json.loads(state_path.read_text())["update"]
        if at > n:
            raise SystemExit(f"{state_path}: checkpoint is at update {at}, past leg end {n}, "
                             f"and {path.name} is missing; it cannot be recorded as the {n} point")
        # The pretrained judge first, as EXP-075's run does; train_judge loads judge.pt over it.
        judge = c75.make_judge(seed, cells.SRC_ARM, src_root)
        resumed_from = t75._resumed_from(ck)
        res = vi.train_judge(
            judge, t75.VI_ARM, n_updates=n, batch=batch, max_len=t75.MAX_LEN,
            n_actions=N_ACTIONS, exclude=exclude, probe=probe, seed=seed,
            sync_every=sync_every, probe_every=probe_every, draws=draws, ckpt_dir=ck)
        if n != legs[-1]:
            g81._copy_atomic(ck / "judge.pt", ck / cells.snapshot_name(n))
        judge.eval()
        rec = {**res, "arm": cells.ARM, "seed": seed, "hidden": c75.HIDDEN[cells.SRC_ARM],
               "readout": judge.readout, "n_updates": n, "resumed_from": resumed_from,
               "batch": batch, "max_len": t75.MAX_LEN, "n_actions": N_ACTIONS,
               "sync_every": sync_every, "probe_every": probe_every, "draws": draws,
               "n_exclude": len(exclude), "n_probe": len(probe),
               "continued_from": continued_from,
               "encoder_drift": t75._encoder_drift(judge, cells.SRC_ARM, seed, src_root),
               "gate_l": gatec.measure(judge, probe, distance, seed),
               "git_commit": g81._git_commit()}
        g81._write_json(path, rec)
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
    print(f"EXP-082 train: {len(jobs)} seeds, {args.workers} workers, legs {cells.LEGS}, "
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
