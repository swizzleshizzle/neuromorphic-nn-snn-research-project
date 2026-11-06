"""EXP-085 training: each arm's rebased X16k checkpoint continued on the device by EXP-075's value
iteration, EXP-082's settings unchanged except the max walk length, which follows the arm's
schedule over the absolute update (cells.max_len_schedule). Both arms exclude EXP-075's exclusion
set and the seed's probe states. Legs: a record at 20,000, a full checkpoint snapshot at 23,751
(Gate H (b) replays from it), a record at 24,000.

A leg runs only when its record (or snapshot) is missing, and refuses a checkpoint already past
its end, so a seed killed mid-leg resumes there and is never recorded at the wrong update.

Spec: docs/superpowers/specs/2026-11-03-exp085-outward-walks-design.md, sections 2 and 6.

Usage (laptop, repo root, PYTHONPATH=src):
    python -u experiments/085_outward_walks/train.py --device cpu --workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rebase = _load("exp085_rebase", HERE / "rebase.py")
gatec = rebase.gatec
cells = gatec.cells  # one module object, so a patched constant reaches all three
c75, t75, g81 = gatec.c75, gatec.t75, gatec.g81

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402
from neuromorphic.training.device import setup_device  # noqa: E402


def _judge(seed, src_root):
    # The pretrained judge first, as EXP-075's run does; train_judge loads judge.pt over it.
    return c75.make_judge(seed, cells.SRC_ARM, src_root)


def train_to(ck: Path, arm: str, seed: int, device, n_updates: int, *, exclude=frozenset(),
             probe=None, batch=None, sync_every=None, probe_every=None, draws=None,
             src_root=None, log=print):
    """Train the checkpoint in `ck` to `n_updates` with the arm's schedule; returns
    (judge, train_judge's result). Used by the legs and by the pre-flight and Gate H replays."""
    judge = _judge(seed, cells.E75_OUT if src_root is None else src_root)
    res = vi.train_judge(
        judge, t75.VI_ARM, n_updates=n_updates, batch=batch or cells.BATCH,
        max_len=cells.max_len_schedule(arm), n_actions=N_ACTIONS, exclude=exclude,
        probe=probe if probe is not None else [], seed=seed,
        sync_every=sync_every or cells.SYNC_EVERY, probe_every=probe_every or cells.PROBE_EVERY,
        draws=draws or cells.DRAWS, ckpt_dir=ck, log=log, device=device)
    return judge, res


def _snapshot(ck: Path, snap: Path) -> None:
    snap.mkdir(parents=True, exist_ok=True)
    for name in gatec.CKPT_FILES:  # state.json last: its presence marks a whole snapshot
        g81._copy_atomic(ck / name, snap / name)


def run(arm, seed, device, out=cells.OUT, *, batch=None, sync_every=None, probe_every=None,
        draws=None, src_root=None) -> list:
    torch.set_num_threads(1)
    out = Path(out)
    ck = cells.ckpt_dir(arm, seed, out)
    state_path = ck / "state.json"
    if not (out / cells.rebase_name(arm, seed)).exists() or not state_path.exists():
        raise SystemExit(f"arm {arm} seed {seed} is not rebased in {out}; run rebase.py first")
    continued_from = json.loads((out / cells.gatec_name(seed)).read_text())["continued_from"]
    exclude, probe, distance = gatec.sets(seed)
    src_root = cells.E75_OUT if src_root is None else Path(src_root)
    dev = torch.device(device)
    written = []
    for n in cells.LEGS:
        snap = n == cells.SNAP_UPDATE
        done = ((cells.snap_dir(arm, seed, out) / "state.json").exists() if snap
                else (out / cells.record_name(arm, n, seed)).exists())
        if done:
            continue
        at = json.loads(state_path.read_text())["update"]
        if at > n:
            raise SystemExit(f"{state_path}: checkpoint is at update {at}, past leg end {n}, and "
                             f"its {'snapshot' if snap else 'record'} is missing")
        resumed_from = t75._resumed_from(ck)
        judge, res = train_to(ck, arm, seed, device, n, exclude=exclude, probe=probe,
                              batch=batch, sync_every=sync_every, probe_every=probe_every,
                              draws=draws, src_root=src_root)
        if snap:
            _snapshot(ck, cells.snap_dir(arm, seed, out))
            continue
        judge.eval()
        rec = {**res, "arm": arm, "seed": seed, "hidden": c75.HIDDEN[cells.SRC_ARM],
               "readout": judge.readout, "n_updates": n, "resumed_from": resumed_from,
               "batch": batch or cells.BATCH, "max_len_schedule": cells.schedule_label(arm),
               "n_actions": N_ACTIONS, "sync_every": sync_every or cells.SYNC_EVERY,
               "probe_every": probe_every or cells.PROBE_EVERY, "draws": draws or cells.DRAWS,
               "n_exclude": len(exclude), "n_probe": len(probe),
               "continued_from": continued_from,
               "s_per_update": res["wall_s"] / max(1, n - resumed_from),
               "device": str(device),
               "device_name": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
               "torch_version": torch.__version__,
               "encoder_drift": t75._encoder_drift(judge.cpu(), cells.SRC_ARM, seed, src_root),
               "gate_l": gatec.measure(judge.to(dev), probe, distance, seed),
               "git_commit": g81._git_commit()}
        g81._write_json(out / cells.record_name(arm, n, seed), rec)
        written.append(rec)
    return written


def pending(out=cells.OUT) -> list[tuple[int, str]]:
    """(seed, arm) pairs without the final record, seed-major and C before O, so both arms of a
    seed finish together."""
    return [(s, a) for s in cells.SEEDS for a in cells.ARMS
            if not (Path(out) / cells.record_name(a, cells.END, s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=cells.DEVICE)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    setup_device(args.device)
    jobs = pending(args.out_dir)
    print(f"EXP-085 train: {len(jobs)} arm-seeds, {args.workers} workers, on {args.device}, "
          f"legs {cells.LEGS}", flush=True)
    # Spawn on every platform, so the VPS tests run the laptop's (Windows) start-up path.
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futs = {pool.submit(run, a, s, args.device, args.out_dir): (s, a) for s, a in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            s, a = futs[f]
            for r in f.result():
                print(f"  {i}/{len(jobs)}  {a} s{s} u{r['n_updates']} margin "
                      f"{r['gate_l']['margin']:.4f} {r['s_per_update']:.3f} s/update", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
