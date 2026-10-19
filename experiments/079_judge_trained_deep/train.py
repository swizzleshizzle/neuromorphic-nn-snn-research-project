"""EXP-079 training driver: judge S by EXP-075's value iteration on walks of length 1 to
`--max-len` (40 in every launched run), then Gate L.

Identical to EXP-075's X driver except for the walk length, where the starting encoder is read
from, and two diagnostics: the walk lengths `random_walk_states` was actually called with, and
the realised training-distance histogram over the last `probe_every` updates, counted from the
BFS instrument as the walks are drawn. Neither reaches the judge: the recorder returns the
library's walks unchanged, and the full BFS table is built only for the exclusion and probe sets,
Gate L and this histogram, never for training.

Spec: docs/superpowers/specs/2026-10-18-exp079-judge-trained-where-scrambles-live-design.md,
sections 3, 4 and 6.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/079_judge_trained_deep/train.py --pilot --seeds 12 13 \
        --n-updates 4000 --max-len 40 --workers 2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp079_cells", HERE / "cells.py")
gate_l = _load("exp074_gate_l", REPO / "experiments" / "074_wide_judge" / "gate_l.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

torch.set_num_threads(1)

# EXP-075's X optimizer: encoder lr 1e-4, head lr 1e-3.
VI_ARM = "W"


class WalkRecorder:
    """Wraps `random_walk_states`: returns its walks unchanged, notes every max_len it was
    called with, and keeps a distance histogram of the last `keep` calls (one call per update)."""

    def __init__(self, walk_fn, distance, keep):
        self.walk_fn = walk_fn
        self.distance = distance
        self.recent = deque(maxlen=keep)
        self.max_lens = []
        self.calls = 0

    def __call__(self, n, max_len, rng, n_actions, exclude, **kw):
        states = self.walk_fn(n, max_len, rng, n_actions, exclude, **kw)
        if max_len not in self.max_lens:
            self.max_lens.append(max_len)
        self.recent.append(Counter(self.distance(s) for s in states))
        self.calls += 1
        return states

    def summary(self) -> dict:
        total = Counter()
        for c in self.recent:
            total.update(c)
        n = sum(total.values())
        return {"walk_max_lens": sorted(self.max_lens),
                "train_hist": {str(d): total[d] / n for d in sorted(total)} if n else {},
                "train_hist_updates": len(self.recent)}


@contextmanager
def recording(distance, keep):
    """Route value_iteration's own `random_walk_states` lookups through a WalkRecorder, and
    restore the library function however the block exits."""
    real = vi.random_walk_states
    rec = WalkRecorder(real, distance, keep)
    vi.random_walk_states = rec
    try:
        yield rec
    finally:
        vi.random_walk_states = real


def train_core(judge, *, max_len, distance, n_updates, batch, exclude, probe, seed, sync_every,
               probe_every, draws, ckpt_dir, log=print):
    """`vi.train_judge` with S's settings, observed by a WalkRecorder. Returns train_judge's
    result and the recorder's summary."""
    with recording(distance, probe_every) as rec:
        result = vi.train_judge(
            judge, VI_ARM, n_updates=n_updates, batch=batch, max_len=max_len,
            n_actions=N_ACTIONS, exclude=exclude, probe=probe, seed=seed,
            sync_every=sync_every, probe_every=probe_every, draws=draws, ckpt_dir=ckpt_dir,
            log=log)
    return result, rec.summary()


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _resumed_from(ckpt_dir) -> int:
    """Updates already done when this session starts (0 for a fresh run). train_judge's wall_s
    covers only the session that wrote the record, so s/update needs this."""
    path = Path(ckpt_dir) / "state.json"
    return int(json.loads(path.read_text())["update"]) if path.exists() else 0


def _rel(path) -> str:
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def run(seed, n_updates, batch, sync_every, probe_every, draws, max_len, out_dir, enc_root,
        provider=None) -> dict:
    torch.set_num_threads(1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    judge = cells.make_judge(seed, enc_root)  # first: a missing encoder fails before the BFS
    if provider is None:
        provider = ExactBFSDistance(max_depth=None)
    excl = cells.exclusion_set(seed, provider)
    probe = cells.probe_set(seed, provider)
    ck = cells.ckpt_dir(seed, out_dir)
    resumed_from = _resumed_from(ck)
    result, walks = train_core(
        judge, max_len=max_len, distance=provider.distance, n_updates=n_updates, batch=batch,
        exclude=excl, probe=probe, seed=seed, sync_every=sync_every, probe_every=probe_every,
        draws=draws, ckpt_dir=ck)
    judge.eval()
    e0 = cells.c75.load_pretrained(cells.BASE_ARM, seed, enc_root).state_dict()
    rec = {
        **result, **walks,
        "arm": cells.ARM, "seed": seed, "hidden": cells.c75.HIDDEN[cells.BASE_ARM],
        "readout": judge.readout, "n_updates": n_updates, "resumed_from": resumed_from,
        "batch": batch, "max_len": max_len, "n_actions": N_ACTIONS, "sync_every": sync_every,
        "probe_every": probe_every, "draws": draws, "n_exclude": len(excl),
        "n_probe": len(probe), "encoder_root": _rel(enc_root),
        "encoder_drift": cells.drift(judge.sensory.state_dict(), e0),
        "gate_l": gate_l.leaf_rank_margin(judge, probe, provider.distance, seed, N_ACTIONS),
        "git_commit": _git_commit(),
    }
    (out_dir / cells.record_name(seed)).write_text(
        json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def pending_jobs(seeds, out_dir) -> list[int]:
    return [s for s in seeds if not (Path(out_dir) / cells.record_name(s)).exists()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--n-updates", type=int, required=True)
    # Required, with no default: a run must state its walk length (spec section 4).
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--sync-every", type=int, default=100)
    ap.add_argument("--probe-every", type=int, default=250)
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="outputs/ (outputs_pilot/ with --pilot) when omitted")
    ap.add_argument("--encoder-root", type=Path, default=None,
                    help="EXP-075's outputs/ (outputs_pilot/ with --pilot) when omitted")
    return ap


def resolve_dirs(args) -> tuple[Path, Path]:
    out = args.out_dir if args.out_dir is not None else (
        cells.PILOT_OUT if args.pilot else cells.OUT)
    enc = args.encoder_root if args.encoder_root is not None else cells.encoder_root(args.pilot)
    return out, enc


def main() -> None:
    args = build_parser().parse_args()
    cells.check_seeds(args.seeds, args.pilot)
    out_dir, enc_root = resolve_dirs(args)
    jobs = pending_jobs(args.seeds, out_dir)
    print(f"EXP-079: {len(jobs)} S training runs, {args.workers} workers, pilot={args.pilot}, "
          f"n_updates={args.n_updates}, max_len={args.max_len}, out {out_dir}, "
          f"encoders {enc_root}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run, s, args.n_updates, args.batch, args.sync_every,
                            args.probe_every, args.draws, args.max_len, out_dir, enc_root): s
                for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            loss = "None" if r["loss_last"] is None else format(r["loss_last"], ".6f")
            print(f"  {i}/{len(jobs)}  S seed {r['seed']}  updates {r['updates']}  "
                  f"loss_last {loss}  wall_s {r['wall_s']:.1f}  "
                  f"encoder_drift {r['encoder_drift']:.6f}  "
                  f"gate_l margin {r['gate_l']['margin']:.4f}  "
                  f"walk_max_lens {r['walk_max_lens']}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
