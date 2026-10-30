"""EXP-083 sets: replay arm X's training walk stream, check it against the committed checkpoints
(Gate V), and write each seed's 200 never-trained-on states at depths 9 and 10.

`vi.train_judge` draws every walk from one `random.Random(seed)` and nothing else draws from it,
so replaying the 16,000 batches reproduces every training state exactly. Gate V proves it per
seed: the replayed random state must equal EXP-075's state.json at update 4000 and EXP-082's at
16,000. EXP-075's and EXP-082's directories are only ever read.

Spec: docs/superpowers/specs/2026-10-30-exp083-unseen-depth9-design.md, sections 2, 3 and 5.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/083_unseen_depth9/sets.py --workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp083_cells", HERE / "cells.py")
c75 = cells.c75

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402
from neuromorphic.training.cube_baseline import shell_states  # noqa: E402


def _rstate(rng) -> list:
    """A random state in state.json's JSON form."""
    v, internal, gauss = rng.getstate()
    return [v, list(internal), gauss]


def replay(seed, exclude, ends, *, batch=cells.BATCH, max_len=cells.MAX_LEN):
    """Every state in a training batch through update max(ends), and the random state at each
    update count in `ends`."""
    rng, seen, at = random.Random(seed), set(), {}
    for u in range(max(ends)):
        seen.update(vi.random_walk_states(batch, max_len, rng, N_ACTIONS, exclude))
        if u + 1 in ends:
            at[u + 1] = _rstate(rng)
    return seen, at


def pick(shell, seen, exclude, depth, seed, cap=cells.CAP) -> list:
    """Sort, drop trained-on and held-out states, shuffle by the registered seed, keep `cap`."""
    cands = sorted(s for s in shell if s not in seen and s not in exclude)
    random.Random(cells.SHUFFLE_BASE + 100 * depth + seed).shuffle(cands)
    return cands[:cap]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=cells.REPO, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _state_matches(ck: Path, update: int, want: list) -> bool:
    p = ck / "state.json"
    if not p.exists():
        return False
    st = json.loads(p.read_text())
    return st.get("update") == update and st.get("random_state") == want


def _default_shells(provider) -> dict:
    return {d: shell_states(provider, d) for d in cells.DEPTHS}


def build(seed, out=cells.OUT, *, exclude_fn=c75.exclusion_set, shells_fn=_default_shells,
          provider_fn=lambda: ExactBFSDistance(max_depth=None), e75_root=cells.E75_OUT,
          e82_root=cells.E82_OUT, ends=(cells.X_END, cells.X16K_END),
          batch=cells.BATCH) -> dict:
    out = Path(out)
    provider = provider_fn()
    exclude = exclude_fn(seed, provider)
    seen, at = replay(seed, exclude, ends, batch=batch)
    e75_end, e82_end = ends
    gate_v = {"u4000": _state_matches(c75.ckpt_dir("X", seed, e75_root), e75_end, at[e75_end]),
              "u16000": _state_matches(c75.ckpt_dir("X", seed, e82_root), e82_end, at[e82_end])}
    ok = all(gate_v.values())
    shells = shells_fn(provider)
    cand, chosen = {}, {}
    for d in cells.DEPTHS:
        cand[str(d)] = sum(1 for s in shells[d] if s not in seen and s not in exclude)
        chosen[str(d)] = [list(s) if isinstance(s, tuple) else s
                          for s in pick(shells[d], seen, exclude, d, seed)] if ok else []
    rec = {"seed": seed, "gate_v": gate_v, "pass": ok, "n_seen": len(seen), "candidates": cand,
           "sets": chosen, "git_commit": _git_commit()}
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / (cells.sets_name(seed) + ".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(out / cells.sets_name(seed))
    return rec


def pending(seeds, out=cells.OUT) -> list[int]:
    return [s for s in seeds if not (Path(out) / cells.sets_name(s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=cells.OUT)
    args = ap.parse_args()
    cells.check_seeds(args.seeds, False)
    jobs = pending(args.seeds, args.out_dir)
    print(f"EXP-083 sets: {len(jobs)} seeds, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(build, s, args.out_dir): s for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"  {i}/{len(jobs)}  seed {r['seed']}  gate_v {r['gate_v']}  seen "
                  f"{r['n_seen']}  candidates {r['candidates']}", flush=True)
    print("done.")


if __name__ == "__main__":
    main()
