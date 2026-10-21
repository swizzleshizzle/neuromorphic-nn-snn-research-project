"""EXP-080 mint: the only place this experiment builds the unbounded BFS table.

Per seed: draws the training states (4000 blocks of 1000, `random_walk_states(1000, 40, ...)`
once per block from one `random.Random(seed)`, excluding held-out states, probe states and probe
leaves), looks up their true distances, and writes them with the seed's probe set and the true
distance of every probe leaf. Training and every margin cell read this file, so no other worker
holds the 1.9 GB table. Arrays are written first and the sidecar (which carries their sha256)
last, each atomically, so a mint killed mid-write leaves the seed pending.

Spec: docs/superpowers/specs/2026-10-20-exp080-oracle-ordering-ceiling-design.md, sections 3
and 4.

Usage (repo root, PYTHONPATH=src):
    .venv/bin/python -u experiments/080_oracle_ordering/mint.py --pilot --seeds 12 13 --workers 2
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cells = _load("exp080_cells", HERE / "cells.py")

from neuromorphic.envs.cube import N_ACTIONS  # noqa: E402
from neuromorphic.envs.cube_distance import ExactBFSDistance  # noqa: E402
from neuromorphic.training import value_iteration as vi  # noqa: E402

N_FACELETS = cells.N_FACELETS


def mint_dir(out_dir) -> Path:
    return Path(out_dir) / "mint"


def npz_name(seed: int) -> str:
    return f"mint_s{seed}.npz"


def meta_name(seed: int) -> str:
    return f"mint_s{seed}.json"


def draw_blocks(n_blocks, block, max_len, rng_seed, exclude, distance, n_actions=N_ACTIONS):
    """`n_blocks` calls of random_walk_states(block, ...) from one random.Random(rng_seed), in
    order, with each state's label from `distance`."""
    rng = random.Random(rng_seed)
    n = n_blocks * block
    states = np.empty((n, N_FACELETS), dtype=np.uint8)
    labels = np.empty(n, dtype=np.uint8)
    i = 0
    for _ in range(n_blocks):
        for s in vi.random_walk_states(block, max_len, rng, n_actions, exclude):
            d = distance(s)
            if d is None:
                raise SystemExit(f"state {s} has no distance: mint needs the unbounded table")
            states[i] = s
            labels[i] = d
            i += 1
    return states, labels


def leaf_table(probe, distance, n_actions=N_ACTIONS):
    """Every probe leaf leaf_rank_margin ranks, sorted, with its true distance."""
    leaves = sorted(cells.leaf_exclusion(probe, n_actions))
    ls = np.array(leaves, dtype=np.uint8).reshape(-1, N_FACELETS)
    ld = np.array([distance(s) for s in leaves], dtype=np.uint8)
    return ls, ld


def digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _arrays(states, labels, probe, leaf_states, leaf_d) -> dict:
    return {"states": states, "labels": labels,
            "probe_states": np.array([s for s, _ in probe], dtype=np.uint8).reshape(-1, N_FACELETS),
            "probe_d": np.array([d for _, d in probe], dtype=np.uint8),
            "leaf_states": leaf_states, "leaf_d": leaf_d}


_ORDER = ("states", "labels", "probe_states", "probe_d", "leaf_states", "leaf_d")


def write_mint(out_dir, seed, states, labels, probe, leaf_states, leaf_d, meta) -> dict:
    d = mint_dir(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    arrays = _arrays(states, labels, probe, leaf_states, leaf_d)
    sidecar = {**meta, "seed": seed, "sha256": digest(*(arrays[k] for k in _ORDER))}
    tmp = d / f"mint_s{seed}.tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, d / npz_name(seed))
    tmp_meta = d / f"mint_s{seed}.tmp.json"
    tmp_meta.write_text(json.dumps(sidecar, indent=1, sort_keys=True) + "\n")
    os.replace(tmp_meta, d / meta_name(seed))
    return sidecar


# A NamedTuple, not a dataclass: dataclasses look the class's module up in sys.modules,
# and a module loaded through importlib.util is not registered there.
class Mint(NamedTuple):
    seed: int
    states: np.ndarray
    labels: np.ndarray
    probe: list
    leaf_dist: dict
    sha: str
    meta: dict


def _tuple(row) -> tuple:
    return tuple(int(x) for x in row)


def read_mint(out_dir, seed) -> Mint:
    d = mint_dir(out_dir)
    npz, js = d / npz_name(seed), d / meta_name(seed)
    for p in (npz, js):
        if not p.exists():
            raise SystemExit(f"missing {p}: run mint.py for seed {seed} first")
    meta = json.loads(js.read_text())
    if meta.get("seed") != seed:
        raise SystemExit(f"{js} is the mint of seed {meta.get('seed')}, not seed {seed}")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in _ORDER}
    sha = digest(*(arrays[k] for k in _ORDER))
    if sha != meta.get("sha256"):
        raise SystemExit(f"{npz}: sha256 {sha} does not match its sidecar's "
                         f"{meta.get('sha256')}; re-mint seed {seed}")
    probe = [(_tuple(s), int(dd)) for s, dd in zip(arrays["probe_states"], arrays["probe_d"])]
    leaf_dist = {_tuple(s): int(dd) for s, dd in zip(arrays["leaf_states"], arrays["leaf_d"])}
    return Mint(seed, arrays["states"], arrays["labels"], probe, leaf_dist, sha, meta)


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


_PROVIDER = None


def _unbounded_provider():
    """The unbounded BFS table, built on first use and held for the life of the process, so a
    mint worker builds it once and not once per seed."""
    global _PROVIDER
    if _PROVIDER is None:
        _PROVIDER = ExactBFSDistance(max_depth=None)
    return _PROVIDER


def mint_seed(seed, out_dir, n_blocks=cells.N_UPDATES, block=cells.BATCH, max_len=cells.MAX_LEN,
              provider=None) -> dict:
    t0 = time.time()
    if provider is None:
        provider = _unbounded_provider()
    probe = cells.probe_set(seed, provider)
    excl = cells.oracle_exclusion(seed, provider, probe)
    states, labels = draw_blocks(n_blocks, block, max_len, seed, excl, provider.distance)
    ls, ld = leaf_table(probe, provider.distance)
    hist = Counter(int(x) for x in labels)
    meta = {"seed": seed, "n_blocks": n_blocks, "block": block, "max_len": max_len,
            "rng": "random.Random(seed)", "n_exclude": len(excl), "n_probe": len(probe),
            "n_leaves": int(len(ld)), "label_hist": {str(k): hist[k] for k in sorted(hist)},
            "git_commit": _git_commit(), "wall_s": round(time.time() - t0, 1)}
    return write_mint(out_dir, seed, states, labels, probe, ls, ld, meta)


def pending_seeds(seeds, out_dir) -> list[int]:
    return [s for s in seeds if not (mint_dir(out_dir) / meta_name(s)).exists()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(cells.EVAL_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="outputs/ (outputs_pilot/ with --pilot) when omitted")
    args = ap.parse_args()
    cells.check_seeds(args.seeds, args.pilot)
    out = args.out_dir if args.out_dir is not None else (
        cells.PILOT_OUT if args.pilot else cells.OUT)
    jobs = pending_seeds(args.seeds, out)
    print(f"EXP-080 mint: {len(jobs)} seeds, {args.workers} workers, into {mint_dir(out)}",
          flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(mint_seed, s, out): s for s in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            m = f.result()
            print(f"  {i}/{len(jobs)}  seed {m['seed']}  n_exclude {m['n_exclude']}  "
                  f"n_leaves {m['n_leaves']}  wall_s {m['wall_s']}  sha {m['sha256'][:12]}",
                  flush=True)
    print("done.")


if __name__ == "__main__":
    main()
