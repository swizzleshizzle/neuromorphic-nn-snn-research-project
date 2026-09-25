"""EXP-069: at depth 5 with a pretrained encoder, does the ENCODER carry the seed effect?

EXP-068 decomposed the seed effect at depth 3 on an encoder-free cell: interaction 0.650, task draw
0.267, trajectory 0.084. `docs/seed-effect.md` measured the effect at DEPTH 5 WITH AN E0 ENCODER,
where a seed fixes a third thing, the pretrained encoder. EXP-068 named it the next suspect.

Two factors, fully crossed, 8 x 8 = 64 cells:

    E  the encoder: its pretraining seed AND `encoder_seed` (the frozen brain's init), index e
    R  the rest: `split_seed` = `train_seed` = r

> THE OBVIOUS DESIGN LEAKS, AND THIS ONE DOES NOT. Each shipped E0 encoder was pretrained with its
> OWN seed's held-out states forbidden (EXP-040's `rl_heldout_union`). Crossing E0 encoder e with
> split r != e would evaluate on states encoder e may have pretrained on, inflating exactly the
> off-diagonal cells, and the contamination would land in the interaction term and imitate
> EXP-068's headline. So phase `pretrain` builds 8 NEW encoders, each forbidding the UNION of all 8
> grid splits' held-out states. Cost: 26.4% of the pretraining pool forbidden instead of 4.5%.
> These are not the shipped E0 encoders and the diagonal does not reproduce EXP-043; that is the
> price of a clean encoder factor, and it is paid knowingly.

Everything else is EXP-043's capped depth-5 cell via `dataclasses.replace`, never retyped, and
EXP-040's own pretraining recipe (`PRETRAIN`, `PRETRAIN_DEPTHS`, `rl_heldout_union`), imported.

Spec: docs/superpowers/specs/2026-09-25-exp069-encoder-decomposition-design.md

Usage:
    .venv/bin/python -u experiments/069_encoder_decomposition/run.py --phase pretrain --workers 8
    .venv/bin/python -u experiments/069_encoder_decomposition/run.py --phase calibrate --workers 8
    .venv/bin/python -u experiments/069_encoder_decomposition/run.py --phase rl --workers 8 --skip-existing
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import torch

from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.cube_baseline import record_filename, run_cube_baseline, shell_states
from neuromorphic.training.encoder_pretrain import (
    PretrainConfig,
    build_pairs,
    save_encoder,
    train_inverse_model,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
EXPS = HERE.parent


def _load(name: str, folder: str):
    spec = importlib.util.spec_from_file_location(name, EXPS / folder / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_exp040 = _load("exp040_run", "040_pretrained_encoder_policy")
_exp043 = _load("exp043_run", "043_cap_at_depth_5_6")

DEPTH = 5
GRID = tuple(range(8))            # 8 x 8 = 64 cells; 8 workers divides it into 8 clean waves


def encoder_path(out_dir: Path, e: int) -> Path:
    return out_dir / f"exp069_encoder_e{e}.pt"


def union_forbidden(provider) -> set:
    """Every state the POLICY is evaluated on, for EVERY split in the grid.

    EXP-040 forbade one seed's held-out union. Forbidding all eight is what makes encoder e
    clean against every split r it is crossed with.
    """
    held = set()
    for r in GRID:
        held |= _exp040.rl_heldout_union(provider, r)
    return held


def pretrain_one(e: int, out_dir: Path) -> dict:
    torch.set_num_threads(1)
    t0 = time.time()
    provider = ExactBFSDistance(max_depth=max(_exp040.PRETRAIN_DEPTHS) + 1)
    states = []
    for d in _exp040.PRETRAIN_DEPTHS:
        states.extend(shell_states(provider, d))
    forbidden = union_forbidden(provider)
    pairs = build_pairs(states, forbidden=forbidden)
    assert not (forbidden & {p[0] for p in pairs}), "held-out state leaked in as a source"
    assert not (forbidden & {p[2] for p in pairs}), "held-out state leaked in as a successor"
    result = train_inverse_model(pairs, PretrainConfig(seed=e, **_exp040.PRETRAIN))
    save_encoder(result.sensory, encoder_path(out_dir, e))
    return {"e": e, "n_pairs": len(pairs), "n_forbidden": len(forbidden),
            "seconds": round(time.time() - t0, 1)}


def base_config(out_dir: Path):
    cells = _exp043.sweep_configs([0], out_dir, depths=[DEPTH])
    if len(cells) != 1:
        raise SystemExit(f"expected one EXP-043 depth-{DEPTH} cell, got {len(cells)}")
    return cells[0]


def cell_tag(e: int, r: int) -> str:
    """Both indices in the tag, because `record_filename` encodes neither."""
    return f"exp069_en{e}rs{r}"


def sweep_configs(out_dir: Path, enc_dir: Path, diagonal_only: bool = False):
    base = base_config(out_dir)
    return [
        replace(
            base,
            tag=cell_tag(e, r),
            seed=r,
            encoder_seed=e,
            split_seed=r,
            train_seed=r,
            encoder_state_path=str(encoder_path(enc_dir, e)),
            out_dir=out_dir,
        )
        for e in GRID
        for r in GRID
        if not diagonal_only or e == r
    ]


def _run(cfg):
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("pretrain", "calibrate", "rl"), required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "pretrain":
        todo = [e for e in GRID
                if not (args.skip_existing and encoder_path(args.out_dir, e).exists())]
        print(f"EXP-069 pretrain: {len(todo)} encoders, union of {len(GRID)} splits forbidden, "
              f"EXP-040 recipe {_exp040.PRETRAIN}", flush=True)
        if args.dry_run or not todo:
            return
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(pretrain_one, e, args.out_dir) for e in todo]
            for f in as_completed(futs):
                r = f.result()
                print(f"  encoder e{r['e']}: {r['n_pairs']:,} pairs, {r['n_forbidden']:,} forbidden, "
                      f"{r['seconds']} s", flush=True)
        return

    missing = [e for e in GRID if not encoder_path(args.out_dir, e).exists()]
    if missing:
        raise SystemExit(f"encoders missing for e={missing}; run --phase pretrain first")

    configs = sweep_configs(args.out_dir, args.out_dir, diagonal_only=args.phase == "calibrate")
    expected = len(GRID) if args.phase == "calibrate" else len(GRID) ** 2
    if len(configs) != expected:
        raise SystemExit(f"expected {expected} cells, built {len(configs)}")
    if any(c.depth != DEPTH or c.readout != "concept" for c in configs):
        raise SystemExit("every cell must be EXP-043's depth-5 concept-readout cell")
    if any("exp040_encoder" in str(c.encoder_state_path) for c in configs):
        raise SystemExit("a shipped E0 encoder slipped in; they LEAK when crossed")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision: the tag must carry both indices")
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-069 phase {args.phase!r}: depth {DEPTH}, {len(configs)} cells, {args.workers} workers",
          flush=True)
    if args.dry_run or not configs:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            print(f"  {i}/{len(configs)}  {r['tag']}  success {r['success_rate']:.3f}", flush=True)
    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
