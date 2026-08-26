# experiments/052_pretraining_optimum/pretrain_sweep.py
"""EXP-052 phase 1: pretrain encoders FROM SCRATCH at several epoch counts.

40 epochs was never chosen by measurement. EXP-039 calibrated the learning RATE and recorded the
epoch count as fixed configuration, and every cube result since EXP-040 starts from a 40-epoch
encoder that was never compared against any other.

The curve is already bracketed - 0 epochs gives 0.0000 (EXP-036), 40 gives 0.1800, 80 gives
0.0887 (EXP-050) - so it rises from 0 and falls by 80 and an interior optimum must exist. This
builds the encoders that locate it.

FROM SCRATCH, not warm-started, so each arm is "N epochs of pretraining" rather than "40 then
N-40 more with a fresh head". Everything else is EXP-040's configuration unchanged, including its
`rl_heldout_union` exclusions - without those an arm could win by leakage rather than by epochs.

Run (repo root):
    .venv/bin/python -u experiments/052_pretraining_optimum/pretrain_sweep.py --workers 10
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.envs.cube_distance import ExactBFSDistance
from neuromorphic.training.cube_baseline import shell_states
from neuromorphic.training.encoder_pretrain import (
    PretrainConfig,
    build_pairs,
    save_encoder,
    train_inverse_model,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

_EXP040_PATH = HERE.parent / "040_pretrained_encoder_policy" / "run.py"
_spec = importlib.util.spec_from_file_location("exp040_run", _EXP040_PATH)
exp040 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp040)

PRETRAIN = dict(exp040.PRETRAIN)          # epochs=40, batch_size=256, lr=3e-3
PRETRAIN_DEPTHS = list(exp040.PRETRAIN_DEPTHS)
EPOCH_ARMS = (10, 20)                     # 40 exists as E0; 80 exists as EXP-050's E0+


def encoder_path(out_dir: Path, epochs: int, seed: int) -> Path:
    return out_dir / f"exp052_encoder_e{epochs}_s{seed}.pt"


def pretrain_one(epochs: int, seed: int, out_dir: Path) -> dict:
    torch.set_num_threads(1)
    # `main` already creates this, but a direct call (a diagnostic, a notebook) otherwise dies
    # only at the very end, after paying for the whole pretrain. Cheap insurance.
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    provider = ExactBFSDistance(max_depth=max(PRETRAIN_DEPTHS) + 1)

    states = []
    for d in PRETRAIN_DEPTHS:
        states.extend(shell_states(provider, d))

    forbidden = exp040.rl_heldout_union(provider, seed)
    pairs = build_pairs(states, forbidden=forbidden)
    assert not (forbidden & {p[0] for p in pairs}), "RL held-out state leaked in as a source"
    assert not (forbidden & {p[2] for p in pairs}), "RL held-out state leaked in as a successor"

    # From scratch: no `sensory=`, so `train_inverse_model` builds make_sensory(cfg.seed) itself,
    # exactly as EXP-040 did. Only `epochs` differs from EXP-040's configuration.
    cfg = PretrainConfig(seed=seed, **{**PRETRAIN, "epochs": epochs})
    result = train_inverse_model(pairs, cfg)
    save_encoder(result.sensory, encoder_path(out_dir, epochs, seed))

    return {"epochs": epochs, "seed": seed, "n_pairs": len(pairs),
            "move_accuracy": result.final_accuracy,
            "seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=list(EPOCH_ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [(e, s) for e in args.epochs for s in args.seeds
            if not encoder_path(args.out_dir, e, s).exists()]

    print(f"EXP-052 phase 1: epoch arms {args.epochs}, {len(args.seeds)} seeds, "
          f"{len(todo)} encoder(s) to build, {args.workers} workers")
    print(f"  FROM SCRATCH. {PRETRAIN} with `epochs` varied; EXP-040's exclusions unchanged.")
    print(f"  40 epochs already exists as E0 (0.1800); 80 as EXP-050's E0+ (0.0887).\n", flush=True)
    if not todo:
        print("nothing to do.")
        return
    if args.dry_run:
        print(f"  --dry-run: {len(todo)} encoder(s) NOT started.")
        return

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(pretrain_one, e, s, args.out_dir): (e, s) for e, s in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            print(f"  {i}/{len(todo)} e{r['epochs']} seed {r['seed']}: "
                  f"move-acc {r['move_accuracy']:.3f}, {r['seconds']}s", flush=True)

    for e, s in [(e, s) for e in args.epochs for s in args.seeds]:
        if not encoder_path(args.out_dir, e, s).exists():
            raise SystemExit(f"encoder e{e} seed {s} missing after phase 1")
    print(f"\ndone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
