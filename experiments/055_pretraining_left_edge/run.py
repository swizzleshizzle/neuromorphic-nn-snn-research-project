"""EXP-055 phase 3: the RL arms at 1, 2, 3 and 5 pretraining epochs.

    epochs   depth-6 policy      S (EXP-054)
      0          0.0000            0.0100     EXP-036 policy, EXP-054 S
    1,2,3,5    UNMEASURED        UNMEASURED   <- this experiment
     10          0.2012            0.0242     EXP-052
     20          0.1850            0.0241     EXP-052
     40          0.1800            0.0246     EXP-043
     80          0.0887            0.0244     EXP-050

Both curves are flat from 10 onward. Both have one unexplained jump at the left edge.

PRE-REGISTERED CONTRACT: docs/superpowers/specs/2026-08-31-exp055-pretraining-left-edge-design.md

  1. PRIMARY: e10 minus e1. CONFIRMED at >= +0.05, p <= 0.05, meaning epochs 2 to 10 buy
     something real. IF NOT SIGNIFICANT THE OUTPUT IS A BOUND, NOT AN EQUIVALENCE.
  2. THE FLOOR: e1 against 0 epochs (EXP-036, 0.0000 on all twelve seeds). If e1 already lands
     near 0.20, pretraining's contribution is almost entirely escaping random init.
  3. SHAPE: adjacent contrasts, and a shape may be named ONLY where one is significant.
  4. Does S saturate at the same point as policy? A dissociation would separate "the encoder
     has the structure" from "the policy can use it".

Every arm is FROZEN (390 trainable) and runs 10,000 episodes, so there is no episode-budget
confound and EXP-046's curve does not apply.

Run (repo root), after phase 1:
    .venv/bin/python -u experiments/055_pretraining_left_edge/run.py --epochs 1 --workers 6
"""

from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import (
    CubeConfig,
    record_filename,
    run_cube_baseline,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

_PRETRAIN_PATH = HERE / "pretrain_left_edge.py"
_spec = importlib.util.spec_from_file_location("exp055_pretrain", _PRETRAIN_PATH)
_pretrain_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pretrain_mod)
encoder_path = _pretrain_mod.encoder_path
EPOCH_ARMS = _pretrain_mod.EPOCH_ARMS
SEEDS = _pretrain_mod.SEEDS

DEPTH = 6
EPISODES = 10_000
CAP = ((1, 2),)

# The anchors, none of which is re-run: (records dir, tag, published mean).
ANCHORS = {
    10: (Path("experiments/052_pretraining_optimum/outputs"), "exp052_e10_d6", 0.2012),
    20: (Path("experiments/052_pretraining_optimum/outputs"), "exp052_e20_d6", 0.1850),
    40: (Path("experiments/043_cap_at_depth_5_6/outputs"), "exp043_capped_d6", 0.1800),
    80: (Path("experiments/050_objective_vs_gradient/outputs"), "exp050_pre2_d6", 0.0887),
}
# 0 epochs is EXP-036, measured at exactly 0.0000 on all twelve seeds, so it has no variance
# and is handled descriptively rather than as a paired contrast. See spec Claim 2.
ZERO_EPOCH_MEAN = 0.0000

BAR = 0.05


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def tag_for(epochs: int) -> str:
    """`record_filename` does not encode the encoder, so the epoch count must live in the tag
    or the four arms would silently overwrite each other."""
    return f"exp055_e{epochs}_d{DEPTH}"


def sweep_configs(seeds, out_dir: Path, epoch_arms) -> list[CubeConfig]:
    """EXP-052's Phase 2 copied field for field. ONE variable: which encoder, hence epochs."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=tag_for(e),
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(DEPTH), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(encoder_path(out_dir, e, seed)),
            max_depth=DEPTH, out_dir=out_dir,
        )
        for e in epoch_arms
        for seed in seeds
    ]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, nargs="+", default=list(EPOCH_ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(encoder_path(args.out_dir, e, s))
               for e in args.epochs for s in args.seeds
               if not encoder_path(args.out_dir, e, s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} encoder(s), first {missing[:3]}. Run phase 1 first: "
            "pretrain_left_edge.py"
        )

    configs = sweep_configs(args.seeds, args.out_dir, args.epochs)
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("EXP-055 arms are FROZEN: encoder_lr must stay None.")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-055: depth {DEPTH}, {EPISODES:,} episodes, epochs {tuple(args.epochs)}, "
          f"{len(configs)} runs, {args.workers} workers")
    print(f"  FROZEN encoder, 390 trainable. ONE VARIABLE vs EXP-052's phase 2: the epoch count.")
    print(f"  Anchors NOT re-run: 0 -> {ZERO_EPOCH_MEAN}, "
          + ", ".join(f"{e} -> {v[2]}" for e, v in sorted(ANCHORS.items())))
    print(f"  Claim 1 is e10 minus e1, CONFIRMED at >= +{BAR}, p <= 0.05. A NON-significant")
    print(f"  result is reported as a BOUND, never as an equivalence.\n", flush=True)

    if not configs:
        print("nothing to do.")
        return
    if args.dry_run:
        print(f"  --dry-run: {len(configs)} cell(s) NOT started.")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
