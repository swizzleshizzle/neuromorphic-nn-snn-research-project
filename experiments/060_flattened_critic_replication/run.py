"""EXP-060: an independent replication of EXP-056's flattened critic, on FRESH seeds.

EXP-056 found that flattening `V(s_t)` to its own episode mean costs -0.0646 at p 0.0234, against
a Bonferroni threshold of 0.025 - it cleared by 6.4% of its margin, at n=12, and two experiments
lean on it. This re-runs both arms on seeds 14-23, which have never been used for this question.

> THE PRIMARY IS SEEDS 14-23 ALONE, NOT A POOLED n=22. Extending an experiment because its
> p-value was marginal and then pooling is optional stopping: the decision to add seeds was made
> after seeing a significant result, so a pooled p-value inherits that selection. The pooled
> figure is reported as a SECONDARY with that sentence attached. See the spec.

> BOTH ARMS ARE RE-RUN. EXP-053's arm B exists only at seeds 0-11 and the contrast is paired by
> seed, so pairing a new `F` seed against an old `B` seed would compare different seeds. Arm B is
> not a free control here.

Spec: docs/superpowers/specs/2026-09-10-exp060-flattened-critic-replication-design.md

Usage:
    .venv/bin/python -u experiments/060_flattened_critic_replication/run.py --workers 10
"""

from __future__ import annotations

import argparse
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
E1_DIR = Path("experiments/047_encoder_finetuning/outputs")

# FRESH seeds only. 0-11 are EXP-056's own cells and 12-13 were EXP-047's lr pilot, so every seed
# below 14 is contaminated for this question in one way or the other.
SEEDS = tuple(range(14, 24))
FIRST_FRESH_SEED = 14
DEPTH = 7
EPISODES = 10_000
CAP = ((1, 2),)

# FIXED, not selected, exactly as in EXP-056. EXP-053's pilot chose this blind to success rate and
# re-selecting it would make this something other than "EXP-056 again on new seeds". There is no
# new hyperparameter here and therefore no pilot.
CRITIC_LR = 0.01

ARMS = {
    "B": (False, "exp060_full_d7"),    # the full critic, EXP-053 arm B
    "F": (True,  "exp060_flat_d7"),    # flattened, EXP-056 arm F
}

# EXP-056's own numbers, for the banner only. NOT controls: no claim is paired against them, and
# the pooled comparison is a secondary computed in the aggregator, not here.
EXP056_F = 0.1358
EXP056_B = 0.2004
BAR = 0.05


def e1_encoder(seed: int) -> Path:
    """EXP-047's fine-tuned encoder, exactly as EXP-053 arm B and EXP-056 loaded it.

    NOT tracked in git. Exists for seeds 0-13 only; 14-23 must be manufactured first, and that
    chain is longer than it looks - EXP-047's confirm mode refuses without EXP-043's depth-6
    baseline records for the same seeds, which exist only for 0-11. See the launcher's phases.
    """
    return E1_DIR / f"exp047_ft_d6_lr0.0001_regionalized_d6_s{seed}_sig0.0_encoder.pt"


def sweep_configs(seeds, out_dir: Path, arms) -> list[CubeConfig]:
    """EXP-056 copied field for field. The ONLY variable is `flatten_critic`."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=tag,
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=tuple(range(1, DEPTH + 1)), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e1_encoder(seed)),
            critic_lr=CRITIC_LR, flatten_critic=flatten,
            max_depth=DEPTH, out_dir=out_dir,
        )
        for key in arms
        for flatten, tag in [ARMS[key]]
        for seed in seeds
    ]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # The whole point of this experiment is that its seeds are uncontaminated. A seed below 14
    # would silently turn an independent replication into a re-run of cells that already exist.
    stale = [s for s in args.seeds if s < FIRST_FRESH_SEED]
    if stale:
        raise SystemExit(
            f"seeds {stale} are below {FIRST_FRESH_SEED} and are NOT fresh: 0-11 are EXP-056's own "
            "cells and 12-13 were EXP-047's lr pilot. This experiment exists to be an INDEPENDENT "
            "replication, and a contaminated seed destroys that without changing anything visible "
            "in the numbers."
        )

    missing = [str(e1_encoder(s)) for s in args.seeds if not e1_encoder(s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} E1 encoder(s), first {missing[:2]}. They are NOT tracked in "
            "git and exist only for seeds 0-13. Manufacturing them for 14-23 needs EXP-043's "
            "depth-6 baseline for those seeds FIRST (it exists only for 0-11), then EXP-047 "
            "--mode confirm. Run the launcher's `baseline` and `finetune` phases in that order."
        )

    configs = sweep_configs(args.seeds, args.out_dir, args.arms)
    if any(c.critic_lr is None for c in configs):
        raise SystemExit("every arm here has a critic; critic_lr must not be None.")
    if any(c.normalize_advantages for c in configs):
        raise SystemExit("normalize_advantages must stay False, as in EXP-053 arm B.")
    if len({(c.tag, c.seed) for c in configs}) != len(configs):
        raise SystemExit("duplicate (tag, seed) cells")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-060: independent replication of EXP-056, depth {DEPTH}, {EPISODES:,} episodes, "
          f"{len(configs)} runs, {args.workers} workers")
    print(f"  seeds {tuple(args.seeds)} - FRESH, never used for this question.")
    print(f"  arms {tuple(args.arms)}, ONE variable: flatten_critic. Encoder FROZEN (E1).")
    print(f"  PRIMARY is seeds 14-23 ALONE at n={len(args.seeds)}, NOT a pooled n=22. Pooling after")
    print(f"  extending a marginal result is optional stopping; the pooled figure is a SECONDARY.")
    print(f"  Claim 1 bar {-BAR:+.2f} on success, directional (EXP-056 found F BELOW B).")
    print(f"  POWER: ~50-60% at EXP-056's -0.0646, and that estimate is upward-biased because it")
    print(f"        was selected for significance. A null primary is a BOUND, not a refutation.")
    print(f"  context, not controls: EXP-056 measured F {EXP056_F}, B {EXP056_B}\n", flush=True)

    if not configs:
        print("nothing to do.")
        return
    if args.dry_run:
        print(f"  --dry-run: {len(configs)} cell(s) NOT started.")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            print(f"  {i}/{len(configs)}  {r['tag']} s{r['seed']}  "
                  f"success {r['success_rate']:.3f}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
