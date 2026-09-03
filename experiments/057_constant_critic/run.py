"""EXP-057: the critic, with NO state input at all.

EXP-056 established that flattening V(s_t) to its own episode mean costs -0.0646 at p 0.0234 and
collapses the arm to the EMA baseline's level, so within-episode state-dependence is load-bearing.
What it could NOT separate, and says so in its own section 5, is the original question:

    CALIBRATION            the critic's level is fitted by MSE on current data; the EMA LAGS
    BETWEEN-EPISODE STATE  V's episode mean still varies with the scramble

A constant critic separates them. One learned scalar, no state input, fitted by the same MSE loss
with the same optimizer and rate. It differs from arm B in exactly one way (it cannot see the
state) and from the EMA baseline in exactly one way (it is fitted rather than exponentially
averaged).

It is NOT the per-episode batch mean. That arm forms G_t - mean(G), exactly zero on a one-step
episode, and depth 1 averages 1.22 steps per episode; it was disqualified before running.

PRE-REGISTERED CONTRACT: docs/superpowers/specs/2026-09-03-exp057-constant-critic-design.md

  1. PRIMARY: C minus the EMA control. CONFIRMED at >= +0.05, p <= 0.05. The calibration
     hypothesis, asked at last with a control that can answer it.
  2. SECONDARY: C minus arm B. The total cost of removing all state-dependence. Not directional.
  3. TERTIARY: C minus EXP-056's arm F. Isolates the BETWEEN-episode contribution.
  4. VALIDITY GATE: critic_within_rms must be below 1e-6 at every stage, because every timestep
     of an episode reads the same scalar. Above that, the arm is not state-blind and every claim
     is void.

POWER, stated up front: at the measured paired sd of 0.0826, n=12 gives about 28% power for a
+0.05 effect. This experiment is well powered for nothing. It is run because the question is
load-bearing and the arm is cheap.

Run (repo root):
    .venv/bin/python -u experiments/057_constant_critic/run.py --workers 6
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

SEEDS = tuple(range(12))
DEPTH = 7
EPISODES = 10_000
CAP = ((1, 2),)
TAG = "exp057_const_d7"

# FIXED, not selected. EXP-053's pilot chose this blind to success rate, and re-selecting it
# would make this something other than "arm B with one change". There is no new hyperparameter
# in this experiment and therefore no pilot.
CRITIC_LR = 0.01

# Neither control is re-run. Both are on disk with the same 12 seeds and the same E1 encoders.
CONTROLS = {
    "EXP-051, the EMA baseline": (Path("experiments/051_depth7_transfer/outputs"),
                                  "exp051_transfer_d7", 0.1471),
    "arm B, the full critic": (Path("experiments/053_neuromod_stage3/outputs"),
                               "exp053_critic_d7", 0.2004),
    "EXP-056 arm F, within-episode removed": (
        Path("experiments/056_flattened_critic/outputs"), "exp056_flat_d7", 0.1358),
}
BAR = 0.05


def e1_encoder(seed: int) -> Path:
    """EXP-047's fine-tuned encoder, exactly as EXP-053 arm B loaded it."""
    return E1_DIR / f"exp047_ft_d6_lr0.0001_regionalized_d6_s{seed}_sig0.0_encoder.pt"


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    """EXP-053 arm B copied field for field. ONE change: `flatten_critic`."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=TAG,
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=tuple(range(1, DEPTH + 1)), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e1_encoder(seed)),
            critic_lr=CRITIC_LR, constant_critic=True,
            max_depth=DEPTH, out_dir=out_dir,
        )
        for seed in seeds
    ]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(e1_encoder(s)) for s in args.seeds if not e1_encoder(s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} E1 encoder(s), first {missing[:2]}. These are EXP-047's "
            "fine-tuned encoders and they ARE tracked in git; a checkout should have them."
        )

    configs = sweep_configs(args.seeds, args.out_dir)
    # The three ways this stops being "arm B with one change", each fatal rather than a warning.
    if any(not c.constant_critic for c in configs):
        raise SystemExit("EXP-057's whole point is constant_critic=True.")
    if any(c.flatten_critic for c in configs):
        raise SystemExit(
            "flatten_critic is EXP-056's switch and is redundant here: a constant critic has no "
            "within-episode variation to flatten. Setting both would confuse the two arms."
        )
    if any(c.critic_lr != CRITIC_LR for c in configs):
        raise SystemExit(f"critic_lr must stay at EXP-053's selected {CRITIC_LR}.")
    if any(c.normalize_advantages for c in configs):
        raise SystemExit(
            "normalize_advantages subtracts the advantage mean, which is algebraically the "
            "returns' mean. Turning it on imposes the DISQUALIFIED batch-mean baseline on top "
            "of this arm."
        )
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-057: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} runs, "
          f"{args.workers} workers")
    print(f"  arm B with ONE change: the critic is a single learned scalar, 1 parameter")
    print(f"  against arm B's 65, and it CANNOT SEE THE STATE.")
    print(f"  critic_lr {CRITIC_LR} FIXED (EXP-053's blind pilot), encoder FROZEN.")
    for label, (_, _, mean) in CONTROLS.items():
        print(f"  control NOT re-run: {label} = {mean}")
    print(f"  Claim 1 is C minus the EMA control, CONFIRMED at >= +{BAR}, p <= 0.05.")
    print("  POWER: at a paired sd of 0.0826, n=12 gives ~28% power for a +0.05 effect, so")
    print("  'indistinguishable' is the modal outcome on Claims 2 and 3 whatever is true.")
    print("  Every null is a BOUND with an interval, never an equivalence.")
    print("  Claim 4 gate: critic_within_rms must stay below 1e-6 at every stage.\n",
          flush=True)

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
