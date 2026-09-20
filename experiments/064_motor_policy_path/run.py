"""EXP-064: put the brain's OWN pathway on the policy path, and train it.

`Brain.step` computes utilities -> router -> motor -> an action on EVERY step of EVERY experiment
ever run, and that action is consumed only by the dashboard. Training and evaluation both ignore
it and read `head(concept)`. So 34,912 of the brain's 61,728 parameters (56.6%) have been frozen
at random init forever, and 318 of 510 neurons sit off the policy path.

> THE CAPSTONE'S CENTRAL CLAIM HAS NEVER BEEN TESTED. "A regionalized spiking brain that learns
> to solve a cube" rests on a topology that was never on the policy path. The Phase 3 assessment
> established that no arm-versus-arm contrast can answer this without an architecture change.
> This is that change.

> TWO NEW ARMS, CAPACITY-MATCHED TO 0.36%. P trains prefrontal + motor + a 6x6 head (15,540);
> C trains one MLP head at hidden=218 with the brain frozen (15,484). The question is exactly
> whether a spiking prefrontal-to-motor pathway learns a policy as well as a conventional MLP of
> the same size reading the same features.

> THE HIPPOCAMPUS IS OFF IN BOTH ARMS. Memory hurts here (EXP-059/061/063).

Spec: docs/superpowers/specs/2026-09-19-exp064-motor-policy-path-design.md

Usage:
    .venv/bin/python -u experiments/064_motor_policy_path/run.py --workers 6 --skip-existing
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
E0_DIR = Path("experiments/040_pretrained_encoder_policy/outputs")

SEEDS = tuple(range(12))
DEPTH = 5
EPISODES = 10_000
CAP = ((1, 2),)

# region_lr EQUALS the head's lr (CubeConfig.lr defaults to 1e-2) so the policy is optimized as
# one object rather than as a head plus a separately-tuned brain. 1e-3 was also calibrated
# (drift 0.093) and deliberately NOT swept; see the spec's section 4.
REGION_LR = 1e-2
# 71*218 + 6 = 15,484 trainable, against the motor arm's 15,540. A 0.36% capacity gap.
CONTROL_HIDDEN = 218

ARMS = (
    dict(key="P", readout="motor", tag="exp064_motor_d5", region_lr=REGION_LR, head_hidden=None),
    dict(key="C", readout="concept", tag="exp064_mlp_d5", region_lr=None,
         head_hidden=CONTROL_HIDDEN),
)

BAR = 0.05
GATE_MIN_REGION_DRIFT = 0.01    # calibrated 0.598 at this lr; a frozen pathway reads exactly 0.0
GATE_MIN_MOTOR_RATE = 0.02      # calibrated 0.105-0.150 at depth 5; a silent pathway reads 0.0

MOTOR_TRAINABLE = 15_540
CONTROL_TRAINABLE = 15_484


def e0_encoder(seed: int) -> Path:
    """EXP-040's pretrained encoder, frozen, exactly as EXP-059/061/063 loaded it.

    NOT tracked in git. All 24 exist on the laptop's MAIN checkout and in the worktree.
    """
    return E0_DIR / f"exp040_encoder_s{seed}.pt"


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    """EXP-059's arm copied field for field. The readout and what trains are the only changes."""
    return [
        CubeConfig(
            arm="regionalized", readout=arm["readout"], tag=arm["tag"],
            region_lr=arm["region_lr"], head_hidden=arm["head_hidden"],
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=tuple(range(1, DEPTH + 1)), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e0_encoder(seed)),
            max_depth=DEPTH, out_dir=out_dir,
        )
        for arm in ARMS
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

    missing = [str(e0_encoder(s)) for s in args.seeds if not e0_encoder(s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} E0 encoder(s), first {missing[:2]}. exp040_encoder_s*.pt are "
            "NOT tracked in git; they live on the laptop's MAIN checkout."
        )

    configs = sweep_configs(args.seeds, args.out_dir)
    # Each of these makes the experiment answer a different question than the spec claims.
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("the encoder is FROZEN in both arms: encoder_lr must stay None.")
    if any(c.normalize_advantages for c in configs):
        raise SystemExit("normalize_advantages must stay False, as in EXP-059.")
    if any(c.episodes != EPISODES or c.depth != DEPTH for c in configs):
        raise SystemExit("budget and depth must match EXP-059 exactly or the arms are not paired")
    motor = [c for c in configs if c.readout == "motor"]
    ctrl = [c for c in configs if c.readout == "concept"]
    if len(motor) != len(ctrl):
        raise SystemExit("the arms must be paired seed for seed")
    if any(c.region_lr is None for c in motor) or any(c.region_lr is not None for c in ctrl):
        raise SystemExit("only the motor arm may train the regions")
    if any(c.head_hidden is not None for c in motor):
        raise SystemExit("the motor arm's head is a 6x6 affine, not an MLP")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    total = len(configs)
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-064: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} of {total} runs, "
          f"{args.workers} workers")
    print( "  THE CAPSTONE'S CENTRAL CLAIM, TESTED FOR THE FIRST TIME. Before this, 34,912 of the")
    print( "        brain's 61,728 parameters were frozen at random init forever and the")
    print( "        prefrontal->motor pathway fed only the dashboard.")
    print(f"  ARM P {ARMS[0]['tag']!r}: reads MOTOR spike rates; trains prefrontal + motor + a")
    print(f"        6x6 head at region_lr={REGION_LR}. Trainable {MOTOR_TRAINABLE:,}.")
    print(f"  ARM C {ARMS[1]['tag']!r}: reads the concept through an MLP head (hidden="
          f"{CONTROL_HIDDEN}), brain FROZEN. Trainable {CONTROL_TRAINABLE:,}.")
    print( "        Capacity-matched to 0.36%. The ROUTE changes; the capacity does not.")
    print( "  Hippocampus OFF in both arms: memory hurts here (EXP-059/061/063).")
    print(f"  CLAIM 1 (PRIMARY) is P - C, NOT directional, bar +/-{BAR}, exact over 2**12 = 4,096")
    print( "        flips. A NULL means the brain's own pathway is neither better nor measurably")
    print( "        worse than a conventional MLP of the same size - notable, and still a BOUND.")
    print(f"  GATE 1: arm P mean region_drift >= {GATE_MIN_REGION_DRIFT} (calibrated 0.598; a")
    print( "        frozen pathway reads exactly 0.0, so it can fail AND pass)")
    print(f"  GATE 2: arm P mean motor_rate_mean >= {GATE_MIN_MOTOR_RATE} (calibrated 0.105-0.150")
    print( "        at depth 5; the pathway was silent on 67% of steps at depth 1, so this is a")
    print( "        real risk and not a formality)\n", flush=True)

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
            drift = r.get("region_drift")
            rate = r.get("motor_rate_mean")
            print(f"  {i}/{len(configs)}  {r['tag']} s{r['seed']}  "
                  f"success {r['success_rate']:.3f}  "
                  f"drift {drift if drift is None else f'{drift:.4f}'}  "
                  f"rate {rate if rate is None else f'{rate:.4f}'}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
