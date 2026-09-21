"""EXP-066: can the brain's own pathway learn AT ALL? A `region_lr` sweep.

EXP-064 put prefrontal -> router -> motor on the policy path and trained it. The mechanism
worked (drift 2.7051, firing 0.1534, zero silent steps, gradient verified at every region
parameter). The experiment still failed: both arms scored exactly 0.0000, because the
capacity-matched control collapsed and a contrast between two arms on the floor is 0 by
construction.

> THIS EXPERIMENT HAS NO CONTROL, AND THAT IS THE DESIGN POINT. It asks a THRESHOLD question
> with a one-sample reading, the same shape as EXP-062's depth-8 containment test. There is no
> control arm, therefore no control can break and zero out the comparison. EXP-064's failure
> mode is structurally unreachable here.

> THE HYPOTHESIS comes from EXP-064's own RESULTS, where it is recorded as a hypothesis and not
> a finding: region_drift of 2.71 means the regions moved nearly three times their own initial
> magnitude, consistent with region_lr=1e-2 destroying the pathway rather than training it.

> NO CODE IS ADDED. `readout="motor"` and `region_lr` shipped in EXP-064 and are unchanged, so
> EXP-064's 1e-2 arm is reusable as a third point without an inertness proof: only a
> configuration differs.

> IF ALL THREE ARMS ARE AT THE FLOOR THAT IS A REAL RESULT, not a failed experiment: the
> pathway does not learn anywhere across three orders of magnitude, on a config where a
> 390-parameter linear head scores 0.3229. Stated before any number exists.

Spec: docs/superpowers/specs/2026-09-20-exp066-region-lr-sweep-design.md

Usage:
    .venv/bin/python -u experiments/066_region_lr_sweep/run.py --workers 6 --skip-existing
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

# Three points across three orders of magnitude. 1e-2 is EXP-064's arm, already measured.
ARMS = (
    dict(key="L4", region_lr=1e-4, tag="exp066_motor_lr1e4"),
    dict(key="L3", region_lr=1e-3, tag="exp066_motor_lr1e3"),
)

EXP064_DIR = Path("experiments/064_motor_policy_path/outputs")
EXP064_TAG, EXP064_LR, EXP064_MEAN = "exp064_motor_d5", 1e-2, 0.0000

# Context, NEVER a control: 390 trainable against the motor arm's 15,540, a 40x difference.
EXP043_REFERENCE = 0.3229

BAR = 0.02                      # EXP-062's floor bar, reused unchanged so readings stay comparable
GATE_MIN_REGION_DRIFT = 0.001   # a frozen pathway reads exactly 0.0; EXP-064 read 2.7051
GATE_MIN_MOTOR_RATE = 0.02      # calibrated 0.105-0.150 at depth 5; EXP-064's arm read 0.1534
MOTOR_TRAINABLE = 15_540


def e0_encoder(seed: int) -> Path:
    """EXP-040's pretrained encoder, frozen, exactly as EXP-064 loaded it."""
    return E0_DIR / f"exp040_encoder_s{seed}.pt"


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    """EXP-064 arm P copied field for field. ONE change per arm: `region_lr`."""
    return [
        CubeConfig(
            arm="regionalized", readout="motor", tag=arm["tag"], region_lr=arm["region_lr"],
            head_hidden=None,
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
    # Each of these makes the sweep answer a different question than the spec claims.
    if any(c.readout != "motor" for c in configs):
        raise SystemExit("every cell must use the motor readout")
    if any(c.region_lr is None for c in configs):
        raise SystemExit("every cell must train the regions; region_lr is the ONLY variable")
    if any(c.head_hidden is not None for c in configs):
        raise SystemExit(
            "head_hidden must stay None. An MLP head is what collapsed EXP-064's control from "
            "EXP-043's 0.3229 to 0.0000, and it is not part of this question."
        )
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("the encoder is FROZEN: encoder_lr must stay None.")
    if any(c.episodes != EPISODES or c.depth != DEPTH for c in configs):
        raise SystemExit("budget and depth must match EXP-064 exactly or the sweep is not paired")
    lrs = {c.region_lr for c in configs}
    if lrs != {a["region_lr"] for a in ARMS}:
        raise SystemExit(f"unexpected region_lr set {lrs}")
    if EXP064_LR in lrs:
        raise SystemExit(f"region_lr {EXP064_LR} is EXP-064's arm and must be REUSED, not re-run")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    total = len(configs)
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-066: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} of {total} runs, "
          f"{args.workers} workers")
    print( "  CAN THE BRAIN'S OWN PATHWAY LEARN AT ALL? A one-sample THRESHOLD question, so")
    print( "        there is NO CONTROL - and therefore no control that can break and zero out")
    print( "        the comparison, which is exactly how EXP-064 failed.")
    for a in ARMS:
        print(f"  ARM {a['key']}: region_lr {a['region_lr']:g}, tag {a['tag']!r}")
    print(f"  THIRD POINT, REUSED: EXP-064's {EXP064_TAG!r} at region_lr {EXP064_LR:g}, "
          f"measured {EXP064_MEAN:.4f}")
    print(f"  CLAIM 1 (PRIMARY), per arm: mean >= {BAR} AND 95% CI lower bound > 0.")
    print(f"        {BAR} is EXP-062's floor bar, reused so the readings stay comparable.")
    print( "        ALL THREE AT THE FLOOR IS A REAL RESULT, not a failed experiment.")
    print(f"  CONTEXT, never a control: EXP-043's linear head scored {EXP043_REFERENCE} on this")
    print( "        config at 390 trainable params against this arm's 15,540, a 40x gap.")
    print(f"  GATE 1: region_drift >= {GATE_MIN_REGION_DRIFT} (frozen reads exactly 0.0). NO upper")
    print( "        bound: the 10k-episode drift at these lrs has never been measured, and a")
    print( "        ceiling guessed from a 120-episode calibration is EXP-057's regime error.")
    print(f"  GATE 2: motor_rate_mean >= {GATE_MIN_MOTOR_RATE} (silent reads 0.0)\n", flush=True)

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
                  f"success {r['success_rate']:.3f}  drift {r['region_drift']:.4f}  "
                  f"rate {r['motor_rate_mean']:.4f}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
