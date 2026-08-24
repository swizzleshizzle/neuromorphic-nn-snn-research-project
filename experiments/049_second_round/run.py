# experiments/049_second_round/run.py
"""EXP-049 driver: does the two-stage recipe compound, or buy back its own compute?

EXP-047 fine-tuned the encoder during RL; EXP-048 froze the result and trained a fresh head.
Depth 6 went 0.1800 -> 0.3112. But priced against EXP-046's budget curve, BOTH arms beat it by
about the same +0.05:

    arm                          compute  budget-equiv  actual  excess
    A  E0 frozen, fresh head      1.00     +0.0000      0.1800    -
    C  E1 joint (EXP-047)         1.33     +0.0272      0.2700  +0.0628
    B  E1 frozen, fresh (EXP-048) 2.33     +0.0808      0.3112  +0.0504

That is a CONSTANT-RETURN model. This runs the third and fourth points to test it:

    D  E1 fine-tuned AGAIN -> E2, co-adapted head   2.66  +0.0935
    E  E2 frozen, fresh head                        3.66  +0.1240

ONE VARIABLE between arm E and arm B: whether the encoder went through one fine-tuning round or
two. `encoder_lr` stays at 1e-4 and is deliberately NOT re-piloted - to ask whether a second
IDENTICAL round helps, the round must be identical. Verified safe: round 1 left the weight scale
unchanged (|fc1| 72.048 -> 72.074) and moved the weights by only 2.90% of their norm.

PRE-REGISTERED CONTRACT, committed at aaca17c before any number existed. Full version:
docs/superpowers/specs/2026-08-24-exp049-second-round-design.md

  1. PRIMARY, PAIRED. E minus B. CONFIRMED at >= +0.05, p <= 0.05.

     >> I PREDICT THIS REFUTES. The constant-return model puts arm E near 0.354, so E - B is
     >> about +0.043, BELOW the bar. Refutation is the prediction; CONFIRMATION would mean the
     >> recipe compounds. The extra round's own compute is worth +0.0432 on the budget curve, so
     >> a delta between +0.0432 and +0.05 is pre-committed as UNINTERPRETABLE, not a near-miss.

  2. THE REAL QUANTITY, descriptive: arm E's excess over its budget-equivalent (E - 0.3040),
     read beside C's +0.0628 and B's +0.0504. ~+0.05 = constant returns; >+0.08 = compounding;
     <+0.03 = diminishing, i.e. the first round was special.

  3. D minus C, free - arm D has to run anyway to produce E2. Does the joint arm improve too?

  4. MECHANISM, falsifiable. EXP-048 localised the gain to TRAJECTORIES: revisits fell
     0.4652 -> 0.3808 (p 0.0454), optimality rose 0.6445 -> 0.7716 (p 0.0132), while single-step
     probe accuracy went slightly DOWN. Arm E should push both further. If success rises while
     both stay flat, that mechanism does not generalise and the explanation is incomplete.

  5. The probe should drift DOWN again (E2 below E1). A rise would break the decoupling story.

  6. A refuted Claim 1 with Claim 2 near +0.05 is the EXPECTED result and is not a
     disappointment: it makes the cost of any target depth calculable and closes "just iterate
     it" as a strategy. The next move would be a different second-stage objective, not a round 3.

Run (repo root), in this order - arm E CANNOT start until arm D has written all 12 encoders:
    .venv/bin/python -u experiments/049_second_round/run.py --arm D --workers 6
    .venv/bin/python -u experiments/049_second_round/run.py --arm E --workers 6
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import (
    CubeConfig,
    curriculum_schedule,
    encoder_filename,
    max_steps_for,
    record_filename,
    run_cube_baseline,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
E1_DIR = Path("experiments/047_encoder_finetuning/outputs")
ARM_B_DIR = Path("experiments/048_fresh_head/outputs")
ARM_C_DIR = E1_DIR

DEPTH = 6
EPISODES = 10_000
CAP = ((1, 2),)
SEEDS = tuple(range(12))
ENCODER_LR = 1e-4            # EXP-047's selected rate, reused ON PURPOSE

ARM_A_MEAN = 0.1800
ARM_B_MEAN = 0.3112
ARM_C_MEAN = 0.2700
ARM_E_BUDGET_EQUIV = 0.3040  # 0.1800 + 0.22*log10(3.66)
BAR = 0.05
E_MINUS_B_BUDGET_EQUIV = 0.0432

TAG_D = f"exp049_ft2_d{DEPTH}"
TAG_E = f"exp049_fresh2_d{DEPTH}"


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def e1_encoder(seed: int) -> Path:
    """EXP-047's fine-tuned encoder for this seed - the input to round 2."""
    return E1_DIR / (f"exp047_ft_d{DEPTH}_lr{ENCODER_LR:g}_regionalized_"
                     f"d{DEPTH}_s{seed}_sig0.0_encoder.pt")


def e2_encoder(out_dir: Path, seed: int) -> Path:
    """Arm D's twice-fine-tuned encoder. Written by arm D, read by arm E."""
    return out_dir / encoder_filename(_cfg_d(seed, out_dir))


def _cfg_d(seed: int, out_dir: Path) -> CubeConfig:
    return CubeConfig(
        arm="regionalized", readout="concept", tag=TAG_D,
        depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
        curriculum=curriculum_for(DEPTH), max_steps_by_depth=CAP,
        entropy_beta=0.0, normalize_advantages=False,
        encoder_state_path=str(e1_encoder(seed)),
        encoder_lr=ENCODER_LR,
        max_depth=DEPTH, out_dir=out_dir,
    )


def _cfg_e(seed: int, out_dir: Path) -> CubeConfig:
    return CubeConfig(
        arm="regionalized", readout="concept", tag=TAG_E,
        depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
        curriculum=curriculum_for(DEPTH), max_steps_by_depth=CAP,
        entropy_beta=0.0, normalize_advantages=False,
        encoder_state_path=str(e2_encoder(out_dir, seed)),
        encoder_lr=None,          # FROZEN. Arm E's whole point.
        max_depth=DEPTH, out_dir=out_dir,
    )


def env_steps(finetune: bool) -> int:
    override = dict(CAP)
    train = sum(n * override.get(d, max_steps_for(d))
                for d, n in curriculum_schedule(curriculum_for(DEPTH), EPISODES, None))
    return int(train * (1.33 if finetune else 1.0)) + 6_000


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["D", "E"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.arm == "D":
        for s in args.seeds:
            if not e1_encoder(s).exists():
                raise SystemExit(
                    f"missing EXP-047 encoder {e1_encoder(s)}. Round 2 starts from round 1's "
                    "output; those files are tracked in git."
                )
        configs = [_cfg_d(s, args.out_dir) for s in args.seeds]
    else:
        missing = [s for s in args.seeds if not e2_encoder(args.out_dir, s).exists()]
        if missing:
            raise SystemExit(
                f"arm D has not written encoders for seeds {missing}. Arm E freezes arm D's "
                "OUTPUT, so it cannot start until arm D has finished. Run --arm D first."
            )
        configs = [_cfg_e(s, args.out_dir) for s in args.seeds]

    # The two arms differ in exactly one field, and getting it backwards would silently answer a
    # different question: arm D must train the encoder, arm E must not.
    want_lr = ENCODER_LR if args.arm == "D" else None
    if any(c.encoder_lr != want_lr for c in configs):
        raise SystemExit(f"arm {args.arm} requires encoder_lr={want_lr!r}")

    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit(f"record filename collision: {sorted({n for n in names if names.count(n) > 1})[:5]}")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    ft = args.arm == "D"
    print(f"EXP-049 arm {args.arm}: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} runs, "
          f"{args.workers} workers")
    if ft:
        print(f"  ROUND 2 fine-tuning from EXP-047's encoders, encoder_lr {ENCODER_LR:g} "
              f"(reused on purpose, NOT re-piloted)")
        print(f"  trainable 27,206. Writes E2, which arm E then freezes.")
        print(f"  vs arm C ({ARM_C_MEAN}): does a second round help the joint arm too?")
    else:
        print(f"  arm E: arm D's twice-fine-tuned encoder, FROZEN, with a FRESH head")
        print(f"  trainable back to 390. ONE VARIABLE vs arm B ({ARM_B_MEAN}): one round or two.")
        print(f"  CONFIRMED at >= +{BAR} over arm B, p <= 0.05.")
        print(f"  >> PREDICTED TO REFUTE. Constant returns put arm E near 0.354, i.e. E-B ~ +0.043.")
        print(f"  >> The extra round's compute alone is worth +{E_MINUS_B_BUDGET_EQUIV}, so")
        print(f"  >> +{E_MINUS_B_BUDGET_EQUIV} to +{BAR} is UNINTERPRETABLE, not a near-miss.")
        print(f"  Claim 2 carries the answer: excess over budget-equivalent {ARM_E_BUDGET_EQUIV}")
        print(f"  beside arm C's +0.0628 and arm B's +0.0504.")
    print(f"  {env_steps(ft):,} equivalent env steps per run\n", flush=True)

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
