# experiments/048_fresh_head/run.py
"""EXP-048 driver: is EXP-047's fine-tuned encoder better, or just co-adapted to its head?

EXP-047 took depth 6 from 0.1800 to 0.2700 by fine-tuning the sensory encoder during RL
(+0.0900, p 0.0020). Its MECHANISM claim split: the standard probe said the representation
improved (+0.0398 at depth 4, 11-0), the leak-free slice said it did not (+0.0050, p 0.5732),
and the probe's gain went NEGATIVE at depth 6 - the depth the policy is actually scored on.

The remaining explanation was co-adaptation: the encoder became a better feature extractor FOR
THAT HEAD without becoming a better encoder of optimality. That predicts the gain does not
transfer to a fresh head. This runs the test.

    A  EXP-040 encoder,  FROZEN, fresh head   ->  0.1800   (EXP-043, NOT re-run)
    B  EXP-047 encoder,  FROZEN, fresh head   ->  ?        (THIS EXPERIMENT)
    C  EXP-047 encoder, trained, its own head ->  0.2700   (EXP-047, NOT re-run)

ONE VARIABLE between A and B: which frozen encoder the fresh head reads. And because B is
FROZEN, it costs 1.0x per step exactly like A - unlike EXP-047 there is no compute confound to
price here.

NO TRAINER CHANGE. EXP-047's serialised encoders load through EXP-040's existing
`encoder_state_path` seam with `encoder_lr=None`. Verified 2026-08-23: state-dict keys match,
weights load exactly, and the loaded encoder differs from its EXP-040 starting point by 6.49e-02
max on `fc1.weight`.

PRE-REGISTERED CONTRACT, committed at 652b517 before any number existed. Full version:
docs/superpowers/specs/2026-08-23-exp048-fresh-head-design.md

  1. PRIMARY, PAIRED. B minus A. CONFIRMED at >= +0.05, p <= 0.05. "Did the encoder itself get
     better", stripped of the head trained alongside it.

  2. B minus C, with its exact p. Significantly negative means part of EXP-047's gain lived in
     the pairing. A NULL IS NOT EVIDENCE OF EQUIVALENCE and is pre-committed not to be called one.

  3. RETENTION, descriptive: (B - 0.1800) / 0.0900. 1.0 = the encoder carries all of EXP-047's
     gain; 0.0 = none of it, and the whole effect was co-adaptation.

  4. THE INTERPRETATION GRID IS FIXED IN THE SPEC, including the incoherent cell: Claim 1
     refuted AND Claim 2 null means B is indistinguishable from both a 0.1800 arm and a 0.2700
     arm, i.e. UNDERPOWERED, not "partial". Do not interpret that cell.

  5. A refuted Claim 1 is the MOST informative outcome available and is a real result: it would
     align three independent measurements and force EXP-047's headline to be restated as "joint
     training beats training the head alone" rather than "the encoder improves".

> WHAT THIS DOES NOT CONTROL FOR: whether it is RL's objective or merely more gradient steps of
> any kind. There is no honest exchange rate between RL episodes and pretraining epochs, so that
> control would smuggle in a free parameter. See spec section 4. A confirmed Claim 1 licenses
> "the fine-tuned encoder is better", NOT "the RL objective is what made it better".

Run (repo root):
    .venv/bin/python -u experiments/048_fresh_head/run.py --workers 6
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import (
    CubeConfig,
    curriculum_schedule,
    max_steps_for,
    record_filename,
    run_cube_baseline,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
FINETUNED = Path("experiments/047_encoder_finetuning/outputs")
ARM_A = Path("experiments/043_cap_at_depth_5_6/outputs")

DEPTH = 6
EPISODES = 10_000            # matched to A and C exactly
CAP = ((1, 2),)              # EXP-042's confirmed arm, unchanged
SEEDS = tuple(range(12))
SELECTED_LR = 1e-4           # the rate EXP-047's pilot chose; its confirmatory encoders

ARM_A_MEAN = 0.1800          # EXP-043 depth 6, frozen EXP-040 encoder + fresh head
ARM_C_MEAN = 0.2700          # EXP-047, fine-tuned encoder + co-adapted head
EXP047_GAIN = 0.0900
BAR = 0.05

TAG = f"exp048_freshhead_d{DEPTH}"


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def finetuned_encoder(seed: int) -> Path:
    """EXP-047's CONFIRMATORY encoder for this seed, at the selected rate.

    Seeds 12-13 also have encoders in that directory, at three different rates - they are the
    PILOT. They are not used here: EXP-048 explains the +0.0900 that seeds 0-11 produced.
    """
    return FINETUNED / (f"exp047_ft_d{DEPTH}_lr{SELECTED_LR:g}_regionalized_"
                        f"d{DEPTH}_s{seed}_sig0.0_encoder.pt")


def env_steps(episodes: int = EPISODES) -> int:
    override = dict(CAP)
    return sum(n * override.get(d, max_steps_for(d))
               for d, n in curriculum_schedule(curriculum_for(DEPTH), episodes, None))


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    """Copied from EXP-043's depth-6 cell in EVERY field except `encoder_state_path` and the tag.

    `encoder_lr` is left at its default None, which is the point: the encoder is FROZEN. If this
    ever gains an `encoder_lr`, the experiment silently becomes EXP-047 again and answers nothing.
    """
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=TAG,
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(DEPTH), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(finetuned_encoder(seed)),
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
    # SIX, not ten. 12 cells on 10 workers is two waves with the second running 8 workers idle;
    # 12 on 6 is two clean waves, and 6 workers measured 0.115 s/step against 10 workers' ~0.16.
    # 6.4 h against 8.8 h. See the 2026-08-22 correction in the remote-runs playbook.
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the pre-flight banner and STOP, without starting anything.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for s in args.seeds:
        p = finetuned_encoder(s)
        if not p.exists():
            raise SystemExit(
                f"missing EXP-047 fine-tuned encoder {p}. EXP-048 reads the CONFIRMATORY "
                f"encoders (seeds 0-11 at lr {SELECTED_LR:g}); they are tracked in git, so a "
                "checkout that lacks them is out of date."
            )
    missing = [s for s in args.seeds
               if not list(ARM_A.glob(f"exp043_capped_d{DEPTH}_regionalized_d{DEPTH}_s{s}_*.json"))]
    if missing:
        raise SystemExit(
            f"EXP-043 depth-{DEPTH} records missing for seeds {missing}. They are arm A, the "
            f"PAIRED baseline for Claim 1; fetch them into {ARM_A} first."
        )

    configs = sweep_configs(args.seeds, args.out_dir)
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("encoder_lr must be None: EXP-048's whole question is a FROZEN encoder.")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"record filename collision, cells would overwrite: {dupes[:5]}")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    print(f"EXP-048: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} runs, "
          f"{args.workers} workers")
    print(f"  arm B: EXP-047's fine-tuned encoder, FROZEN, with a FRESH head")
    print(f"  {env_steps():,} env steps per run, 1.0x (frozen - compute-identical to arm A)")
    print(f"  trainable surface back to 390. ONE VARIABLE vs arm A: which frozen encoder.")
    print(f"    A  EXP-040 encoder, frozen, fresh head   {ARM_A_MEAN}   (EXP-043)")
    print(f"    C  EXP-047 encoder, trained, own head    {ARM_C_MEAN}   (EXP-047)")
    print(f"  CONFIRMED at >= +{BAR} over A, p <= 0.05.")
    print(f"  A REFUTED Claim 1 is the most informative outcome here: it would mean EXP-047's")
    print(f"  +{EXP047_GAIN} was co-adaptation, not a better encoder.\n", flush=True)

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

    print(f"\ndone. records in {args.out_dir}. Run aggregate.py for the verdicts.")


if __name__ == "__main__":
    main()
