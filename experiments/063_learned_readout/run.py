"""EXP-063: can ANY readout use the stored content? A learned attention over episodic memory.

EXP-059 found memory HURTS (M-A = -0.0954, p 0.0056). EXP-061 found the recall block is
indistinguishable from matched-magnitude noise (M-N = +0.0210, p 0.4989). Both measured ONE
readout - a raw hippocampal read concatenated onto the concept - and neither can say whether
the episodic information is UNUSABLE or merely UNUSED. EXP-059's own gate says the content is
there: `recall_content_cos` 0.8514, well below 1.0.

> TWO NEW ARMS. Arms A (EXP-059 amnesic), M (EXP-059 memory) and N (EXP-061 noise) are REUSED
> at all 24 seeds and are NOT re-run. That is legitimate only because every pre-existing readout
> was verified numerically INERT - identical to full float repr before and after, measured
> against a clean worktree at the pre-change commit - and because the attention module and the
> noise generator are constructed only for the two new modes, so no other mode's RNG stream
> shifts. See `tests/training/test_attention_readout.py`.

> ARM T IS A CEILING INSTRUMENT, NOT AN ARCHITECTURE. It attends over a PERFECT cache of prior
> concepts and bypasses the hippocampal attractor. If a learned readout with perfect episodic
> recall cannot beat a control with nothing real to attend to, no readout over the lossy
> attractor read can.

Spec: docs/superpowers/specs/2026-09-17-exp063-learned-readout-design.md

Usage:
    .venv/bin/python -u experiments/063_learned_readout/run.py --workers 6 --skip-existing
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import (
    ATTN_MODES,
    CubeConfig,
    record_filename,
    run_cube_baseline,
)

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
E0_DIR = Path("experiments/040_pretrained_encoder_policy/outputs")

SEEDS = tuple(range(24))
DEPTH = 5
EPISODES = 10_000
CAP = ((1, 2),)

# The two new arms, in the order they are dispatched. Both are new; neither is a control for
# the other in the "hold everything but one thing fixed" sense - they ARE the pair.
ARMS = (
    ("memory_attn", "exp063_attn_d5"),
    ("memory_attn_noise", "exp063_attnnoise_d5"),
)

# Reused arms. Context for the banner; the aggregator loads them for the contrasts.
EXP059_DIR = Path("experiments/059_memory_depth5/outputs")
EXP061_DIR = Path("experiments/061_noise_matched_recall/outputs")
EXP059_A, EXP059_A_MEAN = "exp059_amnesic_d5", 0.3138
EXP059_M, EXP059_M_MEAN = "exp059_memory_d5", 0.2183
EXP061_N = "exp061_noise_d5"

BAR = 0.05

# Both calibrated 2026-09-17 at depth 5 on the real frozen E0 encoder, with the real curriculum
# (1..5) and the real step caps, BEFORE the spec was committed and before any EXP-063 outcome
# existed. Measured on the representation directly rather than through `run_cube_baseline`,
# whose depth-5 held-out evaluation dominates its runtime and is not what either gate reads.
# Seeds 0-3, both arms, 120 training episodes per cell.
GATE_MIN_NORM_RATIO = 0.05
"""Gate 1: mean ||read|| / ||concept||. Measured 0.7836-0.7972 across the 8 calibration cells,
so the floor carries a 15.7x margin at the worst seed. The floor is EXP-061's, reused unchanged
because it was already shown able to both pass and fail; a negligible read block reads near 0.
The two arms measured 0.7836-0.7972 (T) against 0.7845-0.7937 (U), which is the magnitude match
the control depends on, verified rather than assumed."""

GATE_MIN_CHOICE_FRAC = 0.15
"""Gate 2: mean `attn_choice_steps / train_steps`. Measured 0.7601-0.7672.

MAXIMUM ATTAINABLE, computed before the threshold was set - the EXP-058 arithmetic. A step
contributes only when at least two prior states exist, so an episode of L steps contributes
max(0, L-2) of L, and the depth-1 curriculum stage runs under a 2-step cap that contributes
EXACTLY ZERO for a fifth of the budget. The calibration measures an UNTRAINED policy, whose
episodes run to the cap (mean 8.3 steps) and therefore sit near the ceiling; a trained policy
solves sooner and drives the fraction DOWN, so the calibration is an upper reading and must not
be used as the floor.

The floor is set against the other end instead. Under near-optimal play - the regime that
MINIMISES this fraction - the five stages contribute about 2, 2, 3, 4 and 5 steps and 0, 0, 1,
2 and 3 choice steps, giving 12,000/32,000 = 0.375. So 0.15 sits 2.5x below the worst credible
trained-regime value and 5.1x below the measurement, and still fails loudly if the cache never
populates."""


def e0_encoder(seed: int) -> Path:
    """EXP-040's pretrained encoder, frozen, exactly as EXP-059 and EXP-061 loaded it.

    NOT tracked in git. All 24 exist on the laptop's MAIN checkout and in the worktree.
    """
    return E0_DIR / f"exp040_encoder_s{seed}.pt"


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    """EXP-059's `memory` arm copied field for field. ONE change per arm: the readout."""
    return [
        CubeConfig(
            arm="regionalized", readout=readout, tag=tag,
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=tuple(range(1, DEPTH + 1)), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e0_encoder(seed)),
            max_depth=DEPTH, out_dir=out_dir,
        )
        for readout, tag in ARMS
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
            "NOT tracked in git; all 24 live on the laptop's MAIN checkout."
        )

    configs = sweep_configs(args.seeds, args.out_dir)
    # The ways this stops being "EXP-059's memory arm with one variable", each fatal.
    if any(c.readout not in ATTN_MODES for c in configs):
        raise SystemExit(f"every cell must use one of {ATTN_MODES}")
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("EXP-059's arms are FROZEN: encoder_lr must stay None.")
    if any(c.normalize_advantages for c in configs):
        raise SystemExit("normalize_advantages must stay False, as in EXP-059.")
    if any(c.episodes != EPISODES or c.depth != DEPTH for c in configs):
        raise SystemExit("budget and depth must match EXP-059 exactly or the arms are not paired")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    total = len(configs)
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-063: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} of {total} runs, "
          f"{args.workers} workers")
    print(f"  TWO new arms: {ARMS[0][1]!r} (T) and {ARMS[1][1]!r} (U). Encoder FROZEN (exp040 E0).")
    print(f"  Arms A, M, N are REUSED from EXP-059/061 and NOT re-run "
          f"(A {EXP059_A_MEAN}, M {EXP059_M_MEAN}).")
    print( "  T attends over a PERFECT cache of prior concepts: a CEILING instrument, not an")
    print( "        architecture. U is matched on parameters, width AND read magnitude, and")
    print( "        differs only in the CONTENT of the attended set.")
    print(f"  CLAIM 1 (PRIMARY) is T - U, confirmed at >= +{BAR}, p <= 0.05. A NULL IS A BOUND,")
    print( "        never evidence of absence - stated in the spec before any number existed.")
    print(f"  CLAIM 2 is T - A, CLAIM 3 is T - M, both at Bonferroni p <= 0.0167.")
    print( "        Claim 3 is capacity-confounded by construction (T has 3.6x M's trainable")
    print( "        surface); Claim 1 is the clean version of that question.")
    print(f"  GATE 1: both arms' mean recall/concept norm ratio >= {GATE_MIN_NORM_RATIO}")
    print(f"  GATE 2: both arms' mean attn_choice_steps/train_steps >= {GATE_MIN_CHOICE_FRAC}")
    print( "        Both calibrated at depth 5 on the real frozen encoder with the real")
    print( "        curriculum and caps, BEFORE the spec was committed.\n", flush=True)

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
            frac = (r["attn_choice_steps"] / r["train_steps"]) if r["train_steps"] else 0.0
            print(f"  {i}/{len(configs)}  {r['tag']} s{r['seed']}  "
                  f"success {r['success_rate']:.3f}  ratio "
                  f"{r.get('recall_concept_norm_ratio') or float('nan'):.4f}  "
                  f"choice {frac:.3f}  attn_H {r.get('attn_entropy_norm') or float('nan'):.3f}",
                  flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
