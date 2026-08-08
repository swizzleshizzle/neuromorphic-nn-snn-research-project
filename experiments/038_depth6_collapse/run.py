# experiments/038_depth6_collapse/run.py
"""EXP-038 driver: do the trainer stabilizers help where collapse IS the failure?

EXP-032 swept `entropy_beta` x `normalize_advantages` at depths 2 and 3 and refuted them.
Its Finding 3 is why: de-collapsing the policy did not make it solve cubes, because the
entropy bonus lowers modal fraction by INJECTING RANDOMNESS rather than by teaching the
policy to read its input. But EXP-031 had already shown the depth-3 failure was NOT
collapse-limited, so there was nothing there for a collapse fix to unlock.

Depth 6 is the opposite case, and two experiments say so:

    EXP-036  depth 6 held-out 0.0000 on all 12 seeds, modal action fraction 0.975
    EXP-037  depth 6 at 3x the episodes: still 0.0000, modal 0.982

EXP-037 Claim 4 also removed the competing explanation - tripling the episodes at depth 6
moved nothing off the floor, so the failure is NOT starvation. The instruments say collapse.

THE MEASUREMENT DECISION THAT DEFINES THIS EXPERIMENT

At depth 6 the measured RANDOM FLOOR (0.0008) sits ABOVE the trained result (0.0000). So
"did success rise above the baseline?" is a check that CANNOT DISTINGUISH THE STATES IT
EXISTS TO SEPARATE: a policy that is merely more random passes it, and EXP-032 Finding 3
established that is precisely how the entropy bonus operates. Every claim here is therefore
paired against EXP-036's RANDOM arm, and Claim 2 refutes any gain whose modal fraction
reached the uniform floor. EXP-037 Claim 4's wording ("any seed above zero is worth
reporting") carries this defect and is superseded.

PRE-REGISTERED CONTRACT, committed before any number exists. Full version and the reasoning
behind each threshold: docs/superpowers/specs/2026-08-07-exp038-depth6-collapse-design.md

  All comparisons are PAIRED per seed against EXP-036, same twelve seeds, same machine,
  exact permutation over all 2**12 = 4096 sign flips, two-sided.

  1. PRIMARY. Is it a lever at depth 6? Held-out success paired against EXP-036's depth-6
     RANDOM arm. CONFIRMED needs mean >= 0.02 AND p <= 0.017 (Bonferroni over three cells)
     AND modal >= 0.45. Otherwise REFUTED at this budget.

  2. THE DISCRIMINATOR. A gain in a cell whose modal fraction reached the uniform floor
     (0.354) is randomization, NOT learning, and refutes the lever however the arithmetic
     lands. The top beta is chosen to saturate, which makes it an instrument check: a
     near-uniform policy MUST score the random floor. If it does not, the measurement is
     broken and no other claim may be read.

  3. DEPTH 5 COHERENCE, the powered arm. Depth 6 has twelve seeds at exactly 0.0000 and a
     per-seed resolution of 1/200, so a partial effect is invisible there. Depth 5 is also
     BROKEN and substantially collapsed (modal 0.779) but sits at 0.0396 +- 0.027, where an
     effect is measurable. CONFIRMED needs >= +0.02 at p <= 0.05 against EXP-036's depth-5
     TRAINED arm (depth 5 is above its floor, so the question there is genuinely "better
     than the current policy", not "better than noise").

  4. MECHANISM. Report modal fraction WITH entropy, never entropy alone: EXP-037 Claim 5
     found the two rising TOGETHER in the back-loaded arms, where entropy alone would have
     suggested they were more exploratory when they were more degenerate.

  5. THE NULL IS PRE-COMMITTED AS A RESULT. Depth 6 was the strongest remaining case for the
     stabilizers. If Claims 1 and 3 both refute, they are CLOSED as a lever and the next move
     is the encoder (vault Stage 2). Written down in advance so a null cannot later be
     re-described as "inconclusive, worth another sweep".

`normalize_advantages` is PINNED True rather than crossed, and that is measured rather than
assumed. EXP-032: beta alone moves nothing (depth 3, beta=0.1, normalize=False gives modal
0.932 against the baseline's 0.932), normalization alone is the worst cell in the sweep, and
only the conjunction moves anything. Crossing it again would spend 48 runs re-deriving a
result this repo already owns. Stated limitation: EXP-038 cannot speak about the
normalize=False half of the plane at depth 6.

No baseline or random arms are re-run. EXP-036 measured all four comparators on this machine
with these seeds:  depth 6 random 0.0008, depth 6 trained 0.0000, depth 5 trained 0.0396,
depth 5 random 0.0000. Re-measuring unchanged quantities would cost 48 runs for nothing -
the same reasoning by which EXP-037 declined to re-run a random arm.

The betas were chosen from a PILOT (depth 6, 1,000 episodes, 10 runs, 37 min, 2026-08-07)
rather than guessed, because EXP-032's single largest limitation was a sweep bounded too low:
every trend was still moving at its boundary. Measured:

    beta  normalize   modal   entropy   success   entropy % of ceiling
    0.0   False       1.000     0.436    0.0000    24%
    0.0   True        1.000     0.211    0.0000    12%
    0.05  True        0.848     0.602    0.0000    34%
    0.2   True        0.776     1.365    0.0000    76%
    0.8   True        0.675     1.698    0.0000    95%

The pilot corrected TWO things this experiment would otherwise have got wrong, both recorded
in spec section 5a:

  A. The uniform modal anchor is BUDGET-DEPENDENT. 0.354 is the 9-step figure; depth 6 runs
     2d+3 = 15 steps and depth 5 runs 13. Measured from EXP-036's random arms the anchors are
     0.309 (d6) and 0.321 (d5). Using 0.354 would have compared depth-6 policies against a
     depth-3 constant.

  B. ENTROPY SATURATES WHILE GREEDY MODAL FRACTION DOES NOT. At beta=0.8 entropy is at 95% of
     its ceiling while modal has only fallen to 0.675. They are not the same axis:
     `mean_train_entropy` describes the STOCHASTIC SAMPLED policy during training, while
     `greedy_modal_action_frac` describes the DETERMINISTIC ARGMAX policy at evaluation
     (`evaluate_states` does greedy rollouts). Pushing beta higher would likely drive modal
     BACK UP, because flat logits make argmax a deterministic tie-break, i.e. a constant
     action. So the original instrument check ("the top beta must reach uniform modal") was
     unreachable in principle, and is restated as entropy saturation (>= 90% of ceiling).

Note also that the pilot's short budget made the policy MORE collapsed, not less: its
(0.0, False) cell reads modal 1.000 against EXP-036's 0.975 at 10,000 episodes. So these modal
figures are an UPPER BOUND on what the sweep will show.

Run (repo root):
    .venv/bin/python -u experiments/038_depth6_collapse/run.py \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 10
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

EPISODES = 10000
NORMALIZE = True        # pinned, see the module docstring

# Chosen from the pilot, not guessed, and these ARE the piloted values.
#   B_LOW  moves modal measurably off the normalize-only anchor (1.000 -> 0.848)
#   B_MID  the midpoint of the dose axis, and what the depth-5 coherence arm uses
#   B_TOP  SATURATES ENTROPY at 95% of the log 6 ceiling, which is Claim 2's instrument
#          check. Not "reaches uniform modal" - see the docstring, correction B.
B_LOW = 0.05
B_MID = 0.2
B_TOP = 0.8

# (depth, beta). Depth 6 carries the dose axis; depth 5 carries the statistical power.
CELLS = [
    (6, B_LOW),
    (6, B_MID),
    (6, B_TOP),
    (5, B_MID),
]

# EXP-036, same machine, same seeds. The arms every claim is measured against.
EXP036 = {
    ("random", 6): 0.0008,
    ("trained", 6): 0.0000,
    ("trained", 5): 0.0396,
    ("random", 5): 0.0000,
}


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def cell_tag(depth: int, beta: float) -> str:
    """Unique per cell. REQUIRED, not cosmetic: `record_filename` encodes tag/arm/depth/seed/
    sigma and NOT `entropy_beta` or `normalize_advantages`, so without the beta in the tag the
    three depth-6 cells would collapse into one set of files, each holding whichever cell
    finished last, silently. This has bitten EXP-032 and EXP-037 by design."""
    return f"exp038_b{str(beta).replace('.', 'p')}_n{int(NORMALIZE)}_d{depth}"


def sweep_configs(seeds, out_dir) -> list[CubeConfig]:
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=cell_tag(depth, beta),
            depth=depth, seed=seed, sigma=0.0, episodes=EPISODES,
            entropy_beta=beta, normalize_advantages=NORMALIZE,
            curriculum=curriculum_for(depth),   # equal stage weights: EXP-037 found the
                                                # equal split at or near optimal
            max_depth=max(6, depth), out_dir=out_dir,
        )
        for depth, beta in CELLS
        for seed in seeds
    ]


def env_steps(depth: int) -> int:
    """Upper-bound environment steps for one run, for the cost estimate."""
    sched = curriculum_schedule(curriculum_for(depth), EPISODES, None)
    return sum(n * max_steps_for(d) for d, n in sched)


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    # 10, not 16. EXP-036 ran 16 at ~920 MB private each, drove system commit to 48.6 of
    # 50.4 GB and held utilisation at a measured 43.1%. Ten measured 74.2%.
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = sweep_configs(args.seeds, args.out_dir)

    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        dupes = {n for n in names if names.count(n) > 1}
        raise SystemExit(f"record filename collision, cells would overwrite: {sorted(dupes)[:5]}")

    print(f"EXP-038: {len(configs)} runs, {args.workers} workers, "
          f"normalize_advantages={NORMALIZE}")
    total = 0
    for depth, beta in CELLS:
        steps = env_steps(depth)
        total += steps * len(args.seeds)
        sched = curriculum_schedule(curriculum_for(depth), EPISODES, None)
        print(f"  d{depth} beta {beta:<5} {[n for _, n in sched]}  {steps:,} steps/run")
    print(f"  total {total:,} env steps")
    print("  measured against EXP-036: "
          f"d6 random {EXP036[('random', 6)]}, d6 trained {EXP036[('trained', 6)]}, "
          f"d5 trained {EXP036[('trained', 5)]}")
    print("  PRIMARY claim is paired against the RANDOM arm, not the 0.0000 trained arm:")
    print("  at depth 6 the random floor is ABOVE the trained result, so beating the trained")
    print("  arm is a check that cannot distinguish learning from randomization.\n", flush=True)

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. one record + one head checkpoint per run in {args.out_dir}")
    print("Run aggregate.py for the pre-registered verdicts.")


if __name__ == "__main__":
    main()
