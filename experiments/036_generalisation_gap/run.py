# experiments/036_generalisation_gap/run.py
"""EXP-036 driver: the train/held-out gap, and where the curriculum breaks with depth.

The vault note `road-to-a-solved-cube.md` sets as Stage 1 "does the depth-3 policy generalise
or memorise its 90 training states?". THAT QUESTION WAS ALREADY ANSWERED and the note is wrong
on the premise. `split_shell` partitions depth 3 into 90 train / 30 eval, and EXP-035's
headline 50.0% is HELD-OUT success on states the policy never trained on. It generalises.

What has never been measured, in any cube experiment, is the GAP: train-side success has never
been reported, so it is unknown whether the policy overfits its 90 states or is capacity-limited
in a way coverage cannot fix. That decides whether a train-fraction sweep is worth running.

Combined with the depth 4/5/6 break point because measuring the gap needs a retrain either way
(no trained weights were saved anywhere before this experiment), and a retrain that sweeps depth
answers both for one night of laptop time.

PRE-REGISTERED CONTRACT, committed before any number exists. Full version and the reasoning
behind each threshold: docs/superpowers/specs/2026-08-03-exp036-generalisation-gap-design.md

  1. THE GAP. gap_d = train_success(d) - heldout_success(d), both on samples capped at 200 so
     the two sides carry matched noise.

     The threshold is on the TWELVE-SEED MEAN, not a single seed. A first draft of this
     contract put the bar at 0.05 on one seed; measuring the `random` arm (which cannot
     overfit, so its gap is zero by construction) over seeds 0-5 gave single-seed gaps from
     -0.100 to +0.011, because the depth-3 held-out side is 30 states and three lucky solves
     read as a tenth of a gap. The bar was BELOW THE NOISE FLOOR. Corrected before dispatch.

       mean gap  < 0.05 and p >  0.05  ->  coverage REFUTED as a lever. The Stage-1a
                                           train-fraction sweep is cancelled, not run.
       mean gap >= 0.15 and p <= 0.05  ->  overfitting established, the sweep is justified.
       anything else                   ->  inconclusive. Report it, act on neither.

     Significance by exact paired permutation over all 2**12 = 4096 sign flips. No scipy in
     the venv, n = 12, so this is cheap, assumption-free and better than a normal approximation.

  2. THE BREAK POINT. "Working at depth d" requires held-out success to clear BOTH bars:
     at least TWICE the MEASURED random floor at d, AND at least 0.10 absolute. Broken is
     failing either. The floor is measured per depth via the `random` arm, never assumed: on
     this cube it is 21% at depth 1, not 1/6, because a random walk with a 2d+3 budget can
     stumble into solved.

     REVISED 2026-08-03 on synthetic records, before any real number existed. The rule was
     "below twice the measured floor" alone. But the floor COLLAPSES with depth, so that bar
     is 0.029 at depth 3 and only 0.009 at depth 6: it vanishes exactly where it has to bite,
     and a synthetic depth-6 policy at 0.017 (plainly failing) was reported "working". Third
     instance tonight of a check that cannot distinguish the states it exists to separate.

     0.10 is one quarter of the depth-3 result at this budget (EXP-035's 0.397), and clear of
     resolution: held-out sets are 30/133/200/200 states, so 0.10 is 3 to 20 solves.

     PREDICTION, recorded so it can be wrong: the curriculum breaks at DEPTH 5. From EXP-033's
     raw-facelet linear probe, which falls 0.956 / 0.766 / 0.598 at depths 3/4/5 against a
     chance of about 0.19.

  3. If depth 6 is still above twice its floor, Wall 1 sits further out than the probe trend
     implied and the linear head has more room than EXP-033 suggested. That REFUTES claim 2's
     prediction and gets logged as a refutation, not smoothed over.

  4. INSTRUMENTS. Report greedy_modal_action_frac and mean_train_entropy per cell. EXP-035
     established that entropy alone cannot separate collapse from convergence: collapse is low
     entropy with HIGH modal fraction (0.987), convergence is low entropy with LOW modal
     fraction (0.580). A depth that fails must be diagnosed as one or the other, not just scored.

  5. THE BUDGET CONFOUND, stated up front. This measures where the curriculum breaks AT 10,000
     EPISODES. EXP-035 showed depth 3 climbs 0.397 -> 0.500 between 10k and 30k and had not
     saturated, so a depth that fails here may only be under-trained. DO NOT REPORT THE BREAK
     POINT AS A PROPERTY OF THE ARCHITECTURE.

BUILT-IN VALIDATION: the depth-3 cell is exactly EXP-035's 10,000-episode cell (same seeds,
same curriculum, same budget). Run on the same machine it must reproduce 0.397. That checks the
EXP-036 code changes against real experiment data, not only the depth-1 smoke values.

Run (repo root):
    .venv/bin/python -u experiments/036_generalisation_gap/run.py \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 16
"""

from __future__ import annotations

import argparse
import itertools
import statistics as st
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import CubeConfig, record_filename, run_cube_baseline

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

DEPTHS = [3, 4, 5, 6]
EPISODES = 10000
EXP035_DEPTH3_AT_10K = 0.397      # the replication target
GAP_REFUTE_BELOW = 0.05
GAP_CONFIRM_AT = 0.15
BREAK_MULTIPLE = 2.0              # must clear this multiple of the MEASURED floor...
BREAK_ABSOLUTE = 0.10             # ...AND this absolute bar. See the docstring.


def curriculum_for(depth: int) -> tuple[int, ...]:
    """(1..depth). `episodes` is SPLIT across stages, so a deeper curriculum never buys compute."""
    return tuple(range(1, depth + 1))


def cell_tag(depth: int) -> str:
    """Unique per cell. record_filename encodes tag/arm/depth/seed/sigma and NOT episodes or
    curriculum, so any sweep over those must vary the tag or records overwrite silently."""
    return f"exp036_d{depth}_e{EPISODES}"


def sweep_configs(seeds, out_dir) -> list[CubeConfig]:
    trained = [
        CubeConfig(
            arm="regionalized", readout="concept", tag=cell_tag(depth),
            depth=depth, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(depth), max_depth=max(6, depth), out_dir=out_dir,
        )
        for depth in DEPTHS
        for seed in seeds
    ]
    # The floors, and the empirical null for the gap. No training, so these are nearly free.
    floors = [
        CubeConfig(
            arm="random", tag=cell_tag(depth), depth=depth, seed=seed,
            max_depth=max(6, depth), out_dir=out_dir,
        )
        for depth in DEPTHS
        for seed in seeds
    ]
    return trained + floors


def permutation_p(diffs: list[float]) -> float:
    """Exact paired permutation over all 2**n sign flips. Two-sided.

    n = 12 is 4096 flips, so this is exhaustive rather than sampled. Guarded because the
    repo has no scipy and a normal approximation at n = 12 is not trustworthy.
    """
    n = len(diffs)
    if n == 0 or n > 20:
        raise ValueError(f"exact permutation needs 1 <= n <= 20, got {n}")
    observed = abs(sum(diffs))
    hits = sum(
        1
        for signs in itertools.product((1, -1), repeat=n)
        if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12
    )
    return hits / (2 ** n)


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = sweep_configs(args.seeds, args.out_dir)

    # Collision guard against the REAL naming function, not a copy of it. EXP-032's lesson.
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        dupes = {n for n in names if names.count(n) > 1}
        raise SystemExit(f"record filename collision, cells would overwrite: {sorted(dupes)[:5]}")

    trained = [c for c in configs if c.arm != "random"]
    print(f"EXP-036: {len(configs)} runs ({len(trained)} trained + {len(configs) - len(trained)} floors)")
    print(f"depths {DEPTHS} at {EPISODES:,} episodes, curricula {[curriculum_for(d) for d in DEPTHS]}")
    print(f"replication target: depth 3 must reproduce EXP-035's {EXP035_DEPTH3_AT_10K}")
    print(f"prediction (pre-registered): the curriculum breaks at depth 5\n", flush=True)

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. one record + one head checkpoint per trained run in {args.out_dir}\n")

    def cells(depth, arm):
        return [r for r in records if r["depth"] == depth and r["arm"] == arm]

    print(f"{'depth':>6}{'heldout':>18}{'train':>10}{'gap':>9}{'p':>9}"
          f"{'floor':>8}{'modal':>8}{'entropy':>9}{'verdict':>12}")
    for depth in DEPTHS:
        sub, flo = cells(depth, "regionalized"), cells(depth, "random")
        if not sub:
            continue
        held = [r["success_rate"] for r in sub]
        train = [r["train_success_rate"] for r in sub]
        gaps = [r["generalisation_gap"] for r in sub]
        floor = st.mean(r["success_rate"] for r in flo) if flo else float("nan")
        p = permutation_p(gaps) if len(gaps) > 1 else float("nan")
        broken = not (st.mean(held) >= BREAK_MULTIPLE * floor
                      and st.mean(held) >= BREAK_ABSOLUTE)
        print(f"{depth:>6}{st.mean(held):>11.4f}+-{st.stdev(held) if len(held) > 1 else 0:<5.3f}"
              f"{st.mean(train):>10.4f}{st.mean(gaps):>+9.4f}{p:>9.4f}{floor:>8.4f}"
              f"{st.mean(r['greedy_modal_action_frac'] for r in sub):>8.3f}"
              f"{st.mean(r['mean_train_entropy'] for r in sub):>9.3f}"
              f"{'BROKEN' if broken else 'working':>12}")

    # The empirical null: the random arm cannot overfit, so its mean gap is what zero looks like.
    for depth in DEPTHS:
        flo = cells(depth, "random")
        if len(flo) > 1:
            fg = [r["generalisation_gap"] for r in flo]
            print(f"  null gap @ depth {depth}: {st.mean(fg):+.4f} "
                  f"(sd {st.stdev(fg):.4f}, p {permutation_p(fg):.4f})")

    d3 = cells(3, "regionalized")
    if d3:
        got = st.mean(r["success_rate"] for r in d3)
        delta = got - EXP035_DEPTH3_AT_10K
        print(f"\nreplication: depth 3 = {got:.4f} vs EXP-035 {EXP035_DEPTH3_AT_10K} ({delta:+.4f})")
        if abs(delta) > 0.02:
            print("  MISMATCH. Either the EXP-036 code changes moved something, or this did not")
            print("  run on the same machine as EXP-035. Resolve before trusting any row above.")

        gaps = [r["generalisation_gap"] for r in d3]
        mg, p = st.mean(gaps), permutation_p(gaps)
        print(f"\nCLAIM 1 at depth 3: mean gap {mg:+.4f}, exact p {p:.4f}")
        if mg < GAP_REFUTE_BELOW and p > 0.05:
            print("  COVERAGE REFUTED. The Stage-1a train-fraction sweep is cancelled, not run.")
        elif mg >= GAP_CONFIRM_AT and p <= 0.05:
            print("  OVERFITTING ESTABLISHED. The train-fraction sweep is justified.")
        else:
            print("  INCONCLUSIVE. Report it, act on neither.")


if __name__ == "__main__":
    main()
