"""EXP-062: the honest depth frontier, and an out-of-sample test of the budget law.

`road-to-a-solved-cube` calls Stage 4 - depth-11 random scrambles - "the actual deliverable" and
"genuinely achievable... nothing about it requires new science". That was written before week 20
discovered the budget law, and the law refutes it: at 4.4x per depth, depth 11 at depth-7 parity
costs about 1,796 h per cell, ~75 days per seed.

> THE PRICING EXTRAPOLATES A LAW FITTED AT DEPTHS 3-7 OUT TO 11. An extrapolation used to declare
> a deliverable unreachable deserves a test, and depth 8 is where it can be tested: the law
> predicts 0.0588 there. Claim 1 is a ONE-SAMPLE containment check, not a paired contrast.

> CLAIM 2 (depth 9) IS A FRONTIER MEASUREMENT, NOT A TEST OF THE LAW. Success is bounded below at
> zero, so the law's -0.0827 prediction can only manifest as "about zero" - depth 9 cannot
> discriminate -0.08 from -0.5. Reading a zero there as confirmation would be reading a bound as
> evidence.

Spec: docs/superpowers/specs/2026-09-15-exp062-depth-frontier-design.md

Usage:
    .venv/bin/python -u experiments/062_depth_frontier/run.py --workers 6
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
DEPTHS = (8, 9)
EPISODES = 10_000
CAP = ((1, 2),)

# FIXED, not selected, exactly as EXP-056 and EXP-060 did. EXP-053's pilot chose 0.01 BLIND to
# success rate and recorded it in `selected_critic_lr.json`; that file is gitignored and so does
# not reach the worktree, and re-selecting would make this something other than "arm B at a new
# depth". There is no new hyperparameter here and therefore no pilot.
CRITIC_LR = 0.01

# The budget law from EXP-044/045/046, anchored at depth 7's measured 0.2004 (EXP-053 arm B).
# 4.4x budget buys about one depth, and success is ~0.22 per log10 of spend.
ANCHOR_DEPTH, ANCHOR_SUCCESS = 7, 0.2004
PER_DEPTH = 0.1416              # 0.22 * log10(4.4)
# MEASURED before the spec was written, 12 seeds, random policy at a 2d+3 budget.
CHANCE_FLOOR = {7: 0.0000, 8: 0.0000, 9: 0.0000}
FLOOR_BAR = 0.02                # Claim 2: depth 9 is "at the floor" below this
GATE_MIN_RATIO = 0.05           # Claim 3, reused from EXP-056/060 in its every-stage form


def law_prediction(depth: int) -> float:
    return ANCHOR_SUCCESS - PER_DEPTH * (depth - ANCHOR_DEPTH)


def tag_for(depth: int) -> str:
    return f"exp062_frontier_d{depth}"


def e1_encoder(seed: int) -> Path:
    """EXP-047's fine-tuned encoder, exactly as EXP-053 arm B and EXP-056 loaded it."""
    return E1_DIR / f"exp047_ft_d6_lr0.0001_regionalized_d6_s{seed}_sig0.0_encoder.pt"


def sweep_configs(seeds, out_dir: Path, depths) -> list[CubeConfig]:
    """EXP-053 arm B copied field for field. The ONLY variable is `depth`."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=tag_for(depth),
            depth=depth, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=tuple(range(1, depth + 1)), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e1_encoder(seed)),
            critic_lr=CRITIC_LR,
            max_depth=depth, out_dir=out_dir,
        )
        for depth in depths
        for seed in seeds
    ]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=int, nargs="+", default=list(DEPTHS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Depths 1-7 already have records from earlier experiments. Re-running one here under a new
    # tag would produce a second, differently-tagged copy of a measured number and invite it
    # being read as independent. This experiment is about the frontier BEYOND the fitted range.
    shallow = [d for d in args.depths if d <= ANCHOR_DEPTH]
    if shallow:
        raise SystemExit(
            f"depths {shallow} are at or below the law's anchor depth {ANCHOR_DEPTH} and are "
            "already measured by earlier experiments. This tests the frontier BEYOND the fitted "
            "range; re-running a fitted depth here would duplicate a known number under a new tag."
        )

    missing = [str(e1_encoder(s)) for s in args.seeds if not e1_encoder(s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} E1 encoder(s), first {missing[:2]}. EXP-047's fine-tuned "
            "encoders; seeds 0-13 were made in week 20 and 14-23 by EXP-060."
        )

    configs = sweep_configs(args.seeds, args.out_dir, args.depths)
    if any(c.critic_lr is None for c in configs):
        raise SystemExit("every cell here has a learned critic; critic_lr must not be None.")
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("the encoder is FROZEN in arm B: encoder_lr must stay None.")
    if any(c.normalize_advantages for c in configs):
        raise SystemExit("normalize_advantages must stay False, as in EXP-053 arm B.")
    if any(c.max_depth != c.depth for c in configs):
        raise SystemExit("max_depth must equal depth so the BFS table covers the target shell.")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-062: the depth frontier. {len(configs)} runs, {args.workers} workers, "
          f"{EPISODES:,} episodes each.")
    print(f"  depths {tuple(args.depths)}, seeds {tuple(args.seeds)}")
    print("  EXP-053 arm B field for field, ONE variable: depth. Encoder FROZEN (E1), "
          f"critic_lr {CRITIC_LR} FIXED.")
    print("\n  THE BUDGET LAW'S OUT-OF-SAMPLE PREDICTIONS (fitted at depths 3-7):")
    for d in args.depths:
        pred = law_prediction(d)
        floor = CHANCE_FLOOR.get(d)
        note = "" if pred > 0 else "  <- NEGATIVE, so censored at the zero floor"
        print(f"    depth {d}: predicted {pred:+.4f}   measured chance floor "
              f"{floor if floor is not None else '?'}{note}")
    print(f"\n  CLAIM 1 (PRIMARY) is a ONE-SAMPLE containment test at depth 8: does the 95%")
    print(f"        interval on mean success contain {law_prediction(8):.4f}? All three outcomes")
    print(f"        are results, and the two where the law FAILS are the more interesting ones.")
    print(f"  CLAIM 2 is the FRONTIER at depth 9, confirmed at the floor below {FLOOR_BAR}. It is")
    print(f"        NOT a test of the law: success is bounded below at zero, so a negative")
    print(f"        prediction can only show up as about zero.")
    print(f"  GATE: critic within-episode RMS ratio >= {GATE_MIN_RATIO} at EVERY stage, both arms.")
    print(f"        EXP-060 measured 0.3489 at its deepest stage; extrapolated ~0.25 at depth 8")
    print(f"        and ~0.19 at depth 9, so four to five times the floor and no more.\n",
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
            r = fut.result()
            print(f"  {i}/{len(configs)}  {r['tag']} s{r['seed']}  "
                  f"success {r['success_rate']:.3f}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
