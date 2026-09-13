"""EXP-061: matched-magnitude noise in the recall block. WHY does the memory read hurt?

EXP-059 found memory HURTS (M-A = -0.0954, p 0.0056) and that correct memory is
indistinguishable from wrong memory (M-S = +0.0204, p 0.4268). Two explanations survive and it
cannot separate them:

  H1  the recall is NOISE on the policy path        -> noise performs like M
  H2  the stored content is ACTIVELY MISLEADING     -> noise performs BETTER than M

> ONE NEW ARM. Arms A and M are REUSED from EXP-059 at all 24 seeds and are NOT re-run. That is
> legitimate only because the instruments added for this experiment were verified numerically
> INERT for `memory` and `memory_amnesic` - identical to full float repr before and after - and
> because the noise generator is constructed only for `memory_noise`, so no other mode's RNG
> stream shifts. See `tests/training/test_memory_noise_readout.py`.

> THE PRIMARY'S INTERESTING OUTCOME IS A NULL, which is this design's central weakness and is
> pre-registered as such. H1 is supported by M ~= N, and a null at n=24 is a BOUND. Claim 2
> (N - A) carries the positive directional prediction.

Spec: docs/superpowers/specs/2026-09-13-exp061-noise-matched-recall-design.md

Usage:
    .venv/bin/python -u experiments/061_noise_matched_recall/run.py --workers 6
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

SEEDS = tuple(range(24))
DEPTH = 5
EPISODES = 10_000
CAP = ((1, 2),)

READOUT = "memory_noise"
TAG = "exp061_noise_d5"

# EXP-059's arms, reused. Context for the banner; the aggregator loads them for the contrasts.
EXP059_DIR = Path("experiments/059_memory_depth5/outputs")
EXP059_A, EXP059_A_MEAN = "exp059_amnesic_d5", 0.3138
EXP059_M, EXP059_M_MEAN = "exp059_memory_d5", 0.2183
BAR = 0.05
GATE_MIN_NORM_RATIO = 0.05      # calibrated at depth 5 before the spec: worst of 4 seeds 0.1653


def e0_encoder(seed: int) -> Path:
    """EXP-040's pretrained encoder, frozen, exactly as EXP-059 loaded it.

    NOT tracked in git. All 24 exist on the laptop's MAIN checkout and in the worktree.
    """
    return E0_DIR / f"exp040_encoder_s{seed}.pt"


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    """EXP-059's `memory` arm copied field for field. ONE change: the readout."""
    return [
        CubeConfig(
            arm="regionalized", readout=READOUT, tag=TAG,
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=tuple(range(1, DEPTH + 1)), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e0_encoder(seed)),
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

    missing = [str(e0_encoder(s)) for s in args.seeds if not e0_encoder(s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} E0 encoder(s), first {missing[:2]}. exp040_encoder_s*.pt are "
            "NOT tracked in git; all 24 live on the laptop's MAIN checkout."
        )

    configs = sweep_configs(args.seeds, args.out_dir)
    # The three ways this stops being "EXP-059's memory arm with one variable", each fatal.
    if any(c.readout != READOUT for c in configs):
        raise SystemExit(f"every cell must use the {READOUT!r} readout")
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("EXP-059's arms are FROZEN: encoder_lr must stay None.")
    if any(c.normalize_advantages for c in configs):
        raise SystemExit("normalize_advantages must stay False, as in EXP-059.")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-061: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} runs, "
          f"{args.workers} workers")
    print(f"  ONE new arm: readout {READOUT!r}, tag {TAG!r}. Encoder FROZEN (exp040 E0).")
    print(f"  Arms A and M are REUSED from EXP-059 and NOT re-run "
          f"(A {EXP059_A_MEAN}, M {EXP059_M_MEAN}).")
    print(f"  H1 recall is NOISE -> N ~= M.  H2 content is MISLEADING -> N BETTER than M.")
    print(f"  CLAIM 1 (primary, NOT directional) is M - N, bar +/-{BAR}. Its interesting")
    print(f"        outcome is a NULL, which is a BOUND and never proof of H1 - stated in the")
    print(f"        spec before any number existed.")
    print(f"  CLAIM 2 is N - A, confirmed at <= -{BAR}. That is the directional prediction, and")
    print(f"        EXP-059 measured M - A = -0.0954 at p 0.0056 on the same seeds and config.")
    print(f"  GATE: arm N mean recall/concept norm ratio >= {GATE_MIN_NORM_RATIO} (calibrated at")
    print(f"        depth 5 before the spec; worst of 4 seeds was 0.1653, 3.3x the bar).\n",
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
                  f"success {r['success_rate']:.3f}  ratio "
                  f"{r.get('recall_concept_norm_ratio') or float('nan'):.4f}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
