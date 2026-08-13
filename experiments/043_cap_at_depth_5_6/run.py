# experiments/043_cap_at_depth_5_6/run.py
"""EXP-043 driver: does the depth-1 cap transfer to depths 5 and 6?

EXP-041 found that curriculum stage 1 pays a CONSTANT-ACTION policy 0.3333 against a random
policy's 0.2208, because a face move has order 4 and the shipped 2d+3 = 5 budget lets a repeated
move cycle back to solved. EXP-042 capped the depth-1 TRAINING budget at 2, dropping that to
0.1667 - below random - and at depth 4 it was worth +0.188 (0.3471 -> 0.5351), took seeds at zero
from 2/12 to 0/12, and more than halved the spread.

It also helped the ten seeds that never failed, by +0.1195. The trap was degrading every run.

EXP-040 measured depths 5 and 6 WITH THE TRAP STILL IN PLACE: 0.2304 and 0.1037. Those numbers
are now known to be depressed by a defect that has since been fixed.

24 runs: the capped arm at depths 5 and 6, 12 seeds each, on EXP-040's pretrained encoders.
The baseline is EXP-040 itself - same encoders, same seeds, same machine, same budget, same
curriculum, differing by the cap and nothing else - and is NOT re-run.

PRE-REGISTERED CONTRACT, committed before any number exists. Full version:
docs/superpowers/specs/2026-08-13-exp043-cap-at-depth-5-6-design.md

  1. PRIMARY. Success paired against EXP-040, per depth. CONFIRMED at >= +0.05 with p <= 0.05.

  2. DOES DEPTH 6 BECOME "WORKING"? EXP-036's rule is >= 2x floor AND >= 0.10, so the bar is
     0.10. EXP-040 reached 0.1037 - clearing it by 0.11 SE with 5/12 seeds above, which was
     reported as AT the bar. WORKING here additionally requires a margin >= 1.0 SE and >= 8 of
     12 seeds individually above 0.10, because EXP-040 showed the bare rule fires on noise.

  3. FAILURE COUNT, descriptive. EXP-040 had 2/12 zeros at depth 5 and 3/12 at depth 6. NO
     p-value is attached; Fisher's exact cannot distinguish these counts at n=12.

  4. VARIANCE. At depth 4 the cap took sd 0.2242 -> 0.1012. A repeat is evidence that EXP-040's
     "powerful but unreliable" caveat was largely THE TRAP, not the encoder.

  5. THE NULL IS PRE-COMMITTED. If Claim 1 refutes at both depths while EXP-042 stands, the
     finding is that the trap is a DEPTH-4 phenomenon - deeper runs spend proportionally less of
     the budget in stage 1 and have more stages to recover in. That is a scoping result, not a
     failure.

> THE BASELINE HAS NO `stage_trace`, so the mechanism cannot be paired. EXP-040 predates that
> telemetry, which is why EXP-042 re-ran its baseline. Here the primary claim is on SUCCESS
> instead, and the capped arm's stage traces are descriptive only. Re-running 24 baseline cells
> to re-measure what EXP-040 already has would not be worth it - the mechanism is established at
> depth 4 and this experiment asks whether the fix TRANSFERS.

Run (repo root):
    .venv/bin/python -u experiments/043_cap_at_depth_5_6/run.py \
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
ENCODERS = Path("experiments/040_pretrained_encoder_policy/outputs")

EPISODES = 10000
DEPTHS = [5, 6]
CAP = ((1, 2),)          # EXP-042's confirmed arm

# EXP-040, measured with the trap in place. The paired baseline.
EXP040 = {5: 0.2304, 6: 0.1037}
EXP040_ZEROS = {5: 2, 6: 3}


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def cell_tag(depth: int) -> str:
    """`record_filename` covers tag/arm/depth/seed/sigma and NOT max_steps_by_depth, so the
    arm must live in the tag or these would collide with EXP-040's records."""
    return f"exp043_capped_d{depth}"


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=cell_tag(depth),
            depth=depth, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(depth), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(ENCODERS / f"exp040_encoder_s{seed}.pt"),
            max_depth=6, out_dir=out_dir,
        )
        for depth in DEPTHS
        for seed in seeds
    ]


def env_steps(depth: int) -> int:
    override = dict(CAP)
    sched = curriculum_schedule(curriculum_for(depth), EPISODES, None)
    return sum(n * override.get(d, max_steps_for(d)) for d, n in sched)


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in args.seeds:
        p = ENCODERS / f"exp040_encoder_s{s}.pt"
        if not p.exists():
            raise SystemExit(f"missing pretrained encoder {p}. EXP-043 runs on EXP-040's "
                             "encoders, the same ones EXP-042 used.")

    configs = sweep_configs(args.seeds, args.out_dir)
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit(f"record filename collision: {sorted({n for n in names if names.count(n) > 1})[:5]}")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    print(f"EXP-043: {len(configs)} runs, {args.workers} workers, cap {CAP}")
    total = 0
    for depth in DEPTHS:
        n = sum(1 for c in configs if c.depth == depth)
        total += env_steps(depth) * n
        print(f"  d{depth}: {n} runs, {env_steps(depth):,} steps/run, "
              f"vs EXP-040's {EXP040[depth]} ({EXP040_ZEROS[depth]}/12 at zero)")
    print(f"  total {total:,} env steps")
    print("  ONE VARIABLE vs EXP-040: the depth-1 TRAINING budget, 5 -> 2.")
    print("  Depth-6 'working' needs >= 0.10 AND >= 1.0 SE AND >= 8/12 seeds above 0.10;")
    print("  EXP-040's 0.1037 cleared the bare rule by 0.11 SE, which was noise.\n", flush=True)

    if not configs:
        print("nothing to do.")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. records in {args.out_dir}. Run aggregate.py for the verdicts.")


if __name__ == "__main__":
    main()
