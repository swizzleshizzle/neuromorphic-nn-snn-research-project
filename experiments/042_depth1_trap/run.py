# experiments/042_depth1_trap/run.py
"""EXP-042 driver: close the depth-1 curriculum trap.

EXP-041 found that curriculum stage 1 pays a CONSTANT-ACTION policy 0.3333 against a uniform
random policy's 0.2208. A face move has order 4, so from a one-move scramble any repeated move
either inverts it (1 step) or cycles back to solved (3 steps), and the shipped `2d+3 = 5` budget
covers both. Depth 1 is the ONLY stage where degeneracy beats exploring - by depth 2 it is 0.037
against 0.051, and by depth 3 exactly zero.

EXP-040's two dead seeds die there and never recover: entropy 0.005 leaving stage 1, 0.0002 to
0.0012 entering the final stage, against working seeds' 0.41 to 0.49.

THREE ARMS, 12 seeds each, depth 4, on EXP-040's pretrained encoders:

    A baseline   curriculum (1,2,3,4), shipped budget      depth-1 const reward 0.3333
    B capped     max_steps_by_depth=((1,2),)               depth-1 const reward 0.1667
    C skipped    curriculum (2,3,4)                        no depth-1 stage at all

Arm B admits the inverse and not the cycle, putting degeneracy BELOW random. Arm C needs no code
change. The baseline is RE-RUN rather than reused because EXP-040's records predate the
`stage_trace` telemetry, and comparing against a differently-instrumented arm would confound the
fix with the instrument.

PRE-REGISTERED CONTRACT, committed before any number exists. Full version:
docs/superpowers/specs/2026-08-12-exp042-depth1-trap-design.md

  1. PRIMARY, MECHANISM. Entropy entering the final curriculum stage, paired against arm A.
     CONFIRMED at >= +0.05 with p <= 0.05.

  2. SECONDARY, DESCRIPTIVE. Seeds at exactly 0.000, mean success, sd. NO p-value is attached
     to the failure count.

  3. COST. A cost is flagged if an arm's mean falls >= 0.05 below A among the ten seeds that did
     NOT fail in EXP-040. Arm C is likelier to pay it - it deletes the stage rather than
     repricing it, and EXP-037 showed the bootstrap stages do real work.

  4. MECHANISM CHECK, deterministic and already true: 0.3333 shipped, 0.1667 capped, against
     random's 0.2208. If arm B fails to raise entropy despite this, EXP-041's story is wrong.

> THE STATISTICAL LIMIT, STATED IN ADVANCE. The effect is 2 of 12 seeds. A paired permutation
> test where two seeds carry the difference and ten are ~0 gives p about 0.5 BY CONSTRUCTION,
> and Fisher's exact on 2/12 against 0/12 gives about 0.48. Neither can reach significance at
> this n. That is why Claim 1 sits on a quantity that varies across every seed, and why Claim 2
> is reported without a p-value. A Claim 1 refutation whose effect is confined to the two
> previously-failing seeds must be reported as UNDERPOWERED BY CONSTRUCTION, not as "the fix
> does not work".

Run (repo root):
    .venv/bin/python -u experiments/042_depth1_trap/run.py \
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

DEPTH = 4
EPISODES = 10000

# (arm, curriculum, max_steps_by_depth)
ARMS = [
    ("baseline", (1, 2, 3, 4), ()),
    ("capped", (1, 2, 3, 4), ((1, 2),)),
    ("skipped", (2, 3, 4), ()),
]

# EXP-040, the failure this exists to close.
EXP040_MEAN = 0.3471
EXP040_ZEROS = [2, 4]


def cell_tag(arm: str) -> str:
    """`record_filename` covers tag/arm/depth/seed/sigma and NOT curriculum or
    max_steps_by_depth, so the arm must live in the tag or all three would overwrite."""
    return f"exp042_{arm}_d{DEPTH}"


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=cell_tag(name),
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum, max_steps_by_depth=budget,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(ENCODERS / f"exp040_encoder_s{seed}.pt"),
            max_depth=6, out_dir=out_dir,
        )
        for name, curriculum, budget in ARMS
        for seed in seeds
    ]


def env_steps(curriculum, budget) -> int:
    override = dict(budget)
    sched = curriculum_schedule(curriculum, EPISODES, None)
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
            raise SystemExit(f"missing pretrained encoder {p}. EXP-042 runs on EXP-040's "
                             "encoders; fetch them before dispatching.")

    configs = sweep_configs(args.seeds, args.out_dir)
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        dupes = {n for n in names if names.count(n) > 1}
        raise SystemExit(f"record filename collision: {sorted(dupes)[:5]}")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    print(f"EXP-042: {len(configs)} runs, {args.workers} workers, depth {DEPTH}")
    for name, curriculum, budget in ARMS:
        print(f"  {name:<9} curriculum {curriculum}  budget override {budget or 'none'}  "
              f"{env_steps(curriculum, budget):,} steps/run")
    print(f"  vs EXP-040: mean {EXP040_MEAN}, seeds at zero {EXP040_ZEROS}")
    print("  PRIMARY is entropy entering the final stage - a per-seed MECHANISM measure.")
    print("  The 2/12 failure count is reported WITHOUT a p-value: none is meaningful at n=12.")
    print(flush=True)

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
