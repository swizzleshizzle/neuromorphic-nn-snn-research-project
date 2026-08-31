"""EXP-053 pilot: choose the critic learning rate on seeds 12 and 13, at depth 7.

Six cells, one wave at 6 workers, about 3.2 h. Seeds 12 and 13 and NOT 0-11, which are the
confirmatory seeds - selecting a hyperparameter on the seeds you will later report is how a
pilot turns into an uncontrolled experiment.

Depth 7 and not depth 6, because depth 7 is the regime the critic is used in.

The selection rule is fixed in the spec section 4.3 and executed by `select_critic_lr.py`,
which reads only `critic_ev` and cannot see a success rate.

Run (repo root):
    .venv/bin/python -u experiments/053_neuromod_stage3/pilot_critic_lr.py --workers 6
    .venv/bin/python experiments/053_neuromod_stage3/select_critic_lr.py
"""

from __future__ import annotations

import argparse
import statistics as st
import time
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

DEPTH = 7
EPISODES = 10_000
CAP = ((1, 2),)
PILOT_SEEDS = (12, 13)
LRS = (1e-3, 1e-2, 1e-1)


def e1_encoder(seed: int) -> Path:
    return E1_DIR / f"exp047_ft_d6_lr0.0001_regionalized_d6_s{seed}_sig0.0_encoder.pt"


def sweep_configs(out_dir: Path) -> list[CubeConfig]:
    """EXP-051's config in every field except `critic_lr` and the tag."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=f"exp053_pilot_lr{lr:g}",
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=tuple(range(1, DEPTH + 1)), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e1_encoder(seed)),
            critic_lr=lr,
            max_depth=DEPTH, out_dir=out_dir,
        )
        for lr in LRS
        for seed in PILOT_SEEDS
    ]


def _run(cfg: CubeConfig) -> tuple[float, dict]:
    """Times the cell for the LOG only (Claim 6). The elapsed seconds are returned
    alongside the record, never merged into it: records are seeded and byte-identical
    across worker scheduling, and diffing a re-run seed's record is a standing correctness
    check that a wall-clock field would permanently break."""
    torch.set_num_threads(1)
    start = time.perf_counter()
    record = run_cube_baseline(cfg)
    elapsed = time.perf_counter() - start
    return elapsed, record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for s in PILOT_SEEDS:
        if not e1_encoder(s).exists():
            raise SystemExit(f"missing EXP-047 encoder {e1_encoder(s)} (tracked in git)")

    configs = sweep_configs(args.out_dir)
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision: the tag must encode the lr")
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-053 pilot: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} cells, "
          f"{args.workers} workers")
    print(f"  critic_lr grid {LRS} on pilot seeds {PILOT_SEEDS} (NOT the confirmatory 0-11)")
    print(f"  selection reads critic_ev only and cannot see success_rate\n", flush=True)

    if args.dry_run or not configs:
        print(f"  {len(configs)} cell(s) NOT started.")
        return

    cell_seconds: list[float] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            elapsed, _record = fut.result()
            cell_seconds.append(elapsed)
            print(f"  {i}/{len(configs)}  ({elapsed:.1f}s)", flush=True)

    if cell_seconds:
        print(f"\n  CLAIM 6 timing: mean {st.mean(cell_seconds):.1f}s/cell over "
              f"{len(cell_seconds)} cell(s), min {min(cell_seconds):.1f}s max "
              f"{max(cell_seconds):.1f}s. Seconds-per-cell only: the returned record does not "
              "carry a step count cheaply, and adding one would put a wall-clock-derived "
              "quantity into the JSON record.")

    print(f"\ndone. now run select_critic_lr.py")


if __name__ == "__main__":
    main()
