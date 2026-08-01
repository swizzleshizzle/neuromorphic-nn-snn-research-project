# experiments/032_collapse_sweep/run.py
"""EXP-032 driver: does anything actually fix the policy collapse on the cube?

EXP-031 established that the trained cube policy is effectively constant-action: at depth 3,
seven of twelve concept seeds play ONE action for all nine steps of every episode, against a
0.354 uniform floor. A policy that ignores its input cannot respond to a change in its input,
so EXP-030's depth-3 memory result measured a degenerate policy rather than memory.

ADR 0001 Amendment 2 characterised the same failure on the GRID WORLD and found the fix:
`entropy_beta=0.01` **alone did nothing** (the summed entropy term was dwarfed by
un-normalized advantages), while **advantage normalization plus `entropy_beta=0.05`
eliminated collapse entirely** (0/10 runs, policy entropy 1.11-1.35 against a log 4 = 1.386
ceiling).

**That was 4 actions with an MLP head. This is 6 actions with a linear head.** The whole
point of this sweep is that the transfer is unproven, so `normalize_advantages` is CROSSED
rather than pinned on: pinning it would assume ADR 0001's central claim instead of testing it.

Grid: 4 betas x 2 normalization levels x 2 depths x 12 seeds = 192 runs.

Depth 2 is included because EXP-030's +10.8 point primary effect lives there and EXP-031
found it partially collapsed (2 to 5 of 12 seeds). Depth 1 is excluded: 83% of its episodes
end early and a one-move solve scores modal fraction 1.0 by construction, so the instrument
cannot measure anything there.

The `(beta=0.0, normalize=False)` cells are EXP-030/031's exact configuration, so those 24
runs double as a free correctness check: their measured fields must reproduce the existing
records exactly. Only `tag` and `out_dir` differ, and both are provenance.

Run (repo root):
    .venv/bin/python -u experiments/032_collapse_sweep/run.py --seeds 0 1 2 3 4 5 6 7 8 9 10 11
"""

from __future__ import annotations

import argparse
import statistics as st
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import CubeConfig, run_cube_baseline

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent

BETAS = [0.0, 0.01, 0.05, 0.1]
NORMALIZE = [False, True]
DEPTHS = [2, 3]

# Pre-registered gate, written before any number exists. The memory arms may not be re-run
# until some cell clears BOTH. Stated here so it cannot be quietly relaxed later.
GATE_MODAL_MAX = 0.60      # vs 0.932 measured for the depth-3 baseline in EXP-031
GATE_ENTROPY_MIN = 1.20    # 67% of the log 6 = 1.792 ceiling


def cell_tag(beta: float, normalize: bool) -> str:
    """Unique per sweep cell.

    REQUIRED, not cosmetic: `record_filename` encodes tag/arm/depth/seed/sigma and NOT
    entropy_beta or normalize_advantages, so without this the 192 runs would collapse into
    24 files, each holding whichever cell happened to finish last, silently.
    """
    return f"exp032_b{str(beta).replace('.', 'p')}_n{int(normalize)}"


def sweep_configs(seeds, episodes: int, out_dir) -> list[CubeConfig]:
    return [
        CubeConfig(
            arm="regionalized",
            readout="concept",
            tag=cell_tag(beta, normalize),
            depth=depth,
            seed=seed,
            sigma=0.0,
            episodes=episodes,
            entropy_beta=beta,
            normalize_advantages=normalize,
            out_dir=out_dir,
        )
        for beta in BETAS
        for normalize in NORMALIZE
        for depth in DEPTHS
        for seed in seeds
    ]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(12)))
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = sweep_configs(args.seeds, args.episodes, args.out_dir)

    names = {f"{c.tag}_d{c.depth}_s{c.seed}" for c in configs}
    if len(names) != len(configs):
        raise SystemExit(f"tag collision: {len(configs)} configs, {len(names)} distinct names")

    print(f"EXP-032: {len(configs)} runs "
          f"({len(BETAS)} betas x {len(NORMALIZE)} normalize x {len(DEPTHS)} depths "
          f"x {len(args.seeds)} seeds), {args.episodes} episodes each")
    print(f"gate (pre-registered): modal_frac < {GATE_MODAL_MAX} AND "
          f"train_entropy > {GATE_ENTROPY_MIN}\n")

    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if i % 10 == 0 or i == len(configs):
                print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. one record per run in {args.out_dir}\n")
    print("depth 3 summary (the decisive depth; 99% of episodes run the full budget):")
    print(f"{'beta':>8}{'normalize':>12}{'modal_frac':>13}{'entropy':>11}"
          f"{'success':>10}   gate")
    for beta in BETAS:
        for normalize in NORMALIZE:
            sub = [r for r in records
                   if r["depth"] == 3
                   and r["config"]["entropy_beta"] == beta
                   and r["config"]["normalize_advantages"] == normalize]
            if not sub:
                continue
            modal = st.mean(r["greedy_modal_action_frac"] for r in sub)
            ent = st.mean(r["mean_train_entropy"] for r in sub)
            succ = st.mean(r["success_rate"] for r in sub)
            passed = modal < GATE_MODAL_MAX and ent > GATE_ENTROPY_MIN
            print(f"{beta:>8}{str(normalize):>12}{modal:>13.3f}{ent:>11.3f}{succ:>10.3f}"
                  f"   {'PASS' if passed else 'fail'}")


if __name__ == "__main__":
    main()
