# Week 19 diagnosis: WHEN do seeds 2 and 4 collapse?
"""Reproduce four EXP-040 depth-4 cells with per-stage telemetry.

EXP-040 left two seeds at exactly 0.000 with modal fraction 1.000 and entropy 0.04, against
working seeds at 0.19-0.67. Established so far, all measured:

  - the ENCODERS are fine. Seeds 2 and 4 sit mid-pack on mean rate, dead units and weight
    scale, and seed 2 has the HIGHEST across-state discrimination of all twelve (0.207).
    Encoder statistics do not even correlate with policy success - seed 8, the best performer,
    has the lowest across-state sd.
  - the INITIAL policy is fine. Before any training, every seed sits at 96-99% of the log-6
    entropy ceiling, failing seeds included. The collapse is not present at initialisation.
  - the failure is on the TRAIN side too (train_success 0.000), so it is not generalisation.

So the collapse develops DURING training and `mean_train_entropy` - one number for the whole
run - cannot say when. `stage_trace` now records it per curriculum stage.

Two failing seeds (2, 4) and two working ones (0, 1), matched. Full EXP-040 configuration, so
each run should reproduce its EXP-040 record EXACTLY; that equality is a free check that the
telemetry perturbed nothing and that the failure is deterministic rather than a fluke.

Run on the laptop from the repo root:
    python -u C:\\Users\\mlgbr\\diag_seeds.py
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import CubeConfig, record_filename, run_cube_baseline

torch.set_num_threads(1)

REPO = Path(__file__).resolve().parent
E040 = Path("experiments/040_pretrained_encoder_policy/outputs")

FAILING = [2, 4]
WORKING = [0, 1]
DEPTH = 4
EPISODES = 10000

# EXP-040's measured held-out success, for the reproduction check.
EXP040 = {0: 0.188, 1: 0.526, 2: 0.000, 4: 0.000}


def cfg_for(seed: int, out_dir: Path) -> CubeConfig:
    return CubeConfig(
        arm="regionalized", readout="concept", tag=f"diag_d{DEPTH}",
        depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
        curriculum=tuple(range(1, DEPTH + 1)),
        entropy_beta=0.0, normalize_advantages=False,
        encoder_state_path=str(E040 / f"exp040_encoder_s{seed}.pt"),
        max_depth=6, out_dir=out_dir,
    )


def _run(seed: int, out_dir: Path) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg_for(seed, out_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=Path("diag_out"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seeds = FAILING + WORKING
    print(f"diagnosis: depth {DEPTH}, seeds {seeds} (failing {FAILING}, working {WORKING})")
    print(f"expect each to reproduce its EXP-040 value exactly: {EXP040}\n", flush=True)

    records = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, s, args.out_dir): s for s in seeds}
        for fut in as_completed(futures):
            r = fut.result()
            records[r["seed"]] = r
            print(f"  seed {r['seed']} done: success {r['success_rate']:.4f} "
                  f"(EXP-040 {EXP040[r['seed']]:.3f}) modal {r['greedy_modal_action_frac']:.3f}",
                  flush=True)

    print("\n=== REPRODUCTION CHECK ===")
    for s in seeds:
        got, want = records[s]["success_rate"], EXP040[s]
        ok = "MATCH" if abs(got - want) < 0.002 else "DIVERGED"
        print(f"  seed {s}: {got:.4f} vs {want:.3f}  {ok}")

    print("\n=== PER-STAGE ENTROPY TRACE ===")
    for s in seeds:
        label = "FAILS" if s in FAILING else "works"
        print(f"\nseed {s} ({label}), final success {records[s]['success_rate']:.4f}, "
              f"modal {records[s]['greedy_modal_action_frac']:.3f}")
        print(f"  {'stage':>6}{'eps':>7}{'ent_first':>11}{'ent_last':>10}{'ent_min':>9}"
              f"{'train_solved':>14}")
        for row in records[s]["stage_trace"]:
            print(f"  {row['depth']:>6}{row['episodes']:>7}{row['entropy_first_10pct']:>11.4f}"
                  f"{row['entropy_last_10pct']:>10.4f}{row['entropy_min']:>9.4f}"
                  f"{row['train_solved_frac']:>14.3f}")

    (args.out_dir / "diag_summary.json").write_text(json.dumps(
        {str(s): {"success": records[s]["success_rate"],
                  "modal": records[s]["greedy_modal_action_frac"],
                  "stage_trace": records[s]["stage_trace"]} for s in seeds}, indent=2))
    print(f"\nwrote {args.out_dir / 'diag_summary.json'}")


if __name__ == "__main__":
    main()
