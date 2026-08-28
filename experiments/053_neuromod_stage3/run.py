"""EXP-053: a learned critic at depth 7, and a neuromodulated plasticity gate at depth 6.

Three arms, twelve seeds each. TWO CONTROLS ALREADY EXIST ON DISK AND ARE NOT RE-RUN.

    arm B   depth 7, E1 frozen + a learned critic     vs  EXP-051          0.1471
    arm G   depth 6, E1 fine-tuned, gated on the bus  vs  EXP-047 lr1e-4   0.2700
    arm R   depth 6, same, rate-matched random gate   vs  arm G

PRE-REGISTERED CONTRACT, committed before any number existed. Full version:
docs/superpowers/specs/2026-08-28-exp053-stage3-design.md

  1. PRIMARY, critic. Arm B minus EXP-051. CONFIRMED at >= +0.05, p <= 0.05.
  2. PRIMARY, gate.   Arm G minus EXP-047. Same bar. DIRECTION NOT PRE-COMMITTED.
  3. ATTRIBUTION, and it governs how Claim 2 may be written. Arm G minus arm R, >= +0.03.
     If G ~ R, the gate's RATE is the whole effect and the neuromorphic claim is REFUTED,
     not deferred. "We need a better gate" is not an available conclusion.
  4. MECHANISM: revisit_rate and optimality. THE PROBE IS DELIBERATELY ABSENT.
  5. DEAD SEEDS: count at exactly 0.0000. The real depth-7 failure is discrete.
  6. COMPUTE: arms G and R do strictly FEWER optimizer steps than their control, so a WIN
     cannot be a compute artifact, but a LOSS might be.

THE ENTROPY TRACE IS RECORDED AND APPEARS IN NO CLAIM. It is the fourth instrument in this
project to move against policy quality: Spearman +0.881 with success WITHIN EXP-044 arm A,
while BETWEEN arms the better arm has the LOWER entropy. See the spec section 0.

Run (repo root), in this order:
    .venv/bin/python -u experiments/053_neuromod_stage3/run.py --arm B --workers 6
    .venv/bin/python -u experiments/053_neuromod_stage3/run.py --arm G --workers 6
    .venv/bin/python -u experiments/053_neuromod_stage3/run.py --arm R --workers 6
"""

from __future__ import annotations

import argparse
import json
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
EXP051_DIR = Path("experiments/051_depth7_transfer/outputs")

SEEDS = tuple(range(12))
EPISODES = 10_000
CAP = ((1, 2),)

# Filled in from the pilot. `select_critic_lr.py` prints it; paste the number here and
# commit that edit BEFORE dispatching arm B, so the value is in git history rather than in
# a shell command nobody can reconstruct.
SELECTED_CRITIC_LR = 1e-2

CONTROL_MEANS = {"B": 0.1471, "G": 0.2700}   # EXP-051, EXP-047 lr1e-4 seeds 0-11
BAR, ATTRIBUTION_BAR = 0.05, 0.03

TAGS = {"B": "exp053_critic_d7", "G": "exp053_gate_d6", "R": "exp053_rgate_d6"}
DEPTHS = {"B": 7, "G": 6, "R": 6}


def e1_encoder(seed: int) -> Path:
    return E1_DIR / f"exp047_ft_d6_lr0.0001_regionalized_d6_s{seed}_sig0.0_encoder.pt"


def arm_g_rates(out_dir: Path) -> dict[int, float]:
    """Arm G's realized per-seed update rates. Arm R cannot be built without them."""
    rates = {}
    for p in sorted(out_dir.glob(f"{TAGS['G']}_*.json")):
        r = json.loads(p.read_text())
        rates[int(r["seed"])] = float(r["gate_rate"])
    return rates


def sweep_configs(arm: str, seeds, out_dir: Path, rates: dict[int, float] | None = None):
    """One arm's configs. Each is its control copied field for field, with ONE change."""
    depth = DEPTHS[arm]
    common = dict(
        arm="regionalized", readout="concept", tag=TAGS[arm],
        depth=depth, sigma=0.0, episodes=EPISODES,
        curriculum=tuple(range(1, depth + 1)), max_steps_by_depth=CAP,
        entropy_beta=0.0, normalize_advantages=False,
        max_depth=depth, out_dir=out_dir,
    )
    if arm == "B":
        # EXP-051's config. ONE change: a learned baseline replaces the scalar EMA.
        return [CubeConfig(seed=s, encoder_state_path=str(e1_encoder(s)),
                           critic_lr=SELECTED_CRITIC_LR, **common) for s in seeds]
    if arm == "G":
        # EXP-047's confirmatory config. ONE change: the encoder steps only when the bus says so.
        return [CubeConfig(seed=s, encoder_lr=1e-4,
                           plasticity_gate="dopamine", **common) for s in seeds]
    if arm == "R":
        if rates is None:
            rates = arm_g_rates(out_dir)
        missing = [s for s in seeds if s not in rates]
        if missing:
            raise SystemExit(
                f"arm R needs arm G's realized gate_rate for seeds {missing}. Run arm G first."
            )
        return [CubeConfig(seed=s, encoder_lr=1e-4, plasticity_gate="random",
                           gate_rate_by_seed=((s, rates[s]),), **common) for s in seeds]
    raise SystemExit(f"unknown arm {arm!r} (expected B, G or R)")


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=("B", "G", "R"))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.arm == "B":
        for s in args.seeds:
            if not e1_encoder(s).exists():
                raise SystemExit(f"missing EXP-047 encoder {e1_encoder(s)} (tracked in git)")
        missing = [s for s in args.seeds
                   if not list(EXP051_DIR.glob(f"exp051_transfer_d7_*_s{s}_*.json"))]
        if missing:
            raise SystemExit(
                f"EXP-051 records missing for seeds {missing}; they are the PAIRED control."
            )

    configs = sweep_configs(args.arm, args.seeds, args.out_dir)
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    print(f"EXP-053 arm {args.arm}: depth {DEPTHS[args.arm]}, {EPISODES:,} episodes, "
          f"{len(configs)} runs, {args.workers} workers")
    if args.arm in CONTROL_MEANS:
        print(f"  ONE VARIABLE vs its control ({CONTROL_MEANS[args.arm]}). "
              f"CONFIRMED at >= +{BAR}, p <= 0.05.")
    if args.arm == "G":
        print(f"  DIRECTION IS NOT PRE-COMMITTED. A significant loss is as informative.")
    if args.arm == "R":
        print(f"  rate-matched control for arm G. G must beat R by >= +{ATTRIBUTION_BAR} "
              "before the gate's SIGNAL may be claimed to matter.")
    print(f"  THE PROBE AND THE ENTROPY TRACE APPEAR IN NO CLAIM.\n", flush=True)

    if not configs:
        print("nothing to do.")
        return
    if args.dry_run:
        print(f"  --dry-run: {len(configs)} cell(s) NOT started.")
        return

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run, c): c for c in configs}
        for i, fut in enumerate(as_completed(futures), 1):
            fut.result()
            print(f"  {i}/{len(configs)}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
