# experiments/047_encoder_finetuning/run.py
"""EXP-047 driver: fine-tune the sensory encoder during RL. Does it beat paying for budget?

Week 20 priced the only lever this project had been using. The depth series is a BUDGET series
(EXP-044/045/046) and the curve is log-linear at ~0.22 success per log10 of spend with no knee,
so depth 8 at matched exposure is ~194,000 episodes for ~0.18. Chasing depth with compute is a
losing exchange rate.

Fine-tuning the encoder is the only untried lever that could CHANGE that exchange rate rather
than pay it. Everything to date runs on a FROZEN encoder and a `Linear(64 -> 6)` head: 390
trainable parameters. This makes the encoder trainable, taking that to 27,206.

    EXP-043 depth 6   frozen encoder,     10,000 episodes  ->  0.1800   (paired baseline, NOT re-run)
    EXP-047 depth 6   fine-tuned encoder, 10,000 episodes  ->  ?

PRE-REGISTERED CONTRACT, committed at 69bf1dc before any number existed. Full version:
docs/superpowers/specs/2026-08-20-exp047-encoder-finetuning-design.md

  1. PRIMARY, PAIRED. Delta against EXP-043 depth 6, exact permutation over 2**12. CONFIRMED at
     >= +0.05 with p <= 0.05. Fine-tuning costs 1.33x per step (56.17 -> 74.87 ms, measured), and
     EXP-046's curve prices 1.33x of budget at +0.027 - so the +0.05 bar sits 1.9x above what the
     extra compute alone would buy a frozen encoder. A delta between +0.027 and +0.05 is
     pre-committed to be reported as AMBIGUOUS, not as a small win.

  2. MECHANISM. Re-probe the fine-tuned encoders (`probe_encoders.py`). Carries a pre-registered
     asymmetry: RL trains on the RL split and the probe holds out a different one, so
     DEGRADATION IS CLEAN but IMPROVEMENT IS CONFOUNDED with memorising probed states. The
     leak-free slice is depth 6's RL held-out states, which EXP-040 also excluded from
     pretraining, so neither stage ever saw them.

  3. ARCHITECTURE ACCOUNTING, descriptive: 390 -> 27,206 trainable, 1.33x per step. This arm is
     NEVER a cell of the depth series.

  4. COLLAPSE, descriptive: deepest-stage entropy against EXP-045's signature
     (0.5914 -> 0.0979, min 2.7e-06, solve rate 0.0218). A 70x larger surface to collapse.

  5. THE NULL IS PRE-COMMITTED AND IS A REAL RESULT. A refuted Claim 1 means REINFORCE's
     gradient cannot improve the encoder faster than budget can be bought, the frozen 390-param
     result stands as published with no caveat, and the next move is a DIFFERENT PRETRAINING
     OBJECTIVE (value/heuristic rather than inverse dynamics) - not more RL and not more episodes.

> THE PILOT RUNS ON SEEDS 12-13, NOT 0-11, AND THAT IS THE POINT.
> `encoder_lr` has no prior here, so it is chosen by a pilot. EXP-039 section 6a refused to pick
> its lr by the probe because the probe was its outcome metric, and recorded that the probe
> WOULD HAVE PICKED THE OTHER RATE. The same trap applies. Running the pilot on seeds disjoint
> from the confirmatory set means no seed contributing to any claim was used for selection, so
> the choice cannot tune either claim's metric and n stays at 12.

Run (repo root), in this order:
    .venv/bin/python -u experiments/047_encoder_finetuning/pretrain_seeds.py --seeds 12 13
    .venv/bin/python -u experiments/047_encoder_finetuning/run.py --mode pilot --workers 6
    .venv/bin/python -u experiments/047_encoder_finetuning/probe_encoders.py --mode pilot
    .venv/bin/python -u experiments/047_encoder_finetuning/select_lr.py
    .venv/bin/python -u experiments/047_encoder_finetuning/run.py --mode confirm --workers 12
"""

from __future__ import annotations

import argparse
import json
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
BASELINE = Path("experiments/043_cap_at_depth_5_6/outputs")

DEPTH = 6
EPISODES = 10_000            # matched to EXP-043 exactly. ONE variable: the encoder trains.
CAP = ((1, 2),)              # EXP-042's confirmed arm, unchanged
BASELINE_MEAN = 0.1800       # EXP-043 depth 6, the paired baseline

# Section 5.1 of the spec. Log-spaced, bracketing pretraining's 3e-3 from below: REINFORCE's
# gradient is far noisier than a supervised MSE at batch 256 and arrives one short episode at a
# time, so the usable rate is expected to be smaller by orders of magnitude, not by a factor.
LR_GRID = (1e-3, 1e-4, 1e-5)
PILOT_SEEDS = (12, 13)
CONFIRM_SEEDS = tuple(range(12))

# What the compute overhead alone is worth, from EXP-046's 0.22 per log10 at depth 6.
COMPUTE_EQUIVALENT = 0.027
BAR = 0.05

SELECTED_LR_FILE = "selected_lr.json"


def curriculum_for(depth: int) -> tuple[int, ...]:
    return tuple(range(1, depth + 1))


def cell_tag(encoder_lr: float) -> str:
    """`record_filename` covers tag/arm/depth/seed/sigma and NOT `encoder_lr`, so the rate must
    live in the tag or the three pilot arms would overwrite each other - the same collision
    EXP-046 had to route around for `episodes`."""
    return f"exp047_ft_d{DEPTH}_lr{encoder_lr:g}"


def env_steps(episodes: int = EPISODES) -> int:
    override = dict(CAP)
    return sum(n * override.get(d, max_steps_for(d))
               for d, n in curriculum_schedule(curriculum_for(DEPTH), episodes, None))


def sweep_configs(seeds, out_dir: Path, encoder_lr: float) -> list[CubeConfig]:
    """Copied from EXP-043's depth-6 cell in every field except `encoder_lr` and the tag."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=cell_tag(encoder_lr),
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=curriculum_for(DEPTH), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(ENCODERS / f"exp040_encoder_s{seed}.pt"),
            encoder_lr=encoder_lr,
            max_depth=DEPTH, out_dir=out_dir,
        )
        for seed in seeds
    ]


def read_selected_lr(out_dir: Path) -> float:
    """The rate `select_lr.py` chose. Refuses to guess if selection has not run."""
    path = out_dir / SELECTED_LR_FILE
    if not path.exists():
        raise SystemExit(
            f"{path} not found. The confirmatory arm runs at the rate `select_lr.py` chose from "
            "the pilot, and that choice is pre-registered and mechanical. Run the pilot, then "
            "probe_encoders.py --mode pilot, then select_lr.py. Do NOT pass --encoder-lr by hand "
            "to skip this: choosing the rate yourself is the exact move the spec forbids."
        )
    chosen = json.loads(path.read_text(encoding="utf-8"))
    if chosen.get("selected_lr") is None:
        raise SystemExit(
            f"select_lr.py HALTED the chain: {chosen.get('reason', 'no rate passed the gate')}\n"
            "Per spec section 5.2 step 3 this is a RESULT, not a reason to widen the grid. "
            "REINFORCE's gradient damages the pretrained representation at every rate tried. "
            "Write it up; a wider grid would need a new pre-registration."
        )
    return float(chosen["selected_lr"])


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pilot", "confirm"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override the mode's default seed set (pilot: 12 13, confirm: 0-11)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--encoder-lr", type=float, default=None,
                    help="confirm mode only, and only to REPRODUCE a recorded selection. The "
                         "rate normally comes from selected_lr.json.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the pre-flight banner and STOP, without starting anything.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "pilot":
        seeds = list(args.seeds) if args.seeds else list(PILOT_SEEDS)
        overlap = sorted(set(seeds) & set(CONFIRM_SEEDS))
        if overlap:
            raise SystemExit(
                f"pilot seeds {overlap} overlap the confirmatory set {CONFIRM_SEEDS[0]}-"
                f"{CONFIRM_SEEDS[-1]}. The pilot selects `encoder_lr`, so a shared seed would let "
                "the selection tune a claim's own metric - the EXP-039 section 6a trap the spec "
                "exists to avoid. Use seeds >= 12."
            )
        rates = list(LR_GRID)
    else:
        seeds = list(args.seeds) if args.seeds else list(CONFIRM_SEEDS)
        rates = [args.encoder_lr if args.encoder_lr is not None
                 else read_selected_lr(args.out_dir)]

    for s in seeds:
        p = ENCODERS / f"exp040_encoder_s{s}.pt"
        if not p.exists():
            raise SystemExit(
                f"missing pretrained encoder {p}. EXP-047 fine-tunes EXP-040's encoders, the "
                "same ones EXP-042/043/044/045/046 used. For seeds >= 12, generate them first "
                f"with:  .venv/bin/python experiments/047_encoder_finetuning/pretrain_seeds.py "
                f"--seeds {' '.join(str(x) for x in seeds)}"
            )

    if args.mode == "confirm":
        missing = [s for s in seeds if not list(
            BASELINE.glob(f"exp043_capped_d{DEPTH}_regionalized_d{DEPTH}_s{s}_*.json"))]
        if missing:
            raise SystemExit(
                f"EXP-043 depth-{DEPTH} records missing for seeds {missing}. They are the PAIRED "
                f"baseline for Claim 1; fetch them into {BASELINE} first."
            )

    configs = [c for lr in rates for c in sweep_configs(seeds, args.out_dir, lr)]
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise SystemExit(f"record filename collision, cells would overwrite: {dupes[:5]}")

    if args.skip_existing:
        before = len(configs)
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]
        if before != len(configs):
            print(f"  skipping {before - len(configs)} already-recorded cell(s)")

    print(f"EXP-047 {args.mode}: depth {DEPTH}, {EPISODES:,} episodes, "
          f"{len(configs)} runs, {args.workers} workers")
    print(f"  encoder_lr {', '.join(f'{r:g}' for r in rates)} on seeds "
          f"{seeds[0]}-{seeds[-1]} ({len(seeds)})")
    print(f"  {env_steps():,} env steps per run, ~1.33x EXP-043's wall clock per step")
    print(f"  TRAINABLE SURFACE 390 -> 27,206 (70x). THIS IS A DIFFERENT ARCHITECTURE and is")
    print(f"  never reported as another cell of the depth series.")
    if args.mode == "pilot":
        print(f"  Pilot seeds are DISJOINT from the confirmatory set, so selecting the rate")
        print(f"  cannot tune either claim's metric (spec 5.1).")
    else:
        print(f"  ONE VARIABLE vs EXP-043 depth 6 ({BASELINE_MEAN}): the encoder trains.")
        print(f"  CONFIRMED at >= +{BAR}, p <= 0.05. Compute alone is worth "
              f"+{COMPUTE_EQUIVALENT} (EXP-046), so +{COMPUTE_EQUIVALENT} to +{BAR} is AMBIGUOUS.")
    print(flush=True)

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
