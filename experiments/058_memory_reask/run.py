"""EXP-058: does episodic memory help a policy that ACTUALLY WORKS?

EXP-030 asked this and found a null, on a 2.2% policy at depth 3 that EXP-031 then showed was
collapsed to a constant action on 7 of 12 seeds. A memory that helps you avoid revisiting states
cannot help an agent that is not navigating. Depth 6 now runs at 0.3579.

EXP-030'S REAL LESSON DICTATES THIS DESIGN. Its headline was "memory beat the shuffle-null by 10.8
points (p 0.078)". Against its AMNESIC control the same result was +1.2 at p 0.91. The shuffle-null
contrast measures the harm of INCORRECT memory, not the benefit of correct memory, and two arms
would have published a false positive. So the primary here is M minus A, fixed before any number
exists because the tempting comparison is the wrong one.

THE CONTROL IS THE AMNESIC ARM, NOT `concept`. A readout change alters the feature width, so
comparing memory against concept would confound memory with width. The amnesic arm has the
identical width and code path with the stored content zeroed. exp049_fresh2_d6's 0.3579 is
context, not a control, and no claim is paired against it.

PRE-REGISTERED CONTRACT: docs/superpowers/specs/2026-09-04-exp058-memory-reask-design.md

  1. PRIMARY:   M minus A on success.      CONFIRMED at >= +0.05, p <= 0.05.
  2. MECHANISM: M minus A on revisit_rate. CONFIRMED at <= -0.02, p <= 0.05. Note the SIGN.
  3. GATE:      arm M's mean_n_stored must exceed 10, and arm S's unshuffled_frac must be
                below 0.20, or every claim is void.
  4. SECONDARY: M minus S. The harm of INCORRECT memory, and not evidence about the benefit
                of correct memory.

COST IS NOT KNOWN. A concept cell at this config runs about 2.8 h on the laptop; a memory cell
measured at least 1.5x that on the VPS and the upper bound was not established. Wave 1's recorded
`seconds` settles it. --skip-existing makes stopping and resuming free.

Run (repo root):
    .venv/bin/python -u experiments/058_memory_reask/run.py --workers 6
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
E2_DIR = Path("experiments/049_second_round/outputs")

SEEDS = tuple(range(12))
DEPTH = 6
EPISODES = 10_000
CAP = ((1, 2),)

# ONE variable: the readout. Everything else is exp049_fresh2_d6 field for field.
ARMS = {
    "A": ("memory_amnesic", "exp058_amnesic_d6"),
    "M": ("memory", "exp058_memory_d6"),
    "S": ("memory_shuffled", "exp058_shuffled_d6"),
}

# Context, NOT a control. No claim is paired against this number.
FRESH2_CONTEXT = 0.3579
BAR = 0.05
REVISIT_BAR = -0.02
GATE_MIN_STORED = 10
GATE_MAX_UNSHUFFLED = 0.20


def e2_encoder(seed: int) -> Path:
    """EXP-049's second-round encoder, frozen. Tracked in git for all 12 seeds."""
    return E2_DIR / f"exp049_ft2_d6_regionalized_d6_s{seed}_sig0.0_encoder.pt"


def sweep_configs(seeds, out_dir: Path, arms) -> list[CubeConfig]:
    return [
        CubeConfig(
            arm="regionalized", readout=readout, tag=tag,
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=tuple(range(1, DEPTH + 1)), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e2_encoder(seed)),
            max_depth=DEPTH, out_dir=out_dir,
        )
        for key in arms
        for readout, tag in [ARMS[key]]
        for seed in seeds
    ]


def _run(cfg: CubeConfig) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing = [str(e2_encoder(s)) for s in args.seeds if not e2_encoder(s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} E2 encoder(s), first {missing[:2]}. These are EXP-049's "
            "second-round encoders and they ARE tracked in git; a checkout should have them."
        )

    configs = sweep_configs(args.seeds, args.out_dir, args.arms)
    # The three ways this stops being "exp049_fresh2_d6 with one variable", each fatal.
    if any(c.readout == "concept" for c in configs):
        raise SystemExit("the concept readout is CONTEXT here, not an arm. See the spec section 1.")
    if any(c.encoder_lr is not None for c in configs):
        raise SystemExit("EXP-058 arms are FROZEN: encoder_lr must stay None.")
    if any(c.normalize_advantages for c in configs):
        raise SystemExit("normalize_advantages must stay False; exp049_fresh2_d6 had it off.")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-058: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} runs, "
          f"{args.workers} workers")
    print(f"  arms {tuple(args.arms)}, ONE variable: the readout. Encoder FROZEN (E2).")
    print(f"  PRIMARY is M minus A, NOT M minus S. EXP-030's headline came from the")
    print(f"  shuffle-null and was misread; that contrast measures the harm of INCORRECT")
    print(f"  memory. Against its amnesic control the same result was +1.2 pts at p 0.91.")
    print(f"  Claim 1 bar {BAR:+.2f} on success; Claim 2 bar {REVISIT_BAR:+.2f} on revisit_rate.")
    print(f"  GATE: arm M mean_n_stored > {GATE_MIN_STORED}, arm S unshuffled_frac < "
          f"{GATE_MAX_UNSHUFFLED}.")
    print(f"  context, not a control: exp049_fresh2_d6 = {FRESH2_CONTEXT}\n", flush=True)

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
                  f"success {r['success_rate']:.3f}  {r.get('seconds', 0):.0f}s", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
