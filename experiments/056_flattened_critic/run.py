"""EXP-056: the critic, with its WITHIN-EPISODE state-dependence removed.

EXP-053's Claim 1 is the only claim that experiment confirmed: a learned critic raised depth-7
success by +0.0533 at p 0.0498. Its mechanism was written up as absent on an explained variance
of 0.0021 - but that figure is the FINAL STAGE only, and the by-stage numbers were in the records
all along:

    depth      1       2       3       4       5       6       7
    critic_ev  +0.470  +0.205  +0.098  -0.029  -0.067  -0.058  +0.002
    seeds>0    12/12   12/12   12/12    3/12    0/12    0/12    8/12

V(s) predicts well early and not at all late, and Claim 1 evaluates depth-7 success AFTER the
whole curriculum, so the benefit is free to originate where the critic actually predicts.

PRE-REGISTERED CONTRACT: docs/superpowers/specs/2026-09-02-exp056-flattened-critic-design.md

  1. PRIMARY: F minus EXP-053 arm B. Three readings are pre-registered, including F ABOVE B,
     which would mean V's within-episode variation was harmful NOISE in the baseline.
  2. SECONDARY: F minus EXP-051's EMA control. CONFIRMED at >= +0.05, p <= 0.05, the same bar
     and alpha as EXP-053's Claim 1 so the two numbers are directly comparable.
  3. VALIDITY GATE, a condition and not a report: if V barely varies within an episode then
     flattening removed nothing and Claim 1's null is vacuous.

THE ONE CHANGE against arm B is `flatten_critic=True`, which forms advantages against
`v.detach().mean()` instead of `v.detach()`. It is NOT `returns.mean()`: that arm was proposed
by the 2026-08-31 handoff and is DISQUALIFIED, because it is exactly zero on a one-step episode
and depth 1 averages 1.22 steps per episode, so it would lose gradient precisely where the
critic predicts best. See the spec section 0.

Run (repo root):
    .venv/bin/python -u experiments/056_flattened_critic/run.py --workers 6
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
DEPTH = 7
EPISODES = 10_000
CAP = ((1, 2),)
TAG = "exp056_flat_d7"

# FIXED, not selected. EXP-053's pilot chose this blind to success rate, and re-selecting it
# would make this something other than "arm B with one change". There is no new hyperparameter
# in this experiment and therefore no pilot.
CRITIC_LR = 0.01

# Neither control is re-run. Both are on disk with the same 12 seeds and the same E1 encoders.
CONTROLS = {
    "arm B, the full critic": (Path("experiments/053_neuromod_stage3/outputs"),
                               "exp053_critic_d7", 0.2004),
    "EXP-051, the EMA baseline": (Path("experiments/051_depth7_transfer/outputs"),
                                  "exp051_transfer_d7", 0.1471),
}
BAR = 0.05


def e1_encoder(seed: int) -> Path:
    """EXP-047's fine-tuned encoder, exactly as EXP-053 arm B loaded it."""
    return E1_DIR / f"exp047_ft_d6_lr0.0001_regionalized_d6_s{seed}_sig0.0_encoder.pt"


def sweep_configs(seeds, out_dir: Path) -> list[CubeConfig]:
    """EXP-053 arm B copied field for field. ONE change: `flatten_critic`."""
    return [
        CubeConfig(
            arm="regionalized", readout="concept", tag=TAG,
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=tuple(range(1, DEPTH + 1)), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e1_encoder(seed)),
            critic_lr=CRITIC_LR, flatten_critic=True,
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

    missing = [str(e1_encoder(s)) for s in args.seeds if not e1_encoder(s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} E1 encoder(s), first {missing[:2]}. These are EXP-047's "
            "fine-tuned encoders and they ARE tracked in git; a checkout should have them."
        )

    configs = sweep_configs(args.seeds, args.out_dir)
    # The three ways this stops being "arm B with one change", each fatal rather than a warning.
    if any(not c.flatten_critic for c in configs):
        raise SystemExit("EXP-056's whole point is flatten_critic=True.")
    if any(c.critic_lr != CRITIC_LR for c in configs):
        raise SystemExit(f"critic_lr must stay at EXP-053's selected {CRITIC_LR}.")
    if any(c.normalize_advantages for c in configs):
        raise SystemExit(
            "normalize_advantages subtracts the advantage mean, which is algebraically the "
            "returns' mean. Turning it on imposes the DISQUALIFIED batch-mean baseline on top "
            "of this arm."
        )
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-056: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} runs, "
          f"{args.workers} workers")
    print(f"  arm B with ONE change: advantages use v.mean(), not v per timestep.")
    print(f"  critic_lr {CRITIC_LR} FIXED (EXP-053's blind pilot), encoder FROZEN.")
    for label, (_, _, mean) in CONTROLS.items():
        print(f"  control NOT re-run: {label} = {mean}")
    print(f"  Claim 2 is F minus the EMA control, CONFIRMED at >= +{BAR}, p <= 0.05.")
    print("  A NON-significant Claim 1 is a BOUND, never an equivalence, and is vacuous")
    print("  unless the validity gate shows V varied within episodes at all.\n", flush=True)

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
