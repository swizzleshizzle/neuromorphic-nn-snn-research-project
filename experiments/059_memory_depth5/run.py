"""EXP-059: the memory question at depth 5, at n=24, with a gate that works.

EXP-058 asked this at depth 6 and was VOID: its gate required mean_n_stored > 10, a quantity
bounded by episode length in a setting where episodes average 7.76 steps. Unsatisfiable, and it
gated on STORING when the arms differ at the READ site. Its seeds are also burned, because runs
here are byte-identical.

This changes venue and fixes both. exp043_capped_d5 is a working depth-5 policy at 0.3229, frozen,
already measured at 24 seeds, and every exp040 encoder it needs exists. Depth 5 is a different
measurement, seeds 12-23 are fresh for this question, n=24 halves the standard error, and no
encoder has to be manufactured.

EXP-030'S TRAP DICTATES THE PRIMARY, AS IT DID IN EXP-058. That experiment's headline came from
the shuffle-null; against its amnesic control the same result was +1.2 at p 0.91. EXP-058's
unlicensed ordering reproduced it exactly: memory beat the shuffle-null and lost to amnesic. So
M vs A is primary, fixed before any number exists.

PRE-REGISTERED CONTRACT: docs/superpowers/specs/2026-09-09-exp059-memory-depth5-design.md

  1. PRIMARY:   M minus A on success. NOT directional: >= +0.05 confirms memory helps, <= -0.05
                is a real finding that memory HURTS, anything else is a bound.
  2. MECHANISM: M minus A on revisit_rate. CONFIRMED at <= -0.02. The better-powered instrument.
  3. GATE:      arm M's recall_content_cos < 0.95, and arm S's unshuffled_frac < 0.20, or every
                claim is void. Calibrated at bb1efa5 BEFORE the spec: empty attractor 1.000000,
                random loaded 0.9437, real depth-5 run 0.8128.
  4. SECONDARY: M minus S. The harm of INCORRECT memory, not evidence about correct memory.

POWER, stated up front: n=24 gives roughly 60-70% power at a 0.05 effect and only 30-40% at 0.03,
and EXP-058's ordering suggested about -0.03. An indistinguishable Claim 1 is plausible even if
memory does something small.

Run (repo root):
    .venv/bin/python -u experiments/059_memory_depth5/run.py --workers 6
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

# ONE variable: the readout. Everything else is exp049_fresh2_d6 field for field.
ARMS = {
    "A": ("memory_amnesic", "exp059_amnesic_d5"),
    "M": ("memory", "exp059_memory_d5"),
    "S": ("memory_shuffled", "exp059_shuffled_d5"),
}

# Context, NOT a control. No claim is paired against this number.
CAPPED_D5_CONTEXT = 0.3229
BAR = 0.05
REVISIT_BAR = -0.02
# Calibrated at bb1efa5 BEFORE the spec was written: empty attractor 1.000000, random loaded
# 0.9437, real depth-5 run 0.8128, MemoryReadout docstring 0.802 independently.
GATE_MAX_RECALL_COS = 0.95
GATE_MAX_UNSHUFFLED = 0.20


def e0_encoder(seed: int) -> Path:
    """EXP-040's pretrained encoder, frozen, exactly as exp043_capped_d5 loaded it.

    NOT tracked in git (it is reproducible from EXP-040's phase 1 in about 15 min per seed), so
    a fresh worktree will be missing it and the pre-flight below refuses rather than guessing.
    All 24 exist on the laptop's MAIN checkout.
    """
    return E0_DIR / f"exp040_encoder_s{seed}.pt"


def sweep_configs(seeds, out_dir: Path, arms) -> list[CubeConfig]:
    return [
        CubeConfig(
            arm="regionalized", readout=readout, tag=tag,
            depth=DEPTH, seed=seed, sigma=0.0, episodes=EPISODES,
            curriculum=tuple(range(1, DEPTH + 1)), max_steps_by_depth=CAP,
            entropy_beta=0.0, normalize_advantages=False,
            encoder_state_path=str(e0_encoder(seed)),
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

    missing = [str(e0_encoder(s)) for s in args.seeds if not e0_encoder(s).exists()]
    if missing:
        raise SystemExit(
            f"missing {len(missing)} E0 encoder(s), first {missing[:2]}. exp040_encoder_s*.pt are "
            "NOT tracked in git. All 24 live on the laptop's MAIN checkout; copy them into the "
            "worktree, or regenerate from EXP-040 phase 1 at ~15 min per seed."
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

    print(f"EXP-059: depth {DEPTH}, {EPISODES:,} episodes, {len(configs)} runs, "
          f"{args.workers} workers")
    print(f"  arms {tuple(args.arms)}, ONE variable: the readout. Encoder FROZEN (exp040 E0).")
    print(f"  PRIMARY is M minus A, NOT M minus S, and it is NOT directional: -0.05 at")
    print(f"  significance is a real finding that memory HURTS, not a failed confirmation.")
    print(f"  Claim 1 bar +/-{BAR:.2f} on success; Claim 2 bar {REVISIT_BAR:+.2f} on revisit_rate.")
    print(f"  GATE: arm M recall_content_cos < {GATE_MAX_RECALL_COS} (empty attractor = 1.0,")
    print(f"        real depth-5 run measured 0.8128), arm S unshuffled_frac < "
          f"{GATE_MAX_UNSHUFFLED}.")
    print(f"  POWER: n=24 gives ~60-70% at a 0.05 effect, ~30-40% at 0.03. EXP-058's ordering")
    print(f"        suggested about -0.03, so an indistinguishable Claim 1 is plausible.")
    print(f"  context, not a control: exp043_capped_d5 = {CAPPED_D5_CONTEXT}\n", flush=True)

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
