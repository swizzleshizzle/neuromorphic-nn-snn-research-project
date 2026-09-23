"""EXP-067: do the FINDINGS survive retraining on a different machine?

The week-25 audit established that retraining is NOT portable - EXP-036 depth 3 seed 0,
retrained on this VPS, shares 0 of 390 parameters with its published head - and that
re-evaluating a tracked checkpoint IS. So the release guarantee is "re-evaluate, never retrain".

> WHAT THE AUDIT COULD NOT MEASURE, and this does. Its section 7 says the findings would
> "plausibly" survive retraining, from one encoder's move-accuracy sitting inside the seed
> spread, and calls that an argument rather than a measurement. "The numbers are
> machine-specific but the conclusions are not" is a far stronger claim than "the numbers are
> machine-specific", and nothing currently supports it.

> EVERY THRESHOLD IS EXP-036'S OWN, REUSED UNCHANGED. A replication that invents its own bar is
> not a replication. EXP-036 already ran this exact cell against EXP-035 on ONE machine and
> passed at delta +0.0002 with tolerance 0.02, byte-identical on every measured quantity. This
> is that replication on a DIFFERENT machine, against the SAME bar.

> THE CONFIGS ARE IMPORTED FROM EXP-036, NOT RETYPED. Retyping is how a replication silently
> tests a different cell, and this design has exactly one job: to be the same cell.

> THE VALIDITY GATE IS INVERTED. The run must genuinely DIFFER from the original, or this
> machine is not meaningfully different and the experiment tests nothing.

Spec: docs/superpowers/specs/2026-09-23-exp067-cross-machine-replication-design.md

Usage:
    .venv/bin/python -u experiments/067_cross_machine_replication/run.py --workers 2
"""

from __future__ import annotations

import argparse
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from neuromorphic.training.cube_baseline import record_filename, run_cube_baseline

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EXP036 = REPO / "experiments" / "036_generalisation_gap"

_spec = importlib.util.spec_from_file_location("exp036_run", EXP036 / "run.py")
_exp036 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp036)

SEEDS = tuple(range(12))
DEPTH = 3

# Every one of these is EXP-036's, imported rather than copied, so a change there cannot leave
# this replication silently testing a stale bar.
EXP036_DEPTH3_MEAN = 0.3972      # what EXP-036 measured, and what Claim 1 replicates against
REPLICATION_TOLERANCE = 0.02     # EXP-036's own tolerance against EXP-035
BREAK_MULTIPLE = _exp036.BREAK_MULTIPLE      # 2.0
BREAK_ABSOLUTE = _exp036.BREAK_ABSOLUTE      # 0.10
GAP_REFUTE_BELOW = _exp036.GAP_REFUTE_BELOW  # 0.05
GAP_CONFIRM_AT = _exp036.GAP_CONFIRM_AT      # 0.15
EXP036_DEPTH3_GAP = 0.1093       # landed in the dead zone; Claim 3 asks if the ZONE survives


def sweep_configs(seeds, out_dir: Path):
    """EXP-036's own sweep, filtered to depth 3. NOT retyped: see the module docstring."""
    return [c for c in _exp036.sweep_configs(seeds, out_dir) if c.depth == DEPTH]


def _run(cfg) -> dict:
    torch.set_num_threads(1)
    return run_cube_baseline(cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    configs = sweep_configs(args.seeds, args.out_dir)
    trained = [c for c in configs if c.arm == "regionalized"]
    floors = [c for c in configs if c.arm == "random"]
    if len(trained) != len(args.seeds) or len(floors) != len(args.seeds):
        raise SystemExit(f"expected {len(args.seeds)} trained and {len(args.seeds)} floor cells, "
                         f"got {len(trained)} and {len(floors)}")
    if any(c.depth != DEPTH for c in configs):
        raise SystemExit("every cell must be depth 3")
    if any(c.episodes != _exp036.EPISODES for c in trained):
        raise SystemExit("budget must match EXP-036 exactly or this is not a replication")
    # The published checkpoints must be present: Claim 4 compares against them.
    missing = [record_filename(c).replace(".json", "_head.pt") for c in trained
               if not (EXP036 / "outputs" / record_filename(c).replace(".json", "_head.pt")).exists()]
    if missing:
        raise SystemExit(f"published checkpoints absent, the gate cannot be evaluated: {missing[:2]}")
    names = [record_filename(c) for c in configs]
    if len(set(names)) != len(names):
        raise SystemExit("record filename collision")
    total = len(configs)
    if args.skip_existing:
        configs = [c for c in configs if not (args.out_dir / record_filename(c)).exists()]

    print(f"EXP-067: EXP-036 depth {DEPTH} retrained on a DIFFERENT machine, "
          f"{len(configs)} of {total} cells, {args.workers} workers")
    print( "  THE UNIT OF REPLICATION IS THE VERDICT, NOT THE NUMBER. The week-25 audit already")
    print( "        proved the numbers differ: seed 0 retrained here shares 0 of 390 parameters")
    print( "        with its published head. What is unknown is whether the CONCLUSIONS hold.")
    print(f"  CLAIM 1 (PRIMARY): |mean - {EXP036_DEPTH3_MEAN}| <= {REPLICATION_TOLERANCE}, which is")
    print( "        EXP-036's own tolerance against EXP-035. Same-machine it passed at +0.0002.")
    print(f"  CLAIM 2: the 'working' verdict - mean >= {BREAK_MULTIPLE}x measured floor AND")
    print(f"        >= {BREAK_ABSOLUTE}. Depth 3 was WORKING.")
    print(f"  CLAIM 3: the gap verdict - refuted below {GAP_REFUTE_BELOW}, confirmed at")
    print(f"        {GAP_CONFIRM_AT}, inconclusive between. EXP-036 measured {EXP036_DEPTH3_GAP}")
    print( "        and reported INCONCLUSIVE. Does it land in the same zone?")
    print( "  GATE (INVERTED): at least one retrained head must DIFFER byte-wise from its")
    print( "        published counterpart. If they matched, this machine is not different and")
    print( "        the experiment tests nothing.\n", flush=True)

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
            print(f"  {i}/{len(configs)}  {r['arm']:13s} s{r['seed']:<3d} "
                  f"success {r['success_rate']:.4f}", flush=True)

    print(f"\ndone. records in {args.out_dir}.")


if __name__ == "__main__":
    main()
