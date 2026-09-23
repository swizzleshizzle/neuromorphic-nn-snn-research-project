"""Does a fresh clone reproduce a PUBLISHED result end to end, on a different machine?

`verify_regeneration.py` showed the E0 encoders are deterministic on the laptop that made them
and **not portable** off it: 0 of 2 byte-identical here, 7.1% of parameters matching, cosine
0.77. That was ENCODER PRETRAINING. This asks the broader and more important question:

> Is the non-portability specific to pretraining, or does EVERY seeded run in this project
> depend on the machine it was run on?

**EXP-036 is the right probe** because it takes **no pretrained encoder** - its brain is frozen
at random init, so the only inputs are code and a seed. If EXP-036 reproduces byte-for-byte
here, the encoder finding is contained to pretraining. If it does not, then `CLAUDE.md`'s
"seeded runs are byte-identical" holds only WITHIN a machine, and every published number in the
project is machine-specific.

The comparison is against `*_head.pt`, which is tracked in git, because the JSON records are
gitignored and a checkout does not carry them. Byte comparison of the trained policy head is a
far more sensitive test than a success rate: two runs can agree on a success rate to 3 decimals
and hold completely different weights.

> **The scratch directory is NOT the experiment's `outputs/`, deliberately.** Writing there
> would overwrite tracked checkpoints with a differing reproduction, which is the one mistake
> that would destroy the thing being measured.

Usage:
    .venv/bin/python -u scripts/verify_e2e_reproduction.py --depth 3 --seeds 0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments" / "036_generalisation_gap"

_spec = importlib.util.spec_from_file_location("exp036_run", EXP / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

from neuromorphic.training.cube_baseline import record_filename, run_cube_baseline  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def head_path(out_dir: Path, cfg) -> Path:
    return out_dir / record_filename(cfg).replace(".json", "_head.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--scratch", type=Path, default=REPO / "experiments/036_generalisation_gap/outputs_e2e_check")
    args = ap.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)

    # Rebuild the EXACT config the published run used, from the experiment's own sweep, rather
    # than retyping its fields. Retyping is how a reproduction silently tests a different cell.
    configs = [c for c in _run.sweep_configs(args.seeds, args.scratch)
               if c.arm == "regionalized" and c.depth == args.depth and c.seed in args.seeds]
    if not configs:
        raise SystemExit(f"no regionalized cell at depth {args.depth} for seeds {args.seeds}")

    published = EXP / "outputs"
    missing = []
    for c in configs:
        ref = head_path(published, c._replace(out_dir=published) if hasattr(c, "_replace") else c)
        # CubeConfig is a dataclass, not a namedtuple; the filename does not depend on out_dir.
        ref = published / record_filename(c).replace(".json", "_head.pt")
        if not ref.exists():
            missing.append(ref.name)
    if missing:
        raise SystemExit(f"published checkpoint(s) absent, nothing to compare: {missing}")

    print(f"END-TO-END REPRODUCTION CHECK - EXP-036, depth {args.depth}, seeds {args.seeds}")
    print( "  EXP-036 takes NO pretrained encoder: the only inputs are code and a seed.")
    print(f"  published : {published}")
    print(f"  scratch   : {args.scratch}")
    print( "  Comparing the TRAINED POLICY HEAD byte for byte, not a success rate.\n", flush=True)

    rows = []
    for c in configs:
        t0 = time.time()
        r = run_cube_baseline(c)
        rows.append((c, r, round(time.time() - t0, 1)))

    print(f"{'seed':>5s} {'identical':>10s} {'published':>18s} {'reproduced':>18s} "
          f"{'success':>8s} {'wall_s':>8s}")
    identical = 0
    for c, r, wall in rows:
        name = record_filename(c).replace(".json", "_head.pt")
        a, b = published / name, args.scratch / name
        same = a.read_bytes() == b.read_bytes()
        identical += int(same)
        print(f"{c.seed:>5d} {'YES' if same else 'NO':>10s} {sha(a):>18s} {sha(b):>18s} "
              f"{r['success_rate']:>8.4f} {wall:>8.1f}")

    print(f"\n{identical} of {len(rows)} byte-identical.")
    if identical == len(rows):
        print("PORTABLE. The non-portability is contained to encoder PRETRAINING, and the")
        print("main training path reproduces across machines.")
    else:
        print("NOT PORTABLE, and this is NOT about pretraining: EXP-036 uses no pretrained")
        print("encoder. 'Seeded runs are byte-identical' holds only WITHIN a machine, and every")
        print("published number in this project is machine-specific. Phase 4 must say so.")
        for c, r, _w in rows:
            a = published / record_filename(c).replace(".json", "_head.pt")
            b = args.scratch / record_filename(c).replace(".json", "_head.pt")
            wa, wb = torch.load(a, map_location="cpu"), torch.load(b, map_location="cpu")
            keys = sorted(set(wa) & set(wb))
            fa = torch.cat([wa[k].flatten() for k in keys])
            fb = torch.cat([wb[k].flatten() for k in keys])
            cos = float(torch.dot(fa, fb) / (fa.norm() * fb.norm()))
            print(f"  seed {c.seed}: cosine {cos:.6f}, max|delta| "
                  f"{float((fa - fb).abs().max()):.3e}, identical elements "
                  f"{int((fa == fb).sum())}/{fa.numel()}")


if __name__ == "__main__":
    main()
