# experiments/029_cube_baseline/aggregate.py
"""Aggregate EXP-029 records into the collapse table.

Reports mean success per (arm, depth) across seeds, with the paired regionalized-minus-
monolithic difference. Depths 1 and 2 are training-distribution; 3 to 6 are held-out, and
the table says which is which rather than leaving the reader to guess.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(out_dir: Path) -> list[dict]:
    """Every per-run record. One file per run, written by the workers."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("exp029_*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=HERE / "outputs")
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "029_curve.md")
    args = ap.parse_args()

    records = load(args.runs)
    if not records:
        raise SystemExit(f"no run records found in {args.runs}")
    cells = defaultdict(list)
    heldout = {}
    for r in records:
        cells[(r["arm"], r["depth"])].append(r["success_rate"])
        heldout[r["depth"]] = r["is_heldout"]

    by_seed = defaultdict(dict)
    for r in records:
        if r["arm"] in ("regionalized", "monolithic"):
            by_seed[(r["depth"], r["seed"])][r["arm"]] = r["success_rate"]

    lines = [
        "# EXP-029 collapse curve",
        "",
        "| depth | eval | regionalized | monolithic | random floor | paired diff | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for depth in sorted({d for _, d in cells}):
        def mean(arm):
            vals = cells.get((arm, depth), [])
            return f"{100 * sum(vals) / len(vals):.0f}%" if vals else "n/a"

        pairs = [
            v["regionalized"] - v["monolithic"]
            for (d, _), v in by_seed.items()
            if d == depth and "regionalized" in v and "monolithic" in v
        ]
        diff = f"{100 * sum(pairs) / len(pairs):+.0f} pts" if pairs else "n/a"
        label = "held-out" if heldout.get(depth) else "train-dist"
        lines.append(
            f"| {depth} | {label} | {mean('regionalized')} | {mean('monolithic')} | "
            f"{mean('random')} | {diff} | {len(pairs)} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
