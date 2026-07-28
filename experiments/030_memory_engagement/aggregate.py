# experiments/030_memory_engagement/aggregate.py
"""Aggregate EXP-030 records.

The headline is the PAIRED memory minus memory_shuffled difference at matched
(depth, seed): both arms have identical head width, so a gap is memory content rather
than capacity. The revisit rate and mean stored-pattern count are reported alongside,
because a null with near-zero revisits is a statement about the task.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(out_dir.glob("exp030_*.json"))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=HERE / "outputs")
    ap.add_argument("--out", type=Path, default=HERE / "outputs" / "030_curve.md")
    args = ap.parse_args()

    records = load(args.runs)
    if not records:
        raise SystemExit(f"no run records found in {args.runs}")

    episodes_counts = defaultdict(int)
    for r in records:
        episodes_counts[r["episodes"]] += 1
    if len(episodes_counts) > 1:
        print("WARNING: Records from multiple episode budgets detected:")
        for ep in sorted(episodes_counts.keys()):
            print(f"  episodes={ep}: {episodes_counts[ep]} records")
        print("  This may indicate stale records from a previous run. Ensure you intend to blend these.\n")

    cells = defaultdict(list)
    revisit = defaultdict(list)
    stored = defaultdict(list)
    by_seed = defaultdict(dict)
    for r in records:
        cells[(r["readout"], r["depth"])].append(r["success_rate"])
        revisit[(r["readout"], r["depth"])].append(r["revisit_rate"])
        stored[(r["readout"], r["depth"])].append(r["mean_n_stored"])
        by_seed[(r["depth"], r["seed"])][r["readout"]] = r["success_rate"]

    lines = [
        "# EXP-030 memory engagement",
        "",
        "Primary test is the paired memory minus memory_shuffled column: identical head",
        "width, so a gap is memory content rather than capacity. A near-zero revisit rate",
        "means there were no cycles to avoid, and any null must be read that way.",
        "",
        "| depth | concept | memory | shuffled | paired mem-shuf | revisit rate | mean stored | n |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for depth in sorted({d for _, d in cells}):
        def mean(mode, table=cells, fmt="{:.0f}%", scale=100):
            vals = table.get((mode, depth), [])
            return fmt.format(scale * sum(vals) / len(vals)) if vals else "n/a"

        pairs = [v["memory"] - v["memory_shuffled"]
                 for (d, _), v in by_seed.items()
                 if d == depth and "memory" in v and "memory_shuffled" in v]
        diff = f"{100 * sum(pairs) / len(pairs):+.0f} pts" if pairs else "n/a"
        lines.append(
            f"| {depth} | {mean('concept')} | {mean('memory')} | {mean('memory_shuffled')} "
            f"| {diff} | {mean('concept', revisit, '{:.3f}', 1)} "
            f"| {mean('memory', stored, '{:.1f}', 1)} | {len(pairs)} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
