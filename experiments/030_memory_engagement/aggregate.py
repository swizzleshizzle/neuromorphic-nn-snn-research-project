# experiments/030_memory_engagement/aggregate.py
"""Aggregate EXP-030 records.

Two paired columns are reported, at matched (depth, seed):

- ``memory minus shuffled``: memory vs memory_shuffled. Both arms have identical head
  width and receive real, in-distribution recall/familiarity features, so this isolates
  memory CORRESPONDENCE (does it matter that the memory matches the current state?).
- ``memory minus amnesic``: memory vs memory_amnesic. Both arms feed the head the same
  feed-forward expansion of the CURRENT concept; amnesic differs only in that the
  attractor is emptied (W_rec zeroed) at read time. This isolates memory CONTENT (does
  having any stored content help at all, as opposed to just a wider head reading the
  state it is already looking at?).

The greedy-policy revisit rate and mean stored-pattern count are reported alongside,
because a null with near-zero revisits is a statement about the task. The greedy rate
(not the training-policy one) is the decision metric: it is computed over the
deterministic evaluation rollouts, where a repeated state means a genuine cycle, rather
than over the stochastic training policy, which revisits by construction (a random walk
undoes a move about 1 step in 6).
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
        revisit[(r["readout"], r["depth"])].append(r["eval_revisit_rate"])
        stored[(r["readout"], r["depth"])].append(r["mean_n_stored"])
        by_seed[(r["depth"], r["seed"])][r["readout"]] = r["success_rate"]

    lines = [
        "# EXP-030 memory engagement",
        "",
        "Two paired columns, both at matched (depth, seed). ``memory minus shuffled``",
        "isolates memory CORRESPONDENCE (does it matter that the memory matches the",
        "current state?): both arms have identical head width and real, in-distribution",
        "recall/familiarity features. ``memory minus amnesic`` isolates memory CONTENT",
        "(does having any stored content help at all, versus a wider head reading only",
        "the current state?): both arms feed the head the same feed-forward expansion of",
        "the current concept, differing only in whether the attractor holds anything at",
        "read time. A near-zero GREEDY revisit rate means there were no cycles to avoid,",
        "and any null must be read that way. The greedy rate is computed over the",
        "deterministic evaluation policy, not the stochastic training policy, which",
        "revisits by construction and is not a decision metric.",
        "",
        "| depth | concept | memory | shuffled | amnesic | paired mem-shuf | paired mem-amn "
        "| greedy revisit | mean stored | n shuf | n amn |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for depth in sorted({d for _, d in cells}):
        def mean(mode, table=cells, fmt="{:.0f}%", scale=100):
            vals = table.get((mode, depth), [])
            return fmt.format(scale * sum(vals) / len(vals)) if vals else "n/a"

        def paired(a, b):
            vals = [v[a] - v[b]
                    for (d, _), v in by_seed.items()
                    if d == depth and a in v and b in v]
            diff = f"{100 * sum(vals) / len(vals):+.0f} pts" if vals else "n/a"
            return diff, len(vals)

        diff_shuf, n_shuf = paired("memory", "memory_shuffled")
        diff_amn, n_amn = paired("memory", "memory_amnesic")
        lines.append(
            f"| {depth} | {mean('concept')} | {mean('memory')} | {mean('memory_shuffled')} "
            f"| {mean('memory_amnesic')} | {diff_shuf} | {diff_amn} "
            f"| {mean('concept', revisit, '{:.3f}', 1)} "
            f"| {mean('memory', stored, '{:.1f}', 1)} | {n_shuf} | {n_amn} |"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
