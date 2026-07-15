"""Aggregate EXP-028 ablation cells into per-operator dose-response curves."""

from __future__ import annotations


def aggregate_curve(cells: list[dict]) -> dict:
    """cells -> {operator: {dose: mean heldout success across seeds}}."""
    buckets: dict = {}
    for c in cells:
        buckets.setdefault(c["operator"], {}).setdefault(c["dose"], []).append(
            c["heldout_success"]
        )
    return {
        op: {dose: sum(v) / len(v) for dose, v in sorted(doses.items())}
        for op, doses in buckets.items()
    }


def format_curve(curve: dict) -> str:
    """Markdown table: one row per dose (ascending union), one column per operator."""
    operators = sorted(curve)
    doses = sorted({d for op in operators for d in curve[op]})
    header = "| dose | " + " | ".join(operators) + " |"
    sep = "| --- | " + " | ".join("---" for _ in operators) + " |"
    rows = []
    for dose in doses:
        cells = []
        for op in operators:
            v = curve[op].get(dose)
            cells.append("-" if v is None else f"{v:.0%}")
        rows.append(f"| {dose} | " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows]) + "\n"
