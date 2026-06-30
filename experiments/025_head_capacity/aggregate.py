"""Aggregate EXP-025 per-run summaries into a paired head-vs-regime evidence table."""

from __future__ import annotations


def _regime(shaping: bool) -> str:
    return "shaped" if shaping else "sparse"


def aggregate(summaries: list[dict]) -> dict:
    """Group summaries by (head_type, regime); mean and spread (max-min) of success."""
    groups: dict[tuple[str, str], dict[str, list[float]]] = {}
    for s in summaries:
        key = (s["config"]["head_type"], _regime(s["config"]["shaping"]))
        g = groups.setdefault(key, {"heldout": [], "train": []})
        g["heldout"].append(s["eval"]["heldout"]["success_rate"])
        g["train"].append(s["eval"]["train"]["success_rate"])

    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, vals in groups.items():
        h, t = vals["heldout"], vals["train"]
        out[key] = {
            "n": len(h),
            "heldout_mean": round(sum(h) / len(h), 10),
            "heldout_spread": round(max(h) - min(h), 10),
            "train_mean": round(sum(t) / len(t), 10),
            "train_spread": round(max(t) - min(t), 10),
        }
    return out


def format_table(agg: dict) -> str:
    """Render the aggregate as a markdown table sorted by (regime, head_type)."""
    lines = [
        "| regime | head | n | heldout mean | heldout spread | train mean |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for (head, regime) in sorted(agg, key=lambda k: (k[1], k[0])):
        m = agg[(head, regime)]
        lines.append(
            f"| {regime} | {head} | {m['n']} | {m['heldout_mean']:.0%} | "
            f"{m['heldout_spread']:.0%} | {m['train_mean']:.0%} |"
        )
    return "\n".join(lines)
