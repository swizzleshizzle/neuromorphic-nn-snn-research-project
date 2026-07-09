"""Aggregate EXP-027 per-seed characterization into paired region contrasts vs sensory."""

from __future__ import annotations


def aggregate_regions(per_seed: list) -> dict:
    """For each non-sensory region, paired win-fraction of sensory-concept over it, per target."""
    regions = [r for r in per_seed[0]["regions"] if r != "sensory"]
    out = {}
    for reg in regions:
        d_wins = a_wins = n = 0
        for s in per_seed:
            sen, oth = s["regions"]["sensory"], s["regions"][reg]
            d_wins += int(sen["displacement_r2"] > oth["displacement_r2"])
            a_wins += int(sen["optimal_action_acc"] > oth["optimal_action_acc"])
            n += 1
        out[reg] = {"n": n, "displacement_win_fraction": d_wins / n,
                    "optimal_action_win_fraction": a_wins / n}
    return out


def format_matrix(agg: dict) -> str:
    """Markdown region x target table: paired win-fraction of the sensory concept over each region."""
    lines = [
        "| region | n | displacement win-frac (sensory > region) | optimal-action win-frac (sensory > region) |",
        "| --- | --- | --- | --- |",
    ]
    for reg, m in agg.items():
        lines.append(
            f"| {reg} | {m['n']} | {m['displacement_win_fraction']:.0%} | "
            f"{m['optimal_action_win_fraction']:.0%} |"
        )
    return "\n".join(lines)
