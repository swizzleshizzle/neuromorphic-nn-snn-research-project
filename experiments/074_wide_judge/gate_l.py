"""Gate L (spec section 6): does the judge's lowest-J leaf of the 3-move tree lie closer to
solved than the root, more often than chance? Measured on probe states only. `distance` is the
BFS yardstick, passed in as a callable; the judge never sees it."""

from __future__ import annotations

import statistics as st

import torch

from neuromorphic.training.lookahead import tree_levels


def leaf_rank_margin(judge, probe, distance, seed, n_actions, k=3, lo=7, hi=11) -> dict:
    g = torch.Generator().manual_seed(seed)
    by: dict[int, tuple[list, list]] = {}
    for s, d in probe:
        if not lo <= d <= hi:
            continue
        leaves = tree_levels(s, k, n_actions)[k]
        with torch.no_grad():
            j = judge(leaves, g)
        leaf_d = [distance(x) for x in leaves]
        hits, chances = by.setdefault(d, ([], []))
        hits.append(int(leaf_d[int(j.argmin())] < d))  # first minimum
        chances.append(sum(x < d for x in leaf_d) / len(leaf_d))
    all_h = [h for hs, _ in by.values() for h in hs]
    all_c = [c for _, cs in by.values() for c in cs]
    hit = st.mean(all_h) if all_h else float("nan")
    chance = st.mean(all_c) if all_c else float("nan")
    return {
        "hit": hit, "chance": chance, "margin": hit - chance, "n": len(all_h),
        "by_distance": {str(d): {"hit": st.mean(hs), "chance": st.mean(cs), "n": len(hs)}
                        for d, (hs, cs) in sorted(by.items())},
    }
