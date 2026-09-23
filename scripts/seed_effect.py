"""How much of a "level shift" between two seed blocks is just seed sampling?

EXP-060 found both of its arms sitting 0.05 to 0.08 lower on seeds 14-23 than the seeds
0-11 block it was replicating, flagged it as unexplained, and correctly refused to pool
across it. This measures the thing that explains it.

THE POINT: a cube seed fixes the E0 encoder, the train/held-out split and the head init
all at once, so seed quality is a single persistent property that EVERY downstream
experiment inherits. It is large (sd about 0.09 in success rate) and it reproduces across
experiments that share nothing else. Against that variance, a 0.05 gap between two
disjoint seed sets is the expected amount of noise, not a finding.

Read `docs/seed-effect.md` for the worked conclusion. This script regenerates its numbers.

Usage:
    .venv/bin/python -u scripts/seed_effect.py
    .venv/bin/python -u scripts/seed_effect.py --depth 5 --block-a 0-11 --block-b 14-23
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from itertools import combinations
from pathlib import Path

RECORD_GLOB = "experiments/*/outputs/*.json"


def load_records(root: Path = Path(".")) -> dict[tuple[str, int], dict[int, float]]:
    """Every record keyed by (tag, depth) -> {seed: success_rate}.

    Some experiments write a JSON list per file rather than one object, so both are
    handled. Anything without a success rate and a seed is not a cell record.
    """
    out: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    for path in glob.glob(str(root / RECORD_GLOB)):
        try:
            obj = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        for rec in obj if isinstance(obj, list) else [obj]:
            if not isinstance(rec, dict):
                continue
            if "success_rate" not in rec or "seed" not in rec or rec.get("depth") is None:
                continue
            out[(rec.get("tag", "?"), int(rec["depth"]))][int(rec["seed"])] = float(
                rec["success_rate"]
            )
    return dict(out)


def spanning_arms(records, depth, block_a, block_b, min_per_block=10):
    """Arms at `depth` that ran enough seeds in BOTH blocks to be residualised.

    An arm that covers only one block cannot separate a seed effect from that arm's own
    level, which is exactly the confound that makes depth 7 unreadable here.
    """
    return {
        key: seeds
        for key, seeds in records.items()
        if key[1] == depth
        and sum(1 for s in block_a if s in seeds) >= min_per_block
        and sum(1 for s in block_b if s in seeds) >= min_per_block
    }


def per_seed_effect(arms) -> dict[int, float]:
    """Mean success residual per seed, after removing each arm's own mean.

    Residualising is load-bearing, not tidiness. The arms differ enormously in level (an
    amnesic readout beats an attention readout by 0.2), so a raw per-seed average across
    arms would be dominated by which arms happened to cover which seeds.
    """
    residuals: dict[int, list[float]] = defaultdict(list)
    for seeds in arms.values():
        arm_mean = st.mean(seeds.values())
        for seed, value in seeds.items():
            residuals[seed].append(value - arm_mean)
    return {seed: st.mean(vals) for seed, vals in residuals.items()}


def pearson(xs, ys) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def reproducibility(arms, min_common=20):
    """Mean per-seed correlation over all arm pairs: does seed quality REPLICATE?

    This is the check that separates a real seed property from per-run noise. Noise
    cannot correlate across two experiments that share no records.
    """
    keys = sorted(arms)
    cors = []
    for a, b in combinations(keys, 2):
        common = sorted(set(arms[a]) & set(arms[b]))
        if len(common) < min_common:
            continue
        cors.append(pearson([arms[a][s] for s in common], [arms[b][s] for s in common]))
    if not cors:
        return float("nan"), 0, 0
    return st.mean(cors), len(cors), sum(1 for c in cors if c > 0)


def exact_block_p(values, block_a, block_b):
    """Two-sided exact permutation test on the block labels.

    Enumerates every way of splitting the pooled seeds into groups of the observed sizes,
    so there is no normal approximation and no scipy. Returns (observed difference,
    p-value, number of partitions).

    The difference is b minus a, and the test is two-sided: a level shift is interesting
    in either direction, and the direction here was chosen after seeing the data.
    """
    a = [values[s] for s in block_a if s in values]
    b = [values[s] for s in block_b if s in values]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each block needs at least two seeds with values")
    pool = a + b
    total, n_b, n_a = sum(pool), len(b), len(a)
    observed = sum(b) / n_b - sum(a) / n_a
    hits = count = 0
    for combo in combinations(pool, n_b):
        s = sum(combo)
        count += 1
        if abs(s / n_b - (total - s) / n_a) >= abs(observed) - 1e-12:
            hits += 1
    return observed, hits / count, count


def parse_block(text: str) -> list[int]:
    lo, hi = text.split("-")
    return list(range(int(lo), int(hi) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--block-a", default="0-11")
    ap.add_argument("--block-b", default="14-23")
    args = ap.parse_args()

    block_a, block_b = parse_block(args.block_a), parse_block(args.block_b)
    records = load_records()
    arms = spanning_arms(records, args.depth, block_a, block_b)
    if not arms:
        raise SystemExit(f"no arms at depth {args.depth} span both blocks")

    effect = per_seed_effect(arms)
    mean_r, n_pairs, n_pos = reproducibility(arms)
    observed, p, n_parts = exact_block_p(effect, block_a, block_b)
    sd = st.pstdev(list(effect.values()))

    print(f"depth {args.depth}: {len(arms)} arms span seeds {args.block_a} and {args.block_b}")
    for key in sorted(arms):
        print(f"    {key[0]}")

    print("\nIS THE SEED EFFECT REAL? (it cannot correlate across arms if it is noise)")
    print(f"  mean per-seed correlation over {n_pairs} arm pairs   {mean_r:+.3f}")
    print(f"  pairs with positive correlation                  {n_pos}/{n_pairs}")
    print(f"  per-seed effect sd                               {sd:.4f}")

    print("\nPER-SEED EFFECT (success residual, averaged over arms)")
    for seed in sorted(effect):
        print(f"  seed {seed:>2}  {effect[seed]:+.4f}")

    print(f"\nIS THE {args.block_a} vs {args.block_b} SPLIT UNUSUAL?")
    print(f"  observed block difference                        {observed:+.4f}")
    print(f"  exact two-sided p over {n_parts:,} partitions      {p:.4f}")

    print("\nWHAT THIS COSTS A COMPARISON")
    print(f"  se of one {len(block_a)}-seed arm mean                    "
          f"{sd / len(block_a) ** 0.5:.4f}")
    print(f"  se of a DISJOINT {len(block_a)}-vs-{len(block_b)} seed comparison      "
          f"{(sd * sd / len(block_a) + sd * sd / len(block_b)) ** 0.5:.4f}")
    print("  se of a PAIRED same-seed comparison              0.0000 from this source")


if __name__ == "__main__":
    main()
