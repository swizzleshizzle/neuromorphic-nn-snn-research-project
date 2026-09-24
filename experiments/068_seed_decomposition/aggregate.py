"""EXP-068 aggregator. Written and committed BEFORE any EXP-068 number exists.

Two-way ANOVA without replication over the split_seed x train_seed grid. There is no replication
to have: within one machine a cell is a deterministic function of its two seeds, so the
interaction term IS the error term.

Significance by randomization rather than an F table, because the repo has no scipy. The null is
"neither factor matters", so the test shuffles all cell values across the grid. The shuffle uses a
FIXED seed, so the aggregator is deterministic and a rerun reproduces its p-values exactly.

Spec: docs/superpowers/specs/2026-09-24-exp068-seed-decomposition-design.md
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent

GRID = tuple(range(10))
TASK_SHARE_BAR = 0.35
EXP067_PREDICTED_SHARE = 0.183
MIN_TOTAL_SD = 0.05
N_SHUFFLES = 20_000
SHUFFLE_SEED = 20260924
EXP036_D3_MEAN = 0.3972
EXP036_D3_SD = 0.1158


def load_grid(out_dir: Path):
    """grid[i][j] = success for split_seed i, train_seed j. None where a cell is missing."""
    grid = [[None] * len(GRID) for _ in GRID]
    for path in out_dir.glob("exp068_sp*tr*.json"):
        rec = json.load(open(path))
        if not isinstance(rec, dict) or "success_rate" not in rec:
            continue
        stem = path.name.split("_")[0] + "_" + path.name.split("_")[1]
        body = stem.split("_")[1]                      # sp<i>tr<j>
        split_s, train_s = body[2:].split("tr")
        grid[int(split_s)][int(train_s)] = float(rec["success_rate"])
    return grid


def anova(grid):
    """Two-way decomposition. Returns mean squares and variance components."""
    rows = len(grid)
    cols = len(grid[0])
    flat = [x for row in grid for x in row]
    if any(x is None for x in flat):
        raise ValueError("grid has missing cells; the decomposition needs all of them")
    grand = st.mean(flat)
    row_mean = [st.mean(r) for r in grid]
    col_mean = [st.mean([grid[i][j] for i in range(rows)]) for j in range(cols)]

    ss_a = cols * sum((m - grand) ** 2 for m in row_mean)
    ss_b = rows * sum((m - grand) ** 2 for m in col_mean)
    ss_i = sum(
        (grid[i][j] - row_mean[i] - col_mean[j] + grand) ** 2
        for i in range(rows) for j in range(cols)
    )
    df_a, df_b, df_i = rows - 1, cols - 1, (rows - 1) * (cols - 1)
    ms_a, ms_b, ms_i = ss_a / df_a, ss_b / df_b, ss_i / df_i

    # Negative estimates are set to zero: a variance cannot be negative, and the estimator can go
    # under when the true component is near zero. Reported as 0.0 rather than hidden.
    # A COLLAPSED GRID MUST NOT CRASH THE AGGREGATOR. If every cell is identical (a dead
    # policy, which is exactly what the gate exists to catch) then ms_i is 0 and the F ratios
    # are 0/0. The gate has to be reachable to report that, so the ratios are defined here
    # rather than left to raise: an F of 1.0 is "no more structure than noise", which is the
    # honest reading of a grid with no variance at all.
    f_a = ms_a / ms_i if ms_i > 0 else (float("inf") if ms_a > 0 else 1.0)
    f_b = ms_b / ms_i if ms_i > 0 else (float("inf") if ms_b > 0 else 1.0)

    sigma2_i = ms_i
    sigma2_a = max(0.0, (ms_a - ms_i) / cols)
    sigma2_b = max(0.0, (ms_b - ms_i) / rows)
    total = sigma2_a + sigma2_b + sigma2_i
    return {
        "grand": grand, "total_sd": st.pstdev(flat),
        "ms_a": ms_a, "ms_b": ms_b, "ms_i": ms_i,
        "f_a": f_a, "f_b": f_b,
        "sigma2_a": sigma2_a, "sigma2_b": sigma2_b, "sigma2_i": sigma2_i,
        "share_a": sigma2_a / total if total else 0.0,
        "share_b": sigma2_b / total if total else 0.0,
        "share_i": sigma2_i / total if total else 0.0,
    }


def randomization_p(grid, n_shuffles: int = N_SHUFFLES, seed: int = SHUFFLE_SEED):
    """Two p-values, for the row (task draw) and column (trajectory) main effects.

    Null: neither factor matters, so every cell value is exchangeable with every other.
    """
    observed = anova(grid)
    rows, cols = len(grid), len(grid[0])
    flat = [x for row in grid for x in row]
    rng = random.Random(seed)
    hits_a = hits_b = 0
    for _ in range(n_shuffles):
        rng.shuffle(flat)
        shuffled = [flat[i * cols:(i + 1) * cols] for i in range(rows)]
        got = anova(shuffled)
        if got["f_a"] >= observed["f_a"] - 1e-12:
            hits_a += 1
        if got["f_b"] >= observed["f_b"] - 1e-12:
            hits_b += 1
    return (hits_a + 1) / (n_shuffles + 1), (hits_b + 1) / (n_shuffles + 1)


def check_gate(stats, p_a: float, p_b: float):
    """Claim 3. BOTH conditions, and the second is the resolution check.

    Condition 1 alone would pass on a grid whose variance is entirely interaction, and Claim 1's
    share would then be a ratio of two noise estimates.
    """
    sd_ok = stats["total_sd"] >= MIN_TOTAL_SD
    resolves = min(p_a, p_b) < 0.05
    return sd_ok and resolves, sd_ok, resolves


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    grid = load_grid(args.out_dir)
    filled = sum(1 for row in grid for x in row if x is not None)
    print(f"EXP-068: split_seed x train_seed at depth 3")
    print(f"  cells {filled} of {len(GRID) ** 2}")
    if filled != len(GRID) ** 2:
        raise SystemExit("incomplete grid; refusing to decompose a partial design.")

    stats = anova(grid)
    p_a, p_b = randomization_p(grid)
    diag = [grid[i][i] for i in GRID]

    print(f"\n--- the grid ---")
    print(f"  grand mean {stats['grand']:.4f}   total sd {stats['total_sd']:.4f}")
    print(f"  diagonal (the ordinary config) mean {st.mean(diag):.4f}  sd {st.pstdev(diag):.4f}")
    print(f"  EXP-036 published this cell at mean {EXP036_D3_MEAN}, per-seed sd {EXP036_D3_SD}")

    print(f"\n--- variance decomposition (two-way, no replication) ---")
    print(f"  {'component':<28}{'MS':>10}{'F vs MS_I':>12}{'variance':>12}{'share':>9}")
    print(f"  {'A  task draw (split_seed)':<28}{stats['ms_a']:>10.5f}{stats['f_a']:>12.3f}"
          f"{stats['sigma2_a']:>12.5f}{stats['share_a']:>9.3f}")
    print(f"  {'B  trajectory (train_seed)':<28}{stats['ms_b']:>10.5f}{stats['f_b']:>12.3f}"
          f"{stats['sigma2_b']:>12.5f}{stats['share_b']:>9.3f}")
    print(f"  {'I  interaction / residual':<28}{stats['ms_i']:>10.5f}{'-':>12}"
          f"{stats['sigma2_i']:>12.5f}{stats['share_i']:>9.3f}")
    print(f"  randomization p ({N_SHUFFLES:,} shuffles, fixed seed {SHUFFLE_SEED}): "
          f"A {p_a:.4f}   B {p_b:.4f}")

    ok, sd_ok, resolves = check_gate(stats, p_a, p_b)
    print(f"\n--- CLAIM 3, VALIDITY GATE ---")
    print(f"  total sd {stats['total_sd']:.4f} >= {MIN_TOTAL_SD}: {'PASS' if sd_ok else 'FAIL'}")
    print(f"  at least one main effect resolves (min p {min(p_a, p_b):.4f} < 0.05): "
          f"{'PASS' if resolves else 'FAIL'}")
    print(f"  GATE: {'PASS' if ok else 'FAIL - the experiment is VOID'}")
    if not ok:
        raise SystemExit("gate failed; the claims below may not be read.")

    print(f"\n--- CLAIM 1, PRIMARY: is the task draw a MINORITY of the seed effect? ---")
    share = stats["share_a"]
    verdict = "CONFIRMED" if share < TASK_SHARE_BAR else "REFUTED"
    print(f"  task-draw share {share:.3f}   bar < {TASK_SHARE_BAR}   EXP-067 predicted "
          f"~{EXP067_PREDICTED_SHARE}")
    print(f"  VERDICT: {verdict}")
    if 0.25 <= share <= 0.45:
        print(f"  ==UNRESOLVED BAND==: the spec pre-registered that a share between 0.25 and 0.45")
        print(f"     is not cleanly resolved by this design and must be reported as unresolved")
        print(f"     rather than rounded to the nearest verdict.")

    print(f"\n--- CLAIM 2, SECONDARY: does the TRAJECTORY main effect resolve? ---")
    print(f"  F_B {stats['f_b']:.3f}, p {p_b:.4f}, bar p < 0.05")
    print(f"  VERDICT: {'CONFIRMED' if p_b < 0.05 else 'NOT CONFIRMED'}")
    if share < TASK_SHARE_BAR and p_b >= 0.05:
        print(f"  NOTE: Claim 1 confirmed with Claim 2 null means the effect is mostly")
        print(f"     INTERACTION, which is a third answer and is reported as such.")


if __name__ == "__main__":
    main()
