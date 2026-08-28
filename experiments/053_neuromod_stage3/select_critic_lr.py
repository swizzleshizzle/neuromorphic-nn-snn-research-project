"""EXP-053: choose the critic learning rate, without ever seeing a success rate.

`critic_lr` has no prior. EXP-047 faced the same problem with `encoder_lr` and solved it with
a rule fixed in the spec before any pilot number existed, executed by a script that read only
the probe output. Same discipline here.

THE RULE, fixed in the spec section 4.3 before the pilot ran:

    choose the lr maximising the critic's EXPLAINED VARIANCE of realized returns, averaged
    over the pilot seeds, measured on the deepest curriculum stage.
    Ties break toward the SMALLER lr.

`select()` is handed only `critic_lr`, `seed`, `critic_ev` and `critic_n`. It never reads
`success_rate`, and `tests/training/test_select_critic_lr.py` proves it by feeding it a grid
where the best explained variance has the worst success rate.

`explained_variance()` returns exactly 0.0 both when a stage has no data (`critic_n == 0`)
and when the critic genuinely explains nothing. Those are not the same thing, and averaging
a degenerate 0.0 into a real mean would silently change which lr wins. `select()` therefore
REFUSES (raises) on any cell with `critic_n == 0` rather than folding it into the mean or
dropping it - a refusal is safe because it is symmetric, it cannot favour any learning rate,
it can only stop. The ranking rule itself does not change.

Usage:
    .venv/bin/python experiments/053_neuromod_stage3/select_critic_lr.py
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PILOT_SEEDS = (12, 13)


def select(records: list[dict]) -> float:
    """The pre-registered rule. Reads critic_lr, seed, critic_ev and critic_n. Nothing else.

    Raises ValueError if any cell has critic_n == 0: that cell's critic_ev of 0.0 cannot be
    told apart from a genuine zero, and selecting on it would silently distort that lr's
    mean. This is a refusal, not a re-ranking - it stops the whole selection rather than
    dropping or reweighting the offending cell, so it cannot move the choice between two
    valid cells.
    """
    by_lr: dict[float, dict[int, float]] = defaultdict(dict)
    for r in records:
        lr = float(r["config"]["critic_lr"])
        seed = int(r["seed"])
        if r["critic_n"] == 0:
            raise ValueError(
                f"critic_n is 0 for critic_lr={lr:g} seed={seed}: this cell's critic_ev is "
                "degenerate (no data), not a measured value, and cannot be selected on."
            )
        by_lr[lr][seed] = float(r["critic_ev"])

    sizes = {lr: len(seeds) for lr, seeds in by_lr.items()}
    if len(set(sizes.values())) != 1:
        raise ValueError(
            f"incomplete grid: cells have different seed counts {sizes}. Selecting on "
            "different samples per lr would compare rates on unequal evidence."
        )
    means = {lr: st.mean(seeds.values()) for lr, seeds in by_lr.items()}
    best = max(means.values())
    # Ties break toward the smaller lr: less parameter movement for the same fit.
    return min(lr for lr, m in means.items() if m == best)


def main() -> None:
    records = []
    for p in sorted((HERE / "outputs").glob("exp053_pilot_lr*.json")):
        r = json.loads(p.read_text())
        records.append({"config": {"critic_lr": r["config"]["critic_lr"]},
                        "seed": r["seed"], "critic_ev": r["critic_ev"],
                        "critic_n": r["critic_n"]})
    if not records:
        raise SystemExit("no pilot records in outputs/; run pilot_critic_lr.py first")

    by_lr: dict[float, list[tuple[float, int]]] = defaultdict(list)
    for r in records:
        by_lr[float(r["config"]["critic_lr"])].append((float(r["critic_ev"]), int(r["critic_n"])))

    print(f"{'critic_lr':>12} {'mean EV':>10} {'per-seed EV':>24} {'per-seed n':>24}")
    for lr in sorted(by_lr):
        cells = by_lr[lr]
        evs = [ev for ev, _ in cells]
        ns = [n for _, n in cells]
        print(f"{lr:>12.0e} {st.mean(evs):>10.4f}   {['%.4f' % e for e in evs]!s:>22} "
              f"{ns!s:>24}")
    print(f"\nSELECTED critic_lr = {select(records):g}")
    print("Rule fixed in the spec section 4.3. This script cannot see a success rate.")


if __name__ == "__main__":
    main()
