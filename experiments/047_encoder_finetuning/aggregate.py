"""EXP-047 aggregator: apply the pre-registered rules to the records on disk.

WRITTEN BEFORE THE RUN WAS DISPATCHED. No record existed. Rules on disk before numbers.

Thresholds are the ones committed at 69bf1dc in
docs/superpowers/specs/2026-08-20-exp047-encoder-finetuning-design.md.

> CLAIM 1 IS PAIRED AND CARRIES A P-VALUE: EXP-043's depth-6 cell, same seeds, same encoders at
> init, same cap, same curriculum, same 10,000 episodes, differing only in whether the encoder
> trains.

> THE AMBIGUOUS BAND IS PRE-REGISTERED AND IS PRINTED AS SUCH. Fine-tuning costs 1.33x per step
> and EXP-046's curve prices that at +0.027, so a delta between +0.027 and +0.05 is explainable
> by the extra compute alone. It is reported as AMBIGUOUS, never as a small win.

> AND AS ALWAYS: n=12 cannot show a failure count went to zero. Claim 4 carries no p-value.

Usage:
    .venv/bin/python experiments/047_encoder_finetuning/aggregate.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "043_cap_at_depth_5_6" / "outputs"

# --- pre-registered. Do not edit after data exists. ---
CLAIM1_DELTA = 0.05
COMPUTE_EQUIVALENT = 0.027    # EXP-046's 0.22 per log10, applied to the measured 1.33x
ALPHA = 0.05
DEPTH = 6
EPISODES = 10_000
BASELINE_MEAN = 0.1800        # EXP-043 depth 6
FROZEN_TRAINABLE = 390
FINETUNE_TRAINABLE = 27_206
STEP_COST_RATIO = 1.33        # 56.17 -> 74.87 ms, measured 2026-08-20 on the VPS
HEADLINE_DEPTH = 4            # EXP-039's primary probe depth
LEAKFREE_DEPTH = 6
# EXP-045's collapse signature, for the mechanism comparison.
EXP045 = {"entropy_first": 0.5914, "entropy_last": 0.0979, "entropy_min": 2.7e-06,
          "solve_frac": 0.0218}


def permutation_p(diffs: list[float]) -> float:
    """Exact paired permutation over all 2**n sign flips, two-sided. No scipy, no approximation."""
    n = len(diffs)
    if not 1 <= n <= 20:
        raise ValueError(f"exact permutation needs 1 <= n <= 20, got {n}")
    observed = abs(sum(diffs))
    hits = sum(1 for signs in itertools.product((1, -1), repeat=n)
               if abs(sum(s * d for s, d in zip(signs, diffs))) >= observed - 1e-12)
    return hits / 2 ** n


def load(d: Path, tag: str, *, exact: bool = False) -> dict:
    """Records for one cell, keyed by seed.

    `exact=True` for EXP-047's own records. The three pilot rates differ only by a float
    rendered into the tag (`lr0.001`, `lr0.0001`, `lr1e-05`), and a substring match over that
    is one unlucky rate away from silently pooling two arms into one mean. EXP-046's `in` test
    was safe because its tags differed by an integer episode count; here it is not worth the
    risk. The EXP-043 baseline keeps the substring form, since its tag carries a depth suffix.
    """
    out = {}
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text())
        # EXP-047's outputs directory holds more than records: `probe_pilot.json` and
        # `probe_confirm.json` are LISTS and `selected_lr.json` is a dict with no `tag`.
        # EXP-046's aggregator could assume every *.json was a record; this one cannot.
        # Fixed 2026-08-22 after the run, and it moves NO threshold - it only stops the
        # aggregator crashing on files that are not records. Same class of post-run change
        # as EXP-040's "print each margin in standard errors".
        if not isinstance(r, dict):
            continue
        hit = (r.get("tag") == tag) if exact else (tag in r.get("tag", ""))
        if hit and r.get("depth") == DEPTH and r.get("arm") == "regionalized":
            out[r["seed"]] = r
    return out


def sd(xs):
    return st.stdev(xs) if len(xs) > 1 else 0.0


def deepest_stage(rec):
    trace = rec.get("stage_trace") or []
    return next((s for s in reversed(trace) if s.get("depth") == DEPTH), None)


def verdict_claim1(delta: float, p: float) -> str:
    """The pre-registered three-way call, including the band compute alone can explain."""
    if delta >= CLAIM1_DELTA and p <= ALPHA:
        return "CONFIRMED"
    if delta > COMPUTE_EQUIVALENT:
        return (f"AMBIGUOUS (above the +{COMPUTE_EQUIVALENT} that 1.33x compute alone buys, "
                f"but not the pre-registered +{CLAIM1_DELTA} at p <= {ALPHA})")
    return "REFUTED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    sel_path = args.out_dir / "selected_lr.json"
    if not sel_path.exists():
        raise SystemExit(f"{sel_path} not found. Selection must have run before aggregation.")
    selected = json.loads(sel_path.read_text(encoding="utf-8"))["selected_lr"]
    if selected is None:
        raise SystemExit("select_lr.py halted the chain; there is no confirmatory arm to "
                         "aggregate. The pilot IS the result. See spec 5.2 step 3.")

    tag = f"exp047_ft_d{DEPTH}_lr{selected:g}"
    new = load(args.out_dir, tag, exact=True)
    base = load(BASELINE, f"exp043_capped_d{DEPTH}")
    new = {s: r for s, r in new.items() if s < 12}      # confirmatory seeds only
    if not new:
        raise SystemExit(f"no EXP-047 confirmatory records tagged {tag} in {args.out_dir}")

    seeds = sorted(set(new) & set(base))
    missing = sorted(set(new) - set(base))
    if missing:
        print(f"WARNING: no EXP-043 baseline for seeds {missing}; excluded from the pairing.\n")

    ft = [new[s]["success_rate"] for s in seeds]
    fr = [base[s]["success_rate"] for s in seeds]
    diffs = [a - b for a, b in zip(ft, fr)]

    print("=" * 78)
    print(f"EXP-047: fine-tuned encoder vs frozen, depth {DEPTH}, {EPISODES:,} episodes")
    print(f"encoder_lr {selected:g}, selected from the pilot on seeds "
          f"{json.loads(sel_path.read_text())['pilot_seeds']} (probe-only, disjoint)")
    print(f"n = {len(seeds)} paired seeds")
    print("=" * 78)

    # ---- Claim 3 first: it says what is being compared. ----
    counts = {r["trainable_params"] for r in new.values()}
    print(f"\nCLAIM 3 - ARCHITECTURE ACCOUNTING (descriptive)")
    print(f"  trainable parameters   {FROZEN_TRAINABLE} (frozen)  ->  "
          f"{sorted(counts)} (fine-tuned), a factor of "
          f"{FINETUNE_TRAINABLE / FROZEN_TRAINABLE:.0f}")
    print(f"  wall clock per step    {STEP_COST_RATIO}x")
    print(f"  budget-equivalent gain +{COMPUTE_EQUIVALENT} (EXP-046: 0.22 per log10 of spend)")
    if counts != {FINETUNE_TRAINABLE}:
        print(f"  !! expected every cell to report {FINETUNE_TRAINABLE}. Some cell did not "
              f"actually fine-tune.")
    print(f"  >> THIS IS A DIFFERENT ARCHITECTURE. Never a cell of the depth series.")

    # ---- Claim 1 ----
    delta = sum(diffs) / len(diffs)
    p = permutation_p(diffs)
    w = sum(1 for d in diffs if d > 0)
    lose = sum(1 for d in diffs if d < 0)
    tie = len(diffs) - w - lose

    print(f"\nCLAIM 1 (PRIMARY, PAIRED) - does fine-tuning beat the frozen encoder?")
    print(f"  fine-tuned  mean {st.mean(ft):.4f}  sd {sd(ft):.4f}")
    print(f"  frozen      mean {st.mean(fr):.4f}  sd {sd(fr):.4f}   (EXP-043, {BASELINE_MEAN})")
    print(f"  paired delta {delta:+.4f}   W-L-T {w}-{lose}-{tie}   exact p {p:.4f}")
    print(f"  bar: >= +{CLAIM1_DELTA} at p <= {ALPHA}")
    print(f"  VERDICT: {verdict_claim1(delta, p)}")
    print(f"  per-seed: " + "  ".join(f"{d:+.3f}" for d in diffs))
    if 0 < delta <= COMPUTE_EQUIVALENT:
        print(f"  NOTE: this delta is at or below what the 1.33x compute alone buys. It is not")
        print(f"        evidence that fine-tuning does anything.")

    # ---- Claim 2 ----
    probe_path = args.out_dir / "probe_confirm.json"
    print(f"\nCLAIM 2 (MECHANISM) - did the REPRESENTATION improve, or only the head's fit?")
    if not probe_path.exists():
        print(f"  {probe_path.name} not found. Run probe_encoders.py --mode confirm.")
    else:
        rows = {r["seed"]: r for r in json.loads(probe_path.read_text(encoding="utf-8"))}
        pseeds = sorted(set(rows) & set(seeds))
        std = [rows[s]["delta_headline"] for s in pseeds]
        lf = [rows[s]["delta_leakfree"] for s in pseeds]
        for name, ds, d_at in (("standard split", std, HEADLINE_DEPTH),
                               ("leak-free slice", lf, LEAKFREE_DEPTH)):
            m = sum(ds) / len(ds)
            print(f"  {name:16s} depth {d_at}: {m:+.4f}  "
                  f"W-L {sum(1 for x in ds if x > 0)}-{sum(1 for x in ds if x < 0)}  "
                  f"exact p {permutation_p(ds):.4f}")
        print(f"  ASYMMETRY (pre-registered): degradation is clean; improvement on the standard")
        print(f"  split is confounded with memorising probed states. The leak-free slice is the")
        print(f"  only one clean at BOTH stages. Where they disagree, report the WEAKER.")
        direction = ("representation improved" if sum(std) / len(std) > 0
                     else "head fitted itself to a degrading code")
        print(f"  reading: score {'up' if delta > 0 else 'down'}, standard probe "
              f"{'up' if sum(std) / len(std) > 0 else 'down'}  ->  {direction}")

    # ---- Claim 4 ----
    print(f"\nCLAIM 4 (COLLAPSE) - descriptive, NO p-value")
    stages = [deepest_stage(new[s]) for s in seeds]
    stages = [s for s in stages if s]
    if stages:
        first = st.mean(s["entropy_first_10pct"] for s in stages)
        last = st.mean(s["entropy_last_10pct"] for s in stages)
        emin = min(s["entropy_min"] for s in stages)
        solve = st.mean(s["train_solved_frac"] for s in stages)
        print(f"  deepest stage entropy  {first:.4f} -> {last:.4f}   min {emin:.2e}")
        print(f"  train solve rate       {solve:.4f}")
        print(f"  EXP-045 collapse       {EXP045['entropy_first']:.4f} -> "
              f"{EXP045['entropy_last']:.4f}   min {EXP045['entropy_min']:.2e}   "
              f"solve {EXP045['solve_frac']:.4f}")
        print(f"  a 70x larger trainable surface is a 70x larger surface to collapse; this is")
        print(f"  where that would show, along with stage 1's 2-step depth-1 cap shaping the")
        print(f"  encoder as well as the head.")
    else:
        print("  no stage_trace found.")

    zeros_ft = sum(1 for x in ft if x == 0.0)
    zeros_fr = sum(1 for x in fr if x == 0.0)
    print(f"  seeds at exactly 0.0000:  fine-tuned {zeros_ft}/{len(ft)}, "
          f"frozen {zeros_fr}/{len(fr)}")

    print(f"\nCLAIM 5 - if Claim 1 refuted: REINFORCE's gradient cannot improve the encoder")
    print(f"  faster than budget can be bought. The frozen 390-parameter result stands as")
    print(f"  published, WITH NO CAVEAT ADDED, and the next move is a different pretraining")
    print(f"  OBJECTIVE (value/heuristic, not inverse dynamics) - not more RL, not more episodes.")
    print("=" * 78)


if __name__ == "__main__":
    main()
