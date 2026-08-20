# experiments/047_encoder_finetuning/select_lr.py
"""Choose EXP-047's `encoder_lr` from the pilot. Mechanical, pre-registered, PROBE-ONLY.

The rule is fixed in section 5.2 of
docs/superpowers/specs/2026-08-20-exp047-encoder-finetuning-design.md, committed at 69bf1dc
before any pilot number existed. This file only executes it.

  1. GATE. A rate passes only if, on BOTH pilot seeds, the fine-tuned encoder's depth-4 probe
     top-1 is no more than 0.02 below that same seed's own starting encoder. 0.02 is under 6% of
     the +0.3396 pretraining bought at depth 4 (EXP-039), so the gate says: fine-tuning may not
     give back more than a sixteenth of what pretraining earned.

  2. CHOICE. Among the rates that pass, the highest mean depth-4 probe top-1 across the two
     seeds. Tie-break to the LARGER rate.

  3. STOP. If no rate passes, this writes `selected_lr: null` and `run.py --mode confirm`
     REFUSES TO START. "REINFORCE's gradient damages the pretrained representation at every rate
     tried" is a real result and is written up as one. It is NOT a reason to widen the grid;
     that would need a new pre-registration.

> [!important] WHY THIS FILE NEVER OPENS A SUCCESS RATE.
> EXP-039 section 6a chose its pretraining lr by the pretraining objective and explicitly
> refused to choose it by the probe, because the probe was its outcome metric - and recorded
> that the probe would have picked the other rate. EXP-047 inherits the trap one level up. The
> defences are stacked, because either alone is leaky:
>
>   - This file reads ONLY `probe_pilot.json`. It cannot see a success rate; that is enforced by
>     construction, not by intention.
>   - The pilot runs on seeds 12-13, DISJOINT from the confirmatory seeds 0-11. So even though
>     the probe is Claim 2's own metric, no seed that contributes to Claim 2 was used to select.
>
> Together: nothing that decides a claim was used to make this choice.

Run (repo root):
    .venv/bin/python -u experiments/047_encoder_finetuning/select_lr.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

HEADLINE_DEPTH = 4
GATE_MAX_DROP = 0.02          # spec 5.2 step 1
PRETRAIN_GAIN_D4 = 0.3396     # EXP-039 Claim 1, what the gate is a fraction of
EXPECTED_PILOT_SEEDS = (12, 13)


def rows_by_rate(rows):
    out: dict[float, list] = {}
    for r in rows:
        out.setdefault(float(r["encoder_lr"]), []).append(r)
    return out


def evaluate_rate(rate: float, rows: list) -> dict:
    """Apply the gate to one rate. Returns its verdict row, with every input shown."""
    per_seed = []
    for r in sorted(rows, key=lambda r: r["seed"]):
        before = r["before"]["by_depth"][str(HEADLINE_DEPTH)]["top1"]
        after = r["after"]["by_depth"][str(HEADLINE_DEPTH)]["top1"]
        per_seed.append({"seed": r["seed"], "before": before, "after": after,
                         "delta": after - before, "passes": (after - before) >= -GATE_MAX_DROP})
    return {
        "encoder_lr": rate,
        "per_seed": per_seed,
        "mean_after": sum(s["after"] for s in per_seed) / len(per_seed),
        "mean_delta": sum(s["delta"] for s in per_seed) / len(per_seed),
        "worst_drop": min(s["delta"] for s in per_seed),
        # ALL seeds, not the mean: a rate that preserves the representation on one seed and
        # destroys it on the other is not a rate this experiment should carry to 12 seeds.
        "gate_passed": all(s["passes"] for s in per_seed),
    }


def select(verdicts: list[dict]) -> tuple[float | None, str]:
    passing = [v for v in verdicts if v["gate_passed"]]
    if not passing:
        worst = min(v["worst_drop"] for v in verdicts)
        return None, (
            f"no rate passed the gate (worst depth-{HEADLINE_DEPTH} drop {worst:+.4f}, gate is "
            f"-{GATE_MAX_DROP}). REINFORCE's gradient damages the pretrained representation at "
            f"every rate in the pre-registered grid."
        )
    # Highest mean depth-4 top-1; tie-break to the LARGER rate (spec 5.2 step 2).
    best = max(passing, key=lambda v: (v["mean_after"], v["encoder_lr"]))
    return best["encoder_lr"], (
        f"highest mean depth-{HEADLINE_DEPTH} probe top-1 ({best['mean_after']:.4f}) among "
        f"{len(passing)} rate(s) passing the gate"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=HERE / "outputs")
    args = ap.parse_args()

    src = args.out_dir / "probe_pilot.json"
    if not src.exists():
        raise SystemExit(
            f"{src} not found. Run the pilot, then probe_encoders.py --mode pilot, then this."
        )
    rows = json.loads(src.read_text(encoding="utf-8"))

    seeds = sorted({r["seed"] for r in rows})
    if any(s < 12 for s in seeds):
        raise SystemExit(
            f"pilot probe rows include confirmatory seeds {[s for s in seeds if s < 12]}. "
            "Selecting on a seed that also carries a claim is the exact contamination the "
            "design forbids (spec 5.1). Refusing to select."
        )

    by_rate = rows_by_rate(rows)
    verdicts = [evaluate_rate(rate, by_rate[rate]) for rate in sorted(by_rate, reverse=True)]
    for v in verdicts:
        if len(v["per_seed"]) != len(seeds):
            raise SystemExit(
                f"rate {v['encoder_lr']:g} has {len(v['per_seed'])} seed(s), expected {len(seeds)}. "
                "An incomplete pilot cannot be selected from; re-run the missing cells."
            )

    chosen, reason = select(verdicts)

    print(f"EXP-047 rate selection. PROBE-ONLY, seeds {seeds} (disjoint from claims).")
    print(f"gate: depth-{HEADLINE_DEPTH} top-1 may not fall more than {GATE_MAX_DROP} "
          f"({GATE_MAX_DROP / PRETRAIN_GAIN_D4:.1%} of pretraining's +{PRETRAIN_GAIN_D4})\n")
    print(f"{'lr':>8}  {'mean d4':>8}  {'mean delta':>11}  {'worst':>8}  gate   per-seed")
    for v in verdicts:
        detail = "  ".join(f"s{s['seed']}:{s['before']:.3f}->{s['after']:.3f}"
                           for s in v["per_seed"])
        print(f"{v['encoder_lr']:>8.0e}  {v['mean_after']:>8.4f}  {v['mean_delta']:>+11.4f}  "
              f"{v['worst_drop']:>+8.4f}  {'PASS' if v['gate_passed'] else 'FAIL':<5}  {detail}")

    dest = args.out_dir / "selected_lr.json"
    dest.write_text(json.dumps({
        "selected_lr": chosen, "reason": reason, "gate_max_drop": GATE_MAX_DROP,
        "headline_depth": HEADLINE_DEPTH, "pilot_seeds": seeds, "verdicts": verdicts,
        "spec": "docs/superpowers/specs/2026-08-20-exp047-encoder-finetuning-design.md",
    }, indent=1), encoding="utf-8")

    if chosen is None:
        print(f"\nHALT: {reason}")
        print("Per spec 5.2 step 3 this is a RESULT. Write it up. Do not widen the grid without")
        print("a new pre-registration. The confirmatory arm will refuse to start.")
        print(f"\nwritten to {dest}")
        return

    print(f"\nSELECTED encoder_lr = {chosen:g}")
    print(f"  {reason}")
    print(f"\nwritten to {dest}. The confirmatory arm reads it; do not pass --encoder-lr by hand.")


if __name__ == "__main__":
    main()
