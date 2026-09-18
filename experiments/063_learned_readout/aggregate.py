"""EXP-063 aggregator: the pre-registered rules from the spec, applied to the records on disk.

Spec: `docs/superpowers/specs/2026-09-17-exp063-learned-readout-design.md`.

> WRITTEN BEFORE ANY EXP-063 NUMBER EXISTS. Confirm completion from COUNTS - 48 records, zero
> python processes - and NOT from the run log, which prints one success rate per line. EXP-060's
> aggregator could not claim that; EXP-061's and EXP-062's could. It is the practice to keep.

> THE GATES ARE CONDITIONS AND ARE CHECKED FIRST, FOR BOTH NEW ARMS.
>
> A MISSING READING IS A FAILURE, NEVER A PASS. `None` coerced to 0.0 would sail under a
> "must be at least X" floor while measuring nothing - the mirror of EXP-058's gate that could
> not pass. Both gates refuse to average over a missing value.
>
> A PARTIAL failure - one arm passing, the other not - is a FAILURE. EXP-059's aggregator was
> missing exactly this case and only mutation testing found it.

> THE PRIMARY'S INTERESTING OUTCOME MAY WELL BE A NULL, AND THIS FILE MUST SAY SO WHEN IT
> HAPPENS. A null at n=24 is a BOUND: "with a perfect episodic cache and a learned attention
> over it, no usable episodic signal was found". It is never "the content is unusable", and the
> spec fixed that wording before dispatch precisely so it could not be upgraded afterwards.
>
> AND A NULL WITH UNIFORM ATTENTION IS A WEAKER BOUND STILL. If `attn_entropy_norm` sits near
> 1.0 the readout never learned to SELECT, so the null is about optimisation rather than about
> information. That joint reading is pre-registered and is printed automatically.

Usage:
    .venv/bin/python -u experiments/063_learned_readout/aggregate.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

_spec = importlib.util.spec_from_file_location("exp063_run", HERE / "run.py")
_run = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run)

SEEDS = _run.SEEDS
TAG_T, TAG_U = _run.ARMS[0][1], _run.ARMS[1][1]
READOUT_T, READOUT_U = _run.ARMS[0][0], _run.ARMS[1][0]
BAR = _run.BAR                                   # 0.05
GATE1_MIN = _run.GATE_MIN_NORM_RATIO
GATE2_MIN = _run.GATE_MIN_CHOICE_FRAC

ALPHA = 0.05
N_CONTRASTS = 3                                  # Claims 1, 2 and 3. The gates are conditions.
BONFERRONI = ALPHA / N_CONTRASTS                 # 0.0167

EXP059 = REPO / "experiments/059_memory_depth5/outputs"
EXP061 = REPO / "experiments/061_noise_matched_recall/outputs"
TAG_A, TAG_M, TAG_N = _run.EXP059_A, _run.EXP059_M, _run.EXP061_N

# Context, never a control: measured on the SAME seeds and config by EXP-059/061.
EXP059_M_MINUS_A, EXP059_M_MINUS_A_P = -0.0954, 0.0056
EXP061_M_MINUS_N, EXP061_M_MINUS_N_P = +0.0210, 0.4989

EXACT_MAX_N = 20
SAMPLED_DRAWS = 200_000
SAMPLED_SEED = 20260917

T95 = {5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
       20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069}


def t95_for(n: int) -> float:
    """df-indexed, deliberately. EXP-055/056/057 all hardcode `T95_DF11 = 2.201`, the multiplier
    for n=12; reusing that at n=24 reports every interval about 6% too wide."""
    df = n - 1
    if df in T95:
        return T95[df]
    if df > 23:
        return 1.96
    raise ValueError(f"n={n} is too small to report an interval for")


def permutation_p(diffs) -> tuple[float, str]:
    """Two-sided paired permutation p, plus the NAME of the method - an exact proportion and a
    Monte Carlo estimate are not interchangeable and the write-up must say which it is."""
    n, obs = len(diffs), abs(sum(diffs))
    if n <= EXACT_MAX_N:
        hits = sum(1 for s in itertools.product((1, -1), repeat=n)
                   if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12)
        return hits / 2 ** n, f"exact over all 2**{n} = {2 ** n:,} sign flips"
    rng = random.Random(SAMPLED_SEED)
    hits = 0
    for _ in range(SAMPLED_DRAWS):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total) >= obs - 1e-12:
            hits += 1
    # Add-one: plain hits/draws can print 0.0000 and claim an exactness sampling cannot support.
    return (hits + 1) / (SAMPLED_DRAWS + 1), (
        f"sampled, {SAMPLED_DRAWS:,} draws at seed {SAMPLED_SEED} (exhaustive would be "
        f"2**{n} = {2 ** n:,}); add-one estimator, floor {1 / (SAMPLED_DRAWS + 1):.1e}")


def ci95(diffs) -> tuple[float, float]:
    n = len(diffs)
    d = st.mean(diffs)
    se = st.stdev(diffs) / (n ** 0.5) if n > 1 else 0.0
    t = t95_for(n)
    return d - t * se, d + t * se


def load(directory: Path, tag: str, expect_readout: str | None = None) -> dict:
    """Records by seed. `expect_readout` is checked when given: every arm here shares
    `arm="regionalized"` and differs ONLY in the readout, so a tag pointing at the wrong readout
    would contrast an arm against itself with nothing in the numbers looking wrong."""
    out: dict[int, dict] = {}
    for p in sorted(Path(directory).glob("*.json")):
        r = json.loads(p.read_text())
        if not isinstance(r, dict) or r.get("tag") != tag:
            continue
        if expect_readout is not None and r.get("readout") != expect_readout:
            raise SystemExit(f"{p.name}: tag {tag} carries readout {r.get('readout')!r}, "
                             f"expected {expect_readout!r}; refusing.")
        out[int(r["seed"])] = r
    return out


def paired(arm: dict, ctrl: dict, field="success_rate"):
    use = sorted(set(arm) & set(ctrl) & set(SEEDS))
    missing = [s for s in use if arm[s].get(field) is None or ctrl[s].get(field) is None]
    if missing:
        raise SystemExit(f"field {field!r} is None for seeds {missing}; cannot pair on it")
    return use, [arm[s][field] - ctrl[s][field] for s in use]


def choice_frac(record: dict) -> float | None:
    """Gate 2's reading for one cell. `None` when either part is missing OR when `train_steps`
    is zero - a zero denominator is a broken cell, not a fraction of 0.0, and coercing it would
    quietly drag an arm mean toward the floor."""
    choice, steps = record.get("attn_choice_steps"), record.get("train_steps")
    if choice is None or not steps:
        return None
    return choice / steps


def gate_reading(records: dict, reader) -> tuple[float | None, list[int], float | None]:
    """Mean over seeds, the seeds where the reading is MISSING, and the per-seed minimum.

    Missing seeds are RETURNED, not skipped: a `None` dropped from a mean or coerced to 0.0
    turns a floor gate into one that cannot fail, which is the trap this project has now hit
    twice from the other direction.
    """
    vals, missing = [], []
    for s in sorted(SEEDS):
        r = records.get(s)
        v = None if r is None else reader(r)
        if v is None:
            missing.append(s)
        else:
            vals.append(v)
    if not vals:
        return None, missing, None
    return st.mean(vals), missing, min(vals)


def check_gates(arms: dict) -> tuple[bool, list[str]]:
    """BOTH gates, on BOTH new arms. Any partial failure is a failure."""
    lines, ok = [], True
    for gate_no, label, reader, floor in (
        (1, "recall_concept_norm_ratio", lambda r: r.get("recall_concept_norm_ratio"), GATE1_MIN),
        (2, "attn_choice_steps / train_steps", choice_frac, GATE2_MIN),
    ):
        if floor is None:
            ok = False
            lines.append(f"  GATE {gate_no} ({label}): NO THRESHOLD SET - refusing to pass a "
                         "gate whose floor was never calibrated.")
            continue
        for name, records in arms.items():
            mean, missing, lo = gate_reading(records, reader)
            if missing:
                ok = False
                lines.append(f"  GATE {gate_no} arm {name}: MISSING for seeds {missing} - "
                             "FAIL. A missing reading is never a pass.")
                continue
            passed = mean >= floor
            ok = ok and passed
            lines.append(f"  GATE {gate_no} arm {name}: mean {mean:.4f}, min {lo:.4f}, "
                         f"floor {floor} -> {'PASS' if passed else 'FAIL'}")
    return ok, lines


def _sig(p: float, alpha: float) -> str:
    return f"p {p:.4f} {'<=' if p <= alpha else '>'} {alpha:g}"


def verdict(diffs, p: float, alpha: float, bar: float) -> tuple[str, str]:
    """The pre-registered three-way reading. Directional: the arm is predicted ABOVE the ctrl."""
    d = st.mean(diffs)
    if d >= bar and p <= alpha:
        return "CONFIRMED", f"+{d:.4f} clears +{bar} and {_sig(p, alpha)}"
    if d <= -bar and p <= alpha:
        return "REFUTED", f"{d:.4f} clears -{bar} in the WRONG direction and {_sig(p, alpha)}"
    return "NULL", (f"{d:+.4f} against a +/-{bar} bar, {_sig(p, alpha)}. "
                    "A NULL IS A BOUND, NOT EVIDENCE OF ABSENCE.")


def report(label: str, arm: dict, ctrl: dict, alpha: float, field="success_rate"):
    seeds, diffs = paired(arm, ctrl, field)
    p, method = permutation_p(diffs)
    lo, hi = ci95(diffs)
    v, why = verdict(diffs, p, alpha, BAR)
    print(f"\n{label}   n={len(seeds)}")
    print(f"  mean diff {st.mean(diffs):+.4f}   approx 95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  {method}")
    print(f"  VERDICT: {v} - {why}")
    return v, st.mean(diffs), p


def mechanism(arms: dict) -> dict:
    print("\n--- mechanism readings (NEVER gates) ---")
    out = {}
    for name, records in arms.items():
        ent, _, _ = gate_reading(records, lambda r: r.get("attn_entropy_norm"))
        rec, _, _ = gate_reading(records, lambda r: r.get("attn_recency_mass"))
        out[name] = ent
        e = "n/a" if ent is None else f"{ent:.4f}"
        rr = "n/a" if rec is None else f"{rec:.4f}"
        print(f"  arm {name}: attn_entropy_norm {e} (1.0 = uniform), attn_recency_mass {rr}")
    return out


def main() -> None:
    T = load(HERE / "outputs", TAG_T, READOUT_T)
    U = load(HERE / "outputs", TAG_U, READOUT_U)
    A = load(EXP059, TAG_A, "memory_amnesic")
    M = load(EXP059, TAG_M, "memory")
    N = load(EXP061, TAG_N, "memory_noise")

    print("EXP-063: can ANY readout use the stored content?")
    print(f"  cells on disk: T {len(T)}, U {len(U)} (new)   "
          f"A {len(A)}, M {len(M)}, N {len(N)} (reused)")
    for name, recs in (("T", T), ("U", U)):
        short = sorted(set(SEEDS) - set(recs))
        if short:
            print(f"  INCOMPLETE: arm {name} is missing seeds {short}")

    print("\n--- validity gates (conditions, checked FIRST) ---")
    ok, lines = check_gates({"T": T, "U": U})
    print("\n".join(lines))
    if not ok:
        print("\nGATE FAILED. The claims below are VOID and must not be read. Reporting them "
              "anyway is the outcome-dependent reading the pre-registration exists to prevent.")
        return

    v1, d1, _ = report("CLAIM 1, PRIMARY - T minus U (capacity-matched)", T, U, ALPHA)
    v2, d2, _ = report(f"CLAIM 2 - T minus A (amnesic, the best memory-width arm)", T, A,
                       BONFERRONI)
    v3, d3, _ = report(f"CLAIM 3 - T minus M (the raw attractor read; CAPACITY-CONFOUNDED)",
                       T, M, BONFERRONI)

    ents = mechanism({"T": T, "U": U})

    print("\n--- context, never controls (same seeds, same config) ---")
    print(f"  EXP-059  M - A = {EXP059_M_MINUS_A:+.4f}, p {EXP059_M_MINUS_A_P}")
    print(f"  EXP-061  M - N = {EXP061_M_MINUS_N:+.4f}, p {EXP061_M_MINUS_N_P}")

    print("\n--- pre-registered joint reading ---")
    if v1 == "CONFIRMED" and v2 == "CONFIRMED":
        print("  The episodic content IS usable and the failure was the READOUT. Phase 3's")
        print("  memory line REOPENS: EXP-059/061's negative result is about the raw")
        print("  hippocampal read, not about episodic memory on this task.")
    elif v1 == "CONFIRMED":
        print("  The content is usable but not enough to beat a memory-free nonlinear expansion")
        print("  of the current state. Memory stays CLOSED NEGATIVE, with a sharper sentence:")
        print("  the signal is real and too small to matter here.")
    elif v1 == "REFUTED":
        print("  The added capacity HURT. Report as such; do not narrate it as evidence about")
        print("  the content, which this design cannot separate from the capacity.")
    else:
        e = ents.get("T")
        print("  NULL: a BOUND, not evidence of absence. With a perfect episodic cache and a")
        print("  learned attention over it, no usable episodic signal was found at n=24.")
        if e is not None and e >= 0.95:
            print(f"  WEAKER STILL: attn_entropy_norm {e:.4f} is near-uniform, so the readout")
            print("  never learned to SELECT. This null is about OPTIMISATION as much as about")
            print("  information, and must be written up that way.")
        elif e is not None:
            print(f"  attn_entropy_norm {e:.4f}: the attention DID concentrate, so the null is")
            print("  about the information rather than about a failure to train.")


if __name__ == "__main__":
    main()
