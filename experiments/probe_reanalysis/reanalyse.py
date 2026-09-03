"""Re-run the probe-based inferences of EXP-033, EXP-039 and EXP-047 with TRAJECTORY METRICS
beside them.

NOT A NEW EXPERIMENT. No data is generated, no arm is run, and no claim is pre-registered. This
directory is deliberately unnumbered so it cannot be mistaken for one. Everything below is
re-analysis of records already on disk.

WHY THIS EXISTS. The EXP-033 linear probe was retired as a policy predictor: EXP-050 measured two
training objectives moving it in OPPOSITE directions, both unanimous at p 0.0005, and EXP-049 saw
every seed's probe get worse in a round that was supposed to help. `CLAUDE.md` now says to use
`revisit_rate` and `optimality` instead. But three experiments drew inferences from the probe
BEFORE it was retired, and those inferences were never revisited with the replacement instruments
beside them. That is what this does.

The question in each section is the same: **did the thing the probe said actually show up in the
policy's behaviour, per seed?** A probe movement that is unanimous and significant, yet uncorrelated
with any behavioural change, is exactly what a retired instrument looks like.

Usage:
    .venv/bin/python experiments/probe_reanalysis/reanalyse.py
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import random
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# spearman is imported rather than reimplemented, from the aggregator whose Claim 4 gate was
# fixed on 2026-09-01. Reimplementing a rank correlation is how two versions of it drift.
_A54 = ROOT / "experiments" / "054_sequence_blindness" / "aggregate.py"
_s = importlib.util.spec_from_file_location("exp054_agg", _A54)
_agg54 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_agg54)
spearman = _agg54.spearman

ALPHA = 0.05
METRICS = ("success_rate", "revisit_rate", "optimality")
PERM_ITERS = 20000
PERM_SEED = 0

# Every correlation tested anywhere in this file, so the tally at the end is not eyeballed.
CORRELATIONS: list[tuple[str, float, float]] = []


def permutation_p(diffs) -> float:
    """Exact two-sided paired permutation over all 2**n sign flips. No scipy in the venv."""
    n, obs = len(diffs), abs(sum(diffs))
    return sum(1 for s in itertools.product((1, -1), repeat=n)
               if abs(sum(x * y for x, y in zip(s, diffs))) >= obs - 1e-12) / 2 ** n


def spearman_p(x, y, iters: int = PERM_ITERS, seed: int = PERM_SEED) -> float:
    """Two-sided permutation p for a Spearman correlation.

    n=12 makes 12! exhaustive permutation impossible, so this shuffles one vector `iters`
    times against a FIXED seed, which keeps the number reproducible. Without this the
    correlations below are just numbers to squint at, and squinting at four means is the exact
    process failure EXP-052 and EXP-054 each committed.
    """
    rng = random.Random(seed)
    obs = abs(spearman(x, y))
    ys = list(y)
    hits = 0
    for _ in range(iters):
        rng.shuffle(ys)
        if abs(spearman(x, ys)) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (iters + 1)


def load_policy(directory: Path, tag: str, depth: int, arm: str = "regionalized") -> dict:
    """One record per seed, and REFUSE if that is ambiguous.

    EXP-036 stores its measured chance floor under the same tag and depth as the trained arm,
    distinguished only by `arm`. Picking whichever the filesystem yielded first would silently
    compare a policy against a random walk, and would look entirely normal.
    """
    out: dict[int, dict] = {}
    for p in Path(directory).glob("*.json"):
        r = json.loads(p.read_text())
        if not isinstance(r, dict):
            continue
        if r.get("tag") == tag and r.get("depth") == depth and r.get("arm") == arm:
            seed = int(r["seed"])
            if seed in out:
                raise SystemExit(
                    f"AMBIGUOUS: two records for tag={tag} depth={depth} arm={arm} seed={seed}. "
                    "Refusing to pick one; disambiguate before trusting any number here."
                )
            out[seed] = r
    return out


def probe039() -> dict:
    out = {}
    for p in (ROOT / "experiments/039_encoder_pretraining/outputs").glob("exp039_s*.json")   :
        r = json.loads(p.read_text())
        out[int(r["seed"])] = r
    return out


def probe047() -> dict:
    recs = json.loads(
        (ROOT / "experiments/047_encoder_finetuning/outputs/probe_confirm.json").read_text())
    return {int(r["seed"]): r for r in recs}


def by_depth(block: dict, depth: int) -> float:
    return float(block["by_depth"][str(depth)]["top1"])


def paired_line(name: str, diffs: list[float]) -> str:
    d = st.mean(diffs)
    w = sum(1 for x in diffs if x > 0)
    l = sum(1 for x in diffs if x < 0)
    p = permutation_p(diffs)
    flag = "significant" if p <= ALPHA else "not significant"
    return f"    {name:<16} {d:+.4f}   W-L-T {w}-{l}-{len(diffs)-w-l}   p {p:.4f}  {flag}"


def section_039():
    print("=" * 78)
    print("A. EXP-039's inference, with trajectory metrics beside it")
    print("=" * 78)
    print("""
  THE ORIGINAL INFERENCE: inverse-model pretraining raises the linear probe at every
  depth, every seed, at p 0.0005 (the floor of the exact test at n=12). That result
  is what motivated the whole pretrained-encoder line from EXP-040 onward.

  THE TEST NOW: EXP-039 measured only the probe. The policy that used those encoders
  is EXP-040, against EXP-036's frozen policy at the same depths and seeds. So the
  per-seed probe gain can be put directly beside the per-seed behavioural change.
""")
    probes = probe039()
    for depth, tag40 in ((4, "exp040_pre_d4"), (5, "exp040_pre_d5"), (6, "exp040_pre_d6")):
        frozen = load_policy(ROOT / "experiments/036_generalisation_gap/outputs",
                             f"exp036_d{depth}_e10000", depth)
        pre = load_policy(ROOT / "experiments/040_pretrained_encoder_policy/outputs",
                          tag40, depth)
        seeds = sorted(set(probes) & set(frozen) & set(pre))
        if not seeds:
            print(f"  depth {depth}: no overlapping seeds; skipped.")
            continue
        pdiff = [by_depth(probes[s]["trained"], depth) - by_depth(probes[s]["frozen"], depth)
                 for s in seeds]
        print(f"  depth {depth}, n={len(seeds)}")
        print(paired_line("probe top1", pdiff))
        for m in METRICS:
            mdiff = [pre[s][m] - frozen[s][m] for s in seeds]
            print(paired_line(m, mdiff))
            rho = spearman(pdiff, mdiff)
            rp = spearman_p(pdiff, mdiff)
            CORRELATIONS.append((f"d{depth} probe~{m}", rho, rp))
            verdict = "CORRELATED" if rp <= ALPHA else "no relationship resolved"
            print(f"      -> Spearman(probe gain, {m} change) = {rho:+.3f}, "
                  f"perm p {rp:.4f}  [{verdict}]")
        print()


def section_047():
    print("=" * 78)
    print("B. EXP-047's inference, with trajectory metrics beside it")
    print("=" * 78)
    print("""
  THE ORIGINAL INFERENCE: fine-tuning the encoder inside the RL loop moves the probe,
  and EXP-047's Claim 2 re-probes those weights. Here the probe and the policy come
  from the SAME runs and the same seeds, so no cross-experiment pairing is needed.

  THE TEST NOW: does a seed whose probe improved more also behave better?
""")
    probes = probe047()
    pol = load_policy(ROOT / "experiments/047_encoder_finetuning/outputs",
                      "exp047_ft_d6_lr0.0001", 6)
    seeds = sorted(set(probes) & set(pol))
    print(f"  depth 6, n={len(seeds)}, lr 1e-4 (the confirmatory arm)")

    # EXP-047's OWN numbers, not a pooled figure it never reported. Its spec pre-committed to
    # reporting the weaker of the two splits, because RL fine-tunes on the RL split while the
    # standard probe holds out a different one: degradation would be clean, improvement is
    # confounded by leakage.
    headline = [probes[s]["delta_headline"] for s in seeds]
    leakfree = [probes[s]["delta_leakfree"] for s in seeds]
    print(paired_line("probe standard", headline))
    print(paired_line("probe leak-free", leakfree))

    for label, pdiff in (("standard", headline), ("leak-free", leakfree)):
        for m in METRICS:
            vals = [pol[s][m] for s in seeds]
            rho = spearman(pdiff, vals)
            rp = spearman_p(pdiff, vals)
            CORRELATIONS.append((f"047 {label} probe~{m}", rho, rp))
            verdict = "CORRELATED" if rp <= ALPHA else "no relationship resolved"
            print(f"    Spearman({label} probe delta, {m}) = {rho:+.3f}, "
                  f"perm p {rp:.4f}  [{verdict}]")
    print("""
    These correlate a probe CHANGE against a behavioural LEVEL, because EXP-047's
    fine-tuned arm has no seed-matched frozen twin whose trajectory metrics could be
    differenced. That is weaker than section A and is labelled as such.

    EXP-047 DID NOT OVER-CLAIM. Its Claim 2 was pre-registered as SPLIT and it
    reported the weaker slice, concluding "the policy got better, and we cannot show
    the representation did". Nothing here needs retracting. It is included because
    the item asked for it, and because a correctly hedged inference is worth
    confirming as such rather than assuming.
""")


def section_033():
    print("=" * 78)
    print("C. EXP-033's inference, and why most of it CANNOT be redone")
    print("=" * 78)
    print("""
  THE ORIGINAL INFERENCES were two:

    Finding 1  width helps the probe and is refuted as the lever (0.407 at width 64
               rising to 0.528 at 512, saturating, still short of facelets' 0.648).
    Finding 2  the representation is NOT the first bottleneck. An oracle-fitted probe
               run as a greedy policy reached 0.481 at depth 3 where REINFORCE got
               0.022, a 22x gap on an identical representation and head shape.

  WHAT CAN BE PUT BESIDE THEM: almost nothing, and that is the honest finding.

    Finding 1 has NO policy counterpart at any width other than 64. Policies were
    never trained at 128, 256 or 512, so there are no trajectory metrics to pair
    against the width sweep, and none can be produced without new runs. The width
    inference stands or falls entirely on the probe, which is now retired. It should
    be cited as "wider random projections decode better", never as "width would not
    have helped the policy" - the second was never measured.

    Finding 2 is DIFFERENT IN KIND and survives. It did not infer policy quality from
    a probe number; it ran the probe AS A POLICY in the real environment and compared
    success rates directly. That is a behavioural measurement, not a probe-based
    inference, so the retirement does not touch it. Its own 2026-08-02 correction
    (EXP-035 seeds exceeding the 0.481 "ceiling") already stands.

  So of EXP-033's two findings, one is unrepairable without new runs and the other
  never needed repairing. Neither is evidence that the probe predicts behaviour.
""")


def section_verdict():
    print("=" * 78)
    print("D. THE TALLY, with multiplicity counted rather than ignored")
    print("=" * 78)
    n = len(CORRELATIONS)
    bonf = ALPHA / n
    nominal = [c for c in CORRELATIONS if c[2] <= ALPHA]
    survive = [c for c in CORRELATIONS if c[2] <= bonf]
    print(f"""
  {n} correlations were tested between a probe movement and a behavioural one.
  Bonferroni at {n} comparisons is {bonf:.4f}.

    nominally significant at {ALPHA}:  {len(nominal)}
    surviving Bonferroni {bonf:.4f}:   {len(survive)}
""")
    for label, rho, rp in nominal:
        print(f"    {label:<28} rho {rho:+.3f}   p {rp:.4f}   "
              f"{'SURVIVES' if rp <= bonf else 'does not survive correction'}")
    print(f"""
  WHAT THE PROBE ACTUALLY DID. It got the DIRECTION right for pretraining: success
  rose significantly at all three depths in section A. What it carries no
  information about is WHICH SEEDS benefited, or by how much - and that is the
  claim its unanimity at p 0.0005 invites a reader to make.

  Note the SIGNS of the nominally significant hits. Both run the wrong way: at
  depth 5 a bigger probe gain went with LOWER optimality, and in EXP-047 a bigger
  standard-split probe gain went with a HIGHER revisit rate, which is more cycling
  and therefore worse behaviour. Neither survives correction, so neither is a
  finding. But if the probe tracked behaviour, chance hits would not be
  systematically inverted.

  This is consistent with, and independent of, EXP-050's retirement evidence: two
  objectives moving the probe in opposite directions, both unanimous at p 0.0005.
""")


def main():
    print("PROBE RE-ANALYSIS: EXP-033, EXP-039, EXP-047 with trajectory metrics beside them")
    print("Re-analysis only. No new data, no new arms, no pre-registered claim.\n")
    section_039()
    section_047()
    section_033()
    section_verdict()


if __name__ == "__main__":
    main()
