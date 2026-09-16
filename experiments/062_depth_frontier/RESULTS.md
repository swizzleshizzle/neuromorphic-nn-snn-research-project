# EXP-062 results - the budget law holds out of sample, and Stage 4 is priced out

> **COMPLETE.** 24 cells, depths 8 and 9 x 12 seeds. **Validity gate PASSED in both arms.**
>
> **HEADLINE: the budget law HOLDS OUT OF SAMPLE.** Depth 8 measured **0.0783**, 95% interval
> **[0.0523, 0.1044]**, which **contains** the law's out-of-sample prediction of **0.0588**. The
> law was fitted at depths 3-7 and has now been tested a depth beyond its range.
>
> **So the Stage-4 pricing is evidence-backed rather than extrapolated.** `road-to-a-solved-cube`
> calls depth-11 random scrambles "the actual deliverable" and "genuinely achievable... nothing
> about it requires new science". **It is not achievable at this budget, and Phase 3 should report
> that plainly rather than quietly miss it.**
>
> **The frontier is depth 8.** Depth 9 measures **0.0163**, below the pre-registered 0.02 bar.

**Pre-registration:** `docs/superpowers/specs/2026-09-15-exp062-depth-frontier-design.md`.
**The aggregator was written before any number was seen** - completion was confirmed from counts
alone (24 records, 24 checkpoints, zero processes) **without reading the run log**, which prints one
success rate per line.

## Provenance - and this is NOT a single continuous run

| | |
|---|---|
| Depth 8 | 2026-09-15 01:03 to ~06:47 UTC, **6 workers** |
| Depth 9 | 2026-09-16 00:24 to 05:43 UTC, **12 workers** |
| Between them | **a Windows Update reboot at 07:31 UTC killed the first attempt** with depth 8 complete and depth 9 not started |

**Same commit (`d05b855`), same code, same seeds, same config - so the arms are comparable and the
contrast is unaffected.** But they were produced ~15 h apart at different worker counts, and that
belongs in the record rather than being implied away. Worker count changes wall-clock cost, not
arithmetic: `torch.set_num_threads(1)` and per-cell seeding make a cell's result independent of how
many siblings it runs beside.

**The reboot cost ~4 CPU-hours** because depth 8 had already completed. EXP-059's first attempt lost
**all ~19** to the same failure because no cell had finished. **Per-cell durability is what sets the
bill.**

```bash
powershell -File C:\Users\mlgbr\launch062_wt.ps1 -Phase rl -Workers 6  -SkipExisting   # depth 8
powershell -File C:\Users\mlgbr\launch062_wt.ps1 -Phase rl -Workers 12 -SkipExisting   # depth 9
.venv/bin/python -u experiments/062_depth_frontier/aggregate.py
```

## Claim 3, the validity gate - PASSED in both arms

The critic is the recipe component that makes depth 7 work at all, so these numbers are only
readable if it is present and state-dependent. Ratio of `V`'s within-episode RMS to the returns':

| stage depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| depth-8 arm | 1.931 | 1.468 | 0.956 | 0.818 | 0.660 | 0.526 | 0.443 | **0.426** | - |
| depth-9 arm | 1.877 | 1.475 | 0.983 | 0.808 | 0.660 | 0.518 | 0.442 | 0.424 | **0.394** |

**Worst ratio 0.394, nearly eight times the 0.05 floor.**

> [!note] **My pre-registered extrapolation of this gate was too pessimistic, and saying so is the
> point of writing it down.** The spec predicted "~0.25 at depth 8 and ~0.19 at depth 9" from
> EXP-060's geometric decline. Measured: **0.426 and 0.394**. The decline flattens rather than
> continuing geometrically. **The gate passed by more than promised** - which is the harmless
> direction, but the extrapolation was still wrong.

## Claim 1, PRIMARY - the law HOLDS out of sample

**A one-sample containment test, not a paired contrast** - there is nothing to pair depth 8 against.

| | |
|---|---|
| depth-8 mean | **0.0783** (sd 0.0410) |
| 95% interval, t df=11 | **[0.0523, 0.1044]** |
| law's prediction | **0.0588** |
| verdict | **CONTAINED -> LAW SUPPORTED OUT OF SAMPLE** |
| seeds above the measured zero floor | **12 of 12** |

Per-seed: 0.115, 0.085, 0.015, 0.110, 0.110, 0.060, 0.035, 0.135, 0.130, 0.055, 0.055, 0.035.

**Every seed solves something at depth 8**, against a chance floor of **exactly 0.0000** measured
across 12 seeds. So depth 8 genuinely works, just weakly.

The point estimate sits slightly **above** the prediction, so the law is marginally pessimistic -
but the interval contains it and the pre-registered reading is containment.

## Claim 2 - the frontier is depth 8

**Depth 9 measures 0.0163**, below the pre-registered 0.02 bar, so **AT THE FLOOR** by the spec.

Per-seed: 0.000, 0.000, 0.000, 0.080, 0.015, 0.005, 0.020, 0.025, 0.045, 0.000, 0.000, 0.005.

> [!warning] **"At the floor" is the pre-registered verdict, NOT "at zero", and the difference is
> real.** **7 of 12 seeds score above a chance floor of exactly 0.0000.** Depth 9 is not dead - it
> is below the bar that makes a depth count as working, and far below anything usable. Both
> statements are true and the write-up should not collapse them.

**This was never a test of the law.** Success is bounded below at zero, so the law's **-0.0828**
prediction can only manifest as "about zero"; the design cannot discriminate -0.08 from -0.5.
Reading the zero as *confirmation* would be reading a bound as evidence.

**Quality collapses faster than success does.** `optimality` falls **0.6853 -> 0.3720** from depth 8
to 9 while `mean_steps` drops 11.75 -> 8.26: the depth-9 policies are not merely failing more often,
they are failing differently - giving up earlier rather than searching badly.

## The frontier, and what Stage 4 actually costs

| depth | success | source |
|---|---|---|
| 7 | **0.2004** | EXP-053 arm B |
| 8 | **0.0783** | this experiment |
| 9 | **0.0163** | this experiment |
| chance floor at 7, 8, 9 | **exactly 0.0000** | measured, 12 seeds, random policy |

> [!danger] **THE MOST IMPORTANT NUMBER IN THIS FILE IS THE ONE THAT MUST NOT BE USED.**
> The two measured steps are **not equal**: depth 7->8 costs **0.1221** and depth 8->9 costs
> **0.0620**. It is tempting to average them into a cheaper exchange rate - and that would be
> **wrong**, because ==success is bounded below at zero, so an arm approaching the floor MUST show
> a smaller absolute decline==. The 8->9 step is **compressed by censoring**, not genuinely cheaper.
>
> | basis | per depth | budget/depth | depth-11 at depth-7 parity |
> |---|---|---|---|
> | week-20 fitted | 0.1416 | 4.40x | 1,796 h/cell = **74.8 days/seed** |
> | **measured 7->8 only (the clean step)** | **0.1221** | **3.59x** | **795 h/cell = 33.1 days/seed** |
> | naive 7->9 average - **CENSORED, DO NOT USE** | 0.0920 | 2.62x | 226 h/cell = 9.4 days/seed |
>
> **The honest figure is ~33 days per seed**, from the one step with both endpoints clear of the
> floor. The law was mildly pessimistic; Stage 4 got cheaper by a factor of two and **remains
> entirely out of reach** - about 18 days of wall clock for a 12-seed arm at 6 workers.

## What this changes

1. **Stage 4 is priced out, on evidence.** `road-to-a-solved-cube`'s "genuinely achievable...
   nothing about it requires new science" was written before the budget law existed. Nothing new is
   needed *scientifically*; it is arithmetic that stops it. **Phase 3 should state the deliverable
   is unreachable at this budget rather than quietly missing it.**
2. **The budget law survives a genuine out-of-sample test**, one depth past its fitted range. That
   makes every earlier depth projection in this project more trustworthy, not just this one.
3. **The frontier is depth 8 at 0.0783.** That is the honest maximum for the checkpoint.
4. **A ceiling on extrapolation, for free:** any future depth projection using data near the zero
   floor will under-price. Use steps whose endpoints are both well clear of it.

## What is NOT claimed

- **Not that depth 9 is at zero.** 7 of 12 seeds beat a floor of exactly 0.0000; it is below the
  0.02 bar, which is a different statement.
- **Not that depth 9 tests the law.** Censoring makes that impossible by construction.
- **Not that 33 days/seed is precise.** It rests on a single clean step between two depths at n=12,
  and the interval on depth 8 alone spans 0.052 to 0.104.
- **Not that a better recipe is impossible.** Everything here prices the *current* recipe. A method
  that changes the exchange rate rather than paying it is out of scope and untested.
- **Nothing about depths 10 or 11 directly** - both are projections, and the projection is exactly
  what this experiment was built to check one step of.
