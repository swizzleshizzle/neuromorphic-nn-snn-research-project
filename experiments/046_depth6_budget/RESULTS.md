# EXP-046 Results - the depth series is a BUDGET series. 4.4x buys about one depth.

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract and
> `aggregate.py` were both committed at `84fd7aa`, **before the run was dispatched**. No threshold
> was edited.
>
> **Provenance:** 12 records + 12 head checkpoints. Depth 6, **44,000 episodes**, curriculum
> `(1..6)` uniform, EXP-040 pretrained encoders, `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`,
> `heldout_cap=200`, seeds 0-11. Dispatched 2026-08-18 01:05 UTC on the laptop `SwizzlesDuo`,
> `--workers 12`, finished **16:37 UTC, 15.5 h**, zero tracebacks. Paired baseline is EXP-043's
> `exp043_capped_d6` cell, not re-run.
>
> **The dispatching ssh reported exit 1 and the run was fine.** `*>&1 | Tee-Object` puts torch's
> `UserWarning` into the output stream and the PowerShell wrapper exits non-zero on it. The driver
> printed `12/12` and its normal completion line. A client-side exit code says nothing about the
> job - the same lesson the playbook already records for 124 and 255.

## Headline

**Depth 6 responds to budget exactly as depth 7 did. Claim 1 CONFIRMED, 12-0-0.**

| | mean | sd | zeros |
|---|---|---|---|
| EXP-043 depth 6 @ 10,000 | 0.1800 | 0.0985 | 1 / 12 |
| **EXP-046 depth 6 @ 44,000** | **0.3225** | **0.0373** | **0 / 12** |

Paired delta **+0.1425**, **W-L-T 12-0-0**, exact **p 0.0005**. **Every seed improved.** Depth 7
gained +0.1350 from the same multiplier, so depth 6 recovers **106%** of it - the same effect size,
at a depth that already worked.

> [!important] THE DEPTH SERIES IS A BUDGET SERIES
> Every published depth in this project was measured at a fixed 10,000 episodes. Two depths have
> now been given 4.4x and both moved by about +0.14. **"Depth N stopped working" has always meant
> "depth N stopped working at 10,000 episodes."**

## The regularity: 4.4x the budget buys about one depth

| | @ 10,000 | | @ 44,000 | difference |
|---|---|---|---|---|
| depth 5 | **0.3412** | depth 6 | **0.3225** | -0.0187 |
| depth 6 | **0.1800** | depth 7 | **0.1971** | +0.0171 |

Both differences are inside noise. **A depth at 4.4x the budget scores like the depth above it at
1x.** Two independent pairs, in opposite directions, is weak evidence for a clean exponent - but it
is a concrete, falsifiable prediction: **depth 8 at ~194,000 episodes should land near 0.18.**

## Claim 1 (PRIMARY) - does 4.4x help depth 6? CONFIRMED

Pre-registered: **>= +0.05** at **p <= 0.05**.

| condition | value | verdict |
|---|---|---|
| paired delta | **+0.1425** | PASS |
| exact p (2^12 flips) | **0.0005** | PASS |
| W-L-T | **12-0-0** | - |

Per-seed deltas: `+0.155 +0.085 +0.320 +0.130 +0.190 +0.135 +0.255 +0.065 +0.135 +0.050 +0.160 +0.030`

The two seeds the depth-1 cap had left behind at depth 6 are the biggest winners: **seed 2
(+0.320)** and **seed 6 (+0.255)**, which was EXP-043's only depth-6 failure and EXP-044's only
depth-7 zero. Budget rescued the seeds that looked seed-specific.

## Claim 2 - the escalation. TRIGGERED, RUN, AND THERE IS NO KNEE

The pre-registered midpoint ran on 2026-08-19/20 (dispatched 23:53 UTC, <= 9.8 h, 12 records,
zero tracebacks; the dispatching ssh again reported **exit 1** on a run that finished cleanly).

| budget | mean | sd | se | paired vs 10k | W-L-T | exact p |
|---|---|---|---|---|---|---|
| 10,000 | 0.1800 | 0.0985 | 0.0284 | - | - | - |
| **25,000** | **0.2729** | 0.0838 | 0.0242 | **+0.0929** | 11-1-0 | **0.0010** |
| **44,000** | **0.3225** | 0.0373 | 0.0108 | **+0.1425** | 12-0-0 | **0.0005** |

**The response is log-linear in budget.** Slope per log10:

| interval | slope |
|---|---|
| 10k -> 25k | 0.2335 |
| 25k -> 44k | 0.2020 |
| 10k -> 44k (overall) | **0.2215** |

Fitting a straight line through the two endpoints predicts **0.2681** at 25,000. The midpoint
measured **0.2729** - a deviation of **+0.0048** against its own standard error of **0.0242**.
==There is no knee to exploit.==

The slope does drop from 0.2335 to 0.2020 between the intervals, a hint of mild flattening, but
the midpoint's error bar is five times that difference. **This data cannot resolve curvature; what
it can rule out is a large one.**

> [!warning] Why "no knee" is the expensive answer
> A knee would have meant most of the benefit arrives early and you could buy 2.5x instead of 4.4x.
> Log-linear means the opposite: **success grows linearly in the LOGARITHM of spend**, so each
> additional +0.14 costs another 4.4x. There is no cheap fraction of this curve.

**Variance also falls monotonically with budget** (sd 0.0985 -> 0.0838 -> 0.0373, regressions
1/12 -> 0/12). More budget does not merely raise the mean; it makes the outcome reliable.

## Claim 3 - MECHANISM: no collapse, and the reward signal is why

| | EXP-046 (this) | EXP-043 @ 10k | EXP-045 (back-loaded) |
|---|---|---|---|
| deepest-stage entropy, first -> last 10% | **0.2946 -> 0.2810** | - | 0.5914 -> 0.0979 |
| deepest-stage entropy minimum | **1.07e-03** | - | **2.7e-06** |
| deepest-stage training solve rate | **0.3343** | - | **0.0218** |
| modal action fraction | **0.393** | 0.473 | 0.847 |

**Entropy barely moves across the whole deep stage** (0.2946 -> 0.2810) and the policy solves
**33.4%** of its training episodes there. Compare EXP-045, which spent a comparable number of
episodes at depth 7 and solved **2.2%**: entropy died and the policy collapsed.

**This is the same mechanism from both sides.** What kills a run is a long stretch with no reward
signal; what a bigger total budget buys is arriving at the deep stage *competent enough to keep
earning reward there*. EXP-046 is also **less collapsed than its own baseline** (modal 0.393
against 0.473), so more budget does not merely delay collapse - it avoids it.

## Claim 4 - failure counts, DESCRIPTIVE, no p-value

**0 / 12** at exactly 0.000, against EXP-043's 1 / 12. Reported as a count, **no test**: n=12
cannot show a count went to zero.

## Claim 5 - what this means

The pre-committed CONFIRMED reading, unchanged: **the depth series is a budget series.** Practical
consequences:

1. **Every depth number in this project needs "at 10,000 episodes" attached.** They are not wrong;
   they are conditional in a way nobody wrote down until now.
2. **"The break point" was never a property of the architecture.** It was the depth at which
   10,000 episodes stopped being enough. It has now moved twice by spending, not by design.
3. **Depth 8 is predictable and expensive**: ~194,000 episodes for ~0.18, roughly 4 days. The
   budget grows about 4.4x per depth while success stays flat, which is a **losing exchange rate**
   for chasing depth by compute alone.
4. **The remaining interesting levers are the ones that change the exchange rate**, not the budget.
   Fine-tuning the encoder during RL is the only untried one.

## Limitations

- **Two depths, one multiplier.** 4.4x at depths 6 and 7. Nothing tested at 2x, 10x, or at depths
  4 and 5.
- **No saturation point known.** Three points spanning 4.4x show no flattening, but nothing was
  tested above 44,000 at depth 6, and a knee could sit anywhere beyond it.
- **Depths 3 and 4 still do not fit a pure budget story.** Depth 3 scores **lower** than depth 4
  (0.3972 against 0.5351) despite a shell 4.5x smaller, so something other than budget is binding
  at the shallow end.
- **One cap value, pretrained encoders only.** As in EXP-042/043/044/045.
- Depths 1-7 are **0.9%** of the state space. A random scramble sits at **depth 11**.

## Lead for the next experiment

1. **DONE - the midpoint ran and the curve is log-linear.** ~0.22 success per log10 of budget at
   depth 6, no knee.
2. **Fine-tune the encoder during RL.** Now the only untried lever that is not about spending more,
   and - given a log-linear budget curve - the only one that could change the exchange rate rather
   than pay it.
3. **Restate the depth series** in `road-to-a-solved-cube` and the vault progress tracker with the
   budget caveat attached.
4. **Re-run depth 5 with 24+ seeds** to settle EXP-043's Claim 1 (+0.1108 at p 0.0815).

**Still refuted and CLOSED:** width (EXP-033), volume alone (EXP-034), curriculum stage weighting
(EXP-037, confirmed to generalise by EXP-045), starvation at depth 6 (EXP-037 - **note this is now
superseded in scope**: EXP-037 refuted *reallocating* a fixed budget, and EXP-046 confirms
*raising* it works), trainer stabilizers (EXP-038), deleting the depth-1 stage (EXP-042),
deepest-shell coverage as the explanation (EXP-045).

## Regenerate

```bash
.venv/bin/python -u experiments/046_depth6_budget/run.py \
    --seeds 0 1 2 3 4 5 6 7 8 9 10 11 --workers 12 --skip-existing
.venv/bin/python experiments/046_depth6_budget/aggregate.py
# EXP-043's exp043_capped_d6 records are the paired baseline; run.py refuses without them.
```
