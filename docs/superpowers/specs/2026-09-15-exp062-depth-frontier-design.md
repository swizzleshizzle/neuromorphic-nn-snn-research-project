# EXP-062 design - the honest depth frontier, and an out-of-sample test of the budget law

> **PRE-REGISTERED. Committed before any EXP-062 number exists.** Thresholds fixed at commit time.
> **Date:** 2026-09-15 · **Phase:** 3 · **Grounds:** EXP-044/045/046 (the budget law), EXP-053
> (the recipe), EXP-056/057/060 (the critic mechanism and its gate).

## 0. Why this exists: Stage 4 is priced out, and the checkpoint has to say so

`road-to-a-solved-cube` calls **Stage 4 - depth-11 random scrambles - "the actual deliverable"** and
says it is *"genuinely achievable... nothing about it requires new science."* **That was written
2026-08-02/07, BEFORE week 20 discovered the budget law**, and the law refutes it.

Success is linear in the logarithm of spend at about **0.22 per log10**, with **4.4x buying roughly
one depth** (EXP-044/045/046). Anchored at depth 7's measured **0.2004** (EXP-053 arm B):

| depth | projected at 10k episodes | budget for depth-7 parity | h/cell at parity |
|---|---|---|---|
| 8 | **0.0588** | 4x | 17 h |
| 9 | -0.0827 | 19x | 80 h |
| 10 | -0.2243 | 85x | 381 h |
| **11** | **-0.3658** | **375x** | **1,796 h** |

**Depth-11 at depth-7 parity is ~75 days of compute per seed, 150 days for a 12-seed arm.** Nothing
new is needed *scientifically*; it is priced out arithmetically. **Phase 3 must report that
plainly rather than quietly miss its deliverable.**

**But the pricing rests on extrapolating a law fitted at depths 3-7 out to 11, and an extrapolation
used to declare a deliverable unreachable deserves a test.** That is this experiment.

## 1. What is being measured

**Two arms, 12 seeds each, 24 cells. EXP-053 arm B copied field for field, varying ONLY `depth`.**

| arm | depth | tag |
|---|---|---|
| **D8** | 8 | `exp062_frontier_d8` |
| **D9** | 9 | `exp062_frontier_d9` |

Base config: `arm="regionalized"`, `readout="concept"`, 10,000 episodes,
`curriculum=(1..depth)`, `max_steps_by_depth=((1,2),)`, `entropy_beta=0.0`,
`normalize_advantages=False`, `max_depth=depth`, **E1 encoder FROZEN**
(`exp047_ft_d6_lr0.0001_*`), `critic_lr` from `selected_critic_lr.json`.

**This is the full current recipe**: capped depth-1 training (EXP-042), a fine-tuned encoder
(EXP-047), and a learned critic (EXP-053, mechanism confirmed by EXP-056/057 and replicated by
EXP-060). Nothing is new here except the depth.

### Pre-flight facts, measured before this spec was written

| | |
|---|---|
| depth-8 shell | **114,149 states**, split to 113,949 train / 200 held-out |
| depth-9 shell | **360,508 states**, split to 360,308 train / 200 held-out |
| `is_heldout` at both depths | **True** - the same instrument as depths 5-7, not a different one |
| BFS table build | 1.19 s at depth 8, 3.25 s at depth 9 - negligible against a ~4 h cell |
| **measured chance floor, 12 seeds** | **exactly 0.0000 at depths 7, 8 AND 9** |

**The floor being exactly zero matters.** At depth 1 it is 21%, because a random walk with a `2d+3`
budget can stumble into solved; by depth 7 that is impossible. **So at these depths any non-zero
success is above chance**, and the usual "measure the floor, do not assume it" caveat resolves in
the simplest possible direction.

## 2. Claims

### Claim 1, PRIMARY - does the budget law hold OUT OF SAMPLE at depth 8?

The law was fitted at depths 3-7. Its prediction for depth 8 at 10,000 episodes is
**0.0588**.

**This is a ONE-SAMPLE test against a predicted value, not a paired contrast.** Compute the
two-sided 95% interval on depth-8 mean success (t, df=11) and ask whether **0.0588** lies inside.

| outcome | reading |
|---|---|
| interval CONTAINS 0.0588 | **LAW SUPPORTED out of sample.** The Stage-4 pricing is evidence-backed rather than extrapolated. |
| interval EXCLUDES 0.0588, mean ABOVE | **the law is PESSIMISTIC** past its fitted range - Stage 4 is cheaper than priced, and the frontier question reopens. |
| interval EXCLUDES 0.0588, mean BELOW | **the law is OPTIMISTIC** - Stage 4 is even further out of reach, and earlier depth projections were too kind. |

**Any of the three is a result.** The middle and bottom rows are the more interesting ones, because
a law that fails to extrapolate is a bigger finding than one that holds.

### Claim 2 - the frontier: is depth 9 distinguishable from zero?

**CONFIRMED AT THE FLOOR if depth-9 mean success < 0.02.** The measured chance floor is exactly
0.0000, so this is a comparison against true zero rather than against an estimate.

> [!warning] **CLAIM 2 IS A FRONTIER MEASUREMENT, NOT A TEST OF THE LAW, AND IT CANNOT BE ONE.**
> The law predicts **-0.0827** at depth 9. **Success is bounded below at zero**, so a negative
> prediction can only manifest as "about zero" - and depth 9 therefore **cannot discriminate -0.08
> from -0.5**. Reading a zero at depth 9 as *confirmation* of the law would be reading a bound as
> evidence. **It establishes where the frontier is; the law is tested at depth 8 only.**

### Claim 3, THE VALIDITY GATE - the recipe must actually be in place

**A CONDITION, not a report.** These numbers are only interpretable if the critic - the recipe
component that makes depth 7 work at all - is present and state-dependent. Reused from EXP-056 and
EXP-060 in its **stricter every-stage form**.

**Condition: the ratio of `V`'s within-episode RMS to the returns' within-episode RMS must be
`>= 0.05` at EVERY stage, in both arms.**

**Why reuse is safe here**, per the gate-calibration rule: it is a **ratio**, so bounded and
scale-free; it was calibrated in this regime (depth 7, this config) where EXP-060 measured
**0.3489** at the deepest stage and 2.2184 at the shallowest; and it can both pass and fail.

**Extrapolated to the new stages:** EXP-060's ratios decline roughly geometrically, about 0.73 per
depth at the tail, giving **~0.25 at depth 8 and ~0.19 at depth 9** - still four to five times the
floor. **Stated now so a pass is not mistaken for a wide margin it does not have.**

### Multiplicity

**Two inferential statements** (Claims 1, 2), **Bonferroni 0.025** where a p-value applies.
Claim 3 is a condition.

## 3. Cost

**24 cells, 6 workers, 4 waves.** Six workers because per-cell time is better there than at ten and
24 divides by 6 exactly.

| | steps at 10k episodes | h/cell |
|---|---|---|
| depth 8 | 120,000 | ~4.1 h |
| depth 9 | 129,987 | ~4.4 h |

**Derived from `steps()` at 0.115 s/step and then corrected UPWARD by 6%**, because EXP-061
measured **3.05 h** at depth 5 where the same formula predicted 2.88 h. **Expect ~17 h, and treat
it as a floor** - five of the last six estimates came in low, and the one that held was priced from
a same-worker-count measurement exactly like this one.

## 4. What this cannot answer

- **It cannot rescue Stage 4.** Even the most favourable Claim 1 outcome leaves depth 11 at tens of
  days per seed. This prices the deliverable; it does not deliver it.
- **It cannot test the law at depth 9 or beyond.** See Claim 2's warning: the zero floor censors
  every negative prediction into the same observation.
- **It says nothing about a better recipe.** A learned readout, or anything that changes the
  exchange rate rather than paying it, is out of scope and is Phase 3's stated next question.
- **n=12, not 24.** The interval on Claim 1 will be wide; that is accepted because the claim is a
  containment test against a point prediction rather than a small-effect contrast.
