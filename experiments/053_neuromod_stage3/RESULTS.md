# EXP-053 Results - the critic clears its bar by a hair, and the neuromorphic claim is refuted

> **Why this file exists:** the standing habit from the 2026-07-13 Phase-2 audit. The contract was
> committed at `6165468` and amended twice - `8419e26` (signed dopamine) and `e3881fd` (the
> significant-sub-bar branch, the Claim 6 instrument, and the rate-vs-spacing limitation) - **all
> before any number existed**. No threshold was edited while filling this in.
>
> **Provenance:** 36 new records (12 per arm) plus a 6-cell critic-lr pilot. Laptop `SwizzlesDuo`,
> 6 workers, 2026-08-28 to 08-30, run from a git worktree at `C:\Users\mlgbr\wt-exp053` on
> `exp-053-stage3`. Zero tracebacks. Controls are EXP-051 (depth 7) and EXP-047's `lr0.0001`
> confirmatory arm restricted to seeds 0-11 (depth 6); **neither was re-run**.
>
> **Regenerate:**
> ```bash
> .venv/bin/python -u experiments/053_neuromod_stage3/pilot_critic_lr.py --workers 6
> .venv/bin/python experiments/053_neuromod_stage3/select_critic_lr.py
> .venv/bin/python -u experiments/053_neuromod_stage3/run.py --arm B --workers 6   # then G, then R
> .venv/bin/python experiments/053_neuromod_stage3/aggregate.py
> ```

## Headline

**A learned critic helps at depth 7 and we cannot say why. Gating encoder plasticity on the
neuromodulatory bus does not clear its bar, and the pre-registered rule retires the neuromorphic
claim rather than deferring it.**

| | arm | control | delta | W-L-T | exact p | bar | verdict |
|---|---|---|---|---|---|---|---|
| **Claim 1** critic, d7 | **0.2004** | 0.1471 | **+0.0533** | 8-4-0 | **0.0498** | +0.05 | **CONFIRMED** |
| **Claim 2** gate, d6 | 0.3004 | 0.2700 | +0.0304 | 9-3-0 | 0.1323 | +0.05 | not confirmed |
| **Claim 3** G vs R, d6 | 0.3004 | 0.2654 | +0.0350 | 7-5-0 | 0.1167 | +0.03 | not confirmed |

## Claim 1 (PRIMARY) - the critic. **CONFIRMED, and it clears both thresholds by a hair.**

**+0.0533 against a +0.05 bar, at p 0.0498 against an alpha of 0.05.** Both margins are
approximately 6% and 0.4% respectively. This is a confirmation, and it is not a robust one.

**The mechanism it was theorised to work through is measurably absent.**

| | value |
|---|---|
| critic explained variance, final stage | **0.0021** |
| revisit_rate, paired | -0.0177, p 0.9189 |
| optimality, paired | +0.0569, p 0.6011 |

An explained variance of 0.002 means that **at depth 7** `V(s)` is no better than a constant. So
"a learned state-dependent baseline reduces gradient variance **at the evaluated depth**" is not
supported, even though "a learned critic raises depth-7 success" is.

> [!warning] CORRECTED 2026-09-02: "THE MECHANISM IS ABSENT" WAS GENERALISED FROM ONE STAGE, AND
> THE BY-STAGE NUMBERS WERE IN THESE RECORDS THE WHOLE TIME.
> The table above is labelled "final stage" and is accurate. The sentence that followed it was not:
> it read a depth-7 number as a statement about the critic, and a later handoff hardened that into
> "its mechanism is measurably absent". **The critic is strongly predictive in the early
> curriculum.** `stage_trace[i]["critic_ev"]`, mean over the same 12 seeds:
>
> | depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
> |---|---|---|---|---|---|---|---|
> | critic explained variance | **+0.4702** | **+0.2054** | +0.0983 | -0.0292 | -0.0668 | -0.0577 | +0.0021 |
> | seeds with ev > 0 | 12/12 | 12/12 | 12/12 | 3/12 | 0/12 | 0/12 | 8/12 |
> | steps per episode | 1.22 | 2.50 | 4.15 | 6.76 | 9.77 | 12.55 | 15.57 |
>
> **This is systematic, not noise**: every seed is positive at depths 1 to 3 and every seed is
> negative at depths 5 and 6. `V(s)` explains 47% of return variance at depth 1, where episodes are
> 1.22 steps long and 78% are solved, and stops explaining anything from depth 4 on.
>
> **Claim 1 evaluates depth-7 success after training through the WHOLE curriculum**, so the
> critic's benefit is free to originate in the stages where it genuinely predicts. "The mechanism
> is absent" is not an available reading. What is established is narrower: the mechanism is absent
> **at the final stage**.

**The leading alternative, restated after the correction.** Two candidates now, not one:

1. **Calibration, not state-dependence.** The advantage is `G_t` minus a constant fitted by MSE
   over the episode, against an EMA baseline with `beta=0.1` that **lags**. The critic may simply
   be a better-calibrated constant.
2. **Early-curriculum shaping.** The critic really is state-dependent where it matters, at depths
   1 to 3, and that advantage carries forward through the curriculum.

**A per-episode batch-mean baseline does NOT separate these, and must not be used to.** That arm
forms `G_t - mean(G)`, which is **exactly zero for a one-step episode** - and depth 1 averages 1.22
steps per episode. It would be starved of gradient precisely in the stages where the critic
predicts best, so a loss against the critic would be uninterpretable: the confound is aligned with
the hypothesis. **EXP-056 flattens the critic's prediction to its own episode mean instead**, which
holds calibration and fitting fixed and removes only within-episode state-dependence.

**Claim 5 - dead seeds. 0/12 against the control's 1/12.** The critic rescued the seed that died
outright. The spec pre-registered this precisely because a change that rescues a dead seed while
barely moving the mean is invisible in an average, and at depth 7 the failure mode is discrete.

> The critic-lr pilot predicted this arm would be null. It was selected blind to success rate and
> reported explained variance of 0.0204 at best, four of six cells negative. **That prediction was
> wrong, and running the pre-registered arm rather than cancelling on a pilot signal is what caught
> it.** The pilot's own reading - that the critic explains almost no return variance - was
> confirmed at 0.0021 and is what makes Claim 1 hard to interpret.

## Claim 2 - the gate. **NOT CONFIRMED.**

**+0.0304, 61% of the bar, p 0.1323.** Nine of twelve seeds improved, so the direction is positive
and consistent, but it does not reach the pre-registered threshold and is not significant.

**The gate mechanism itself worked exactly as designed.** The running-median threshold produced a
realized update rate of **0.4987** (min 0.4328, max 0.5443) - within 0.13% of the intended 50%,
with no tuned constant anywhere. `trainable_params` is **27,206**, so the split optimizer did not
double-count. `NeuromodBus.learning_enabled` and `Brain.learn()` were read and called for the first
time since L11.

**The bus became load-bearing. It did not buy anything.**

## Claim 3 (ATTRIBUTION) - G against its rate-matched control. **NOT CONFIRMED.**

| | value |
|---|---|
| arm G, dopamine-gated | 0.3004, sd 0.0904 |
| arm R, random-gated at G's realized rate | 0.2654, sd 0.0658 |
| paired delta | **+0.0350**, exceeding the +0.03 bar |
| W-L-T | 7-5-0 |
| exact p | **0.1167** |

**The delta clears its bar and the significance does not.** With 7-5 splits at n=12, this is
unresolved rather than null.

### The pre-registered verdict, applied as written

```
CLAIM 2 NOT CONFIRMED and CLAIM 3 NOT CONFIRMED. Encoder updates are redundant at this rate.
The neuromorphic claim is REFUTED, not deferred. 'We need a better gate' is NOT an available
conclusion from this experiment.
```

That verdict stands, and it is the one the spec committed to before any number existed.

### One mechanism number moved, and it is a lead, not a rescue

| G vs R | delta | p |
|---|---|---|
| **revisit_rate** | **-0.0262** | **0.0083** |
| optimality | +0.0180 | 0.3599 |

**Arm G revisits states significantly less often than arm R at an identical update rate.** That is
the trajectory instrument `CLAUDE.md` prefers over performance, moving in the predicted direction,
on the one comparison that isolates the gate's *signal* from its *rate*.

**It does not rescue Claim 3 and must not be reported as if it did.** Claim 4 is descriptive with no
bar; six such tests were run across three contrasts, so a single p 0.0083 sits exactly at a
Bonferroni threshold of 0.0083 and no lower. The honest statement is: **the dopamine signal changes
trajectories relative to random gating, and that change did not convert into a significant success
difference.**

## A gap in the pre-registered decision table, found by the data

Claim 3's table has four rows and **no row for "G does not beat its control, yet G beats R"** -
which is close to what happened (+0.0350 over R, above the bar, below significance). As written,
`claim3_verdict` falls through to the catch-all and reports REFUTED regardless of the R comparison.

**The table was not edited after seeing the numbers.** Modifying a pre-registration once an arm has
reported is the outcome-dependent editing it exists to prevent. The gap is recorded here as a
limitation of the design, and any successor should add that row **before** running anything.

> [!note] THE ROW WAS ADDED 2026-08-31, WITH NOTHING PENDING THAT COULD USE IT, AND THIS RUN'S
> VERDICT IS UNCHANGED.
> It fires only when `beats_r` is true, which requires the G-vs-R contrast to clear the +0.03
> attribution bar **and** reach significance. This run's is +0.0350 at **p 0.1167**, so it does not
> fire, and `claim3_verdict` still returns the same catch-all string byte for byte. Re-running the
> aggregator on these records confirms it, and a regression test pins the exact string against the
> measured pair `(+0.0304, 0.1323)` and `(+0.0350, 0.1167)` so a later edit cannot quietly move it.
>
> The row is **not** another early return. Every Claim-2-did-not-confirm branch used to return
> immediately and discard the R contrast, so putting the new row in any one of them would have left
> the same hole one row over. The Claim 2 verdict is now settled first and the R comparison is
> appended, which covers all four states.
>
> What it may say is bounded by this experiment's own limitation note: **arm R controls for RATE,
> not SPACING**, so an R win licenses "the gate's timing carries information" and never "the
> dopamine signal is what matters". The row says so in its own text, and says explicitly that a
> signal which reorders trajectories without moving success is a lead rather than a result.

## Claim 6 - compute, priced

Per-cell wall clock, measured by the instrument added at `e3881fd` (Claim 6 had no instrument at
all until the final review caught it):

| arm | mean s/cell | min | max |
|---|---|---|---|
| pilot (d7) | 9,950 | 9,478 | 10,511 |
| B (d7) | 7,915 | 7,252 | 8,686 |
| G (d6) | 10,067 | 8,708 | 12,302 |
| R (d6) | 11,522 | 10,406 | 12,867 |

**Every arm and every control ran exactly 10,000 episodes**, and EXP-046's budget curve is
denominated in episodes, so **no arm carries an episode-budget confound**. Arm B adds 65 parameters
and one `Linear(64 -> 1)` forward per step against a `brain.step` that dominates at about 90 ms, so
its per-step cost is essentially unchanged and its measured wall clock did not rise.

**Do not read the G-versus-R wall-clock difference as a cost difference.** They perform the same
number of optimizer steps by construction (rates 0.4987 and 0.4985). R ran later on the same laptop
and the gap is machine state, not arithmetic.

## Limitations recorded before the run

- **Arm R controls for RATE, not SPACING.** Arm G's opens are temporally autocorrelated - dopamine
  is measured against a `beta=0.1` EMA, so a good episode raises the bar for the next - while arm
  R's are i.i.d. Bernoulli. A G-over-R win would therefore license "the gate's timing carries
  information", not specifically "the dopamine signal is what matters". Recorded at `e3881fd`.
- **Arm G force-opens its first 100 episodes** as warmup while arm R draws at G's realized rate from
  episode 1, so roughly 50 of ~5,000 updates differ in placement at the highest-leverage point of
  the curriculum.
- **The entropy trace and the EXP-033 probe appear in no claim**, per the spec's section 0. Entropy
  is recorded and printed and decides nothing.

## What this changes

1. **A learned critic is a real lever at depth 7** and the cheapest one found so far: +0.0533 at
   matched episodes for 65 parameters. Its mechanism is unknown and the EMA-lag hypothesis is the
   obvious next test.
2. **Neuromodulated plasticity gating is refuted at this rate**, not deferred. The bus is now
   genuinely load-bearing in code, which is worth keeping, but it bought nothing measurable.
3. **`road-to-a-solved-cube` and `CLAUDE.md` should record that Week 20's "nothing neuromorphic
   participates in the learning" changed ONCE** - when the spiking encoder started training. Arm G
   wired the bus in and did not clear its bar, so it must not be written up as a second change.
