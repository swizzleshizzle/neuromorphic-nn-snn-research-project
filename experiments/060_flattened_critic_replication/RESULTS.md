# EXP-060 results - EXP-056 REPLICATES on fresh seeds

> **COMPLETE.** 20 cells, 10 fresh seeds, two arms, no tracebacks. **Validity gate PASSED at every
> stage**, so Claim 1 may be read.
>
> **HEADLINE: EXP-056 REPLICATES. `F` minus `B` is -0.0925 at p 0.0156** on seeds 14-23 alone,
> clearing the pre-registered -0.05 bar downward and clearing Bonferroni 0.025. W-L-T 2-8-0.
>
> **The effect is LARGER than the original's -0.0646, not smaller** - which is the opposite of what
> regression to the mean would predict for a result selected for significance.
>
> **The critic conclusion no longer rests on one knife-edge contrast.** EXP-056 cleared its
> threshold by 6.4% of its margin at n=12; this is an independent test on seeds that have never
> been used for the question.

**Pre-registration:** `docs/superpowers/specs/2026-09-10-exp060-flattened-critic-replication-design.md`.
Cost section amended upward before dispatch; **no claim, threshold or test was changed at any point.**

> [!warning] **DISCLOSURE: this aggregator was written AFTER the run, unlike EXP-059's.**
> Verifying phase 2 had finished meant reading the launcher log's tail, which prints one line per
> cell, so **roughly twelve individual cell success rates had been seen before the aggregator was
> written.** No contrast, arm mean or gate value had been computed. Every rule applied - the bar,
> the direction, the three readings, the multiplicity, the test, the gate threshold and its
> every-stage form - was fixed in the spec before dispatch, so the discretion left to contaminate
> was small. **It is recorded because the standard here is to say so, not because it is harmless.**

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo`, worktree `C:\Users\mlgbr\wt-exp053` at `2363871`, 10 workers |
| Wall clock | 2026-09-11 21:33 to 2026-09-12 18:06 UTC, **20.6 h across three phases** |
| Arms | EXP-056 field for field, `flatten_critic` the only variable, `critic_lr` 0.01 FIXED |
| Seeds | **14-23, fresh.** The driver refuses any seed below 14 |
| Encoders | E1, `exp047_ft_d6_lr0.0001_*`, manufactured for these seeds by this experiment |

**Three phases, because the dependency chain is longer than the spec assumed.** E1 encoders exist
only for seeds 0-13, and EXP-047's `confirm` mode refuses without EXP-043's **depth-6** baseline for
the same seeds - which existed only for 0-11. Depth **5** has all 24, which is why the gap was easy
to miss: EXP-059 ran on depth-5 encoders and found everything it needed.

| phase | what | est. | **measured** |
|---|---|---|---|
| 0 `baseline` | EXP-043 depth 6, seeds 14-23 | 3 h | **4.3 h** |
| 1 `finetune` | EXP-047 `--mode confirm` -> E1 | 6 h | **7.1 h** |
| 2 `rl` | arms B and F, 20 cells | 8 h | **9.1 h** |

**All three overran, in the same direction, by the same mechanism**: each estimate was scaled from a
6-worker measurement, and the real per-cell penalty at 10 workers is steeper than the playbook's
0.16-vs-0.115 s/step ratio implies. That is now three instances; **the ratio itself is the thing to
re-measure**, not each estimate in turn.

```bash
powershell -File C:\Users\mlgbr\launch060_wt.ps1 -Phase baseline -Workers 10   # then finetune, then rl
.venv/bin/python -u experiments/060_flattened_critic_replication/aggregate.py
```

## Claim 3, the validity gate - PASSED at every stage

If `V` barely varied within an episode there was nothing for flattening to remove and Claim 1's
result would be vacuous. Measured on the **flattened** arm, where the removed variation lived:

| depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| `V` within-episode RMS | 0.7303 | 2.1809 | 2.1249 | 2.3149 | 2.1860 | 1.8498 | 1.5562 |
| returns' within RMS | 0.3292 | 1.3471 | 1.9353 | 2.6655 | 3.3291 | 3.8985 | 4.4606 |
| **ratio** | 2.2184 | 1.6190 | 1.0980 | 0.8685 | 0.6566 | 0.4745 | **0.3489** |

**The smallest ratio is 0.3489, seven times the 0.05 threshold.**

> [!note] **This gate is STRICTER than EXP-056's own, deliberately.** That experiment's aggregator
> passed on `any(ratio >= 0.05)`; this spec pre-registered **every** stage. The spec's wording
> binds, because relaxing a gate once numbers exist is exactly the EXP-058 mistake. Both forms pass
> here, so nothing turns on it - but the divergence is recorded, and **the every-stage form is the
> one to carry forward**, since it is the one that cannot be satisfied by a single favourable stage.

## Claim 1, PRIMARY - `F` minus `B` on seeds 14-23 ALONE. **-0.0925 at p 0.0156.**

| | fresh seeds 14-23 (this run) | original seeds 0-11 (EXP-056/053) |
|---|---|---|
| arm `F`, flattened | **0.0540** (sd 0.0530) | 0.1358 (sd 0.0747) |
| arm `B`, full critic | **0.1465** (sd 0.0785) | 0.2004 (sd 0.0530) |
| **paired delta** | **-0.0925** (sd 0.0891) | -0.0646 (sd 0.0814) |
| W-L | **2-8** | 3-9 |
| p | **0.0156**, exact over 1,024 flips | 0.0234 |

**REPLICATED at the pre-registered bar.** The point estimate grew rather than shrank, and the
paired sd is nearly identical across the two blocks (0.0891 against 0.0814), so this is not a
noisier measurement that happened to land well.

## Claim 2, SECONDARY - the pooled contrast, and read the caveat every time

**n=22: -0.0773, p 0.0005, approx 95% interval [-0.1146, -0.0400].**

> **THIS FIGURE INHERITS OPTIONAL STOPPING.** The decision to add seeds 14-23 was made *because*
> EXP-056's p-value was marginal, so a pooled p-value is selected-on and is not the number it
> appears to be. **The primary is Claim 1, on the fresh seeds alone.**

> [!warning] **AND THERE IS A SECOND, INDEPENDENT REASON TO DISTRUST THE POOLED FIGURE: it pools
> across a LEVEL SHIFT.** Both arms sit far lower on the fresh seeds - `F` 0.0540 against 0.1358,
> `B` 0.1465 against 0.2004, roughly **0.05 to 0.08 lower in each arm**. The *contrast* replicates
> because it is paired within seed, which is precisely what a paired design buys; but the two
> blocks are not samples from the same level, and a pooled mean silently averages them.
>
> **The level shift is unexplained and this experiment cannot explain it.** Same recipe, same
> `selected_lr.json`, same config; the only differences are the seeds themselves and that these
> encoders were manufactured on 2026-09-12 rather than 2026-08-21. **Do not quote the pooled
> number without both caveats.**

## What this changes

1. **The critic conclusion is now supported by an independent replication.** EXP-056 cleared
   Bonferroni by 6.4% of its margin at n=12; with a fresh-seed replication at p 0.0156, "within-
   episode state-dependence is the critic's benefit" no longer rests on a single thin contrast.
2. **Read beside EXP-057, the decomposition holds.** Calibration alone was worth nothing
   (+0.0088, p 0.7822); flattening within-episode structure costs 0.065 to 0.093 in two
   independent runs. **Those are the two halves of the same claim and both have now been tested
   twice.**
3. **A paired design survived a large level shift that would have destroyed an unpaired one.**
   Both arms moved ~0.06 between seed blocks while the contrast reproduced. Worth remembering the
   next time a "just compare the arm means" shortcut is tempting.
4. **The 10-vs-6-worker cost ratio in the playbook is wrong and should be re-measured.** Three
   phases, three overruns, all in the same direction.

## What is NOT claimed

- **Not that the pooled -0.0773 is a clean n=22 estimate.** It carries optional stopping *and*
  pools across a level shift. Quote Claim 1 instead.
- **Not that the level shift is understood.** It is flagged, not explained, and nothing here
  distinguishes seed-set difficulty from anything about the encoder manufacturing run.
- **Not that -0.0925 is the true effect size.** n=10, and the interval is wide; the replication
  establishes the direction and that the effect clears the bar, not its magnitude.
- **Not a resolved decomposition of the critic's benefit.** EXP-057's Claim 3 remains unresolvable
  at these sample sizes, as its own spec pre-registered.
- **Not anything about depths other than 7**, or about a critic on a non-frozen encoder.
