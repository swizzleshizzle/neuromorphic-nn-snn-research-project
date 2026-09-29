# EXP-069 results - at depth 5 the task draw and trajectory vanish, and the encoder takes their place

> **COMPLETE.** 64 cells plus 8 pretrained encoders, no tracebacks. **Validity gate PASSED.**
> Completion confirmed from **counts** (64 records, 64 heads, 8 encoders, zero python processes)
> **without reading the run log**.
>
> **HEADLINE: the interaction replicates almost exactly (0.652 here against 0.650 in EXP-068), and
> the rest of the seed effect has changed hands.** At depth 5 with a pretrained encoder, the task
> draw and trajectory together carry a main effect of **0.000** (F 0.291, p 0.9605). The
> **encoder** carries **0.348** (F 5.264, p 0.0001).
>
> **Claim 1 is UNRESOLVED**, and this time the aggregator says so itself. 0.348 misses the
> pre-registered MAJOR bar of 0.35 by **0.002**. The band exists precisely so that a number this
> close to a line cannot be rounded across it.

**Pre-registration:** `docs/superpowers/specs/2026-09-25-exp069-encoder-decomposition-design.md`,
committed with `run.py` and `aggregate.py` at `9edab30` **before any cell ran**. Nothing in the spec,
driver or aggregator was changed afterwards.

## Provenance

| | |
|---|---|
| Run | `SwizzlesDuo`, worktree `C:\Users\mlgbr\wt-exp053` at `9edab30`, **8 workers**, unattended |
| Pretraining | 8 union-excluded encoders, 2026-09-25 16:48 to 17:28 local, **40 min** |
| RL | 64 cells, 2026-09-25 17:28 to 2026-09-26 14:23 local, **20.9 h**, 8 waves at ~2.6 h each |
| Total | **~21.6 h against a pre-registered estimate of 28-40 h** |
| Collected | 2026-09-29, three days after it finished |

**The estimate came in HIGH this time, by roughly a third.** It was scaled from EXP-059's 6-worker
depth-5 figure, with a penalty added for EXP-068's E-core spill at 10 workers. At 8 workers the spill
was smaller than feared. Scaling was wrong in the safe direction for once, and it is still scaling:
this spec skipped the calibration wave because the run was launched unattended.

```bash
powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\launch069_wt.ps1 -Phase all -Workers 8
.venv/bin/python -u experiments/069_encoder_decomposition/aggregate.py
```

## Claim 3, the validity gate - PASSED

| condition | value | bar |
|---|---|---|
| grand mean, a WORKING policy | **0.3893** | >= 0.10 |
| total sd | **0.1098** | >= 0.05 |
| a main effect resolves | encoder **p 0.0001** | < 0.05 |

**The union-excluded encoders work well despite losing 26.4% of their pretraining pool.** The
grid's mean of 0.3893 sits at or above the 0.29 to 0.34 that EXP-043 measured with the shipped
encoders. That comparison is descriptive (different encoders, different cells), but it removes the
worry the spec named: that a weaker encoder family would make the share a measurement of weakness.

## The decomposition

| component | F vs MS_I | share | p |
|---|---|---|---|
| **E, encoder** | **5.264** | **0.348** | **0.0001** |
| **R, rest** (split and train together) | 0.291 | **0.000** | 0.9605 |
| **I, interaction** | - | **0.652** | - |

Descriptive, and carrying no verdict:

| | means | range |
|---|---|---|
| encoder rows, e0..e7 | 0.427 0.249 0.417 0.423 0.416 0.421 0.292 0.468 | **0.219** |
| rest columns, r0..r7 | 0.381 0.389 0.394 0.386 0.406 0.383 0.358 0.418 | **0.060** |

**Most of the encoder effect is two weak encoders**, e1 and e6, about 0.15 below the other six,
which cluster tightly. That shape is recorded because it matters for what "the encoder carries the
effect" means: it looks like occasional bad draws, not a smooth quality gradient. **It is not
tested.** Eight encoders cannot distinguish the two, and no claim was registered about it.

## Claim 1, PRIMARY - **UNRESOLVED.** Encoder share 0.348 against a MAJOR bar of 0.35.

**The verdict function returned UNRESOLVED, and that is the reading.** 0.348 is 0.002 under the bar.
The spec's simulation showed that at 8x8 a true share of 0.30 estimates anywhere in [0.046, 0.505],
so 0.348 is consistent with a true share well below or well above the line. The p-value of 0.0001
says the encoder effect is **real**. It does not say the effect is **major**, which is what Claim 1
asked, and the two must not be run together.

> [!important] **This is EXP-068's lesson paying out on the very next experiment.** EXP-068's band
> lived only in prose, and its aggregator printed CONFIRMED for a result inside it. Here the band is
> in `encoder_verdict()`, and the aggregator itself printed UNRESOLVED for a number that would
> otherwise have been very tempting to call MAJOR, missing by 0.002. **Nothing had to be overridden
> by hand, and nothing had to be disclosed.** That is the point of encoding it.

## Claim 2 - **REPLICATED.** Interaction share 0.652.

EXP-068 measured **0.650** at depth 3 on an encoder-free cell. Here, at depth 5, with a pretrained
encoder and a different pair of factors, it is **0.652**. **A seed's quality is mostly the specific
combination of its parts, in both regimes.** The practical rule, pair within seed, now rests on
two independent decompositions.

## What this changes

1. **The seed effect's main effects are regime-dependent; its interaction is not.** At depth 3 with
   no encoder, the task draw carried 0.267 and the trajectory 0.084. At depth 5 with an encoder,
   split and trajectory together carry **0.000**, and the encoder carries what main effect there
   is. The pretrained encoder appears to absorb what the task draw and trajectory did.
2. **Encoder draws matter, apparently in the form of occasional bad ones.** Two of eight were
   ~0.15 worse. Any encoder-dependent experiment compared across encoder seeds carries that risk;
   pairing on the encoder removes it, as pairing on the seed always has.
3. **The unresolved band worked as designed** on its first real test.

## What is NOT claimed

- **Not that the encoder is a MAJOR carrier.** UNRESOLVED, by the verdict function, at 0.002 under.
- **Not that split and trajectory are irrelevant at depth 5.** Their main effect is zero; they still
  participate in the 0.652 interaction.
- **Not anything about the SHIPPED E0 encoders.** These are union-excluded siblings, built so that
  the encoder factor could be crossed without leaking evaluation states into pretraining.
- **Not that encoder quality is bimodal.** Two low rows out of eight is a shape, not a test.
- **Not a comparison with EXP-068 beyond the interaction.** Different depth, different factors.
