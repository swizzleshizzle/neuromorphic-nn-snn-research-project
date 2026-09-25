# EXP-069 design - at depth 5 with a pretrained encoder, does the ENCODER carry the seed effect?

> **PRE-REGISTERED. Committed before any EXP-069 number exists.**
> **Date:** 2026-09-25 · **Phase:** 3, final compute · **Grounds:** EXP-068, `docs/seed-effect.md`,
> EXP-040, EXP-043.

## 0. The question

EXP-068 decomposed the seed effect at depth 3, on a cell with **no pretrained encoder**:
interaction **0.650**, task draw **0.267**, trajectory **0.084**. But `docs/seed-effect.md`
measured the confound at **depth 5 with an E0 encoder**, where a seed fixes a third thing, the
pretrained encoder, and EXP-068's RESULTS named it the next suspect. **This measures its share.**

## 1. The obvious design leaks, and this one does not

Each shipped E0 encoder (`exp040_encoder_s{e}.pt`) was pretrained with **its own seed's held-out
states forbidden**: EXP-040's `rl_heldout_union` forbids seed e's depth-4/5/6 evaluation states.
Crossing E0 encoder e with split r != e would therefore evaluate on states encoder e **may have
pretrained on**.

==That contamination inflates exactly the off-diagonal cells, lands in the interaction term, and
would imitate EXP-068's headline.== It was caught while designing, not after running.

**The fix:** phase `pretrain` builds **8 new encoders**, e = 0..7, using EXP-040's own recipe
(`PRETRAIN`, `PRETRAIN_DEPTHS`, imported), each forbidding the **union of all 8 grid splits'**
held-out states. Measured before dispatch:

| | states forbidden | share of the 11,912-state pretraining pool |
|---|---|---|
| one seed (shipped E0) | 533 | 4.5% |
| **union of 8 splits (this design)** | **3,144** | **26.4%** |

**The costs, paid knowingly:** these are not the shipped encoders and were trained on less data, and
**the diagonal no longer reproduces EXP-043**, so there is no free byte-identity validation this
time. The run asserts no shipped `exp040_encoder` path can enter a cell.

## 2. Design

**Factors, fully crossed, 8 x 8 = 64 cells** on EXP-043's capped depth-5 cell, obtained by
`dataclasses.replace` on the config EXP-043's own driver builds:

| factor | what varies | fields |
|---|---|---|
| **E, encoder** | pretraining seed and the frozen brain's init | `encoder_state_path` = encoder e, `encoder_seed` = e |
| **R, rest** | task draw and trajectory together | `split_seed` = `train_seed` = r |

R bundles split and train deliberately: EXP-068 already separated those, and holding them together
keeps the grid at a size the laptop can run. **Tags carry both indices** (`exp069_en3rs5`), because
`record_filename` encodes neither; the driver asserts 64 distinct names.

**Analysis:** EXP-068's two-way decomposition and randomization test, **imported, not copied**, so
both experiments are read by the same audited code.

## 3. Claims, with the bands IN the verdict functions

**The EXP-068 lesson is applied.** Each band below is returned by `aggregate.py` as a verdict of
its own; none lives only in this prose.

### Claim 1, PRIMARY - is the encoder a carrier?

| encoder share | verdict |
|---|---|
| < 0.10 | **MINOR** |
| 0.10 to 0.35 | **UNRESOLVED** |
| >= 0.35 | **MAJOR** |

**What these bands can detect, calibrated by simulation BEFORE dispatch** (400 replicate grids at
8x8, total variance matched to EXP-068's):

| true share | estimate, 5-95% |
|---|---|
| 0.05 | 0.000 to 0.175 |
| 0.15 | 0.000 to 0.338 |
| 0.30 | 0.046 to 0.505 |
| 0.45 | 0.102 to 0.610 |

==This design distinguishes "the encoder carries almost nothing" from "it carries a third or more",
and nothing finer.== Landing in the wrong extreme is about 5% at worst. A wide UNRESOLVED band is the
honest consequence of 7 df, not a hedge.

### Claim 2 - does EXP-068's headline replicate here?

Interaction share **>= 0.50 REPLICATED**, **< 0.30 NOT REPLICATED**, **UNRESOLVED** between. At a
true 0.60 the estimate spans about 0.39 to 0.87.

### Claim 3, VALIDITY GATE - three conditions, all required

1. **Grand mean >= 0.10**, EXP-036's working bar. A grid of dead policies decomposes nothing, which is
   EXP-064's lesson. EXP-043 measured this cell at **0.29 to 0.34** with shipped encoders, so the bar
   can pass; weaker encoders or a collapse would fail it.
2. **Total sd >= 0.05.**
3. **At least one main effect resolves at p < 0.05**, so the shares are not ratios of noise.

## 4. Cost

| | |
|---|---|
| Phase `pretrain` | 8 encoders at 8 workers. EXP-040 did 12 at 10 workers in ~1.6 h, so **~1.3 h** |
| Phase `rl` | 64 cells, 8 workers, **8 clean waves** |
| Per cell, estimate | EXP-059 measured 3.05 h at 6 workers on this cell; 8 workers will be slower (EXP-068 saw 71% CPU-to-wall at 10) |
| **Total, estimate** | **~28 to 40 h**, to be MEASURED from wave 1 and reported in RESULTS |

**No separate calibration wave this time**, because the run is launched unattended. The first
wave of 8 cells prices the run just as well, and no claim depends on the cost. Every cell writes its
own record and the launcher passes `--skip-existing`, so an interruption costs one wave.

## 5. What this cannot answer

- **Split versus train at depth 5.** R bundles them; EXP-068 separated them at depth 3 only.
- **Anything about the SHIPPED E0 encoders specifically.** These are union-excluded siblings.
  If they turn out much weaker, the encoder share is measured on a weaker encoder family.
- **Depths other than 5.**
