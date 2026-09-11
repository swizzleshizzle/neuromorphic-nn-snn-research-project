# EXP-060 design - an independent replication of EXP-056's flattened critic

> **PRE-REGISTERED. Written before any EXP-060 number exists.** Thresholds fixed at commit time.
> **Date:** 2026-09-10 · **Phase:** 3 · **Grounds:** EXP-051, EXP-053, EXP-056, EXP-057.
> **NOT YET DISPATCHED.** The laptop is running EXP-059 until ~2026-09-11 16:30 UTC.

## 0. Why this exists

**EXP-056 is load-bearing and its margin is thin.** It found that flattening `V(s_t)` to its own
episode mean costs **-0.0646 at p 0.0234**, against a Bonferroni threshold of 0.025 - **it cleared
by 6.4% of its margin**, the same knife-edge EXP-053's Claim 1 sat on at p 0.0498 against 0.05.

Two experiments now lean on it. EXP-056 plus EXP-057 together are what closed the critic question:
everything without within-episode state-dependence sits between 0.1358 and 0.1558, and the one arm
that has it sits at 0.2004. **That conclusion currently rests on a single contrast that cleared its
threshold by a hair, at n=12.**

**This is a REPLICATION, not a re-run.** Seeds 14-23 have never been used for this question. It is
not the EXP-058 situation, where re-running the same seeds would have reproduced byte-identical
records; these are new cells that have never existed.

## 1. The problem this design exists to solve, and it is not a compute problem

**Extending an experiment BECAUSE its p-value was marginal, then pooling and reporting the combined
p, is optional stopping.** The decision to add seeds was made after seeing a significant result, so
a pooled p-value inherits that selection and is not the number it appears to be.

The design therefore splits the two questions rather than blending them:

| | contrast | contamination |
|---|---|---|
| **PRIMARY** | `F` minus `B` on **seeds 14-23 only**, n=10 | **none.** No seed here influenced the decision to run it. |
| **SECONDARY** | `F` minus `B` pooled over seeds 0-11 and 14-23, n=22 | **optional stopping.** Reported with that sentence attached, never as the headline. |

**The primary is the only contamination-free test, and it is underpowered. That is stated now.**

### Power, honestly, before any number exists

At this project's measured paired-difference sd of 0.10 to 0.14, `se` at n=10 is **0.032 to 0.044**:

| true effect | approx power at alpha 0.05, n=10 |
|---|---|
| -0.0646 (EXP-056's point estimate) | **~50-60%** |
| -0.05 | ~35-45% |

**And -0.0646 is an UPWARD-BIASED estimate of the true effect**, because it was selected for being
significant. A replication should expect something smaller, which puts the realistic case in the
lower row. **So a null primary is a plausible outcome even if the effect is real**, and it is
pre-registered here as a BOUND, never as a refutation of EXP-056.

**If a decisive test is what is wanted, this is not it.** Seeds 14-23 are the only fresh seeds whose
E0 encoders already exist; going past seed 23 requires manufacturing new E0 and is a different,
larger experiment. That trade is recorded here rather than discovered afterwards.

## 2. The arms

**Two arms, seeds 14-23, 20 cells. EXP-056 copied field for field.**

| arm | change | tag |
|---|---|---|
| **B**, full critic | EXP-053 arm B, unchanged | `exp060_full_d7` |
| **F**, flattened | `flatten_critic=True`, the ONLY change | `exp060_flat_d7` |

Base config: depth 7, 10,000 episodes, `CAP = ((1, 2),)`, E1 encoders
`exp047_ft_d6_lr0.0001_regionalized_d6_s{seed}_sig0.0_encoder.pt`.

**Arm B must be re-run, not reused.** EXP-053's arm B exists only at seeds 0-11, and the contrast is
paired by seed. Pairing a new `F` seed against an old `B` seed would compare different seeds and is
not the same measurement.

## 3. Claims

Paired by seed, **exact** permutation - n=10 is 1,024 sign flips, so no sampling is needed.

### Claim 1, PRIMARY - `F` minus `B` on held-out success, seeds 14-23

**Directional.** EXP-056 pre-registered a direction and found it; a replication tests that same
direction rather than re-opening it.

| outcome | reading |
|---|---|
| `delta <= -0.05`, `p <= 0.05` | **REPLICATED.** The EXP-056/057 picture stands. |
| `delta >= +0.05`, `p <= 0.05` | **A significant reversal.** Report in the headline; it would put the critic conclusion in serious doubt. |
| otherwise | a **BOUND with its interval**, never a refutation and never an equivalence |

### Claim 2, SECONDARY - the pooled n=22 contrast

Reported **with the optional-stopping caveat in the same sentence**, every time it is quoted.

### Claim 3, THE VALIDITY GATE - reused from EXP-056 deliberately, and here is why that is safe

**Condition: the ratio of `V`'s within-episode RMS to the returns' within-episode RMS must be
`>= 0.05` at every stage.** If `V` barely varies within an episode there is nothing for flattening
to remove and Claim 1's null is vacuous.

Per the gate-calibration rule, a reused gate needs its regime checked, and this one passes that
check where EXP-057's did not:

- **It is a RATIO, so it is scale-free.** EXP-057's gate failed precisely by being an absolute
  calibrated at depth 3 and evaluated at depth 7.
- **It was measured in THIS regime** - depth 7, this config - where the smallest observed ratio was
  **0.415**, more than **eight times** the threshold.
- **It is satisfiable and it can fail**, which is the pair of checks EXP-058's gate failed.

### Multiplicity

**Two inferential contrasts, Bonferroni 0.025.** Claim 3 is a condition and belongs to no family.
**Note that 0.025 is the exact threshold EXP-056 cleared by 6.4%**, so the replication is held to
the same bar rather than a looser one.

## 4. Cost - corrected DOWNWARD, and the corrections matter

**The standing ~25 h estimate is wrong. It is ~14 h, and one of its three phases is already paid.**

| phase | cells | workers | cost | basis |
|---|---|---|---|---|
| E0 pretraining | - | - | **0 h, ALREADY DONE** | all 24 `exp040_encoder_s*.pt` exist; EXP-059 needed exactly these |
| E1 fine-tune, seeds 14-23 | 10 | 10 | **~6 h** | EXP-047 confirmatory measured **11.95 h for 12 cells at 10 workers = 2 waves**, so one wave is ~5.98 h and 10 cells on 10 workers is one wave |
| RL, both arms | 20 | 10 | **~8 h** | EXP-056 measured **5.72 h for 12 cells at 6 workers = 2 waves**, so 2.86 h per cell at 6 workers; scaled by the measured 10-vs-6-worker per-cell penalty (0.16 vs 0.115 s/step) and rounded to 2 clean waves |

**Three corrections to the old figure:**

1. **E1 is needed for 10 seeds, not 12.** It already exists for seeds 0-13, not 0-11 - EXP-047's
   pilot ran on seeds 12 and 13, and those encoders were kept.
2. **E0 is already manufactured for all 24 seeds**, as a side effect of EXP-059. The ~1.7 h that
   every previous estimate carried is spent.
3. **Worker count is chosen to DIVIDE the cell count.** 10 cells and 20 cells both divide by 10,
   giving clean waves. This is the one place more workers help: not by parallelism alone, but by
   removing a ragged final wave. Per-cell time is *worse* at 10 workers than at 6, and the playbook
   records that; it is still the right choice here because 20 cells on 6 workers is 4 waves.

**These are wave-rounded extrapolations from measured anchors, not measurements.** Read the first
wave before trusting the total - which is exactly the discipline EXP-059 was dispatched without.

## 5. What this cannot answer

- **It cannot rescue a marginal result by pooling.** If the primary is null, the honest summary is
  "one significant result at n=12 and one bound at n=10", not "significant at n=22".
- **Nothing about depths other than 7.**
- **It does not decompose the critic's benefit.** EXP-057 already established that calibration is
  not the mechanism; this only asks whether EXP-056's specific contrast reproduces.
- **It cannot distinguish a smaller-than-reported true effect from no effect**, at n=10. See the
  power table.
