# EXP-054 design - is the concept sequence-blind, and does that explain the pretraining collapse?

> **PRE-REGISTERED. Committed before any number exists.** Thresholds fixed at commit time.
> **Date:** 2026-08-29 · **Phase:** 3 · **Grounds:** `docs/handoffs/SESSION-HANDOFF-2026-08-27.md`
> open item 3, EXP-039, EXP-050, EXP-052.

## 0. The gap this closes

**A hypothesis has been leaned on four times and never measured.** The encoder is pretrained on
an inverse model: predict the move from a state pair, `CrossEntropy(head([rate(concept(s)),
rate(concept(s'))]), a)`. That objective is **purely single-step**. The recurring explanation for
why more pretraining hurts the policy is that over-training it produces a code good at "which move
just happened" and bad at "how far along a sequence we are" - **sequence-blindness**.

It now has four consistent non-significant trends behind it (EXP-050, EXP-052) and zero direct
evidence. The handoff asks for **a metric that does not depend on solving**, and names the shape:
whether the concept distinguishes a state one move away from one two moves away.

**Why it is worth a session.** The pretraining series is a paradox nobody has explained:

| epochs | move-accuracy | depth-6 policy |
|---|---|---|
| 0 | - | **0.0000** |
| 10 | 0.383 | **0.2012** |
| 20 | 0.414 | 0.1850 |
| 40 | 0.437 | 0.1800 |
| 80 | **0.452** | **0.0887** |

**The pretext metric climbs monotonically while the policy halves.** EXP-052 established that and
could not say why. If sequence-sensitivity falls across the same series, the hand-wave becomes a
measured fact and the collapse acquires a mechanism.

## 1. The metric, and why it trains nothing

For one encoder:

1. Build exact BFS shells with `ExactBFSDistance(max_depth=6)` - a bounded build is near free
   (11,913 states, about 0.04s). `shell_states(provider, d)` gives every state at exact distance
   `d`.
2. Sample up to `N_PER_SHELL = 60` states per shell, `d = 1..6`, with a fixed RNG.
3. Encode each state and take the concept rate, giving a 64-vector per state.
4. **Centre**: subtract the grand mean concept over all sampled states.
5. `sim(d1, d2)` = mean cosine over all cross-pairs from shell `d1` and shell `d2`.
6. **Sequence-sensitivity `S`** = the negated least-squares slope of `sim` against `|d1 - d2|`
   over all pairs with `d1 <= d2`. Higher `S` means similarity falls off faster with separation,
   i.e. the code distinguishes shells.

**Nothing is trained.** No classifier, no probe, no fitted parameters other than a slope over
already-computed similarities. This is deliberate: **every instrument this project has retired was
a trained linear probe**, and a metric with no capacity cannot overfit its way into a story.

> [!important] CENTRING IS LOAD-BEARING, NOT COSMETIC
> Concept vectors are **firing rates and therefore non-negative**, so raw cosine between any two
> of them is compressed near 1 and is dominated by overall activity rather than structure. The
> uncentred version would report "everything is similar to everything" for every encoder and
> would look like a clean null. The grand mean is subtracted over the whole sampled set, once,
> before any similarity is computed.

**Encoding is stochastic** (Poisson spiking), so every measurement uses a fixed `torch.Generator`
seeded from the encoder's seed. Re-running an encoder must reproduce its `S` exactly; the driver
asserts this on one cell.

**The shallow shells are small and that is a real limit.** Depth 1 has **6** states and depth 2
has **27**, against 8,969 at depth 6. Their per-shell means are noisy, `N_PER_SHELL` cannot rescue
them, and no claim rests on a single shell pair.

## 2. Design

**Five arms, 12 seeds each, 60 encoders. All already on disk. Nothing is trained and nothing runs
on the laptop.**

| arm | epochs | encoders | policy |
|---|---|---|---|
| E0 | **0** | rebuilt from `encoder_seed` (random init, no file) | 0.0000 |
| E10 | 10 | `experiments/052_pretraining_optimum/outputs/exp052_encoder_e10_s*.pt` | 0.2012 |
| E20 | 20 | `experiments/052_pretraining_optimum/outputs/exp052_encoder_e20_s*.pt` | 0.1850 |
| E40 | 40 | `experiments/040_pretrained_encoder_policy/outputs/exp040_encoder_s*.pt` | 0.1800 |
| E80 | 80 | `experiments/050_objective_vs_gradient/outputs/exp050_encoder_plus_s*.pt` | 0.0887 |

**The 0-epoch arm is mandatory**, per the standing rule to measure the floor rather than assume
it. It is reconstructed from `encoder_seed` rather than loaded, because a random init is exactly
reproducible and no file was ever saved.

## 3. The prediction, and the trap named in advance

> [!warning] A RANDOM ENCODER MAY SCORE HIGHEST, AND THAT IS NOT A BUG
> Random projections tend to **preserve** geometry (Johnson-Lindenstrauss). A randomly-initialised
> encoder may separate shells better than any trained one, while scoring **0.0000** policy.
>
> If so, the shape is **sequence-sensitivity monotone in epochs while policy is an inverted U**:
>
> | epochs | 0 | 10 | 20 | 40 | 80 |
> |---|---|---|---|---|---|
> | policy | 0.0000 | **0.2012** | 0.1850 | 0.1800 | 0.0887 |
> | `S`, if the hypothesis holds | highest | | | | lowest |
>
> **That is the result, not a failure of it.** It would say the 10-epoch optimum is a **tradeoff**:
> pretraining buys move-structure and spends sequence-structure, from the very first epoch, and the
> policy peak is where the two curves cross.
>
> This reading is written down **now** so that nobody discovers it afterwards and reports it as a
> prediction. That is precisely the EXP-050 Claim 4 failure, where a satisfied condition carried a
> wrong inference.

**`S` is therefore NOT expected to predict policy on its own, and Claim 1 does not ask it to.**

## 4. Claims

Paired per seed, exact permutation over all `2**12` sign flips (4096; no scipy in the venv).

### Claim 1 - PRIMARY. Does sequence-sensitivity fall as pretraining continues?

Three adjacent contrasts, paired by seed: **10 vs 20**, **20 vs 40**, **40 vs 80**.

**CONFIRMED if `S` decreases in at least two of the three at `p <= 0.05`.**

Two of three rather than all three, because EXP-052 showed 10/20/40 are statistically
indistinguishable in policy terms (p 0.49 to 0.84) and there is no reason to demand a resolution
the series may not contain. Requiring all three would make the claim fail on noise; requiring one
would let noise confirm it.

### Claim 2 - THE TRADEOFF. `S` against the pretext metric.

Descriptive, no bar. Report `S` beside move-accuracy (0.383 / 0.414 / 0.437 / 0.452) and beside
policy. **If `S` falls while move-accuracy rises, the four-experiment hand-wave is confirmed as a
measured fact** and the pretraining collapse has a named mechanism.

### Claim 3 - THE FLOOR. The 0-epoch arm.

Descriptive, no bar. Report E0's `S` against E10's, paired by seed.

**If E0 is at or above E10, state plainly that pretraining degrades sequence structure from the
first epoch**, and that no amount of it is protective. Do not soften this into "early pretraining
preserves structure" if the number does not say so.

### Claim 4 - THE DISQUALIFIER. This claim governs whether `S` may ever be used again.

Compute Spearman correlation between `S` and per-seed held-out policy success:

- **within** each trained arm (12 seeds, four separate correlations), and
- **between** arms (the four arm means).

**If the within-arm and between-arm correlations carry OPPOSITE SIGNS, `S` is reported as a fifth
inverted instrument and is retired on the spot.** It may not be used as a diagnostic, may not
appear in a later spec, and the write-up must say so in its headline rather than in a caveat.

E0 is excluded from this correlation: every seed scores exactly 0.0000, so there is no variance to
correlate against. It still counts for Claim 3.

> [!danger] WHY THIS IS A HARD RULE AND NOT A CAVEAT
> This is exactly how entropy behaved: **Spearman +0.881 with success within EXP-044 arm A, and
> the opposite sign between arms**, where the better arm had the LOWER entropy. The EXP-033 probe
> did the same thing twice, at p 0.0005 in both directions.
>
> Each was reported with caveats, and the caveats did not stop EXP-033, EXP-039 and EXP-047 from
> building inferences on the probe anyway. **A rule that fires automatically is the only kind that
> survives contact with a result you like.**

## 5. Implementation

- **Library**: `src/neuromorphic/analysis/sequence_sensitivity.py` - shell sampling, centring,
  the similarity matrix, and the slope. Pure functions over an encoder and a provider, so the
  statistic is testable without touching a driver.
- **Driver**: `experiments/054_sequence_blindness/run.py` - iterates the five arms, writes one
  JSON record per encoder.
- **Aggregator**: `experiments/054_sequence_blindness/aggregate.py` - Claims 1 to 4, with Claim
  4's sign test as a function, not prose, following EXP-053's `claim3_verdict` precedent.

### Tests, to the repo's strength rule

Each must fail against a broken implementation:

1. **A synthetic shell-structured encoder scores high.** Construct a fake encoder whose output is
   a known function of shell index; assert `S` exceeds a measured threshold. Prototype first, then
   set the bar with margin.
2. **A shell-blind encoder scores about zero.** Construct one whose output ignores the state;
   assert `|S|` is below a measured threshold. This is the test that fails if centring is dropped.
3. **A shuffled-label null.** Recompute `S` with shell labels permuted across states; assert it
   collapses toward zero. Guards against `S` measuring sampling artefacts rather than structure.
4. **Determinism.** The same encoder and seed reproduce `S` exactly.

## 6. What would refute the whole thing

- **Claim 1 flat across all three contrasts.** Sequence structure does not change with pretraining,
  the four-experiment hypothesis is wrong, and EXP-052's collapse needs a different explanation.
- **Claim 4 trips.** `S` is inverted and joins the retired list. The experiment then produces a
  negative methodological result, which is worth reporting precisely because three instruments got
  here before it.
- **E0 scores near zero.** Would mean a random encoder has no shell structure at all, contradicting
  the Johnson-Lindenstrauss expectation, and would make the whole metric suspect rather than the
  encoders.

## 7. Deliberately not in this experiment

- **Any trained probe**, including the gap classifier that was the obvious alternative. The
  no-capacity version was chosen precisely because of the retirement history.
- **Random k-move walks** as a way to build state pairs. On a 2x2 a move and its inverse cancel and
  faces have order 4, so a k-walk does not land k moves away - the same trap EXP-041/042 found at
  depth 1. Exact BFS shells avoid it entirely.
- **Any policy run.** This is offline analysis over encoders already on disk; it must not compete
  with EXP-053's arms for the laptop.
- **The EXP-033 probe.** Retired.
