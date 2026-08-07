# EXP-038 design - do the stabilizers help where collapse IS the failure?

**Status: pre-registration. Written 2026-08-07, before any EXP-038 number exists.**
Every threshold below is fixed. If one is edited after data arrives, that edit is the finding
and must be reported as such.

## 1. The question

EXP-032 swept the trainer stabilizers (`entropy_beta`, `normalize_advantages`) at depths 2 and 3
and refuted them. Its Finding 3 is the reason:

> De-collapsing the policy does NOT make it solve cubes. The entropy bonus lowers modal fraction
> by **injecting randomness, not by teaching the policy to read its input**.

But EXP-031 had already established that the depth-3 failure was **not collapse-limited**: a
de-collapsed depth-3 policy still solved essentially nothing, so there was nothing for the
stabilizers to unlock. **EXP-032 tested a collapse fix on a problem whose binding constraint was
not collapse.**

Depth 6 is the opposite case, and two experiments now say so:

| evidence | value |
|---|---|
| EXP-036, depth 6 held-out success | **0.0000** on all 12 seeds |
| EXP-036, depth 6 modal action fraction | **0.975** (uniform floor 0.354) |
| EXP-037, depth 6 at 3x the episodes | **0.0000**, modal **0.982** |

EXP-037 Claim 4 also removed the competing explanation: tripling the episodes at depth 6 moved
nothing off the floor, so **depth 6's failure is not starvation**. The instruments say collapse.

**So: same intervention, different regime, and this time the diagnosis matches the fix.**

## 2. What makes this NOT a re-run of EXP-032

Three things differ, and all three are load-bearing:

1. **The diagnosis.** Depth 3 was not collapse-limited (EXP-031). Depth 6 demonstrably is.
2. **The budget and schedule.** EXP-032 ran 600 episodes with no curriculum. This is 10,000
   episodes across a `(1..d)` curriculum, which is the configuration every result since EXP-034
   is stated in.
3. **The beta scale is re-derived from measurement.** EXP-032's own limitations section says the
   sweep was bounded too low - every trend was still moving at `beta=0.1`. Baseline entropy at
   depth 6 / 10k is 0.204 against 0.452 at depth 3 / 600, so that scale is not assumed to
   transfer. See section 5.

## 3. Arms - 48 runs, every comparator free

| cell | depth | beta | normalize | n | source |
|---|---|---|---|---|---|
| stabilized | 6 | b1 | True | 12 | new |
| stabilized | 6 | b2 | True | 12 | new |
| stabilized | 6 | b3 | True | 12 | new |
| coherence | 5 | b2 | True | 12 | new |
| trained baseline | 6 | 0.0 | False | - | **EXP-036, 0.0000** |
| random floor | 6 | - | - | - | **EXP-036, 0.0008** |
| trained baseline | 5 | 0.0 | False | - | **EXP-036, 0.0396** |
| random floor | 5 | - | - | - | **EXP-036, 0.0000** |

All four comparators are EXP-036 cells at the **same twelve seeds on the same machine**, so they
pair directly per seed. Re-running them would cost 48 runs to re-measure unchanged quantities -
the same reasoning by which EXP-037 declined to re-run a random arm.

Everything else matches EXP-036 exactly: `arm="regionalized"`, `readout="concept"`, frozen brain
at random init, `sigma=0.0`, 10,000 episodes, curriculum `(1..d)` with **equal** stage weights
(EXP-037 established equal is at or near optimal), evaluated on the held-out shell at `cfg.depth`.

### `normalize_advantages` is pinned True, and that is measured rather than assumed

EXP-032 crossed it and found:

- **beta alone does nothing.** At depth 3, `beta=0.1, normalize=False` gives modal 0.932 against
  the baseline's 0.932. Zero movement.
- **normalization alone is actively harmful**, the worst cell in the whole sweep at both depths
  (depth 3: modal 0.987, 11/12 collapsed).
- Only the **conjunction** moves anything: `beta=0.1 + normalize` gives modal 0.824, entropy
  1.253, 3/12 collapsed.

Crossing it again would spend 48 runs re-deriving a result this repo already owns. Pinning it is
therefore evidence-based. **This is a stated limitation, not a hidden one:** EXP-038 cannot speak
about the `normalize=False` half of the plane at depth 6.

### Why depth 5 is in the design

Depth 6 has the strongest motivation and the **worst measurement resolution**: all twelve seeds
sit at exactly 0.0000, so a partial improvement is invisible against a per-seed resolution of
1/200. Depth 5 is also **BROKEN** by EXP-036's rule and substantially collapsed (modal 0.779),
but it sits at **0.0396 +- 0.027**, where an effect of realistic size is measurable.

Depth 5 supplies the statistical power that depth 6 structurally cannot. A null at both closes
the stabilizers far more firmly than a null at depth 6 alone, because the depth-6 null on its own
is confounded with "the effect was below resolution".

## 4. The pre-registered contract

All tests are **exact paired permutation over all 2^12 = 4096 sign flips**, two-sided. No scipy
in the venv, and at n=12 the exact test is cheap and assumption-free.

### Claim 1 (PRIMARY) - is it a lever at depth 6?

Held-out success per seed, paired against **EXP-036's depth-6 RANDOM arm**.

> **CONFIRMED** requires, for at least one depth-6 cell:
> **mean >= 0.02** AND **p <= 0.017**.
> Anything else: **REFUTED at this budget.**

**The comparator is the random arm, not the 0.0000 trained baseline.** This is the whole design
decision of the experiment:

> [!danger] At depth 6 the random floor is ABOVE the trained result
> EXP-036 measured floor **0.0008** against trained **0.0000**. So "did success rise above the
> trained baseline?" is a check that **cannot distinguish the two states it exists to separate** -
> a policy that is merely more random passes it. EXP-032 Finding 3 established that injecting
> randomness is exactly how the entropy bonus operates.
>
> EXP-037 Claim 4's wording ("any seed above zero is worth reporting") has this defect. It is
> superseded here.

Thresholds:
- **0.02** is 25x the measured floor (0.0008), 4x the per-seed resolution (1/200 = 0.005), and
  about half of depth 5's current 0.0396 - clearing it would put depth 6 onto the part of the
  curve where depth 5 already sits.
- **0.017** is Bonferroni over the three depth-6 cells. Testing three betas and reporting the best
  at p <= 0.05 carries a family-wise error near 0.14. The exact test's floor is 2/4096, so 0.017
  is reachable.

### Claim 2 (THE DISCRIMINATOR) - is a gain learning, or just randomness?

> A success gain in a cell whose **modal action fraction has fallen to the uniform floor (0.354)**
> is **randomization, not learning**, and MUST be reported as refuting the lever regardless of
> whether Claim 1's arithmetic passes.

This encodes EXP-032 Finding 3 as a rule rather than as a lesson to be remembered. The
operational bar: a CONFIRMED cell must hold **modal fraction >= 0.45**, comfortably above the
0.354 uniform floor, so the policy is still selecting rather than sampling.

The highest beta `b3` is chosen (section 5) so that it **does** reach near-uniform. That makes it
a **built-in instrument check**: a near-uniform policy must score the random floor, ~0.0008. If
`b3` lands near-uniform and scores materially above the floor, **the measurement is broken and no
other claim in this experiment may be read.**

### Claim 3 - depth 5 coherence

`b2` at depth 5, paired against EXP-036's depth-5 **trained** arm (0.0396).

> **CONFIRMED**: mean delta **>= +0.02** with **p <= 0.05**.
> **REFUTED**: anything else.

+0.02 is about 0.75 sd of the measured 0.0272 and a 50% relative gain. Depth 5's comparator is the
trained arm rather than the random arm because depth 5 **is** above its floor (0.0396 vs 0.0000),
so the question there is genuinely "better than the current policy", not "better than noise".

### Claim 4 - mechanism

Report `greedy_modal_action_frac` and `mean_train_entropy` per cell, plus the dose ordering.

- Does modal fraction fall from 0.975, and **monotonically in beta**?
- **Entropy alone is not sufficient evidence.** EXP-037 Claim 5 found entropy and modal fraction
  rising **together** in the back-loaded arms, where entropy alone would have suggested the arms
  were more exploratory when they were more degenerate. Read modal fraction, and read both.

### Claim 5 - the null is a real result, and is pre-committed as one

Depth 6 was the strongest remaining case for the stabilizers: the one regime where the failure
the intervention targets is the failure the instruments actually diagnose.

> If Claims 1 and 3 both refute, **the trainer stabilizers are CLOSED as a lever** and join width
> (EXP-033), volume alone (EXP-034), curriculum stage weighting (EXP-037) and starvation
> (EXP-037) on the do-not-revisit list. The next move is the encoder (vault Stage 2).

This is written down in advance so that a null cannot later be re-described as "inconclusive,
worth another sweep".

## 5. Choosing b1, b2, b3 - measured, not guessed

EXP-032's single largest limitation was that its sweep was **bounded too low**: every metric was
still moving at the boundary, so it could not say what happened past it. Guessing again risks
repeating exactly that.

A **pilot** runs first: depth 6, **1,000 episodes**, cells `(beta, normalize)` =
`(0.0, False)`, `(0.0, True)`, `(0.05, True)`, `(0.2, True)`, `(0.8, True)`, two seeds each,
10 runs, about 25 minutes at 10 workers. Its records live **outside the repo** and it produces no
pre-registered claim.

**How it is read - the shift, not the absolute values.** 1,000 episodes under-converges relative
to 10,000, so the policy is less collapsed there and every modal fraction reads low. The
`(0.0, False)` cell is EXP-036's exact configuration, so the distance between it and EXP-036's
measured 0.975 **is** the under-convergence offset, and the dose axis is read relative to it.

Selection rule, fixed in advance:

- **b3** is the smallest piloted beta whose modal fraction approaches the uniform floor (0.354),
  so the top of the sweep demonstrably saturates and Claim 2's instrument check has teeth.
- **b1** is near the bottom of the range that moves modal fraction measurably off the
  `(0.0, True)` anchor.
- **b2** sits between them, and is the value the depth-5 coherence arm uses.

If **no piloted beta approaches uniform**, the span is still too low and must be raised before
dispatch. The pilot has then done its job by saying so before 21 hours were spent.

## 6. Cost

EXP-037 measured 4.62M env steps in 20.5 h at 10 workers, about **62 steps/s**.

| | steps/run | runs | steps |
|---|---|---|---|
| depth 6, equal split | 166 x (5+7+9+11+13+15) = 99,960 | 36 | 3.60M |
| depth 5, equal split | 2,000 x (5+7+9+11+13) = 90,000 | 12 | 1.08M |
| **total** | | **48** | **4.68M** |

**About 21 h**, matching EXP-037's 20.5 h for the same run count. One overnight.

Ten workers, not sixteen: EXP-036 ran 16 at ~920 MB private each, drove system commit to 48.6 of
50.4 GB and held utilisation at a measured 43.1%. Ten measured 74.2%.

## 7. Constraints on the implementation

- **`src/neuromorphic/training/cube_baseline.py` must not change.** Every knob EXP-038 needs
  already exists (`entropy_beta`, `normalize_advantages`, `curriculum`). If trainer code does
  change, a `beta=0, normalize=False` depth-6 replication cell becomes **mandatory**, because
  without one nothing verifies the change was neutral. As long as nothing changes, that cell is a
  byte-identical duplicate of EXP-036 and would buy nothing.
- **`cell_tag` must encode beta and normalize.** `record_filename` encodes tag/arm/depth/seed/
  sigma and **not** `entropy_beta` or `normalize_advantages`. Without it the 48 runs collapse into
  24 files, each holding whichever cell finished last, silently. This has bitten EXP-032 and
  EXP-037 by design and the driver carries an explicit collision guard.
- **`aggregate.py` must print the ORDERING, not only the single comparison each rule names.**
  EXP-037's aggregator printed "INTERIOR OPTIMUM" and "CONTROL HOLDS" on a result that supported
  neither, because the rules were written expecting the other outcome. Every verdict here must be
  printed next to the dose table it was derived from.
- **No assertion that cannot fail.** Any test added for this experiment must fail against the
  pre-change code.

## 8. What this experiment cannot say

- Nothing about `normalize_advantages=False` at depth 6 (pinned; section 3).
- Nothing about betas outside the piloted span.
- Nothing at budgets other than 10,000 episodes. EXP-036 Claim 5 stands: no statement of the form
  "the architecture cannot do depth N" is supported, because depth 3 was still climbing between
  10k and 30k in EXP-035.
- Nothing about curricula other than `(1..d)` with equal weights.
- Nothing about the encoder, which both EXP-032 and ADR 0001 independently point at as the wall.
