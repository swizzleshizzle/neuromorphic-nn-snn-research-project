# EXP-075: a wider sensory region for the learned judge

**Written 2026-10-09, week 28.** Michael chose the first option in the EXP-074 handoff: a wider
sensory region for the judge. EXP-074 (`experiments/074_wide_judge/RESULTS.md`) confirmed that a
value-iteration judge reading all 192 sensory neurons beats both the best policy recipe and the
same judge reading only the 64-unit concept. This experiment asks whether a wider region helps
again. Roadmap: `docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md`, stage 3.

## 1. Why: the readout is the cap, measured before this spec

EXP-074's best judge (J3V-W) solves 36/200 at depth 9, seed 0. A judge fitted on TRUE distances
over raw facelets solves 88/200 on the same cell. Two things differ between them: the
representation (192 spiking mean rates vs raw one-hot facelets) and the training signal (value
iteration vs true distances). Diagnostics run 2026-10-09 separated them. Every row below is the
same 300,000 random-walk training states (every seed 0-11 evaluation held-out state excluded),
the same `Linear(n,128)-ReLU-Linear(128,1)` head fitted on true distance for 8 epochs, Gate L's
margin on seed 0's probe set, and the J3V search on depth 9, seed 0. Script:
`experiments/074_wide_judge/diagnostics/oracle_wide.py`.

| readout (frozen, head fitted on true distance) | width | Gate L margin | J3V d9 s0 |
|---|---|---|---|
| raw one-hot facelets (not spiking; the ceiling) | 144 | 0.414 | **88/200** |
| frozen E1 region, concept + hidden, 1 Poisson draw | 192 | 0.206 | 33/200 |
| frozen E1 region, concept + hidden, 8 draws averaged | 192 | 0.150 | 34/200 |
| random-init region, hidden 128 | 192 | 0.150 | 30/200 |
| random-init region, hidden 512 | 576 | 0.194 | **44/200** |
| *for reference: EXP-074's VI-trained W judge, seed 0* | 192 | 0.206 | 36/200 |

Read as follows. One seed and one cell, so these are directions, not effects (binomial se about
3 solves per cell):

1. **The headroom is in the readout, not the training signal.** A head fitted on TRUE distance
   over the FROZEN E1 192 features scores 33/200 and margin 0.206. The VI-trained W judge, whose
   encoder also trained, scores 36/200 and margin 0.206. A supervised fit with a trainable encoder
   was not measured. So this shows that the frozen 192 readout caps even a supervised head at
   about where value iteration already is. It does not show that no training signal could do
   better.
2. **Poisson noise is not the bottleneck.** Averaging 8 draws gives 34/200, not more.
3. **Width helps, even without pretraining.** A random region at hidden 512 beats one at 128:
   44/200 vs 30/200, margin 0.194 vs 0.150.
4. **Pretraining helps at fixed width.** At hidden 128, E1 (inverse-model pretrained, then
   fine-tuned) has margin 0.206 vs 0.150 for random init.

So the experiment combines the two levers: a region pretrained by the same inverse-model recipe,
at hidden 512, then trained as a judge by value iteration.

**Standing record against width.** EXP-033 Finding 1 refuted concept width for a frozen random
encoder read by a linear probe (concept@512 reached 0.638 against facelets 0.766, with doublings
saturating). This case differs in three ways: the region trains, the readout includes the hidden
layer, and the consumer is a search, where row 3 above is a direct measurement.

## 2. Disclosure

The diagnostics in section 1 ran J3V on depth 9, seed 0, an evaluation cell. That cell was already
disclosed in EXP-074 spec section 2. They also used seed 0's probe set, which Gate L uses. No
diagnostic ran a VI-trained judge of the new width, and no other evaluation cell was looked at.
**Every primary therefore keeps EXP-074's 10-seed sensitivity line (seeds 0 and 3 dropped)**,
printed by the aggregator next to the 12-seed verdict. If the two lines disagree in verdict,
RESULTS.md leads with the disagreement.

## 3. The judges

Every judge has the EXP-074 W readout: concept mean rate concatenated with hidden mean rate,
from one forward pass. The head is `Linear(n_in,128)-ReLU-Linear(128,1)-softplus`, lr 1e-3. The
encoder trains at lr 1e-4. The concept stays at 64 units.

| arm | sensory region | where it starts | readout |
|---|---|---|---|
| **X** (new) | `SensoryCortex(144 -> 512 -> 64)` | inverse-model pretrained at hidden 512 (section 4) | 64 + 512 = **576** |
| **Y** (new control) | `SensoryCortex(144 -> 128 -> 64)` | inverse-model pretrained at hidden 128, same recipe, same machine as X | 64 + 128 = 192 |
| **W** (EXP-074, reused) | `SensoryCortex(144 -> 128 -> 64)` | the seed's E1 (EXP-039 pretraining, then EXP-047 fine-tune) | 192 |

- **X vs Y** isolates width. Pretraining recipe, machine, VI training and readout form are all
  held fixed.
- **X vs W** asks whether X beats the best judge so far. It confounds width with E1's provenance:
  E1 was fine-tuned by RL after pretraining (about 3% of its norm) and made in a separate run.
- W is not retrained. Its 12 committed judges (`experiments/074_wide_judge/outputs/judge_W_s*`)
  and evaluation records are reused, which Gate 0(b) checks.
- No parameter matching (CLAUDE.md, EXP-064). X's head has 73,985 parameters against 24,833 for
  W and Y. That difference is part of the intervention.
- The policy brain is untouched. The judge's region is its own module, so the policy path, the
  agents, the heads and the real-state stream are those of EXP-074.

## 4. Pretraining (X and Y)

EXP-039's recipe, unchanged except for two things: the hidden width, and a forbidden set that also
includes the seed's evaluation held-out states.

- Self-supervised inverse model: name the move between two states. No distance labels.
- 40 epochs, batch 256, lr 3e-3.
- State pairs from depths 1 to 6, built by `build_pairs`. EXP-039's probe held-out states are
  forbidden, **plus every state in `exclusion_set(seed)`**. EXP-039's pairs could reach depth 7 as
  successors, and some evaluation states are at depth 7. Unlabelled, but excluded anyway.
- The inverse head reads the concept (64), as in EXP-039.

`make_sensory` gains a `hidden` argument whose default (128) leaves every existing caller and its
init byte-identical. A test pins that. Pretraining runs on the laptop: 24 runs, plus 4 pilot runs.
EXP-039 measured 1000 s per seed at hidden 128 on the VPS.

## 5. Value-iteration training (fixed now; EXP-074 section 4, unchanged)

- `N = 1000` positions per update, from random walks of length uniform in 1..14, rejecting the
  seed's evaluation held-out sets (depths 7, 8, 9, 11).
- `sync_every = 100`, `jt_draws = 1`, `probe_every = 250`, **4000 updates**.
- 24 runs: arms X and Y, seeds 0 to 11.

**Cost, measured on the VPS 2026-10-09 (1 thread):**

| hidden | s per update | peak RSS |
|---|---|---|
| 128 | 1.2 | 0.84 GB |
| 512 | 2.9 (2.4x) | 2.1 GB |

EXP-074's W took 16,900 s per run at 12 laptop workers, so X is likely about 40,000 s (11 h) per
run. At about 2.1 GB per worker, 12 X workers would not fit in the laptop's 31.4 GB alongside
anything else. The pilot measures laptop RSS and throughput, and the amendment (section 6) sets
the worker count. Expect training to take about one to one and a half days of laptop time.
Checkpoints are resumable and bank every 250 updates, so the 03:31 Windows Update reboot costs at
most one probe interval per run.

## 6. Pilot (seeds 12 and 13 only)

- Pretrain X and Y on seeds 12 and 13: 4 pretraining runs.
- Train X and Y by value iteration at section 5's settings: 4 runs.
- Measure: Gate P, Gate L margin, Gate E, s per update, and per-worker working set.

**A dated amendment then fixes**, before any seed 0-11 runs:

- X's and Y's Gate P and Gate L thresholds (forms fixed in section 7);
- the training worker count, keeping total peak working set under 24 GB;
- the schedule.

The width is NOT a pilot decision. It stays at 512.

**Y must be a working control (CLAUDE.md, EXP-064).** The amendment compares Y's mean pilot Gate L
margin with W's mean pilot margin, 0.2037 (EXP-074 section 11). If Y's is below W's Gate L
threshold, 0.1018, laptop pretraining is not producing a region of E1's grade. Then stop and
report rather than amend, because Claim 2 would otherwise be tested against a weak control.

## 7. Gates

**Gate P (pretraining worked, X and Y).** Every seed's final pretraining move accuracy must be at
least **0.9 times that arm's mean pilot accuracy**. The amendment states both numbers. The
threshold comes from each arm's own pilot because EXP-039's range (0.449 to 0.457 over seeds 0-11)
was measured at hidden 128 only, and the CLAUDE.md gate-calibration rule forbids carrying it to 512.
Chance is 1/6 and the maximum is 1, so the gate can pass. If either arm's mean pilot accuracy is
below 0.30, the pilot has failed: stop and report, with no amendment. A failure voids that arm's
claims.

**Gate L (VI worked, X and Y).** EXP-074 section 6's instrument, unchanged: on the seed's 250
probe states at true distances 7 to 11, the margin is the 3-move lowest-J leaf hit rate minus
chance. It passes when, over the 12 seeds:

- (a) the per-seed margin exceeds 0 by an exact one-sided sign-flip test, p < 0.05; and
- (b) the mean margin is at least **half that arm's mean pilot margin**.

Gate L is a floor check, not a predictor of X vs W. In EXP-074, W and A had equal depth-9 leaf
rates while W solved 0.064 more.

**Gate E (encoders trained).** X and Y: drift from the seed's own pretrained encoder is > 0 on
every seed.

**Gate R (leaf ranking at evaluation scale).** As in EXP-074: per arm and depth, on the evaluation
held-out states, above chance at p < 0.05.

**Gate 0.**
- (a) Determinism: J3V-X at depth 9, seed 0, re-run into a separate directory. The result must be
  identical except `wall_s` and `git_commit`.
- (b) Continuity: J3V-W at seed 0, depths 9 and 11, re-evaluated on the laptop from its committed
  judge, must equal EXP-074's records in every outcome field (36/200 at depth 9). Depth 11 is
  checked separately because it takes a different path: the depth-9 agent and a 25-move budget.
  This is what licenses reusing W's records for Claims 1 and 3.

**Gate 1 (resolution).** A contrast is UNRESOLVED if both arms are below 0.02 or above 0.98. The
verdict function returns UNRESOLVED as a verdict, using EXP-070's `gate1_verdict`.

## 8. Evaluation

Held-out sets, budgets, stream discipline, no-revisit rule, the 3-move goal-tested tree, and the
state-reading critic seam are all exactly EXP-074 section 7's. Depth 11 uses the seed's depth-9
policy agent, head and train seed, with a 25-move budget, as in EXP-074.

| arm | source | depths |
|---|---|---|
| **J3V-X**, **J3V-Y** | new cells | 7, 8, 9, 11 |
| **J3V-W** | EXP-074 records (Gate 0(b) re-check) | 7, 8, 9, 11 |
| P3V, R3V | EXP-071/072/074 records, as cited in EXP-074 RESULTS | 7, 8, 9, 11 |

96 new cells (2 arms x 4 depths x 12 seeds), plus the Gate 0 cells. 12 seeds, paired across every
arm.

## 9. Pre-registered claims (one-sided exact sign-flip, 4096 flips; alpha 0.0167 each, 0.05 / 3)

**Claim 1 (primary): the wider region beats the best judge so far at depth 9.**
mean(J3V-X - J3V-W) > 0.

**Claim 2 (primary): width is what helps, pretraining held fixed, depth 9.**
mean(J3V-X - J3V-Y) > 0.

**Claim 3 (primary): the wider region beats the best judge so far on the typical random scramble.**
mean(J3V-X - J3V-W) > 0 at depth 11. EXP-074 left depth 11 exploratory. It is promoted here because
it is the real goal and it is resolvable: J3V-W solved 0.101 at depth 11, and the measured paired
sd of J3V-W against P3V there was 0.038.

Verdicts: CONFIRMED, REFUTED (mean <= 0), NOT SIGNIFICANT, UNRESOLVED (Gate 1), VOID (Gate 0, or
Gate P, L, E or R for an arm in the contrast). Each primary also prints its 10-seed sensitivity
line (section 2).

**Secondary (patterns, never confirmations):**
- Claims 1 and 2 at depths 7 and 8.
- Claim 2 at depth 11.
- Y vs W at depth 9: does E1's provenance (its RL fine-tune) matter?
- The X - Y leaf-rate difference (Gate R) as the mechanism line.

**What the primaries can detect.** EXP-074's paired per-seed sd between judges was 0.0375 at
depth 9 and 0.038 at depth 11, so se is about 0.011 at n = 12. At alpha 0.0167 a true difference of
about 0.03 is detectable. Section 1's random-init width effect was +0.07 on one cell.

## 10. Order

1. Build and test (VPS):
   - `hidden` in `make_sensory`, with the default byte-identical;
   - the EXP-075 pretraining driver, with the extended forbidden set;
   - judge construction from a pretrained region;
   - drift against the seed's pretrained encoder;
   - the training and evaluation drivers, reusing EXP-074's;
   - the aggregator, with all three primaries, their sensitivity lines and Gate 1 in the verdict;
   - a launcher with EXP-074's `.Handle` fix.
2. Pilot on the laptop: pretraining, then value iteration, seeds 12 and 13.
3. Dated amendment: Gate L thresholds, worker count, schedule. Shown to Michael and committed
   before step 4.
4. Pretrain the 24 seed 0-11 encoders. Gate P is committed before any VI run starts.
5. Train the 24 VI runs. Gates L and E are committed before any evaluation cell runs.
6. Evaluation: Gate 0(b) first, then every J3V-X and J3V-Y cell, then Gate 0(a).
7. Aggregate. Commit RESULTS.md, the records, the pretrained encoders and the trained judges.

## 11. Not in scope

- Widths other than 512.
- A wider concept.
- A non-spiking judge on raw facelets. It is the ceiling in section 1, never an arm.
- A deeper tree or a learned no-revisit rule.
- Training the policy from J.
- A learned world model.
- 3x3.
