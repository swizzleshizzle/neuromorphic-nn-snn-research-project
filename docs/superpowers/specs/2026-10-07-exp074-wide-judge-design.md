# EXP-074: a learned judge that reads the whole sensory region, with a training gate that measures what search uses

**Written 2026-10-07, week 27.** Michael chose option 2 after EXP-073's pilots (issue #14): a judge
that reads all 192 sensory neurons, with EXP-073's 64-unit judge kept as its control, and a training
gate redesigned so it measures what the search actually needs. EXP-073 is superseded by this spec;
no EXP-073 evaluation seed ever trained. Roadmap:
`docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md`, stage 3.

## 1. Why

EXP-073's judge (spec `2026-10-07-exp073-learned-judge-design.md`, sections 10 and 11) trained by
value iteration from its own look-ahead, never from answers. Both pilots predicted its training gate
(Gate T: Spearman over true distances 7 to 11 >= 0.30, and mean J strictly increasing 7 to 11) would
fail for both arms. Diagnostics afterwards (all throwaway, scripts in
`experiments/074_wide_judge/diagnostics/`) found:

1. **The 64-unit concept is the cap, not the training procedure.** A head fitted directly on TRUE
   distances (an instrument, never the method) over the frozen E1 concept reaches spearman_7_11
   **0.260** (500k states); value iteration reached 0.289 (A) and 0.257 (B). Reading concept AND
   hidden layer (192 units) reaches **0.341 to 0.370**; raw one-hot facelets **0.388 to 0.399**.
   Seed 12, 250 probe points, se about 0.06: the 64 vs 192 gap is real, differences inside the
   second group are not.
2. **Gate T(b) fails even for those fitted judges** (mean J at 10 above mean J at 11 for the
   192-unit and raw 512x512 fits). With 50 probe states per distance a strict-increase check at the
   top of the range is close to a coin flip. It could not pass at the ceiling.
3. **spearman_7_11 does not predict search value.** A raw-facelet judge fitted on true distances,
   trained excluding every seed 0-11 evaluation state, has spearman_7_11 **0.371** yet drives the
   3-move search (J3V) to **88/200** at depth 9, seed 0, against P3V's published **5/200**.

So Gate T guarded the wrong quantity, and the 64-unit readout is a measured bottleneck. This
experiment widens the readout and gates on leaf ranking, which is what J3V consumes.

## 2. Disclosure: evaluation cells were looked at before this spec

While diagnosing EXP-073, EXP-073 PILOT judges (seed 12, never an evaluation seed, no answers in
training) were run through J3V on two EVALUATION cells at depth 9, published P3V for comparison:

| cell | P3V (published) | pilot judge B_s12 | pilot judge A_s12 | raw judge fitted on true distances |
|---|---|---|---|---|
| depth 9, seed 0 | 5 | 10 | 24 | 88 |
| depth 9, seed 3 | 26 | 12 | 30 | not run |

The pilot judge's training excluded only seed 12's held-out sets, so some of these evaluation states
may have appeared in its random walks (unlabelled). These numbers motivated replacing Gate T, which
is an outcome-informed change. **Therefore every primary is reported twice: on all 12 seeds (the
verdict) and on the 10 seeds excluding 0 and 3 (a sensitivity line, printed by the aggregator next to
the verdict).** If the two disagree in verdict, RESULTS.md must lead with that disagreement.

## 3. The judges

All three start from the seed's E1 encoder and use the same head shape and value-iteration training
as EXP-073 (section 3 of its spec), unchanged except for the readout:

| arm | readout into the head | encoder | head |
|---|---|---|---|
| **W** (new) | concept mean rate (64) concatenated with hidden-layer mean rate (128) = **192** | trains (lr 1e-4) | `Linear(192,128)-ReLU-Linear(128,1)-softplus`, lr 1e-3 |
| **A** (EXP-073 arm A) | concept mean rate (64) | trains (lr 1e-4) | `Linear(64,128)-ReLU-Linear(128,1)-softplus`, lr 1e-3 |
| **B** (EXP-073 arm B) | concept mean rate (64) | frozen | as A |

The hidden mean rate is taken from the same forward pass that produces the concept (the same
spikes), so W sees strictly more of one computation, not a second noisy encoding. W's head has 24,833
parameters against A's 8,449: W differs from A in both readout width and head input size, which is
the intervention. No parameter matching (CLAUDE.md, EXP-064: matching can cost competence).

**The concept path must stay byte-identical to EXP-073** so arm A and B here ARE EXP-073's arms: a
judge built with readout "concept" must reproduce EXP-073 pilot 1's record for (A, seed 12) when
re-run on the same machine (Gate 0(c)).

## 4. Training (fixed now; the EXP-073 pilots ran exactly these for A and B)

`N = 1000` positions per update from random walks of length uniform in 1..14, rejecting the seed's
evaluation held-out sets (depths 7, 8, 9, 11); `sync_every = 100`; `jt_draws = 1` (pilot 2 showed 4
draws lift values but not ordering, at 3.2x cost); `probe_every = 250`; **4000 updates**. 36 runs
(3 arms x seeds 0-11) on the laptop. Pilot throughput 1.6 to 1.7 s/update at 4 workers; at 20 workers
assume up to 3x slower per run (measured for eval cells, not training), so about 36 x 4000 x 5 s / 20
= **10 h**, an upper estimate.

## 5. Pilot (W only, seeds 12 and 13)

A and B need no new pilot: EXP-073 pilot 1 ran them at exactly section 4's settings. W runs at the
same settings on seeds 12 and 13. The pilot measures W's Gate L margin (section 6) and throughput,
and checks Gate E on W. **A dated amendment then sets W's Gate L threshold** (the form is fixed
below) before any seed 0-11 trains.

**Amendment 2026-10-07 (plan writing, before any pilot or seed 0-11 run):** the pilot also re-runs
A and B on seeds 12 and 13 (6 runs, about 2 h at 6 workers). That gives Gate 0(c) on both control
arms and lets every arm's Gate L threshold come from the PRODUCTION instrument rather than the
diagnostic script; the section 6 calibration table is superseded wherever the two differ.

## 6. Gates

**Gate L (training worked, in the quantity search uses), replaces EXP-073's Gate T. Per arm.**
On the seed's PROBE states at true distances 7 to 11 (250 states, the EXP-073 probe set, disjoint
from every evaluation held-out set): build the 3-move tree (216 leaves), take the leaf with the
lowest J, record a hit if its true distance is below the root's. Chance per state is the fraction of
its 216 leaves that are closer. The **margin** is mean hit minus mean chance over the 250 states. The
BFS table is the yardstick only. Gate L passes when, over the 12 seeds:
(a) the per-seed margin exceeds 0 by an exact one-sided sign-flip test, p < 0.05; and
(b) the mean margin is at least **half that arm's mean pilot margin**.

Calibration on the EXP-073 pilot judges (pilot seeds only), measured 2026-10-07:

| arm | margin s12 | margin s13 | mean | threshold (b) |
|---|---|---|---|---|
| A | 0.148 | 0.143 | 0.146 | **0.073** |
| B | 0.104 | 0.043 | 0.074 | **0.037** |
| W | pilot | pilot | pilot | set by the amendment |

Maximum attainable margin is 1 minus chance (about 0.75), so (b) can pass. Both pilot arms clear
their own (b) threshold by 2x. A failed Gate L voids that arm's claims.

**Gate E (encoders trained or frozen as designed).** A and W: encoder drift > 0 on every seed. B:
bit-identical to E1 on every seed. A failure voids every claim involving that arm.

**Gate R (leaf ranking at evaluation scale)**, as EXP-073 section 6, per arm and depth, on
evaluation held-out states; chance at depth 7 0.1344, 8 0.1477, 9 0.1750, 11 0.5209. The policy's
own leaf rate (P3) at depth 9 is recorded first as a reference.

**Gate 0.** (a) Determinism: one J3V-W and one J3V-A cell (seed 0, depth 9) re-run, identical
except `wall_s` and `git_commit`. (b) Continuity: P3V at seed 0, depths 8 and 9, re-run, equals
EXP-072's records in all 8 outcome fields. (c) Concept-path identity: re-running EXP-073 pilot 1's
(A, seed 12) for its full 4000 updates on the laptop reproduces its record's probe history exactly.

**Gate 1 (resolution).** A contrast is UNRESOLVED if both arms are below 0.02 or above 0.98; the
aggregator returns UNRESOLVED as a verdict, not a warning.

## 7. Evaluation

Held-out sets, budget, stream discipline, no-revisit rule and the 3-move goal-tested tree exactly as
EXP-073 section 5. **J arms call the judge directly on the imagined leaf states** (one batched call
through the judge's own encoder and readout) instead of reading `agent.step`'s concept; a W judge
cannot be read through the concept alone. This needs a small seam in `lookahead.py`: a critic that
declares it reads states is called as `critic(states, generator)`; every existing caller is
unchanged.

| arm | what it is | depths |
|---|---|---|
| **J3V-W, J3V-A, J3V-B** | 3-move tree, goal test, lowest-J leaf, no-revisit | 7, 8, 9, 11 |
| **P3V** | best recipe so far; EXP-071 (7) and EXP-072 (8, 9) records, Gate 0(b) re-check; re-run at 11 | 7, 8, 9, 11 |
| **R3V** | random-guided floor, same tree and rule | 7, 8, 9, 11 |

12 seeds, paired across every arm.

**Amendment 2026-10-07 (final review, before any data): depth 11 has no published policy cell.**
EXP-070 published agents and heads for depths 7 to 9 only. At depth 11 every arm uses the seed's
DEPTH-9 policy agent, head and train seed (they drive only the real-state stream for J arms, and
the move scores for P3V), the held-out states are exactly `heldout_states(11, seed)` (the set judge
training excluded), and the budget is depth 11's, `2d + 3 = 25` moves. Depth 11 stays exploratory.

## 8. Pre-registered claims (depth 9; alpha 0.025 each; one-sided exact sign-flip, 4096 flips)

**Claim 1 (primary): the wide learned judge beats the best recipe so far.** mean(J3V-W - P3V) > 0.

**Claim 2 (primary): reading the whole sensory region helps.** mean(J3V-W - J3V-A) > 0.

Verdicts: CONFIRMED, REFUTED (mean <= 0), NOT SIGNIFICANT, UNRESOLVED (Gate 1), VOID (Gate 0, or Gate
L, E or R for an arm in the contrast). Each primary also prints its 10-seed sensitivity line
(section 2).

**Secondary (a pattern, never confirmations):** J3V-A vs P3V and J3V-A vs J3V-B at depth 9 (EXP-073's
two questions); both primaries at depths 7 and 8.

**Exploratory (no claims):** every arm at depth 11, the typical random scramble.

**What the primaries can detect.** Paired per-seed sd at depth 9 for P3V vs G0 was about 0.012
(EXP-072); for J arms it is unknown. The disclosed cells (section 2) put a J3V-A vs P3V difference
between +0.02 and +0.095 on two seeds, which is an anecdote, not a power estimate. State the
detectable effect after evaluation from the measured paired sd.

## 9. Order

1. Build and test (VPS): readout option in the judge, the state-reading critic seam, Gate L
   instrument, experiment folder reusing EXP-073's cells, training driver and evaluation.
2. W pilot (laptop, seeds 12 and 13) and Gate 0(c).
3. Dated amendment: W's Gate L threshold. Committed before step 4.
4. Train 36 runs, seeds 0 to 11 (laptop, banked checkpoints).
5. Gates L and E computed and the policy's leaf reference measured, committed before any J
   evaluation cell runs.
6. Evaluation: Gate 0(b) cells, all J cells, R3V and P3V, then Gate 0(a).
7. Aggregate, RESULTS.md, records and trained judges committed. EXP-073 gets a short RESULTS.md
   pointing here (superseded, no evaluation).

## 10. Not in scope

A wider or retrained sensory region (option 3, after this answers whether readout width matters).
3x3. A learned no-revisit rule. Using J to train the policy. A judge fitted on true distances is a
diagnostic only and never an arm.
