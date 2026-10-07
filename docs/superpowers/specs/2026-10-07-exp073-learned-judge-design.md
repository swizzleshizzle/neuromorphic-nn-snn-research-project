# EXP-073: a learned "distance to solved" judge, trained from its own look-ahead (stage 3)

**Written 2026-10-07, week 27.** Design agreed with Michael in session `week27`: arm A trains the
spiking encoder, arm B is its frozen-encoder control, and a pilot sizes the run before any
evaluation seed trains. Roadmap: `docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md`,
stage 3.

## 1. Why

Two judges have failed. The policy head scores look-ahead below about 0.01 better than a goal check
(EXP-070), and the reused critic, though above chance at ranking, makes search far worse (EXP-071,
C3 0.0729 vs E3 0.2321). The best recipe, P3V, is search scaffolding around a frozen network (depth
9: 0.0629, EXP-072). The roadmap's stage 3 asks whether a judge TRAINED FOR THE JOB, from its own
look-ahead and never from the answers, does better, and whether training the spiking network itself
is what makes the difference.

## 2. The judge

`J(s)`: estimated moves to solved, `>= 0`.

- **Encoder:** the seed's spiking sensory region, initialised from its E1 encoder
  (`experiments/047_encoder_finetuning/outputs/exp047_ft_d6_lr0.0001_regionalized_d6_s{seed}_sig0.0_encoder.pt`),
  Poisson-encoded facelets in, concept mean rate (64) out, exactly the path every earlier
  experiment used.
- **Head:** `Linear(64, 128) -> ReLU -> Linear(128, 1) -> softplus`.
- **Arm A:** encoder AND head train (surrogate gradients, as EXP-047). **Arm B:** encoder frozen,
  head only. Everything else identical, including seeds, batches and generators.

## 3. Training: approximate value iteration (DeepCubeA-style)

Per update:

1. **Positions:** `N` states, each from a random walk of length uniform in `1..14` from solved
   (14 is the 2x2 quarter-turn diameter). Any state in the seed's EVALUATION held-out sets (depths
   7, 8, 9 and 11) is rejected and redrawn, so no evaluated position is ever trained on.
2. **Targets:** `y(s) = 0` if `s` is solved, else `1 + min over the n children c of Jt(c)`, with
   `Jt(c) = 0` for a solved child. `Jt` is a frozen copy of `J`, refreshed every `sync_every`
   updates.
3. **Loss:** mean squared error between `J(s)` and `y(s)`. Adam: head lr 1e-3; encoder lr 1e-4 in
   arm A (EXP-047's rate), 0 in arm B.

**No answers anywhere in training.** The BFS distance table is never a target and is never read
during training. The one filter on a batch is membership in a FIXED LIST of evaluation states
(built once, before training, from the BFS shells); no distance is consulted to apply it. The table
is otherwise read only by the instrument below and by evaluation scoring. The simulator
(`apply_move`, `is_solved`) is the only source of structure.

**Instrument (read, never trained on):** every `probe_every` updates, J is evaluated on a fixed
probe set of 50 states per true distance 1 to 11, drawn from BFS shells and disjoint from every
held-out set. Recorded: Spearman rank correlation between J and true distance, and mean J per true
distance. Checkpoints (J, Jt, optimiser state, generator state, update count) are banked at every
probe so a lost run resumes rather than restarts.

**Jt is a frozen copy of the WHOLE judge**, encoder and head both. Copying only the head while
reading arm A's live encoder would make the targets move every update.

**Defaults the pilot may revise** (section 4): `N = 1000`, `sync_every = 500`,
`probe_every = 500`, and `jt_draws = 1` (Poisson draws averaged per child when computing `Jt`).

## 4. Pilot (seeds 12 and 13 only; they never enter an evaluation)

Both arms, both seeds, a few hours each on the laptop. It measures: updates per second at the
chosen `N`; the probe Spearman and mean-J-per-distance curves over training; whether `sync_every`
lets J's values climb past depth 9 (DAVI propagates one step of distance per sync, so a sync
interval that is too long or too short shows up as a stalled curve).

**A named risk the pilot must check:** each target takes a MIN over `n` Poisson-noisy `Jt` values,
which is biased low, and the bias compounds every sync. Its symptom is mean J per distance
compressing (flattening) past about distance 8. **The pre-named mitigation** the amendment may
adopt is raising `jt_draws` (averaging several Poisson draws per child inside `Jt`); choosing it in
the amendment is pre-registered, inventing a different fix afterwards is not.

**Then a dated amendment, before any seed 0-11 trains,** fixes: `N`, `sync_every`, the number of
updates per run (sized so 24 runs fit in about a day on the laptop), and the **training gate**
threshold below. Pilot numbers may not be compared to evaluation numbers: different seeds, and
they are calibration, not results.

## 5. Evaluation

Re-evaluation over the published held-out sets (EXP-070 cells for depths 7 to 9; depth 11 uses
`split_shell` on the depth-11 shell with the same split rule), budget `2d + 3` real moves, 12 seeds
paired across every arm.

**J-driven search** reuses the look-ahead procedure's critic scorer with `critic = -J` read from
the J-trained encoder's concept (lower J is better), so no new search code is needed:

| arm | what it is |
|---|---|
| **J1V-A, J1V-B** | step to the child with the lowest J, no-revisit rule |
| **J3V-A, J3V-B** | 3-move tree, goal test, lowest-J leaf, no-revisit rule |
| **P3V** | the best recipe so far; EXP-071 (depth 7) and EXP-072 (depths 8, 9) records, re-checked by Gate 0(b) |
| **R3V** | random-guided floor, same tree and rule |

The real-state stream draw uses the policy head exactly as before (stream discipline); J arms
ignore it, as C arms did.

## 6. Gates

**Gate 0(a) determinism.** One J3V-A and one J3V-B cell (seed 0, depth 9) re-run into a separate
directory: identical except `wall_s` and `git_commit`.

**Gate 0(b) continuity.** P3V at seed 0, depths 8 and 9, re-run under this harness, must equal
EXP-072's records in all 8 outcome fields.

**Gate T (training worked where the claims live), per arm.** Computed on probe distances **7 to 11
only**, because a correlation pooled from distance 1 is dominated by the easy shallow end and
passes a judge that is flat past distance 6, which is exactly DAVI's failure with too few syncs.
Both must hold, on the mean over the 12 seeds:
(a) Spearman of J against true distance over probe distances 7 to 11 reaches the threshold, whose
FORM is fixed now: **half of that arm's mean pilot end-of-training value of the same quantity, and
never below 0.30**; (b) mean J per distance is **strictly increasing from 7 through 11**.
If an arm fails Gate T, its claims are VOID.

**Gate E (arm A's encoder actually trained; arm B's did not).** EXP-047's first implementation
trained nothing and produced an ordinary-looking run. So: arm A's encoder parameter drift (L2 norm
of the change from its E1 starting point) must be **greater than zero on every seed**, and arm B's
encoder must be **bit-identical** to E1 on every seed. If arm A fails, Claim 2 is VOID; if arm B
fails, it is not a frozen control and Claim 2 is VOID.

**Gate R (J ranks at the scale it searches at).** Per depth, J's top-rated leaf of the 3-move
tree is closer than the root more often than chance (exact one-sided sign-flip over 12 seeds,
p < 0.05). Chance, measured before this spec on 1200 held-out states per depth (100 per seed):

| depth | chance: top leaf closer | top leaf at `d - 3` |
|---|---|---|
| 7 | 0.1344 (EXP-071) | 0.0082 |
| 8 | **0.1477** | 0.0090 |
| 9 | **0.1750** | 0.0104 |
| 11 | 0.5209 | 0.0277 |

At depth 11 more than half of all 3-move leaves are closer than the root, so this check is weak
there; depth 11 carries no claim. A failed Gate R voids that arm's claims at that depth.

**Reference, recorded before any J cell is evaluated:** the same leaf-level hit rate for the
POLICY's own sequence ranking (P3's summed log-probability, top leaf closer than root) at depth 9.
EXP-071 showed that clearing chance is not the same as steering well; the policy's rate is the
useful bar, and it goes on record first.

**Gate 1 (resolution).** A contrast is UNRESOLVED if both arms are below 0.02 or above 0.98.

## 7. Pre-registered claims (alpha 0.025 each, one-sided exact sign-flip, 4096 flips)

**Claim 1 (primary): the learned judge beats the best recipe so far.** mean(J3V-A - P3V) > 0 at
depth 9 (P3V 0.0629, EXP-072).

**Claim 2 (primary): training the spiking network matters.** mean(J3V-A - J3V-B) > 0 at depth 9.

Verdicts: CONFIRMED, REFUTED (mean <= 0), NOT SIGNIFICANT, UNRESOLVED (Gate 1), VOID (Gate 0, or
Gate T or Gate R for an arm in the contrast).

**Secondary (a pattern, never confirmations):** the same two contrasts at depths 7 and 8; J1V-A
against G0V (EXP-071/072 records); J3V-B against P3V.

**Exploratory (no claims):** J3V-A, J3V-B, P3V and R3V on 200 held-out states at **depth 11**, the
typical random scramble.

**What each primary can detect.** Paired per-seed sd at depth 9 from EXP-072 (P3V - G0) is about
0.012; for a J arm against P3V it is not yet known, and the pilot cannot measure it (it runs no
evaluation). The amendment states a detectable effect from the depth-9 noise in EXP-070/072
records, about **0.010 to 0.02**, and says so as an estimate.

## 8. Order

1. Build and test (VPS).
2. Pilot (laptop, seeds 12 and 13, both arms).
3. Dated amendment: training settings, run length, Gate T threshold. Committed before step 4.
4. Train seeds 0 to 11, both arms (laptop; banked checkpoints).
5. Gates T and E computed, and the policy's leaf-level reference measured, then committed before
   any J evaluation cell runs.
6. Evaluation: Gate 0(b) continuity cells, then all J cells, R3V at depth 11, P3V at depth 11,
   then the determinism re-run.
7. Aggregate, RESULTS.md, records and trained judges committed.

## 9. Not in scope

3x3. A learned or spiking no-revisit rule. Deeper search. Using J to train the policy.

## 10. Pilot 1 report, 2026-10-07 (NOT yet the amendment; training settings PENDING pilot 2)

Seeds 12 and 13, both arms, `N = 1000`, `sync_every = 100` (40 Jt refreshes), `probe_every = 250`,
`jt_draws = 1`, 4000 updates, 4 workers on the laptop, commit `9a92aff`. Records:
`experiments/073_learned_judge/outputs/exp073_train_{A,B}_s1{2,3}.json` (tracked).

**Throughput:** 1.70 s/update (arm A), 1.59 s/update (arm B), at 4 workers. A 20-worker figure is
not measured; eval cells ran about 3x slower per cell at 20 workers than at 3, so sizing must name
the factor it assumes or measure it.

**Gate E passes cleanly:** arm A encoder drift 8.885 and 9.110; arm B exactly 0.0 on both seeds.

**The named risk happened: mean J compresses past about distance 7.** Seed-mean final values:

| true distance | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|
| arm A mean J | 5.51 | 5.95 | 6.36 | 6.59 | 6.65 | 7.07 | 7.01 |
| arm B mean J | 5.26 | 5.63 | 5.91 | 6.07 | 6.19 | 6.58 | 6.50 |

The curves plateau by about update 2000 of 4000, so this is a fixed point, not slow propagation.
Pooled Spearman (distances 1 to 11) reaches 0.83 / 0.80 (A) and 0.78 / 0.74 (B); the deep end is
where it fails.

**Gate T, applied to the pilot with the form fixed in section 6:**

| arm | final spearman_7_11 (s12, s13) | mean | threshold max(0.30, mean/2) | (a) | (b) seed-mean J strictly increasing 7..11 |
|---|---|---|---|---|---|
| A | 0.313, 0.264 | 0.289 | **0.30** (floor binds) | below | **fails** (10: 7.07 > 11: 7.01) |
| B | 0.267, 0.247 | 0.257 | **0.30** (floor binds) | below | **fails** (10: 6.58 > 11: 6.50) |

**Under pilot 1's settings, the 12-seed run is predicted to fail Gate T on both halves for both
arms**, which would void both claims. That is the gate-calibration trap, caught at pilot cost. The
0.30 floor is part of the pre-registered form and is NOT changed here.

**Where the compression comes from (diagnostic on the pilot judges, seeds 12 and 13 only):** per
state, the Poisson spread of J across 32 encoding draws is about 0.19 to 0.25 moves at distances 7
to 11; the spread of J BETWEEN states at the same true distance, after averaging 32 draws, is about
0.68 to 0.83. Averaging 32 draws at inference leaves spearman_7_11 unchanged (A s12 0.344 to 0.314,
B s12 0.256 to 0.256). So Poisson noise is the smaller part of the error the min in each target is
biased by; most of it is the judge's approximation error. This does NOT settle what a judge TRAINED
on `jt_draws = 4` targets would learn, since training-time averaging shrinks the min-bias that
compounds every sync. Only a run can tell.

**Pilot 2 (running from 2026-10-07 ~07:45 UTC):** the pre-named mitigation, `jt_draws = 4`, all else
identical to pilot 1, seeds 12 and 13, records in `outputs_pilot2/`. About 3.6x the encodings per
update, so roughly 6 h at 4 workers.

**If pilot 2 also misses 0.30, the choices are Michael's** (no evaluation number exists, so each is a
legitimate pre-data amendment if dated): (i) run anyway and let Gate T void, a mechanism finding at
a day's cost; (ii) explicitly amend the floor, with this report attached; (iii) revise the judge
design under a new spec (the 64-unit concept from a 192-neuron sensory region may be what limits the
deep end).

## 11. Pilot 2 report, 2026-10-07: the pre-named mitigation does not rescue Gate T

`jt_draws = 4`, everything else as pilot 1. Records:
`experiments/073_learned_judge/outputs_pilot2/exp073_train_{A,B}_s1{2,3}.json` (tracked).
Throughput 5.46 / 5.35 s/update at 4 workers, 3.2x pilot 1. Gate E again clean (A drift 9.175,
9.357; B 0.0).

| arm | jt_draws | spearman_7_11 (s12, s13) | mean | seed-mean J at 7, 8, 9, 10, 11 | (b) |
|---|---|---|---|---|---|
| A | 1 | 0.313, 0.264 | 0.289 | 6.36, 6.59, 6.65, 7.07, 7.01 | fails |
| A | 4 | 0.328, 0.266 | 0.297 | 6.43, 6.71, 6.76, 7.19, 7.14 | fails |
| B | 1 | 0.267, 0.247 | 0.257 | 5.91, 6.07, 6.19, 6.58, 6.50 | fails |
| B | 4 | 0.248, 0.258 | 0.253 | 6.21, 6.39, 6.55, 6.88, 6.86 | fails |

Averaging four draws lifts every value by 0.1 to 0.35 moves (the min-bias shrinks, as expected) but
leaves the ORDERING of distances 7 to 11 where it was: both arms stay under the 0.30 floor and both
seed-mean curves still fall from 10 to 11. This matches the section 10 diagnostic: the binding error
is the judge's approximation error between states, not Poisson noise.

**Status: no seed 0-11 has trained; `GATE_T_THRESHOLD` and `$TrainUpdates` remain unset**, so the
launcher refuses every claim-bearing phase. The section 10 choices (run and let Gate T void; amend
the floor; revise the judge design under a new spec) are now Michael's decision.
