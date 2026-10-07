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

**No answers anywhere in training.** The BFS distance table is never a target, never filters a
batch, and is read only by the instrument below and by evaluation scoring. The simulator
(`apply_move`, `is_solved`) is the only source of structure.

**Instrument (read, never trained on):** every `probe_every` updates, J is evaluated on a fixed
probe set of 50 states per true distance 1 to 11, drawn from BFS shells and disjoint from every
held-out set. Recorded: Spearman rank correlation between J and true distance, and mean J per true
distance. Checkpoints (J, Jt, optimiser state, generator state, update count) are banked at every
probe so a lost run resumes rather than restarts.

**Defaults the pilot may revise** (section 4): `N = 1000`, `sync_every = 500`,
`probe_every = 500`.

## 4. Pilot (seeds 12 and 13 only; they never enter an evaluation)

Both arms, both seeds, a few hours each on the laptop. It measures: updates per second at the
chosen `N`; the probe Spearman and mean-J-per-distance curves over training; whether `sync_every`
lets J's values climb past depth 9 (DAVI propagates one step of distance per sync, so a sync
interval that is too long or too short shows up as a stalled curve).

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

**Gate T (training worked), per arm.** End-of-training probe Spearman of J against true distance,
mean over the 12 seeds, must reach the threshold the pilot amendment fixes. The threshold's FORM
is fixed now: **half of that arm's mean pilot end-of-training Spearman, and never below 0.30.**
If an arm fails Gate T, its claims are VOID.

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
5. Gate T computed and committed before any J evaluation cell runs.
6. Evaluation: Gate 0(b) continuity cells, then all J cells, R3V at depth 11, P3V at depth 11,
   then the determinism re-run.
7. Aggregate, RESULTS.md, records and trained judges committed.

## 9. Not in scope

3x3. A learned or spiking no-revisit rule. Deeper search. Using J to train the policy.
