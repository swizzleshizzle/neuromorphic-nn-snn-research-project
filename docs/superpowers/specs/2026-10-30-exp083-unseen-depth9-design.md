# EXP-083: does X16k's depth-9 gain hold on states it never trained on, and reach depth 10?

Date: 2026-10-30. Chosen by Michael on neuromorphic#29 (option 3, "confirm the depth-9 gain and
extend to depth 10", the quickest of the three), design approved in chat the same day ("go on the
spec").

## 1. The question

EXP-082 read **VALUE ITERATION SATURATED AT 4X BUDGET** at depth 11, but reported, outside its
pre-registration, that J4V-X16k solved **0.6200** of depth-9 scrambles against J4V-X's **0.4004**
(+0.2196, 12 of 12 seeds, p 0.0002), the best depth-9 number in the repo. A reported number found
after the fact is not a confirmed one. This experiment pre-registers it on **new** scrambles and
asks whether the gain reaches **depth 10**.

Distance-to-solved stays an instrument only: it builds the evaluation sets and is the search's
success test, never a training target or a model input.

## 2. Why the scrambles must be chosen, not just drawn

A rerun on EXP-082's own depth-9 sets would replay 0.6200 exactly (same states, same streams,
same machine), so it confirms nothing. New states are needed, and they cannot simply be drawn
from the depth-9 shell:

- The judges' training excluded only **200 held-out states per evaluated depth** (7, 8, 9, 11;
  `experiments/073_learned_judge/cells.py`, `exclusion_set`). Every other depth-9 state, and
  **the whole depth-10 shell** (10 is not an evaluated depth there), was open to the training
  walks: 16,000 batches of 1,000.
- A state drawn at random from the rest of the shell is therefore likely a training state, and
  the comparison would move to an easier population without saying so.

**The walk stream can be replayed exactly.** `vi.train_judge` draws walks from one
`random.Random(seed)`, used by `random_walk_states` and nothing else (probes use their own
generator, encoding uses a torch generator). Replaying the 16,000 batches in plain Python, with
EXP-075's exclusion set, batch 1000 and max length 14, gives every state either judge was trained
on: X's updates 0 to 4000 are a prefix of X16k's 0 to 16,000, because EXP-082 continued EXP-075's
checkpoint, random state included.

**Measured 2026-10-30 on the laptop, seed 0** (`/root/scratch/exp083hw/replay_walks.py`, 232 s):
the replayed random state at update 16,000 **equals** the committed `state.json` exactly; 1,177,203
distinct states were trained on; **76,216** of the 360,508 depth-9 states and **496,965** of the
930,588 depth-10 states were never a training state and are not in the exclusion set.

"Trained on" means a state appeared in a training batch. Children scored only by the frozen target
network are not counted; they receive no gradient.

## 3. The evaluation sets

Per seed and depth d in {9, 10}: **U_d(seed)** = the depth-d shell, minus every replayed training
state, minus that seed's exclusion set; sorted; shuffled by `random.Random(83_000 + 100 * d +
seed)`; the first **200** kept. 200 matches every earlier cell. Written per seed to
`exp083_sets_s<seed>.json` with the replay checks (section 5), so the search reads the sets and
never rebuilds them.

## 4. Arms and the search cell

| arm | judge |
|---|---|
| **J4V-X16k** | EXP-082's X16k at update 16,000 (`experiments/082_x_long_budget/outputs/judge_X_s<seed>/`) |
| **J4V-X** | EXP-075's X at update 4000 (`experiments/075_wide_region/outputs/judge_X_s<seed>/`) |

Both arms are **rerun** on U_9 and U_10: EXP-076's J4V-X records are on the old held-out set and
cannot be the paired baseline. The cell is EXP-082's, unchanged except for its states: mode C,
k 4, goal check at 4, no-revisit, through `e75.JudgeCritic`, 200 scrambles per seed. A judge at
the wrong update count is refused.

**Depth 10 has no published cell.** It mirrors EXP-074's depth-11 cell: the seed's **depth-9**
policy agent and head from EXP-070's `load_cell(9, seed)`, and that cell's train seed driving the
generator, `rng_seed` and `imag_seed`, over U_10. Depth 9 uses the same agent, head and train seed
over U_9.

**Disclosed:** the depth-9 policy head was trained on EXP-070's training side of the depth-9
shell, which U_9 may overlap, while the old held-out set did not. Mode C scores moves with the
judge, not the head, and both arms share the head, so the paired contrast is unaffected. U_d is
also not a uniform draw from the shell: it holds the states the walks happened to miss. The
reported block (section 9) shows how far that moves each arm.

## 5. Gates

- **Gate V (voiding): the replay is the true training set.** Per seed, the replayed random state
  equals EXP-075's committed `state.json` at update **4000** and EXP-082's committed
  `state.json` at update **16,000**. A mismatch means the stream was not what section 2 assumes;
  the seed is refused and the experiment is VOID.
- **Gate R (voiding): the cell still reproduces.** J4V-X16k at depth 9, seed 0, on EXP-082's
  own held-out states, rerun at this checkout, must equal EXP-082's record exactly (`solved`
  127, `success_rate` 0.635). This proves the code and machine have not drifted since EXP-082,
  which every comparison here relies on. Same machine, so exact equality is the right bar.
- **Gate S: the sets are full.** Each U_d(seed) holds 200 states. Seed 0 has 76,216 candidates
  at depth 9, so this cannot bind short of a bug; a failure is UNRESOLVED.
- **Gate F (resolution): the contrast is off the floor.** The chance floor on U_d is measured,
  not assumed: a uniform random policy with the search cell's step budget, over the same 200
  states per seed and the same train seed. A claim at depth d is read only if the better arm's
  12-seed mean exceeds that floor by at least **0.05**. Otherwise that claim is UNRESOLVED.
  (J4V-X solved 0.1950 at depth 11, so this should pass easily at 10; it is gated because two
  arms on the floor contrast to zero by construction, EXP-064's lesson.)

## 6. Claims

One-sided exact sign-flip over all 4096 flips, paired by seed.

| claim | contrast | CONFIRMED when | CONFIRMED (SMALL) when |
|---|---|---|---|
| 1 (primary) | J4V-X16k minus J4V-X, success on U_9 | p < 0.025 and mean >= **0.10** | p < 0.025, mean below 0.10 |
| 2 | J4V-X16k minus J4V-X, success on U_10 | p < 0.025 and mean >= **0.05** | p < 0.025, mean below 0.05 |

Otherwise NOT CONFIRMED.

**Why these bars.** Claim 1's bar is under half the reported +0.2196, so it confirms "a gain of
the reported order", not the exact number. Claim 2's bar sits between EXP-082's depth-11 gain
(+0.0133) and depth-9 gain: depth 10 is where the outward spread of value iteration says the gain
should be fading.

**Power.** EXP-082's 12 paired depth-9 differences had sd **0.0358** (se 0.0103 at n = 12). On a
new population, assume up to twice that: sd 0.07, se 0.020. Claim 1's 0.10 bar then sits 6 se
under the reported gain: a true gain of 0.22 confirms almost surely, 0.15 about 99% of the time,
0.12 about 84%, and gains under about 0.06 cannot reach the bar. Claim 2 at sd 0.05 (se 0.0145)
needs a true gain of about 0.065 to confirm 85% of the time. **A NOT CONFIRMED claim bounds the
gain; it does not show zero.**

## 7. Verdict (in the aggregator, one line)

| condition | VERDICT |
|---|---|
| Gate V or Gate R failed | VOID |
| Gate S failed, or Gate F failed at depth 9 | UNRESOLVED |
| Claim 1 CONFIRMED and Claim 2 CONFIRMED | DEPTH-9 GAIN CONFIRMED, REACHES DEPTH 10 |
| Claim 1 CONFIRMED, Claim 2 anything else | DEPTH-9 GAIN CONFIRMED, STOPS SHORT OF DEPTH 10 |
| Claim 1 CONFIRMED (SMALL) | DEPTH-9 GAIN SMALLER THAN REPORTED |
| otherwise | DEPTH-9 GAIN DOES NOT HOLD ON UNSEEN STATES |

The unresolved band is a verdict of its own (the EXP-068 rule): a significant gain under 0.10 at
depth 9 prints SMALLER THAN REPORTED, never CONFIRMED and never DOES NOT HOLD. When Gate F fails
at depth 10 only, Claim 2 prints UNRESOLVED and the row reads STOPS SHORT OF DEPTH 10.

## 8. Code and outputs

`experiments/083_unseen_depth9/`, importing EXP-082's `cells`, `gatec` and `evaluate` and
EXP-075's / EXP-074's loaders unchanged:

- `sets.py`: the replay, Gate V, Gate S, the sets file per seed (4 workers; each loads the full
  BFS table, about 1 GB).
- `evaluate.py`: the search cell over a sets file, both arms, depths 9 and 10, plus the Gate R
  rerun; **6 workers** (EXP-082 measured 12 workers slower per cell than EXP-081's 6, for no
  throughput).
- `floor.py`: Gate F's random-policy floor.
- `aggregate.py`: gates, claims, verdict, reported block; writes the verdict file only on a clean
  aggregate.
- A chain script for the laptop, the same shape as EXP-082's: each stage skips finished work, so a
  03:31 restart costs at most the cells in flight.

## 9. Reported, never confirmations

- Both arms on U_9 against their EXP-082 / EXP-076 numbers on the old held-out set: how far the
  new population moves each arm.
- Optimality, mean steps, goal-fired fraction, fallback fraction, per arm and depth.
- The chance floors on U_9 and U_10.
- Per seed: distinct states trained on, and U_9 and U_10 candidate counts before the cap.

## 10. No pilot

The cell is EXP-082's, already run on all 12 seeds; Gate R reruns one of its cells exactly. The
replay is measured (232 s for seed 0). The only new code path is reading states from a sets
file, covered by tests.

## 11. Cost and schedule

- Sets: 12 seeds at about 4 min each, 4 workers, **about 15 min**.
- Search: 48 cells plus Gate R's one. At 6 workers EXP-081's depth-11 cells took about 5,200 s;
  EXP-082's depth-9 cells took about 0.5 times its depth-11 cells, so depth 9 is about 2,600 s
  and depth 10 about 3,900 s per cell. **About 7.5 h.** U_d may be harder or easier than the old
  sets, so allow 6 to 10 h.
- Floor: minutes.
- **Total about 8 h on the laptop**, which has nothing else queued while the desktop is set up
  for GPU training. Build about a day; verdict the evening it runs if it starts in the morning,
  otherwise it crosses one 03:31 window and resumes.

## 12. What this does not answer

- Whether X16k's ordering keeps improving with more training (EXP-082 showed it spreads about one
  distance per 12,000 updates; that is options 1 and 2 on #29, which need the GPU).
- Depth 11 on U_11: EXP-082 already measured depth 11 on states the judges never trained on.
- Whether the gain holds for a judge trained on another machine. Retraining is not portable; this
  re-evaluates committed checkpoints only.
