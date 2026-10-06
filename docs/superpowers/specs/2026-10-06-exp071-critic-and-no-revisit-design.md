# EXP-071: the critic as judge, and a no-revisit rule in the search

**Written 2026-10-06, week 27**, after EXP-070 closed. Design agreed with Michael in session
`week27`. Roadmap: `docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md` section 4,
stage 2, with the visited-state recommendation from `experiments/070_lookahead_existing/RESULTS.md`
run alongside it at Michael's request.

## 1. Why

EXP-070 showed the policy head cannot judge a look-ahead: P3 adds below about 0.01 over a bare
goal check at depth 8 (NOT SIGNIFICANT, p 0.3818). It also showed the reflex revisits a state on
about 40% of moves at depth 7 (0.3997) and policy-guided search inherits that. Two questions:

1. **Can the critic judge?** It was trained (EXP-053 arm B) to predict return from the sensory
   concept, which makes it the natural "how close is this position" signal. It was never asked
   to rank neighbouring positions.
2. **Does refusing to revisit help?** The memory line tested memory as a recall INPUT to the head
   (EXP-059/061/063) and `revisit_rate` did not move. A hard rule in the SEARCH is a different,
   untested mechanism. It is a plain lookup table, not a spiking component, and is reported as such.

Both are re-evaluations of the same 12 depth-7 checkpoints with the EXP-070 harness, so one
experiment with paired seeds answers both. Depth 7 only: it is the one depth with saved critics
(`experiments/053_neuromod_stage3/outputs/exp053_critic_d7_*_critic.pt`, tracked 2026-10-06).

## 2. What is loaded (nothing is trained)

EXP-070's depth-7 cells exactly (`experiments/070_lookahead_existing/cells.py`: EXP-053 arm B
heads, E1 encoders, the published held-out split, 200 states), plus each seed's critic,
`nn.Linear(64, 1)` on the sensory concept. Higher value means a better expected return.

## 3. Procedure additions

**Scorer C (critic-guided), look-ahead `k`.** Goal test first, exactly as E, P and R. Otherwise
encode the `n**k` leaf states in ONE batched call on the imagined stream, score each sequence by
the critic's value of its leaf, and play the first move of the best (first maximum on ties).
`C1` therefore picks the child the critic rates highest.

**Modifier V (no revisit), any scorer.** The rollout keeps the set of real states visited this
episode, including the current one. A sequence is ELIGIBLE iff no state it passes through (after
each of its moves) is in that set. Ineligible sequences are excluded from the argmax; for G0+V
and E+V the greedy move is the highest-logit move whose resulting state is unvisited. If NO
candidate is eligible, the rule falls back to the unmasked choice and the fallback is counted.
The goal test is unaffected (solved is never in the visited set). The set is the agent's own
history: no oracle.

**Stream discipline unchanged** (EXP-070 section 2.3): one real-stream draw per real move in every
arm; imagined encodings and random scores on a per-(state, step) imagined generator.

**No BFS inside the procedure.** The distance table builds held-out shells and, in step 0 only,
serves as the yardstick for ranking. It is never read by a scorer, the goal test, or V.

## 4. Step 0: the ranking check (runs first; decides whether the critic arms may be read)

For every held-out state of every seed, encode its `n` children in one batched call (imagined
stream, seeded per (seed, state)). Record whether the critic's top-rated child is an IMPROVING
child (exact distance 6), and the same for the policy head's argmax at the parent (the reflex's
own first move). Pre-flight facts measured before this spec, depth 7, all 2400 held-out states:

| | |
|---|---|
| every move changes distance by | exactly +1 or -1 |
| chance: mean fraction of improving children | **0.2203** (min 1/6, max 1.0) |
| maximum attainable hit rate | 1.0 |

**Each critic arm is gated at the scale it runs at** (the gate-calibration rule: a k = 1 hit rate
says little about an argmax over 216 noisy leaves). Per state, step 0 therefore also encodes the
`n**3` leaves of the depth-3 tree in one batched call and records whether the critic's top-rated
leaf is CLOSER than the root, and whether it is at exactly `d - 3`.

| check | chance at depth 7, measured before this spec (2400 states) | governs |
|---|---|---|
| R1: top child improves | **0.2203** | C1, C1V (secondary) |
| R3: top leaf closer than root | **0.1344** | C3, C3V, and therefore Claim 1 |
| top leaf at `d - 3` | 0.0082 | reported only |

Leaf distances at depth 7, k = 3, are only ever 4, 6, 8 or 10. Maximum attainable hit rate is 1.0
for each check.

**Gate R1 / Gate R3.** Per seed, `hit - chance` with both averaged over that seed's 200 states
(chance computed per state). A gate PASSES iff mean(hit - chance) > 0 by the exact one-sided
sign-flip test at p < 0.05. If R3 fails, Claim 1 is **VOID** and the roadmap goes to stage 3; if
R1 fails, the C1 secondaries are reported as data only. The policy head's child-level hit rate is
reported beside R1 as a reference, not gated.

> **AMENDMENT 2026-10-06, before any number exists:** the single child-level Gate R of the first
> draft was replaced by R1 and R3 above, after review noted that it was calibrated in C1's regime
> while guarding a C3 claim.

## 5. Arms

12 seeds each, depth 7, held-out 200, real-move budget `2d + 3 = 17`.

| arm | k | scorer | V |
|---|---|---|---|
| G0, G0V | 0 | greedy | no / yes |
| E3, E3V | 3 | greedy + goal test | no / yes |
| P3, P3V | 3 | policy | no / yes |
| C1, C1V | 1 | critic | no / yes |
| C3, C3V | 3 | critic | no / yes |
| R3, R3V | 3 | random | no / yes |

12 arms x 12 seeds = 144 cells.

## 6. Gates

**Gate 0(a) determinism.** Three cells (G0V, C3, R3V, seed 0) re-run into a separate directory are
identical in every field except `wall_s` and `git_commit`.

**Gate 0(b) continuity with EXP-070, EXACT.** EXP-071's G0, E3, P3 and R3 run the identical code
path and stream as EXP-070's depth-7 cells, so their records must match EXP-070's committed
records in every outcome field (`solved`, `n`, `success_rate`, `mean_steps`, `optimality`,
`eval_revisit_rate`, `greedy_modal_action_frac`, `goal_fired_frac`) for all 48 cells. Unlike
EXP-070's Gate 0, this is re-evaluation against re-evaluation, where exactness is the expected
regime. Any mismatch means the C or V code changed the shared path, and no claim may be read.

**Gates R1 and R3** (section 4).

**Gate V (the rule engaged).** mean `eval_revisit_rate` of G0V must be at most **0.5x** that of
G0 (EXP-070 measured G0 at 0.3997, so the bar is about 0.20). A ratio, per the gate-calibration
rule; maximum attainable reduction is to 0, so it is satisfiable. If it fails, V did not engage
and Claim 2 is **VOID**. Also reported: the fallback fraction per V arm.

**Gate 1 (resolution).** As EXP-070: a contrast is UNRESOLVED if both arms are below 0.02 or both
above 0.98. At depth 7 E3 is 0.2321 and G0 0.2067, so neither primary is expected to sit on a
bound; the gate is encoded in the aggregator regardless.

## 7. Pre-registered claims

Two primaries, one per track, each at **alpha 0.025** (one-sided exact sign-flip over 4096 flips),
so that running both together holds the family-wise error at 0.05.

**Claim 1 (primary, critic track): the critic is a better judge than reflex plus endgame.**
mean(C3 - E3) > 0. Verdicts: CONFIRMED, REFUTED (mean <= 0), NOT SIGNIFICANT, UNRESOLVED (Gate 1),
VOID (Gate R3 failed).

**Claim 2 (primary, no-revisit track): refusing to revisit helps the reflex.**
mean(G0V - G0) > 0. Same verdict ladder, VOID if Gate V fails.

**Secondary (reported with p-values, read as a pattern, never as confirmations):** C3 - P3
(critic against policy as judge), C1 - G0 (one-step critic against the reflex), and V within each
scorer: E3V - E3, P3V - P3, C3V - C3. R3 and R3V are floors.

**What each primary can detect.** From EXP-070's depth-7 records, paired per-seed sd is 0.0231
(P3 - E3) and 0.0249 (E3 - G0), se about 0.007, so at alpha 0.025 one-sided each primary has 80%
power at a true effect of about **0.02**. A NOT SIGNIFICANT verdict means "below about 0.02".

## 8. Cost and order

Step 0 (two batched calls per state) runs first, on the laptop, then the base arms and the
determinism re-run. **Those decide every gate, and the gate results are committed in a dated
amendment BEFORE any C cell runs.** Only then does one calibration cell each of C3 and C3V at
seed 0 run, for wall-clock (C3 encodes 216 leaves per real move); its cost is appended, dated,
before the full launch. Records bank per cell and
are committed (force-added), as EXP-070's are.

## 9. Not in scope

Depths 8 and 9 (no saved critics; a retrain is its own decision). Any spiking implementation of V.
Training the critic for ranking (that is stage 3).
