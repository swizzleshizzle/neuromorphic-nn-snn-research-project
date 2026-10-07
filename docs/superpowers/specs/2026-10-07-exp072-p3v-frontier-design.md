# EXP-072: does look-ahead plus no-revisit move the depth 8-9 frontier?

**Written 2026-10-07, week 27.** Option 1 of the decision in `docs/handoffs/SESSION-HANDOFF-2026-10-06.md`
section 2, chosen by Michael. Re-evaluation only; nothing trains.

## 1. Why

EXP-071 (depth 7) found that a no-revisit rule helps the reflex (CONFIRMED, +0.0279) and that
policy-guided 3-move look-ahead with the rule, P3V, reaches 0.3217 against P3's 0.2454 (a
SECONDARY, 12 of 12 seeds). EXP-070 had left the frontier at depth 8 (G0 0.0688) and depth 9
(G0 0.0171, at the floor), where look-ahead without the rule added below about 0.01. The
question: does P3V's gain survive at the depths where the reflex fails, and does depth 9 leave
the floor?

## 2. What is loaded and what runs

EXP-070's depth-8 and depth-9 cells exactly (`experiments/070_lookahead_existing/cells.py`: EXP-062
heads, E1 encoders, the published held-out split, 200 states, budget `2d + 3`). No critic.

**New arms, 12 seeds each, depths 8 and 9:** G0V, P3V, R3V. That is 3 x 2 x 12 = 72 cells.

**Paired references, NOT re-run:** EXP-070's committed G0, E3, P3 and R3 records at depths 8 and 9.
They come from the identical code path and stream discipline (EXP-071's Gate 0(b) showed the C/V
changes leave those arms bit-identical), and Gate 0(b) below re-checks that on four cells.

## 3. Gates

**Gate 0(a) determinism.** P3V depth 8 seed 0 and G0V depth 9 seed 0 re-run into a separate
directory: identical in every field except `wall_s` and `git_commit`.

**Gate 0(b) continuity.** G0 and P3 at seed 0, depths 8 and 9 (4 cells), re-run under EXP-072's
harness, must equal EXP-070's committed records in all 8 outcome fields (`solved`, `n`,
`success_rate`, `mean_steps`, `optimality`, `eval_revisit_rate`, `greedy_modal_action_frac`,
`goal_fired_frac`). A mismatch means the paired references are not comparable, and no claim may
be read.

**Gate V, per depth.** mean `eval_revisit_rate` of G0V at most **0.5x** that of G0 (EXP-070: 0.454
at depth 8, 0.556 at depth 9). Maximum attainable reduction is to 0, so it is satisfiable.

**Gate 1 (resolution).** As EXP-070/071: a contrast is UNRESOLVED if both arms are below 0.02 or
both above 0.98. At depth 9, G0 is 0.0171, so Claim 2 resolves only if P3V clears 0.02.

## 4. Pre-registered claims (alpha 0.025 each, one-sided exact sign-flip over 4096 flips)

**Claim 1 (primary, depth 8): no-revisit unlocks look-ahead where the reflex fails.**
mean(P3V - P3) > 0 at depth 8. This replicates EXP-071's secondary at a new depth.

**Claim 2 (primary, depth 9): the frontier moves.** mean(P3V - G0) > 0 at depth 9. Read as a
practical gain of the whole recipe (search plus rule) over the reflex, never as evidence about
the network alone.

Verdicts: CONFIRMED, REFUTED (mean <= 0), NOT SIGNIFICANT, UNRESOLVED (Gate 1), VOID (Gate 0 or,
for any contrast involving a V arm, Gate V failed).

**Secondary (a pattern, never confirmations):** depth 8 P3V - G0; depth 9 P3V - P3; G0V - G0 at
both depths; P3V - G0V at both depths (the interaction EXP-071 suggested). R3V is the floor.

**What each primary can detect** (paired per-seed sd from EXP-070's records, 80% power at alpha
0.025 one-sided): depth 8 P3 - E3 sd 0.0144 and P3 - G0 sd 0.0241, so about **0.012 to 0.02**;
depth 9 P3 - G0 sd 0.0127, so about **0.010**. A NOT SIGNIFICANT verdict means "below roughly
that", not zero.

## 5. Cost and order

P3V cells cost about what P3 did in EXP-070 (up to about 2000 s at depth 9 with 20 workers). Order:
continuity (4 cells) and determinism (2 cells) first; if Gate 0(b) fails, stop. Then the 72 new
cells. Records are committed (force-added).

## 6. Not in scope

Depth 7 (EXP-071 covered it). Any critic. Any learned or spiking version of the rule.
