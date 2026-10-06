# EXP-070 results: look-ahead on the networks we already have

**Run 2026-10-06** on SwizzlesDuo (laptop), main at `0af516b` (harness) and `cad0a9c` (pre-launch
amendment, which fixed both gates before any P number existed). 324 cells: 9 arms x depths 7, 8, 9
x seeds 0-11, re-evaluation only, nothing trained. Records are committed in `outputs/`.

Spec: `docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md` (section 2, its 2026-10-06
amendments). Regenerate the verdicts with:

```bash
.venv/bin/python experiments/070_lookahead_existing/aggregate.py --determinism-ok
```

> **HEADLINE: scoring the look-ahead with the policy head adds nothing measurable.** Primary
> cell (depth 8, k = 3): P3 **0.0854** against E3 **0.0838**, +0.0017, p **0.3818**, **NOT
> SIGNIFICANT**. The one-sided 95% upper bound on the benefit is about **+0.009**, so if scoring
> helps at all it is by less than one cube in a hundred. **The policy head is not a usable judge,
> which is the outcome stage 2 (the critic) exists for.**
>
> **Look-ahead guided by the policy also does not break the policy's loops.** The reflex is known
> to cycle: EXP-062's published records already show `eval_revisit_rate` **0.445 at depth 8 and
> 0.552 at depth 9**, and this run reproduces that (G0 0.454, 0.556) against 0.19 for a random
> walk. What EXP-070 adds is that P3 leaves it unchanged (0.457 vs E3 0.449 at depth 8): a search
> that asks the same policy to judge inherits the same cycles.

## Gates

| gate | verdict | evidence |
|---|---|---|
| 0(a) determinism | **PASS** | G0, E3, R3 at depth 8 seed 0 re-run into a separate directory: identical in every field except `wall_s` |
| 0(b) published match, Wilson form | **PASS** | per-depth pooled G inside the published Wilson 95% interval at all three depths (depth 8 by 0.0006). The exact form was ruled out in pre-flight: 27 of 36 cells differ from their published counts because evaluation encodes with Poisson spikes on a different stream. The stronger evidence the harness is right is `test_mode_g_reproduces_evaluate_states_exactly`. |
| 1 resolution | primary **RESOLVED** | E3 at depth 8 is 0.0838, clear of the 0.02 floor |

## Mean held-out success (12 seeds, 200 states each)

| depth | G0 | E1 | E2 | E3 | P2 | P3 | R1 | R2 | R3 |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 0.2067 | 0.2179 | 0.2150 | 0.2321 | 0.2158 | 0.2454 | 0.0000 | 0.0008 | 0.0025 |
| 8 | 0.0688 | 0.0708 | 0.0750 | 0.0838 | 0.0746 | 0.0854 | 0.0000 | 0.0004 | 0.0013 |
| 9 | 0.0171 | 0.0179 | 0.0200 | 0.0200 | 0.0192 | 0.0250 | 0.0000 | 0.0004 | 0.0008 |

G is the no-look-ahead reflex, E adds only a k-move goal check, P scores k-move sequences by
summed policy log-probability, R scores them at random.

## Pre-registered claims

**Claim 1 (primary, depth 8, k = 3): scoring beats reflex plus endgame. NOT SIGNIFICANT.**
P3 - E3 = +0.0017, exact one-sided sign-flip p 0.3818. Per-seed differences in solves out of 200:
4, 1, -3, -1, 3, -5, 0, 3, -2, 4, -1, 1 (sd 0.0146, se 0.0042). The amendment stated in advance
that this cell can detect about 0.02 at 80% power from a noise bound of 0.0244; the realised
noise was smaller (0.0146), so the realised sensitivity is about 0.01. **Read: below about 0.01,
not zero.**

**Secondary cells (a pattern, never independent confirmations):**

| cell | P | E | diff | p | verdict |
|---|---|---|---|---|---|
| depth 8, k = 2 | 0.0746 | 0.0750 | -0.0004 | 0.5781 | REFUTED (point estimate <= 0) |
| depth 9, k = 2 | 0.0192 | 0.0200 | -0.0008 | 0.7500 | REFUTED (point estimate <= 0) |
| depth 9, k = 3 | 0.0250 | 0.0200 | +0.0050 | 0.1250 | NOT SIGNIFICANT |

The two REFUTED verdicts are the spec's mechanical rule for a non-positive point estimate. Their
magnitudes are far inside the noise; they say "no benefit", not "harm".

**Claim 2 (practical, depth 9 P3 >= 0.10): NOT MET.** 0.0250 against the reflex's 0.0171.
Look-ahead of 3 moves does not move the depth-9 frontier off the floor.

**Claim 3 (mechanism, revisit rate, no threshold):**

| depth | G0 | E3 | P3 | R3 |
|---|---|---|---|---|
| 7 | 0.400 | 0.393 | 0.387 | 0.187 |
| 8 | 0.454 | 0.449 | 0.457 | 0.189 |
| 9 | 0.556 | 0.552 | 0.551 | 0.189 |

## Exploratory, NOT pre-registered (read as leads, not findings)

- **The goal check alone helps where the policy reaches near-solved states.** E3 - G0: depth 7
  +0.0254 (11 of 12 seeds, one-sided p 0.0022), depth 8 +0.0150 (8 of 12, p 0.0400), depth 9
  +0.0029 (7 of 12, p 0.1387). This was not a pre-registered contrast, and a 3-move goal check is
  search the network did not do, so it is a property of the procedure, not of the SNN.
- **Optimality rises under P3 at depth 9** (0.456 to 0.595) on very few solves, so P3's successes
  there are shorter. Too few solves to read further.
- **The loops look like one face turned repeatedly.** G0's modal-action fraction is 0.454,
  0.489 and **0.668** at depths 7, 8 and 9: at depth 9 two thirds of a typical episode's moves are
  a single action, and a quarter turn has order 4. Per-move traces were not recorded, so this is
  an inference from two aggregates, not an observation.
- **The goal check rarely fires** (4.6% of moves at depth 7, 1.4% at 8, 0.3% at 9). The policy
  seldom gets within 3 moves of solved at depths 8 and 9, which is why E barely moves G there.

## What this decides (roadmap section 4)

1. **Stage 2 goes ahead, and it is cheap.** The policy head cannot rank sequences, as expected
   in the spec ("summed log-probability usually picks the greedy first move"). The EXP-053 arm B
   critics (`exp053_critic_d7_*_critic.pt`) exist untracked on the VPS, so critic-guided search
   at depth 7 needs no retraining. Track them before anything else.
2. **Recommendation, not a decision: consider a visited-state rule inside the search.** Roughly
   half of all moves at depths 8 and 9 revisit a state. The memory line (EXP-059/061/063) tested
   memory as a RECALL INPUT to the policy head, and found `revisit_rate` flat (EXP-059: M minus A
   +0.0157, p 0.4450). A hard rule in the SEARCH that refuses to re-enter a visited state is a
   different mechanism and is untested. If the modal-action inference above holds, much of what it
   would do is "do not make the fourth quarter turn", which is narrower than "memory", and should
   be stated that way. The roadmap's stage order is Michael's to change; this file only records
   the evidence.

## Provenance

- Machine: SwizzlesDuo, Intel Ultra 9 185H, Windows, Python 3.13 venv, torch CPU. 20 workers for
  G/E/R/P phases, 3 for the determinism re-run, 2 for calibration.
- Cost: G/E/R cells 250 s at 3 workers, 760 to 1640 s at 20; P3 cells up to 2055 s at 20.
- Pre-flight order and both gate decisions are recorded, dated, in the spec's 2026-10-06
  amendments, committed before the full P launch.
- Seeds 0-11, held-out shells capped at 200 states, budget `2d + 3` real moves.
