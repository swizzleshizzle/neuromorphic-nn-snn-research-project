# EXP-071 results: the critic as judge, and a no-revisit rule in the search

**Run 2026-10-06** on SwizzlesDuo (laptop), main at `c568c04` (harness), gates fixed in `e6f0347`
and cost appended in `c8f1658`, both BEFORE any C or V cell. 144 cells (12 arms x 12 seeds, depth
7, 200 held-out states each) plus 12 step-0 ranking records, re-evaluation only, nothing trained.
Records are committed in `outputs/`.

Spec: `docs/superpowers/specs/2026-10-06-exp071-critic-and-no-revisit-design.md` (sections 1 to 9,
gate amendment section 10). Regenerate the verdicts with:

```bash
.venv/bin/python experiments/071_critic_and_no_revisit/aggregate.py --determinism-ok
```

> **HEADLINE 1: the critic is NOT a usable judge. Claim 1 REFUTED, decisively.** Critic-guided
> 3-move search solves **0.0729** against reflex-plus-endgame's **0.2321** (-0.1592, every seed
> worse). One-step critic choice (C1) solves **0.0175**. The critic ranks positions above chance
> (both ranking gates passed on 12 of 12 seeds) but far worse than the policy head does, and
> steering by it compounds that error over every move.
>
> **HEADLINE 2: refusing to revisit helps. Claim 2 CONFIRMED.** The reflex with a no-revisit rule
> solves **0.2346** against **0.2067** (+0.0279, 9 of 12 seeds, one-sided p 0.0115, alpha 0.025).
>
> **The strongest result is a secondary, and it is large: no-revisit UNLOCKS look-ahead.**
> Policy-guided 3-move look-ahead with the rule (P3V) solves **0.3217**, against P3's 0.2454
> (+0.0762, p 0.0002) and the plain reflex's 0.2067. That is **+0.115 at depth 7 with no
> training**, a 56% relative gain. EXP-070 found policy-guided look-ahead worth below about 0.01
> on its own; the reason was the loops, not the judge.

## Gates

| gate | verdict | evidence |
|---|---|---|
| 0(a) determinism | **PASS** | G0V, C3, R3V at seed 0 re-run into a separate directory: identical except `wall_s` and `git_commit`; C3 also identical between the det and calibration runs |
| 0(b) continuity with EXP-070 (exact) | **PASS** | all 48 base cells (G0, E3, P3, R3) equal EXP-070's committed depth-7 records in all 8 outcome fields, 0 mismatches |
| R1 child-level ranking | **PASS** | critic top child improves 0.3375 vs chance 0.2203, 12/12 seeds, p 1/4096 |
| R3 leaf-level ranking | **PASS** | critic top leaf closer 0.2329 vs chance 0.1344, 12/12 seeds, p 1/4096 |
| V engaged | **PASS** | G0V revisit rate 0.0000 vs G0 0.3997 (bar: at most 0.5x) |
| 1 resolution | RESOLVED | no primary pair on a bound |

Reference recorded in the gate amendment before any C number existed: the POLICY HEAD's own first
move improves **0.4950** of the time against the critic's 0.3375.

## Mean held-out success and revisit rate (12 seeds, depth 7)

| arm | success | revisit | | arm | success | revisit |
|---|---|---|---|---|---|---|
| G0 | 0.2067 | 0.3997 | | G0V | 0.2346 | 0.0000 |
| E3 | 0.2321 | 0.3925 | | E3V | 0.2575 | 0.0000 |
| P3 | 0.2454 | 0.3867 | | **P3V** | **0.3217** | 0.0000 |
| C1 | 0.0175 | 0.6082 | | C1V | 0.0163 | 0.0000 |
| C3 | 0.0729 | 0.3963 | | C3V | 0.0717 | 0.0000 |
| R3 | 0.0025 | 0.1874 | | R3V | 0.0033 | 0.0000 |

The V rule never had to fall back (fallback fraction 0.0000 in every V arm): an unvisited
candidate always existed, so every V rollout is loop-free by construction.

## Pre-registered claims

**Claim 1 (primary, critic track, alpha 0.025): C3 beats E3. REFUTED.** -0.1592, p 1.0000.

**Claim 2 (primary, no-revisit track, alpha 0.025): G0V beats G0. CONFIRMED.** +0.0279, p 0.0115,
9 of 12 seeds. The spec stated in advance that each primary can detect about 0.02; this effect
sits just above that.

**Secondary (a pattern, never independent confirmations):**

| contrast | diff | p | verdict |
|---|---|---|---|
| C3 - P3 | -0.1725 | 1.0000 | REFUTED |
| C3V - C3 | -0.0013 | 0.6416 | REFUTED (point estimate <= 0, inside noise) |
| C1 - G0 | -0.1892 | 1.0000 | REFUTED |
| E3V - E3 | +0.0254 | 0.0361 | NOT SIGNIFICANT at 0.025 |
| **P3V - P3** | **+0.0762** | **0.0002** | **CONFIRMED** (per-seed solves +4 to +31, 12/12) |

## Exploratory, NOT pre-registered

- **P3V - G0V: +0.0871, 12 of 12 seeds** (one-sided p 0.0002). With loops removed, scoring the
  look-ahead by the policy adds a large amount. Without the rule it adds below about 0.01
  (EXP-070). **Look-ahead and no-revisit interact; neither alone explains P3V.**
- **P3V - G0: +0.1150, 12 of 12 seeds.** The practical gain over the published reflex.
- **Quality does not degrade:** optimality G0 0.709, G0V 0.711, P3 0.748, P3V 0.732. The extra
  solves are not long detours.
- **Why the critic fails as a driver, as far as this run shows:** a one-step ranker that is right
  34% of the time against the policy head's 50% compounds over ~10 moves into near-total failure
  (C1 0.0175, and C1 revisits 61% of moves). Above chance is not the same as good enough to steer.
  The critic was trained as a baseline for policy-gradient variance reduction, not as a
  cost-to-go, which is exactly the gap stage 3 of the roadmap targets.

## What this decides

1. **The critic, as trained, is retired as a search judge.** Stage 3 (train a judge from its own
   look-ahead, DeepCubeA-style) is the route to a learned judge; this run is the evidence that a
   reused baseline critic is not one.
2. **The working recipe is now P3V:** policy-guided 3-move look-ahead plus no-revisit, on the
   existing networks, no retraining. It is the new reference arm for depth 7. The obvious next
   measurement is P3V at depths 8 and 9, where EXP-070 left the frontier (and where no critic is
   needed, so it costs only re-evaluation).
3. **No-revisit is a plain lookup table, not a spiking component.** That is stated, not hidden: the
   gain is real and the mechanism is ordinary search bookkeeping.

## Provenance

- Machine: SwizzlesDuo, Intel Ultra 9 185H, Windows, Python 3.13 venv, torch CPU. 12 workers for
  ranking, 20 for base and full phases, 3 for det, 2 for calibration.
- Cost: C3 and C3V up to 2383 s per cell at 20 workers; 830 to 880 s at 2 to 3.
- Critics: the 12 committed EXP-053 arm B critics, verified byte-identical (SHA-256) to the
  originals in the laptop's `wt-exp053` worktree before any run.
- Seeds 0-11, held-out shells capped at 200 states, budget `2d + 3 = 17` real moves.
