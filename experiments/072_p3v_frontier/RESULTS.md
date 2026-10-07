# EXP-072 results: look-ahead plus no-revisit moves the depth 8-9 frontier

**Run 2026-10-07** on SwizzlesDuo (laptop), main at `f72b796`. 72 new cells (G0V, P3V, R3V x depths
8 and 9 x 12 seeds) plus 4 continuity cells, re-evaluation only, nothing trained. Paired
references are EXP-070's committed G0, E3, P3 and R3 records. Records are committed in `outputs/`.

Spec: `docs/superpowers/specs/2026-10-07-exp072-p3v-frontier-design.md`. Regenerate the verdicts:

```bash
.venv/bin/python experiments/072_p3v_frontier/aggregate.py --determinism-ok
```

> **HEADLINE: both primaries CONFIRMED.**
>
> - **Claim 1, depth 8:** policy-guided 3-move look-ahead with the no-revisit rule (P3V) solves
>   **0.1496** against look-ahead without the rule (P3) **0.0854**: +0.0642, one-sided p 0.0005,
>   11 of 12 seeds better and one tied. This replicates EXP-071's depth-7 secondary at a new depth.
> - **Claim 2, depth 9:** P3V solves **0.0629** against the reflex's **0.0171**: +0.0458,
>   p 0.0002, 12 of 12 seeds. Depth 9 leaves the floor (Gate 1 resolved: P3V clears 0.02).
>
> As in EXP-071, the gain is search scaffolding (the simulator, a 3-move tree, a hand-coded
> visited table) around the same frozen networks. Nothing was trained; the SNN learned nothing new.

## Gates

| gate | verdict | evidence |
|---|---|---|
| 0(a) determinism | **PASS** | P3V and G0V at seed 0, both depths, re-run into a separate directory: identical except `wall_s` and `git_commit` (4 of 4) |
| 0(b) continuity with EXP-070 | **PASS** | G0 and P3 at seed 0, both depths, re-run under this harness: equal to EXP-070's records in all 8 outcome fields, 0 mismatches |
| V, depth 8 | **PASS** | G0V revisit 0.000 vs G0 0.454 |
| V, depth 9 | **PASS** | G0V revisit 0.000 vs G0 0.556 |
| 1 resolution | RESOLVED | both primaries clear the 0.02 floor |

The rule never fell back (fallback fraction 0.0 in every V cell).

## Mean held-out success (12 seeds, 200 states each; G0, E3, P3, R3 from EXP-070)

| depth | G0 | G0V | E3 | P3 | **P3V** | R3 | R3V |
|---|---|---|---|---|---|---|---|
| 8 | 0.0688 | 0.0838 | 0.0838 | 0.0854 | **0.1496** | 0.0013 | 0.0013 |
| 9 | 0.0171 | 0.0283 | 0.0200 | 0.0250 | **0.0629** | 0.0008 | 0.0004 |

Per-seed solves (of 200), Claim 1, P3V - P3 at depth 8: 24, 10, 16, 10, 21, 5, 1, 24, 15, 0, 12, 16.
Claim 2, P3V - G0 at depth 9: 5, 7, 4, 15, 13, 11, 10, 9, 12, 13, 2, 9.

## Secondary (a pattern, never confirmations)

The aggregator prints the primary verdict ladder beside these for reference; the spec reserves
CONFIRMED for the two primaries.

| contrast | diff | p |
|---|---|---|
| depth 8, P3V - G0 | +0.0808 | 0.0002 |
| depth 9, P3V - P3 | +0.0379 | 0.0002 |
| depth 8, G0V - G0 | +0.0150 | 0.0547 |
| depth 9, G0V - G0 | +0.0113 | 0.0508 |
| depth 8, P3V - G0V | +0.0658 | 0.0002 |
| depth 9, P3V - G0V | +0.0346 | 0.0005 |

The pattern matches EXP-071 at depth 7: the rule alone gives a small gain (here just short of
significance at both depths), and look-ahead adds a large gain only once the rule is in place.

## Exploratory, NOT pre-registered

- **Quality:** P3V optimality 0.693 at depth 8 (G0 0.718) and 0.609 at depth 9 (G0 0.456). At
  depth 8 the extra solves are slightly longer on average; at depth 9 they are shorter. Neither is
  a pre-registered quantity.
- **Across EXP-071 and EXP-072,** P3V over the reflex is +0.115 at depth 7, +0.081 at depth 8 and
  +0.046 at depth 9. The absolute gain shrinks with depth as the reflex's base rate does; the
  relative gain grows (1.56x, 2.17x, 3.68x). Three depths from two experiments, so read as a trend.

## What this decides

The recipe of policy-guided look-ahead plus no-revisit holds at every depth tested (7, 8, 9) on
the existing networks. It does not reach the goal: a random 2x2 is usually 11 moves from solved,
and depth 9 is at 6%. Choosing the next stage is Michael's; the roadmap's remaining routes are a
learned judge (stage 3), a learned or spiking form of the no-revisit rule, and deeper search.

## Provenance

- Machine: SwizzlesDuo, Intel Ultra 9 185H, Windows, Python 3.13 venv, torch CPU. 4 workers for
  continuity and determinism, 20 for the full run.
- Cost: P3V up to 2600 s per cell at 20 workers.
- Seeds 0-11, held-out shells capped at 200 states, budget `2d + 3` real moves.
