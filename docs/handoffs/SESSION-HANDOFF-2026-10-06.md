# Session Handoff - 2026-10-06 - WEEK 27. EXP-070 and EXP-071 done; no-revisit plus look-ahead is the best arm.

> **Nothing is running. The laptop is FREE.** `main` at the commit carrying this file. Suite:
> **844 tests, 822 not slow, 22 slow** (plus 88 vitest + 2 e2e in `dashboard/`). Worktree
> `/root/projects/.wt/report` on `phase4-report` belongs to a parallel docs session; leave it.
>
> **This supersedes the 2026-09-23 handoff and `NEXT-SESSION-PROMPT.md`**, which describe a
> project whose deliverable was a public release. It is not any more.

## 0. The direction changed on 2026-10-05

Michael dropped the "public release". **The goal is a spiking network that actually solves a
cube, eventually beyond 2x2.** The plan is look-ahead: a general procedure (imagine moves with the
simulator, compare, choose) plus a learned judgement of "how close is this position". Roadmap and
its staged decision points: `docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md`
section 4. The report continues as documentation on the parallel track.

## 1. EXP-070, the first stage, ran and closed

`experiments/070_lookahead_existing/RESULTS.md`. Re-evaluation of the tracked EXP-053/062
networks with 1-3 move look-ahead, 324 cells, nothing trained.

- **Primary NOT SIGNIFICANT:** scoring sequences with the policy head adds below about 0.01 at
  depth 8, k = 3 (P3 0.0854 vs E3 0.0838, p 0.3818). The policy head is not a usable judge.
- A bare 3-move goal check helps a little (exploratory, not pre-registered).
- The reflex policy loops on about half its moves at depths 8-9 (already known from EXP-062),
  and policy-guided look-ahead does not break the loops.

## 2. EXP-071 ran too, and the decision waiting on Michael changed

`experiments/071_critic_and_no_revisit/RESULTS.md` (depth 7, 144 cells, gates all passed).

- **The critic is not a judge:** it ranks above chance but critic-guided search is REFUTED
  (C3 0.0729 vs E3 0.2321). Retired as a judge by the roadmap's stage-2 rule.
- **No-revisit helps:** CONFIRMED, G0V +0.0279, p 0.0115.
- **Best arm: P3V at 0.3217** (policy look-ahead plus no-revisit), +0.0762 over P3 as a
  SECONDARY, 12/12 seeds. Search scaffolding around the frozen network, not network learning.

**Decision waiting on Michael**, each a short spec with R3V as its matched floor:

1. **P3V at depths 8 and 9** (re-evaluation only, no critic needed): does the frontier move?
2. **Stage 3:** train a judge from its own look-ahead (DeepCubeA-style), the route to a learned
   "how close am I".
3. **Make the no-revisit function learned or spiking** (e.g. in the hippocampal region).

## 3. Things learned this session that will bite again

- **Worktrees import MAIN's src** through the venv's editable install. Prefix `PYTHONPATH=src`.
  `.venv` is in `.git/info/exclude` so a symlinked one stays out of commits.
- **The bundled `sdd-runner` workflow assumed Windows paths and the main checkout.** Its patched
  copy takes a `cwd` arg; without it, reviewers diff `main` and pass every task on an empty diff.
- **Evaluation is stochastic** (Poisson spikes). Re-evaluation does not reproduce every published
  per-seed number; CLAUDE.md's reproducibility bullet now says exactly what does reproduce. Always
  re-evaluate a baseline rather than pairing against a published per-seed value.
- **EXP-070 records are tracked** (force-added). Untracked published records are why Gate 0 had to
  depend on files only two machines held. Do the same for future experiments' small JSON records.
- **Laptop throughput:** a cube eval cell is about 3x slower per cell at 20 workers than at 3.
  Total throughput still favours 20, but price runs from a same-worker-count measurement.

## 4. What NOT to do

- Do not re-run P1: it is E1 by construction (spec amendment 2026-10-06).
- Do not read EXP-070's REFUTED secondary verdicts as harm; they are a non-positive point
  estimate far inside the noise.
- Do not schedule work against a publishing date. There is none.
