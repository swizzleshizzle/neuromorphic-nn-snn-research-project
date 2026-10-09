# Session Handoff - 2026-10-09 - WEEK 27. EXP-070 to 074 done; a learned judge solves deeper cubes.

> **Nothing is running. The laptop is FREE.** `main` at the commit carrying this file. Suite:
> **976 tests, 953 not slow, 23 slow** (plus 88 vitest + 2 e2e in `dashboard/`). Worktree
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

**EXP-072 then ran (2026-10-07), Michael's option 1:** `experiments/072_p3v_frontier/RESULTS.md`.
P3V at depth 8 is **0.1496** (P3 0.0854, reflex 0.0688) and at depth 9 **0.0629** (reflex 0.0171);
both primaries CONFIRMED, every gate passed. The recipe holds at depths 7, 8 and 9.

**Decision waiting on Michael**, each a short spec with R3V as its matched floor:

1. **Stage 3:** train a judge from its own look-ahead (DeepCubeA-style), the route to a learned
   "how close am I" and to scrambles beyond depth 9.
2. **Make the no-revisit function learned or spiking** (e.g. in the hippocampal region).
3. **Deeper or wider search with the same networks** (k = 4, or beam search), the cheapest way to
   push P3V further, though it scales as 6^k per move.

## 2b. EXP-074 (stage 3, a learned judge) CONFIRMED; EXP-073 superseded

EXP-073's 64-unit judge was blocked on its own training gate (issue #14). Michael chose option 2:
EXP-074, a judge reading all 192 sensory neurons (W), with EXP-073's arms kept as A (encoder
trains) and B (frozen), and a training gate (Gate L) that measures 3-move leaf ranking, which is
what search uses. Spec `docs/superpowers/specs/2026-10-07-exp074-wide-judge-design.md`, results
`experiments/074_wide_judge/RESULTS.md`.

**Both primaries CONFIRMED at depth 9 on 12 seeds and on the 10-seed sensitivity line:** J3V-W
0.1933 vs P3V 0.0629 (+0.130, 12/12 seeds) and vs J3V-A 0.1292 (+0.064). Depth 7: 0.79.
Exploratory depth 11 (typical random scramble): J3V-W 0.101 vs P3V 0.031. Every gate passed. The
judge was trained only from its own look-ahead; the BFS table is a yardstick, never an input.

**What is open for Michael to choose next** (roadmap `2026-10-05-lookahead-roadmap-design.md`):
a wider or retrained sensory region (EXP-074 section 10's option 3, now motivated since width
mattered); a deeper tree or a learned no-revisit rule; using J to train the policy; stage B
(a learned world model in place of `apply_move`). Ask before building any of them.

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
- **Value-iteration training workers use 0.7 to 0.94 GB each**, not 195 MB. 12 workers ran three
  waves of 36 runs in about 14 h; total throughput at 12 was about the same as at 6. The first
  hour of a wave is slower than its average: price a run from a whole wave, not its first probe.
- **`Start-Process -PassThru` reports `ExitCode` as `$null` unless `.Handle` is read**, so a
  launcher pool counts every cell as failed. Fixed in `launch074.ps1` (`5ca5ead`); copy that
  pattern into any new launcher. `launch073.ps1` still has the bug (never ran an eval phase).

## 4. What NOT to do

- Do not re-run P1: it is E1 by construction (spec amendment 2026-10-06).
- Do not read EXP-070's REFUTED secondary verdicts as harm; they are a non-positive point
  estimate far inside the noise.
- Do not schedule work against a publishing date. There is none.
