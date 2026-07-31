# Session Handoff - 2026-07-30 (Thu) -> first VPS session

> Pickup point for the move from the desktop to the VPS. **Two PRs are open and unmerged**; nothing is
> stranded, but nothing from this session is on `main` yet either.
>
> Read `CLAUDE.md` first for standing knowledge, then
> `docs/playbooks/remote-experiment-runs.md` before dispatching anything to the laptop. That playbook is
> new and it matters: **the old `192.168.50.62` LAN address is unreachable from a VPS.**

## 0. Start here

```bash
git fetch --all --prune
gh pr list --state open          # expect #8 and #9
```

On the VPS the repo commands differ from the Windows ones in `CLAUDE.md`:

| | Windows (old sessions) | Linux VPS |
|---|---|---|
| python | `.venv\Scripts\python.exe` | `.venv/bin/python` |
| fast suite | `.venv\Scripts\python.exe -m pytest tests/ -q -m "not slow"` | `.venv/bin/python -m pytest tests/ -q -m "not slow"` |

First-time VPS setup: create the venv and install from `requirements.txt`, then run the fast suite once
to confirm the environment before trusting any result from it.

## 1. What shipped this session

Nothing merged to `main`. Two independent branches, both reviewed and both recommended SHIP:

| PR | Branch | Contents |
|---|---|---|
| [#8](https://github.com/swizzleshizzle/neuromorphic-nn-snn-research-project/pull/8) | `week17-exp030-results` | `experiments/030_memory_engagement/RESULTS.md`, 1 commit |
| [#9](https://github.com/swizzleshizzle/neuromorphic-nn-snn-research-project/pull/9) | `week17-dashboard-cube` | Cube support for the trace contract and dashboard, 18 commits |

Verification at the tip of #9: **python 370 passed** (full suite including the slow BFS test),
**dashboard 20 files / 84 tests**, `npm run build` zero TypeScript errors.

## 2. EXP-030 concluded: memory content does nothing

144 records, 12 seeds, run on the laptop 2026-07-29. Full numbers and the claim-by-claim contract check
are in `experiments/030_memory_engagement/RESULTS.md` (on branch `week17-exp030-results`).

Depth-2 arm ordering, where all the headroom is:

**shuffled 20.4% < amnesic 29.9% ~ memory 31.2% < concept 38.0%**

| comparison | depth 2 | sd | W-L-T | exact p |
|---|---|---|---|---|
| `memory` minus `shuffled` (PRIMARY) | +10.8 pts | 18.1 | 8-3-1 | 0.078 |
| `memory` minus `amnesic` (CONTENT) | +1.2 pts | 28.0 | 5-6-1 | 0.908 |
| `memory` minus `concept` (secondary) | -6.8 pts | 36.1 | 4-6-2 | 0.535 |

Removing stored content entirely costs 1.2 points. Feeding a wrong state costs 10.8. **The
pre-registered primary comparison is positive and it measures the harm of incorrect memory, not the
benefit of correct memory.** With three arms this would have read as "memory content helps"; the
`memory_amnesic` arm inverted the conclusion.

**Second, independent null: memory did not reduce cycling.** The memory arm revisits slightly more than
plain concept at every depth (0.368 vs 0.327 at depth 2; 0.609 vs 0.604 at depth 3) despite the gate
establishing abundant cycles. That localises the failure to the readout rather than to storage.

Gate numbers, measured before the memory arms ran: greedy revisit 0.089 / 0.327 / 0.604 at depths 1/2/3.
At depth 3, 7 of 12 seeds sat at exactly 0.667, which on a 9-step budget is 3 unique states: an absorbing
cycle entered on step one.

## 3. THE JOB NEXT: test the policy-collapse hypothesis before anything else

While recording a depth-3 trace for the dashboard, **the policy played `F'` nine times in a row.** `F'`
has order 4, so four of them return the cube to its start: an absorbing 4-cycle, revisit rate 0.556.

If the policy has collapsed to a constant action then it is not reading its input at all, and no change
to the feature vector (memory or otherwise) could alter behaviour. **That would make the depth-3 memory
null a fact about a degenerate policy rather than about memory.** It is consistent with `CubeConfig`'s
defaults (`entropy_beta=0.0`, `normalize_advantages=False`), which is precisely the collapse EXP-025
already fixed once with advantage normalisation plus entropy.

Caveat, and it is a real one: **n = 1 seed, from the recorder's own training, and depth 2 does not show
it** (that trace played `F'` then `R`). This is a lead, not a result.

Cheapest decisive test: measure action entropy, or the fraction of steps taking the modal action, across
EXP-030's existing seeds at each depth. If depth-3 policies are constant-action, re-run engagement with
`entropy_beta > 0` and advantage normalisation before drawing any further conclusion about memory.

## 4. Dashboard cube support (PR #9)

The handoff called this "the cube task panel is grid-hardcoded". Measuring found five independent breaks,
the worst being that `runner.py` read facelet colours as x/y coordinates, which does not raise, it
renders a plausible wrong picture.

New library surface:

| Piece | Path |
|---|---|
| `TaskAdapter` protocol, `GridworldAdapter`, `CubeAdapter` | `src/neuromorphic/monitor/tasks.py` |
| Adapter-driven header/frame/encoding | `monitor/schema.py`, `frame.py`, `runner.py` |
| Cube trace recorder | `scripts/record_cube_trace.py` |
| Unfolded-net geometry (pure) | `dashboard/src/panels/cubeNet.ts` |

`SCHEMA_VERSION` is now `1.1`. Frames carry the task type stamped at ingest, in both `parseTrace` and the
store, because the live WebSocket path bypasses `parseTrace` entirely.

**Cube frames describe the POST-move state.** `facelets`, `solved` and `distance` all describe the same
state and the final frame shows the solved cube. This was a defect found from real output: frames used to
pair pre-move facelets with post-move status, so a scrambled cube rendered as "solved: yes" and the solved
state never rendered at all.

Regenerate a trace (it is gitignored, per `dashboard/.gitignore`):

```bash
.venv/bin/python scripts/record_cube_trace.py --depth 2 --seed 0 --episodes 600 --out dashboard/public/cube_trace.jsonl
```

Depth 2 solves in 2 moves. Depth 3 gives 9 frames and shows the absorbing cycle from section 3.

## 5. Open debt

1. **Nothing in PR #9 has been seen rendering in a browser.** All 84 dashboard tests pass and the build is
   clean, but no browser was driven. Visual correctness is asserted by tests only.
2. **`cubeNet.ts` applies the same row-major ordering to all six faces**, so the B and D faces are probably
   not oriented for a physically coherent unfolded net. Cosmetic, needs eyes.
3. **Move correctness and net geometry are each pinned independently; their composition is not.** No test
   asserts that a facelet moved by a real permutation lands in the geometrically correct net cell.
4. The held DLB corner highlight (facelets 12, 16, 21) renders but no test pins its placement.
5. **The shuffle control is diluted at shallow depth.** `unshuffled_frac` was 0.321 at depth 1 and 0.152 at
   depth 2: that fraction of steps fell back to the current state because fewer than two states had been
   visited. Design it out of any follow-up shuffle control.
6. Live trace streaming during training remains a follow-up spec, not built. It needs a decision about
   which of N parallel worker runs emits.
7. EXP-029 test debt, still open from the 2026-07-27 handoff:
   `test_random_arm_scores_above_zero_but_well_below_one` asserts `0.0 <= x <= 1.0`, which cannot fail.

## 6. Process notes worth carrying

**The fourth control arm paid for the whole experiment.** `memory` vs `memory_shuffled` was the
pre-registered primary and on its own it would have licensed a false claim. The habit that saved it is
already in `CLAUDE.md`: ask what a control holds fixed besides the thing you named.

**Look at real output, not just green tests.** The "solved: yes on a scrambled cube" defect passed every
unit test in the suite. Two minutes reading an actual recorded trace found it. Same pattern as the
2026-07-27 filename collision, which was visible in an implementer's own smoke output.

**Plan and spec text was again the largest defect source.** On this branch: a `zip` loop whose length
guard sat in a different test (would pass vacuously), a spec requirement mapped onto one component and
silently dropped for another, and a stub test body that referenced a helper that did not exist. Writing
complete code into plans transfers the author's errors at high fidelity; the review loop is what makes it
survivable. Six defects were caught by review on this branch, three of them from plan text.

**A mechanism null beats a performance null.** "Memory did not help" is weak. "Memory was present, cycles
were abundant, and cycling did not decrease" localises the failure.

## 7. Pointers

- Standing knowledge: `CLAUDE.md`
- **Remote runs: `docs/playbooks/remote-experiment-runs.md` (new, read before dispatching)**
- EXP-030 design and pre-registration: `docs/superpowers/specs/2026-07-27-memory-engagement-design.md`
- EXP-030 results: `experiments/030_memory_engagement/RESULTS.md` (PR #8)
- Cube dashboard design: `docs/superpowers/specs/2026-07-29-dashboard-cube-support-design.md` (PR #9)
- EXP-029 results: `experiments/029_cube_baseline/RESULTS.md`
- Architecture: `docs/architecture-spec-v3.md`
- Previous handoff: `docs/handoffs/SESSION-HANDOFF-2026-07-27.md`
- Vault: `Weekly Notes/week-17-memory-engagement.md`
