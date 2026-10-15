# Session Handoff - 2026-10-12 - WEEK 27 closed, WEEK 28 open. EXP-076 DONE: all three claims confirmed.

> **2026-10-15: EXP-077 DONE** (section 2c): Claim 2 CONFIRMED, Claims 1 and 3 REFUTED. **EXP-078
> is PILOTED, its amendment committed (`007521f`, corrected `89c42c0`), and it waits on Michael's
> go on neuromorphic#21** (about 40 to 48 h, 6 workers). The laptop is idle. Suite: **1122 tests,
> 1097 not slow, 25 slow**. Section 2b is the run procedure; step 4 below is where it stands.
>
> **This supersedes `SESSION-HANDOFF-2026-10-06.md`** (which covers 2026-10-06 to 11 and is
> misnamed; its header says 2026-10-11). Read that one for EXP-070 to 075 and its section 3, the
> gotchas, which all still hold.

## 0. Week numbering

Session names and journals count the 2026-09-28 gap week, the vault's original plan does not. So
**week 27 = Oct 5-11 (closed), week 28 = Oct 12-18 (now)**. The vault tracker rows for those two
weeks are relabelled to match; the later plan rows are stale and left alone.

## 1. Week 27 in one paragraph

Six experiments closed and a seventh built. EXP-070 to 072: the policy head and the critic are not
judges, refusing to revisit states helps, and policy look-ahead plus no-revisit (P3V) holds at
depths 7 to 9 (depth 9 0.0629). EXP-074: a judge trained by value iteration from its own
look-ahead, reading all 192 sensory neurons, CONFIRMED at depth 9 (0.1933 vs P3V 0.0629). EXP-075:
a sensory region four times wider CONFIRMED at depth 9 (0.2571 vs 0.1933), NOT SIGNIFICANT at
depth 11 (0.1200 vs 0.1013). Sessions Mon 5 to Thu 8 and Sun 11 (Pacific nights); Fri 9 and Sat
10 were unattended laptop runs plus the EXP-075 close-out.

## 2a. EXP-076 outcome

**All three claims CONFIRMED, 12 seeds and the 10-seed line, every gate passed.** Depth 11: J4V-X
0.1950 vs J3V-X 0.1200 (+0.0750, 12/12). Depth 9: 0.4004 vs 0.2571 (+0.1433, 12/12). The 4-move
goal check alone (J3VG4-X) equalled J3V-X on all 24 cells, so the gain is the judge scoring
leaves one move nearer the goal. A 5-move full tree is about 20 h per depth-11 cell, so going
deeper needs a cheaper search (beam or pruned). Section 2 below is the run as dispatched, kept
for the relaunch procedure.

## 2b. EXP-077 then EXP-078 (Michael's decision, relayed 2026-10-14): run both, cheap first

Specs: `docs/superpowers/specs/2026-10-14-exp077-beam-search-design.md` (a width-64, depth-5 beam,
at most 1026 judge evaluations per move vs the 4-move tree's 1296; arms J4V-X reused from EXP-076,
J4VG5-X, B5-X) and `docs/superpowers/specs/2026-10-14-exp078-full-five-move-tree-design.md` (J5V-X,
7776 leaves, ~200-250 cell-hours, each cell checkpoints every 10 states and resumes). Plans beside
them in `docs/superpowers/plans/`. Review ask on #18 (non-blocking).

**Procedure once the laptop is reachable** (`tailscale status | grep swizzles`):

1. `git pull` on the laptop; `launch077.ps1 -Phase check`, then `-Phase pilot -Workers 2`.
   Read `phase_pilot.log`: the `peak commit MB` lines must be far above 2 MB (the launcher now
   samples the `python3.13` child; a WARNING line under 50 MB means it is reading the redirector).
2. `aggregate.py --pilot-report` (EXP-077), then the dated amendment: set
   `AMENDMENT = {date, workers, spec_section}` in `experiments/077_beam_search/aggregate.py`, add the
   amendment section to the spec, commit, and put it to Michael with `pm ask` (blocking for the
   full run only).
3. After his go: `chain077.ps1` (cont, Gate 0(b), eval, det; eval workers come from AMENDMENT).
4. EXP-078's pilot may run alongside EXP-077's eval: `launch078.ps1 -Phase pilot -Workers 1
   -AllowConcurrent`, if EXP-077's amendment leaves memory headroom. Then its own amendment and
   `pm ask`. Its chain (`chain078.ps1`: cont, gate, eval, det with a forced stop at 50 and resume,
   rank) starts only after EXP-077's eval has finished (spec section 9).
5. Aggregators compute every gate; EXP-078's Claims 3-4 read EXP-077's records and VOID if
   EXP-077 is invalid. Close-out: force-add records AND each `outputs_det` record (the
   `.gitignore` ignores `outputs_*`; EXP-078's aggregator reads EXP-077's det record).

**Disclosures for the RESULTS files (rulings made while building):** EXP-077's Gate B voids per cell,
stricter than the per-depth mean in the spec; `fallback_frac` is over all real moves, not beam
moves; EXP-078's Gate 0(b) compares EXP-071's 8 outcome fields (not first_fire_len or
fallback_frac); J5V-X `wall_s` is loop time only (EXP-076/077 include load time); rank workers
hold the full BFS table plus a 7776-leaf batch, so rank memory may exceed the eval pilot's.

## 2. EXP-076: what is running

A 4-move look-ahead tree with EXP-075's 12 committed X judges, nothing retrained.
Spec `docs/superpowers/specs/2026-10-11-exp076-deeper-tree-design.md` (section 12 is the dated
pilot amendment), plan `docs/superpowers/plans/2026-10-11-exp076-deeper-tree.md`.

- Arms: J3V-X (k 3, EXP-075's committed records), J3VG4-X (k 3, goal check at 4), J4V-X (k 4).
  Depths 9 and 11, seeds 0 to 11, mode C with no-revisit.
- Claims (alpha 0.05/3, exact sign-flip): 1 J4V-X vs J3V-X at d11; 2 the same at d9; 3 J4V-X vs
  J3VG4-X at d11. Gate D voids every claim; Gate G and Gate R (k 4) are reported only.
- Pilot (20 states, d11 seed 0): J4V-X 26.0 s per state, J3VG4-X 9.9 s; 41.9 cell-hours of
  evaluation in all. Real workers peak at 2.0 GB commit. Disclosed, decides nothing: J4V-X solved
  2/20, J3VG4-X 4/20.

**The chain.** `C:\Users\mlgbr\chain076.ps1` (tracked copy: `experiments/076_deeper_tree/chain076.ps1`;
`scp` it back if the laptop copy is lost) runs, in order: `cont` (2 workers) -> `aggregate.py --stage cont`, stopping
unless Gate 0(b) says PASS -> `rank` (6) -> `eval` (6) -> `det` (1). Every phase but det passes
`-SkipExisting`. Expected 11 to 15 h, so done around Tue 2026-10-13 midday Pacific.

- Chain log: `C:\Users\mlgbr\chain076.log` (one line per phase start, exit, and the Gate 0(b) line).
- Phase logs: `experiments\076_deeper_tree\phase_<phase>.log` on the laptop.
- Progress: count `exp076_*.json` in `experiments\076_deeper_tree\outputs` (2 cont, 24 rank, 48
  eval) and `^python` processes (each worker shows as `python` plus `python3.13`).

**After a Windows Update restart (about 03:31 laptop time)** the chain dies with the pool. Read the
chain log for the last phase that started, then relaunch from it; finished cells are skipped:

```bash
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\chain076.ps1 -From eval'
```

`-From` takes `cont`, `gate`, `rank`, `eval` or `det`. Run it as a background Bash call; the ssh
dies around 2-3 h with the run intact, which is not a failure signal.

**What the chain does NOT do (the close-out):**

1. Diff `outputs_det/exp076_J4V-X_d9_s0.json` against `outputs/exp076_J4V-X_d9_s0.json` on the
   outcome fields (Gate 0(a)).
2. `aggregate.py --determinism-ok` only if that diff is clean; read every gate and claim.
3. `RESULTS.md` with provenance, the Gate G note (spec section 11), the pilot disclosure, and the
   launcher memory note below.
4. `git add -f` the records (outputs, outputs_det), roadmap outcome bullet, report section 9,
   handoff, `secretary log`.

**Parked:** `launch076.ps1`'s peak-commit log reads about 2 MB because it samples the venv's
`python.exe` redirector, not the `python3.13.exe` worker. The memory probe
(`probe076m.ps1`, `^python` `PrivateMemorySize64`) is the instrument; fix the launcher before
reusing it.

## 2c. EXP-077 outcome and EXP-078 status (2026-10-15)

**EXP-077:** every gate passed, 10-seed line agrees. Depth 9: B5-X 0.5571 vs J4V-X 0.4004
(+0.1567, p 0.0002, 12/12), **CONFIRMED**. Depth 11: 0.1475 vs 0.1950 (-0.0475, B5-X wins 1 of
12), Claims 1 and 3 **REFUTED**. The beam helps near the goal and hurts far from it. J4VG5-X equals
J4V-X on all 24 cells (late fires run out of step budget). RESULTS.md `b9c23f3`; roadmap and the
playbook lesson below `a399d73`.

**Lesson, now in the playbook section 3:** a 2-minute tailnet drop killed the ssh pipe the cells
inherit, so finished cells failed on their last `print` and the chain stopped with every record
intact. Launch chains with OS-level redirection:
`ssh -n laptop 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File <chain>.ps1 > C:\Users\mlgbr\<chain>.out 2>&1"'`,
in the background from this side. Never wrap it in `Start-Process`: that died with its ssh client.

**EXP-078:** pilot J5V-X 115 s/state (6.4 h per depth-11 cell at 3 processes), 7776 judge evals
per move, 2.8 GB per worker, 5/20 solved (B5-X 3, J4VG5-X 2 on the same states). Amendment:
6 workers, 125.5 cell-hours, depth 11 first, 40 to 48 h once 6-worker contention is applied.
**On Michael's go (#21):** `pm resolve neuromorphic#21`, `git pull` on the laptop, then
`chain078.ps1` launched as above. The determinism phase stops itself at 50 states and resumes.

## 3. Week 28 (Oct 12-18)

1. ~~Collect EXP-076~~ done 2026-10-13 (section 2a).
2. **Report:** EXP-075 is now in `docs/writeup/report.md` section 9, the README and the index
   (merged `746e633`). EXP-076 goes in once it has a number.
3. **Branch `report-phase4-sync`** (a parallel docs session's header and sections 6.2, 6.3,
   8.1) was reviewed and merged (`9d63a94`), then brought up to EXP-075 with section 1's counts
   (`934abdb`). Branch and worktree deleted.
4. ~~EXP-077~~ done 2026-10-15 (section 2c). EXP-078 waits on #21.
5. **The next stage after EXP-078 is Michael's call.** The menu, from handoff 2026-10-06 section 2c plus
   EXP-076: a cheaper deeper search (beam or pruned 5-move), since depth pays off; wider still or a supervised-then-VI region; a learned no-revisit rule;
   using J to train the policy; a learned world model (stage B). Ask before building any.

## 4. What NOT to do

- Do not read the pilot's 2 vs 4 solves as a result; the full cells reversed it.
- Do not schedule work against a publishing date. There is none.
