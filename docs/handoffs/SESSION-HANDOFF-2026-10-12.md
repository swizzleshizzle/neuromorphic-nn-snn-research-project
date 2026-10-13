# Session Handoff - 2026-10-12 - WEEK 27 closed, WEEK 28 open. EXP-076 is RUNNING on the laptop.

> **EXP-076 IS RUNNING.** Launched Mon 2026-10-12 20:05 laptop time, after Michael said go on
> issue #17 (resolved). `main` at the commit carrying this file; the laptop checkout is at
> `79930b2`, which is all the run needs. Suite: **1059 tests, 1034 not slow, 25 slow** (plus 88
> vitest + 2 e2e in `dashboard/`).
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

## 3. Week 28 (Oct 12-18)

1. Collect EXP-076 (section 2's close-out).
2. **Report:** EXP-075 is now in `docs/writeup/report.md` section 9, the README and the index
   (merged `746e633`). Section 1 still says 74 experiments and 58 with results; the index says 76
   and 59. EXP-076 goes in once it has a number.
3. **Branch `report-phase4-sync`** (worktree `/root/projects/.wt/report-phase4-sync`) holds one
   unmerged commit from 2026-10-09, a parallel docs session: header and sections 6.2, 6.3, 8.1
   brought in line with Phase 4. Not this session's; ask Michael or review and merge it.
4. **The next stage is Michael's call.** The menu, from handoff 2026-10-06 section 2c and
   whatever EXP-076 says: wider still or a supervised-then-VI region; a learned no-revisit rule;
   using J to train the policy; a learned world model (stage B). Ask before building any.

## 4. What NOT to do

- Do not start a new experiment on the laptop while EXP-076 holds it.
- Do not read the pilot's 2 vs 4 solves as a result.
- Do not schedule work against a publishing date. There is none.
