# EXP-076: a deeper look-ahead tree with the committed wide judges

**Written 2026-10-11, week 28.** EXP-075 (`experiments/075_wide_region/RESULTS.md`) confirmed that
the wide judge X beats every earlier judge at depth 9, but at depth 11, the typical random
scramble, its lead over W (+0.019, p 0.076) was not significant and it solves about one cube in
eight. Michael chose the "deeper tree" option from the EXP-075 handoff (section 2c). This
experiment retrains nothing: it searches four moves ahead instead of three with EXP-075's 12
committed X judges. Roadmap: `docs/superpowers/specs/2026-10-05-lookahead-roadmap-design.md`.

## 1. Why a deeper tree, and what it confounds

The judge is good near the goal and poor far from it. J3V-X solves 0.891 at depth 7, 0.257 at
depth 9 and 0.120 at depth 11. Its Gate R leaf ranking (lowest-J 3-move leaf closer to solved than
the root) is 0.833 against chance 0.134 at depth 7, but 0.570 against chance 0.521 at depth 11.
A 4-move tree scores positions one move nearer the goal along the best line, which is where the
judge is more reliable.

**The confound.** `choose_move` does two things with k: it takes a solve if one exists within k
moves (`solving_prefix`), and otherwise scores the 6^k leaves with the judge. Raising k to 4
extends both. A gain from J4V-X over J3V-X could come entirely from the longer goal check, with
the judge contributing nothing. A random-judge control cannot separate them: R3V solves 1 cube in
2400 at both depths 9 and 11 (`exp072_R3V_d9_*`, `exp074_R3V_d11_*`), so a 4-move random control
would also sit at the floor. Instead the two parts are separated by a third arm that extends only
the goal check (section 3).

## 2. Cost, measured before this spec

One real move at a depth-11 held-out state, seed 0, X judge, mode C with no-revisit, on the VPS
(2026-10-11, scratch probe, one state):

| k | leaves | time per move |
|---|---|---|
| 3 | 216 | 0.13 s |
| 4 | 1296 | 0.49 s |
| 5 | 7776 | 4.06 s |

On the laptop a full J3V-X cell took about 2400 s at depth 11 and 1780 s at depth 9 (EXP-075
records, `wall_s`), so about 0.7 s per real move including the policy's own `brain.step`. If the
judge's share of that scales by the measured 3.8x, a J4V-X cell is 1.3 to 2.5 h at depth 11.

**k = 5 is out of scope by cost:** 31x the k = 3 judge cost, roughly 20 h per depth-11 cell. The
pilot (section 6) replaces these estimates for k = 4.

## 3. Arms (every arm uses EXP-075's committed X judge for that seed, read through the 576-unit readout)

| arm | goal check (`k_goal`) | judge tree (`k`) | source |
|---|---|---|---|
| **J3V-X** | 3 | 3 | EXP-075's committed records, reused under Gate 0(b) |
| **J3VG4-X** | 4 | 3 | new cells |
| **J4V-X** | 4 | 4 | new cells |

Everything else is EXP-075 section 8 unchanged: the held-out sets, budgets (`2d + 3`, so 21 at
depth 9 and 25 at depth 11), the evaluation stream, the per-step imagination seeds, the
no-revisit rule, and the depth-11 cell (the seed's depth-9 policy agent, head and train seed).
Depths 9 and 11, seeds 0 to 11, paired across arms.

**What each difference measures.**
- J3VG4-X minus J3V-X: what the longer goal check alone buys. The two arms play identical moves
  until J3VG4-X first finds a solve exactly 4 moves away, because `solving_prefix` tries shorter
  lengths first and every other input to the move is the same. From that move it solves for
  certain. So J3VG4-X solves **at least as many** states as J3V-X on every cell (Gate D).
- J4V-X minus J3VG4-X: what scoring the 4-move leaves with the judge adds, with the goal check
  held at 4.

## 4. Code change

`choose_move` and `evaluate_lookahead` gain `k_goal: int | None = None`, meaning `k_goal = k`.
`solving_prefix` is called with `k_goal` instead of `k`; nothing else changes. Defaults are
therefore byte-identical to today, which a unit test pins and Gate 0(b) checks on the laptop.
`k_goal < k` is refused (ValueError): it would let the judge tree see past the goal check, a
regime nobody asked for.

`evaluate_lookahead` also records `first_fire_len`: a histogram (length -> episodes) of the length
of the solving prefix at each episode's FIRST goal fire. `fired4` is that histogram's count at
length 4. These are new keys. `k_goal` is recorded too.

## 5. Disclosure

The cost probe in section 2 ran one move on one depth-11 held-out state of seed 0 and did not look
at its outcome. The pilot (section 6) runs the first 20 held-out states of seed 0 at depth 11 for
both new arms and does see their outcomes. **Every primary therefore prints EXP-074's 10-seed
sensitivity line (seeds 0 and 3 dropped)** next to the 12-seed verdict, as EXP-075 did. If the two
disagree in verdict, RESULTS.md leads with the disagreement.

## 6. Pilot (seed 0, depth 11, first 20 held-out states, into `outputs_pilot/`)

Run J4V-X and J3VG4-X with `--limit-states 20`, plus one rank cell (J-X, k 4, depth 11, seed 0,
limit 20). Measure seconds per state for each arm, and **peak committed memory per worker**
(`PrivateMemorySize64`, not working set: EXP-075's pool died budgeting by working set).

**A dated amendment then fixes**, before any full cell runs:
- the worker count, keeping total committed memory under 24 GB and at most 6 workers (EXP-075:
  more than 4 to 6 single-thread workers bought no throughput);
- the schedule.

No gate threshold is set from the pilot. Every gate below has its form and threshold fixed here.

## 7. Gates

**Gate 0.**
- (a) Determinism: J4V-X at depth 9, seed 0, re-run into `outputs_det/`. Identical except
  `wall_s` and `git_commit`.
- (b) Continuity: J3V-X at seed 0, depths 9 and 11, re-run on the laptop through this
  experiment's code with the default `k_goal`, must equal EXP-075's committed records in every
  outcome field (56/200 and 30/200). This licenses reusing J3V-X's records for every claim, and
  it proves the new parameter's default changes nothing in the regime that matters.

**Gate D (dominance; the instrument check).** J3VG4-X `solved` >= J3V-X `solved` on every one of
the 24 (depth, seed) cells. It holds by construction (section 3), so a violation means the stream
or the goal check is broken. **Any violation voids every claim** and is reported as a bug.
Maximum attainable: it passes whenever the code is right, so it cannot be unsatisfiable.

**Gate 1 (resolution).** A contrast is UNRESOLVED if both arms are below 0.02 or above 0.98,
returned by the verdict function, using EXP-070's `gate1_verdict`. J3V-X is 0.120 and 0.257, so
the J3V side is clear of both bounds.

**Gate G (does the longer goal check fire at all).** For J3VG4-X minus J3V-X to be readable, the
4-move goal check has to fire. If summed `fired4` over the 12 seeds of J3VG4-X at a depth is 0,
the decomposition at that depth is UNRESOLVED (returned as a verdict by the aggregator), and
Claim 3's reading is limited to "the judge's 4-move tree against a goal check that never fired".
This is a reported verdict, not a voiding gate: Claims 1 and 2 do not depend on it.

**Gate R (reported, not voiding).** Per depth, the per-seed fraction of held-out states whose
lowest-J **4-move** leaf is closer to solved than the root, against its chance rate, exact
one-sided sign-flip p < 0.05. BFS is the yardstick only (`ExactBFSDistance(max_depth=None)`,
since depth + 4 reaches 15, past the 2x2's diameter). EXP-074 and EXP-075 gated their claims on
Gate R. It is reported here but voids nothing. In EXP-075 root leaf ranking did not predict the
episode-level difference (X, Y and W within 0.018 at depth 9 while X solved 0.06 to 0.09 more).
And Claim 3 measures the judge's contribution directly, which Gate R only approximates.

## 8. Pre-registered claims (one-sided exact sign-flip, 4096 flips; alpha 0.0167 each, 0.05 / 3)

**Claim 1 (primary): a deeper tree solves more of the typical random scramble.**
mean(J4V-X - J3V-X) > 0 at depth 11.

**Claim 2 (primary): a deeper tree solves more at depth 9.**
mean(J4V-X - J3V-X) > 0 at depth 9.

**Claim 3 (primary): the judge's 4-move tree adds to the longer goal check.**
mean(J4V-X - J3VG4-X) > 0 at depth 11.

Verdicts: CONFIRMED, REFUTED (mean <= 0), NOT SIGNIFICANT, UNRESOLVED (Gate 1), VOID (Gate 0 or
Gate D). Each primary also prints its 10-seed sensitivity line (section 5).

**Secondary (patterns, never confirmations):**
- Claim 3 at depth 9.
- J3VG4-X minus J3V-X at depths 9 and 11, with summed `fired4`. Positive by construction (Gate D),
  so only its size is informative.
- Gate R at k 4 against EXP-075's k 3 rates.
- Mean steps to solve and optimality per arm. A deeper tree should waste fewer moves.

**What the primaries can detect.** EXP-075's paired per-seed sd between X and W was 0.039 at depth
9 and 0.041 at depth 11, so se is about 0.012 at n = 12. At alpha 0.0167 a true difference of about
0.03 is detectable. A J4V-X vs J3V-X pairing shares the judge, so its sd is probably smaller, but
that is not assumed. **Unresolved band:** a CONFIRMED claim whose mean difference is under 0.01 is
reported by the aggregator as CONFIRMED (SMALL), a verdict of its own. That distinguishes a real
but practically negligible gain from one that matters to the goal of solving the cube.

## 9. Order

1. Build and test (VPS):
   - `k_goal` in `choose_move` and `evaluate_lookahead`, with defaults byte-identical, `k_goal < k`
     refused, and `first_fire_len` recorded;
   - the EXP-076 evaluation and rank drivers (EXP-075's judge loading and cells, reused directly);
   - the aggregator with all three primaries, the sensitivity lines, Gate 1, Gate D and Gate G as
     verdicts, and CONFIRMED (SMALL);
   - a launcher from EXP-075's, with phases `pilot`, `cont`, `rank`, `eval`, `det`.
2. Pilot on the laptop (section 6).
3. Dated amendment: worker count and schedule. Shown to Michael and committed before step 4.
4. Gate 0(b) cells, then the 24 rank cells, then the 48 new evaluation cells (J4V-X and J3VG4-X x
   depths 9, 11 x seeds 0-11), then Gate 0(a).
5. Aggregate. Commit RESULTS.md and the records.

## 10. Not in scope

- k = 5 or deeper (section 2).
- Pruned or beam search: a different procedure, not a deeper one.
- Retraining any judge, or the Y and W judges.
- Depths 7 and 8: J3V-X already solves 0.89 and 0.52 there; the goal is depth 11.
- A learned no-revisit rule, training the policy from J, a learned world model, 3x3.

## 11. Amendment 2026-10-11: from the final code review, before any pilot or full cell runs

No pilot record and no evaluation number existed when this was written.

- **Gate D gains two instrument clauses.** A move changes the distance to solved by at most 1,
  and every start is at depth 9 or 11. So the first time a walk comes within 4 moves of solved,
  it is exactly 4 away, and the goal check fires there. Two more things therefore hold by
  construction, on every (depth, seed) cell:
  - (a) for J3VG4-X and J4V-X, `first_fire_len` has no key other than "4";
  - (b) J3VG4-X `fired4` >= J3V-X `solved`, because every J3V-X solve passes distance 4, where
    J3VG4-X fires.
  A violation of either is a bug, exactly like a dominance violation, and **voids every claim**.
- **Gate G cannot in practice be UNRESOLVED**, by clause (b): summed `fired4` is at least J3V-X's
  summed solves, which is in the hundreds. It stays in the aggregator, and RESULTS.md reports
  only its size.
- **Peak committed memory is logged by the launcher** per worker (5 s polls, so a peak between
  polls can be missed). This replaces a hand-taken sample.
- **The launcher refuses more than 6 workers.**
- **Rank `wall_s` includes the unbounded BFS build** (about 67 s per process), so the pilot's
  rank seconds per state are conservative.
- **Gate 0(b)'s field list is EXP-071's `OUTCOME_FIELDS`**, which omits `fallback_frac`, as in
  EXP-074 and EXP-075. The code-path test compares `fallback_frac` too.
