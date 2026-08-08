# Session Handoff - 2026-08-08 (Sat) - Week 19 session 1

> **EXP-038 is IN FLIGHT on the laptop.** Dispatched 2026-08-07 20:15:11 at commit `6416379`,
> 48 runs, 10 workers, ~21 h, **ETA about 17:15 Saturday**. The repo is clean and pushed.
>
> Read `CLAUDE.md` first, then this file. Strategy is **not** in this repo: it is the vault at
> `300 Efforts/Active/Coding/Neuromorphic Development/road-to-a-solved-cube.md`.

## 0. State check

```bash
git log --oneline -1                 # expect 6416379
git status --short                   # expect clean
ssh -n laptop 'powershell -NoProfile -Command "(Get-ChildItem C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\038_depth6_collapse\outputs -Filter *.json).Count"'
```

Expect the record count climbing toward **48**. The laptop travels with Michael and is
frequently off the tailnet; **an ssh timeout is not a problem signal** - check
`tailscale status | grep swizzlesduo` for last-seen before concluding anything. An SSH drop
does not kill the run (Windows has no SIGHUP semantics); `exit code 255` means the connection
dropped, not the job.

**Health signal, in order of trustworthiness:**

1. **Effective cores over a fixed interval** - the only reliable one. Measured 7.06 at launch.
   ```bash
   ssh -n laptop 'powershell -NoProfile -Command "$a=(Get-Process | Where-Object { $_.Name -match \"^python\" } | Measure-Object -Property CPU -Sum).Sum; Start-Sleep -Seconds 45; $b=(Get-Process | Where-Object { $_.Name -match \"^python\" } | Measure-Object -Property CPU -Sum).Sum; Write-Output (\"effective_cores=\" + [math]::Round(($b-$a)/45,2))"'
   ```
2. Record count in `outputs/`.
3. **Worker count is NOT a health signal** - a hung pool looks identical to a working one.

> [!warning] Do not compute utilisation from wall time you inferred between tool calls
> I did exactly that mid-session and concluded the pilot was running at 15% and possibly
> broken. A clean 60-second sample said **7.34 effective cores**, i.e. 73%, matching the
> documented 74.2%. **An agent has no reliable clock between tool calls.** Sample over an
> interval you explicitly control, inside a single command.

## 1. What this session did

| | |
|---|---|
| `470419f` | **The dispatch pull trap, fixed structurally** and verified against the live failure |
| `32b72a4` | EXP-038 pre-registration, aggregator, and its instrument test |
| `6416379` | EXP-038 driver + launcher, and the two corrections the pilot forced |

### The pull trap is closed

`scripts/laptop/sync_repo.ps1` moves **only** the untracked files the incoming tree actually
contains into a timestamped attic at `C:\Users\mlgbr\repo-attic\<stamp>\`, then **verifies**
`git rev-list --count HEAD..origin/main` is 0 and exits non-zero otherwise. It is **not**
`git clean`, which would destroy the gitignored `outputs/*.json` records that are frequently
an experiment's only copy. It aborts rather than touching modified tracked files.

Verified against the real thing: the laptop was at `022d8b8` with **48 colliding `_head.pt`
files** and reached `66f7c5d` cleanly. `launch038.ps1` now calls it and **aborts the dispatch**
if it exits non-zero, then separately greps the driver for `B_TOP` - the commit being right
does not prove the code is.

Playbook section 1 is rewritten around this. Todo `^3ef2` (exit-255 guidance) is still open.

## 2. EXP-038 - what it asks and how to read it

**Question:** EXP-032 refuted the trainer stabilizers at depth 3, where EXP-031 had shown the
failure was *not* collapse-limited. At depth 6 it demonstrably is (modal 0.975 in EXP-036,
0.982 under 3x episodes in EXP-037, which also ruled out starvation). Same intervention, and
this time the diagnosis matches the fix.

**Arms:** 48 runs. Depth 6 x beta {0.05, 0.2, 0.8} x 12 seeds, plus depth 5 at beta 0.2 x 12.
`normalize_advantages` pinned True. All four comparators are **EXP-036 cells, not re-run**.

**Depth 5 is in the design for statistical power**, not curiosity: depth 6 has twelve seeds at
exactly 0.0000 with 1/200 resolution, so a partial effect is invisible there. Depth 5 is also
BROKEN but sits at 0.0396 +- 0.027 where an effect is measurable.

```bash
.venv/bin/python experiments/038_depth6_collapse/aggregate.py
```

> [!important] The measurement decision that defines this experiment
> **At depth 6 the measured random floor (0.0008) is ABOVE the trained result (0.0000).** So
> "did success rise above the baseline?" is a check that cannot distinguish the states it
> exists to separate - a policy that is merely more random passes it, and EXP-032 Finding 3
> established that is exactly how the entropy bonus operates.
>
> Claim 1 is therefore paired against the **random arm**. **EXP-037 Claim 4's wording ("any
> seed above zero is worth reporting") carries this defect and is superseded.**

**The null is pre-committed as a result** (Claim 5). Depth 6 was the strongest remaining case
for the stabilizers. If Claims 1 and 3 both refute, they are **CLOSED** and join width,
volume alone, curriculum weighting and starvation. Do not re-describe a null as "inconclusive,
worth another sweep".

## 3. The pilot earned its 37 minutes, twice

A calibration pilot ran first (depth 6, 1,000 episodes, 10 runs, 19:31-20:09) so the betas came
from measurement rather than a guess. It corrected **two things the spec had wrong**, before any
EXP-038 number existed. Full detail in spec section 5a.

| beta | normalize | modal | entropy | success | % of ceiling |
|---|---|---|---|---|---|
| 0.0 | False | 1.000 | 0.436 | 0.0000 | 24% |
| 0.0 | True | 1.000 | 0.211 | 0.0000 | 12% |
| 0.05 | True | 0.848 | 0.602 | 0.0000 | 34% |
| 0.2 | True | 0.776 | 1.365 | 0.0000 | 76% |
| 0.8 | True | **0.675** | **1.698** | 0.0000 | **95%** |

**A. The 0.354 uniform modal anchor is the 9-step figure and does not apply at depth 6.**
`modal_action_fraction`'s own docstring says it is 0.354 over 9 steps and 0.429 over 5 - it
falls as the budget lengthens. Depth 6 runs `2d+3 = 15` steps, depth 5 runs 13. Measured from
EXP-036's random arms: **0.309 (d6), 0.321 (d5)**. The draft would have compared depth-6
policies against a depth-3 constant. **Todo `^bbd0` asks whether other experiments carry the
same wrong-budget anchor.**

**B. Entropy saturates while greedy modal fraction does not.** At beta=0.8, entropy is at 95%
of ceiling while modal has only fallen to 0.675. They are **different axes**:
`mean_train_entropy` describes the **stochastic sampled policy during training**;
`greedy_modal_action_frac` describes the **deterministic argmax at evaluation**
(`evaluate_states` does greedy rollouts). Worse, pushing beta higher would likely drive modal
**back up**, because flat logits make argmax a deterministic tie-break, i.e. a constant action.

So the original instrument check ("the top beta must reach uniform modal") was **unreachable in
principle**, not merely unreached. It is restated as **entropy saturation >= 90% of ceiling**,
which is reachable and measured. Without the pilot this would have been discovered in the
aggregator after 21 hours, against a rule that could never fire.

**C. The short budget made the policy MORE collapsed, not less** - the pilot's `(0.0, False)`
cell reads modal **1.000** against EXP-036's **0.975** at 10k, the opposite of what the spec
predicted. So the pilot's modal figures are an **upper bound** on the sweep's.

This adds a third pattern to a relationship the project keeps re-measuring:

| source | pattern |
|---|---|
| EXP-035 | collapse = low entropy **with** high modal fraction |
| EXP-037 | both rose **together** across the back-loaded arms |
| **EXP-038 pilot** | entropy rises to its ceiling while modal **plateaus** |

**Read both, always. No single one of these is the general rule.**

## 4. Three launcher failure modes I re-introduced by accident

The first pilot launcher used `$ErrorActionPreference = 'Stop'` with `2>&1` **and**
`Tee-Object`, both of which `launch037.ps1` warns against in its header. I had not read it
first. The `reinforce.py:188` UserWarning that triggers the first failure is visible in every
log this session, so it would very likely have died mid-run and orphaned ten workers that go on
looking healthy.

Caught before it cost anything, killed, and relaunched against the known-good pattern.
`launch038.ps1` now restates all three constraints in its header **with a note that they were
re-introduced once**, because "documented in another file" was not enough to stop it.

**If you write a launcher: copy `launch038.ps1`, do not write one from memory.**

## 5. What to do when it lands

1. `aggregate.py`, then write `experiments/038_depth6_collapse/RESULTS.md` marking each claim
   confirmed or refuted (todo `^e172`). Include provenance: seeds, date, machine, wall clock,
   regeneration command.
2. **Read a real trace, not only the verdicts.** A cube frame labelled `solved: yes` on a
   scrambled cube once passed every unit test in the suite.
3. Fetch the records before doing anything else - `outputs/*.json` is gitignored and the laptop
   is the only copy. The `*_head.pt` checkpoints ARE tracked; **commit them from the VPS and
   the attic mechanism handles the collision next dispatch.**

**Then the decision the handoff before this one framed and this session did not close:** the
remaining candidates are **the encoder (vault Stage 2)** - the biggest build, most
thesis-relevant, and where both EXP-032 and ADR 0001 independently point - and **the EXP-030
memory re-ask** (`^7741`), still cheap because 96 heads are committed and EXP-038 adds 48 more.
If EXP-038 refutes, Claim 5 says the encoder is next.

## 6. Open debt

Tracked (`secretary todo list --project neuromorphic`):

1. `^e172` **NEW** - read EXP-038 verdicts and write its `RESULTS.md`.
2. `^bbd0` **NEW** - audit other experiments for the wrong-budget 0.354 modal anchor.
3. `^f772` cubeNet.ts B/D face orientation - pinned with `it.fails`, needs a browser. **Remove
   the `.fails` when fixed or the test starts erroring.** B is the only face row-major gets
   *right*; U, R, F, D, L are wrong.
4. `^0576` Nothing in the cube dashboard has been seen rendering in a browser.
5. `^7741` The EXP-030 memory re-ask.
6. `^0817` Phase 0 / Phase 1 checkpoints unticked in the vault `progress-tracker`. **Needs
   Michael.**
7. `^c16a` `requirements.txt` omits the `[server]` extra.
8. `^3ef2` Playbook: exit-255 guidance should mention the tailscale peer last-seen.

Not tracked:

9. `verify_instrument_neutrality.py` reports a `config` mismatch as one opaque key.
10. EXP-025 still has no committed `RESULTS.md`; cite ADR 0001 Amendment 2.
11. Live trace streaming during training is a spec, not built.
12. The Aug 8 calendar milestone duplicates the recurring coding block in the same slot.

## 7. Pointers

- Pre-registration: `docs/superpowers/specs/2026-08-07-exp038-depth6-collapse-design.md`
  (**section 5a** is the pilot corrections)
- Analysis: `experiments/038_depth6_collapse/aggregate.py`, tested by
  `tests/experiments/test_exp038_aggregate.py` (8 tests; removing the Claim 2 guard or defeating
  the saturation threshold each fails 3 of them)
- Prior results: `experiments/03{2,6,7}_*/RESULTS.md`
- Remote runs: `docs/playbooks/remote-experiment-runs.md` (**section 1 rewritten**)
- Sync: `scripts/laptop/sync_repo.ps1`
- **Strategy: vault `Neuromorphic Development/road-to-a-solved-cube.md`**
- Previous handoff: `SESSION-HANDOFF-2026-08-07.md`
