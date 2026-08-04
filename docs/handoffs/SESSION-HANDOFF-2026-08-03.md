# Session Handoff - 2026-08-03 (Mon) - Week 18 session 1

> [!note] **RESOLVED 09:35 ET. The run never died; the network flapped.**
> Boot time is `07/27/2026`, a week before launch, so there was no reboot. 18 processes alive,
> records went 14 -> 32 straight through the outage, zero tracebacks. Tailscale is connected
> over the **DERP relay "iad" rather than a direct link**, which flaps; that produced both the
> "offline, last seen 5m ago" reading and later ssh timeouts while the host was fine.
> **Retry a failed probe two or three times before concluding anything.**
>
> **The schedule is off by 1.8x though. Revised finish is about 17:00-17:30 ET, not 08:00.**
> Diagnosis in section 1.6.

> **EXP-036 was dispatched** 2026-08-03 at 20:09 ET, expected to take **11.8 hours**, so it was
> due to finish around **08:00 ET on 2026-08-04**. Repo is clean at `87a965b` and pushed. The
> laptop is on the same commit.
>
> Read `CLAUDE.md`, then this file, then
> `docs/superpowers/specs/2026-08-03-exp036-generalisation-gap-design.md` for the
> pre-registered contract this run is answering.

## 0. State check

```bash
git log --oneline -1                                    # expect 87a965b
git status --short                                      # expect clean
ssh -n laptop 'powershell -NoProfile -Command "(Get-Process | Where-Object { $_.Name -match \"^python\" }).Count"'
```

## 0.5 The analysis path is already built. Do not write it in the morning.

```bash
# 1. fetch (records AND the head checkpoints, which are tracked)
scp "laptop:C:/Users/mlgbr/Desktop/Projects/neuromorphic-nn-snn-research-project/experiments/036_generalisation_gap/outputs/*" experiments/036_generalisation_gap/outputs/

# 2. apply the pre-registered rules. Re-runnable any number of times.
.venv/bin/python experiments/036_generalisation_gap/aggregate.py

# 3. confirm the code changes were neutral against EXP-035's depth-3 cell
.venv/bin/python scripts/verify_instrument_neutrality.py \
    experiments/035_budget_scaling/outputs experiments/036_generalisation_gap/outputs \
    --key-by cell --new-fields train_success_rate,n_train_eval,generalisation_gap --exempt tag
```

`experiments/036_generalisation_gap/RESULTS.md` is committed as a skeleton with every claim
already written in as an unfilled row. **Fill it in, then mark each claim CONFIRMED or REFUTED.
Do not edit a threshold while filling it in.** If a result lands awkwardly against a bar, that
is the finding.

## 1. First thing: check on EXP-036

```bash
ssh -n laptop 'powershell -NoProfile -Command "$d=\"C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\036_generalisation_gap\outputs\"; \"pyprocs=\" + (Get-Process | Where-Object { $_.Name -match \"^python\" }).Count; \"records=\" + (Get-ChildItem $d -Filter *.json).Count; Get-Content (Join-Path $d run.log) -Tail 40"'
```

**96 records expected** (48 trained + 48 floors), plus 48 `*_head.pt` checkpoints.

> [!warning] **Zero records for the first ~3 HOURS is HEALTHY, not dead.**
> `sweep_configs` submits all 48 trained runs before the 48 floors, so the first wave of 16
> workers is entirely depth-3 training and nothing completes until a whole run finishes. A
> depth-3 run at 10,000 episodes is 70,000 steps, which at the measured 153 ms/step is
> **about 3.0 hours**. The cheap floors are queued LAST and produce nothing early.
>
> Expected first-record times, per wave of 16 on 16 workers:
>
> | wave | depth | steps/run | first records at |
> |---|---|---|---|
> | 1 | 3 | 70,000 | ~3.0 h |
> | 2 | 4 | 80,000 | ~6.4 h |
> | 3 | 5, 6 | 90,000 / 100,000 | ~10 h, then the floors |
>
> A record-count check with a short grace period would call this run dead while it is working
> perfectly. Same shape as the five monitoring bugs logged on 2026-08-02: **before trusting a
> health check, ask what it would report about a run that is perfectly healthy.**
>
> The real early health signal is worker count (expect 16 plus the parent) and memory.
>
> **Report PRIVATE bytes, not working set.** Measured 1.9 h in: 16 workers at 80 to 103 MB
> working set but **920 MB private commit each**, with system commit at **36 GB of a 43.4 GB
> limit (83%)**. Quoting working set makes a run look 10x lighter than it is, and "10.2 GB
> free physical" is not the constraint. The playbook's 195 MB per worker is a working-set
> figure and carries the same blind spot.
>
> **Worker CPU utilisation is about 44%, and that is NORMAL here, not a fault.** 16 workers on
> 22 logical cores each averaged 3,041 CPU-s over 6,881 s elapsed. It looks like the run is
> starved and should take 2.3x longer than estimated. It is not, because the 153 ms/step
> calibration was derived from EXP-035's WALL CLOCK times workers, not from CPU time, so the
> same 44% is already inside it. Cross-check end to end instead: EXP-035 sustained 376,672
> steps/hour aggregate, and 4,462,068 / 376,672 = 11.8 h.
>
> **Do not "fix" the utilisation by adding workers without re-measuring.** At 83% commit there
> is not room, and the throughput figure that all estimates rest on was measured at 16.

Fetch the results when it is done:

```bash
scp "laptop:C:/Users/mlgbr/Desktop/Projects/neuromorphic-nn-snn-research-project/experiments/036_generalisation_gap/outputs/*" experiments/036_generalisation_gap/outputs/
```

The `*_head.pt` checkpoints **are tracked in git**, unlike the `.json` records. Commit them.

## 2. What EXP-036 answers, and the thresholds it must be held to

Full contract in the spec and in `experiments/036_generalisation_gap/run.py`'s docstring. The
driver prints its own verdicts. Two claims:

1. **The gap.** `train_success - heldout_success` at each depth, on 200-capped samples.
   Mean gap under 0.05 with p > 0.05 **refutes coverage as a lever** and the vault's Stage-1a
   train-fraction sweep is cancelled, not run. At or above 0.15 with p <= 0.05 justifies it.
2. **The break point.** Held-out success below twice the measured random floor.
   **Pre-registered prediction: it breaks at depth 5.** If depth 6 still works, that refutes
   the prediction and means Wall 1 is further out than EXP-033 implied. Log it as a refutation.

**Replication gate before trusting anything:** the depth-3 cell is exactly EXP-035's 10k cell
and must reproduce **0.397**. The driver checks this and prints MISMATCH if it is off by more
than 0.02. If it mismatches, resolve that before reading any other row.

## 1.5 The laptop went offline mid-run. Triage this first.

**Timeline, all 2026-08-04 UTC:**

| time | evidence |
|---|---|
| 00:09 ET | run launched, log header correct |
| ~00:53 ET | last healthy probe: **18 processes, 14/96 records, 14 checkpoints, 0 tracebacks** |
| ~01:05 ET | `ssh` returns **Connection timed out**; `tailscale status` shows `swizzlesduo ... offline, last seen 5m ago` |

**This is NOT the ssh-drop case the playbook covers.** That case is "the connection died, the
machine kept working, exit 255 means nothing". Here the **machine left the tailnet**, which the
Tailscale peer status distinguishes and a bare ssh failure does not. The dispatch task did report
exit 255, and reacting to that alone would have been reading the wrong signal.

**Most likely cause: the laptop slept.** It went offline right around the time work stopped for
the night, and `CLAUDE.md` already records that **Tailscale cannot wake a sleeping machine.**

**What to check when it is back:**

```bash
ssh -n laptop 'powershell -NoProfile -Command "(Get-CimInstance Win32_OperatingSystem).LastBootUpTime; (Get-Process | Where-Object { $_.Name -match \"^python\" }).Count"'
```

- **Boot time BEFORE 20:09 ET Aug 3** plus workers alive -> it only slept; the run resumed and
  should still complete, just later than 08:00.
- **Boot time AFTER 20:09 ET** -> it rebooted and **the run is dead**. Records already written
  survive on disk.

**Either way nothing is lost beyond time.** Records are one file per run, written on completion
and never appended to a shared file, and each trained run's head checkpoint is written beside it.
`aggregate.py` handles a partial set and prints `INCOMPLETE`, flagging every verdict as
provisional. To resume, re-run the driver: completed cells rewrite identically on the same
machine, so there is no partial-state problem.

**A watcher is armed** on the VPS polling every 4 minutes for up to 7 hours; it reports process
count, record count and boot time the moment the laptop returns.

> [!note] Follow-on for the playbook, not done
> `docs/playbooks/remote-experiment-runs.md` says exit 255 means the connection dropped and the
> job is fine. That is true but **incomplete**: it does not distinguish a dropped connection from
> a vanished host. Add "check `tailscale status` for the peer's last-seen before concluding
> anything" to the monitoring section.

## 1.6 Why it is 1.8x slow: memory, not CPU. Do NOT restart it.

Measured 2026-08-04 at 09:35 ET, 13.4 h into the run:

| | |
|---|---|
| completed | 32/96 (depth 3 and 4 trained done, depth 5 at 8/12, depth 6 and the 48 floors pending) |
| worker CPU utilisation | **43.1%** |
| CPU cost per step | 132 ms (fine) |
| **wall cost per step** | **306 ms** (132 / 0.431) |
| private per worker | **920 MB** |
| working set per worker | **80 MB** |
| system commit | **48.6 GB of a 50.4 GB limit (96%)** |
| free physical | 5.8 GB of 31.4 GB |

**Workers hold 920 MB private but only 80 MB resident.** Roughly 840 MB per worker is paged
out, so they spend 57% of their time waiting on the pagefile rather than computing. Windows has
already grown the commit limit from 43.4 GB to 50.4 GB, which is the tell. **The constraint is
memory, not CPU.**

**Why the estimate was wrong, again.** 153 ms/step came from EXP-035, but EXP-035 ran **24 tasks
on 16 workers**, so the pool was not saturated end to end and commit stayed manageable; back out
its numbers and it achieved about **86% utilisation**. EXP-036 runs **96 tasks on 16 workers**,
saturating the pool for the whole run and pushing commit to 96%. Same machine, same code, half
the throughput.

> [!important] Calibrate throughput at the concurrency AND memory footprint you will actually run
> at. A previous experiment on the same machine is not a valid reference unless its pool was
> saturated the same way. This is the second time in one session that a number inherited from
> elsewhere was wrong by a large factor.

**Do not restart with fewer workers.** It is a wash and costs the in-flight work:

```
16 workers x 43.1% utilisation = 6.9 effective
 8 workers x ~85% (optimistic)  = 6.8 effective
```

Halving the workers roughly halves our 14.7 GB footprint, but our python is already the largest
single consumer at **15.2 GB of the 48.6 GB committed** and the rest is spread thin (WSL 1.7,
mcp-server 1.0, Podman 0.9, svchost 0.9, claude 0.8, NordVPN 0.8). There is no one process to
close for a big win, and a restart would discard 16 runs that are each an hour or more in.

**Let it finish.** The 32 completed records and their checkpoints are already safe on disk.

**For the NEXT sweep**, budget from private bytes: `16 x 920 MB = 14.7 GB`, not from the
playbook's 195 MB working-set figure. That figure understates the real footprint by about 4.7x
and is what made 16 workers look comfortable here.

## 2.5 The one habit that earned its keep tonight

**Measure the instrument on a case where the answer is already known.** Four defects, all the
same shape - a check that cannot distinguish the states it exists to separate - and the three
that were caught before costing anything were caught this way:

| what was measured | known answer | what it exposed |
|---|---|---|
| the `random` arm's gap | zero, it cannot overfit | the 0.05 threshold sat below the noise floor |
| synthetic records with a failing depth 6 | BROKEN | "2x the floor" reported it *working* |
| derived cube corners vs the net | must share a corner | row-major is wrong on 5 of 6 faces |

The fourth, the 90 ms cost figure, was caught only by cross-checking against a real run's wall
clock. **Estimate from measured throughput, never from a latency figure.**

## 3. Three errors caught during design, each worth a run

Recorded because the pattern matters more than the instances.

1. **The vault's Stage 1 premise was already false.** `road-to-a-solved-cube.md` asks whether
   the depth-3 policy memorises its 90 training states. It does not: `split_shell` partitions
   depth 3 into 90 train / 30 eval, and **EXP-035's 50.0% headline was always held-out
   success**. The vault note has not been corrected yet; see section 6.

2. **The first gap threshold sat below its own noise floor.** The draft contract tested a
   single-seed gap against 0.05. The `random` arm cannot overfit, so its gap is zero by
   construction, yet over seeds 0-5 it produced single-seed gaps from **-0.100 to +0.011** -
   the depth-3 held-out side is 30 states, so its resolution is 1/30 and three lucky solves
   read as a tenth of a gap. Now stated on the twelve-seed mean with an exact paired
   permutation test over 4096 sign flips, against the random arm as a **measured** null.

3. **The cost estimate was 43% low.** It used CLAUDE.md's "`brain.step` costs about 90 ms",
   which is single-step latency, not what a 16-worker run achieves. See section 4.

## 4. `brain.step` is 90 ms; a parallel run achieves 153 ms/step

Calibrated against EXP-035's own wall clock on the same machine: 3,359,916 steps in 8h55m
across 16 workers is 143 core-hours, so **153 ms/step**. The rule of thumb understates a
parallel run by about 70%. EXP-036 is 11.8 h, not the 6.7 h first written into the spec.

`docs/playbooks/remote-experiment-runs.md` now carries the calibration and this step-counting
formula. Use it for every future estimate:

```python
def steps(depth, episodes):
    stages = list(range(1, depth + 1))
    per = episodes // len(stages)
    return sum(per * (2 * d + 3) for d in stages)
# wall_hours = total_steps * 0.153 / 3600 / workers, plus n_states * (2d+3) per evaluation
```

## 5. Code changes that outlive this experiment

- **Head checkpointing.** No trained weights were saved anywhere before today, so every
  "evaluate that policy differently" question cost a full retrain. `head_filename(cfg)` writes
  `<record stem>_head.pt` beside the record. **This makes the EXP-030 memory re-ask cheap**,
  which is the single biggest consequence of this session for what comes next.
- **Checkpoints are tracked in git**, records are not. Two `.gitignore` fixes were needed and
  **one was already silently broken**: `experiments/*/outputs/` excluded the *directory*, and
  git does not descend into an excluded directory, so the `!.gitkeep` negation beneath it had
  never worked. Now `experiments/*/outputs/*`, with `!experiments/*/outputs/*_head.pt` placed
  after the global `*.pt` because last match wins.
- **`sample_train_eval()`**, capped at `heldout_cap`. **The cap is load-bearing.** Uncapped at
  depth 6 the train-side evaluation is 8,769 states x 15 steps, about 3.3 h per seed and 40
  core-hours over twelve seeds, against roughly 2.6 h of actual training per run. It also
  matches the sampling noise on the two sides, which a gap measurement needs.
- **New record fields:** `train_success_rate`, `n_train_eval`, `generalisation_gap`.
- **Ordering is load-bearing.** Both evaluations run after training, and the train side runs
  strictly after the held-out side, because `greedy_action` draws on the shared torch
  generator. The five EXP-030 reference values are what catch a swap: all reproduce exactly.

## 6. Open, and carried forward

**New, from this session:**

1. **The vault note `road-to-a-solved-cube.md` still states the false Stage-1 premise.** It
   should be corrected to say the depth-3 policy already generalises and that the live question
   is the gap. Not done tonight.
2. ~~`verify_instrument_neutrality.py` cannot compare across tags.~~ **DONE.** It gained
   `--key-by cell`, matching on (arm, depth, seed, sigma), plus `--new-fields` and `--exempt`,
   with defaults preserving the EXP-030 behaviour. Eight tests, each checking it reports a
   *different* verdict for a different input. The exact invocation for the EXP-035 replication
   check is at the bottom of `experiments/036_generalisation_gap/RESULTS.md`.
3. ~~Vault notes stale.~~ **DONE.** `experiment-log.md` Phase 3 section replaced (it held eight
   March placeholder rows whose numbering never survived contact with the work - EXP-030 was
   planned as "1-move scramble" and turned out to be the memory experiment).
   `weekly-notes-index.md` had **dead wikilinks for weeks 13 to 25**, pointing at titles no note
   was ever written under; relinked against the files that exist.
   `week-18-generalisation-and-depth.md` created. `road-to-a-solved-cube.md` carries a dated
   correction block on the Stage-1 premise, with the plan itself left intact.
4. Phase 0 and Phase 1 checkpoints in the vault `progress-tracker` are still unticked although
   every week beneath them is done. Unverifiable from here, needs Michael.
5. The Google Calendar milestone on 2026-08-08, "MILESTONE: First 1-Move Cube Solve Attempt",
   is stale by months and says "do not rush past 1-move scrambles until solve rate > 80%".

**Closed this session, from the 2026-08-02 list:**

- **Item 7 (unfalsifiable assertion) is DONE**, along with two others found by sweeping for the
  same shape. `0.0 <= success_rate <= 1.0`, `0.0 <= rate <= 1.0` and `0 <= dist.sample() < 6`
  were all tautologies. Replaced with measured thresholds; see commit `ad19c41` and the one
  after it.
- **Items 2 and 4 (cube net orientation, DLB highlight) are PINNED, not fixed.** No rendering
  changed, because this box has no browser and nothing visual could be looked at.

> [!warning] **Item 2's description was wrong and is corrected.**
> It read "cubeNet.ts uses row-major on all six faces; B and D are probably mis-oriented".
> **That is close to backwards.** The corner structure was derived from the pre-verified move
> permutations (orbit of one corner under U/R/F, plus the fixed DLB corner the orbit cannot
> reach) and validates exactly against cube geometry. Solving the five net-border constraints
> gives 32 consistent orientation assignments, and **B is the ONLY face for which row-major
> appears in any of them** - because this net constrains B through R alone. **U, R, F, D and L
> are all inconsistent.**
>
> Worked example, the U/F border: U's F-side stickers are facelets 0 and 1 (corners UFL, UFR),
> so they belong on U's *bottom* row; row-major puts them on the top. F's U-side stickers are
> 10 and 11, which row-major puts on the bottom.
>
> `dashboard/src/panels/cubeNet.test.ts` now carries the derived corner table and a border
> test marked `it.fails`, so the suite stays green while the defect stays pinned. **Remove the
> `.fails` when cubeNet.ts is fixed** - it will start erroring rather than silently passing.

**Carried forward from 2026-08-02, still open:** items 1, 3, 5, 6, 8, 9, 10 of that handoff's
section 6. Item 1 (nothing seen rendering in a browser) is what blocks actually fixing item 2.

## 7. What was updated outside the repo

`progress-tracker.md` in the vault was rewritten for Phase 3: weeks 16 and 17 marked done with
what actually happened, week 18 in progress, weeks 19 to 25 re-mapped onto the Stage 1-5 plan
from `road-to-a-solved-cube`, and the empty metric lines filled in with the depth-3 result, the
390-parameter framing, and EXP-033's measured probe table. The vault is live-synced, so that is
already on every device.

## 8. Pointers

- Pre-registration: `docs/superpowers/specs/2026-08-03-exp036-generalisation-gap-design.md`
- Driver and its contract: `experiments/036_generalisation_gap/run.py`
- Launcher, with the three Windows failure modes documented: `experiments/036_generalisation_gap/launch036.ps1`
- Strategy: vault `Neuromorphic Development/road-to-a-solved-cube.md` (Stage 1 premise needs fixing)
- Remote runs: `docs/playbooks/remote-experiment-runs.md` (now correct on ssh and on throughput)
- Previous handoff: `docs/handoffs/SESSION-HANDOFF-2026-08-02.md`
