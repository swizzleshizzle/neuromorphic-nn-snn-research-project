# Session Handoff - 2026-08-13 (Thu) - Week 19 session 3

> [!important] **SUPERSEDED. Start from `docs/handoffs/SESSION-HANDOFF-2026-08-14.md`.**
> EXP-043 finished and is written up. Nothing is in flight; the job is now the render.

> **EXP-043 is IN FLIGHT on the laptop.** Dispatched 2026-08-13 13:13:46 at commit `cbbe9d4`.
> 24 runs, ~2.15M env steps, **ETA about 02:15 Friday**. The repo is clean and pushed.
>
> Read `CLAUDE.md` first, then this file. Strategy is **not** in this repo: it is the vault at
> `300 Efforts/Active/Coding/Neuromorphic Development/road-to-a-solved-cube.md`.

## 0. State check

```bash
git log --oneline -1              # expect cbbe9d4 or later
git status --short                # expect clean
scp scripts/laptop/probe_run.ps1 laptop:C:/Users/mlgbr/probe_run.ps1
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\probe_run.ps1 -OutDir "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\experiments\043_cap_at_depth_5_6\outputs"'
```

`VERDICT=HEALTHY | ORPHANED | STALLED | NO_RUN`, non-zero exit on the bad ones. Expect ~6.5-7.5
effective cores at 10 workers.

**Things that look like problems and are not:** a client-side `exit 124`/`255` (the dispatching
ssh has its own timeout; the run outlives it - probe before reacting); a record count frozen for
hours (runs of one depth finish as a group); `log_growth_bytes=0` (progress lines are ~30 min
apart). **The laptop sleeping does not kill a run** - Windows suspends processes. Verified
2026-08-12 when the power button was pressed by accident and EXP-042 rode straight through it.

## 1. Where the project now stands

**The headline: depth 4 scores 0.5351.** That is higher than depth 3's best-ever 0.500, which
needed three times the episode budget.

| depth | EXP-036 (frozen encoder) | EXP-040 (pretrained) | EXP-042 (pretrained + cap) |
|---|---|---|---|
| 3 | 0.3972 | - | - |
| 4 | 0.1591 | 0.3471 | **0.5351** |
| 5 | 0.0396 (broken) | 0.2304 | **running now** |
| 6 | 0.0000 | 0.1037 | **running now** |

Two levers found, in order:

1. **Train the encoder** (EXP-039/040). Self-supervised: predict the move between two cube
   states. Depth 4 went 0.1591 -> 0.3471, and the linear probe beat the raw-observation ceiling.
2. **Reprice depth 1** (EXP-041/042). See below - this is the one that is easy to under-rate.

> [!important] Refuted and CLOSED. Do not revisit without a new reason.
> width (EXP-033) - volume alone (EXP-034) - curriculum stage weighting (EXP-037) - starvation at
> depth 6 (EXP-037) - trainer stabilizers (EXP-038) - **deleting the depth-1 stage (EXP-042:
> reprice it, do not remove it)**

## 2. The depth-1 trap, because everything recent depends on it

`max_steps_for(d) = 2d+3` gave depth 1 a budget of **5** where optimal is **1**. A cube face has
**order 4**, so from a one-move scramble any repeated move either inverts it (1 step) or cycles
back to solved (3 steps). Both fit in 5.

So a **constant-action policy** scored **0.3333** at depth 1 - against a random policy's
**0.2208**. Enumerated, not sampled.

**Curriculum stage 1 paid more for the worst possible policy than for exploring.** Two of twelve
seeds learned that, collapsed to one action, and never recovered: entropy 0.005 leaving stage 1,
0.000 success at every later depth.

**Depth 1 is the only stage where this happens** - 0.037 vs 0.051 at depth 2, exactly 0.000 at
depth 3+.

**The fix** (`max_steps_by_depth=((1,2),)`) caps the depth-1 **training** budget at 2. That
admits the inverse and not the cycle: 0.1667, below random. Evaluation is untouched, so no arm is
scored differently.

**What it bought at depth 4:** 0.3471 -> 0.5351, seeds at zero 2/12 -> **0/12**, sd 0.2242 ->
**0.1012**, worst seed 0.000 -> 0.241.

> [!important] The part that is easy to miss
> It gained **+0.1195 on the ten seeds that never failed**. The trap was degrading **every** run,
> not only the two that visibly collapsed - which also means EXP-040's "powerful but unreliable"
> caveat was **largely the trap, not the encoder**.

## 3. EXP-043 - what is running and how to read it

Same capped arm, at **depths 5 and 6**, 12 seeds each, on EXP-040's pretrained encoders.
**EXP-040 is the paired baseline and is not re-run** - same encoders, seeds, machine, budget and
curriculum, differing by the cap and nothing else.

```bash
.venv/bin/python experiments/043_cap_at_depth_5_6/aggregate.py     # NOT WRITTEN YET
```

| claim | test |
|---|---|
| **1 PRIMARY** | success paired vs EXP-040, per depth. **>= +0.05 at p <= 0.05** |
| **2** | does depth 6 become "working"? **>= 0.10 AND >= 1.0 SE AND >= 8/12 seeds above 0.10** |
| **3** | failure count, **descriptive, no p-value** |
| **4** | does the variance collapse repeat? (sd 0.2242 -> 0.1012 at depth 4) |
| **5** | the null is pre-committed as a **scoping** result |

> [!warning] `aggregate.py` is NOT written. Write it BEFORE reading the records.
> Rules on disk before numbers - the standing habit. Model it on
> `experiments/042_depth1_trap/aggregate.py`.
>
> **The baseline has no `stage_trace`** (EXP-040 predates that telemetry), so the primary claim
> is on **success**, not entropy. Do not try to pair the mechanism.

**Claim 2's extra conditions exist for a reason.** EXP-040's depth 6 hit 0.1037 and cleared
EXP-036's bare `>= 0.10` rule by **0.11 standard errors** with 5/12 seeds above it. That was
noise dressed as a verdict, and it was reported as *at* the bar rather than *working*.

## 4. Statistics: what n=12 cannot do here

> [!danger] Read this before interpreting any failure count
> The failure effect is **2 of 12 seeds**. In a paired permutation test where two seeds carry the
> difference and ten are ~0, only 2 of the 4 sign assignments on those two exceed the observed
> sum, so **p is about 0.5 BY CONSTRUCTION**. Fisher's exact on 2/12 against 0/12 gives ~0.48.
>
> **No arrangement of 12 seeds can prove a failure rate went to zero.** Claim the mechanism, or
> a quantity that moves on every seed. Report counts descriptively, with no p-value.
>
> EXP-042's aggregator has a worked example: it reports the effect **excluding** the two
> previously-failing seeds, which is what showed capping helps broadly (+0.0754) while skipping
> **hurts** the other ten (-0.0496).

## 5. Open, in rough priority

1. **Write `experiments/043_cap_at_depth_5_6/aggregate.py`, then read the records.** `^07e2`'s
   sibling; the EXP-042 one is the template.
2. **`^5bdb` render the manim scenes - Content Day is SAT AUG 16.** They have never been
   rendered, so layout is unverified. `manimpango` will **not** build on the VPS; three attempts
   are recorded in `viz/manim/README.md`. **Render on the laptop** once EXP-043 clears. Story arc
   and the data layer are done (`viz/manim/story.md`, `data.py`).
3. **If EXP-043 confirms, find the new break point.** Depth 7 needs
   `ExactBFSDistance(max_depth=8)`.
4. `^bbd0` audit other experiments for the wrong-budget 0.354 modal anchor (depth 6 is 0.309,
   depth 5 is 0.321).
5. `^7741` the EXP-030 memory re-ask - 228 serialised heads now, still cheap.
6. `^0817` **needs Michael**: Phase 0 / Phase 1 checkpoints in the vault `progress-tracker`.

## 6. Operational notes worth not rediscovering

- **`scp` to the laptop can time out on the banner exchange.** It did on the EXP-043 dispatch,
  and the launcher silently was not there. **Verify with `Test-Path` before dispatching**, and
  retry - it succeeded first try on the retry.
- **Do not background a command that itself ends in `&`.** The wrapper exits immediately and
  takes the child with it. The EXP-043 dispatch failed this way once.
- **The VPS is 2 cores / 4 GB: one Python worker, `nice -n 10`.** Two workers once drove it into
  swap thrash, which makes the box unresponsive **without** tripping the OOM killer - so
  `journalctl` shows nothing afterwards. Absence of OOM records is not absence of trouble.
- **Do not extrapolate throughput across concurrency levels.** Measured: the same work ran
  **1.8x faster** with 2 workers than with 10, because the laptop is memory-bound.
- **An agent has no reliable clock between tool calls.** Sample inside one command.
- **`LastWriteTime` is unusable on an open log** on Windows; measure byte growth.
- **A waiter condition must require well-formed output.** `grep -qv "^0,0$"` once fired on a
  transient ssh hiccup and reported a phase transition that had not happened.
- **`pgrep -f <name>` matches the guard's own command line.** Bit twice; a completion check built
  that way can never fire.

## 7. Pointers

- Results: `experiments/04{0,1,2}_*/RESULTS.md` - EXP-041 is the **diagnosis** that explains
  everything since
- Pre-registrations: `docs/superpowers/specs/2026-08-{07,08,09,12,13}-*.md`
- Trainer seams: `CubeConfig.encoder_state_path` (EXP-040), `max_steps_by_depth` (EXP-042),
  `stage_trace` telemetry. **All three are measured-neutral** -
  `tests/training/test_encoder_seam.py`, slow marker, reproduces a pre-change baseline to 1e-6
- Remote runs: `docs/playbooks/remote-experiment-runs.md`
- Laptop scripts: `scripts/laptop/sync_repo.ps1`, `scripts/laptop/probe_run.ps1`
- Visual story: `viz/manim/story.md`
- **Strategy: vault `Neuromorphic Development/road-to-a-solved-cube.md`**
- Previous handoff: `SESSION-HANDOFF-2026-08-10.md`
