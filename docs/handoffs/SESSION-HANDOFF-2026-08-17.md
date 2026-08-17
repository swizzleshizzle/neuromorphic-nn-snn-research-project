# Session Handoff - 2026-08-17 (Mon) - EXP-044 and EXP-045: it was TOTAL BUDGET

> **Nothing is in flight. Both machines are idle. The repo is clean and pushed.**
>
> **EXP-044 (both arms) and EXP-045 are finished and written up.** Read
> `experiments/045_budget_vs_coverage/RESULTS.md` first - it settles what EXP-044 could not - then
> `experiments/044_depth7_frontier/RESULTS.md`. This file is the short version.

## 1. What EXP-044 found

| arm | episodes | coverage | mean | margin | seeds above bar | verdict |
|---|---|---|---|---|---|---|
| A | 10,000 | 0.044 | 0.0621 | -2.38 SE | 4 / 12 | REFUTED |
| **B** | **44,000** | **0.191** | **0.1971** | **+7.86 SE** | **12 / 12** | **CONFIRMED** |

**Depth 7 works once it is fed like depth 6 was.** By the reading fixed in the spec before either
number existed, arm A's failure was **starvation, not difficulty**.

The comparison that carries it: depth 6 at 10,000 episodes and depth 7 at 44,000 sit at the same
coverage - **0.190 against 0.191** episodes per training state - and score **0.1800 against
0.1971**, a gap well inside a combined SE of 0.031. **At matched coverage the two depths are
indistinguishable.** Feeding depth 7 properly also cut its coefficient of variation from **0.89 to
0.22**, the tightest in the series: the seed sensitivity that looked like a property of depth was
substantially a property of a starved budget.

**The break point is not found, for the second time.** What is found is that depth 7 was
under-trained.

## 1b. EXP-045 settled WHICH quantity, and it is not coverage

EXP-044 arm B raised the total budget 4.4x, which raised episodes at *every* stage, so it could not
say whether depth 7 was rescued by total training or by exposure at the deepest shell. EXP-045 gave
stage 7 arm B's episode count **inside arm A's 10,000-episode budget**, by weighting the curriculum
`(1,1,1,1,1,1,10)`. One variable against arm A.

| | mean | zeros | modal action frac |
|---|---|---|---|
| arm A (uniform, 10k) | 0.0621 | 1 / 12 | 0.561 |
| **EXP-045 (back-loaded, 10k)** | **0.0142** | **9 / 12** | **0.847** |
| arm B (uniform, 44k) | 0.1971 | 0 / 12 | - |

**Paired delta -0.0479, W-L-T 0-11-1, exact p 0.0010.** Not one seed improved. **The operative
variable is TOTAL BUDGET**, and raising deep-end exposure at a fixed budget is actively harmful.

**The mechanism is the finding.** Inside the long stage-7 block, entropy fell 0.591 -> 0.098
(minimum 2.7e-06) while only **2.2%** of training episodes solved. 6,250 consecutive episodes at a
depth the policy almost never solves is 6,250 episodes of no reward signal, and it collapses.
EXP-037's depth-4 decline therefore **generalises** to depth 7, a pretrained encoder and the cap.

> [!important] The shallow curriculum stages are not warm-up. They are what stops the collapse.
> Halving them while quadrupling the deep stage produced the most collapsed policy in the series.
> The curriculum's job is to keep the reward signal dense enough that entropy survives - which also
> explains why EXP-042's depth-1 **cap** worked where **deleting** the stage did not. Shallow stages
> must be present and cheap, not absent and not dominant.

**Consequence for everything already published here:** every "depth N stopped working" result was
taken at a fixed 10,000 episodes, so each measures **depth at 10,000 episodes**. Depth 7 shows a
depth can look broken purely for want of budget.

> [!warning] The overreach to refuse
> *"Every depth just needs more episodes"* is still NOT established. Only **depth 7** has been run
> at more than 10,000, and depth 3 scores **lower** than depth 4 (0.3972 against 0.5351) despite a
> far smaller shell, so budget cannot be the whole story either. One depth is one depth.

## 2. The next experiment, and why it is not depth 8

**Re-run depth 6 at raised TOTAL budget** - about 51,000 episodes, ~25 h. If it moves the way depth
7 did, the whole depth series is a budget series and every number here needs restating as "at
10,000 episodes". If it does not move, depth 7 was the special case. That is the open question.

**Depth 8 at matched coverage needs ~174,000 episodes** (shell 114,149), roughly **100 h**. The
matched budget grows about **4x per depth**, so brute-forcing the frontier gets expensive faster
than the frontier moves. Do it after the scaling law is tested, not before.

Also open: fine-tuning the encoder during RL (untested), and re-running depth 5 with 24+ seeds to
settle EXP-043's Claim 1 (+0.1108 at p 0.0815).

## 3. Standing state

- **The visual story is complete.** All nine scenes render at 1080p and `FullStory` assembles them
  into one ~2:31 cut. Nothing is waiting on it - **there is no publishing deadline**, see
  `CLAUDE.md`. The scenes do NOT yet include depth 7; adding it is a ~30 minute job if wanted.
- **The laptop needs no setup.** manim 0.21 in `C:\Users\mlgbr\manim-venv`, ffmpeg 9.0, and
  `scripts/laptop/render_story.ps1` drives renders and pulls stills.
- Records for EXP-044 (36) and all 24 head checkpoints are on the VPS and committed.

## 4. Operational notes earned this run

- **`| Tee-Object` captures stdout only.** Arm A's traceback went to the dispatching ssh's stderr
  and was lost when that session was killed. Use `*>&1 | Tee-Object`, as arm B did.
- **`probe_run.ps1` takes `-LogName` now.** It hardcoded `run.log`, so probing arm B reported arm
  A's finished log and `log_growth_bytes=0` looked like a stall on a healthy run.
- **A killed dispatching ssh does not kill the run** - confirmed twice more, by the launcher
  powershell keeping the same pid throughout. Probe, do not re-dispatch.
- **`run.py --dry-run` exists** because inspecting the pre-flight banner by running the driver and
  killing the pipe nearly started a 44,000-episode run on the 2-core VPS against a seed the laptop
  was already computing.
- Arm B took **25.4 h** for 12 seeds at 12 workers, against a 32 h estimate. Roughly 6.4 effective
  cores sustained. CPU-hours accumulated and wall-clock fraction agreed to within a percent
  mid-run, which is the only progress signal available - nothing writes until a seed finishes.

## 5. Pointers

- `experiments/044_depth7_frontier/RESULTS.md` - the full write-up
- `docs/superpowers/specs/2026-08-14-exp044-depth7-frontier-design.md` - the pre-registration
- Previous handoff: `SESSION-HANDOFF-2026-08-14b.md` (the render session)
- Remote runs: `docs/playbooks/remote-experiment-runs.md`
