# Session Handoff - 2026-08-14 (Fri) - Week 19 session 4

> **Nothing is in flight. Both machines are idle. The repo is clean at `c45f13a` and pushed.**
>
> **The job now is the RENDER, not another experiment.** Content Day is **Sunday 2026-08-16**
> and the visual-story scenes have never been rendered once. Section 3 is the whole task.
>
> Read `CLAUDE.md` first, then this file. Strategy is **not** in this repo: it is the vault at
> `300 Efforts/Active/Coding/Neuromorphic Development/road-to-a-solved-cube.md`.

## 0. State check

```bash
git log --oneline -1     # expect c45f13a
git status --short       # expect clean
ssh -n laptop 'powershell -NoProfile -Command "(Get-Process | Where-Object { $_.Name -match \"^python\" }).Count"'   # expect 0
```

## 1. Where the project stands

**Depths 3 through 6 all work.** The break point was depth 5 from EXP-036 until two days ago.
**It is now past depth 6 and unmeasured** - the first time in the project its location is
genuinely unknown.

| depth | frozen encoder | pretrained | **+ depth-1 cap (current)** |
|---|---|---|---|
| 3 | 0.3972 | - | - |
| 4 | 0.1591 | 0.3471 | **0.5351** |
| 5 | 0.0396 broken | 0.2304 | **0.3412** |
| 6 | 0.0000 broken | 0.1037 at the bar | **0.1800 working** |

Two levers, and they compound. Neither alone reaches depth 6.

1. **Train the encoder** (EXP-039/040) - self-supervised: predict the move between two states.
2. **Reprice depth 1** (EXP-041/042/043) - `max_steps_for(d) = 2d+3` gave depth 1 five steps
   where optimal is one. A cube face has **order 4**, so any repeated move either inverts a
   one-move scramble (1 step) or cycles back to solved (3 steps). A **constant-action policy
   therefore scored 0.3333 at depth 1 against a random policy's 0.2208** - stage 1 paid more for
   the worst possible policy than for exploring. Capping the depth-1 **training** budget at 2
   flips that to 0.1667, below random.

> [!important] Refuted and CLOSED
> width (EXP-033) - volume alone (EXP-034) - curriculum stage weighting (EXP-037) - starvation at
> depth 6 (EXP-037) - trainer stabilizers (EXP-038) - **deleting the depth-1 stage** (EXP-042:
> reprice it, do not remove it)

**One honest caveat carried forward:** EXP-043's depth-5 improvement (+0.1108) is reported
**REFUTED** because p was 0.0815 against a pre-registered 0.05. It is a p-value miss with four
regressing seeds, **not** a demonstrated absence. Do not cite it as "the cap does not help at
depth 5"; 24+ seeds would likely settle it.

## 2. Statistics rule that keeps applying

> [!danger] n=12 cannot show a failure count went to zero
> Fisher's exact on 2/12 against 0/12 gives ~0.48, and a paired permutation test where two seeds
> carry the difference gives p ~ 0.5 **by construction**. Put claims on quantities that move on
> **every** seed; report counts descriptively with **no p-value**.
> `experiments/042_depth1_trap/aggregate.py` is the worked example.

## 3. THE JOB: render the visual story on the laptop

Everything is built except the render itself. **`viz/manim/story.md` is the three-act arc**,
`viz/manim/data.py` is the data layer, `viz/manim/scenes/story_scenes.py` has four scenes.

### 3a. FIRST - `data.py` is STALE and will render last week's story

`depth_curve()` reads **EXP-040** as the "after" arm. Rendered today it shows
**0.3471 / 0.2304 / 0.1037** when the current numbers are **0.5351 / 0.3412 / 0.1800**.

**Update `depth_curve()` to read EXP-042 (depth 4, tag `exp042_capped`) and EXP-043 (depths 5-6,
tag `exp043_capped`) before rendering anything.** `story.md` needs the same pass - its Act 3
table and the `TheBreakPointMoves` caption both predate depth 6 working.

This is the difference between a video that says "we moved the frontier" and one that
understates it by a third.

### 3b. The laptop needs two installs

Probed 2026-08-14:

| | laptop | note |
|---|---|---|
| experiment records | **present** (036: 96, 040: 36, 043: 24) | the scenes read these |
| python | present | |
| **ffmpeg** | **ABSENT** | **manim cannot encode video without it** |
| manim | not installed | `pip install manim` - Windows wheels usually work |

> [!warning] Do NOT try to render on the VPS
> `manimpango` will not build there. Three attempts are recorded in `viz/manim/README.md` -
> plain install, `--no-binary` source build, and cython with `--no-build-isolation` all leave
> only Cython sources and no compiled `.so`. `libcairo2-dev`, `libpango1.0-dev` and
> `pkg-config` **were** installed and did fix `pycairo`, so that part is done.

> [!warning] If installing manim anywhere, keep it OUT of the project venv
> Its dependency tree pulls **scipy**, and this repo deliberately has none - which is **why the
> exact permutation tests exist**. On the VPS it lives at `/root/scratch/manim-venv`. Use a
> separate venv on the laptop too.

### 3c. Rendering

```bash
cd viz/manim
python data.py                                        # sanity: print the real numbers first
manim -ql scenes/story_scenes.py TheBreakPointMoves   # 480p, iterate on layout
manim -qh scenes/story_scenes.py TheBreakPointMoves   # 1080p when it looks right
```

Scenes: `TheBreakPointMoves`, `TheWall`, `ScaleOfTheCube`, `CollapseIsASymptom`.
**None need LaTeX** - they use `Text` (Pango), never `MathTex`.

**They have never been rendered, so layout is unverified.** Expect a fixup pass; positioning is
the part you cannot get right by reading. Render `TheBreakPointMoves` first - it is the payoff
shot and the one most affected by the stale data.

### 3d. Framing notes for the video, all measured

`story.md` has these in full. The ones that keep it credible:

- **Still 0.32% of the cube.** Depths 3-6 of 14. A random scramble sits at **depth 11**, where
  97.5% of the state space lives. Do not imply the cube is solved.
- **The wins came from refutations.** Six things were closed before the two that worked.
- **Both drafted titles for Video 8 are out of date in the project's favour.**

## 4. After the render, if there is time

1. **Find the new break point** - depth 7, probably 8. **Check before dispatching:** the depth-7
   shell is roughly 58,000 states against depth 6's 8,969, so `ExactBFSDistance(max_depth=8)`
   build time and `heldout_cap` both need measuring, not assuming.
2. **Re-run depth 5 with 24+ seeds** to settle EXP-043's Claim 1.
3. **Fine-tune the encoder during RL** - still untested; the frozen version now carries four
   working depths.
4. `^bbd0` audit other experiments for the wrong-budget 0.354 modal anchor (depth 6 is 0.309).
5. `^7741` the EXP-030 memory re-ask - 252 serialised heads now.
6. `^0817` **needs Michael**: Phase 0 / Phase 1 checkpoints in the vault `progress-tracker`.

## 5. Operational notes

- **`Join-Path`, not string concatenation, for Windows paths over ssh.** A trailing backslash in
  a quoted string gets eaten and every `Test-Path` silently returns false. It reported "0
  records on the laptop" today when there were 156.
- **`scp` to the laptop can time out on the banner exchange.** Verify with `Test-Path` before
  dispatching; the EXP-043 launcher silently was not there the first time.
- **Do not background a command that itself ends in `&`** - the wrapper exits and takes the child.
- **Client-side ssh `exit 124`/`255` says nothing about the job.** EXP-043 survived an hour of
  intermittent connectivity; the launcher pid never changed. **Probe, do not re-dispatch** - a
  second launcher would put 20 workers on a memory-bound laptop.
- **Repeated failed ssh connects make things worse.** Back off to long intervals.
- **The VPS is 2 cores / 4 GB: one Python worker, `nice -n 10`.** Swap thrash there leaves **no
  OOM record**, so `journalctl` being clean proves nothing.
- **A sleeping laptop does not kill a run.** Windows suspends processes.

## 6. Pointers

- Results: `experiments/04{1,2,3}_*/RESULTS.md` - **EXP-041 is the diagnosis** that explains
  everything since
- Pre-registrations: `docs/superpowers/specs/2026-08-{07,08,09,12,13}-*.md`
- Trainer seams, all **measured-neutral** against a pre-change baseline
  (`tests/training/test_encoder_seam.py`, slow marker): `encoder_state_path`,
  `max_steps_by_depth`, `stage_trace`
- Visual story: `viz/manim/story.md`, `viz/manim/README.md`
- Remote runs: `docs/playbooks/remote-experiment-runs.md`
- Laptop scripts: `scripts/laptop/sync_repo.ps1`, `scripts/laptop/probe_run.ps1`
- Previous handoff: `SESSION-HANDOFF-2026-08-13.md`
