# Session Handoff - 2026-08-14 (Fri) - Week 19 session 5 - THE RENDER IS DONE

> **Nothing is in flight. Both machines are idle. The repo is clean and pushed.**
>
> **All seven built scenes are rendered at 1080p, and `FullStory` assembles them into one
> 2:21 cut** with act cards and fades. Checked against real frames, on the laptop.
> Content Day (Sat 2026-08-16) has video assets. This file is the shorter one: the previous
> handoff (`SESSION-HANDOFF-2026-08-14.md`) still describes the science correctly and nothing in
> it was refuted today.

## 1. What happened

**The stale-data fix came first, and it mattered.** `viz/manim/data.py` read EXP-040 as the
"after" arm, so a render would have shown 0.3471 / 0.2304 / 0.1037 in place of
**0.5351 / 0.3412 / 0.1800**.

It now carries **three** arms, not two, because the story is two levers that compound and a
two-bar shot credits the wrong one:

| depth | frozen (EXP-036) | + trained encoder (EXP-040) | + depth-1 cap |
|---|---|---|---|
| 4 | 0.1591 | 0.3471 | **0.5351** (EXP-042) |
| 5 | 0.0396 | 0.2304 | **0.3412** (EXP-043) |
| 6 | 0.0000 | 0.1037 | **0.1800** (EXP-043) |

**The on-screen honesty numbers are computed, not transcribed.** `data.py` now runs the exact
paired permutation itself and reproduces EXP-043's committed values exactly: p **0.0815** at
depth 5 and **0.0273** at depth 6, and depth 6 at **+2.81 SE** with **10 of 12** seeds above the
0.10 bar. So the caption cannot drift from the experiment, and the depth-5 REFUTED verdict is on
screen rather than quietly dropped.

**Two more scenes were built the same day**, both from records rather than transcriptions:

- **`TheCurriculumUnlock`** - depth 3, log-x, the climb 0.097 -> 0.256 -> 0.397 -> 0.500 against
  the **control** that makes it a result at all: direct training at 3,000 episodes, five times
  the budget with no curriculum, scores **0.019**, below the 600-episode run. A rising line on
  its own is just more compute, so both series are on screen.
- **`TheEncoderLearns`** - the probe against depth, three series, with the margin over the raw
  observation drawn as the gap itself: **+0.003, +0.044, +0.042, +0.087**. At depth 3 the gap is
  a stub, which is the honest picture; the encoder only pulls ahead where the pixels fail.
- **`WhereWeStarted`** - the closer. EXP-029's opening column morphs into today's, in place,
  because these are the same rows: same head, same 390 parameters, same observation. Depths 5
  and 6 get a dash rather than a zero, since EXP-029 stopped at depth 4. **Its depth-3 "now" cell
  is not on today's recipe** (EXP-035, frozen encoder, 30,000 episodes) and says so on screen; an
  unmarked column would have folded two encoders and a 3x budget into one word.
- **`FullStory`** - all seven in `story.md` order, three act cards, a fade between each, **2:21**.
  It calls each scene's own `construct` rather than copying it, so one definition of every shot
  survives. `--save_sections` writes the pieces alongside the cut, which is what an editor wants.

`curriculum_climb()` replaced the transcribed `PUBLISHED_CURRICULUM`, which was already off in
its last digit (0.256 against a measured 0.2556). Records were local the whole time.

> [!warning] Records do not travel with the repo, and this bit us
> They are gitignored, so **each machine holds only the experiments it ran**. `TheEncoderLearns`
> died on the laptop with `unsupported operand type(s) for -: 'NoneType' and 'float'` inside a
> coordinate function, three layers from the cause: the laptop had **zero** EXP-039 records. They
> were copied over (100 KB), and `data.py` now names the missing experiment instead.
>
> The mirror case still holds: **the VPS has no EXP-029/030 records and the laptop has both**
> (265 and 144). That is where to get per-seed detail for `WhereWeStarted`.

## 2. The laptop is set up and does not need setting up again

| | |
|---|---|
| manim | **0.21.0**, `C:\Users\mlgbr\manim-venv` (python 3.13) |
| ffmpeg | **9.0**, `winget install --id Gyan.FFmpeg -e --scope user` |
| videos | `C:\Users\mlgbr\manim-media\videos\story_scenes\1080p60\*.mp4` |
| stills | `C:\Users\mlgbr\manim-frames\*.png` |
| driver | `scripts/laptop/render_story.ps1` (renders, then extracts stills) |

**That venv is not the project venv, deliberately** - manim pulls scipy and this repo has none on
purpose, which is why the exact permutation tests exist.

**The VPS manimpango blocker is CLOSED as won't-fix.** `python3-manimpango` is not packaged for
Ubuntu 22.04, and the laptop route worked on the first attempt. Do not spend more time on it.

## 3. Two rendering lessons worth keeping

> [!important] manim reports success on a scene whose caption sits on top of a label
> The first four scenes rendered green on the first attempt and had **three** layout defects
> between them, and the two plot scenes each had another. Stills are the check, not the log.
> `render_story.ps1` extracts them for exactly this reason.

The defects, all invisible by reading the code:

1. **Nested `arrange()` cannot build a table.** Row labels differ in width, so each row centres on
   its own width and the columns drift apart. Fixed column x and row y instead.
2. **`next_to()` chained off a left-hanging label runs off the frame edge**, because the
   inheriting line is wider than the label it takes its x from. That put two of
   `ScaleOfTheCube`'s lines against the edge.
3. **The default Pango face uses old-style figures** - 4, 7, 9 descend, 0, 1, 8 do not - and manim
   centres on the bounding box, so a row of numbers visibly steps. Numeric cells now use a lining
   monospaced face; prose keeps the default.

Also: `powershell -File script.ps1 -Scenes A,B,C` passes the whole list as **one string**. manim
then matches no scene, opens its interactive picker, reads EOF from `ssh -n`, and the whole thing
looks exactly like a render failure. The script splits it now.

### Assembling it, and where DaVinci starts

`FullStory` is the manim answer and it is the one to use: real animated transitions, one
definition of every shot, and `--save_sections` writes the pieces alongside the 2:21 cut. Plain
`ffmpeg -f concat -c copy` also works for hard cuts with no re-encode, since every clip is
already the same codec, resolution and frame rate; crossfades would need `xfade` and a re-encode.

**What manim cannot do is audio or retiming.** Every `wait()` is baked in at render time, and
pacing that reads well silently is usually too fast under narration - changing it means editing
the scene and re-rendering. Narration, music, B-roll and retiming belong in an editor, with the
section files as the timeline units. `viz/manim/README.md` has this in full.

## 4. Depth 7 pre-flight - MEASURED, and the handoff's estimate was wrong

The previous handoff said the depth-7 shell is "roughly 58,000 states". **It is 33,058.**
Measured on the VPS today, `nice`d, whole script peaking at **95 MB** resident:

| build | time | table | shell at that depth |
|---|---|---|---|
| `max_depth=6` | 0.07 s | 11,913 | 8,969 |
| `max_depth=7` | 0.27 s | 44,971 | **33,058** |
| `max_depth=8` | 0.96 s | 159,120 | 114,149 |

**So the BFS build is a non-issue and does not need designing around.** Neither does `heldout_cap`:
`split_shell` takes `min(heldout_cap, 0.25 * shell)`, and at 200 the cap already binds at depth 6
(2,242 available) and equally at depth 7 (8,264). **Evaluation cost is constant in shell size** -
it grows only with episode length, so a depth-7 eval is about 17/15 of a depth-6 one.

> [!warning] The real depth-7 confound is COVERAGE, not compute
> The training side goes from 8,769 states at depth 6 to 32,858 at depth 7 - **3.7x** - while the
> episode budget stays at 10,000. If depth 7 fails, "the task got harder" and "each state was seen
> a quarter as often" are **not separable** unless the design says so in advance. Pre-register it:
> either hold episodes-per-state roughly fixed (which means about 37,000 episodes and a much
> longer run) or run a budget arm, as EXP-035 did for depth 3.

## 5. Next, in the order I would take it

1. **Depth 7** (see the coverage warning above). The frontier is unmeasured for the first time
   since EXP-036, so this is the interesting one.
2. **Re-run depth 5 with 24+ seeds** to settle EXP-043's Claim 1. It is the clearest case in the
   project where 12 seeds is the binding constraint rather than the effect.
3. **One scene is still story-only:** `PolicyCollapse`, which needs a cube renderer that does not
   exist yet. It is the most visceral shot in the deck - two cubes side by side, one playing the
   same move forever - and it is the only thing the video is missing.
4. **Fine-tune the encoder during RL** - still untested, and the frozen version now carries four
   working depths.
5. `^0817` **needs Michael**: Phase 0 / Phase 1 checkpoints in the vault `progress-tracker`.

## 6. Pointers

- Visual story: `viz/manim/story.md` (arc), `viz/manim/README.md` (how to render), `data.py`
- Render driver: `scripts/laptop/render_story.ps1`
- Yesterday's handoff, still accurate on the science: `SESSION-HANDOFF-2026-08-14.md`
- Remote runs: `docs/playbooks/remote-experiment-runs.md`
