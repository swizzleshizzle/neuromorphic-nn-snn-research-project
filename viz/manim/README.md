# Manim scenes for the project's visual story

Explanatory animation for talks and video, driven by the committed experiment records.

**The story arc is in `story.md`. The numbers all come from `data.py`.**

## Which manim

**Use ManimCE (Community Edition), not `3b1b/manim`.**

`3b1b/manim` is Grant Sanderson's personal codebase - the one 3Blue1Brown videos are actually
made with - but it is explicitly not supported for outside use, has sparse docs, and changes
without notice. **ManimCE is the maintained fork**: versioned pip releases, real documentation,
same API lineage.

**You do not clone it.** It is a normal package - but **never into the project venv**, because its
dependency tree pulls scipy:

```bash
python -m venv ~/manim-venv && ~/manim-venv/bin/pip install manim
```

## BLOCKED on this box: `manimpango` will not build (2026-08-12)

**Manim installs; importing it fails.** Do not repeat these three attempts.

```
ModuleNotFoundError: No module named 'manimpango._register_font'
```

`site-packages/manimpango/` contains only Cython **sources** (`_register_font.pyx`, `.pxd`) and
**no compiled `.so`** - the build backend produced a wheel without cythonising the extensions.

Tried and did **not** work:

1. plain `uv pip install manim` - pulled the manimpango wheel, same missing module
2. `--reinstall --no-cache --no-binary manimpango` - "Built manimpango==0.6.1", still no `.so`
3. adding `cython`/`setuptools`/`wheel` and `--no-build-isolation` - unchanged

What **was** fixed along the way, and is now in place:

- `libcairo2-dev`, `libpango1.0-dev`, `pkg-config` installed system-wide (cairo 1.16.0,
  pango 1.50.6). `pycairo` builds fine now; it was the original blocker.
- The manim venv is **isolated at `/root/scratch/manim-venv`** (python 3.11), deliberately
  **not** the project venv - manim's dependency tree would drag **scipy** in, and this repo has
  none on purpose, which is why the exact permutation tests exist. Verified after installing:
  `import scipy` in `.venv` still fails, as it should.

**CLOSED as won't-fix, 2026-08-14.** The laptop route was taken instead and worked first time:
`pip install manim` there pulled a working `manimpango` 0.6.1 cp313 wheel. `python3-manimpango`
is **not** packaged for Ubuntu 22.04 (`apt-cache policy` returns nothing), so the one untried
idea with real odds is gone too. Do not spend more time on this box; render on the laptop.

## Dependencies on this box

Checked 2026-08-11:

| | |
|---|---|
| ffmpeg | present |
| cairo / pango | present |
| disk | 53 G free |
| python | 3.10 |
| **LaTeX** | **absent** |

**LaTeX is not needed.** These scenes use `Text` (Pango) and never `MathTex`/`Tex`. Installing
texlive costs 1-4 GB; only do it if a scene genuinely needs typeset formulas.

## Rendering - do it on the laptop

**All four scenes render on `swizzlesduo`, verified 2026-08-14** at `-ql` and `-qh`. The VPS
cannot render at any quality (see the manimpango section above); it can still run
`data.py`, which is the useful half of the sanity check.

The laptop was set up on 2026-08-14 and does not need setting up again:

| | |
|---|---|
| manim | **0.21.0** in `C:\Users\mlgbr\manim-venv` (python 3.13) |
| ffmpeg | **9.0**, `winget install --id Gyan.FFmpeg --scope user` |
| media output | `C:\Users\mlgbr\manim-media` (outside the repo, so nothing lands in git) |
| stills | `C:\Users\mlgbr\manim-frames` |

**That venv is deliberately not the project venv.** It contains scipy, pulled in by manim's
dependency tree; the project venv must stay scipy-free, which is why the exact permutation tests
exist. Verify with `.venv\Scripts\python.exe -c "import scipy"` still failing.

```bash
# from the VPS. sync first: the laptop's checkout blocks on untracked head checkpoints.
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\sync_repo.ps1'
ssh -n laptop 'powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project\scripts\laptop\render_story.ps1 -Quality qh'
scp 'laptop:C:/Users/mlgbr/manim-frames/*.png' ./frames/     # then LOOK at them
```

`render_story.ps1` renders and then pulls stills out of each mp4, because **manim reports success
on a scene whose caption is sitting on top of a label.** Green logs are not a rendered video; the
stills are the actual check. `-Quality ql|qm|qh|qk`, `-Scenes A,B,C`, `-Frames N`.

Scenes: `TheBreakPointMoves`, `TheWall`, `ScaleOfTheCube`, `CollapseIsASymptom`,
`TheCurriculumUnlock`, `TheEncoderLearns`, `WhereWeStarted`, `PolicyCollapse`, and `FullStory`.

## The one scene with recorded input

`PolicyCollapse` replays two real rollouts rather than drawing a cube turning. Regenerate them
only in the **project** venv - it needs torch and the brain, and the manim venv has neither:

```bash
.venv/bin/python viz/manim/record_traces.py     # writes viz/manim/traces.json, committed
.venv/bin/python viz/manim/cube_net.py          # net geometry self-check, prints the layout
```

`record_traces.py` rebuilds each arm's config from **its own record** rather than a
re-specification, so a replay cannot silently differ from the run that trained the checkpoint.
Verify any regenerated `traces.json` against the cube model before committing it - every frame
should replay under `apply_move`, and the `solved` flag should match `is_solved`. This repo has
already had a cube frame labelled `solved: yes` on a scrambled cube pass every unit test.

`cube_net.py` is the Python twin of `dashboard/src/panels/cubeNet.ts` and carries the same corner
check, so the video and the dashboard draw the same cube. **Both had every face's rows inverted
until 2026-08-15.**

## Assembling the whole arc

**`FullStory` does it in manim, and that is the right tool for this part.** It plays all seven
scenes in `story.md` order with act cards and a fade between each: **2 minutes 21 seconds** at
1080p60.

```bash
ssh -n laptop 'powershell ... render_story.ps1 -Quality qh -Full'
# or directly:  manim -qh --save_sections scenes/story_scenes.py FullStory
```

It calls each scene's own `construct`, so there is **one definition of every shot** - fix a
layout once and the assembled cut gets it too. `--save_sections` writes the pieces as well as
the continuous cut:

```
videos/story_scenes/1080p60/FullStory.mp4                    the whole arc
videos/story_scenes/1080p60/sections/FullStory_0003_TheCurriculumUnlock.mp4   ...and each part
```

**If you only want the existing clips glued together**, ffmpeg concatenates without re-encoding,
because every scene is already the same codec, resolution and frame rate. Hard cuts only:

```bash
printf "file '%s'\n" *.mp4 > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c copy out.mp4      # instant, no quality loss
```

Crossfades need the `xfade` filter and therefore a re-encode, at which point `FullStory` is
better: its transitions are real animations rather than an effect laid over two frozen frames.

### What still belongs in DaVinci

Manim assembles **video**; it is not an editor. Take it into DaVinci for:

- **narration, music, and audio** of any kind. Nothing here has a sound track;
- **retiming against a voice track.** This is the big one. Every `wait()` is baked in at render
  time, and a beat that reads well silently is usually too fast under narration. Retiming in
  manim means editing the scene and re-rendering; in an editor it is a drag;
- **B-roll**, screen capture, talking head, thumbnails, titles in your own template.

The practical workflow is both: render `FullStory` for the assembled cut to work against, and
drop the **section files** on the timeline as the editing units.

> [!important] What the first render actually broke
> Three defects, none of which are visible by reading the code:
> - **Nested `arrange()` cannot build a table.** Row labels differ in width, so each row is
>   centred on its own width and the columns drift apart. Use fixed column x and row y.
> - **`next_to()` chained off a left-hanging label runs off the frame**, because the inheriting
>   line is wider than the label it takes its x from.
> - **The default Pango face uses old-style figures** - 4, 7, 9 descend and 0, 1, 8 do not - and
>   manim centres on the bounding box, so a row of numbers visibly steps. Numeric cells use
>   `Consolas`; prose keeps the default face.

## The rule that keeps this honest

**Scenes must not hardcode results.** `data.py` is the single source of truth and reads
`experiments/*/outputs/*.json` directly, so an animation cannot drift from the experiment behind
it. Two provenance classes are kept distinct:

- `measured_*` / the loader functions - read from records present on this machine
- `PUBLISHED_*` - transcribed from a committed `RESULTS.md`, because those raw records are
  gitignored and were never fetched here

`python data.py` prints which experiments have records and which are published-only.

## Data gaps - and they differ per machine

**Records are gitignored, so each machine holds only the experiments it ran.** That is a real
constraint on rendering, not a footnote: `TheEncoderLearns` failed on the laptop because EXP-039
had never been there. The loaders in `data.py` now name the missing experiment instead of dying
as a `NoneType` inside a scene's coordinate function.

| | VPS | laptop |
|---|---|---|
| EXP-029, EXP-030 | **absent** | present (265, 144) |
| EXP-031 | present (144) | **absent** |
| EXP-039 | present (12) | copied there 2026-08-14 |

Everything the six built scenes need is now on the laptop. **EXP-027/029/030's published means
are in `data.py`** and cover the story; they would only be needed at per-seed resolution - a
scatter of EXP-029's twelve seeds, say, which a mean cannot produce. The laptop has those
records if `WhereWeStarted` ever wants one.

**Copy records, never transcribe them.** A second copy of a number is a number that can drift:
`PUBLISHED_CURRICULUM` was already off in its last digit before it was deleted.
