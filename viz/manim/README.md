# Manim scenes for the project's visual story

Explanatory animation for talks and video, driven by the committed experiment records.

**The story arc is in `story.md`. The numbers all come from `data.py`.**

## Which manim

**Use ManimCE (Community Edition), not `3b1b/manim`.**

`3b1b/manim` is Grant Sanderson's personal codebase - the one 3Blue1Brown videos are actually
made with - but it is explicitly not supported for outside use, has sparse docs, and changes
without notice. **ManimCE is the maintained fork**: versioned pip releases, real documentation,
same API lineage.

**You do not clone it.** It is a normal package:

```bash
.venv/bin/pip install manim
```

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

## Rendering

```bash
cd viz/manim
../../.venv/bin/python data.py                              # sanity: print the real numbers first
manim -ql scenes/story_scenes.py TheBreakPointMoves         # 480p, fast, iterate
manim -qh scenes/story_scenes.py TheBreakPointMoves         # 1080p
manim -qk scenes/story_scenes.py TheBreakPointMoves         # 4K, final only
```

Scenes: `TheBreakPointMoves`, `TheWall`, `ScaleOfTheCube`, `CollapseIsASymptom`.

> [!warning] Where to render
> **Not on this VPS beyond `-ql`.** It is 2 cores / 4 GB, and one worker is the standing limit -
> two parallel Python workers once drove it into swap thrash, which makes the box unresponsive
> *without* ever tripping the OOM killer. Final renders belong on the laptop (22 cores, 31 GB)
> or the desktop.

> [!important] These scenes have never been rendered
> They were written while the VPS was busy with the seed diagnosis, so layout is unverified.
> Positioning is the part you cannot get right by reading - expect a first-render fixup pass.

## The rule that keeps this honest

**Scenes must not hardcode results.** `data.py` is the single source of truth and reads
`experiments/*/outputs/*.json` directly, so an animation cannot drift from the experiment behind
it. Two provenance classes are kept distinct:

- `measured_*` / the loader functions - read from records present on this machine
- `PUBLISHED_*` - transcribed from a committed `RESULTS.md`, because those raw records are
  gitignored and were never fetched here

`python data.py` prints which experiments have records and which are published-only.

## Data gaps

Records absent locally: **EXP-027, EXP-029, EXP-030**. Their published means are in `data.py`,
which covers every scene in `story.md`.

**They would only be needed for per-seed detail** - a scatter of EXP-029's twelve seeds, say.
The means alone cannot produce that. If those records exist on the desktop or laptop, copying
`experiments/029_cube_baseline/outputs/` and `030_memory_engagement/outputs/` here would unlock
it; nothing in the current story depends on it.
