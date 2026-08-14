# Prompt for a fresh session

Paste the block below into a new session. It is deliberately short - the handoff carries the
detail, and duplicating it here would create two versions that drift.

---

```
Picking up the neuromorphic cube project. Nothing is running; both machines are idle.

Read docs/handoffs/SESSION-HANDOFF-2026-08-14.md first, then CLAUDE.md.

Where we are: a spiking network learns to solve a 2x2 Rubik's cube. Depths 3 through 6
now all work (0.3972 / 0.5351 / 0.3412 / 0.1800), and the break point - stuck at depth 5
since EXP-036 - is now past depth 6 and unmeasured. Two levers got us here and they
compound: training the sensory encoder self-supervised, and capping the depth-1 training
step budget at 2. That second one closed a trap where curriculum stage 1 paid a
constant-action policy 0.3333 against a random policy's 0.2208, because a cube face has
order 4 and the 2d+3 budget let a repeated move cycle back to solved.

YOUR JOB THIS SESSION: render the visual-story scenes on the laptop. Content Day is
Saturday Aug 16 and they have never been rendered once. Handoff section 3 is the task.

Do these in order:

1. FIX THE STALE DATA FIRST. viz/manim/data.py depth_curve() still reads EXP-040 as the
   "after" arm, so it would render 0.3471/0.2304/0.1037 instead of the current
   0.5351/0.3412/0.1800. Point it at EXP-042 (depth 4, tag exp042_capped) and EXP-043
   (depths 5-6, tag exp043_capped). Update viz/manim/story.md's Act 3 table to match.
   Rendering before this fix understates the result by a third.

2. Set the laptop up. It HAS the experiment records and python, but it does NOT have
   ffmpeg (manim cannot encode video without it) and does not have manim. Install manim
   into a SEPARATE venv, never the project venv - manim pulls scipy, and this repo
   deliberately has none, which is why the exact permutation tests exist.

   Do NOT try to render on the VPS. manimpango will not build there; three attempts are
   recorded in viz/manim/README.md.

3. Render, starting with TheBreakPointMoves at -ql. It is the payoff shot and the one
   most affected by the stale data. The scenes have never been rendered, so expect a
   layout fixup pass - positioning is what you cannot get right by reading. The other
   three are TheWall, ScaleOfTheCube, CollapseIsASymptom. None need LaTeX.

Keep the video framing honest, per viz/manim/story.md: this is still 0.32% of the cube,
a random scramble sits at depth 11, and the wins came from six refutations before the two
things that worked.

If the render finishes early, the next experiment is finding the new break point at depth
7 - but check the BFS build time and heldout_cap first, since the depth-7 shell is ~58,000
states against depth 6's 8,969.
```

---

## Why this prompt is shaped this way

- **It names the file to read rather than restating it**, so the two cannot drift apart.
- **It leads with the two levers**, because every current number depends on both and neither is
  guessable from the code.
- **"Fix the stale data" is task 1, not a footnote.** Rendering first and noticing later would
  mean redoing every scene, two days before a deadline.
- **It names the scipy constraint explicitly**, because installing manim into the project venv
  would silently remove the property the methodology rests on and nothing would fail visibly.
- **It carries the honest framing**, so the video does not overclaim on a genuinely good result.
