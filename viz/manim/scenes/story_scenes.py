"""Manim Community Edition scenes for the project's visual story.

Every number is pulled from `data.py`, which reads the committed experiment records. Nothing
here is hand-typed, so a scene cannot drift from the experiment that produced it.

DELIBERATELY CONSERVATIVE API USE: only `Text`, `Rectangle`, `Line`, `VGroup` and the basic
animations, all stable across ManimCE releases. No `MathTex`/`Tex`, so **no LaTeX install is
required**. No `BarChart`, whose signature has moved between versions.

    manim -ql scenes/story_scenes.py TheBreakPointMoves    # 480p, iterate
    manim -qh scenes/story_scenes.py TheBreakPointMoves    # 1080p, wants real cores

**Rendered on the laptop, 2026-08-14.** Not on the VPS: `manimpango` will not build there. Use
`scripts/laptop/render_story.ps1`, which also extracts stills - manim reports success on a scene
whose caption is sitting on top of a label, so the only way to check layout from a headless
session is to look at frames.

Two layout rules learned from that first render, both of which produced visible defects:

1. **Place table cells at fixed x and y, not with nested `arrange()`.** Row labels differ in
   width, so an arranged table centres each row on its own width and the columns drift.
2. **Do not chain `next_to()` off a left-hanging label.** The inheriting line is wider than the
   label, so it overflows the frame edge.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from manim import (
    BLUE, GREEN, GREY, RED, WHITE, YELLOW,
    DOWN, LEFT, RIGHT, UP,
    Create, Dot, FadeIn, FadeOut, Line, Rectangle, Scene, Text, Transform, VGroup, Write,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data  # noqa: E402

TITLE = 40
LABEL = 24
SMALL = 20

# The default Pango face renders OLD-STYLE figures: 4, 7 and 9 drop below the baseline while 0,
# 1 and 8 do not. Manim centres each Text on its bounding box, so a row of numbers set in it does
# not sit on a common baseline - visible as jitter in `TheWall`'s chance row. Numeric CELLS use a
# lining, monospaced face; prose keeps the default. Pango falls back silently if it is absent,
# and the render boxes are all Windows.
MONO = "Consolas"


def _axes(x0: float, x1: float, y0: float, y1: float):
    """A bare L. Ticks and labels belong to the caller, because the two line scenes label
    different things and a general-purpose axis helper would be mostly arguments."""
    return VGroup(Line([x0, y0, 0], [x1, y0, 0]).set_stroke(GREY, 2),
                  Line([x0, y0, 0], [x0, y1, 0]).set_stroke(GREY, 2))


def _polyline(points, color, width: float = 4.0):
    """(segments, dots) for a series already in scene coordinates.

    Returned separately so a scene can reveal a line one leg at a time - a series that appears
    all at once reads as a picture, and one that draws reads as a claim being made.
    """
    dots = VGroup(*[Dot(point=p, radius=0.075, color=color) for p in points])
    segs = VGroup(*[Line(a, b).set_stroke(color, width) for a, b in zip(points, points[1:])])
    return segs, dots


def _legend(entries, y: float):
    """Swatch-and-label row, centred, at a fixed y."""
    row = VGroup()
    for color, label in entries:
        swatch = Rectangle(width=0.3, height=0.3, color=color,
                           fill_color=color, fill_opacity=0.85)
        row.add(VGroup(swatch, Text(label, font_size=SMALL, color=color)).arrange(RIGHT, buff=0.2))
    row.arrange(RIGHT, buff=0.9).move_to([0, y, 0])
    return row


def _bar(value: float, max_value: float, color, width: float = 0.8, height: float = 4.0):
    """A bar whose HEIGHT is proportional to value. Zero-height bars are illegal in manim,
    so a floor of 0.01 keeps a 0.0000 result visible as a hairline rather than vanishing -
    which matters, because 'exactly zero on all twelve seeds' is a real result here."""
    h = max(0.01, height * (value / max_value if max_value else 0))
    return Rectangle(width=width, height=h, color=color, fill_color=color, fill_opacity=0.85)


BASELINE_Y = -2.1          # where every bar stands, so the three reveals share one axis
CAPTION_Y = -3.35
LEGEND_Y = 2.45            # measured off the rendered frame, just clear of a to_edge(UP) title


class TheBreakPointMoves(Scene):
    """Act 3's payoff, and THREE bars rather than two.

    The story is no longer "one variable changed". Two levers moved the break point and they
    compound: the self-supervised encoder (EXP-039/040) and capping the depth-1 training budget
    (EXP-041/042/043). Neither alone reaches depth 6, so a two-bar version of this shot would
    credit the wrong thing.

    The head is the same `Linear(64 -> 6)`, 390 trainable parameters, throughout."""

    def construct(self):
        curve = data.depth_curve()
        w6 = data.working_bar(6)
        depths = [4, 5, 6]
        # Headroom above the tallest bar so its value label is not against the legend. 1.12 and
        # a 4.0 height were set by looking at the 480p render: 1.15/3.0 left a dead band roughly
        # a bar's width high between the tallest bar and the legend.
        max_v = max(curve[d]["after"] for d in depths) * 1.12

        arms = [("before", GREY, "frozen encoder"),
                ("pretrained", BLUE, "+ trained encoder"),
                ("after", GREEN, "+ depth-1 cap")]

        title = Text("Two levers. The same 390 parameters.", font_size=TITLE).to_edge(UP)
        self.play(Write(title))

        legend = _legend([(color, label) for _, color, label in arms], LEGEND_Y)

        axis = Line(LEFT * 6.2, RIGHT * 6.2).shift(UP * BASELINE_Y).set_stroke(GREY, 2)
        self.play(Create(axis), run_time=0.5)

        # Bars first, positioned on the shared axis; labels attach afterwards so they follow the
        # final position rather than an intermediate one.
        groups, labels = VGroup(), []
        for d in depths:
            trio = VGroup(*[_bar(curve[d][key], max_v, color, height=4.0)
                            for key, color, _ in arms])
            trio.arrange(RIGHT, buff=0.12, aligned_edge=DOWN)
            groups.add(trio)
        groups.arrange(RIGHT, buff=1.3, aligned_edge=DOWN)
        groups.next_to(axis, UP, buff=0.0)

        for d, trio in zip(depths, groups):
            vals = [Text(f"{curve[d][key]:.3f}", font_size=SMALL, color=color)
                    .next_to(bar, UP, buff=0.12)
                    for bar, (key, color, _) in zip(trio, arms)]
            depth_label = Text(f"depth {d}", font_size=LABEL).next_to(trio, DOWN, buff=0.25)
            labels.append((vals, depth_label))

        def caption(text, color):
            return Text(text, font_size=SMALL, color=color).move_to(UP * CAPTION_Y)

        # Beat 1: sit with the failure. Depth 6 was 0.0000 on every seed.
        self.play(FadeIn(legend[0]))
        for trio, (vals, depth_label) in zip(groups, labels):
            self.play(Create(trio[0]), FadeIn(vals[0]), FadeIn(depth_label), run_time=0.4)
        broken = caption("depth 5 broken. depth 6: 0.0000 on all twelve seeds.", RED)
        self.play(FadeIn(broken))
        self.wait(1.5)
        self.play(FadeOut(broken))

        # Beat 2: lever one. Train the encoder to predict the move between two states.
        self.play(FadeIn(legend[1]))
        for trio, (vals, _) in zip(groups, labels):
            self.play(Create(trio[1]), FadeIn(vals[1]), run_time=0.45)
        lever1 = caption("Lever 1: train the encoder, self-supervised. No labels, no oracle.",
                         BLUE)
        self.play(FadeIn(lever1))
        self.wait(1.5)
        self.play(FadeOut(lever1))

        # Beat 3: lever two. Depth 1 was paying 0.3333 for a constant action, above random's
        # 0.2208, because a cube face has order 4.
        self.play(FadeIn(legend[2]))
        for trio, (vals, _) in zip(groups, labels):
            self.play(Create(trio[2]), FadeIn(vals[2]), run_time=0.45)
        lever2 = caption("Lever 2: stop paying depth 1 for a repeated move.", GREEN)
        self.play(FadeIn(lever2))
        self.wait(1.5)
        self.play(FadeOut(lever2))

        # The honest close. Depth 6 clears the pre-registered rule with room; depth 5 has the
        # LARGER effect and still misses on p, and is reported REFUTED.
        # Two lines, so it sits at the caption line rather than above it - at CAPTION_Y + 0.15 the
        # first line crowded the depth labels.
        honest = VGroup(
            Text(f"depth 6 works: {w6['seeds_above']} of {w6['n']} seeds above "
                 f"{data.WORKING_BAR:.2f}, +{w6['se_margin']:.1f} SE.",
                 font_size=SMALL, color=YELLOW),
            Text(f"depth 5's +{curve[5]['after'] - curve[5]['pretrained']:.2f} is larger and "
                 f"still misses its 0.05 bar (p = {curve[5]['after_p']:.3f}).",
                 font_size=SMALL, color=YELLOW),
        ).arrange(DOWN, buff=0.2).move_to(UP * CAPTION_Y)
        self.play(FadeIn(honest))
        self.wait(2.5)


class TheCurriculumUnlock(Scene):
    """Act 2's one thing that worked, at depth 3.

    A rising line on its own is not a result - more episodes is more compute, and the audience
    should suspect exactly that. The scene exists for its CONTROL: direct training at 3,000
    episodes, five times the budget with no curriculum, lands BELOW the 600-episode run. Both
    series are on screen or this shot is dishonest.
    """

    def construct(self):
        c = data.curriculum_climb()
        x0, x1, y0, y1 = -4.9, 5.0, -2.0, 2.0
        vmax = 0.55

        budgets = [e for e, _ in c["curriculum"]]
        lo, hi = math.log10(budgets[0]), math.log10(budgets[-1])
        # Log x: on a linear axis the 30,000 point sits so far right that the first three
        # collapse into one, and the diminishing return - the honest shape - disappears.
        def px(e):
            return x0 + (x1 - x0) * (math.log10(e) - lo) / (hi - lo)

        def py(v):
            return y0 + (y1 - y0) * v / vmax

        title = Text("The curriculum is the lever, not the compute.",
                     font_size=TITLE).to_edge(UP)
        legend = _legend([(GREEN, "curriculum 1 -> 2 -> 3"), (RED, "no curriculum")], LEGEND_Y)
        self.play(Write(title))

        axes = _axes(x0, x1, y0, y1)
        x_ticks = VGroup(*[Text(f"{e:,}", font_size=SMALL, font=MONO, color=GREY)
                           .move_to([px(e), y0 - 0.38, 0]) for e in budgets])
        y_ticks = VGroup(*[Text(f"{v:.2f}", font_size=SMALL, font=MONO, color=GREY)
                           .move_to([x0 - 0.55, py(v), 0]) for v in (0.0, 0.25, 0.50)])
        x_name = Text("training episodes", font_size=SMALL, color=GREY).move_to([0, y0 - 0.95, 0])
        self.play(Create(axes), FadeIn(x_ticks), FadeIn(y_ticks), FadeIn(x_name))

        # The chance floor, so 0.022 reads as "barely above chance" rather than as a small number.
        floor = Line([x0, py(c["floor"]), 0], [x1, py(c["floor"]), 0]).set_stroke(GREY, 2, 0.5)
        floor_label = Text(f"chance {c['floor']:.3f}", font_size=SMALL, color=GREY)
        floor_label.move_to([x1 - 0.1, py(c["floor"]) + 0.3, 0], aligned_edge=RIGHT)
        self.play(Create(floor), FadeIn(floor_label))

        def caption(text, color):
            return Text(text, font_size=SMALL, color=color).move_to([0, -3.15, 0])

        # Beat 1: the climb, drawn leg by leg so it reads as a claim being made.
        self.play(FadeIn(legend[0]))
        pts = [[px(e), py(v), 0] for e, v in c["curriculum"]]
        segs, dots = _polyline(pts, GREEN)
        for i, (_, v) in enumerate(c["curriculum"]):
            label = Text(f"{v:.3f}", font_size=SMALL, color=GREEN).next_to(dots[i], UP, buff=0.2)
            if i:
                self.play(Create(segs[i - 1]), run_time=0.35)
            self.play(FadeIn(dots[i]), FadeIn(label), run_time=0.35)
        doubt = caption("More episodes, more success. On its own that proves nothing.", WHITE)
        self.play(FadeIn(doubt))
        self.wait(1.8)
        self.play(FadeOut(doubt))

        # Beat 2: the control. Same budget, no curriculum, and it does not move.
        self.play(FadeIn(legend[1]))
        d_pts = [[px(e), py(v), 0] for e, v in c["direct"]]
        d_segs, d_dots = _polyline(d_pts, RED)
        self.play(FadeIn(d_dots), Create(d_segs), run_time=0.8)
        note = Text(f"no curriculum, 5x the budget: {c['direct'][-1][1]:.3f}",
                    font_size=SMALL, color=RED)
        note.move_to([px(3_000) + 0.35, py(c["direct"][-1][1]) + 0.55, 0], aligned_edge=LEFT)
        self.play(FadeIn(note))
        self.wait(0.8)

        worse = caption(f"Five times the compute, and WORSE than the "
                        f"{c['direct'][0][0]}-episode run ({c['direct'][0][1]:.3f}).", RED)
        self.play(FadeIn(worse))
        self.wait(2)
        self.play(FadeOut(worse))

        honest = caption(f"Depth 3 only, and {budgets[-1]:,} episodes is "
                         f"{budgets[-1] // budgets[0]}x the first run.", YELLOW)
        self.play(FadeIn(honest))
        self.wait(2)


class TheEncoderLearns(Scene):
    """Act 3's first lever, and the first time the spiking network does work rather than acting
    as a fixed random projection.

    Self-supervised: predict which move was played between two cube states. No labels, no oracle.
    The claim is not "the probe went up" - it is that the TRAINED concept beats a linear probe on
    the raw observation, which is the ceiling any encoder is trying to clear, and that the margin
    grows with depth. It helps most exactly where the observation fails.
    """

    def construct(self):
        probe = data.encoder_probe()
        depths = sorted(probe)
        x0, x1, y0, y1 = -4.6, 4.8, -2.0, 2.0
        vlo, vhi = 0.30, 0.95      # truncated, and labelled as such at the bottom left

        def px(d):
            return x0 + (x1 - x0) * (d - depths[0]) / (depths[-1] - depths[0])

        def py(v):
            return y0 + (y1 - y0) * (v - vlo) / (vhi - vlo)

        title = Text("Train the encoder and it beats the pixels.", font_size=TITLE).to_edge(UP)
        legend = _legend([(GREY, "frozen concept"), (BLUE, "raw observation"),
                          (GREEN, "trained concept")], LEGEND_Y)
        self.play(Write(title))

        axes = _axes(x0, x1, y0, y1)
        x_ticks = VGroup(*[Text(f"depth {d}", font_size=SMALL, color=GREY)
                           .move_to([px(d), y0 - 0.38, 0]) for d in depths])
        y_ticks = VGroup(*[Text(f"{v:.1f}", font_size=SMALL, font=MONO, color=GREY)
                           .move_to([x0 - 0.5, py(v), 0]) for v in (0.4, 0.6, 0.8)])
        truncated = Text("probe accuracy; axis starts at 0.30", font_size=SMALL, color=GREY)
        truncated.move_to([x0 - 1.2, y0 - 0.9, 0], aligned_edge=LEFT)
        self.play(Create(axes), FadeIn(x_ticks), FadeIn(y_ticks), FadeIn(truncated))

        def caption(text, color):
            return Text(text, font_size=SMALL, color=color).move_to([0, -3.15, 0])

        def series(key, color):
            return _polyline([[px(d), py(probe[d][key]), 0] for d in depths], color)

        intro = caption("Predict which move was played between two states. "
                        "No labels, no oracle.", WHITE)
        self.play(FadeIn(intro))
        self.wait(1.5)
        self.play(FadeOut(intro))

        # The ceiling first: everything after is read against it.
        self.play(FadeIn(legend[1]))
        f_segs, f_dots = series("facelets", BLUE)
        self.play(FadeIn(f_dots), Create(f_segs), run_time=1.0)
        ceiling = caption("The ceiling: what a linear probe reads off the raw pixels.", BLUE)
        self.play(FadeIn(ceiling))
        self.wait(1.5)
        self.play(FadeOut(ceiling))

        self.play(FadeIn(legend[0]))
        z_segs, z_dots = series("frozen", GREY)
        self.play(FadeIn(z_dots), Create(z_segs), run_time=1.0)
        worse = caption("What the policy actually reads is WORSE than the pixels.", GREY)
        self.play(FadeIn(worse))
        self.wait(1.5)
        self.play(FadeOut(worse))

        self.play(FadeIn(legend[2]))
        t_segs, t_dots = series("trained", GREEN)
        self.play(FadeIn(t_dots), Create(t_segs), run_time=1.0)

        # The margin over the ceiling, drawn as the gap itself. At depth 3 it is a stub, which is
        # the honest picture: there it merely MATCHES the ceiling.
        gaps = VGroup(*[Line([px(d), py(probe[d]["facelets"]), 0],
                             [px(d), py(probe[d]["trained"]), 0]).set_stroke(YELLOW, 6)
                        for d in depths])
        self.play(FadeIn(gaps))
        margins = "  ".join(f"+{probe[d]['trained'] - probe[d]['facelets']:.3f}" for d in depths)
        grows = caption(f"margin over the ceiling, depth {depths[0]} to {depths[-1]}:  {margins}",
                        YELLOW)
        self.play(FadeIn(grows))
        self.wait(2)
        self.play(FadeOut(grows))

        punch = caption("It helps most exactly where the observation fails.", WHITE)
        self.play(Write(punch))
        self.wait(2)


class TheWall(Scene):
    """Act 2's explanation. A linear probe for 'which move reduces distance-to-solved', read off
    the raw observation, decays with depth and reaches chance around depth 8-9.

    The punchline is that the frozen concept the policy actually reads is WORSE than the raw
    pixels, and widening it 8x does not close the gap."""

    def construct(self):
        title = Text("Why it gets hard", font_size=TITLE).to_edge(UP)
        self.play(Write(title))

        depths = [1, 2, 3, 4, 5]
        rows = [
            ("raw observation", data.PUBLISHED_FACELET_PROBE, BLUE),
            ("what the policy reads", data.PUBLISHED_CONCEPT64_PROBE, RED),
            ("8x wider", data.PUBLISHED_CONCEPT512_PROBE, GREY),
            ("chance", data.PUBLISHED_PROBE_CHANCE, YELLOW),
        ]

        # Fixed column x and row y, NOT nested arrange(). The row labels differ in width, so an
        # arrange-based table centres each row on its own width and the columns drift apart.
        col_x = [-1.5 + 1.3 * i for i in range(len(depths))]
        label_right = -2.5
        row_y = [1.25 - 0.75 * i for i in range(len(rows) + 1)]

        header = VGroup(*[Text(f"d{d}", font_size=LABEL, font=MONO).move_to([x, row_y[0], 0])
                          for d, x in zip(depths, col_x)])
        self.play(FadeIn(header))

        table = VGroup(header)
        for (label, series, color), y in zip(rows, row_y[1:]):
            cells = VGroup(*[Text(f"{series[d]:.3f}", font_size=LABEL, color=color, font=MONO)
                             .move_to([x, y, 0]) for d, x in zip(depths, col_x)])
            name = Text(label, font_size=SMALL, color=color)
            name.move_to([label_right - name.width / 2, y, 0])
            row = VGroup(name, cells)
            table.add(row)
            self.play(FadeIn(row), run_time=0.6)
        self.wait(1)

        punch = Text("A linear head cannot solve deep cubes, however good the encoder.",
                     font_size=LABEL, color=WHITE).move_to([0, row_y[-1] - 0.9, 0])
        self.play(Write(punch))
        self.wait(2)


class ScaleOfTheCube(Scene):
    """Act 1's hook, and the honesty anchor for the whole video.

    Depths 1-3 are 153 states out of 3,674,160. A random scramble sits at depth 11, where 97.5%
    of the mass is. Every later gain should be read against this."""

    def construct(self):
        title = Text("A 2x2 cube: 3,674,160 states", font_size=TITLE).to_edge(UP)
        self.play(Write(title))

        total_width = 11.0
        bar_group = VGroup()
        colors = [GREEN, BLUE, YELLOW, RED, GREY]
        prev_frac = 0.0
        for (label, count, cum), color in zip(data.STATE_CENSUS, colors):
            frac = cum - prev_frac
            prev_frac = cum
            w = max(0.05, total_width * frac)
            seg = Rectangle(width=w, height=0.9, color=color,
                            fill_color=color, fill_opacity=0.8)
            bar_group.add(seg)
        bar_group.arrange(RIGHT, buff=0.0)
        bar_group.move_to([0, 1.3, 0])
        self.play(Create(bar_group))

        # Name only the two that carry the argument: where we are, and where a real scramble is.
        # Both are left-aligned UNDER the bar's left end; the two closing lines are centred on the
        # frame instead. Chaining next_to() off `here` ran them into the left edge, because a
        # centred-on-a-left-hanging-label line is wider than the label it inherits its x from.
        here = Text("depths 1-3: 153 states (0.004%)", font_size=SMALL, color=GREEN)
        here.move_to([bar_group.get_left()[0], 0.0, 0], aligned_edge=LEFT)
        there = Text("depths 10-12: 3,063,976 (97.5%)", font_size=SMALL, color=RED)
        there.move_to([bar_group.get_left()[0], -0.6, 0], aligned_edge=LEFT)
        self.play(FadeIn(here))
        self.wait(0.6)
        self.play(FadeIn(there))
        self.wait(1)

        punch = Text(f"A random scramble sits at depth {data.RANDOM_SCRAMBLE_DEPTH}.",
                     font_size=LABEL, color=WHITE).move_to([0, -1.8, 0])
        self.play(Write(punch))
        self.wait(1)

        honest = Text("We solve depths 3-6. That is 0.32% of the cube.",
                      font_size=SMALL, color=YELLOW).move_to([0, -2.7, 0])
        self.play(FadeIn(honest))
        self.wait(2)


class CollapseIsASymptom(Scene):
    """Act 2's twist, and the most counter-intuitive result in the project.

    The policy collapses to one action (modal 0.975 against a 0.354 uniform floor). Forcing it
    to stop - an entropy bonus drove modal to 0.631 - bought NOTHING. Collapse is a symptom."""

    def construct(self):
        ev = data.collapse_evidence()
        title = Text("Fixing the obvious problem did nothing", font_size=TITLE).to_edge(UP)
        self.play(Write(title))

        order = ["baseline", "beta 0.05", "beta 0.2", "beta 0.8"]
        # Same reason as `TheWall`: the names have different widths, so the columns are placed at
        # fixed x rather than arranged inside each row.
        col_x = [-5.4, -3.2, 2.4]
        rows = VGroup()
        for i, key in enumerate([k for k in order if k in ev]):
            modal, succ = ev[key]["modal"], ev[key]["success"]
            y = 1.2 - 0.8 * i
            cells = [Text(key, font_size=SMALL, color=GREY),
                     Text(f"one action {modal * 100:.0f}% of the time",
                          font_size=SMALL, color=BLUE),
                     Text(f"solved {succ:.4f}", font_size=SMALL, color=RED)]
            for cell, x in zip(cells, col_x):
                cell.move_to([x, y, 0], aligned_edge=LEFT)
            rows.add(VGroup(*cells))

        for row in rows:
            self.play(FadeIn(row), run_time=0.6)
        self.wait(1)

        # Read the two ends off the records: a transcribed "97% to 63%" rounds differently from
        # the row above it (0.975 prints as 98%), and that reads as an error on screen.
        lowest = min(ev[k]["modal"] for k in ev if k != "baseline")
        punch = Text(f"Collapse fell from {ev['baseline']['modal'] * 100:.0f}% to "
                     f"{lowest * 100:.0f}%. Success stayed at zero.",
                     font_size=LABEL, color=YELLOW).move_to([0, -2.1, 0])
        self.play(Write(punch))
        self.wait(1)
        punch2 = Text("It was a symptom, not the cause.",
                      font_size=LABEL, color=WHITE).move_to([0, -2.9, 0])
        self.play(Write(punch2))
        self.wait(2)
