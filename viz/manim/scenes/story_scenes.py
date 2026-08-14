"""Manim Community Edition scenes for the project's visual story.

Every number is pulled from `data.py`, which reads the committed experiment records. Nothing
here is hand-typed, so a scene cannot drift from the experiment that produced it.

DELIBERATELY CONSERVATIVE API USE: only `Text`, `Rectangle`, `Line`, `VGroup` and the basic
animations, all stable across ManimCE releases. No `MathTex`/`Tex`, so **no LaTeX install is
required**. No `BarChart`, whose signature has moved between versions.

    manim -ql scenes/story_scenes.py TheBreakPointMoves    # 480p, iterate anywhere
    manim -qh scenes/story_scenes.py TheBreakPointMoves    # 1080p, wants real cores

> [!warning] These have NOT been rendered yet
> Written before manim was installed, because the VPS was busy with the seed diagnosis. Expect
> to fix layout on first render - positioning is the part you cannot get right by reading.
"""

from __future__ import annotations

import sys
from pathlib import Path

from manim import (
    BLUE, GREEN, GREY, RED, WHITE, YELLOW,
    DOWN, LEFT, RIGHT, UP,
    Create, FadeIn, FadeOut, Line, Rectangle, Scene, Text, Transform, VGroup, Write,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data  # noqa: E402

TITLE = 40
LABEL = 24
SMALL = 20


def _bar(value: float, max_value: float, color, width: float = 0.8, height: float = 4.0):
    """A bar whose HEIGHT is proportional to value. Zero-height bars are illegal in manim,
    so a floor of 0.01 keeps a 0.0000 result visible as a hairline rather than vanishing -
    which matters, because 'exactly zero on all twelve seeds' is a real result here."""
    h = max(0.01, height * (value / max_value if max_value else 0))
    return Rectangle(width=width, height=h, color=color, fill_color=color, fill_opacity=0.85)


BASELINE_Y = -2.1          # where every bar stands, so the three reveals share one axis
CAPTION_Y = -3.35


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
        # Headroom above the tallest bar so its value label is not against the legend.
        max_v = max(curve[d]["after"] for d in depths) * 1.15

        arms = [("before", GREY, "frozen encoder"),
                ("pretrained", BLUE, "+ trained encoder"),
                ("after", GREEN, "+ depth-1 cap")]

        title = Text("Two levers. The same 390 parameters.", font_size=TITLE).to_edge(UP)
        self.play(Write(title))

        legend = VGroup()
        for _, color, label in arms:
            swatch = Rectangle(width=0.3, height=0.3, color=color,
                               fill_color=color, fill_opacity=0.85)
            legend.add(VGroup(swatch, Text(label, font_size=SMALL, color=color)
                              ).arrange(RIGHT, buff=0.2))
        legend.arrange(RIGHT, buff=0.9)
        legend.next_to(title, DOWN, buff=0.35)

        axis = Line(LEFT * 6.2, RIGHT * 6.2).shift(UP * BASELINE_Y).set_stroke(GREY, 2)
        self.play(Create(axis), run_time=0.5)

        # Bars first, positioned on the shared axis; labels attach afterwards so they follow the
        # final position rather than an intermediate one.
        groups, labels = VGroup(), []
        for d in depths:
            trio = VGroup(*[_bar(curve[d][key], max_v, color, height=3.0)
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
        honest = VGroup(
            Text(f"depth 6 works: {w6['seeds_above']} of {w6['n']} seeds above the bar, "
                 f"+{w6['se_margin']:.1f} SE.", font_size=SMALL, color=YELLOW),
            Text(f"depth 5's +{curve[5]['after'] - curve[5]['pretrained']:.2f} is larger and "
                 f"still misses p<=0.05 (p={curve[5]['after_p']:.3f}).",
                 font_size=SMALL, color=YELLOW),
        ).arrange(DOWN, buff=0.22).move_to(UP * (CAPTION_Y + 0.15))
        self.play(FadeIn(honest))
        self.wait(2.5)


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

        table = VGroup()
        header = VGroup(*[Text(f"d{d}", font_size=LABEL) for d in depths])
        header.arrange(RIGHT, buff=0.9)
        table.add(header)

        for label, series, color in rows:
            cells = VGroup(*[Text(f"{series[d]:.3f}", font_size=LABEL, color=color)
                             for d in depths])
            cells.arrange(RIGHT, buff=0.7)
            name = Text(label, font_size=SMALL, color=color).next_to(cells, LEFT, buff=0.8)
            table.add(VGroup(cells, name))

        table.arrange(DOWN, buff=0.55)
        table.shift(DOWN * 0.3)

        self.play(FadeIn(table[0]))
        for row in table[1:]:
            self.play(FadeIn(row), run_time=0.6)
        self.wait(1)

        punch = Text("A linear head cannot solve deep cubes, however good the encoder.",
                     font_size=LABEL, color=WHITE).next_to(table, DOWN, buff=0.7)
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
        bar_group.shift(UP * 0.5)
        self.play(Create(bar_group))

        # Name only the two that carry the argument: where we are, and where a real scramble is.
        here = Text("depths 1-3: 153 states (0.004%)", font_size=SMALL, color=GREEN)
        here.next_to(bar_group, DOWN, buff=0.8).to_edge(LEFT, buff=1.0)
        there = Text("depths 10-12: 3,063,976 (97.5%)", font_size=SMALL, color=RED)
        there.next_to(here, DOWN, buff=0.35).align_to(here, LEFT)
        self.play(FadeIn(here))
        self.wait(0.6)
        self.play(FadeIn(there))
        self.wait(1)

        punch = Text(f"A random scramble sits at depth {data.RANDOM_SCRAMBLE_DEPTH}.",
                     font_size=LABEL, color=WHITE).next_to(there, DOWN, buff=0.7)
        self.play(Write(punch))
        self.wait(1)

        honest = Text("We solve depths 3-6. That is 0.32% of the cube.",
                      font_size=SMALL, color=YELLOW).next_to(punch, DOWN, buff=0.4)
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
        rows = VGroup()
        for key in order:
            if key not in ev:
                continue
            modal, succ = ev[key]["modal"], ev[key]["success"]
            name = Text(key, font_size=SMALL, color=GREY)
            m = Text(f"one action {modal * 100:.0f}% of the time", font_size=SMALL, color=BLUE)
            s = Text(f"solved {succ:.4f}", font_size=SMALL, color=RED)
            row = VGroup(name, m, s).arrange(RIGHT, buff=0.8)
            rows.add(row)
        rows.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        rows.shift(DOWN * 0.2)

        for row in rows:
            self.play(FadeIn(row), run_time=0.6)
        self.wait(1)

        punch = Text("Collapse fell from 97% to 63%. Success stayed at zero.",
                     font_size=LABEL, color=YELLOW).next_to(rows, DOWN, buff=0.8)
        self.play(Write(punch))
        self.wait(1)
        punch2 = Text("It was a symptom, not the cause.",
                      font_size=LABEL, color=WHITE).next_to(punch, DOWN, buff=0.4)
        self.play(Write(punch2))
        self.wait(2)
