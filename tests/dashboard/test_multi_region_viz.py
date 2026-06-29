"""Tests for the multi-region batch dashboard."""

from pathlib import Path

import pytest
from matplotlib.figure import Figure

from dashboard.multi_region_viz import (
    RewardSeries,
    load_trace,
    render_dashboard,
    resolve_step,
    summarize_reward,
)

REAL_TRACE = Path(__file__).resolve().parents[2] / "outputs" / "week11_dashboard_trace.jsonl"


# --- load_trace ---------------------------------------------------------

def test_load_trace_splits_header_and_frames(trace_file):
    header, frames = load_trace(trace_file)
    assert header["brain"]["id"] == "five-region"
    assert len(frames) == 5
    assert frames[0]["step"] == 0


def test_load_trace_skips_blank_lines(trace_file):
    # trace_file is written with a trailing blank line; it must not become a frame.
    _, frames = load_trace(trace_file)
    assert all("step" in f for f in frames)


# --- summarize_reward ---------------------------------------------------

def test_summarize_reward_within_episode_for_single_episode(single_episode_frames):
    series = summarize_reward(single_episode_frames)
    assert isinstance(series, RewardSeries)
    assert series.mode == "within_episode"
    assert len(series.xs) == 5
    assert len(series.ys) == 5
    # cumulative return: -1, -2, -3, -4, -5
    assert series.ys == [-1.0, -2.0, -3.0, -4.0, -5.0]


def test_summarize_reward_per_episode_for_multi_episode(multi_episode_frames):
    series = summarize_reward(multi_episode_frames)
    assert series.mode == "per_episode"
    assert series.xs == [0, 1, 2]
    # final return of each 2-step episode is -2.0
    assert series.ys == [-2.0, -2.0, -2.0]


# --- resolve_step -------------------------------------------------------

def test_resolve_step_defaults_to_last():
    assert resolve_step(5, None) == 4


def test_resolve_step_in_range_is_unchanged():
    assert resolve_step(5, 2) == 2


def test_resolve_step_clamps_high_out_of_range():
    with pytest.warns(UserWarning):
        assert resolve_step(5, 99) == 4


def test_resolve_step_clamps_negative():
    with pytest.warns(UserWarning):
        assert resolve_step(5, -3) == 0


# --- render_dashboard ---------------------------------------------------

def test_render_dashboard_returns_figure_with_panels(header, single_episode_frames):
    fig = render_dashboard(header, single_episode_frames, step=2)
    assert isinstance(fig, Figure)
    # 5 rasters + comm heatmap + grid-world + reward curve = 8 panels minimum.
    assert len(fig.axes) >= 8


def test_render_dashboard_does_not_raise_on_multi_episode(header, multi_episode_frames):
    fig = render_dashboard(header, multi_episode_frames, step=0)
    assert isinstance(fig, Figure)


@pytest.mark.skipif(not REAL_TRACE.exists(), reason="real trace artifact not present")
def test_render_dashboard_on_real_trace():
    header, frames = load_trace(REAL_TRACE)
    fig = render_dashboard(header, frames, step=len(frames) - 1)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 8


def test_region_tag_marks_policy_vs_spectator():
    from dashboard.multi_region_viz import region_tag
    assert region_tag("sensory", ["sensory"]) == "● on policy path"
    assert region_tag("motor", ["sensory"]) == "○ spectator (frozen)"
    # empty/missing policy_regions -> no claim, treat as spectator-unknown (spectator string)
    assert region_tag("sensory", []) == "○ spectator (frozen)"
