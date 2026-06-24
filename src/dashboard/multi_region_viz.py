"""Multi-region batch dashboard (Stage-2 MVP).

Reads a saved NEURO·SCOPE JSONL trace and renders a single multi-panel
matplotlib figure: 5 spike rasters (one per region), an inter-region
communication heatmap, a grid-world state panel, and an auto-detected
reward/return curve.

Design: ``docs/superpowers/specs/2026-06-24-multi-region-dashboard-viz-design.md``

CLI::

    python -m dashboard.multi_region_viz <trace.jsonl> [--step N] [--out PATH] [--show]
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- NEURO·SCOPE theme (single source of styling truth) -----------------

PALETTE = {
    "sensory": "#3fd2ff",      # cyan
    "hippocampus": "#ad8bff",  # purple
    "prefrontal": "#ffd24a",   # gold
    "router": "#ff5a8a",       # pink
    "motor": "#46f0a0",        # green
}
REGION_ORDER = ["sensory", "hippocampus", "prefrontal", "router", "motor"]

BG = "#0a0a0c"      # figure background
PANEL = "#11161f"   # axes background
TEXT = "#e9edf6"    # primary text
MUTED = "#9aa3b6"   # secondary text / ticks
GRIDC = "#646d80"   # grid lines / edges
POS = "#46f0a0"     # positive reward
NEG = "#ffb37a"     # negative reward


@dataclass
class RewardSeries:
    """Auto-detected reward curve.

    Attributes:
        mode: ``"per_episode"`` (final return per episode, multi-episode trace)
            or ``"within_episode"`` (cumulative return per step, single episode).
        x_label: axis label for ``xs``.
        xs: x values (episode indices or step indices).
        ys: y values (returns).
    """

    mode: str
    x_label: str
    xs: list
    ys: list


# --- data ---------------------------------------------------------------

def load_trace(path):
    """Parse a two-part JSONL trace into ``(header, frames)``.

    Line 0 is the run header; subsequent non-blank lines are per-step frames.
    Blank lines are ignored.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    records = [json.loads(ln) for ln in lines if ln.strip()]
    if not records:
        raise ValueError(f"trace {path} is empty")
    return records[0], records[1:]


def summarize_reward(frames):
    """Auto-detect and build the reward curve as a :class:`RewardSeries`.

    Multi-episode trace → final ``return`` per episode. Single episode →
    cumulative ``return`` per step.
    """
    episodes = sorted({f["episode"] for f in frames})
    if len(episodes) > 1:
        final = {}
        for f in frames:  # frames are in order; last write per episode = final
            final[f["episode"]] = f["task"]["return"]
        return RewardSeries("per_episode", "Episode", list(episodes),
                            [final[e] for e in episodes])
    xs = [f["step"] for f in frames]
    ys = [f["task"]["return"] for f in frames]
    return RewardSeries("within_episode", "Step", xs, ys)


def resolve_step(n_frames, step):
    """Resolve/clamp a requested frame index into ``[0, n_frames - 1]``.

    ``None`` → last frame. Out-of-range values are clamped with a warning.
    """
    last = n_frames - 1
    if step is None:
        return last
    if step < 0 or step > last:
        clamped = max(0, min(step, last))
        warnings.warn(
            f"step {step} out of range [0, {last}]; clamped to {clamped}",
            stacklevel=2,
        )
        return clamped
    return step


# --- styling helpers ----------------------------------------------------

def _style_ax(ax, title=None, accent=MUTED):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(GRIDC)
    ax.tick_params(colors=MUTED, labelsize=8)
    if title is not None:
        ax.set_title(title, color=accent, fontsize=10, pad=6)


# --- panels -------------------------------------------------------------

def _raster(ax, spikes, region, label):
    """Spike raster for one region: x = time, y = neuron, tinted by region."""
    arr = np.asarray(spikes)
    color = PALETTE.get(region, TEXT)
    if arr.size:
        T, N = arr.shape
        ts, ns = np.nonzero(arr)
        ax.scatter(ts, ns, s=6, c=color, marker="s", linewidths=0)
        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-0.5, max(N - 0.5, 0.5))
    _style_ax(ax, accent=color)
    ax.set_ylabel(f"{label}\n(n={arr.shape[1] if arr.size else 0})",
                  color=color, fontsize=8, rotation=0, ha="right", va="center")
    ax.yaxis.set_label_coords(-0.04, 0.5)


def _comm_heatmap(ax, header, frame):
    """Inter-region communication: src×dst intensity for the real pathways."""
    order = [r["id"] for r in header["regions"]]
    idx = {rid: i for i, rid in enumerate(order)}
    n = len(order)
    mat = np.full((n, n), np.nan)
    gated = np.zeros((n, n), dtype=bool)
    for p in header["pathways"]:
        i, j = idx[p["src"]], idx[p["dst"]]
        mat[i, j] = frame["pathways"][p["id"]].get("intensity", 0.0)
        gated[i, j] = bool(p.get("gated", False))
    masked = np.ma.masked_invalid(mat)
    vmax = float(np.nanmax(mat)) if np.isfinite(mat).any() else 1.0
    vmax = max(vmax, 1e-6)
    from matplotlib import colormaps
    cmap = colormaps["magma"].with_extremes(bad=PANEL)
    ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    short = [rid[:4] for rid in order]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(short)
    ax.set_xlabel("→ dst", color=MUTED, fontsize=8)
    ax.set_ylabel("src", color=MUTED, fontsize=8)
    for i in range(n):
        for j in range(n):
            if np.isfinite(mat[i, j]):
                mark = "*" if gated[i, j] else ""
                # Luminance-aware: dark text on the bright (high-intensity) cells.
                txt_color = BG if (mat[i, j] / vmax) > 0.55 else TEXT
                ax.text(j, i, f"{mat[i, j]:.2f}{mark}", ha="center", va="center",
                        color=txt_color, fontsize=7, fontweight="bold")
    _style_ax(ax, title="inter-region comm  (* = gated)", accent=TEXT)


def _grid_world(ax, header, frame):
    """Grid-world state: agent position, goal, and current action."""
    gn = header["task"]["grid_n"]
    task = frame["task"]
    ax.set_xlim(-0.5, gn - 0.5); ax.set_ylim(-0.5, gn - 0.5)
    ax.set_xticks(range(gn)); ax.set_yticks(range(gn))
    ax.set_aspect("equal")
    ax.grid(True, color=GRIDC, alpha=0.4)
    ax.invert_yaxis()  # row 0 at top
    gx, gy = task["goal"]
    ax_x, ax_y = task["agent"]
    ax.scatter([gx], [gy], marker="*", s=420, c=PALETTE["prefrontal"],
               edgecolors=TEXT, linewidths=0.5, zorder=3, label="goal")
    ax.scatter([ax_x], [ax_y], marker="o", s=240, c=PALETTE["motor"],
               edgecolors=TEXT, linewidths=0.5, zorder=4, label="agent")
    _style_ax(ax, title=f"grid · action: {task['action_label']}",
              accent=PALETTE["motor"])


def _reward(ax, frames, step):
    """Auto-detected reward/return curve with a playhead at the chosen step."""
    series = summarize_reward(frames)
    line_color = POS if (series.ys and series.ys[-1] >= 0) else NEG
    ax.plot(series.xs, series.ys, color=line_color, lw=1.8)
    ax.fill_between(series.xs, series.ys, min(series.ys + [0.0]),
                    color=line_color, alpha=0.12)
    chosen = frames[step]
    playhead = chosen["episode"] if series.mode == "per_episode" else chosen["step"]
    ax.axvline(playhead, color=MUTED, ls="--", lw=1.0)
    ax.set_xlabel(series.x_label, color=MUTED, fontsize=8)
    ax.set_ylabel("Return", color=MUTED, fontsize=8)
    ax.grid(True, color=GRIDC, alpha=0.25)
    _style_ax(ax, title=f"reward · {series.mode}", accent=line_color)


# --- composition --------------------------------------------------------

def render_dashboard(header, frames, step):
    """Render the 8-panel dashboard for a single frame; return the Figure."""
    import matplotlib.pyplot as plt

    frame = frames[step]
    order = [r["id"] for r in header["regions"]]
    labels = {r["id"]: r.get("label", r["id"]) for r in header["regions"]}

    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor(BG)

    outer = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0],
                             left=0.10, right=0.97, top=0.90, bottom=0.07,
                             wspace=0.22)
    left = outer[0, 0].subgridspec(len(order), 1, hspace=0.30)
    right = outer[0, 1].subgridspec(3, 1, hspace=0.45)

    # Left column: one raster per region, in signal-flow order.
    raster_axes = []
    for i, rid in enumerate(order):
        ax = fig.add_subplot(left[i])
        _raster(ax, frame["field"][rid]["spikes"], rid, labels[rid])
        raster_axes.append(ax)
        if i < len(order) - 1:
            ax.set_xticklabels([])
    raster_axes[-1].set_xlabel("Time (T)", color=MUTED, fontsize=8)

    # Right column: comm heatmap / grid-world / reward curve.
    _comm_heatmap(fig.add_subplot(right[0]), header, frame)
    _grid_world(fig.add_subplot(right[1]), header, frame)
    _reward(fig.add_subplot(right[2]), frames, step)

    b = header["brain"]
    task = frame["task"]
    last = len(frames) - 1
    fig.suptitle(
        f"{b['id']}  ·  seed {b['seed']}  ·  cfg {b['config_hash']}  ·  "
        f"step {step}/{last}  ·  return {task['return']:.1f}",
        color=TEXT, fontsize=13, y=0.96,
    )
    return fig


def main(argv=None):
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Multi-region batch dashboard.")
    parser.add_argument("trace_path", help="path to a NEURO·SCOPE JSONL trace")
    parser.add_argument("--step", type=int, default=None,
                        help="env step / frame index to render (default: last)")
    parser.add_argument("--out", default=None,
                        help="output PNG path (default: outputs/<stem>_dashboard.png)")
    parser.add_argument("--show", action="store_true", help="also display the figure")
    args = parser.parse_args(argv)

    header, frames = load_trace(args.trace_path)
    step = resolve_step(len(frames), args.step)
    fig = render_dashboard(header, frames, step)

    out = Path(args.out) if args.out else Path("outputs") / f"{Path(args.trace_path).stem}_dashboard.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
    print(f"wrote {out}")

    if args.show:
        import matplotlib.pyplot as plt
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
