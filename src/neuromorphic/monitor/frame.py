"""Build one JSON-serializable Frame from one ``Brain.step(record=True)`` output."""

from __future__ import annotations

import torch

from neuromorphic.monitor.schema import REGION_OUTPUT_KEY


def _field_tensor(out: dict, region: str) -> torch.Tensor:
    """The region's output spike train, squeezed to [T, N] (single-agent trace)."""
    key = REGION_OUTPUT_KEY[region]
    rec = out["recordings"][region][key]  # [T, B, N]
    return rec[:, 0, :].float()


def _region_summary(field: torch.Tensor) -> dict:
    """Scalar activity summary for one region from its [T, N] output spikes."""
    return {
        "rate": float(field.mean()),
        "spikes": int(field.sum()),
        "active_frac": float((field.sum(dim=0) > 0).float().mean()),
        "rate_t": field.mean(dim=1).tolist(),  # [T]
    }


def _router_block(out: dict) -> dict:
    gate_open = (1 - out["gate_closed"]).float()  # [T, B, A]
    utilities = out["utilities"].float()          # [T, B, A]
    return {
        "gate_open": gate_open.mean(dim=0)[0].tolist(),   # [A] open-fraction per action
        "gate_open_t": gate_open[:, 0, :].tolist(),       # [T, A]
        "utilities": utilities.mean(dim=0)[0].tolist(),   # [A] utility rate per action
    }


def _pathways(region_rate: dict, pfc_motor_open: list, store: bool, recall: bool) -> dict:
    return {
        "sens_hippo": {"intensity": region_rate["sensory"], "gate_open": 1.0 if store else 0.0},
        "sens_pfc": {"intensity": region_rate["sensory"]},
        "hippo_pfc": {"intensity": region_rate["hippocampus"], "gate_open": 1.0 if recall else 0.0},
        "pfc_motor": {"intensity": region_rate["prefrontal"], "gate_open": pfc_motor_open},
    }


def _encoding_block(out: dict, grid_n: int) -> dict:
    """Truthful sensory input: the encode_gridworld planes [T, 2*grid_n**2]."""
    obs_spk = out["obs_spikes"][:, 0, :]  # [T, n_obs]
    return {
        "sensory_input": {
            "spikes": obs_spk.int().tolist(),
            "grid_n": grid_n,
            "planes": ["agent", "goal"],   # first grid_n**2 = agent, second = goal
            "index": "y*grid_n + x",
        }
    }


def build_frame(
    out: dict,
    *,
    episode: int,
    step: int,
    t: float,
    task: dict,
    store: bool,
    recall: bool,
    grid_n: int | None = None,
) -> dict:
    """Assemble one Frame. ``out`` must come from ``Brain.step(record=True)``.

    ``grid_n`` enables the truthful ``encoding.sensory_input`` block (needs
    ``out["obs_spikes"]``); omit it to skip the block.
    """
    fields = {r: _field_tensor(out, r) for r in REGION_OUTPUT_KEY}
    regions = {r: _region_summary(f) for r, f in fields.items()}
    region_rate = {r: regions[r]["rate"] for r in regions}
    router = _router_block(out)
    frame = {
        "episode": episode,
        "step": step,
        "t": t,
        "task": task,
        "regions": regions,
        "pathways": _pathways(region_rate, router["gate_open"], store, recall),
        "router": router,
        # field = each region's RAW output train. Note router's raw train is the
        # gate_closed mask (1=blocked); router.gate_open above is its inversion.
        "field": {r: {"spikes": f.int().tolist()} for r, f in fields.items()},
    }
    if grid_n is not None and "obs_spikes" in out:
        frame["encoding"] = _encoding_block(out, grid_n)
    return frame
