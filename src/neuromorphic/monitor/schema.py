"""Schema constants and the trace header for the dashboard data contract."""

from __future__ import annotations

import hashlib
import json

SCHEMA_VERSION = "1.1"

# region id -> the per-step recording key whose [T, B, N] tensor is the region's
# OUTPUT spike train (what the hero renders). Distinct from region.n_neurons.
REGION_OUTPUT_KEY = {
    "sensory": "concept",
    "hippocampus": "population",
    "prefrontal": "utility",
    "router": "gate",
    "motor": "action",
}


def render_for_n(n: int) -> str:
    """Hero representation hint as a function of output neuron count."""
    if n <= 2_000:
        return "dots"
    if n <= 100_000:
        return "cloud"
    return "density"


PATHWAYS = [
    {"id": "sens_hippo", "src": "sensory", "dst": "hippocampus", "gated": True, "label": "store/recall"},
    {"id": "sens_pfc", "src": "sensory", "dst": "prefrontal", "gated": False},
    {"id": "hippo_pfc", "src": "hippocampus", "dst": "prefrontal", "gated": True},
    {"id": "pfc_motor", "src": "prefrontal", "dst": "motor", "gated": True, "label": "router-gated"},
]


def region_specs(brain):
    """(id, label, output n_neurons, role) for each region, in signal-flow order.

    n_neurons is the region's OUTPUT width (what the hero renders), derived from
    brain config so it always matches the `field` tensor — not region.n_neurons.
    """
    return [
        ("sensory", "Sensory Cortex", brain.content, "input"),
        ("hippocampus", "Hippocampus", brain.hippo.n_neurons, "memory"),
        ("prefrontal", "Prefrontal", brain.n_actions, "planning"),
        ("router", "Thalamic Router", brain.n_actions, "control"),
        ("motor", "Motor Cortex", brain.n_actions, "output"),
    ]


def _config_hash(brain, seed: int, task_type: str) -> str:
    payload = {
        "content": brain.content,
        "n_actions": brain.n_actions,
        "n_hippo": brain.hippo.n_neurons,
        "T": brain.T,
        # n_obs is meaningful for every task; grid_n is meaningful for exactly one,
        # and a cube brain carries the Brain default of 5, which means nothing.
        "n_obs": brain.n_obs,
        "task": task_type,
        "seed": seed,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def build_header(brain, *, seed: int, adapter, policy_regions=None) -> dict:
    """Build the once-per-run trace header declaring brain topology + run context."""
    task = adapter.header_task()
    regions = [
        {"id": rid, "label": label, "n_neurons": n, "role": role, "render": render_for_n(n)}
        for rid, label, n, role in region_specs(brain)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "brain": {
            "id": "five-region",
            "config_hash": _config_hash(brain, seed, task["type"]),
            "seed": seed,
            "T": brain.T,
        },
        "task": task,
        "regions": regions,
        "pathways": [dict(p) for p in PATHWAYS],
        "policy_regions": list(policy_regions or []),
    }
