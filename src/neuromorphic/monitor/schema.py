"""Schema constants and the trace header for the dashboard data contract."""

from __future__ import annotations

SCHEMA_VERSION = "1.0"

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
