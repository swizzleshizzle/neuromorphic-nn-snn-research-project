"""Inter-region connections — sparse, delayed projections between regions.

See ``docs/architecture-spec-v1.md`` §3 (pathway table) and
``docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md``.
"""

from neuromorphic.connections.gating import apply_gain, apply_gate
from neuromorphic.connections.projection import Projection

__all__ = ["Projection", "apply_gate", "apply_gain"]
