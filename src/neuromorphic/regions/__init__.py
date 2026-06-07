"""Brain regions — the standard ``BrainRegion`` interface and concrete regions.

See ``docs/architecture-spec-v1.md`` and
``docs/superpowers/specs/2026-06-06-phase2-region-framework-design.md``.
"""

from neuromorphic.regions.base_region import BrainRegion
from neuromorphic.regions.motor_cortex import MotorCortex
from neuromorphic.regions.sensory_cortex import SensoryCortex, encode_gridworld

__all__ = ["BrainRegion", "SensoryCortex", "encode_gridworld", "MotorCortex"]
