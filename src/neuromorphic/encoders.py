"""Picklable encoder factories for Brain's encoder seam.

``Brain`` needs an encoder callable invoked as ``enc(obs_tensor, T=..., generator=...)``.
These factories return ``functools.partial`` objects over module-level functions, which
pickle cleanly and therefore survive ``ProcessPoolExecutor`` fan-out. A lambda would not
pickle, and the EXP-029 driver runs its grid across processes.
"""

from __future__ import annotations

from functools import partial

from neuromorphic.regions.sensory_cortex import encode_cube, encode_gridworld


def grid_encoder(grid_n: int = 5):
    """Grid observations ``[B, 4]`` -> Poisson spikes ``[T, B, 2*grid_n**2]``."""
    return partial(encode_gridworld, grid_n=grid_n)


def cube_encoder(cube_n: int = 2, n_colors: int = 6):
    """Cube facelets ``[B, 6*cube_n**2]`` -> Poisson spikes ``[T, B, 6*cube_n**2*n_colors]``."""
    return partial(encode_cube, cube_n=cube_n, n_colors=n_colors)
