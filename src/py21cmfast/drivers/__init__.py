"""
Driving functions for computing 21cmFAST outputs.

In this sub-package, we have functions both for running the low-level outputs (e.g.
initial conditions, perturb fields, spin temperature, etc.) and for high-level outputs
such as coeval boxes and lightcones.
"""

from collections import deque
from collections.abc import Generator

from lightcone import LightCone, generate_lightcone, run_lightcone

from .coeval import Coeval, generate_coeval, run_coeval
from .single_field import (
    compute_halo_grid,
    compute_initial_conditions,
    compute_ionization_field,
    compute_spin_temperature,
    compute_xray_source_field,
    perturb_field,
    perturb_halo_list,
)


def exhaust(generator: Generator):
    """Exhaust a generator without keeping more than one return value in memory."""
    return deque(generator, maxlen=1)[0]


__all__ = [
    "Coeval",
    "LightCone",
    "compute_halo_grid",
    "compute_initial_conditions",
    "compute_ionization_field",
    "compute_spin_temperature",
    "compute_xray_source_field",
    "exhaust",
    "generate_coeval",
    "generate_lightcone",
    "perturb_field",
    "perturb_halo_list",
    "run_coeval",
    "run_lightcone",
]
