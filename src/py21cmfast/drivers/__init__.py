"""
Driving functions for computing 21cmFAST outputs.

In this sub-package, we have functions both for running the low-level outputs (e.g.
initial conditions, perturb fields, spin temperature, etc.) and for high-level outputs
such as coeval boxes and lightcones.
"""

from collections import deque
from typing import Generator

from .coeval import Coeval, run_coeval
from .single_field import (
    compute_initial_conditions,
    compute_ionization_field,
    compute_spin_temperature,
    perturb_field,
)


def exhaust(generator: Generator):
    """Exhaust a generator without keeping more than one return value in memory."""
    return deque(generator, maxlen=1)[0]
