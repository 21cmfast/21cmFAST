"""
Driving functions for computing 21cmFAST outputs.

In this sub-package, we have functions both for running the low-level outputs (e.g.
initial conditions, perturb fields, spin temperature, etc.) and for high-level outputs
such as coeval boxes and lightcones.
"""

from collections import deque
from typing import Generator


def exhaust(generator: Generator):
    """Exhaust a generator without keeping more than one return value in memory."""
    return deque(generator, maxlen=1)[0]
