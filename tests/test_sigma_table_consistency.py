"""Tests that expose stale-state issues in initialiseSigmaMInterpTable() and verify the fix.

Background
----------
``initialiseSigmaMInterpTable(M_min, M_max)`` fills static interpolation tables used by
``EvaluateSigma`` and ``EvaluatedSigmasqdm``. Those tables depend on:

1. The mass range (``M_min``, ``M_max``)
2. Global cosmological / power-spectrum parameters used by ``sigma_z0``

If the global parameters change without rebuilding the sigma table, the interpolation
results become stale.

The desired behaviour after the fix
-----------------------------------
1. ``initialiseSigmaMInterpTable`` is idempotent for unchanged parameters.
2. Parameter changes force a rebuild.
3. ``sigma_table_is_consistent`` exposes cache consistency to Python.
4. ``invalidate_sigma_table_cache`` forces the next call to rebuild.
5. Python wrapper initialization calls are safe with ``skip=True`` because C-side init
   is idempotent and consistency-aware.
"""

from __future__ import annotations

import numpy as np
import pytest
from py21cmfast.c_21cmfast import ffi, lib

from py21cmfast import CosmoParams, InputParameters
from py21cmfast.drivers._param_config import initialize_sigma_tables

SIGMA_MIN_MASS = 5e2
SIGMA_MAX_MASS = 1e20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _broadcast(inputs: InputParameters) -> None:
    """Low-level broadcast of all input structures to the C global pointers."""
    lib.Broadcast_struct_global_all(
        inputs.simulation_options.cstruct,
        inputs.matter_options.cstruct,
        inputs.cosmo_params.cstruct,
        inputs.astro_params.cstruct,
        inputs.astro_options.cstruct,
        inputs.cosmo_tables.cstruct,
    )


def _sigma_value(mass: float) -> float:
    """Evaluate sigma for one mass directly through the low-level C API."""
    masses = np.array([mass], dtype=np.float64)
    sigma = np.zeros(1, dtype=np.float64)
    dsigmasq = np.zeros(1, dtype=np.float64)

    lib.get_sigma(
        masses.size,
        ffi.cast("double *", ffi.from_buffer(masses)),
        ffi.cast("double *", ffi.from_buffer(sigma)),
        ffi.cast("double *", ffi.from_buffer(dsigmasq)),
    )
    return float(sigma[0])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def inputs_cosmo1():
    return InputParameters(random_seed=31, cosmo_params=CosmoParams(hlittle=0.678))


@pytest.fixture
def inputs_cosmo2():
    return InputParameters(random_seed=32, cosmo_params=CosmoParams(hlittle=0.700))


@pytest.fixture(autouse=True)
def reset_sigma_table_state(inputs_cosmo1):
    """Start each test from a known initialized sigma-table state."""
    _broadcast(inputs_cosmo1)
    lib.init_ps()
    lib.initialiseSigmaMInterpTable(SIGMA_MIN_MASS, SIGMA_MAX_MASS)
    yield
    lib.invalidate_sigma_table_cache()


# ---------------------------------------------------------------------------
# Cache API tests
# ---------------------------------------------------------------------------


def test_sigma_table_is_consistent_true_after_init(inputs_cosmo1):
    _broadcast(inputs_cosmo1)
    lib.init_ps()
    lib.invalidate_sigma_table_cache()
    lib.initialiseSigmaMInterpTable(SIGMA_MIN_MASS, SIGMA_MAX_MASS)

    assert lib.sigma_table_is_consistent(SIGMA_MIN_MASS, SIGMA_MAX_MASS)


def test_sigma_table_is_consistent_false_after_param_change(
    inputs_cosmo1, inputs_cosmo2
):
    _broadcast(inputs_cosmo1)
    lib.init_ps()
    lib.invalidate_sigma_table_cache()
    lib.initialiseSigmaMInterpTable(SIGMA_MIN_MASS, SIGMA_MAX_MASS)
    assert lib.sigma_table_is_consistent(SIGMA_MIN_MASS, SIGMA_MAX_MASS)

    _broadcast(inputs_cosmo2)
    assert not lib.sigma_table_is_consistent(SIGMA_MIN_MASS, SIGMA_MAX_MASS)


def test_sigma_table_is_consistent_false_after_mass_range_change(inputs_cosmo1):
    _broadcast(inputs_cosmo1)
    lib.init_ps()
    lib.invalidate_sigma_table_cache()
    lib.initialiseSigmaMInterpTable(SIGMA_MIN_MASS, SIGMA_MAX_MASS)

    assert not lib.sigma_table_is_consistent(1e3, SIGMA_MAX_MASS)


def test_invalidate_sigma_table_cache_forces_reinit(inputs_cosmo1):
    _broadcast(inputs_cosmo1)
    lib.init_ps()
    lib.initialiseSigmaMInterpTable(SIGMA_MIN_MASS, SIGMA_MAX_MASS)
    assert lib.sigma_table_is_consistent(SIGMA_MIN_MASS, SIGMA_MAX_MASS)

    lib.invalidate_sigma_table_cache()
    assert not lib.sigma_table_is_consistent(SIGMA_MIN_MASS, SIGMA_MAX_MASS)

    lib.initialiseSigmaMInterpTable(SIGMA_MIN_MASS, SIGMA_MAX_MASS)
    assert lib.sigma_table_is_consistent(SIGMA_MIN_MASS, SIGMA_MAX_MASS)


# ---------------------------------------------------------------------------
# Staleness regression tests
# ---------------------------------------------------------------------------


def test_sigma_values_not_stale_after_initialize_sigma_tables_skip_true(
    inputs_cosmo1, inputs_cosmo2
):
    """Wrapper path must reinitialise even when skip=True.

    Before the fix, initialize_sigma_tables(skip=True) would skip calling
    ``initialiseSigmaMInterpTable`` entirely, leaving stale interpolation tables
    after a parameter change.
    """
    mass = 1e10

    _broadcast(inputs_cosmo1)
    lib.init_ps()
    lib.invalidate_sigma_table_cache()
    lib.initialiseSigmaMInterpTable(SIGMA_MIN_MASS, SIGMA_MAX_MASS)
    sigma_cosmo1 = _sigma_value(mass)

    _broadcast(inputs_cosmo2)

    # Must still initialise despite skip=True.
    initialize_sigma_tables(inputs=inputs_cosmo2, skip=True)
    sigma_after_wrapper = _sigma_value(mass)

    # Authoritative value for cosmo2.
    lib.invalidate_sigma_table_cache()
    lib.initialiseSigmaMInterpTable(SIGMA_MIN_MASS, SIGMA_MAX_MASS)
    sigma_cosmo2_authoritative = _sigma_value(mass)

    assert not np.isclose(sigma_after_wrapper, sigma_cosmo1), (
        "Sigma table result is stale after parameter change: wrapper call with skip=True "
        "must not leave cosmo1 values active"
    )
    assert np.isclose(sigma_after_wrapper, sigma_cosmo2_authoritative), (
        "Sigma table result after wrapper re-init does not match authoritative cosmo2 value"
    )


def test_initialise_sigma_table_noop_when_unchanged(inputs_cosmo1):
    _broadcast(inputs_cosmo1)
    lib.init_ps()
    lib.invalidate_sigma_table_cache()

    lib.initialiseSigmaMInterpTable(SIGMA_MIN_MASS, SIGMA_MAX_MASS)
    assert lib.sigma_table_is_consistent(SIGMA_MIN_MASS, SIGMA_MAX_MASS)

    # Second call with same args/params should be a cache hit.
    lib.initialiseSigmaMInterpTable(SIGMA_MIN_MASS, SIGMA_MAX_MASS)
    assert lib.sigma_table_is_consistent(SIGMA_MIN_MASS, SIGMA_MAX_MASS)
