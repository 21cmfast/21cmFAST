"""Tests that expose the staleness problem with init_heat() and verify the fix.

Background
----------
``init_heat()`` loads a set of static interpolation tables (RECFAST, kappa tables,
spectral emissivity, optionally Lya heating) used throughout the C spin-temperature
backend.  The spectral emissivity table is scaled by ``astro_params_global->POP2_ION``
and ``astro_params_global->POP3_ION`` at load time, and the Lya heating table is only
loaded when ``astro_options_global->USE_LYA_HEATING`` is True.

If those global parameters are updated via ``Broadcast_struct_global_all`` without
re-calling ``init_heat()``, the loaded tables become stale.

The desired behaviour after the fix
------------------------------------
1. ``init_heat()`` caches the parameter values it was initialised with and returns
   early (no-op) when called again with the same parameters.
2. ``heat_is_consistent()`` can be queried from Python to check whether the cached
   state matches the current global parameters.
3. ``invalidate_heat_cache()`` forces a full re-initialisation on the next
   ``init_heat()`` call.
4. The Python wrapper ``initialize_heat()`` always calls ``lib.init_heat()``
   unconditionally so parameter changes are never silently ignored.
"""

from __future__ import annotations

import pytest
from py21cmfast.c_21cmfast import lib

from py21cmfast import AstroOptions, AstroParams, InputParameters
from py21cmfast.drivers._param_config import initialize_heat

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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def inputs_astro1():
    """InputParameters with default astro params (POP2_ION=5000)."""
    return InputParameters(
        random_seed=21,
        astro_params=AstroParams(POP2_ION=5000.0, POP3_ION=44021.0),
        astro_options=AstroOptions(USE_LYA_HEATING=True),
    )


@pytest.fixture
def inputs_astro2():
    """InputParameters with different astro params (POP2_ION=3000)."""
    return InputParameters(
        random_seed=22,
        astro_params=AstroParams(POP2_ION=3000.0, POP3_ION=20000.0),
        astro_options=AstroOptions(USE_LYA_HEATING=True),
    )


@pytest.fixture
def inputs_no_lya():
    """InputParameters with USE_LYA_HEATING=False."""
    return InputParameters(
        random_seed=23,
        astro_params=AstroParams(POP2_ION=5000.0, POP3_ION=44021.0),
        astro_options=AstroOptions(USE_LYA_HEATING=False),
    )


@pytest.fixture(autouse=True)
def reset_heat_state(inputs_astro1):
    """Ensure a known, clean heat state before every test in this module.

    Broadcasts a fixed set of parameters and calls ``init_heat()`` before each
    test.  After the test, the cache is invalidated so the next test starts fresh.
    """
    _broadcast(inputs_astro1)
    lib.init_heat()
    yield
    lib.invalidate_heat_cache()


# ---------------------------------------------------------------------------
# Baseline tests
# ---------------------------------------------------------------------------


def test_init_heat_succeeds(inputs_astro1):
    """init_heat() must return 0 (success) for valid parameters."""
    _broadcast(inputs_astro1)
    lib.invalidate_heat_cache()
    result = lib.init_heat()
    assert result == 0, f"init_heat() returned error code {result}"


def test_init_heat_is_noop_when_params_unchanged(inputs_astro1):
    """Calling init_heat() twice with identical parameters must be a no-op.

    The second call should hit the cache and return immediately without error.
    heat_is_consistent() must remain True after both calls.
    """
    _broadcast(inputs_astro1)
    lib.init_heat()
    assert lib.heat_is_consistent(), "Cache should be valid after first init_heat()"

    # Second call - params unchanged, should be a cache hit (no-op)
    result = lib.init_heat()
    assert result == 0, f"init_heat() returned error code {result} on repeat call"
    assert lib.heat_is_consistent(), (
        "Cache should still be valid after second init_heat()"
    )


def test_init_heat_with_lya_heating_disabled(inputs_no_lya):
    """init_heat() must succeed when USE_LYA_HEATING=False."""
    _broadcast(inputs_no_lya)
    lib.invalidate_heat_cache()
    result = lib.init_heat()
    assert result == 0, f"init_heat() with USE_LYA_HEATING=False returned {result}"
    assert lib.heat_is_consistent()


# ---------------------------------------------------------------------------
# heat_is_consistent() API tests
# ---------------------------------------------------------------------------


def test_heat_is_consistent_true_after_init(inputs_astro1):
    """heat_is_consistent() must return True immediately after init_heat()."""
    _broadcast(inputs_astro1)
    lib.invalidate_heat_cache()
    lib.init_heat()
    assert lib.heat_is_consistent(), (
        "heat_is_consistent() must be True after init_heat() with current params"
    )


def test_heat_is_consistent_false_after_param_change(inputs_astro1, inputs_astro2):
    """CORE BUG: heat_is_consistent() must detect parameter changes.

    Before the fix: changing parameters without calling init_heat() left the
    tables stale but heat_is_consistent() did not exist to detect this.

    After the fix: heat_is_consistent() correctly returns False after a
    parameter change, signalling that the tables are out of date.
    """
    _broadcast(inputs_astro1)
    lib.invalidate_heat_cache()
    lib.init_heat()
    assert lib.heat_is_consistent(), "Should be consistent after init with astro1"

    # Change global parameters without re-calling init_heat()
    _broadcast(inputs_astro2)
    assert not lib.heat_is_consistent(), (
        "heat_is_consistent() must return False after params changed from astro1 to "
        "astro2 without a re-init"
    )


def test_heat_is_consistent_true_after_reinit(inputs_astro1, inputs_astro2):
    """heat_is_consistent() must return True after re-calling init_heat() with new params."""
    _broadcast(inputs_astro2)
    lib.invalidate_heat_cache()
    lib.init_heat()
    assert lib.heat_is_consistent(), "Should be consistent after init with astro2"


def test_heat_is_consistent_detects_lya_heating_change(inputs_astro1, inputs_no_lya):
    """heat_is_consistent() must detect a change in USE_LYA_HEATING."""
    _broadcast(inputs_astro1)
    lib.invalidate_heat_cache()
    lib.init_heat()
    assert lib.heat_is_consistent()

    # Toggle USE_LYA_HEATING off without reinit
    _broadcast(inputs_no_lya)
    assert not lib.heat_is_consistent(), (
        "Changing USE_LYA_HEATING must be detected as inconsistent"
    )


# ---------------------------------------------------------------------------
# invalidate_heat_cache() API tests
# ---------------------------------------------------------------------------


def test_invalidate_heat_cache_forces_reinit(inputs_astro1):
    """invalidate_heat_cache() must cause the next init_heat() to fully re-init."""
    _broadcast(inputs_astro1)
    lib.init_heat()
    assert lib.heat_is_consistent(), "Should be consistent after first init"

    lib.invalidate_heat_cache()
    assert not lib.heat_is_consistent(), (
        "heat_is_consistent() must be False after invalidate_heat_cache()"
    )

    # Re-init should restore consistency
    result = lib.init_heat()
    assert result == 0
    assert lib.heat_is_consistent(), (
        "heat_is_consistent() must be True after init_heat() following invalidation"
    )


def test_destruct_heat_invalidates_cache(inputs_astro1):
    """destruct_heat() must invalidate the cache so the next init_heat() reinitialises."""
    _broadcast(inputs_astro1)
    lib.init_heat()
    assert lib.heat_is_consistent()

    lib.destruct_heat()
    assert not lib.heat_is_consistent(), (
        "Cache must be invalid after destruct_heat() to prevent stale state"
    )

    # Re-init after destruct must succeed
    result = lib.init_heat()
    assert result == 0
    assert lib.heat_is_consistent()


# ---------------------------------------------------------------------------
# Python wrapper tests
# ---------------------------------------------------------------------------


def test_initialize_heat_always_calls_init_heat_regardless_of_skip(inputs_astro2):
    """initialize_heat() must call lib.init_heat() even when skip=True.

    Because lib.init_heat() is now idempotent (fast no-op on cache hit), it is
    safe to always call it, and doing so ensures param changes are never missed.
    """
    _broadcast(inputs_astro2)
    lib.invalidate_heat_cache()

    # Call with skip=True - must still invoke lib.init_heat() unconditionally
    initialize_heat(inputs=inputs_astro2, skip=True)
    assert lib.heat_is_consistent(), (
        "initialize_heat(skip=True) must still call lib.init_heat() and update the cache"
    )


def test_initialize_heat_updates_cache_on_param_change(inputs_astro1, inputs_astro2):
    """initialize_heat() must produce a consistent cache even after a param change."""
    # Start with astro1 consistent
    _broadcast(inputs_astro1)
    lib.init_heat()
    assert lib.heat_is_consistent()

    # Simulate param change followed by wrapper call
    _broadcast(inputs_astro2)
    assert not lib.heat_is_consistent(), "Cache should be stale after param change"

    initialize_heat(inputs=inputs_astro2)
    assert lib.heat_is_consistent(), (
        "initialize_heat() must restore consistency after param change"
    )
