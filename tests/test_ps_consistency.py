"""Tests that expose the staleness problem with init_ps() and verify the fix.

Background
----------
``init_ps()`` initialises a set of static global constants (``cosmo_consts``) that are
used throughout the C backend.  Because the derived constants are stored as a static
struct they become *stale* if the cosmological input parameters are updated (e.g. via
``Broadcast_struct_global_all``) without calling ``init_ps()`` again.

The desired behaviour after the fix
------------------------------------
1. ``init_ps()`` caches the parameter values it was initialised with and returns early
    (no-op) when called again with the same parameters - avoiding redundant computation.
2. Functions that depend on the derived constants (``power_in_k``, ``power_in_vcb``,
   ``sigma_z0``) detect parameter changes and automatically re-initialise before
   returning so that users never silently obtain stale results.
3. High-level Python wrapper calls remain consistent: consecutive calls with different
   ``InputParameters`` always return results that correspond to those parameters.
4. ``ps_is_consistent()`` can be queried from Python to check whether the cached state
   matches the current global parameters.
5. ``invalidate_ps_cache()`` can be called from Python to force re-initialisation on
   the next ``init_ps()`` call.
"""

from __future__ import annotations

import numpy as np
import pytest
from py21cmfast.c_21cmfast import lib

from py21cmfast import CosmoParams, InputParameters
from py21cmfast.wrapper import cfuncs as cf

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
def inputs_cosmo1():
    """InputParameters with a specific cosmology (hlittle=0.678)."""
    return InputParameters(random_seed=11, cosmo_params=CosmoParams(hlittle=0.678))


@pytest.fixture
def inputs_cosmo2():
    """InputParameters with a distinctly different cosmology (hlittle=0.700)."""
    return InputParameters(random_seed=12, cosmo_params=CosmoParams(hlittle=0.700))


@pytest.fixture(autouse=True)
def reset_ps_state(inputs_cosmo1):
    """Ensure a known, clean PS state before every test in this module.

    Broadcasting a fixed cosmology and calling ``init_ps()`` before each test
    prevents state leaking from previous tests.  After the test we invalidate
    the cache so the next test also starts fresh.
    """
    _broadcast(inputs_cosmo1)
    lib.init_ps()
    yield
    # Invalidate so subsequent tests cannot accidentally reuse cached state.
    lib.invalidate_ps_cache()


# ---------------------------------------------------------------------------
# Baseline correctness tests (must pass before AND after the fix)
# ---------------------------------------------------------------------------


def test_different_cosmologies_give_different_power_spectra(
    inputs_cosmo1, inputs_cosmo2
):
    """Sanity check: two different cosmologies must produce different P(k)."""
    k = 0.1

    _broadcast(inputs_cosmo1)
    lib.init_ps()
    pk1 = lib.power_in_k(k)

    _broadcast(inputs_cosmo2)
    lib.init_ps()
    pk2 = lib.power_in_k(k)

    assert not np.isclose(pk1, pk2), (
        f"P(k={k}) should differ between cosmologies but got pk1={pk1}, pk2={pk2}"
    )


def test_init_ps_returns_consistent_values_on_repeat_call_same_params(inputs_cosmo1):
    """Calling init_ps() twice with identical parameters must give the same result."""
    k = 0.1

    _broadcast(inputs_cosmo1)
    lib.init_ps()
    pk_first = lib.power_in_k(k)

    # Second call - params unchanged, should be a no-op
    lib.init_ps()
    pk_second = lib.power_in_k(k)

    assert np.isclose(pk_first, pk_second), (
        "Repeated init_ps() calls with the same params must not change P(k): "
        f"first={pk_first}, second={pk_second}"
    )


def test_high_level_wrapper_gives_different_results_for_different_cosmo(
    inputs_cosmo1, inputs_cosmo2
):
    """The Python wrapper must return cosmology-consistent P(k) values."""
    k_values = [0.1, 0.5, 1.0]

    pk1 = cf.get_matter_power_values(inputs=inputs_cosmo1, k_values=k_values)
    pk2 = cf.get_matter_power_values(inputs=inputs_cosmo2, k_values=k_values)

    assert not np.allclose(pk1, pk2), (
        "High-level wrapper: P(k) must differ between cosmologies"
    )


def test_high_level_wrapper_is_reproducible(inputs_cosmo1):
    """The Python wrapper must give identical results on repeated calls with same inputs."""
    k_values = [0.1, 0.5, 1.0]

    pk1 = cf.get_matter_power_values(inputs=inputs_cosmo1, k_values=k_values)
    pk2 = cf.get_matter_power_values(inputs=inputs_cosmo1, k_values=k_values)

    assert np.allclose(pk1, pk2), "High-level wrapper: P(k) must be reproducible"


# ---------------------------------------------------------------------------
# Staleness tests: these FAIL before the fix and PASS after
# ---------------------------------------------------------------------------


def test_power_in_k_not_stale_after_broadcast_without_explicit_reinit(
    inputs_cosmo1, inputs_cosmo2
):
    """CORE BUG: power_in_k must not return stale values after a parameter change.

    Before the fix: changing parameters via Broadcast and then calling power_in_k
    without an explicit init_ps() call returns the *old* value (cosmo1).

    After the fix: power_in_k detects the stale state and re-initialises
    automatically, so it returns the *new* correct value (cosmo2).
    """
    k = 0.1

    # Establish a known baseline with cosmo1
    _broadcast(inputs_cosmo1)
    lib.init_ps()
    pk_cosmo1 = lib.power_in_k(k)

    # Change global parameters to cosmo2 - deliberately do NOT call init_ps()
    _broadcast(inputs_cosmo2)
    pk_stale_or_fresh = lib.power_in_k(k)

    # Obtain the authoritative fresh value for cosmo2
    lib.init_ps()
    pk_cosmo2_authoritative = lib.power_in_k(k)

    # The stale value must differ from cosmo1 (meaning: auto-reinit happened)
    assert not np.isclose(pk_stale_or_fresh, pk_cosmo1), (
        "power_in_k should NOT return the old cosmo1 value after params changed to "
        f"cosmo2.  Got {pk_stale_or_fresh}, expected something close to "
        f"{pk_cosmo2_authoritative} (not {pk_cosmo1})"
    )
    # And it must equal the authoritative fresh value
    assert np.isclose(pk_stale_or_fresh, pk_cosmo2_authoritative), (
        f"power_in_k returned {pk_stale_or_fresh} after param change but the "
        f"correct value is {pk_cosmo2_authoritative}"
    )


def test_sigma_z0_not_stale_after_broadcast_without_explicit_reinit(
    inputs_cosmo1, inputs_cosmo2
):
    """sigma_z0 must not return stale values after a parameter change.

    Analogous to test_power_in_k_not_stale_after_broadcast_without_explicit_reinit
    but for the sigma integral used in the halo mass function.
    """
    M = 1e10  # solar masses

    _broadcast(inputs_cosmo1)
    lib.init_ps()
    sigma_cosmo1 = lib.sigma_z0(M)

    # Change global parameters to cosmo2 without explicit reinit
    _broadcast(inputs_cosmo2)
    sigma_stale_or_fresh = lib.sigma_z0(M)

    lib.init_ps()
    sigma_cosmo2_authoritative = lib.sigma_z0(M)

    assert not np.isclose(sigma_cosmo1, sigma_cosmo2_authoritative), (
        "sigma_z0 must differ between cosmologies"
    )
    assert not np.isclose(sigma_stale_or_fresh, sigma_cosmo1), (
        "sigma_z0 should NOT return the old cosmo1 value after params changed.  "
        f"Got {sigma_stale_or_fresh}; old cosmo1 value was {sigma_cosmo1}"
    )
    assert np.isclose(sigma_stale_or_fresh, sigma_cosmo2_authoritative), (
        f"sigma_z0 returned {sigma_stale_or_fresh} after param change but the "
        f"correct value is {sigma_cosmo2_authoritative}"
    )


def test_ps_is_consistent_reflects_param_changes(inputs_cosmo1, inputs_cosmo2):
    """ps_is_consistent() must return False after a parameter change.

    This tests the observable Python-accessible flag exposed by the fix.
    """
    _broadcast(inputs_cosmo1)
    lib.init_ps()
    assert lib.ps_is_consistent(), (
        "ps_is_consistent() must be True immediately after init_ps() with current params"
    )

    # Change params without calling init_ps()
    _broadcast(inputs_cosmo2)
    assert not lib.ps_is_consistent(), (
        "ps_is_consistent() must return False after parameter change without reinit"
    )

    # Re-initialize: must become consistent again
    lib.init_ps()
    assert lib.ps_is_consistent(), (
        "ps_is_consistent() must be True after init_ps() is called again"
    )


def test_invalidate_ps_cache_forces_reinit(inputs_cosmo1):
    """invalidate_ps_cache() must mark the cache as invalid.

    After invalidation, ps_is_consistent() must return False and init_ps() must
    re-run the full initialisation.
    """
    _broadcast(inputs_cosmo1)
    lib.init_ps()
    assert lib.ps_is_consistent()

    lib.invalidate_ps_cache()
    assert not lib.ps_is_consistent(), (
        "ps_is_consistent() must be False after invalidate_ps_cache()"
    )

    # Calling init_ps() restores consistency
    lib.init_ps()
    assert lib.ps_is_consistent(), "ps_is_consistent() must be True after re-init"


def test_init_ps_is_noop_when_params_unchanged(inputs_cosmo1):
    """init_ps() with unchanged params must be idempotent (no detectable side-effects).

    Verifies that the caching logic does not alter the derived P(k) values.
    """
    k_values = np.array([0.01, 0.1, 1.0, 10.0])

    _broadcast(inputs_cosmo1)
    lib.init_ps()
    pk_before = np.vectorize(lib.power_in_k)(k_values)

    # Force several "redundant" init_ps calls
    for _ in range(3):
        lib.init_ps()

    pk_after = np.vectorize(lib.power_in_k)(k_values)

    np.testing.assert_allclose(
        pk_before,
        pk_after,
        rtol=0,
        atol=0,
        err_msg="Repeated init_ps() with unchanged params must not alter P(k)",
    )


# ---------------------------------------------------------------------------
# Python wrapper level: verify the fix propagates correctly
# ---------------------------------------------------------------------------


def test_high_level_wrapper_consistent_after_param_change(inputs_cosmo1, inputs_cosmo2):
    """High-level wrapper must give correct results when params change between calls.

    This verifies that the Python lifecycle management (broadcast + init_ps) is
    correctly hooked up to re-initialize the PS for every top-level call.
    """
    k_values = [0.1, 0.5, 1.0]

    pk1 = cf.get_matter_power_values(inputs=inputs_cosmo1, k_values=k_values)
    pk2 = cf.get_matter_power_values(inputs=inputs_cosmo2, k_values=k_values)
    pk1_again = cf.get_matter_power_values(inputs=inputs_cosmo1, k_values=k_values)

    assert not np.allclose(pk1, pk2), "P(k) must differ between cosmologies"
    np.testing.assert_allclose(
        pk1,
        pk1_again,
        rtol=1e-10,
        err_msg="Wrapper must give the same result when called again with cosmo1",
    )
