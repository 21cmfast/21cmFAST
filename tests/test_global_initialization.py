"""Test the GlobalInitializationManager class."""

import pytest

from py21cmfast import InputParameters
from py21cmfast.drivers._global_initialization import (
    GlobalInitializationManager,
    _GlobalInitManagerSingleton,
)

N_REPEAT = 10


def test_global_initialization_is_singleton():
    """Test that the GlobalInitializationManager is a singleton."""
    with pytest.raises(
        RuntimeError,
        match="GlobalInitializationManager is a singleton and has already been instantiated",
    ):
        GlobalInitializationManager()


def test_init():
    """Test that the init method work as expected."""
    # Let's call the init method with default input parameters (nothing should be initialized due to the built-in call to free)
    _GlobalInitManagerSingleton.init(inputs=InputParameters(random_seed=0))
    assert not _GlobalInitManagerSingleton.inputs_are_broadcast
    assert not _GlobalInitManagerSingleton.ps_inited
    assert not _GlobalInitManagerSingleton.sigma_inited
    assert not _GlobalInitManagerSingleton.heat_inited
    assert not _GlobalInitManagerSingleton.recomb_inited

    # Let's call init again, but with all flag set on True
    _GlobalInitManagerSingleton.init(
        inputs=InputParameters(random_seed=0),
        broadcast_inputs=True,
        ps=True,
        sigma=True,
        heat=True,
        recomb=True,
    )
    assert _GlobalInitManagerSingleton.inputs_are_broadcast
    assert _GlobalInitManagerSingleton.ps_inited
    assert _GlobalInitManagerSingleton.sigma_inited
    assert _GlobalInitManagerSingleton.heat_inited
    assert (
        not _GlobalInitManagerSingleton.recomb_inited
    )  # not initialized, because default is to have no recombinations


@pytest.mark.parametrize("_run", range(N_REPEAT))
def test_direct_initializations(_run):
    """
    Test that direct initializations work as expected.

    We run this test several times because segfaults can still occur unexpectedly, even if one test passes smoothly.

    NOTE: it is NOT a good idea to call directly these initialization functions, as they could lead to segfaults
    with uncautious usage!
    """
    # Ensure we start with a clean slate
    _GlobalInitManagerSingleton.free()

    # Let's give the initializer inputs that will prevent the initialization of sigma and recombination rate tables,
    # as well as the CLASS transfer function tables
    _GlobalInitManagerSingleton.inputs = (
        _GlobalInitManagerSingleton.inputs.evolve_input_structs(
            SOURCE_MODEL="L-INTEGRAL",
            USE_UPPER_STELLAR_TURNOVER=False,
            USE_INTERPOLATION_TABLES="no-interpolation",
            RECOMB_MODEL="none",
        )
    )

    _GlobalInitManagerSingleton._broadcast_input_struct()
    _GlobalInitManagerSingleton._initialize_power_spectrum()
    _GlobalInitManagerSingleton._initialize_sigma_tables()
    _GlobalInitManagerSingleton._initialize_heat()
    _GlobalInitManagerSingleton._initialize_recombination_rate()

    assert _GlobalInitManagerSingleton.inputs_are_broadcast
    assert _GlobalInitManagerSingleton.ps_inited
    assert not _GlobalInitManagerSingleton.sigma_inited
    assert _GlobalInitManagerSingleton.heat_inited
    assert not _GlobalInitManagerSingleton.recomb_inited

    # NOTE: before initializing again below with different inputs, it is very important to free everything that was initialized,
    # otherwise segfaults could occur! Note that these segfaults are not a problem with the user-facing logic of the code, but
    # rather due to our attempts of calling the private functions directly
    _GlobalInitManagerSingleton.free()

    # Now let's change the inputs to ones that will allow the initialization of all tables, and check that it works as expected
    _GlobalInitManagerSingleton.inputs = _GlobalInitManagerSingleton.inputs.with_logspaced_redshifts().evolve_input_structs(
        POWER_SPECTRUM="CLASS",
        V_CB_MODEL="FLUCTS",
        K_MAX_FOR_CLASS=1.0,
        USE_INTERPOLATION_TABLES="sigma-interpolation",
        RECOMB_MODEL="inhomogeneous",
    )

    _GlobalInitManagerSingleton._broadcast_input_struct()
    _GlobalInitManagerSingleton._initialize_power_spectrum()
    _GlobalInitManagerSingleton._initialize_sigma_tables()
    _GlobalInitManagerSingleton._initialize_heat()
    _GlobalInitManagerSingleton._initialize_recombination_rate()

    assert _GlobalInitManagerSingleton.inputs_are_broadcast
    assert _GlobalInitManagerSingleton.ps_inited
    assert _GlobalInitManagerSingleton.sigma_inited
    assert _GlobalInitManagerSingleton.heat_inited
    assert _GlobalInitManagerSingleton.recomb_inited


def test_free():
    """Test that the free method works as expected."""
    # After the above test, all tables should be initialized, so let's call free and check that they are indeed all freed
    _GlobalInitManagerSingleton.free()
    assert not _GlobalInitManagerSingleton.inputs_are_broadcast
    assert not _GlobalInitManagerSingleton.ps_inited
    assert not _GlobalInitManagerSingleton.sigma_inited
    assert not _GlobalInitManagerSingleton.heat_inited
    assert not _GlobalInitManagerSingleton.recomb_inited

    # Let's call free again, just to check that there aren't segfaults
    _GlobalInitManagerSingleton.free()
    assert not _GlobalInitManagerSingleton.inputs_are_broadcast
    assert not _GlobalInitManagerSingleton.ps_inited
    assert not _GlobalInitManagerSingleton.sigma_inited
    assert not _GlobalInitManagerSingleton.heat_inited
    assert not _GlobalInitManagerSingleton.recomb_inited


def test_direct_initializations_for_heat_and_recomb():
    """Test that direct initializations for heat and recombination rate work as expected."""
    # Ensure we start with a clean slate
    _GlobalInitManagerSingleton.free()

    # Let's begin with a direct initialization of the heating tables
    _GlobalInitManagerSingleton._initialize_heat()
    assert _GlobalInitManagerSingleton.inputs_are_broadcast
    assert not _GlobalInitManagerSingleton.ps_inited
    assert not _GlobalInitManagerSingleton.sigma_inited
    assert _GlobalInitManagerSingleton.heat_inited
    assert not _GlobalInitManagerSingleton.recomb_inited

    # Free again
    _GlobalInitManagerSingleton.free()

    # Now let's change the inputs to ones that will allow the initialization of the recombination rate, and check that it works as expected
    _GlobalInitManagerSingleton.inputs = _GlobalInitManagerSingleton.inputs.with_logspaced_redshifts().evolve_input_structs(
        RECOMB_MODEL="inhomogeneous"
    )
    _GlobalInitManagerSingleton._initialize_recombination_rate()
    assert _GlobalInitManagerSingleton.inputs_are_broadcast
    assert not _GlobalInitManagerSingleton.ps_inited
    assert not _GlobalInitManagerSingleton.sigma_inited
    assert not _GlobalInitManagerSingleton.heat_inited
    assert _GlobalInitManagerSingleton.recomb_inited
