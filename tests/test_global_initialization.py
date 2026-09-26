"""Test the GlobalInitializationManager class."""

import pytest

from py21cmfast import InputParameters
from py21cmfast.drivers._global_initialization import (
    GlobalInitializationManager,
    _GlobalInitManagerSingleton,
    c_state,
    init_c_state,
)

N_REPEAT = 10


@pytest.fixture(autouse=True)
def _isolate_backend_state():
    """Give each test a freed backend and the singleton's original inputs.

    These tests drive the singleton directly, so without this they would leak both C
    allocations and parameter changes into one another and into the rest of the suite.
    """
    original_inputs = _GlobalInitManagerSingleton.inputs
    _GlobalInitManagerSingleton.free()
    yield
    _GlobalInitManagerSingleton.free()
    _GlobalInitManagerSingleton.inputs = original_inputs


def _count_frees(monkeypatch) -> list[int]:
    """Record a value in the returned list every time the backend is freed."""
    frees = []
    real_free = GlobalInitializationManager.free

    def counting_free(self):
        frees.append(1)
        real_free(self)

    monkeypatch.setattr(GlobalInitializationManager, "free", counting_free)
    return frees


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
    # NOTE: these inputs are built from a fresh InputParameters rather than from whatever
    # the singleton currently holds, so that repeated runs of this test don't accumulate
    # parameter changes into combinations that the input validators reject.
    # Let's give the initializer inputs that will prevent the initialization of sigma and recombination rate tables,
    # as well as the CLASS transfer function tables
    inputs_no_tables = InputParameters(random_seed=0).evolve_input_structs(
        SOURCE_MODEL="L-INTEGRAL",
        USE_UPPER_STELLAR_TURNOVER=False,
        USE_INTERPOLATION_TABLES="no-interpolation",
        RECOMB_MODEL="none",
    )
    _GlobalInitManagerSingleton.inputs = inputs_no_tables

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
    _GlobalInitManagerSingleton.inputs = (
        inputs_no_tables.with_logspaced_redshifts().evolve_input_structs(
            POWER_SPECTRUM="CLASS",
            V_CB_MODEL="FLUCTS",
            USE_MCGS=True,
            USE_TS_FLUCT=True,
            K_MAX_FOR_CLASS=1.0,
            USE_INTERPOLATION_TABLES="sigma-interpolation",
            RECOMB_MODEL="inhomogeneous",
            R_BUBBLE_MAX=50.0,
            M_TURN_STELLAR_FEEDBACK=5.0,
        )
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
    # Initialize everything we can, then check that free really does free it all
    _GlobalInitManagerSingleton.init(
        inputs=InputParameters(random_seed=0),
        broadcast_inputs=True,
        ps=True,
        sigma=True,
        heat=True,
    )
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

    # Now let's change the inputs to ones that will allow the initialization of the recombination
    # rate tables, and check that it works as expected
    _GlobalInitManagerSingleton.inputs = (
        InputParameters(random_seed=0)
        .with_logspaced_redshifts()
        .evolve_input_structs(RECOMB_MODEL="inhomogeneous", R_BUBBLE_MAX=50.0)
    )

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
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=50.0,
    )
    _GlobalInitManagerSingleton._initialize_recombination_rate()
    assert _GlobalInitManagerSingleton.inputs_are_broadcast
    assert not _GlobalInitManagerSingleton.ps_inited
    assert not _GlobalInitManagerSingleton.sigma_inited
    assert not _GlobalInitManagerSingleton.heat_inited
    assert _GlobalInitManagerSingleton.recomb_inited


def _two_differing_inputs() -> tuple[InputParameters, InputParameters]:
    """Build two input sets that differ in a parameter the backend cares about.

    ``USE_INTERPOLATION_TABLES`` is the parameter at issue in the bug these tests guard
    against; a source model free of discrete halos is needed to be allowed to vary it.
    """
    base = InputParameters(random_seed=0).evolve_input_structs(
        SOURCE_MODEL="L-INTEGRAL",
        USE_UPPER_STELLAR_TURNOVER=False,
    )
    return (
        base.evolve_input_structs(USE_INTERPOLATION_TABLES="sigma-interpolation"),
        base.evolve_input_structs(USE_INTERPOLATION_TABLES="no-interpolation"),
    )


def test_nested_call_with_other_inputs_restores_the_outer_inputs():
    """A nested call with its own inputs must not leave them broadcast to C.

    The outer function may go on to call the backend directly after the nested call
    returns, and those calls have to see the outer function's own inputs.
    """
    outer_inputs, inner_inputs = _two_differing_inputs()

    @init_c_state(broadcast_inputs=True)
    def inner(*, inputs):
        assert _GlobalInitManagerSingleton.inputs == inputs

    @init_c_state(broadcast_inputs=True)
    def outer(*, inputs):
        inner(inputs=inner_inputs)
        assert _GlobalInitManagerSingleton.inputs == inputs

    outer(inputs=outer_inputs)


def test_outer_inputs_restored_when_nested_call_raises():
    """The outer inputs must be restored even if the nested call fails."""
    outer_inputs, inner_inputs = _two_differing_inputs()

    @init_c_state(broadcast_inputs=True)
    def inner(*, inputs):
        raise RuntimeError("boom")

    @init_c_state(broadcast_inputs=True)
    def outer(*, inputs):
        with pytest.raises(RuntimeError, match="boom"):
            inner(inputs=inner_inputs)
        assert _GlobalInitManagerSingleton.inputs == inputs

    outer(inputs=outer_inputs)


def test_top_level_calls_keep_their_initializations():
    """Initializations must survive a top-level call, so that the next one can reuse them."""
    inputs, _ = _two_differing_inputs()

    @init_c_state(ps=True)
    def func(*, inputs):
        pass

    func(inputs=inputs)
    assert _GlobalInitManagerSingleton.ps_inited

    func(inputs=inputs)
    assert _GlobalInitManagerSingleton.ps_inited


def test_outermost_scope_leaves_its_state_in_place():
    """The outermost scope has nothing to return into, so it keeps what it set up.

    The state it would otherwise wind back to is itself just a leftover of whatever ran
    before it, and winding back would throw away tables that the next call would only
    have to build again.
    """
    inputs, _ = _two_differing_inputs()

    with c_state(inputs, ps=True):
        pass

    assert _GlobalInitManagerSingleton.inputs == inputs
    assert _GlobalInitManagerSingleton.ps_inited


@pytest.mark.parametrize("n_calls", [2, 5])
def test_scope_is_not_rebuilt_per_call(monkeypatch, n_calls):
    """Calls sharing a scope must set the backend up once, not once each.

    Restoring the outer inputs after every single nested call would be correct but
    ruinously slow, so the number of times the backend is torn down has to be
    independent of how many calls are made inside the scope.
    """
    outer_inputs, inner_inputs = _two_differing_inputs()

    @init_c_state(ps=True)
    def inner(*, inputs):
        pass

    @init_c_state(ps=True)
    def outer(*, inputs):
        with c_state(inner_inputs):
            for _ in range(n_calls):
                inner(inputs=inner_inputs)

    frees = _count_frees(monkeypatch)
    outer(inputs=outer_inputs)

    # One on entering the outer function, one on entering the scope, one on leaving it.
    assert len(frees) == 3
