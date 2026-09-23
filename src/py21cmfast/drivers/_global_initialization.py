"""
A module that deals with global initialization for the 21cmfast simulation.

This correctly handles backend global state, such as the power spectrum tables
and interpolation tables.

The backend keeps this state in C globals, of which there is a single set, so it can
correspond to only one :class:`~InputParameters` at a time. Functions declare the state
they need with :func:`init_c_state`, which sets it up on entry and hands back whatever
the enclosing scope had on exit. Code that deliberately computes with modified inputs
should hold them open with :func:`c_state`, so that the calls it makes share one scope.
"""

import atexit
import functools
import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

import attrs

from ..c_21cmfast import lib
from ..wrapper.inputs import InputParameters
from ._param_config import _OutputStructComputationInspect

logger = logging.getLogger(__name__)


@attrs.frozen
class _BackendState:
    """Which global tables are initialized in the C backend, and for which inputs."""

    inputs: InputParameters
    inputs_are_broadcast: bool
    ps_inited: bool
    sigma_inited: bool
    heat_inited: bool
    recomb_inited: bool


@attrs.define
class GlobalInitializationManager:
    """Singleton that tracks the initialization states in the C backend."""

    inputs: InputParameters = attrs.field(default=InputParameters(random_seed=0))

    inputs_are_broadcast: bool = False
    ps_inited: bool = False
    sigma_inited: bool = False
    heat_inited: bool = False
    recomb_inited: bool = False

    # The state as it was on entry to each currently-open scope, innermost last.
    _scopes: list[_BackendState] = attrs.field(factory=list, init=False, repr=False)

    def __new__(cls, *args, **kwargs):
        """Ensure this class is a singleton."""
        if not hasattr(cls, "exists"):
            out = super().__new__(cls)
            cls.exists = True
            return out
        else:
            raise RuntimeError(
                "GlobalInitializationManager is a singleton and has already been instantiated."
            )

    def free(self):
        """Free all global state in the C backend."""
        if self.recomb_inited:
            lib.free_MHR()
            self.recomb_inited = False
        if self.heat_inited:
            lib.destruct_heat()
            self.heat_inited = False
        if self.sigma_inited:
            lib.free_sigma_tables()
            self.sigma_inited = False
        if self.ps_inited:
            lib.free_ps()
            self.ps_inited = False
        if self.inputs_are_broadcast:
            lib.Free_cosmo_tables_global()
            self.inputs_are_broadcast = False

    def init(
        self,
        inputs: InputParameters,
        broadcast_inputs: bool = False,
        ps: bool = False,
        sigma: bool = False,
        heat: bool = False,
        recomb: bool = False,
    ):
        """Initialize the global state for a given set of inputs."""
        # First check that we're consistent with existing inputs, if any.
        # If not, free everything and start again. This ensures that we don't have
        # a mix of different inputs and that the global state is always consistent with
        # the inputs of the current run.
        if self.inputs is not None and self.inputs != inputs:
            # Free everything and start again.
            self.free()

            # Note that we ONLY reset the inputs in the case that they're not equal.
            # The backend relies on *pointers* to the underlying C structs, so even
            # if the new inputs is equal to the old, it will have a different memory
            # address. We don't want to use the new memory address for the backend.
            self.inputs = inputs

        if broadcast_inputs:
            self._broadcast_input_struct()
        if ps:
            self._initialize_power_spectrum()
        if sigma:
            self._initialize_sigma_tables()
        if heat:
            self._initialize_heat()
        if recomb:
            self._initialize_recombination_rate()

    @contextmanager
    def scope(
        self,
        inputs: InputParameters,
        broadcast_inputs: bool = False,
        ps: bool = False,
        sigma: bool = False,
        heat: bool = False,
        recomb: bool = False,
    ) -> Iterator[None]:
        """Initialize the global state for a set of inputs, and hand it back on exit."""
        self._scopes.append(self._snapshot())
        try:
            self.init(
                inputs,
                broadcast_inputs=broadcast_inputs,
                ps=ps,
                sigma=sigma,
                heat=heat,
                recomb=recomb,
            )
            yield
        finally:
            previous = self._scopes.pop()
            # Only hand the state back if we're returning into another scope. At the
            # top level nothing is relying on the old state, and keeping what we just
            # built lets consecutive calls with the same inputs reuse the tables.
            if self._scopes:
                self._restore(previous)

    def _snapshot(self) -> _BackendState:
        """Record the current state, so that it can be handed back later."""
        return _BackendState(
            inputs=self.inputs,
            inputs_are_broadcast=self.inputs_are_broadcast,
            ps_inited=self.ps_inited,
            sigma_inited=self.sigma_inited,
            heat_inited=self.heat_inited,
            recomb_inited=self.recomb_inited,
        )

    def _restore(self, state: _BackendState):
        """Return the backend to a previously recorded state."""
        # `init` rebinds self.inputs only when the values actually differ, so identity
        # is an exact -- and much cheaper -- test for whether the inputs have changed.
        if state.inputs is not self.inputs:
            self.free()
            self.inputs = state.inputs

        # Nothing was freed above if the inputs are unchanged, in which case this is a
        # no-op: tables that the inner scope added are kept rather than thrown away.
        self.init(
            state.inputs,
            broadcast_inputs=state.inputs_are_broadcast,
            ps=state.ps_inited,
            sigma=state.sigma_inited,
            heat=state.heat_inited,
            recomb=state.recomb_inited,
        )

    def _broadcast_input_struct(self):
        """Broadcast the parameters to the C library, and construct FFTW wisdoms if necessary."""
        if not self.inputs_are_broadcast:
            lib.Broadcast_struct_global_all(
                self.inputs.simulation_options._cstruct,
                self.inputs.matter_options._cstruct,
                self.inputs.cosmo_params._cstruct,
                self.inputs.astro_params._cstruct,
                self.inputs.astro_options._cstruct,
                self.inputs.cosmo_tables._cstruct,
            )
            if self.inputs.matter_options.USE_FFTW_WISDOM:
                lib.CreateFFTWWisdoms()

            self.inputs_are_broadcast = True

    def _initialize_power_spectrum(self):
        """Initialize power spectrum at the C backend."""
        if not self.inputs_are_broadcast:
            self._broadcast_input_struct()

        if not self.ps_inited:
            lib.init_ps()
            self.ps_inited = True

    def _initialize_sigma_tables(self):
        """Initialize sigma interpolation tables at the C backend."""
        if not self.ps_inited:
            self._initialize_power_spectrum()

        if (
            self.inputs.matter_options.USE_INTERPOLATION_TABLES != "no-interpolation"
            and not self.sigma_inited
        ):
            sigma_min_mass = 5e2
            sigma_max_mass = 1e20
            lib.initialize_sigma_tables(sigma_min_mass, sigma_max_mass)
            self.sigma_inited = True

    def _initialize_heat(self):
        """Initialize heat interpolation tables at the C backend."""
        if not self.inputs_are_broadcast:
            self._broadcast_input_struct()

        if not self.heat_inited:
            lib.init_heat()
            self.heat_inited = True

    def _initialize_recombination_rate(self):
        """Initialize recombination rate interpolation tables at the C backend."""
        if not self.inputs_are_broadcast:
            self._broadcast_input_struct()

        # NOTE: run_global_evolution at the moment does not do recombination calculations, so we only initialize if HII_DIM > 1.
        # If this is changed in the future, it will be necessary to remove the HII_DIM > 1 condition below, since it would cause a segfault!
        if (
            self.inputs.astro_options.RECOMB_MODEL != "none"
            and self.inputs.simulation_options.HII_DIM > 1
            and not self.recomb_inited
        ):
            lib.init_MHR()
            self.recomb_inited = True

    def __atexit__(self):
        """Free the global state when the program exits."""
        self.free()


def c_state(
    inputs: InputParameters,
    *,
    broadcast_inputs: bool = False,
    ps: bool = False,
    sigma: bool = False,
    heat: bool = False,
    recomb: bool = False,
) -> AbstractContextManager[None]:
    """Compute with the backend set up for ``inputs``, handing the state back on exit.

    Use this around a region that deliberately computes with inputs other than those of
    the calling function, so that the backend is not left set up for the wrong inputs.
    """
    return _GlobalInitManagerSingleton.scope(
        inputs,
        broadcast_inputs=broadcast_inputs,
        ps=ps,
        sigma=sigma,
        heat=heat,
        recomb=recomb,
    )


def init_c_state(
    *,
    broadcast_inputs: bool = False,
    ps: bool = False,
    sigma: bool = False,
    heat: bool = False,
    recomb: bool = False,
) -> Callable:
    """Build a decorator that sets up the backend state required by the wrapped function."""

    def _make_wrapper(func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            inputs = _OutputStructComputationInspect._get_inputs(kwargs)
            with c_state(
                inputs,
                broadcast_inputs=broadcast_inputs,
                ps=ps,
                sigma=sigma,
                heat=heat,
                recomb=recomb,
            ):
                return func(**kwargs)

        return wrapper

    return _make_wrapper


# Instantiate the singleton for the global initialization manager (this happens at import time)
_GlobalInitManagerSingleton = GlobalInitializationManager()

# Register the atexit function. When python exits, this will free any tables that were allocated in C
atexit.register(_GlobalInitManagerSingleton.__atexit__)
