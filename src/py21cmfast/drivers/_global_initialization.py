"""
A module that deals with global initialization for the 21cmfast simulation.

This correctly handles backend global state, such as the power spectrum tables
and interpolation tables.
"""

import atexit
import functools
import logging
from collections.abc import Callable

import attrs

from ..c_21cmfast import lib
from ..wrapper.inputs import InputParameters
from ._param_config import _OutputStructComputationInspect

logger = logging.getLogger(__name__)


@attrs.define
class GlobalInitializationManager:
    """Singleton that tracks the initialization states in the C backend."""

    inputs: InputParameters = attrs.field(default=InputParameters(random_seed=0))

    inputs_are_broadcast: bool = False
    ps_inited: bool = False
    sigma_inited: bool = False
    heat_inited: bool = False
    recomb_inited: bool = False

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
            lib.freeSigmaMInterpTable()
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
            self.broadcast_input_struct()
        if ps:
            self.initialize_power_spectrum()
        if sigma:
            self.initialize_sigma_tables()
        if heat:
            self.initialize_heat()
        if recomb:
            self.initialize_recombination_rate()

    def broadcast_input_struct(self):
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

    def initialize_power_spectrum(self):
        """Initialize power spectrum at the C backend."""
        if not self.inputs_are_broadcast:
            self.broadcast_input_struct()

        if not self.ps_inited:
            lib.init_ps()
            self.ps_inited = True

    def initialize_sigma_tables(self):
        """Initialize sigma interpolation tables at the C backend."""
        if not self.ps_inited:
            self.initialize_power_spectrum()

        if (
            self.inputs.matter_options.USE_INTERPOLATION_TABLES != "no-interpolation"
            and not self.sigma_inited
        ):
            sigma_min_mass = 5e2
            sigma_max_mass = 1e20
            lib.initialiseSigmaMInterpTable(sigma_min_mass, sigma_max_mass)
            self.sigma_inited = True

    def initialize_heat(self):
        """Initialize heat interpolation tables at the C backend."""
        if not self.inputs_are_broadcast:
            self.broadcast_input_struct()

        if not self.heat_inited:
            lib.init_heat()
            self.heat_inited = True

    def initialize_recombination_rate(self):
        """Initialize recombination rate interpolation tables at the C backend."""
        if not self.inputs_are_broadcast:
            self.broadcast_input_struct()

        # NOTE: run_global_evolution at the moment does not do recombination calculations, so we only initialize if HII_DIM > 1.
        # If this is changed in the future, it will be necessary to remove the HII_DIM > 1 condition below, since it would cause a segfault!
        if (
            self.inputs.astro_options.INHOMO_RECO
            and self.inputs.simulation_options.HII_DIM > 1
            and not self.recomb_inited
        ):
            lib.init_MHR()
            self.recomb_inited = True

    def __atexit__(self):
        """Free the global state when the program exits."""
        self.free()


def init_c_state(
    *,
    broadcast_inputs: bool = False,
    ps: bool = False,
    sigma: bool = False,
    heat: bool = False,
    recomb: bool = False,
) -> Callable:
    """Build a decorator that calls init_func before and free_func after the wrapped function."""

    def _make_wrapper(func):

        def setup(**kwargs):
            inputs = _OutputStructComputationInspect._get_inputs(kwargs)
            _GlobalInitManagerSingleton.init(
                inputs=inputs,
                broadcast_inputs=broadcast_inputs,
                ps=ps,
                sigma=sigma,
                heat=heat,
                recomb=recomb,
            )

        def wrapper(**kwargs):
            setup(**kwargs)
            return func(**kwargs)

        result = wrapper
        functools.update_wrapper(result, func)
        return result

    return _make_wrapper


# Instantiate the singleton for the global initialization manager (this happens at import time)
_GlobalInitManagerSingleton = GlobalInitializationManager()

# Register the atexit function. When python exits, this will free any tables that were allocated in C
atexit.register(_GlobalInitManagerSingleton.__atexit__)
