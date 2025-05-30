"""The py21cmfast package."""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("21cmFAST")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

from . import lightcones, plotting, wrapper
from ._cfg import config
from ._data import DATA_PATH
from ._logging import configure_logging
from .drivers.coeval import Coeval, generate_coeval, run_coeval
from .drivers.lightcone import LightCone, generate_lightcone, run_lightcone
from .drivers.single_field import (
    brightness_temperature,
    compute_halo_grid,
    compute_initial_conditions,
    compute_ionization_field,
    compute_spin_temperature,
    compute_xray_source_field,
    determine_halo_list,
    perturb_field,
    perturb_halo_list,
)
from .io.caching import CacheConfig, OutputCache, RunCache
from .lightcones import AngularLightconer, RectilinearLightconer
from .run_templates import create_params_from_template
from .utils import get_all_fieldnames
from .wrapper.cfuncs import (
    compute_luminosity_function,
    compute_tau,
    construct_fftw_wisdoms,
)
from .wrapper.inputs import (
    AstroOptions,
    AstroParams,
    CosmoParams,
    InputParameters,
    MatterOptions,
    SimulationOptions,
    get_logspaced_redshifts,
)
from .wrapper.outputs import (
    BrightnessTemp,
    HaloBox,
    HaloField,
    InitialConditions,
    IonizedBox,
    PerturbedField,
    PerturbHaloField,
    TsBox,
    XraySourceBox,
)
from .wrapper.photoncons import setup_photon_cons

from .wrapper.CLASS import run_CLASS, compute_RMS

configure_logging()
