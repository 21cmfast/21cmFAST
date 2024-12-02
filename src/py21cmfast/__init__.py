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

# This just ensures that the default directory for boxes is created.
from os import mkdir as _mkdir
from os import path

from . import cache_tools, lightcones, plotting, wrapper
from ._cfg import config
from ._logging import configure_logging
from .cache_tools import query_cache
from .drivers.coeval import Coeval, run_coeval
from .drivers.lightcone import LightCone, exhaust_lightcone, run_lightcone
from .drivers.param_config import InputParameters, get_logspaced_redshifts
from .drivers.single_field import (
    brightness_temperature,
    compute_halo_grid,
    compute_initial_conditions,
    compute_ionization_field,
    compute_xray_source_field,
    determine_halo_list,
    perturb_field,
    perturb_halo_list,
    spin_temperature,
)
from .lightcones import AngularLightconer, RectilinearLightconer
from .run_templates import create_params_from_template
from .utils import get_all_fieldnames
from .wrapper.cfuncs import (
    compute_luminosity_function,
    compute_tau,
    construct_fftw_wisdoms,
)
from .wrapper.inputs import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    UserParams,
    global_params,
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

configure_logging()

try:
    _mkdir(path.expanduser(config["direc"]))
except FileExistsError:
    pass
