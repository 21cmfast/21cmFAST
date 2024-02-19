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

from . import cache_tools, inputs, lightcones, outputs, plotting, wrapper
from ._cfg import config
from ._logging import configure_logging
from .cache_tools import query_cache
from .lightcones import AngularLightconer, RectilinearLightconer
from .outputs import (
    Coeval,
    InitialConditions,
    IonizedBox,
    LightCone,
    PerturbedField,
    TsBox,
)
from .wrapper import (
    AstroParams,
    BrightnessTemp,
    CosmoParams,
    FlagOptions,
    HaloField,
    PerturbHaloField,
    UserParams,
    brightness_temperature,
    compute_luminosity_function,
    compute_tau,
    construct_fftw_wisdoms,
    determine_halo_list,
    get_all_fieldnames,
    global_params,
    initial_conditions,
    ionize_box,
    perturb_field,
    perturb_halo_list,
    run_coeval,
    run_lightcone,
    spin_temperature,
)

configure_logging()

try:
    _mkdir(path.expanduser(config["direc"]))
except FileExistsError:
    pass
