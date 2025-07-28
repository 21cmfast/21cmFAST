"""The py21cmfast package."""

from importlib.metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("21cmFAST")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

__all__ = [
    "DATA_PATH",
    "AngularLightconer",
    "AstroOptions",
    "AstroParams",
    "BrightnessTemp",
    "CacheConfig",
    "Coeval",
    "CosmoParams",
    "HaloBox",
    "HaloField",
    "InitialConditions",
    "InputParameters",
    "IonizedBox",
    "LightCone",
    "MatterOptions",
    "OutputCache",
    "PerturbHaloField",
    "PerturbedField",
    "RectilinearLightconer",
    "RunCache",
    "SimulationOptions",
    "TsBox",
    "XraySourceBox",
    "__version__",
    "brightness_temperature",
    "compute_halo_grid",
    "compute_initial_conditions",
    "compute_ionization_field",
    "compute_luminosity_function",
    "compute_rms",
    "compute_spin_temperature",
    "compute_tau",
    "compute_xray_source_field",
    "config",
    "configure_logging",
    "construct_fftw_wisdoms",
    "create_params_from_template",
    "determine_halo_list",
    "generate_coeval",
    "generate_lightcone",
    "get_all_fieldnames",
    "get_logspaced_redshifts",
    "lightconers",
    "list_templates",
    "perturb_field",
    "perturb_halo_list",
    "plotting",
    "run_classy",
    "run_coeval",
    "run_lightcone",
    "setup_photon_cons",
    "wrapper",
    "write_template",
]

from . import lightconers, plotting, wrapper
from ._cfg import config
from ._data import DATA_PATH
from ._logging import configure_logging
from ._templates import list_templates, write_template
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
from .lightconers import AngularLightconer, RectilinearLightconer
from .wrapper.cfuncs import (
    compute_luminosity_function,
    compute_tau,
    construct_fftw_wisdoms,
)
from .wrapper.classy_interface import compute_rms, run_classy
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

configure_logging()
