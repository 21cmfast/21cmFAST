__version__ = "3.0.0dev"

# This just ensures that the default directory for boxes is created.
from os import mkdir as _mkdir
from os import path

from ._cfg import config
from . import cache_tools
from ._logging import configure_logging
from .cache_tools import query_cache
from .wrapper import (
    CosmoParams,
    UserParams,
    AstroParams,
    FlagOptions,
    InitialConditions,
    PerturbedField,
    IonizedBox,
    TsBox,
    BrightnessTemp,
    compute_luminosity_function,
    compute_tau,
    initial_conditions,
    perturb_field,
    ionize_box,
    spin_temperature,
    brightness_temperature,
    run_coeval,
    run_lightcone,
    LightCone,
    global_params,
)


configure_logging()

try:
    _mkdir(path.expanduser(config["boxdir"]))
except FileExistsError:
    pass
