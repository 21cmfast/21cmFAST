"""The py21cmfast package."""
__version__ = "3.0.0"

# This just ensures that the default directory for boxes is created.
from os import mkdir as _mkdir
from os import path

from . import cache_tools
from . import inputs
from . import outputs
from . import plotting
from . import wrapper
from ._cfg import config
from ._logging import configure_logging
from .cache_tools import query_cache
from .outputs import Coeval
from .outputs import InitialConditions
from .outputs import IonizedBox
from .outputs import LightCone
from .outputs import PerturbedField
from .outputs import TsBox
from .wrapper import AstroParams
from .wrapper import BrightnessTemp
from .wrapper import CosmoParams
from .wrapper import FlagOptions
from .wrapper import HaloField
from .wrapper import PerturbHaloField
from .wrapper import UserParams
from .wrapper import brightness_temperature
from .wrapper import compute_luminosity_function
from .wrapper import compute_tau
from .wrapper import determine_halo_list
from .wrapper import get_all_fieldnames
from .wrapper import global_params
from .wrapper import initial_conditions
from .wrapper import ionize_box
from .wrapper import perturb_field
from .wrapper import perturb_halo_list
from .wrapper import run_coeval
from .wrapper import run_lightcone
from .wrapper import spin_temperature

configure_logging()

try:
    _mkdir(path.expanduser(config["direc"]))
except FileExistsError:
    pass
