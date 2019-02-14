__version__ = "0.1.0"

from os import path, mkdir

from .mcmc import yaml

from ._21cmfast import *

# This just ensures that the default directory for boxes is created.
try:
    mkdir(path.expanduser(config['boxdir']))
except FileExistsError:
    pass
