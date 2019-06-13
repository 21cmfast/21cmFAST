__version__ = '0.1.0'

from . import cache_tools
from ._logging import configure_logging
from .cache_tools import query_cache
from .wrapper import *

configure_logging()

# This just ensures that the default directory for boxes is created.
from os import mkdir as _mkdir

try:
    _mkdir(path.expanduser(config['boxdir']))
except FileExistsError:
    pass
