__version__ = "0.1.0"

from ._21cmfast import *

from os import path, mkdir
import yaml

# Global Options
with open(path.expanduser(path.join("~", '.21CMMC', "config.yml"))) as f:
    config = yaml.load(f)

# This just ensures that the default directory for boxes is created.
try:
    mkdir(path.expanduser(config['boxdir']))
except:
    pass
