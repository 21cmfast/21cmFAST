from os import path

from . import yaml

# Global Options
with open(path.expanduser(path.join("~", ".21cmfast", "config.yml"))) as f:
    config = yaml.load(f)
