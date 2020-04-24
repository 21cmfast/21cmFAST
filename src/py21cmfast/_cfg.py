"""Open and read the configuration file."""
import contextlib
import copy
import warnings
from os import path

from . import yaml


class ConfigurationError(Exception):
    """An error with the config file."""

    pass


class Config(dict):
    """Simple over-ride of dict that adds a context manager."""

    _defaults = {"direc": "~/21cmFAST-cache", "regenerate": False, "write": True}

    _aliases = {"direc": ("boxdir",)}

    def __init__(self, *args, write=True, **kwargs):

        super().__init__(*args, **kwargs)

        # Ensure the keys that got read in are the right keys for the current version
        do_write = False
        for k, v in self._defaults.items():
            if k not in self:
                if k not in self._aliases:
                    warnings.warn("Your configuration file is out of date. Updating...")
                    do_write = True
                    self[k] = v

                else:
                    for alias in self._aliases[k]:
                        if alias in self:
                            do_write = True
                            warnings.warn(
                                "Your configuration file has old key '{}' which has been re-named '{}'. Updating...".format(
                                    alias, k
                                )
                            )
                            self[k] = self[alias]
                            del self[alias]
                            break
                    else:
                        warnings.warn(
                            "Your configuration file is out of date. Updating..."
                        )
                        do_write = True
                        self[k] = v

        for k, v in self.items():
            if k not in self._defaults:
                raise ConfigurationError(
                    f"The configuration file has key '{alias}' which is not known to 21cmFAST."
                )

        if do_write and write:
            self.write()

    @contextlib.contextmanager
    def use(self, **kwargs):
        """Context manager for using certain configuration options for a set time."""
        backup = self.copy()
        for k, v in kwargs.items():
            self[k] = v
        yield self
        for k in kwargs:
            self[k] = backup[k]

    def write(self, fname=None):
        """Write current configuration to file to make it permanent."""
        fname = fname or self.file_name
        with open(fname, "w") as fl:
            yaml.dump(self._as_dict(), fl)

    def _as_dict(self):
        """The plain dict defining the instance."""
        return {k: v for k, v in self.items()}

    @classmethod
    def load(cls, file_name):
        """Create a Config object from a config file."""
        cls.file_name = file_name
        if path.exists(file_name):
            with open(file_name, "r") as fl:
                config = yaml.load(fl)
                return cls(config)
        else:
            return cls(write=False)


config = Config.load(path.expanduser(path.join("~", ".21cmfast", "config.yml")))

# Keep an original copy around
default_config = copy.deepcopy(config)
