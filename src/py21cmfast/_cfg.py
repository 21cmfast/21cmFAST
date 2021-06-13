"""Open and read the configuration file."""
import contextlib
import copy
import warnings
from pathlib import Path

from . import yaml


class ConfigurationError(Exception):
    """An error with the config file."""

    pass


class Config(dict):
    """Simple over-ride of dict that adds a context manager."""

    _defaults = {"direc": "~/21cmFAST-cache", "regenerate": False, "write": True}

    _aliases = {"direc": ("boxdir",)}

    def __init__(self, *args, write=True, file_name=None, **kwargs):

        super().__init__(*args, **kwargs)
        self.file_name = file_name

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
                                f"Your configuration file has old key '{alias}' which "
                                f"has been re-named '{k}'. Updating..."
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
                    f"The configuration file has key '{k}' which is not known to 21cmFAST."
                )

        self["direc"] = Path(self["direc"]).expanduser().absolute()

        if do_write and write and self.file_name:
            self.write()

    @contextlib.contextmanager
    def use(self, **kwargs):
        """Context manager for using certain configuration options for a set time."""
        backup = self.copy()
        for k, v in kwargs.items():
            self[k] = Path(v).expanduser().absolute() if k == "direc" else v
        yield self
        for k in kwargs:
            self[k] = backup[k]

    def write(self, fname: [str, Path, None] = None):
        """Write current configuration to file to make it permanent."""
        fname = Path(fname or self.file_name)
        if fname:
            if not fname.parent.exists():
                fname.parent.mkdir(parents=True)

            with open(fname, "w") as fl:
                yaml.dump(self._as_dict(), fl)

    def _as_dict(self):
        """The plain dict defining the instance."""
        return {k: str(Path) if isinstance(v, Path) else v for k, v in self.items()}

    @classmethod
    def load(cls, file_name: [str, Path]):
        """Create a Config object from a config file."""
        file_name = Path(file_name).expanduser().absolute()

        if file_name.exists():
            with open(file_name) as fl:
                cfg = yaml.load(fl)
            return cls(cfg, file_name=file_name)
        else:
            return cls(write=False)


config = Config.load(Path("~/.21cmfast/config.yml"))

# Keep an original copy around
default_config = copy.deepcopy(config)
