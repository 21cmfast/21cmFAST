"""Open and read the configuration file."""

from __future__ import annotations

import contextlib
import copy
import warnings
from pathlib import Path
from typing import ClassVar

from . import yaml
from ._data import DATA_PATH
from .c_21cmfast import ffi, lib
from .wrapper.structs import StructInstanceWrapper


class ConfigurationError(Exception):
    """An error with the config file."""


# TODO: force config to be a singleton
class Config(dict):
    """Simple over-ride of dict that adds a context manager."""

    _defaults: ClassVar = {
        "direc": "~/21cmFAST-cache",
        "ignore_R_BUBBLE_MAX_error": False,
        "external_table_path": DATA_PATH,
        "HALO_CATALOG_MEM_FACTOR": 1.2,
    }
    _defaults["wisdoms_path"] = Path(_defaults["direc"]) / "wisdoms"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # keep the config settings from the C library here
        self._c_config_settings = StructInstanceWrapper(lib.config_settings, ffi)

        for k, v in self._defaults.items():
            if k not in self:
                self[k] = v

        for k in self.keys():
            if k not in self._defaults:
                raise ConfigurationError(
                    f"You passed the key '{k}' to config, which is not known to 21cmFAST."
                )

        self["direc"] = Path(self["direc"]).expanduser().absolute()

        # since the subclass __setitem__ is not called in the super().__init__ call, we re-do the setting here
        # NOTE: This seems messy but I don't know a better way to do it
        for k in self._c_config_settings:
            self._pass_to_backend(k, self[k])

    def __setitem__(self, key, value):
        """Set an item in the config. Also updating the backend if it exists there."""
        super().__setitem__(key, value)
        if key in self._c_config_settings:
            self._pass_to_backend(key, value)

    def _pass_to_backend(self, key, value):
        """Set the value in the backend."""
        # we should possibly do a typemap for the ffi
        if isinstance(value, Path | str):
            setattr(
                self._c_config_settings, key, ffi.new("char[]", str(value).encode())
            )
        else:
            setattr(self._c_config_settings, key, value)

    @contextlib.contextmanager
    def use(self, **kwargs):
        """Context manager for using certain configuration options for a set time."""
        backup = self.copy()
        for k, v in kwargs.items():
            self[k] = Path(v).expanduser().absolute() if k == "direc" else v
        yield self
        for k in kwargs:
            self[k] = backup[k]

    def write(self, fname: str | Path | None = None):
        """Write current configuration to file to make it permanent."""
        if fname := Path(fname or self.file_name):
            if not fname.parent.exists():
                fname.parent.mkdir(parents=True)

            with fname.open("w") as fl:
                yaml.dump(self._as_dict(), fl)

    def _as_dict(self):
        """Return a plain dict defining the instance."""
        return {k: str(Path) if isinstance(v, Path) else v for k, v in self.items()}

    @classmethod
    def load(cls, file_name: str | Path):
        """Create a Config object from a config file."""
        file_name = Path(file_name).expanduser().absolute()

        if file_name.exists():
            with file_name.open() as fl:
                cfg = yaml.load(fl)
            return cls(cfg, file_name=file_name)
        else:
            return cls(write=True)


# On import, load the default config
config = Config()
