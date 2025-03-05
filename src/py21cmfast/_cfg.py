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


class ConfigurationError(Exception):
    """An error with the config file."""


# TODO: This has been moved for the sole purpose of avoiding circular imports, and should be moved back to structs.py
# if possible after the output struct overhaul
class StructInstanceWrapper:
    """A wrapper for *instances* of C structs.

    This is as opposed to :class:`StructWrapper`, which is for the un-instantiated structs.

    Parameters
    ----------
    wrapped :
        The reference to the C object to wrap (contained in the ``cffi.lib`` object).
    ffi :
        The ``cffi.ffi`` object.
    """

    def __init__(self, wrapped, ffi):
        self._cobj = wrapped
        self._ffi = ffi

        for nm, _tp in self._ffi.typeof(self._cobj).fields:
            setattr(self, nm, getattr(self._cobj, nm))

        # Get the name of the structure
        self._ctype = self._ffi.typeof(self._cobj).cname.split()[-1]

    def __setattr__(self, name, value):
        """Set an attribute of the instance, attempting to change it in the C struct as well."""
        with contextlib.suppress(AttributeError):
            setattr(self._cobj, name, value)
        object.__setattr__(self, name, value)

    def __iter__(self):
        yield from self.keys()

    def items(self):
        """Yield (name, value) pairs for each element of the struct."""
        for nm, _tp in self._ffi.typeof(self._cobj).fields:
            yield nm, getattr(self, nm)

    def keys(self):
        """Return a list of names of elements in the struct."""
        return [nm for nm, tp in self.items()]

    def __repr__(self):
        """Return a unique representation of the instance."""
        return (
            self._ctype + "(" + ";".join(f"{k}={v!s}" for k, v in sorted(self.items()))
        ) + ")"

    def filtered_repr(self, filter_params):
        """Get a fully unique representation of the instance that filters out some parameters.

        Parameters
        ----------
        filter_params : list of str
            The parameter names which should not appear in the representation.
        """
        return (
            self._ctype
            + "("
            + ";".join(
                f"{k}={v!s}" for k, v in sorted(self.items()) if k not in filter_params
            )
        ) + ")"


# TODO: force config to be a singleton
class Config(dict):
    """Simple over-ride of dict that adds a context manager."""

    _defaults: ClassVar = {
        "direc": "~/21cmFAST-cache",
        "cache_param_sigfigs": 6,
        "cache_redshift_sigfigs": 4,
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
