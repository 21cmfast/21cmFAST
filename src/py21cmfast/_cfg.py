"""Open and read the configuration file."""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Generator, Mapping
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    Self,
    TypedDict,
    Unpack,
    overload,
)

import deprecation
import yaml as pyyaml

from . import yaml
from ._data import _DATA_PATH
from .c_21cmfast import ffi, lib
from .wrapper.structs import StructInstanceWrapper


class ConfigurationError(Exception):
    """An error with the config file."""


type ConfigValue = str | Path | bool | float
"""The type of any value in the configuration."""

type _PathKey = Literal["direc", "external_table_path", "wisdoms_path"]
type _BoolKey = Literal[
    "ignore_R_BUBBLE_MAX_error",
    "EXTRA_EMISSIVITY_FIELDS",
    "safe_read",
    "CACHE_ARRAYS_ON_ACCESS",
]


class ConfigOptions(TypedDict, total=False):
    """The options that can be set in the configuration, with their types."""

    direc: str | Path
    ignore_R_BUBBLE_MAX_error: bool
    external_table_path: str | Path
    HALO_CATALOG_MEM_FACTOR: float
    EXTRA_EMISSIVITY_FIELDS: bool
    safe_read: bool
    CACHE_ARRAYS_ON_ACCESS: bool
    wisdoms_path: str | Path
    # Deprecated alias of EXTRA_EMISSIVITY_FIELDS.
    EXTRA_HALOBOX_FIELDS: bool


_DEFAULT_DIREC = "~/21cmFAST-cache"


# TODO: force config to be a singleton
class Config(dict[str, ConfigValue]):
    """Simple over-ride of dict that adds a context manager."""

    _defaults: ClassVar[dict[str, ConfigValue]] = {
        "direc": _DEFAULT_DIREC,
        "ignore_R_BUBBLE_MAX_error": False,
        "external_table_path": _DATA_PATH,
        "HALO_CATALOG_MEM_FACTOR": 1.2,
        "EXTRA_EMISSIVITY_FIELDS": False,
        "safe_read": True,
        "CACHE_ARRAYS_ON_ACCESS": True,
        "wisdoms_path": Path(_DEFAULT_DIREC) / "wisdoms",
    }

    _deprecated_keys: ClassVar[dict[str, str]] = {
        "EXTRA_HALOBOX_FIELDS": "EXTRA_EMISSIVITY_FIELDS",
    }

    # The C config_settings struct is process-global, but string values are stored in
    # char buffers allocated by cffi. Keep the most recent buffer for each key alive
    # here, so the struct never points at memory freed with a collected instance.
    # (The values are cffi ``char[]`` cdata objects.)
    _c_string_buffers: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *args: Mapping[str, ConfigValue],
        file_name: str | Path | None = None,
        **kwargs: Unpack[ConfigOptions],
    ) -> None:
        if args:
            args = (self._translate_deprecated(dict(args[0])), *args[1:])
        options = self._translate_deprecated({**kwargs})

        super().__init__(*args, **options)
        self.file_name = file_name
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
        for k in ("wisdoms_path", "external_table_path"):
            self[k] = Path(self[k])

        # since the subclass __setitem__ is not called in the super().__init__ call, we re-do the setting here
        # NOTE: This seems messy but I don't know a better way to do it
        for k in self._c_config_settings:
            self._pass_to_backend(k, self[k])

    if TYPE_CHECKING:
        # Precise value types for the known keys (the runtime method is dict's own).

        @overload
        def __getitem__(self, key: _PathKey) -> str | Path: ...
        @overload
        def __getitem__(self, key: _BoolKey) -> bool: ...
        @overload
        def __getitem__(self, key: Literal["HALO_CATALOG_MEM_FACTOR"]) -> float: ...
        @overload
        def __getitem__(self, key: str) -> ConfigValue: ...

    def __setitem__(self, key: str, value: ConfigValue) -> None:
        """Set an item in the config. Also updating the backend if it exists there."""
        key = self._resolve_deprecated_key(key)
        super().__setitem__(key, value)
        if key in self._c_config_settings:
            self._pass_to_backend(key, value)

    def _pass_to_backend(self, key: str, value: ConfigValue) -> None:
        """Set the value in the backend."""
        # we should possibly do a typemap for the ffi
        if isinstance(value, Path | str):
            buffer = ffi.new("char[]", str(value).encode())
            Config._c_string_buffers[key] = buffer
            setattr(self._c_config_settings, key, buffer)
        else:
            setattr(self._c_config_settings, key, value)

    @contextlib.contextmanager
    def use(self, **kwargs: Unpack[ConfigOptions]) -> Generator[Self, None, None]:
        """Context manager for using certain configuration options for a set time."""
        if "direc" in kwargs:
            kwargs["direc"] = Path(kwargs["direc"]).expanduser().absolute()
        options = self._translate_deprecated({**kwargs})
        backup = self.copy()
        for k, v in options.items():
            self[k] = v
        try:
            yield self
        finally:
            for k in options:
                self[k] = backup[k]

    def write(self, fname: str | Path | None = None) -> None:
        """Write current configuration to file to make it permanent."""
        fname = fname or self.file_name
        if fname is None:
            raise ValueError(
                "No file name given to write the config to, and the config has no file_name."
            )

        fname = Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)

        with fname.open("w") as fl:
            yaml.dump(self._as_dict(), fl)

    def _as_dict(self) -> dict[str, str | bool | float]:
        """Return a plain dict defining the instance."""
        return {k: str(v) if isinstance(v, Path) else v for k, v in self.items()}

    @classmethod
    def load(cls, file_name: str | Path) -> Self:
        """Create a Config object from a config file."""
        file_name = Path(file_name).expanduser().absolute()

        if file_name.exists():
            with file_name.open() as fl:
                # Config files only hold plain values, so never construct arbitrary
                # Python objects from tags in them.
                cfg = pyyaml.safe_load(fl) or {}
            return cls(cfg, file_name=file_name)
        else:
            return cls(file_name=file_name)

    @classmethod
    def _resolve_deprecated_key(cls, key: str) -> str:
        """Translate a deprecated config key to its current name, warning once."""
        new_key = cls._deprecated_keys.get(key)
        if new_key is None:
            return key
        warnings.warn(
            deprecation.DeprecatedWarning(
                key,
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details=f"'{key}' has been renamed to '{new_key}'. Please use '{new_key}' instead.",
            ),
            stacklevel=3,
        )
        return new_key

    @classmethod
    def _translate_deprecated(
        cls, d: Mapping[str, ConfigValue]
    ) -> dict[str, ConfigValue]:
        """Translate any deprecated keys in a dict of config options."""
        return {cls._resolve_deprecated_key(k): v for k, v in d.items()}


# On import, load the default config
config = Config()
"""A singleton Config object used to manage global configuration options."""
