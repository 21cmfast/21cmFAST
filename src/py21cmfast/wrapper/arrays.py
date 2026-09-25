"""Module for dealing with arrays that are input/output to C functions."""

from __future__ import annotations

import itertools
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, overload

import attrs
import deprecation
import h5py
import numpy as np
from attrs.validators import instance_of, optional

from .arraystate import ArrayState

if TYPE_CHECKING:  # pragma: no cover
    from .outputs import OutputStruct

logger = logging.getLogger(__name__)


def _tuple_of_ints(x: Sequence[float | int]) -> tuple[int]:
    return tuple(int(i) for i in x)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def read(self) -> np.ndarray:
        """Read an Array from the cache."""

    @abstractmethod
    def write(self, val: np.ndarray) -> None:
        """Write an Array to the cache."""


@attrs.define(frozen=True)
class H5Backend(CacheBackend):
    """Backend for caching arrays in a HDF5 file."""

    path: Path = attrs.field(converter=Path)
    dataset: str = attrs.field(converter=str)

    def read(self) -> np.ndarray:
        """Read an array from the cache."""
        with h5py.File(self.path, "r") as f:
            return f[self.dataset][()]

    def write(self, val: np.ndarray, overwrite: bool = False) -> None:
        """Write an array to the cache."""
        if not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(self.path, "a") as f:
            if self.dataset in f:
                if overwrite:
                    f[self.dataset] = val
            else:
                f.create_dataset(self.dataset, data=val)


def _corner_elements(arr: np.ndarray) -> np.ndarray:
    shape = arr.shape
    dims = len(shape)

    # Get all combinations of 0 and -1 for each dimension
    corner_indices = list(itertools.product(*[(0, -1)] * dims))
    return np.array([arr[idx] for idx in corner_indices]).reshape((2,) * dims)


def _array_value_repr(x: np.ndarray | None) -> str:
    """Return a more compact representation of an array."""
    if x is None:
        return "None"
    elif x.size < 25:
        return str(x)
    else:
        corners = _corner_elements(x)

        return (
            f"Corners: {corners}\n"
            f"Min | Max | Mean: {x.min()} | {x.max()} | {x.mean()}\n"
            f"First elements: {x.flatten()[:6]}"
        )


@attrs.define(slots=False, frozen=True)
class Array:
    """
    A flexible array management class providing  state tracking and initialization capabilities.

    The Array class supports dynamic array creation, caching, and state management with
    immutable semantics.The class allows for creating arrays with configurable shape,
    data type, initialization function, and optional caching backend.
    It provides methods for initializing, setting values, removing values, writing to
    disk, and loading from disk while maintaining a consistent state.

    .. note:: One deliberate exception to this immutability: accessing a purged
              array's data directly (e.g. `arr.mean()`, `np.asarray(arr)`, or any
              other attribute not defined on this class) - rather than through
              `OutputStruct.get()` - transparently loads it from disk and, unless
              `config["CACHE_ARRAYS_ON_ACCESS"]` is set to False, caches the result
              by mutating `_value`, `state` and `cache_backend` on this instance in
              place (bypassing `frozen=True`). Any other reference to the same
              `Array` object will observe that mutation. All other methods on this
              class remain purely functional, returning a new instance rather than
              mutating `self`.

    Attributes
    ----------
    shape
        Dimensions of the array.
    dtype
        Data type of the array (default is float).
    state
        Current state of the array.
    initfunc
        Function used for array initialization (default is np.zeros).
    _value
        Actual array data, or `None` if it is not currently in memory (e.g. it has
        been purged to disk). Private: read the data off the `Array` itself, or via
        `OutputStruct.get()`, so that a purged array is transparently reloaded.
    cache_backend
        Optional backend for disk caching.

    Examples
    --------
    # Create an array with specific shape and initialize
    arr = Array(shape=(10, 10))
    initialized_arr = arr.initialize()

    # Set a value and write to disk
    arr = arr.with_value(np.random.rand(10, 10))
    arr = arr.written_to_disk(backend)

    """

    shape = attrs.field(converter=_tuple_of_ints)
    dtype = attrs.field(default=float, kw_only=True)
    state = attrs.field(factory=ArrayState, kw_only=True)
    initfunc = attrs.field(default=np.zeros, kw_only=True)
    _value = attrs.field(
        converter=attrs.converters.optional(np.asarray),
        default=None,
        kw_only=True,
        repr=_array_value_repr,
    )
    cache_backend = attrs.field(
        default=None, validator=optional(instance_of(CacheBackend)), kw_only=True
    )

    @_value.validator
    def _value_validator(self, att, val):
        if val is None:
            return

        if val.shape != self.shape:
            raise ValueError(f"Shape mismatch: expected {self.shape}, got {val.shape}")

        if val.dtype != self.dtype:
            raise ValueError(
                f"Data type mismatch: expected {self.dtype}, got {val.dtype}"
            )

    def initialize(self):
        """Initialize the array to its initial/default allocated state."""
        if self.state.initialized:
            return self
        else:
            return attrs.evolve(
                self,
                value=self.initfunc(self.shape, dtype=self.dtype),
                state=self.state.initialize(),
            )

    def with_value(self, val: np.ndarray) -> Self:
        """Set the array to a given value and return a new Array."""
        return attrs.evolve(
            self, value=val.astype(self.dtype, copy=False), state=self.state.computed()
        )

    def _with_value_not_computed(self, val: np.ndarray) -> Self:
        """Set the array to a given value and return a new Array, but without indicating the array has been computed."""
        return attrs.evolve(
            self,
            value=val.astype(self.dtype, copy=False),
        )

    def computed(self) -> Self:
        """Set the array to a given value and return a new Array."""
        return attrs.evolve(self, state=self.state.computed())

    def without_value(self) -> Self:
        """Remove the allocated data from the array."""
        return attrs.evolve(self, value=None, state=self.state.dropped())

    def written_to_disk(self, backend: CacheBackend | None) -> Self:
        """Write the array to disk and return a new object with correct state."""
        backend = backend or self.cache_backend

        if backend is None:
            raise ValueError("backend must be specified")

        backend.write(self._value)
        return attrs.evolve(self, cache_backend=backend, state=self.state.written())

    def purged_to_disk(self, backend: CacheBackend | None) -> Self:
        """Move the array data to disk and return a new object with correct state."""
        return self.written_to_disk(backend).without_value()

    def loaded_from_disk(self, backend: CacheBackend | None = None) -> Self:
        """Load values for the array from a cache backend, and return a new instance."""
        if self._value is not None:
            return attrs.evolve(self, cache_backend=backend)

        backend = backend or self.cache_backend

        if backend is None:
            raise ValueError("backend must be specified")

        logger.debug(
            "Reading array of shape %s and dtype %s from disk (%s)",
            self.shape,
            np.dtype(self.dtype).name,
            backend,
        )
        value = backend.read().astype(self.dtype, copy=False)
        return attrs.evolve(
            self,
            value=value,
            cache_backend=backend,
            state=self.state.loaded_from_disk(),
        )

    def trimmed(self, trimmed_shape: tuple[int]) -> Self:
        """Return a new Array with the same data but a different shape, by slicing the original array."""
        slc = tuple(slice(0, n) for n in trimmed_shape)
        trimmed_value = self._value[slc]
        return attrs.evolve(self, shape=trimmed_shape, value=trimmed_value)

    def _resolve_value(self) -> np.ndarray:
        """Return the array's value, transparently loading it from disk if needed.

        If the array was purged to disk, this loads it via `loaded_from_disk()`. If
        `config["CACHE_ARRAYS_ON_ACCESS"]` is True (the default), the loaded value
        (and state) is cached directly onto this instance - bypassing the
        `frozen=True` restriction, the same way every other state-transition method
        *would* if not for their functional, return-a-new-instance convention - so
        that repeated access doesn't keep re-reading from disk. Otherwise, this
        instance is left untouched, and every access re-reads from disk.
        """
        if self._value is not None:
            return self._value

        if not self.state.on_disk and not self.state.initialized:
            raise ValueError("Array is not on disk and not initialized.")

        from .._cfg import config

        loaded = self.loaded_from_disk()

        if config["CACHE_ARRAYS_ON_ACCESS"]:
            object.__setattr__(self, "_value", loaded._value)
            object.__setattr__(self, "state", loaded.state)
            object.__setattr__(self, "cache_backend", loaded.cache_backend)

        return loaded._value

    @property
    def value(self) -> np.ndarray:
        """Deprecated accessor for the array's data.

        `.value` used to be the raw in-memory slot, which meant it read as `None` for
        an array that had been purged to disk - a silent wrong answer, since the data
        was still perfectly available. It now resolves, loading from disk if needed.

        It is deprecated because it no longer has a job to do: an `OutputStruct`'s
        field attribute (or `OutputStruct.get()`) already gives you the data as a
        plain numpy array, so `ic.hires_density` is what you want. This class is the
        *manager* for that data, reached via `OutputStruct.arrays[...]`, and is only
        of interest when you care about where the data lives rather than what it is.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "value",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="Array.value is deprecated and will be removed in a future "
                "version. Read the field off the OutputStruct instead (e.g. "
                "`ic.hires_density`, or `ic.get('hires_density')`), which gives you "
                "the data as a plain numpy array.",
            ),
            stacklevel=2,
        )
        return self._resolve_value()

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Support `np.asarray(array)` and other numpy-protocol consumers."""
        # Some numpy versions' np.asarray() don't accept `copy=None`.
        if copy is None:
            return np.asarray(self._resolve_value(), dtype=dtype)
        return np.asarray(self._resolve_value(), dtype=dtype, copy=copy)

    def __getattr__(self, name: str):
        """Delegate unknown attributes (e.g. `.mean()`, `.sum()`) to the value.

        Leading-underscore names are never delegated, and a `ValueError` raised
        while resolving an unresolvable (never computed) value is translated to
        `AttributeError` - so that `hasattr()`, `copy.deepcopy()`, and pickling
        (which all probe for attributes, including dunder methods, via `getattr`)
        behave normally on an `Array` that hasn't been computed yet, instead of
        raising a confusing `ValueError`. Other failures (e.g. the cache backend
        itself raising `OSError` while reading a purged array) propagate unchanged.
        """
        if name.startswith("_"):
            raise AttributeError(name)

        try:
            value = self._resolve_value()
        except ValueError as e:
            raise AttributeError(str(e)) from e

        return getattr(value, name)


# ======================================================================================
# Exposing arrays on OutputStructs as plain numpy arrays.
# ======================================================================================

#: Attributes and methods that live on `Array` but not on `np.ndarray`. Accessing one
#: of these on the numpy array returned by an `OutputStruct` attribute is always code
#: written against the old API, where that attribute *was* an `Array`.
_ARRAY_ONLY_NAMES = frozenset(
    {
        "value",
        "state",
        "cache_backend",
        "initfunc",
        "initialize",
        "with_value",
        "computed",
        "without_value",
        "written_to_disk",
        "purged_to_disk",
        "loaded_from_disk",
        "trimmed",
    }
)


class _LegacyArrayView(np.ndarray):
    """A plain numpy array that still answers `Array`-only attributes, with a warning.

    `OutputStruct` array attributes used to give you an `Array`; they now give you the
    data itself. This transitional view exists purely so that code written against the
    old API keeps working for one release: it *is* an `np.ndarray` in every respect
    (including `isinstance`), but attributes that only ever existed on `Array` -
    `.value`, `.state`, `.purged_to_disk()` and friends - are forwarded to the
    underlying `Array` after emitting a `DeprecatedWarning`.

    Remove this class, and `_as_legacy_view`, when `Array.value` is removed in v5.
    """

    #: The `Array` this view was resolved from. Deliberately *not* propagated to
    #: derived arrays (see `__array_finalize__`), so only the object handed back by
    #: `ArrayProxy.__get__` answers the legacy API.
    _array: Array | None = None

    def __array_finalize__(self, obj):
        """Deliberately do not carry the originating `Array` into derived arrays.

        Slices, views and computed results are new data, not the struct's array, so
        they must behave as plain numpy arrays. With `_array` left as None,
        `__getattr__` raises `AttributeError` for the legacy names exactly as an
        `np.ndarray` would - which also keeps `hasattr()` probes (astropy does a few)
        answering the same way they would for a real array.

        Nothing is lost for backwards compatibility: the old `Array` defined no
        operators or `__getitem__`, so `ic.hires_density[0].value` never worked.
        """

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Compute on plain arrays, so this transitional type never leaks into results."""
        inputs = tuple(
            np.asarray(i) if type(i) is _LegacyArrayView else i for i in inputs
        )
        if (out := kwargs.get("out")) is not None:
            kwargs["out"] = tuple(
                np.asarray(o) if type(o) is _LegacyArrayView else o for o in out
            )
        return getattr(ufunc, method)(*inputs, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Forward `Array`-only attributes to the originating `Array`, with a warning.

        Only called when normal `np.ndarray` attribute lookup has already failed, so
        genuine array attributes (`.shape`, `.dtype`, `.mean`, ...) never reach here.
        """
        if name not in _ARRAY_ONLY_NAMES or self._array is None:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        warnings.warn(
            deprecation.DeprecatedWarning(
                name,
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details=f"Accessing '{name}' here is deprecated: this attribute is "
                "now a plain numpy array rather than an Array. Use the array "
                "directly, or reach for the Array itself via "
                "`output_struct.arrays['<name>']` if you need to manage its memory.",
            ),
            stacklevel=2,
        )

        if name == "value":
            # It *is* the value; don't round-trip through Array.value and warn twice.
            return self.view(np.ndarray)

        return getattr(self._array, name)


def _as_legacy_view(value: np.ndarray, array: Array) -> _LegacyArrayView:
    """Wrap resolved data in the transitional view, without copying it."""
    view = value.view(_LegacyArrayView)
    view._array = array
    return view


class ArrayProxy:
    """Exposes an `OutputStruct`'s `Array` field as a plain numpy array.

    The `Array` itself is stored in a private attrs field (`_<name>`) and remains
    reachable via `OutputStruct.arrays[<name>]`, which is where the memory-management
    API lives. Reading the public attribute gives you the data, loading it from disk
    first if it has been purged.

    Whether that load is *kept* is governed by `config["CACHE_ARRAYS_ON_ACCESS"]`: by
    default the loaded data is cached back onto the struct, so repeated access is
    cheap; with it disabled, each access re-reads from disk and nothing is retained.
    Either way the read is logged at DEBUG level, so unexpected memory growth or
    repeated disk traffic is traceable.
    """

    def __init__(self, name: str):
        self.public = name
        self.private = f"_{name}"

    def __set_name__(self, owner: type, name: str) -> None:
        """Support declaring the proxy directly in a class body as well."""
        self.public = name
        self.private = f"_{name}"

    @overload
    def __get__(self, obj: None, owner: type | None = ...) -> ArrayProxy: ...

    @overload
    def __get__(
        self, obj: OutputStruct, owner: type | None = ...
    ) -> np.ndarray | None: ...

    def __get__(self, obj, owner=None):
        """Return the array's data, loading it from disk if it has been purged."""
        if obj is None:
            # Class-level access, e.g. `InitialConditions.hires_density`. Hand back the
            # descriptor so that introspection and `attrs`-style tooling still work.
            return self

        array = getattr(obj, self.private)
        if array is None:
            # An optional field that this configuration doesn't produce.
            return None

        from .._cfg import config

        if (
            not config["CACHE_ARRAYS_ON_ACCESS"]
            and array.state.on_disk
            and not array.state.computed_in_mem
        ):
            # Resolve without retaining: don't put it back on the struct.
            return _as_legacy_view(array.loaded_from_disk()._value, array)

        return _as_legacy_view(obj.get(self.public), array)

    def __set__(self, obj, value: np.ndarray | Array | None) -> None:
        """Set the array's data, or replace the underlying `Array` wholesale.

        Assigning an `Array` (or `None`) replaces the field directly - this is how the
        memory-management machinery moves an array between states. Assigning anything
        array-like sets the *data* of the existing `Array`, which is what a user
        writing `ic.hires_density = my_box` means.
        """
        if value is None or isinstance(value, Array):
            setattr(obj, self.private, value)
            return

        current = getattr(obj, self.private)
        if current is None:
            raise AttributeError(
                f"Cannot set data for '{self.public}': it does not exist for this "
                "set of inputs."
            )
        setattr(obj, self.private, current.with_value(np.asarray(value)))


def expose_arrays(cls: type) -> type:
    """Class decorator exposing every private `Array` field as a plain numpy array.

    Must be applied *outside* `attrs.define`, so that it runs once `attrs` has finished
    building the class::

        @expose_arrays
        @attrs.define(slots=False, kw_only=True)
        class InitialConditions(OutputStruct):
            _hires_density = _arrayfield()

    Keeping the `Array` in the private field means `attrs`' generated `__repr__`, `eq`
    and `asdict` continue to see it, so inspecting a purged struct still never touches
    the disk. The public name is purely this descriptor. `attrs` strips the leading
    underscore for `__init__`, so `InitialConditions(hires_density=...)` is unchanged.
    """
    names = []
    for field in attrs.fields(cls):
        if field.type is not Array:
            continue
        if not field.name.startswith("_"):
            raise TypeError(
                f"{cls.__name__}.{field.name} is an Array field and must be declared "
                f"privately, as '_{field.name}', so that it can be exposed as a "
                "numpy array."
            )
        public = field.name[1:]
        setattr(cls, public, ArrayProxy(public))
        names.append(public)

    cls._array_field_names = tuple(names)
    return cls
