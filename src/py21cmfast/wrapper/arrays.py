"""Module for dealing with arrays that are input/output to C functions."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import attrs
import h5py
import numpy as np
from attrs.validators import instance_of, optional

from .arraystate import ArrayState


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


@attrs.define(slots=False, frozen=True)
class Array:
    """
    A flexible array management class providing  state tracking and initialization capabilities.

    The Array class supports dynamic array creation, caching, and state management with
    immutable semantics.The class allows for creating arrays with configurable shape,
    data type, initialization function, and optional caching backend.
    It provides methods for initializing, setting values, removing values, writing to
    disk, and loading from disk while maintaining a consistent state.

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
    value
        Actual array data.
    cache_backend
        Optional backend for disk caching.

    Examples
    --------
    # Create an array with specific shape and initialize
    arr = Array(shape=(10, 10))
    initialized_arr = arr.initialize()

    # Set a value and write to disk
    arr = arr.set_value(np.random.rand(10, 10))
    arr = arr.written_to_disk(backend)

    """

    shape = attrs.field(converter=_tuple_of_ints)
    dtype = attrs.field(default=float, kw_only=True)
    state = attrs.field(factory=ArrayState, kw_only=True)
    initfunc = attrs.field(default=np.zeros, kw_only=True)
    value = attrs.field(
        converter=attrs.converters.optional(np.asarray), default=None, kw_only=True
    )
    cache_backend = attrs.field(
        default=None, validator=optional(instance_of(CacheBackend)), kw_only=True
    )

    @value.validator
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

        backend.write(self.value)
        return attrs.evolve(self, cache_backend=backend, state=self.state.written())

    def purged_to_disk(self, backend: CacheBackend | None) -> Self:
        """Move the array data to disk and return a new object with correct state."""
        return attrs.evolve(self.written_to_disk(backend), value=None)

    def loaded_from_disk(self, backend: CacheBackend | None = None) -> Self:
        """Load values for the array from a cache backend, and return a new instance."""
        if self.value is not None:
            return attrs.evolve(self, cache_backend=backend)

        backend = backend or self.cache_backend

        if backend is None:
            raise ValueError("backend must be specified")

        value = backend.read()
        return attrs.evolve(
            self,
            value=value,
            cache_backend=backend,
            state=self.state.loaded_from_disk(),
        )
