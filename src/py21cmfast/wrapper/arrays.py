"""Module for dealing with arrays that are input/output to C functions."""

import attrs
import h5py
import numpy as np
from abc import ABC, abstractmethod
from attrs.validators import instance_of, optional
from pathlib import Path
from typing import Sequence

from .arraystate import ArrayState


def tuple_of_ints(x: Sequence[float | int]) -> tuple[int]:
    return tuple(int(i) for i in x)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def read(self) -> np.ndarray:
        pass

    @abstractmethod
    def write(self, val: np.ndarray) -> None:
        pass


@attrs.define(frozen=True)
class H5Backend(CacheBackend):
    """Backend for caching arrays in a HDF5 file."""

    path: Path = attrs.field(converter=Path)
    dataset: str = attrs.field(converter=str)

    def read(self) -> np.ndarray:
        with h5py.File(self.path, "r") as f:
            return f[self.dataset][()]

    def write(self, val: np.ndarray, overwrite: bool = False) -> None:
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
    shape = attrs.field(converter=tuple_of_ints)
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
    def value_validator(self, att, val):
        if val is None:
            return

        if val.shape != self.shape:
            raise ValueError(f"Shape mismatch: expected {self.shape}, got {val.shape}")

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

    def set_value(self, val: np.ndarray):
        """Set the array to a given value."""
        return attrs.evolve(self, value=val, state=self.state.as_computed())

    def without_value(self):
        """Remove the allocated data from the array."""
        return attrs.evolve(self, value=None, state=self.state.dropped())

    def written_to_disk(self, backend: CacheBackend | None):
        """Write the array to disk and return a new object with correct state."""
        backend = backend or self.cache_backend

        if backend is None:
            raise ValueError("backend must be specified")

        backend.write(self.value)
        return attrs.evolve(self, cache_backend=backend, state=self.state.written())

    def purged_to_disk(self, backend: CacheBackend | None):
        """Move the array data to disk."""
        return attrs.evolve(self.written_to_disk(backend), value=None)

    def loaded_from_disk(self, backend: CacheBackend | None = None):
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
