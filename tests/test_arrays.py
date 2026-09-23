"""Unit tests for `Array`'s numpy duck-typing and auto-caching-on-access behavior.

See issue #565: once an `Array`'s value has been purged to disk, plain attribute
access (not `.get()`) should be able to transparently resolve it, and by default
that resolved value is cached directly onto the `Array` instance so repeated access
doesn't keep re-reading from disk. This must be:

- Opt-outable via `config["CACHE_ARRAYS_ON_ACCESS"]`, for users who want strict
  control over when memory gets reallocated.
- Invisible to plain inspection (`repr`, `hasattr`, `copy.deepcopy`, `pickle`) - none
  of those should trigger a disk read or raise a surprising error.
"""

import copy
import pickle

import numpy as np
import pytest

from py21cmfast import config
from py21cmfast.wrapper.arrays import Array, CacheBackend


class _FakeBackend(CacheBackend):
    """An in-memory CacheBackend that counts reads/writes, for testing caching."""

    def __init__(self, value: np.ndarray):
        self._value = value
        self.read_count = 0
        self.write_count = 0

    def read(self) -> np.ndarray:
        self.read_count += 1
        return self._value

    def write(self, val: np.ndarray) -> None:
        self.write_count += 1
        self._value = val


@pytest.fixture
def raw_value():
    return np.arange(27.0).reshape((3, 3, 3))


@pytest.fixture
def backend(raw_value):
    return _FakeBackend(raw_value)


@pytest.fixture
def in_memory_array(raw_value):
    return Array(shape=raw_value.shape).with_value(raw_value)


@pytest.fixture
def purged_array(in_memory_array, backend):
    return in_memory_array.purged_to_disk(backend)


@pytest.fixture
def uncomputed_array():
    return Array(shape=(3, 3, 3))


# ---------------------------------------------------------------------------
# Basic duck-typing: Array behaves like an ndarray when a value is available.
# ---------------------------------------------------------------------------


def test_array_type_unchanged_by_duck_typing(in_memory_array):
    """Direct attribute access must keep returning an `Array`, not a bare ndarray."""
    assert isinstance(in_memory_array, Array)


def test_array_proxies_ndarray_methods_when_in_memory(in_memory_array, raw_value):
    assert in_memory_array.mean() == raw_value.mean()
    assert in_memory_array.sum() == raw_value.sum()


def test_array_supports_numpy_array_protocol_when_in_memory(in_memory_array, raw_value):
    assert np.allclose(np.asarray(in_memory_array), raw_value)
    assert np.allclose(np.sum(in_memory_array), raw_value.sum())


def test_array_getattr_of_missing_ndarray_attr_raises_attributeerror(in_memory_array):
    with pytest.raises(AttributeError):
        _ = in_memory_array.this_is_not_a_real_ndarray_attribute


# ---------------------------------------------------------------------------
# Uncomputed arrays: errors, but never a false "surprise".
# ---------------------------------------------------------------------------


def test_array_method_call_on_uncomputed_raises_attributeerror(uncomputed_array):
    # Going through __getattr__ (a method call) should raise AttributeError, not
    # ValueError, so that hasattr()/copy/pickle probing behaves correctly (below).
    with pytest.raises(AttributeError):
        uncomputed_array.mean()


def test_array_protocol_on_uncomputed_raises_valueerror(uncomputed_array):
    # np.asarray() goes through __array__ directly (not __getattr__), so it keeps
    # the same error type .get() already raises for this situation.
    with pytest.raises(ValueError, match="not on disk and not initialized"):
        np.asarray(uncomputed_array)


def test_hasattr_on_uncomputed_array_does_not_raise(uncomputed_array):
    assert hasattr(uncomputed_array, "some_random_attr") is False


def test_deepcopy_of_uncomputed_array_does_not_raise(uncomputed_array):
    copy.deepcopy(uncomputed_array)


# ---------------------------------------------------------------------------
# Purged (on-disk) arrays: transparent loading, cached back by default.
# ---------------------------------------------------------------------------


def test_purged_array_value_is_none(purged_array):
    assert purged_array.value is None
    assert purged_array.state.on_disk
    assert not purged_array.state.computed_in_mem


def test_purged_array_method_call_loads_and_caches(purged_array, backend, raw_value):
    result = purged_array.mean()
    assert result == pytest.approx(raw_value.mean())

    # Cached directly onto the instance as a side effect of the access above.
    assert purged_array.value is not None
    assert purged_array.state.computed_in_mem
    assert backend.read_count == 1

    # A second access must not hit disk again.
    _ = purged_array.mean()
    assert backend.read_count == 1


def test_purged_array_numpy_protocol_loads_and_caches(purged_array, backend, raw_value):
    assert np.allclose(np.asarray(purged_array), raw_value)
    assert purged_array.value is not None
    assert backend.read_count == 1


def test_config_flag_disables_auto_caching(purged_array, backend, raw_value):
    with config.use(CACHE_ARRAYS_ON_ACCESS=False):
        result = purged_array.mean()
        assert result == pytest.approx(raw_value.mean())

        # Must NOT be cached onto the instance.
        assert purged_array.value is None
        assert not purged_array.state.computed_in_mem

        # A second access re-reads from disk.
        _ = purged_array.mean()
        assert backend.read_count == 2

    # Default (caching) behavior resumes outside the context.
    _ = purged_array.mean()
    assert purged_array.value is not None
    assert backend.read_count == 3


def test_cache_arrays_on_access_defaults_true():
    assert config["CACHE_ARRAYS_ON_ACCESS"] is True


def test_repr_of_purged_array_does_not_read_disk(purged_array, backend):
    _ = repr(purged_array)
    assert backend.read_count == 0
    assert purged_array.value is None


def test_dunder_probe_on_purged_array_does_not_read_disk(purged_array, backend):
    assert not hasattr(purged_array, "__deepcopy__")
    assert backend.read_count == 0


def test_deepcopy_of_purged_array_does_not_read_disk(purged_array, backend):
    copy.deepcopy(purged_array)
    assert backend.read_count == 0


def test_pickle_of_purged_array_does_not_read_disk(purged_array, backend):
    unpickled = pickle.loads(pickle.dumps(purged_array))
    assert backend.read_count == 0
    assert unpickled.value is None
    assert unpickled.state.on_disk
