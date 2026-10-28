"""Unit tests for `Array` and for how `OutputStruct` exposes its arrays.

Two related things are covered here.

`Array` itself is the *manager* for a block of array data: it tracks whether the
data is in memory, on disk or shared with the C backend, and it knows how to move
between those states. Per issue #565, once its value has been purged to disk, using
it as array data transparently resolves it, and by default the resolved value is
cached back onto the instance so repeated access doesn't keep re-reading. That must
be opt-outable via `config["CACHE_ARRAYS_ON_ACCESS"]`, and invisible to plain
inspection (`repr`, `hasattr`, `copy.deepcopy`, `pickle`), none of which should
trigger a disk read or raise a surprising error.

Separately, an `OutputStruct` does *not* hand you the `Array`. Each field is stored
privately (`_hires_density`) and exposed under its public name by an `ArrayProxy`
descriptor that gives back a plain `np.ndarray`. The `Array` stays reachable through
`OutputStruct.arrays[...]`. This is what stops the old half-duck-typing problem,
where `ic.hires_density.mean()` worked but `ic.hires_density ** 2` did not.
"""

import copy
import logging
import pickle

import attrs
import deprecation
import numpy as np
import pytest
from astropy import units as un

from py21cmfast import InitialConditions, config
from py21cmfast.wrapper.arrays import (
    Array,
    ArrayProxy,
    CacheBackend,
    _as_legacy_view,
    expose_arrays,
)


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


@pytest.mark.parametrize("unit", [un.K, "K"])
def test_unit_metadata_is_normalized_without_allocating(unit):
    array = Array((2,), unit=unit, description="Temperature.")
    assert array.unit == un.K
    assert array.description == "Temperature."
    assert array._value is None
    assert not array.state.initialized


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_get_quantity_preserves_values_and_dtype(dtype):
    values = np.arange(3, dtype=dtype)
    array = Array(values.shape, dtype=dtype, unit=un.K).with_value(values)
    quantity = array.get_quantity()
    assert isinstance(quantity, un.Quantity)
    assert quantity.unit == un.K
    assert quantity.dtype == values.dtype
    np.testing.assert_array_equal(quantity.value, values)


@pytest.mark.parametrize(
    ("unit", "physical_unit", "scale"),
    [
        (un.Unit(1e-12 / un.s), un.s**-1, 1e-12),
        (
            un.Unit(1e-21 * un.erg / (un.s * un.cm**2 * un.Hz * un.sr)),
            un.erg / (un.s * un.cm**2 * un.Hz * un.sr),
            1e-21,
        ),
    ],
)
def test_get_quantity_applies_stored_unit_scale(unit, physical_unit, scale):
    array = Array((2,), unit=unit).with_value(np.array([1.0, 2.0]))
    np.testing.assert_allclose(
        array.get_quantity().to_value(physical_unit), [scale, 2 * scale], atol=0
    )
    np.testing.assert_array_equal(np.asarray(array), [1.0, 2.0])


def test_get_quantity_distinguishes_dimensionless_from_unspecified_units():
    values = np.array([0.25, 0.75])
    array = Array(values.shape).with_value(values)
    assert array.unit is None
    assert array.description is None
    with pytest.raises(ValueError, match="Unit metadata is not defined"):
        array.get_quantity()

    dimensionless = attrs.evolve(array, unit=un.dimensionless_unscaled)
    assert dimensionless.get_quantity().unit == un.dimensionless_unscaled
    np.testing.assert_array_equal(dimensionless.get_quantity().value, values)


def test_get_quantity_shares_read_only_memory_unless_copy_requested() -> None:
    array = Array((2,), dtype=np.float32, unit=un.K).with_value(np.ones(2))
    view = array.get_quantity()
    independent = array.get_quantity(copy=True)
    assert independent.dtype == array.dtype
    assert np.shares_memory(view, array._value)
    assert not np.shares_memory(independent, array._value)
    assert not view.flags.writeable
    assert independent.flags.writeable
    assert array._value.flags.writeable

    with pytest.raises(ValueError, match="read-only"):
        view[0] = 2 * un.K
    independent[1] = 3 * un.K
    np.testing.assert_array_equal(np.asarray(array), [1, 1])

    # A read-only view still follows changes made through the existing raw API.
    np.asarray(array)[0] = 4
    np.testing.assert_array_equal(view.to_value(un.K), [4, 1])
    np.testing.assert_array_equal(independent.to_value(un.K), [1, 3])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
@pytest.mark.parametrize(
    ("unit", "converted_unit"),
    [
        (un.K, un.mK),
        (un.Unit(1e-12 / un.s), un.s**-1),
        (un.dimensionless_unscaled, un.percent),
    ],
)
def test_shared_quantity_unit_conversion_preserves_stored_values(
    dtype: type[np.generic], unit: un.UnitBase, converted_unit: un.UnitBase
) -> None:
    values = np.array([1, 2], dtype=dtype)
    expected = values.copy()
    array = Array(values.shape, dtype=dtype, unit=unit).with_value(values)
    quantity = array.get_quantity()

    try:
        quantity <<= converted_unit
    except ValueError as error:
        if "read-only" not in str(error):
            raise
    else:
        # Astropy can fall back to a copy if an in-place cast is impossible.
        assert quantity.unit == converted_unit
        assert not np.shares_memory(quantity, np.asarray(array))
        np.testing.assert_allclose(quantity.to_value(unit), expected)

    assert array.unit == unit
    assert array._value.flags.writeable
    np.testing.assert_array_equal(np.asarray(array), expected)


def test_quantity_conversion_can_be_edited_without_changing_stored_values() -> None:
    array = Array((1,), dtype=np.float32, unit=un.K).with_value(
        np.array([100.0], dtype=np.float32)
    )
    converted = array.get_quantity().to(un.mK)
    assert converted.flags.writeable
    assert not np.shares_memory(converted, np.asarray(array))
    np.testing.assert_allclose(converted.value, [100000.0])

    converted[0] = 200000 * un.mK
    np.testing.assert_array_equal(np.asarray(array), [100.0])
    assert array.unit == un.K


def test_get_quantity_on_uncomputed_array_does_not_allocate():
    array = Array((2,), unit=un.K)
    with pytest.raises(ValueError, match="not on disk and not initialized"):
        array.get_quantity()
    assert array._value is None
    assert not array.state.initialized


@pytest.mark.parametrize("cache_on_access", [False, True])
def test_get_quantity_loads_purged_data_with_existing_cache_policy(
    purged_array, backend, raw_value, cache_on_access
):
    array = attrs.evolve(purged_array, unit=un.K, description="Temperature.")
    assert array.unit == un.K
    assert array.description == "Temperature."
    assert backend.read_count == 0

    with config.use(CACHE_ARRAYS_ON_ACCESS=cache_on_access):
        for _ in range(2):
            np.testing.assert_array_equal(array.get_quantity().value, raw_value)

    assert backend.read_count == (1 if cache_on_access else 2)
    assert array.state.computed_in_mem == cache_on_access
    assert (array._value is not None) == cache_on_access


def test_missing_unit_metadata_does_not_load_purged_data(purged_array, backend):
    with pytest.raises(ValueError, match="Unit metadata is not defined"):
        purged_array.get_quantity()
    assert backend.read_count == 0
    assert purged_array._value is None


def test_unit_metadata_survives_array_transitions_and_pickle(raw_value, backend):
    array = Array(raw_value.shape, unit="K", description="Temperature.")
    initialized = array.initialize()
    computed = initialized.with_value(raw_value)
    purged = computed.purged_to_disk(backend)
    loaded = purged.loaded_from_disk()
    for changed in (initialized, computed, purged, loaded, loaded.trimmed((2, 2, 2))):
        assert changed.unit == un.K
        assert changed.description == "Temperature."

    restored = pickle.loads(pickle.dumps(purged))
    assert restored.unit == un.K
    assert restored.description == "Temperature."
    assert restored._value is None
    np.testing.assert_array_equal(restored.get_quantity().value, raw_value)


@pytest.mark.parametrize("state", ["uncomputed", "computed", "purged"])
def test_pickle_without_unit_metadata_retains_existing_array_operations(
    state, raw_value, backend
):
    array = Array(raw_value.shape)
    if state != "uncomputed":
        array = array.with_value(raw_value)
    if state == "purged":
        array = array.purged_to_disk(backend)

    # Match the dictionary stored by Array before metadata was added.
    object.__delattr__(array, "unit")
    object.__delattr__(array, "description")
    restored = pickle.loads(pickle.dumps(array))

    assert restored.unit is None
    assert restored.description is None
    if state == "purged":
        assert restored._value is None
        assert restored.cache_backend.read_count == 0
    updated = restored.with_value(raw_value)
    np.testing.assert_array_equal(np.asarray(updated), raw_value)


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


def test_purged_array_raw_slot_is_none(purged_array):
    assert purged_array._value is None
    assert purged_array.state.on_disk
    assert not purged_array.state.computed_in_mem


def test_purged_array_method_call_loads_and_caches(purged_array, backend, raw_value):
    result = purged_array.mean()
    assert result == pytest.approx(raw_value.mean())

    # Cached directly onto the instance as a side effect of the access above.
    assert purged_array._value is not None
    assert purged_array.state.computed_in_mem
    assert backend.read_count == 1

    # A second access must not hit disk again.
    _ = purged_array.mean()
    assert backend.read_count == 1


def test_purged_array_numpy_protocol_loads_and_caches(purged_array, backend, raw_value):
    assert np.allclose(np.asarray(purged_array), raw_value)
    assert purged_array._value is not None
    assert backend.read_count == 1


def test_config_flag_disables_auto_caching(purged_array, backend, raw_value):
    with config.use(CACHE_ARRAYS_ON_ACCESS=False):
        result = purged_array.mean()
        assert result == pytest.approx(raw_value.mean())

        # Must NOT be cached onto the instance.
        assert purged_array._value is None
        assert not purged_array.state.computed_in_mem

        # A second access re-reads from disk.
        _ = purged_array.mean()
        assert backend.read_count == 2

    # Default (caching) behavior resumes outside the context.
    _ = purged_array.mean()
    assert purged_array._value is not None
    assert backend.read_count == 3


def test_cache_arrays_on_access_defaults_true():
    assert config["CACHE_ARRAYS_ON_ACCESS"] is True


def test_repr_of_purged_array_does_not_read_disk(purged_array, backend):
    _ = repr(purged_array)
    assert backend.read_count == 0
    assert purged_array._value is None


def test_dunder_probe_on_purged_array_does_not_read_disk(purged_array, backend):
    assert not hasattr(purged_array, "__deepcopy__")
    assert backend.read_count == 0


def test_deepcopy_of_purged_array_does_not_read_disk(purged_array, backend):
    copy.deepcopy(purged_array)
    assert backend.read_count == 0


def test_pickle_of_purged_array_does_not_read_disk(purged_array, backend):
    unpickled = pickle.loads(pickle.dumps(purged_array))
    assert backend.read_count == 0
    assert unpickled._value is None
    assert unpickled.state.on_disk


# ---------------------------------------------------------------------------
# OutputStruct array fields are plain numpy arrays, not `Array`s.
# ---------------------------------------------------------------------------


def test_struct_attribute_is_a_plain_ndarray(ic: InitialConditions):
    """The public attribute is the data; the `Array` lives beside it."""
    density = ic.hires_density

    assert isinstance(density, np.ndarray)
    assert not isinstance(density, Array)

    # The manager is reachable, and is exactly the object in the private field.
    assert isinstance(ic.arrays["hires_density"], Array)
    assert ic.arrays["hires_density"] is ic._hires_density


def test_struct_attribute_behaves_fully_like_an_array(ic: InitialConditions):
    """Every array behaviour, not just the ones `__getattr__` could delegate.

    Operators are the ones that used to fail: implicit dunder lookup never consults
    `__getattr__`, so `Array ** 2` raised `TypeError` while `Array.mean()` worked.
    """
    density = ic.hires_density

    assert (density**2).mean() >= 0.0  # __pow__
    assert np.all((density - density) == 0.0)  # __sub__
    assert density[0].shape == density.shape[1:]  # __getitem__
    assert len(density) == density.shape[0]  # __len__
    assert np.all(np.zeros_like(density) == 0.0)  # __eq__ against a scalar
    assert density.mean() == pytest.approx(np.asarray(density).mean())


def test_class_level_access_returns_the_descriptor():
    """Class access must give the descriptor, so introspection still works."""
    assert isinstance(InitialConditions.__dict__["hires_density"], ArrayProxy)
    assert "hires_density" in InitialConditions._array_field_names


def test_absent_optional_field_is_none(ic: InitialConditions):
    """None means "this field doesn't exist for these inputs", nothing else."""
    assert ic.lowres_vcb is None
    assert "lowres_vcb" not in ic.arrays


def test_reading_a_field_with_no_data_raises(ic: InitialConditions):
    """A field that exists but was never computed has no data to give back."""
    fresh = InitialConditions.new(inputs=ic.inputs)

    assert not fresh.has("hires_density")
    with pytest.raises(ValueError, match="not on disk and not initialized"):
        _ = fresh.hires_density


def test_purged_struct_field_loads_on_access_and_caches_back(ic: InitialConditions):
    expected = ic.get("hires_density")
    ic.purge()
    assert not ic.arrays["hires_density"].state.computed_in_mem

    assert np.allclose(ic.hires_density, expected)
    assert ic.arrays["hires_density"].state.computed_in_mem

    ic.load_all()


def test_purged_struct_field_respects_no_cache_config(ic: InitialConditions):
    expected = ic.get("hires_density")
    ic.purge()

    with config.use(CACHE_ARRAYS_ON_ACCESS=False):
        assert np.allclose(ic.hires_density, expected)
        # Resolved, but deliberately not retained on the struct.
        assert not ic.arrays["hires_density"].state.computed_in_mem

    ic.load_all()


def test_disk_read_is_logged_at_debug(ic: InitialConditions, caplog):
    """Loading from disk must be traceable, since it can happen implicitly."""
    ic.purge()

    with caplog.at_level(logging.DEBUG, logger="py21cmfast.wrapper.arrays"):
        _ = ic.hires_density.mean()

    assert any("from disk" in r.message for r in caplog.records)

    ic.load_all()


def test_assigning_an_ndarray_sets_the_data(ic: InitialConditions):
    """`ic.field = arr` is the natural way to write data, and must keep the Array."""
    original = ic.get("hires_density").copy()
    try:
        ic.hires_density = np.zeros_like(original)

        assert np.all(ic.hires_density == 0.0)
        # Still managed by an Array, which now reports the data as in memory.
        assert isinstance(ic.arrays["hires_density"], Array)
        assert ic.arrays["hires_density"].state.computed_in_mem
    finally:
        ic.hires_density = original
        ic.load_all()


def test_assigning_an_array_replaces_the_field(ic: InitialConditions):
    """The memory-management machinery swaps whole `Array`s through the same path."""
    original = ic.arrays["hires_density"]
    try:
        ic.hires_density = original.without_value()
        assert ic.arrays["hires_density"]._value is None
    finally:
        ic.hires_density = original
        ic.load_all()


def test_expose_arrays_rejects_a_public_array_field():
    """Array fields must be private, or the descriptor would have nowhere to live."""
    with pytest.raises(TypeError, match="must be declared"):

        @expose_arrays
        @attrs.define(slots=False, kw_only=True)
        class _Bad:
            density = attrs.field(default=None, type=Array)


# ---------------------------------------------------------------------------
# The transitional `Array`-API shim on the returned numpy array.
# ---------------------------------------------------------------------------


def test_legacy_view_answers_array_api_with_a_warning(in_memory_array, raw_value):
    view = _as_legacy_view(raw_value, in_memory_array)

    assert isinstance(view, np.ndarray)
    with pytest.warns(deprecation.DeprecatedWarning, match="state is deprecated"):
        assert view.state.computed_in_mem
    with pytest.warns(deprecation.DeprecatedWarning, match="value is deprecated"):
        assert np.allclose(view.value, raw_value)


def test_legacy_view_does_not_shadow_real_numpy_attributes(
    in_memory_array, raw_value, recwarn
):
    view = _as_legacy_view(raw_value, in_memory_array)

    assert view.dtype == raw_value.dtype
    assert view.shape == raw_value.shape
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    with pytest.raises(AttributeError):
        _ = view.not_a_real_attribute


def test_legacy_view_does_not_propagate_into_derived_arrays(in_memory_array, raw_value):
    """The transitional type must not leak into unrelated computations.

    It previously did, via `__array_finalize__`, and reached an astropy `Quantity`
    operation deep in the drivers. Nothing is lost by not propagating: the old
    `Array` defined no operators or `__getitem__`, so `ic.hires_density[0].value`
    never worked in the first place.
    """
    view = _as_legacy_view(raw_value, in_memory_array)

    derived = view + 1
    assert type(derived) is np.ndarray
    with pytest.raises(AttributeError):
        _ = derived.state

    # Slices keep the type (numpy insists) but must not answer the legacy API.
    with pytest.raises(AttributeError):
        _ = view[0].state
