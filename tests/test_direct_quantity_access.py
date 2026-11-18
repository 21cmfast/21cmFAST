"""Check optional quantity access on raw output fields without implicit units."""

import copy
import pickle
import warnings
from collections.abc import Callable

import deprecation
import numpy as np
import pytest
from astropy import units as u

from py21cmfast import config
from py21cmfast.wrapper.arrays import Array, CacheBackend, _as_legacy_view
from py21cmfast.wrapper.inputs import InputParameters
from py21cmfast.wrapper.outputs import TsBox


@pytest.fixture
def temperature_box(default_input_struct: InputParameters) -> TsBox:
    """Create a real output struct with synthetic, computed temperature data."""
    box = TsBox.new(inputs=default_input_struct, redshift=10)
    array = box.arrays["spin_temperature"]
    box.spin_temperature = np.ones(array.shape, dtype=array.dtype)
    return box


def test_direct_quantity_access_preserves_raw_field_and_copy_contract(
    temperature_box: TsBox,
) -> None:
    field = temperature_box.spin_temperature
    quantity = field.get_quantity()
    independent = field.get_quantity(copy=True)
    assert isinstance(field, np.ndarray)
    assert not isinstance(field, u.Quantity)
    assert not hasattr(field, "unit")
    assert not hasattr(field, "description")
    assert quantity.unit == temperature_box.arrays["spin_temperature"].unit == u.K
    assert quantity.dtype == independent.dtype == field.dtype
    assert np.shares_memory(quantity, field)
    assert not np.shares_memory(independent, field)
    assert not quantity.flags.writeable
    assert independent.flags.writeable
    assert field.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        quantity <<= u.mK
    field.flat[0] = 3
    assert quantity.flat[0] == 3 * u.K
    assert independent.flat[0] == 1 * u.K


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
@pytest.mark.parametrize("unit", [u.K, u.dimensionless_unscaled, u.Unit(1e-12 / u.s)])
def test_direct_quantity_preserves_dtype_and_scaled_unit(
    dtype: type[np.generic], unit: u.UnitBase
) -> None:
    values = np.array([1, 2], dtype=dtype)
    array = Array((2,), dtype=dtype, unit=unit, description="Test field.").with_value(
        values
    )
    field = _as_legacy_view(values, array)
    quantity = field.get_quantity()
    assert quantity.unit == unit
    assert quantity.dtype == values.dtype
    np.testing.assert_array_equal(quantity.value, values)


@pytest.mark.parametrize(
    "operation",
    [
        lambda a: a + 1,
        lambda a: a[0],
        lambda a: a[:1],
        lambda a: a.copy(),
        lambda a: a.astype(np.float64),
        lambda a: a.reshape(-1),
        lambda a: a.T,
        lambda a: np.sin(a),
        lambda a: np.stack([a, a]),
        lambda a: np.concatenate([a, a]),
        lambda a: np.asarray(a),
        lambda a: copy.copy(a),
        lambda a: copy.deepcopy(a),
        lambda a: pickle.loads(pickle.dumps(a)),
    ],
    ids=[
        "arithmetic",
        "index",
        "slice",
        "copy",
        "cast",
        "reshape",
        "transpose",
        "ufunc",
        "stack",
        "concatenate",
        "asarray",
        "shallow-copy",
        "deep-copy",
        "pickle",
    ],
)
def test_direct_quantity_access_does_not_follow_derived_raw_arrays(
    temperature_box: TsBox, operation: Callable[[np.ndarray], np.ndarray]
) -> None:
    field = temperature_box.spin_temperature
    derived = operation(field)
    np.testing.assert_array_equal(derived, operation(np.asarray(field)))
    assert not hasattr(derived, "unit")
    assert not hasattr(derived, "get_quantity")


@pytest.mark.parametrize(
    "operation",
    [
        lambda a: a * u.K,
        lambda a: u.K * a,
        lambda a: a * (3 * u.s),
        lambda a: (3 * u.s) * a,
        lambda a: np.multiply(3 * u.s, a),
        lambda a: u.Quantity(a),
        lambda a: u.Quantity(a, u.mK),
    ],
    ids=[
        "right-unit",
        "left-unit",
        "right-quantity",
        "left-quantity",
        "ufunc",
        "infer",
        "explicit",
    ],
)
def test_direct_quantity_access_keeps_existing_astropy_interpretation(
    temperature_box: TsBox, operation: Callable[[np.ndarray], u.Quantity]
) -> None:
    field = temperature_box.spin_temperature
    # Astropy can probe the existing deprecated .value shim. Preserve that warning;
    # capture only this category here so its inherited behavior is not a new failure.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", deprecation.DeprecatedWarning)
        actual = operation(field)
    assert all(str(w.message).startswith("value is deprecated") for w in caught)
    expected = operation(np.asarray(field))
    assert actual.unit == expected.unit
    np.testing.assert_array_equal(actual.value, expected.value)


def test_direct_quantity_access_survives_explicit_in_place_outputs(
    temperature_box: TsBox,
) -> None:
    field = temperature_box.spin_temperature
    assert np.add(field, 1, out=field) is field
    field += 1
    assert field.get_quantity().flat[0] == 3 * u.K
    fractional, integer = np.modf(field, out=(None, field))
    assert integer is field
    assert not hasattr(fractional, "get_quantity")
    np.testing.assert_array_equal(fractional, 0)
    result = np.empty((), dtype=field.dtype)
    assert np.add.reduce(field.reshape(-1), out=result) is result


def test_direct_quantity_access_uses_observed_values_after_field_replacement(
    temperature_box: TsBox,
) -> None:
    old_field = temperature_box.spin_temperature
    old_shape = old_field.shape
    temperature_box.spin_temperature = Array(
        old_shape, dtype=old_field.dtype, unit=u.mK, description="Temperature."
    ).with_value(np.full(old_shape, 2000, dtype=old_field.dtype))
    assert old_field.get_quantity().flat[0] == 1 * u.K
    assert temperature_box.spin_temperature.get_quantity().flat[0] == 2000 * u.mK


class _ReadCountingBackend(CacheBackend):
    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.reads = 0

    def read(self) -> np.ndarray:
        self.reads += 1
        return self.values.copy()

    def write(self, val: np.ndarray) -> None:
        self.values = val.copy()


@pytest.mark.parametrize("cache_on_access", [False, True])
def test_direct_quantity_access_uses_resolved_data_without_rereading(
    temperature_box: TsBox, cache_on_access: bool
) -> None:
    array = temperature_box.arrays["spin_temperature"]
    backend = _ReadCountingBackend(np.asarray(array))
    temperature_box.spin_temperature = array.purged_to_disk(backend)
    with config.use(CACHE_ARRAYS_ON_ACCESS=cache_on_access):
        field = temperature_box.spin_temperature
        assert backend.reads == 1
        assert (
            temperature_box.arrays["spin_temperature"].state.computed_in_mem
            == cache_on_access
        )
        for _ in range(2):
            np.testing.assert_array_equal(field.get_quantity().value, 1)
        assert backend.reads == 1


def test_direct_quantity_access_rejects_unknown_legacy_units(
    temperature_box: TsBox,
) -> None:
    array = temperature_box.arrays["spin_temperature"]
    legacy = object.__new__(Array)
    legacy.__setstate__(
        {
            key: value
            for key, value in array.__dict__.items()
            if key not in {"unit", "description"}
        }
    )
    temperature_box.spin_temperature = legacy
    field = temperature_box.spin_temperature
    np.testing.assert_array_equal(field, 1)
    with pytest.raises(ValueError, match="Unit metadata is not defined"):
        field.get_quantity()
