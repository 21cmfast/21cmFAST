"""Read, write, and compare physical quantities in integration reference data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import h5py
import numpy as np
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from powerbox import get_power

if TYPE_CHECKING:
    from py21cmfast.wrapper.outputs import OutputStruct


def assert_reference_fields(group: h5py.Group, expected: Iterable[str]) -> None:
    """Require the complete dataset inventory before reading physical references.

    Parameters
    ----------
    group
        Reference group for one simulation configuration.
    expected
        Dataset names required by that configuration, independently of stored data.
    """
    actual, required = set(group), set(expected)
    location = f"{group.file.filename}:{group.name}"
    assert not (missing := required - actual), (
        f"Missing reference fields in {location}: {sorted(missing)}"
    )
    assert not (extra := actual - required), (
        f"Unexpected reference fields in {location}: {sorted(extra)}"
    )
    for name in sorted(required):
        assert isinstance(group[name], h5py.Dataset), (
            f"Reference {location}/{name} must be a dataset."
        )


def field_unit(boxes: Iterable[OutputStruct], name: str) -> u.UnitBase:
    """Read a field's declared unit without loading its numerical array."""
    for box in boxes:
        if name in box.arrays:
            unit = box.arrays[name].unit
            if unit is None:
                raise ValueError(f"The field {name!r} has no unit metadata.")
            return unit
    raise KeyError(f"No output struct contains the field {name!r}.")


def write_quantity(group: h5py.Group, name: str, quantity: u.Quantity) -> h5py.Dataset:
    """Store numerical samples with an explicit, parseable unit attribute."""
    if not isinstance(quantity, u.Quantity):
        raise TypeError(f"The reference {name!r} must be an Astropy Quantity.")
    dataset = group.create_dataset(name, data=quantity.value)
    dataset.attrs["unit"] = quantity.unit.to_string()
    return dataset


def read_quantity(dataset: h5py.Dataset) -> u.Quantity:
    """Load a typed reference, rejecting absent or invalid unit metadata."""
    if "unit" not in dataset.attrs:
        raise ValueError(f"Reference {dataset.name!r} has no unit metadata.")
    stored_unit = dataset.attrs["unit"]
    if not isinstance(stored_unit, str | bytes):
        raise TypeError(f"Reference {dataset.name!r} has invalid unit metadata.")
    if isinstance(stored_unit, bytes):
        stored_unit = stored_unit.decode("utf-8")
    return u.Quantity(dataset[...], u.Unit(stored_unit), dtype=None, copy=False)


def assert_physical_type(
    actual: u.Quantity, reference: u.Quantity, *, name: str = ""
) -> None:
    """Reject incompatible dimensions independently of numerical tolerances.

    Parameters
    ----------
    actual, reference
        Typed physical values to compare. Scaled representations of the same
        dimensions are accepted; global physical equivalencies are not applied.
    name
        Field name included in an incompatible-dimensions error.
    """
    if not isinstance(actual, u.Quantity) or not isinstance(reference, u.Quantity):
        raise TypeError("Both actual and reference must be Astropy Quantity objects.")
    if not actual.unit.is_equivalent(reference.unit, equivalencies=None):
        raise u.UnitConversionError(
            f"{name}: incompatible physical dimensions: {actual.unit} and {reference.unit}"
        )


def assert_quantity_close(
    actual: u.Quantity,
    reference: u.Quantity,
    *,
    rtol: float,
    atol: u.Quantity | None = None,
    name: str = "",
) -> None:
    """Assert physical agreement without broadcasting or accepting untyped data."""
    assert_physical_type(actual, reference, name=name)
    assert actual.shape == reference.shape, f"{name}: reference shape mismatch"
    assert np.all(np.isfinite(actual)), f"{name}: nonfinite simulation values"
    assert np.all(np.isfinite(reference)), f"{name}: nonfinite reference values"
    assert_quantity_allclose(
        actual, reference, rtol=rtol, atol=atol, equal_nan=False, err_msg=name
    )


def power_spectrum(
    field: u.Quantity, boxlength: u.Quantity
) -> tuple[u.Quantity, u.Quantity]:
    """Attach field-squared times volume units to the existing power estimator."""
    result = get_power(
        field.value,
        boxlength=boxlength.to_value(u.Mpc),
        bins_upto_boxlen=True,
        vol_normalised_power=True,
    )
    return (
        u.Quantity(result.power, field.unit**2 * u.Mpc**field.ndim, copy=False),
        u.Quantity(result.bin_avg, u.Mpc**-1, copy=False),
    )
