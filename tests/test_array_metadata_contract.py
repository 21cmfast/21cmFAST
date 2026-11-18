"""Ensure driver shortcut paths preserve each field's declared metadata."""

import inspect

import numpy as np
import pytest

from py21cmfast import InitialConditions, InputParameters, IonizedBox
from py21cmfast.drivers.global_evolution import compute_global_reionization_at_z
from py21cmfast.drivers.single_field import compute_initial_conditions
from py21cmfast.wrapper import cfuncs


@pytest.mark.parametrize("initial_density", [None, 0.25])
def test_single_cell_initial_conditions_preserve_metadata(
    default_input_struct: InputParameters, initial_density: float | None
) -> None:
    inputs = default_input_struct.evolve_input_structs(HII_DIM=1, DIM=1)
    expected = InitialConditions.new(inputs=inputs)
    # Test the Python shortcut without C-state/cache decorators or a simulation.
    result = inspect.unwrap(compute_initial_conditions)(
        inputs=inputs, initial_density=initial_density
    )
    computed = {
        name: array
        for name, array in result.arrays.items()
        if array.state.computed_in_mem
    }
    assert computed
    for name, array in computed.items():
        assert array.unit == expected.arrays[name].unit
        assert array.description == expected.arrays[name].description
        value = (
            initial_density if initial_density is not None and "density" in name else 0
        )
        np.testing.assert_array_equal(array.get_quantity().value, value)


def test_initial_density_override_preserves_metadata(
    default_input_struct: InputParameters, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = default_input_struct.evolve_input_structs(HII_DIM=2, DIM=4)
    expected = InitialConditions.new(inputs=inputs).arrays["hires_density"]
    values = np.zeros(expected.shape, dtype=np.float32)
    monkeypatch.setattr(InitialConditions, "compute", lambda self: self)
    result = inspect.unwrap(compute_initial_conditions)(
        inputs=inputs, initial_density=values
    )
    array = result.arrays["hires_density"]
    assert array.unit == expected.unit
    assert array.description == expected.description
    assert array.state.initialized
    assert not array.state.computed_in_mem
    np.testing.assert_array_equal(array.get_quantity().value, values)


@pytest.mark.parametrize("previous_computed", [False, True])
def test_global_reionization_preserves_metadata(
    default_input_struct: InputParameters,
    monkeypatch: pytest.MonkeyPatch,
    previous_computed: bool,
) -> None:
    inputs = default_input_struct.evolve_input_structs(HII_DIM=1, DIM=1)
    previous = IonizedBox.new(inputs=inputs, redshift=11)
    if previous_computed:
        previous.neutral_fraction = np.full((1, 1, 1), 0.8, dtype=np.float32)
    monkeypatch.setattr(cfuncs, "evaluate_Nion_z", lambda **kwargs: (0.25, None))
    monkeypatch.setattr(cfuncs, "compute_mturns", lambda **kwargs: (1e8, None))
    result = compute_global_reionization_at_z(
        redshift=10, inputs=inputs, previous_ionized_box=previous, spin_temp=None
    )
    expected = IonizedBox.new(inputs=inputs, redshift=10)
    for name in ("neutral_fraction", "ionisation_rate_G12", "z_reion"):
        array = result.arrays[name]
        assert array.unit == expected.arrays[name].unit
        assert array.description == expected.arrays[name].description
        assert array.get_quantity().dtype == np.float32
    np.testing.assert_array_equal(result.neutral_fraction, 0.75)
    np.testing.assert_array_equal(result.z_reion, -1)
    if previous_computed:
        assert result.ionisation_rate_G12.item() > 0
    else:
        np.testing.assert_array_equal(result.ionisation_rate_G12, 0)
