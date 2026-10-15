"""Regression tests for bugs found while adding type annotations to the drivers."""

import inspect

import numpy as np
import pytest

import py21cmfast as p21c
from py21cmfast.drivers._global_initialization import init_c_state
from py21cmfast.drivers._param_config import (
    _OutputStructComputationInspect,
    single_field_func,
)
from py21cmfast.drivers.global_evolution import GlobalEvolution


def test_negative_mean_initial_density_warns(default_input_struct):
    """A large *negative* mean of initial_density must warn, just like a positive one.

    Previously the check was ``np.abs(mean > 1e-3)``, i.e. the absolute value of a
    boolean, so negative means never warned.
    """
    dim = default_input_struct.simulation_options.DIM
    with pytest.warns(UserWarning, match="initial_density has mean -1.0"):
        p21c.compute_initial_conditions(
            inputs=default_input_struct,
            initial_density=-np.ones((dim,) * 3),
        )


def test_scalar_initial_density_for_multi_cell_box(default_input_struct):
    """A scalar initial_density for a multi-cell box gives the shape error.

    Previously this raised ``AttributeError: 'float' object has no attribute 'mean'``.
    """
    with pytest.raises(
        ValueError,
        match="The shape of your high resolution initial_density is not consistent",
    ):
        p21c.compute_initial_conditions(
            inputs=default_input_struct, initial_density=0.0
        )


def test_global_evolution_without_quantities(default_input_struct, tmp_path):
    """A GlobalEvolution built without quantities can be compared and saved.

    Previously ``quantities`` defaulted to None, and both ``==`` and ``save`` raised
    AttributeError.
    """
    ge = GlobalEvolution(inputs=default_input_struct)
    assert ge.quantities == {}
    assert ge == GlobalEvolution(inputs=default_input_struct)

    ge.save(tmp_path / "global.h5")
    assert (tmp_path / "global.h5").exists()


def test_get_inputs_rejects_non_input_parameters():
    """A non-InputParameters 'inputs' with no OutputStructs is an error.

    Previously the bogus object was returned as if it were the InputParameters.
    """
    with pytest.raises(ValueError, match="No parameter 'inputs' given"):
        _OutputStructComputationInspect._get_inputs({"inputs": "not inputs"})


def test_redshift_consistency_without_current_redshift(default_input_struct_ts):
    """Only previous-redshift outputs, and no current redshift, must not crash.

    Previously the previous-redshift check compared node redshifts with None, raising
    a TypeError.
    """

    def func(*, previous_perturbed_field: p21c.PerturbedField) -> p21c.IonizedBox: ...

    inspector = _OutputStructComputationInspect(func)
    pf = p21c.PerturbedField.new(
        inputs=default_input_struct_ts,
        redshift=default_input_struct_ts.node_redshifts[0],
    )
    inspector.ensure_redshift_consistency(None, {"previous_perturbed_field": pf})


def test_cache_kwargs_are_part_of_the_signature():
    """The cache-control kwargs consumed by @single_field_func are in the signature."""
    for func in (
        p21c.compute_initial_conditions,
        p21c.perturb_field,
        p21c.compute_ionization_field,
        p21c.brightness_temperature,
    ):
        params = inspect.signature(func).parameters
        assert {"cache", "regenerate", "write"} <= params.keys()


def test_decorators_reject_positional_arguments(default_input_struct):
    """The decorated functions only take keyword arguments."""

    @init_c_state(broadcast_inputs=True)
    def func(inputs: p21c.InputParameters) -> None:
        pass

    with pytest.raises(TypeError, match="only accepts keyword arguments"):
        func(default_input_struct)

    sff = single_field_func(p21c.compute_initial_conditions.__wrapped__)
    with pytest.raises(TypeError, match="only accepts keyword arguments"):
        sff(default_input_struct)
