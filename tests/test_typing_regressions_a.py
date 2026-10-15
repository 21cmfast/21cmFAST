"""Regression tests for bugs found while adding type annotations to the input modules."""

import pytest

from py21cmfast.wrapper.inputs import (
    AstroParams,
    CosmoParams,
    CosmoTables,
    InputParameters,
    MatterOptions,
)


def test_scalar_node_redshifts():
    """A single (scalar) node redshift is converted to a one-element tuple.

    Previously, the converter called ``len()`` on the value before checking whether
    it was a sequence, so a scalar raised a TypeError.
    """
    inputs = InputParameters(random_seed=1, node_redshifts=6.0)
    assert inputs.node_redshifts == (6.0,)


def test_node_redshifts_sorted_descending():
    """Node redshifts are stored as a descending tuple of floats."""
    inputs = InputParameters(random_seed=1, node_redshifts=[6, 8.5, 7])
    assert inputs.node_redshifts == (8.5, 7.0, 6.0)


def test_is_compatible_with():
    """is_compatible_with previously raised an AttributeError (no merge_keys)."""
    inputs = InputParameters(random_seed=1)
    assert inputs.is_compatible_with(inputs)
    assert inputs.is_compatible_with(inputs.clone())
    assert not inputs.is_compatible_with(
        inputs.evolve_input_structs(HII_EFF_FACTOR=31.0)
    )
    assert not inputs.is_compatible_with(inputs.clone(random_seed=2))
    assert not inputs.is_compatible_with(AstroParams())


@pytest.mark.parametrize("cls", [CosmoParams, MatterOptions, AstroParams])
def test_new_with_instance_of_other_struct(cls):
    """InputStruct.new used to return an instance of a *different* struct unchanged."""
    other = MatterOptions() if cls is CosmoParams else CosmoParams()
    with pytest.raises(ValueError, match="Cannot instantiate"):
        cls.new(other)


def test_new_with_own_instance():
    """InputStruct.new with an instance of the class clones it with the updates."""
    params = MatterOptions(HMF="PS")
    new = MatterOptions.new(params, PERTURB_ON_HIGH_RES=True)
    assert new.HMF == "PS"
    assert new.PERTURB_ON_HIGH_RES


def test_cosmo_tables_new_with_other_type():
    """CosmoTables.new with a non-CosmoTables struct raises."""
    with pytest.raises(ValueError, match="Cannot instantiate"):
        CosmoTables.new(CosmoParams())
