"""Regression tests for issues found while adding type annotations to the wrapper.

These cover `wrapper/outputs.py`, `wrapper/arrays.py` and `wrapper/structs.py`.
"""

import attrs
import numpy as np
import pytest

from py21cmfast import InputParameters
from py21cmfast.wrapper.arrays import Array
from py21cmfast.wrapper.outputs import HaloCatalog, InitialConditions
from py21cmfast.wrapper.structs import StructWrapper


def test_trim_to_n_halos_returns_self(default_input_struct: InputParameters):
    """`trim_to_n_halos` is annotated as returning the catalog, but returned None."""
    hc = HaloCatalog.new(inputs=default_input_struct, redshift=10.0, buffer_size=10)
    hc._init_arrays()
    hc.n_halos = 3

    assert hc.trim_to_n_halos() is hc
    assert hc.buffer_size == 3
    assert hc.halo_masses.shape == (3,)
    assert hc.halo_coords.shape == (3, 3)


def test_halo_catalog_without_desc_redshift(default_input_struct: InputParameters):
    """A HaloCatalog made without `desc_redshift` used to fail obscurely.

    `get_required_input_arrays` compared None with 0 (a TypeError), and C would have
    been passed None for a double. Now the grids are (conservatively) required, and
    `compute` explains what is missing.
    """
    inputs = default_input_struct.evolve_input_structs(SOURCE_MODEL="CHMF-SAMPLER")
    hc = HaloCatalog.new(inputs=inputs, redshift=10.0, buffer_size=1)
    assert hc.desc_redshift is None

    ics = InitialConditions.new(inputs=inputs)
    assert hc.get_required_input_arrays(ics) == ["hires_density", "lowres_density"]

    with pytest.raises(ValueError, match="desc_redshift must be set"):
        hc.compute(ics=ics, descendant_halos=HaloCatalog.dummy())


def test_array_value_validator():
    """The `_value` validator (no longer a decorated method) still checks the value."""
    with pytest.raises(ValueError, match="Shape mismatch"):
        Array((2, 2), dtype=np.float32, value=np.zeros((3, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Data type mismatch"):
        Array((2, 2), dtype=np.float32, value=np.zeros((2, 2), dtype=np.float64))

    arr = Array((2, 2), dtype=np.float32, value=np.ones((2, 2), dtype=np.float32))
    assert arr._value is not None


def test_struct_wrapper_name_default():
    """The struct name (no longer a decorated default) defaults to the class name."""

    @attrs.define(slots=False)
    class InitialConditions(StructWrapper):
        pass

    wrapper = InitialConditions()
    assert wrapper._name == "InitialConditions"
    assert wrapper.cstruct is not None
    assert "lowres_density" in wrapper.pointer_fields

    named = StructWrapper(name="InitialConditions")
    assert named._name == "InitialConditions"
    assert named.fieldnames == wrapper.fieldnames
