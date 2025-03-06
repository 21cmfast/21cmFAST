"""Unit tests for output structures."""

import copy
import pickle

import attrs
import numpy as np
import pytest

from py21cmfast import (
    InitialConditions,  # An example of an output struct
    InputParameters,
    IonizedBox,
    OutputCache,
    PerturbedField,
    TsBox,
)
from py21cmfast.io import h5
from py21cmfast.wrapper import outputs as ox


@pytest.fixture
def init(default_input_struct: InputParameters):
    return InitialConditions.new(inputs=default_input_struct)


def test_readability(
    ic: InitialConditions, cache: OutputCache, default_input_struct: InputParameters
):
    ic2 = InitialConditions.new(inputs=default_input_struct)
    existing = cache.find_existing(ic2)

    assert existing is not None
    assert existing.exists()

    ic2 = cache.load(ic2)

    assert ic == ic2
    assert ic is not ic2


def test_different_seeds(
    init: InitialConditions,
    default_input_struct: InputParameters,
):
    ic2 = InitialConditions.new(
        inputs=default_input_struct.clone(
            random_seed=default_input_struct.random_seed + 1
        )
    )

    assert init is not ic2
    assert init != ic2

    # make sure we didn't inadvertantly set the random seed while doing any of this
    assert init.random_seed == default_input_struct.random_seed


def test_pickleability(default_input_struct: InputParameters):
    ic_ = InitialConditions.new(inputs=default_input_struct)
    s = pickle.dumps(ic_)

    ic2 = pickle.loads(s)
    assert repr(ic_) == repr(ic2)


def test_match_seed(cache: OutputCache, default_input_struct: InputParameters):
    ic2 = InitialConditions.new(inputs=default_input_struct.clone(random_seed=3))

    # This fails because we've set the seed and it's different to the existing one.
    with pytest.raises(IOError, match="No cache exists for"):
        cache.load(ic2)


def test_reading_purged(ic: InitialConditions):
    lowres_density = ic.get(ic.lowres_density)

    # Remove it from memory
    ic.purge()

    assert not ic.lowres_density.state.computed_in_mem
    assert ic.lowres_density.state.on_disk

    # But we can still get it.
    lowres_density_2 = ic.get(ic.lowres_density)

    assert ic.lowres_density.state.on_disk
    assert ic.lowres_density.state.computed_in_mem

    assert np.allclose(lowres_density_2, lowres_density)

    ic.load_all()


@pytest.mark.parametrize("struct", list(ox._ALL_OUTPUT_STRUCTS.values()))
def test_all_fields_exist(struct: ox.OutputStruct):
    cstruct = ox.StructWrapper(struct.__name__)

    this = attrs.fields_dict(struct)

    # Ensure that all fields in the cstruct are also defined on this class.
    for name in cstruct.pointer_fields:
        assert name in this
        assert this[name].type == ox.Array

    for name in cstruct.primitive_fields:
        assert name in this
