"""
Unit tests for output structures
"""

import pytest

import copy
import numpy as np
import pickle

from py21cmfast import InitialConditions  # An example of an output struct
from py21cmfast import IonizedBox, PerturbedField, TsBox, global_params


@pytest.fixture(scope="module")
def input_struct_noseed(default_input_struct):
    return default_input_struct.evolve(random_seed=None)


@pytest.fixture(scope="function")
def init(input_struct_noseed):
    return InitialConditions(inputs=input_struct_noseed)


@pytest.mark.parametrize("cls", [InitialConditions, PerturbedField, IonizedBox, TsBox])
def test_pointer_fields(cls):
    if cls is InitialConditions:
        inst = cls()
    else:
        with pytest.raises(KeyError):
            cls()

        inst = cls(redshift=7.0)

    # Get list of fields before and after array initialisation
    d = copy.copy(list(inst.__dict__.keys()))
    inst._init_arrays()
    new_names = [name for name in inst.__dict__ if name not in d]

    assert new_names
    assert all(n in inst.pointer_fields for n in new_names)


def test_non_existence(init, test_direc):
    assert not init.exists(direc=test_direc)


def test_writeability(init):
    """init is not initialized and therefore can't write yet."""
    with pytest.raises(IOError):
        init.write()


def test_readability(ic, tmpdirec, input_struct_noseed):
    ic2 = InitialConditions(inputs=input_struct_noseed)

    # without seeds, they are obviously exactly the same.
    assert ic._seedless_repr() == ic2._seedless_repr()

    assert ic2.exists(direc=tmpdirec)

    ic2.read(direc=tmpdirec)

    assert repr(ic) == repr(ic2)  # they should be exactly the same.
    assert str(ic) == str(ic2)  # their str is the same.
    assert hash(ic) == hash(ic2)
    assert ic == ic2
    assert ic is not ic2


def test_different_seeds(init, input_struct_noseed):
    ic2 = InitialConditions(inputs=input_struct_noseed.evolve(random_seed=2))

    assert init is not ic2
    assert init != ic2
    assert repr(init) != repr(ic2)
    assert init._seedless_repr() == ic2._seedless_repr()

    assert init._md5 == ic2._md5

    # make sure we didn't inadvertantly set the random seed while doing any of this
    assert init._random_seed is None


def test_pickleability(input_struct_noseed):
    # TODO: remove init kwarg which does nothing?
    ic_ = InitialConditions(init=True, inputs=input_struct_noseed)
    ic_.filled = True
    ic_.random_seed

    s = pickle.dumps(ic_)

    ic2 = pickle.loads(s)
    assert repr(ic_) == repr(ic2)


def test_fname(input_struct_noseed):
    ic1 = InitialConditions(inputs=input_struct_noseed)
    ic2 = InitialConditions(inputs=input_struct_noseed)

    # we didn't give them seeds, so can't access the filename attribute
    # (it is undefined until a seed is set)
    with pytest.raises(AttributeError):
        assert ic1.filename != ic2.filename

    # *but* should be able to get a skeleton filename:
    assert ic1._fname_skeleton == ic2._fname_skeleton

    ic1.random_seed  # sets the random seed
    ic2.random_seed

    assert ic1.filename != ic2.filename  # random seeds should now be different
    assert ic1._fname_skeleton == ic2._fname_skeleton


def test_match_seed(tmpdirec, input_struct_noseed):
    ic2 = InitialConditions(inputs=input_struct_noseed.evolve(random_seed=3))

    # This fails because we've set the seed and it's different to the existing one.
    with pytest.raises(IOError):
        ic2.read(direc=tmpdirec)


def test_bad_class_definition(input_struct_noseed):
    class CustomInitialConditions(InitialConditions):
        _name = "InitialConditions"

        """
        A class containing all initial conditions boxes.
        """

        def _get_box_structures(self):
            out = super()._get_box_structures()
            out["unknown_key"] = (1, 1, 1)
            return out

    with pytest.raises(TypeError):
        CustomInitialConditions(inputs=input_struct_noseed)


def test_bad_write(init):
    # no random seed yet so shouldn't be able to write.
    with pytest.raises(IOError):
        init.write()


def test_global_params_keys():
    assert "HII_FILTER" in global_params.keys()


def test_reading_purged(ic: InitialConditions):
    lowres_density = ic.lowres_density

    # Remove it from memory
    ic.purge()

    assert "lowres_density" not in ic.__dict__
    assert ic._array_state["lowres_density"].on_disk
    assert not ic._array_state["lowres_density"].computed_in_mem

    # But we can still get it.
    lowres_density_2 = ic.lowres_density

    assert ic._array_state["lowres_density"].on_disk
    assert ic._array_state["lowres_density"].computed_in_mem

    assert np.allclose(lowres_density_2, lowres_density)

    ic.load_all()
