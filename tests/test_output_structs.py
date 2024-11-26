"""
Unit tests for output structures
"""

import pytest

import copy
import numpy as np
import pickle

from py21cmfast import InitialConditions  # An example of an output struct
from py21cmfast import IonizedBox, PerturbedField, TsBox, global_params


@pytest.fixture(scope="function")
def init(default_input_struct):
    return InitialConditions(inputs=default_input_struct)


@pytest.mark.parametrize("cls", [InitialConditions, PerturbedField, IonizedBox, TsBox])
def test_pointer_fields(cls, default_input_struct):
    with pytest.raises(KeyError):
        cls()

    inst = cls(inputs=default_input_struct)

    # Get list of fields before and after array initialisation
    d = copy.copy(list(inst.__dict__.keys()))
    inst._init_arrays()
    new_names = [name for name in inst.__dict__ if name not in d]

    assert new_names
    assert all(n in inst.struct.pointer_fields for n in new_names)


def test_non_existence(init, test_direc):
    assert not init.exists(direc=test_direc)


def test_writeability(init):
    """init is not initialized and therefore can't write yet."""
    with pytest.raises(IOError):
        init.write()


def test_readability(ic, tmpdirec, default_input_struct):
    ic2 = InitialConditions(inputs=default_input_struct)

    # without seeds, they are obviously exactly the same.
    assert ic._seedless_repr() == ic2._seedless_repr()

    assert ic2.exists(direc=tmpdirec)

    ic2.read(direc=tmpdirec)

    assert repr(ic) == repr(ic2)  # they should be exactly the same.
    assert str(ic) == str(ic2)  # their str is the same.
    assert hash(ic) == hash(ic2)
    assert ic == ic2
    assert ic is not ic2


def test_different_seeds(init, default_input_struct, default_seed):
    ic2 = InitialConditions(inputs=default_input_struct.clone(random_seed=2))

    assert init is not ic2
    assert init != ic2
    assert repr(init) != repr(ic2)
    assert init._seedless_repr() == ic2._seedless_repr()

    assert init._md5 == ic2._md5

    # make sure we didn't inadvertantly set the random seed while doing any of this
    assert init.random_seed == default_seed


def test_pickleability(default_input_struct):
    ic_ = InitialConditions(inputs=default_input_struct)
    ic_.filled = True

    s = pickle.dumps(ic_)

    ic2 = pickle.loads(s)
    assert repr(ic_) == repr(ic2)


def test_fname(default_input_struct):
    ic1 = InitialConditions(inputs=default_input_struct)
    ic2 = InitialConditions(inputs=default_input_struct.clone(random_seed=2))

    assert ic1.filename != ic2.filename  # random seeds should now be different
    assert ic1._fname_skeleton == ic2._fname_skeleton


def test_match_seed(tmpdirec, default_input_struct):
    ic2 = InitialConditions(inputs=default_input_struct.clone(random_seed=3))

    # This fails because we've set the seed and it's different to the existing one.
    with pytest.raises(IOError):
        ic2.read(direc=tmpdirec)


def test_bad_class_definition(default_input_struct):
    class CustomInitialConditions(InitialConditions):
        """
        A class containing all initial conditions boxes.
        """

        def __init__(self):
            super.__init__(self)
            self._name = "InitialConditions"

        def _get_box_structures(self):
            out = super()._get_box_structures()
            out["unknown_key"] = (1, 1, 1)
            return out

    with pytest.raises(TypeError):
        CustomInitialConditions(inputs=default_input_struct)


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
