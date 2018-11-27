"""
Unit tests for output structures
"""

import copy
import pickle

import pytest

from py21cmmc import InitialConditions  # An example of an output struct


@pytest.fixture(scope="module")
def ic():
    return InitialConditions()


def test_arrays_inited(ic):
    assert not ic.arrays_initialized


def test_pointer_fields(ic):  # TODO: this is probably good to implement for every output struct defined in code
    # Get list of fields before and after array initialisation
    d = copy.copy(list(ic.__dict__.keys()))
    print(d)
    ic._init_arrays()
    print(ic.__dict__.keys())
    print(d)
    new_names = [name for name in ic.__dict__ if name not in d]

    assert new_names
    assert all([n in ic._pointer_fields for n in new_names])

    # cstruct shouldn't be initialised,
    assert not ic.arrays_initialized

    ic._init_cstruct()
    assert ic.arrays_initialized


def test_non_existence(ic, tmpdirec):
    assert not ic.exists(direc=tmpdirec.strpath)


def test_writeability(ic):
    with pytest.raises(IOError):
        ic.write()


def test_readability(tmpdirec):
    # we update this one, so don't use the global one
    ic_ = InitialConditions(init=True)

    # fake it being filled
    ic_.filled = True

    ic_.write(direc=tmpdirec.strpath)

    ic2 = InitialConditions()

    assert ic2.exists(direc=tmpdirec.strpath)

    print("SEED: ", ic2._current_seed)
    ic2.read(direc=tmpdirec)

    assert repr(ic_) == repr(ic2)


def test_pickleability():
    ic_ = InitialConditions(init=True)
    ic_.filled = True
    s = pickle.dumps(ic_)

    ic2 = pickle.loads(s)
    assert repr(ic_) == repr(ic2)


def test_fname():
    ic1 = InitialConditions(user_params={"HII_DIM":1000})
    ic2 = InitialConditions(user_params={"HII_DIM":1000})
    assert ic1.filename != ic2.filename  # random seeds are different


def test_match_seed(tmpdirec):
    # we update this one, so don't use the global one
    ic_ = InitialConditions(init=True)

    # fake it being filled
    ic_.filled = True

    ic_.write(direc=tmpdirec.strpath)

    ic2 = InitialConditions(cosmo_params={"RANDOM_SEED": 1})
    with pytest.raises(IOError) as e_info:  # should not read in just anything if its random seed is set.
        ic2.read(direc=tmpdirec.strpath)
