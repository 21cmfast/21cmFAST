"""
Unit tests for output structures
"""

import pytest

import copy
import pickle

from py21cmfast import InitialConditions  # An example of an output struct
from py21cmfast import IonizedBox, PerturbedField, TsBox, global_params


@pytest.fixture(scope="function")
def init(default_user_params):
    return InitialConditions(user_params=default_user_params)


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


def test_readability(test_direc, default_user_params):
    # we update this one, so don't use the global one
    ic_ = InitialConditions(init=True, user_params=default_user_params)

    # TODO: fake it being filled (need to do both of the following to fool it.
    # TODO: Actually, we *shouldn't* be able to fool it at all, but hey.
    ic_.filled = True
    ic_.random_seed  # accessing random_seed actually creates a random seed.

    ic_.write(direc=test_direc)
    ic2 = InitialConditions(user_params=default_user_params)

    # without seeds, they are obviously exactly the same.
    assert ic_._seedless_repr() == ic2._seedless_repr()

    assert ic2.exists(direc=test_direc)

    ic2.read(direc=test_direc)

    assert repr(ic_) == repr(ic2)  # they should be exactly the same.
    assert str(ic_) == str(ic2)  # their str is the same.
    assert hash(ic_) == hash(ic2)
    assert ic_ == ic2
    assert ic_ is not ic2


def test_different_seeds(init, default_user_params):
    ic2 = InitialConditions(random_seed=2, user_params=default_user_params)

    assert init is not ic2
    assert init != ic2
    assert repr(init) != repr(ic2)
    assert init._seedless_repr() == ic2._seedless_repr()

    assert init._md5 == ic2._md5

    # make sure we didn't inadvertantly set the random seed while doing any of this
    assert init._random_seed is None


def test_pickleability(default_user_params):
    ic_ = InitialConditions(init=True, user_params=default_user_params)
    ic_.filled = True
    ic_.random_seed

    s = pickle.dumps(ic_)

    ic2 = pickle.loads(s)
    assert repr(ic_) == repr(ic2)


def test_fname(default_user_params):
    ic1 = InitialConditions(user_params=default_user_params)
    ic2 = InitialConditions(user_params=default_user_params)

    # we didn't give them seeds, so can't access the filename attribute
    # (it is undefined until a seed is set)
    with pytest.raises(AttributeError):
        assert ic1.filename != ic2.filename  # random seeds are different

    # *but* should be able to get a skeleton filename:
    assert ic1._fname_skeleton == ic2._fname_skeleton

    ic1.random_seed  # sets the random seed
    ic2.random_seed

    assert ic1.filename != ic2.filename  # random seeds should now be different
    assert ic1._fname_skeleton == ic2._fname_skeleton


def test_match_seed(test_direc, default_user_params):
    # we update this one, so don't use the global one
    ic_ = InitialConditions(init=True, random_seed=12, user_params=default_user_params)

    # fake it being filled
    ic_.filled = True
    ic_.random_seed

    ic_.write(direc=test_direc)

    ic2 = InitialConditions(random_seed=1, user_params=default_user_params)

    # should not read in just anything if its random seed is set.
    with pytest.raises(IOError):
        ic2.read(direc=test_direc)


def test_bad_class_definition(default_user_params):
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
        CustomInitialConditions(init=True, user_params=default_user_params)


def test_pre_expose(init):
    # haven't actually tried to read in or fill the array yet.
    with pytest.raises(Exception):
        init._expose()


def test_bad_write(init):
    init.filled = True

    # no random seed yet so shouldn't be able to write.
    with pytest.raises(ValueError):
        init.write()


def test_global_params_keys():
    assert "HII_FILTER" in global_params.keys()
