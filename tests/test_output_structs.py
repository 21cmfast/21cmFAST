"""
Unit tests for output structures
"""

import copy
import pickle

import pytest

from py21cmmc import InitialConditions, PerturbedField, IonizedBox, TsBox, global_params  # An example of an output struct


@pytest.fixture(scope="module")
def ic():
    return InitialConditions()


def test_arrays_inited(ic):
    assert not ic.arrays_initialized


def test_pointer_fields_ic(ic):  # TODO: this is probably good to implement for every output struct defined in code
    # Get list of fields before and after array initialisation
    d = copy.copy(list(ic.__dict__.keys()))
    ic._init_arrays()
    new_names = [name for name in ic.__dict__ if name not in d]

    assert new_names
    assert all([n in ic.pointer_fields for n in new_names])

    # cstruct shouldn't be initialised,
    assert not ic.arrays_initialized

    ic._init_cstruct()
    assert ic.arrays_initialized


def test_pointer_fields_pf():
    # Get list of fields before and after array initialisation

    with pytest.raises(KeyError):
        pf = PerturbedField()

    pf = PerturbedField(redshift=7.0)

    d = copy.copy(list(pf.__dict__.keys()))
    pf._init_arrays()
    new_names = [name for name in pf.__dict__ if name not in d]

    assert new_names
    assert all([n in pf.pointer_fields for n in new_names])


def test_pointer_fields_ib():
    # Get list of fields before and after array initialisation
    with pytest.raises(KeyError):
        pf = IonizedBox()

    pf = IonizedBox(redshift=7.0)

    d = copy.copy(list(pf.__dict__.keys()))
    pf._init_arrays()
    new_names = [name for name in pf.__dict__ if name not in d]

    assert new_names
    assert all([n in pf.pointer_fields for n in new_names])


def test_pointer_fields_st():
    # Get list of fields before and after array initialisation
    with pytest.raises(KeyError):
        pf = TsBox()

    pf = TsBox(redshift=7.0)

    d = copy.copy(list(pf.__dict__.keys()))
    pf._init_arrays()
    new_names = [name for name in pf.__dict__ if name not in d]

    assert new_names
    assert all([n in pf.pointer_fields for n in new_names])


def test_non_existence(ic, tmpdirec):
    assert not ic.exists(direc=tmpdirec.strpath)


def test_writeability(ic):
    with pytest.raises(IOError):
        ic.write()


def test_readability(tmpdirec):
    # we update this one, so don't use the global one
    ic_ = InitialConditions(init=True)

    # TODO: fake it being filled (need to do both of the following to fool it. Actually, we *shouldn't* be able to
    # TODO: fool it at all, but hey.
    ic_.filled = True
    ic_.random_seed  # accessing random_seed actually creates a random seed.

    ic_.write(direc=tmpdirec.strpath)

    ic2 = InitialConditions()

    assert ic_._seedless_repr() == ic2._seedless_repr()  # without seeds, they are obviously exactly the same.

    assert ic2.exists(direc=tmpdirec.strpath)

    ic2.read(direc=tmpdirec.strpath)

    assert repr(ic_) == repr(ic2)  # they should be exactly the same.
    assert str(ic_) == str(ic2)  # their str is the same.
    assert hash(ic_) == hash(ic2)
    assert ic_ == ic2
    assert ic_ is not ic2

    # make sure we can't read it twice
    with pytest.raises(IOError):
        ic2.read(direc=tmpdirec.strpath)

def test_different_seeds(ic):
    ic2 = InitialConditions(random_seed=1)

    assert ic is not ic2
    assert ic != ic2
    assert repr(ic) != repr(ic2)
    assert ic._seedless_repr() == ic2._seedless_repr()

    assert ic._md5 == ic2._md5

    # make sure we didn't inadvertantly set the random seed while doing any of this
    assert ic._random_seed is None


def test_pickleability():
    ic_ = InitialConditions(init=True)
    ic_.filled = True
    ic_.random_seed

    s = pickle.dumps(ic_)

    ic2 = pickle.loads(s)
    assert repr(ic_) == repr(ic2)


def test_fname():
    ic1 = InitialConditions(user_params={"HII_DIM": 1000})
    ic2 = InitialConditions(user_params={"HII_DIM": 1000})

    # we didn't give them seeds, so can't access the filename attribute (it is undefined until a seed is set)
    with pytest.raises(AttributeError):
        assert ic1.filename != ic2.filename  # random seeds are different

    # *but* should be able to get a skeleton filename:
    assert ic1._fname_skeleton == ic2._fname_skeleton

    ic1.random_seed  # sets the random seed
    ic2.random_seed

    assert ic1.filename != ic2.filename  # random seeds should now be different
    assert ic1._fname_skeleton == ic2._fname_skeleton


def test_match_seed(tmpdirec):
    # we update this one, so don't use the global one
    ic_ = InitialConditions(init=True, random_seed=12)

    # fake it being filled
    ic_.filled = True
    ic_.random_seed

    ic_.write(direc=tmpdirec.strpath)

    ic2 = InitialConditions(random_seed=1)
    with pytest.raises(IOError):  # should not read in just anything if its random seed is set.
        ic2.read(direc=tmpdirec.strpath)


def test_bad_class_definition():
    class CustomInitialConditions(InitialConditions):
        _name = "InitialConditions"

        """
        A class containing all initial conditions boxes.
        """

        def _init_arrays(self):
            super()._init_arrays()

            # remove one of the arrays that needs to be defined.
            del self.hires_density

    with pytest.raises(AttributeError):
        c = CustomInitialConditions(init=True)


def test_pre_expose(ic):
    # haven't actually tried to read in or fill the array yet.
    with pytest.raises(Exception):
        ic._expose()


def test_bad_write():
    ic = InitialConditions()
    ic.filled=True

    # no random seed yet so shouldn't be able to write.
    with pytest.raises(ValueError):
        ic.write()


def test_global_params_keys():
    assert "HII_FILTER" in global_params.keys()
