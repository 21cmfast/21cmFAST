"""
Unit tests for input structures
"""

import pickle

import pytest

from py21cmmc import CosmoParams  # An example of a struct with defaults


@pytest.fixture(scope="module")
def c():
    return CosmoParams(SIGMA_8=0.8)


def test_diff(c):
    "Ensure that the python dict has all fields"
    d = CosmoParams(SIGMA_8=0.9)

    assert c is not d
    assert c != d
    assert hash(c) != hash(d)


def test_constructed_the_same(c):
    c2 = CosmoParams(SIGMA_8=0.8)

    assert c is not c2
    assert c == c2
    assert hash(c) == hash(c2)


def test_constructed_from_itself(c):
    c3 = CosmoParams(c)

    assert c == c3
    assert c is not c3


def test_repr(c):
    assert "SIGMA_8:0.8" in repr(c)


def test_pickle(c):
    # Make sure we can pickle/unpickle it.
    c4 = pickle.dumps(c)
    assert c == pickle.loads(c4)


def test_self(c):
    c5 = CosmoParams(c.self)
    assert c5 == c

    assert c5.pystruct != c.pystruct  # These shouldn't be the same since the RANDOM_SEED is chosen randomly
    assert c5.defining_dict == c.defining_dict  # these should be the same as they omit the random seed.


def test_update():
    c = CosmoParams()
    c_pystruct = c.pystruct

    c.update(RANDOM_SEED=None)  # update random seed
    assert c_pystruct != c.pystruct


def test_c_structures(c):
    # See if the C structures are behaving correctly
    c2 = CosmoParams(SIGMA_8=0.8, RANDOM_SEED=c.RANDOM_SEED)

    assert c() != c2()  # Even with same random seed, these shouldn't be the same
    assert c() is c()  # Re-calling should not re-make the object (object should have persistence)


def test_c_struct_update():
    c = CosmoParams()
    _c = c()
    c.update(RANDOM_SEED=None)
    assert _c != c()
