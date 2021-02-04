"""
Unit tests for input structures
"""

import pytest

import pickle

from py21cmfast import AstroParams  # An example of a struct with defaults
from py21cmfast import CosmoParams, FlagOptions, UserParams, global_params


@pytest.fixture(scope="module")
def c():
    return CosmoParams(SIGMA_8=0.8)


def test_diff(c):
    """Ensure that the python dict has all fields"""
    d = CosmoParams(SIGMA_8=0.9)

    assert c is not d
    assert c != d
    assert hash(c) != hash(d)


def test_constructed_the_same(c):
    c2 = CosmoParams(SIGMA_8=0.8)

    assert c is not c2
    assert c == c2
    assert hash(c) == hash(c2)


def test_bad_construction(c):
    with pytest.raises(TypeError):
        CosmoParams(c, c)

    with pytest.raises(TypeError):
        CosmoParams(UserParams())

    with pytest.raises(TypeError):
        CosmoParams(1)


def test_warning_bad_params(caplog):
    CosmoParams(bad_param=1)
    assert (
        "The following parameters to CosmoParams are not supported: ['bad_param']"
        in caplog.text
    )


def test_constructed_from_itself(c):
    c3 = CosmoParams(c)

    assert c == c3
    assert c is not c3


def test_dynamic_variables():
    u = UserParams()
    assert u.DIM == 3 * u.HII_DIM

    u.update(DIM=200)

    assert u.DIM == 200


def test_clone():
    u = UserParams()
    v = u.clone()
    assert u == v


def test_repr(c):
    assert "SIGMA_8:0.8" in repr(c)


def test_pickle(c):
    # Make sure we can pickle/unpickle it.
    c4 = pickle.dumps(c)
    c4 = pickle.loads(c4)
    assert c == c4

    # Make sure the c data gets loaded fine.
    assert c4._cstruct.SIGMA_8 == c._cstruct.SIGMA_8


def test_self(c):
    c5 = CosmoParams(c.self)
    assert c5 == c
    assert c5.pystruct == c.pystruct
    assert c5.defining_dict == c.defining_dict
    assert (
        c5.defining_dict != c5.pystruct
    )  # not the same because the former doesn't include dynamic parameters.
    assert c5.self == c.self


def test_update():
    c = CosmoParams()
    c_pystruct = c.pystruct

    c.update(
        SIGMA_8=0.9
    )  # update c parameters. since pystruct as dynamically created, it is a new object each call.
    assert c_pystruct != c.pystruct


def test_c_structures(c):
    # See if the C structures are behaving correctly
    c2 = CosmoParams(SIGMA_8=0.8)

    assert c() != c2()
    assert (
        c() is c()
    )  # Re-calling should not re-make the object (object should have persistence)


def test_c_struct_update():
    c = CosmoParams()
    _c = c()
    c.update(SIGMA_8=0.8)
    assert _c != c()


def test_update_inhomo_reco(caplog):
    ap = AstroParams(R_BUBBLE_MAX=25)

    ap.update(INHOMO_RECO=True)

    msg = (
        "You are setting R_BUBBLE_MAX != 50 when INHOMO_RECO=True. "
        + "This is non-standard (but allowed), and usually occurs upon manual update of INHOMO_RECO"
    )

    ap.R_BUBBLE_MAX

    assert msg in caplog.text


def test_mmin():
    fo = FlagOptions(USE_MASS_DEPENDENT_ZETA=True)
    assert fo.M_MIN_in_Mass


def test_globals():
    orig = global_params.Z_HEAT_MAX

    with global_params.use(Z_HEAT_MAX=10.0):
        assert global_params.Z_HEAT_MAX == 10.0
        assert global_params._cobj.Z_HEAT_MAX == 10.0

    assert global_params.Z_HEAT_MAX == orig


def test_fcoll_on(caplog):

    f = UserParams(FAST_FCOLL_TABLES=True, USE_INTERPOLATION_TABLES=False)
    assert not f.FAST_FCOLL_TABLES
    assert (
        "You cannot turn on FAST_FCOLL_TABLES without USE_INTERPOLATION_TABLES"
        in caplog.text
    )
