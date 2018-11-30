"""
These are designed to be unit-tests of the wrapper functionality. They do not test for correctness of simulations,
but whether different parameter options work/don't work as intended.
"""
import numpy as np
import pytest

from py21cmmc import wrapper

REDSHIFT = 12

@pytest.fixture(scope="module")
def user_params():
    # Do a small box, for testing
    return wrapper.UserParams(HII_DIM=35, DIM=70)


@pytest.fixture(scope="module")
def perturb_field(user_params,tmpdirec):
    "A default perturb_field"
    return wrapper.perturb_field(
        redshift=REDSHIFT,
        regenerate=True,  # i.e. make sure we don't read it in.
        user_params=user_params,
        direc=tmpdirec.strpath
    )


def test_perturb_field_no_ic(user_params, perturb_field):
    "Run a perturb field without passing an init box"

    assert len(perturb_field.density) == perturb_field.user_params.HII_DIM == user_params.HII_DIM
    assert perturb_field.redshift == REDSHIFT
    assert np.sum(perturb_field.density) != 0


def test_pf_unnamed_param():
    "Try using an un-named parameter"
    with pytest.raises(TypeError):
        pf = wrapper.perturb_field(7)


def test_perturb_field_ic(user_params, perturb_field, tmpdirec):
    ic = wrapper.initial_conditions(user_params=user_params, regenerate=True, direc=tmpdirec.strpath)
    pf = wrapper.perturb_field(redshift=REDSHIFT, init_boxes=ic, regenerate=True, direc=tmpdirec.strpath)
    assert len(pf.density) == len(ic.lowres_density)
    assert pf.cosmo_params == ic.cosmo_params
    assert pf.user_params == ic.user_params
    assert np.sum(pf.density) != 0
    assert pf.cosmo_params is ic.cosmo_params
    assert pf.user_params is ic.user_params

    assert pf.user_params == perturb_field.user_params
    assert pf.cosmo_params.RANDOM_SEED != perturb_field.cosmo_params.RANDOM_SEED
    assert pf.cosmo_params == perturb_field.cosmo_params # FIXME: this passes, because "equality" here does not include random seed. this is unintuitive and should be fixed!!!!


def test_cache_exists(user_params, perturb_field, tmpdirec):
    pf = wrapper.PerturbedField(redshift=perturb_field.redshift, cosmo_params=perturb_field.cosmo_params,
                                user_params=user_params)

    assert pf.exists(tmpdirec.strpath)

    pf.read(tmpdirec.strpath)
    assert np.all(pf.density == perturb_field.density)
    assert pf.cosmo_params == perturb_field.cosmo_params


def test_pf_new_seed(perturb_field, tmpdirec):
    pf = wrapper.perturb_field(
        redshift=perturb_field.redshift,
        user_params=perturb_field.user_params,
        direc=tmpdirec.strpath,
        cosmo_params={"RANDOM_SEED":1}
    ) #note not passing cosmo params, because of seed.

    assert not np.all(pf.density == perturb_field.density)
    assert pf.cosmo_params.RANDOM_SEED != perturb_field.cosmo_params.RANDOM_SEED

def test_pf_regenerate(perturb_field, tmpdirec):
    pf = wrapper.perturb_field(
        redshift=perturb_field.redshift,
        user_params=perturb_field.user_params,
        direc=tmpdirec.strpath,
        regenerate=True
    ) #note not passing cosmo params, because of seed.

    assert not np.all(pf.density == perturb_field.density)
    assert pf.cosmo_params.RANDOM_SEED != perturb_field.cosmo_params.RANDOM_SEED


def test_ib_from_pf(perturb_field, tmpdirec):
    ib = wrapper.ionize_box(perturbed_field=perturb_field, direc=tmpdirec.strpath)
    assert ib.redshift == perturb_field.redshift
    assert ib.user_params == perturb_field.user_params
    assert ib.cosmo_params == perturb_field.cosmo_params
    assert ib.user_params is perturb_field.user_params
    assert ib.cosmo_params is perturb_field.cosmo_params


def test_ib_from_z(user_params, perturb_field, tmpdirec):
    ib = wrapper.ionize_box(redshift=perturb_field.redshift, user_params=user_params, direc=tmpdirec.strpath)
    assert ib.redshift == perturb_field.redshift
    assert ib.user_params == perturb_field.user_params
    assert ib.cosmo_params == perturb_field.cosmo_params
    assert ib.cosmo_params is not perturb_field.cosmo_params


def test_ib_override_z(perturb_field, tmpdirec):
    ib = wrapper.ionize_box(redshift=perturb_field.redshift+1, perturbed_field=perturb_field, direc=tmpdirec.strpath)
    assert ib.redshift == perturb_field.redshift
    assert ib.user_params == perturb_field.user_params
    assert ib.cosmo_params == perturb_field.cosmo_params

    assert ib.cosmo_params is perturb_field.cosmo_params # because the ib gets its cosmo_params directly from the passed perturb_field.
