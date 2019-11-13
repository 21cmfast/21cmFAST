"""
Various tests of the initial_conditions() function and InitialConditions class.
"""

import pytest

import numpy as np

from py21cmfast import wrapper


@pytest.fixture(scope="module")  # call this fixture once for all tests in this module
def basic_init_box():
    return wrapper.initial_conditions(
        regenerate=True, write=False, user_params=wrapper.UserParams(HII_DIM=35)
    )


def test_box_shape(basic_init_box):
    """Test basic properties of the InitialConditions struct"""
    shape = (35, 35, 35)
    assert basic_init_box.lowres_density.shape == shape
    assert basic_init_box.lowres_vx.shape == shape
    assert basic_init_box.lowres_vy.shape == shape
    assert basic_init_box.lowres_vz.shape == shape
    assert basic_init_box.lowres_vx_2LPT.shape == shape
    assert basic_init_box.lowres_vy_2LPT.shape == shape
    assert basic_init_box.lowres_vz_2LPT.shape == shape
    assert basic_init_box.hires_density.shape == tuple(4 * s for s in shape)

    assert basic_init_box.cosmo_params == wrapper.CosmoParams()


def test_modified_cosmo():
    """Test using a modified cosmology"""
    cosmo = wrapper.CosmoParams(sigma_8=0.9)
    ic = wrapper.initial_conditions(cosmo_params=cosmo, regenerate=True, write=False)

    assert ic.cosmo_params == cosmo
    assert ic.cosmo_params.SIGMA_8 == cosmo.SIGMA_8


def test_transfer_function(basic_init_box):
    """Test using a modified transfer function"""
    ic = wrapper.initial_conditions(
        regenerate=True,
        write=False,
        random_seed=basic_init_box.random_seed,
        user_params=wrapper.UserParams(HII_DIM=35, POWER_SPECTRUM=5),
    )

    rmsnew = np.sqrt(np.mean(ic.hires_density ** 2))
    rmsdelta = np.sqrt(np.mean((ic.hires_density - basic_init_box.hires_density) ** 2))
    assert rmsdelta < rmsnew
    assert rmsnew > 0.0
    assert not np.allclose(ic.hires_density, basic_init_box.hires_density)


def test_relvels():
    """Test for relative velocity initial conditions"""
    ic = wrapper.initial_conditions(
        regenerate=True,
        write=False,
        random_seed=1,
        user_params=wrapper.UserParams(
            HII_DIM=100,
            DIM=300,
            BOX_LEN=300,
            POWER_SPECTRUM=5,
            USE_RELATIVE_VELOCITIES=True,
        ),
    )

    vcbrms = np.sqrt(np.mean(ic.hires_vcb ** 2))
    vcbavg = np.mean(ic.hires_vcb)

    vcbrms_lowres = np.sqrt(np.mean(ic.lowres_vcb ** 2))
    vcbavg_lowres = np.mean(ic.lowres_vcb)

    assert vcbrms > 25.0
    assert vcbrms < 35.0  # it should be about 30 km/s, so we check it is around it
    assert (
        vcbavg < 0.95 * vcbrms
    )  # the average should be 0.92*vrms, since it follows a maxwell boltzmann
    assert vcbavg > 0.90 * vcbrms
    assert vcbrms_lowres > 25.0  # we also test the lowres box
    assert vcbrms_lowres < 35.0
    assert vcbavg_lowres < 0.95 * vcbrms_lowres
    assert vcbavg_lowres > 0.90 * vcbrms_lowres
