"""
Various tests of the initial_conditions() function and InitialConditions class.
"""

import pytest
from py21cmmc import wrapper


@pytest.fixture(scope="module") # call this fixture once for all tests in this module
def basic_init_box():
    return wrapper.initial_conditions(regenerate=True, write=False, user_params=wrapper.UserParams(HII_DIM=35))


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
    assert basic_init_box.hires_density.shape == tuple([4*s for s in shape])

    assert basic_init_box.cosmo_params == wrapper.CosmoParams()


def test_modified_cosmo():
    """Test using a modified cosmology"""
    cosmo = wrapper.CosmoParams(sigma_8=0.9)
    ic = wrapper.initial_conditions(cosmo_params=cosmo, regenerate=True, write=False)

    assert ic.cosmo_params == cosmo
    assert ic.cosmo_params.SIGMA_8 == cosmo.SIGMA_8

