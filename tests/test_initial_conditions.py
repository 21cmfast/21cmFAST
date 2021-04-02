"""
Various tests of the initial_conditions() function and InitialConditions class.
"""
import pytest

import numpy as np
from multiprocessing import cpu_count

from py21cmfast import wrapper


def test_box_shape(ic):
    """Test basic properties of the InitialConditions struct"""
    shape = (35, 35, 35)
    hires_shape = tuple(2 * s for s in shape)
    assert ic.lowres_density.shape == shape
    assert ic.lowres_vx.shape == shape
    assert ic.lowres_vy.shape == shape
    assert ic.lowres_vz.shape == shape
    assert ic.lowres_vx_2LPT.shape == shape
    assert ic.lowres_vy_2LPT.shape == shape
    assert ic.lowres_vz_2LPT.shape == shape
    assert ic.hires_density.shape == hires_shape

    assert ic.hires_vx.shape == hires_shape
    assert ic.hires_vy.shape == hires_shape
    assert ic.hires_vz.shape == hires_shape
    assert ic.hires_vx_2LPT.shape == hires_shape
    assert ic.hires_vy_2LPT.shape == hires_shape
    assert ic.hires_vz_2LPT.shape == hires_shape

    assert not hasattr(ic, "lowres_vcb")

    assert ic.cosmo_params == wrapper.CosmoParams()


def test_modified_cosmo(ic):
    """Test using a modified cosmology"""
    cosmo = wrapper.CosmoParams(SIGMA_8=0.9)
    ic2 = wrapper.initial_conditions(
        cosmo_params=cosmo,
        user_params=ic.user_params,
    )

    assert ic2.cosmo_params != ic.cosmo_params
    assert ic2.cosmo_params == cosmo
    assert ic2.cosmo_params.SIGMA_8 == cosmo.SIGMA_8


def test_transfer_function(ic, default_user_params):
    """Test using a modified transfer function"""
    user_params = default_user_params.clone(POWER_SPECTRUM=5)
    ic2 = wrapper.initial_conditions(
        random_seed=ic.random_seed,
        user_params=user_params,
    )

    rmsnew = np.sqrt(np.mean(ic2.hires_density ** 2))
    rmsdelta = np.sqrt(np.mean((ic2.hires_density - ic.hires_density) ** 2))
    assert rmsdelta < rmsnew
    assert rmsnew > 0.0
    assert not np.allclose(ic2.hires_density, ic.hires_density)


def test_relvels():
    """Test for relative velocity initial conditions"""
    ic = wrapper.initial_conditions(
        random_seed=1,
        user_params=wrapper.UserParams(
            HII_DIM=100,
            DIM=300,
            BOX_LEN=300,
            POWER_SPECTRUM=5,
            USE_RELATIVE_VELOCITIES=True,
            N_THREADS=cpu_count(),  # To make this one a bit faster.
        ),
    )

    vcbrms_lowres = np.sqrt(np.mean(ic.lowres_vcb ** 2))
    vcbavg_lowres = np.mean(ic.lowres_vcb)

    # we test the lowres box
    # rms should be about 30 km/s for LCDM, so we check it is finite and not far off
    # the average should be 0.92*vrms, since it follows a maxwell boltzmann
    assert vcbrms_lowres > 20.0
    assert vcbrms_lowres < 40.0
    assert vcbavg_lowres < 0.97 * vcbrms_lowres
    assert vcbavg_lowres > 0.88 * vcbrms_lowres
