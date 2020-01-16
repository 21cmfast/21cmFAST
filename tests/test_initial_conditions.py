"""
Various tests of the initial_conditions() function and InitialConditions class.
"""

import pytest

import numpy as np

from py21cmfast import wrapper
import pickle
import numpy as np
from powerbox.tools import get_power
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot

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
    assert basic_init_box.hires_density.shape == tuple(3 * s for s in shape)

    assert basic_init_box.cosmo_params == wrapper.CosmoParams()


def test_modified_cosmo():
    """Test using a modified cosmology"""
    cosmo = wrapper.CosmoParams(sigma_8=0.9)
    ic = wrapper.initial_conditions(
        cosmo_params=cosmo,
        regenerate=True,
        write=False,
        user_params={"HII_DIM": 35, "BOX_LEN": 70},
    )

    assert ic.cosmo_params == cosmo
    assert ic.cosmo_params.SIGMA_8 == cosmo.SIGMA_8


def test_transfer_function(basic_init_box):
    """Test using a modified transfer function"""
    ic = wrapper.initial_conditions(
        regenerate=True,
        write=False,
        random_seed=basic_init_box.random_seed,
        user_params=wrapper.UserParams(HII_DIM=35, BOX_LEN=70, POWER_SPECTRUM=5),
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

    # the average should be 0.92*vrms, since it follows a maxwell boltzmann
    assert vcbavg < 0.95 * vcbrms
    assert vcbavg > 0.90 * vcbrms

    # we also test the lowres box. Increase its error margin because its low res.
    assert vcbrms_lowres > 20.0
    assert vcbrms_lowres < 40.0
    assert vcbavg_lowres < 0.97 * vcbrms_lowres
    assert vcbavg_lowres > 0.88 * vcbrms_lowres


# def test_high_res_density(basic_init_box):
#     """Test the high resolution density field mode (added by C.Watkinson) """

#     # I WOULD SUGGEST ADDING A POWER SPECTRUM CHECK TO THIS AS WELL

#     RNG_SEED = 124375

#     ic_hi = wrapper.initial_conditions(user_params=wrapper.UserParams(MOVE_DENSITY_HIGH_RES=True), regenerate=True, write=False, random_seed=RNG_SEED)
#     assert ic_hi.user_params.MOVE_DENSITY_HIGH_RES == True

#     # Compare hi res density field with hard coded one
#     f_in = "tests/DensityFieldTestData/init_hires_density_hist_bins.npz"
#     f = open( f_in, 'rb' )
#     hist_hc = pickle.load( f )
#     f.close()
#     hist, bins = np.histogram(ic_hi.hires_density, bins=100, density=True)
#     hist_test = [hist, bins]
#     for i, histo in enumerate(hist_test[0]):
#         assert histo == hist_hc[0][i], "hi-res density histograms do not match for MOVE_DENSITY_HIGH_RES=True"

#     # Compare velocity hist with hardcoded hist
#     f_in = "tests/DensityFieldTestData/init_hires_vx_hist_bins.npz"
#     f = open( f_in, 'rb' )
#     hist_hc = pickle.load( f )
#     f.close()
#     hist, bins = np.histogram(ic_hi.hires_vx, bins=100, density=True)
#     hist_test = [hist, bins]
#     for i, histo in enumerate(hist_test[0]):
#         assert histo == hist_hc[0][i], "hi-res vx histograms do not match for MOVE_DENSITY_HIGH_RES=True"

#     f_in = "tests/DensityFieldTestData/init_hires_vy_hist_bins.npz"
#     f = open( f_in, 'rb' )
#     hist_hc = pickle.load( f )
#     f.close()
#     hist, bins = np.histogram(ic_hi.hires_vy, bins=100, density=True)
#     hist_test = [hist, bins]
#     for i, histo in enumerate(hist_test[0]):
#         assert histo == hist_hc[0][i], "hi-res vy histograms do not match for MOVE_DENSITY_HIGH_RES=True"

#     f_in = "tests/DensityFieldTestData/init_hires_vz_hist_bins.npz"
#     f = open( f_in, 'rb' )
#     hist_hc = pickle.load( f )
#     f.close()
#     hist, bins = np.histogram(ic_hi.hires_vz, bins=100, density=True)
#     hist_test = [hist, bins]
#     for i, histo in enumerate(hist_test[0]):
#         assert histo == hist_hc[0][i], "hi-res vz histograms do not match for MOVE_DENSITY_HIGH_RES=True"

#     pf_hi = wrapper.perturb_field(user_params=wrapper.UserParams(MOVE_DENSITY_HIGH_RES=True), init_boxes=ic_hi, redshift=6.0, random_seed=RNG_SEED)
#     assert pf_hi.user_params.MOVE_DENSITY_HIGH_RES == True

#     # Test low res density field
#     f_in = "tests/DensityFieldTestData/peturbfield_lowres_density_hist_bins.npz"
#     f = open( f_in, 'rb' )
#     hist_hc = pickle.load( f )
#     f.close()
#     hist, bins = np.histogram(pf_hi.density, bins=100, density=True)
#     hist_test = [hist, bins]
#     for i, histo in enumerate(hist_test[0]):
#         assert histo == hist_hc[0][i], "low-res (post-perturb) density histograms do not match for MOVE_DENSITY_HIGH_RES=True"

#     # Test the power spectrum from the final high resolution
#     f_in = "tests/DensityFieldTestData/k_Pk_hires_perturb.npz"
#     f = open( f_in, 'rb' )
#     k_Pk_hc = pickle.load( f )
#     f.close()

#     res = get_power(
#     		pf_hi.density,
#     		boxlength=pf_hi.user_params.BOX_LEN,
#     		bins=None, bin_ave=False, get_variance=False, log_bins=True
#     	)
#     res = list(res)
#     k = res[1]
#     k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
#     res[1] = k
#     Pk_obs = ( res[0] * k**3 / (2 * np.pi ** 2) )

#     for i, Pk in enumerate(Pk_obs):
#         if not(np.isnan(Pk)):
#             assert Pk == k_Pk_hc[1][i], "low-res (post-perturb) density power spectrum does not match for MOVE_DENSITY_HIGH_RES=True"


#     #---------- TEST FOR MOVE_DENSITY_HIGH_RES=False (i.e. old density field generation method) ------------
#     ic_lo = wrapper.initial_conditions(user_params=wrapper.UserParams(MOVE_DENSITY_HIGH_RES=False), regenerate=True, write=False, random_seed=RNG_SEED)
#     assert ic_lo.user_params.MOVE_DENSITY_HIGH_RES == False

#     # Test low res velocity field
#     f_in = "tests/DensityFieldTestData/init_lowres_density_hist_bins.npz"
#     f = open( f_in, 'rb' )
#     hist_hc = pickle.load( f )
#     f.close()
#     hist, bins = np.histogram(ic_lo.hires_density, bins=100, density=True)
#     hist_test = [hist, bins]
#     for i, histo in enumerate(hist_test[0]):
#         assert histo == hist_hc[0][i], "hi-res density histograms do not match for MOVE_DENSITY_HIGH_RES=False"


#     # Test low res vx field
#     f_in = "tests/DensityFieldTestData/init_lowres_vx_hist_bins.npz"
#     f = open( f_in, 'rb' )
#     hist_hc = pickle.load( f )
#     f.close()
#     hist, bins = np.histogram(ic_lo.lowres_vx, bins=100, density=True)
#     hist_test = [hist, bins]
#     for i, histo in enumerate(hist_test[0]):
#         assert histo == hist_hc[0][i], "low-res vx histograms do not match for MOVE_DENSITY_HIGH_RES=False"

#     # Test low res vy field
#     f_in = "tests/DensityFieldTestData/init_lowres_vy_hist_bins.npz"
#     f = open( f_in, 'rb' )
#     hist_hc = pickle.load( f )
#     f.close()
#     hist, bins = np.histogram(ic_lo.lowres_vy, bins=100, density=True)
#     hist_test = [hist, bins]
#     for i, histo in enumerate(hist_test[0]):
#         assert histo == hist_hc[0][i], "low-res vy histograms do not match for MOVE_DENSITY_HIGH_RES=False"


#     # Test low res vy field
#     f_in = "tests/DensityFieldTestData/init_lowres_vz_hist_bins.npz"
#     f = open( f_in, 'rb' )
#     hist_hc = pickle.load( f )
#     f.close()
#     hist, bins = np.histogram(ic_lo.lowres_vz, bins=100, density=True)
#     hist_test = [hist, bins]
#     for i, histo in enumerate(hist_test[0]):
#         assert histo == hist_hc[0][i], "low-res vz histograms do not match for MOVE_DENSITY_HIGH_RES=False"

#     pf_lo = wrapper.perturb_field(user_params=wrapper.UserParams(MOVE_DENSITY_HIGH_RES=False), init_boxes=ic_lo, redshift=6.0, random_seed=RNG_SEED)
#     assert pf_lo.user_params.MOVE_DENSITY_HIGH_RES == False

#     # Test low res vy field
#     f_in = "tests/DensityFieldTestData/perturb_lowres_lowres_density_hist_bins.npz"
#     f = open( f_in, 'rb' )
#     hist_hc = pickle.load( f )
#     f.close()
#     hist, bins = np.histogram(pf_lo.density, bins=100, density=True)
#     hist_test = [hist, bins]
#     for i, histo in enumerate(hist_test[0]):
#         assert histo == hist_hc[0][i], "low-res (post-perturb) density histograms do not match for:\n MOVE_DENSITY_HIGH_RES=False"

#     # Test the power spectrum from the final high resolution
#     f_in = "tests/DensityFieldTestData/k_Pk_lowres_perturb.npz"
#     f = open( f_in, 'rb' )
#     k_Pk_hc = pickle.load( f )
#     f.close()

#     res = get_power(
#     		pf_lo.density,
#     		boxlength=pf_lo.user_params.BOX_LEN,
#     		bins=None, bin_ave=False, get_variance=False, log_bins=True
#     	)
#     res = list(res)
#     k = res[1]
#     k = np.exp((np.log(k[1:]) + np.log(k[:-1])) / 2)
#     res[1] = k
#     Pk_obs = ( res[0] * k**3 / (2 * np.pi ** 2) )

#     for i, Pk in enumerate(Pk_obs):
#         if not(np.isnan(Pk)):
#             assert Pk == k_Pk_hc[1][i], "low-res (post-perturb) density power spectrum does not match for:\n MOVE_DENSITY_HIGH_RES=False"
