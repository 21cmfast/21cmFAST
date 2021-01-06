"""
A set of large-scale tests which test code updates against previously-run "golden"
results.

The idea here is that any new updates (except for major versions) should be non-breaking;
firstly, they should not break the API, so that the tests should run without crashing without
being changed.
Secondly, the actual results of running the basic functions should remain the same for
the same input code, except for potential bug-fixes. In these cases, these tests should
pick these changes up. The test data should then be changed to reflect the new gold
standard, and if applicable, a new test should be written that reflects the previous
broken code.
Thirdly, it enforces that new features, where possible, are added in such a way as to
keep the default behaviour constant. That is, the tests here should *not* run the added
feature, and therefore should continue to produce the same test results regardless of
the new feature added. The new feature should be accompanied by its own tests, whether
in this or another test module. If a new feature *must* be included by default, then
it must be implemented in a new major version of the code, at which point the test data
is able to be updated.

Comparison tests here are meant to be as small as possible while attempting to form
a reasonable test: they should be of reduced data such as power spectra or global xHI
measurements, and they should be generated with small simulations.
"""
import pytest

import h5py
import logging
import numpy as np

from py21cmfast import config, global_params

from . import produce_integration_test_data as prd

logger = logging.getLogger("21cmFAST")
logger.setLevel(logging.INFO)


options = prd.OPTIONS
options_pt = prd.OPTIONS_PT
options_halo = prd.OPTIONS_HALO

# Skip the USE_MINI_HALOS test because it takes too long.
# This should be revisited in the future.
options = tuple([z, kw] for z, kw in prd.OPTIONS)


@pytest.mark.parametrize("redshift,kwargs", options)
def test_power_spectra_coeval(redshift, kwargs, module_direc):
    print("Options used for the test: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename(redshift, **kwargs), "r") as f:
        power = f["power_coeval"][...]

    with config.use(direc=module_direc, regenerate=False, write=True):
        with global_params.use(zprime_step_factor=prd.DEFAULT_ZPRIME_STEP_FACTOR):
            # Note that if zprime_step_factor is set in kwargs, it will over-ride this.
            k, p, bt = prd.produce_coeval_power_spectra(redshift, **kwargs)

    assert np.allclose(
        power[: len(power) / 2], p[: len(power) / 2], atol=1e-3, rtol=1e-3
    )
    assert np.allclose(
        power[(len(power) / 2) :], p[(len(power) / 2) :], atol=1e-2, rtol=1e-2
    )


@pytest.mark.parametrize("redshift,kwargs", options)
def test_power_spectra_lightcone(redshift, kwargs, module_direc):
    print("Options used for the test: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename(redshift, **kwargs), "r") as f:
        power = f["power_lc"][...]
        xHI = f["xHI"][...]
        Tb = f["Tb"][...]

    with config.use(direc=module_direc, regenerate=False, write=True):
        with global_params.use(zprime_step_factor=prd.DEFAULT_ZPRIME_STEP_FACTOR):
            # Note that if zprime_step_factor is set in kwargs, it will over-ride this.
            k, p, lc = prd.produce_lc_power_spectra(redshift, **kwargs)

    assert np.allclose(
        power[: len(power / 2)], p[: len(power / 2)], atol=1e-5, rtol=1e-2
    )
    assert np.allclose(
        power[(len(power / 2)) :], p[(len(power / 2)) :], atol=5e-2, rtol=5e-2
    )

    assert np.allclose(xHI, lc.global_xH, atol=1e-5, rtol=1e-3)
    assert np.allclose(Tb, lc.global_brightness_temp, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("redshift,kwargs", options_pt)
def test_perturb_field_data(redshift, kwargs):
    print("Options used for the test: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename_pt(redshift, **kwargs), "r") as f:
        power_dens = f["power_dens"][...]
        power_vel = f["power_vel"][...]
        pdf_dens = f["pdf_dens"][...]
        pdf_vel = f["pdf_vel"][...]

    with global_params.use(zprime_step_factor=prd.DEFAULT_ZPRIME_STEP_FACTOR):
        # Note that if zprime_step_factor is set in kwargs, it will over-ride this.
        (
            k_dens,
            p_dens,
            k_vel,
            p_vel,
            x_dens,
            y_dens,
            x_vel,
            y_vel,
            ic,
        ) = prd.produce_perturb_field_data(redshift, **kwargs)

    assert np.allclose(power_dens, p_dens, atol=5e-3, rtol=1e-3)
    assert np.allclose(power_vel, p_vel, atol=5e-3, rtol=1e-3)
    assert np.allclose(pdf_dens, y_dens, atol=5e-3, rtol=1e-3)
    assert np.allclose(pdf_vel, y_vel, atol=5e-3, rtol=1e-3)


@pytest.mark.parametrize("redshift,kwargs", options_halo)
def test_halo_field_data(redshift, kwargs):
    print("Options used for the test: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename_halo(redshift, **kwargs), "r") as f:
        n_pt_halos = f["n_pt_halos"][...]
        pt_halo_masses = f["pt_halo_masses"][...]

    with global_params.use(zprime_step_factor=prd.DEFAULT_ZPRIME_STEP_FACTOR):
        # Note that if zprime_step_factor is set in kwargs, it will over-ride this.
        pt_halos = prd.produce_halo_field_data(redshift, **kwargs)

    assert np.allclose(n_pt_halos, pt_halos.n_halos, atol=5e-3, rtol=1e-3)
    assert np.allclose(
        np.sum(pt_halo_masses), np.sum(pt_halos.halo_masses), atol=5e-3, rtol=1e-3
    )
