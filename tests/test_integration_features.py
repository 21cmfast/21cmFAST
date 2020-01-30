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
import numpy as np

from py21cmfast import global_params

from . import produce_integration_test_data as prd


@pytest.mark.parametrize("redshift,kwargs", prd.OPTIONS)
def test_power_spectra_coeval(redshift, kwargs):
    print("Options used for the test: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename(redshift, **kwargs), "r") as f:
        power = f["power_coeval"][...]

    with global_params.use(zprime_step_factor=prd.DEFAULT_ZPRIME_STEP_FACTOR):
        # Note that if zprime_step_factor is set in kwargs, it will over-ride this.
        k, p, bt = prd.produce_coeval_power_spectra(redshift, **kwargs)

    assert np.allclose(power, p, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("redshift,kwargs", prd.OPTIONS)
def test_power_spectra_lightcone(redshift, kwargs):
    print("Options used for the test: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename(redshift, **kwargs), "r") as f:
        power = f["power_lc"][...]
        xHI = f["xHI"][...]
        Tb = f["Tb"][...]

    with global_params.use(zprime_step_factor=prd.DEFAULT_ZPRIME_STEP_FACTOR):
        # Note that if zprime_step_factor is set in kwargs, it will over-ride this.
        k, p, lc = prd.produce_lc_power_spectra(redshift, **kwargs)

    assert np.allclose(power, p, atol=1e-5, rtol=5e-3)
    assert np.allclose(xHI, lc.global_xHI, atol=1e-5, rtol=1e-3)
    assert np.allclose(Tb, lc.global_brightness_temp, atol=1e-5, rtol=1e-3)
