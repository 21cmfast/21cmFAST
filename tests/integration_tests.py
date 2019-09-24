"""
A set of large-scale tests which test code updates against previously-run "golden"
results.

The idea here is that any new updates (except for major versions) should be non-breaking;
firstly, they should not break the API, so that the tests should run with crashing without
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

import glob
import os

import numpy as np
import pytest
import h5py

from powerbox import get_power
from py21cmfast import (
    run_coeval,
    run_lightcone,
    FlagOptions,
    AstroParams,
    CosmoParams,
    UserParams,
)

_SEED = 12345
DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")

_redshifts = [6, 7, 9, 11, 15, 20, 30]
_z_step_factor = [1.02, 1.05]
_z_heat_max = [None, 40, 25]
_hmf = [None, 0, 2, 3]
_interp_pf, _mdz, _rsd, _inh_reco, _ts, _mmin_mass, _wisdom = [[None, True]] * 7


def _get_defaults(kwargs, cls):
    return {k: kwargs.pop(k.lower(), v) for k, v in cls._defaults_.items()}


def _get_all_defaults(kwargs):
    flag_options = _get_defaults(kwargs, FlagOptions)
    astro_params = _get_defaults(kwargs, AstroParams)
    cosmo_params = _get_defaults(kwargs, CosmoParams)
    user_params = _get_defaults(kwargs, UserParams)
    return user_params, cosmo_params, astro_params, flag_options


@pytest.mark.skip
@pytest.mark.parametrize(
    "fname", glob.glob(os.path.join(DATA_PATH, "power_spectra_coeval_*.h5"))
)
def test_power_spectra_coeval(fname):
    with h5py.File(fname) as f:
        kwargs = dict(f.attrs)
        power = f["power"][...]

    user_params, cosmo_params, astro_params, flag_options = _get_all_defaults(kwargs)

    # Now ensure some properties that make the box small
    user_params["HII_DIM"] = 50
    user_params["DIM"] = 150
    user_params["BOX_LEN"] = 100

    init, perturb, ionize, brightness_temp = run_coeval(
        redshift=kwargs.pop("redshift", 7),
        user_params=user_params,
        cosmo_params=cosmo_params,
        astro_params=astro_params,
        flag_options=flag_options,
        regenerate=True,
        write=False,
        z_step_factor=kwargs.pop("z_step_factor", None),
        z_heat_max=kwargs.pop("z_heat_max", None),
        use_interp_perturb_field=kwargs.pop("use_interp_pf", False),
        random_seed=12345,
    )

    p, k = get_power(brightness_temp.brightness_temp, boxlength=user_params["BOX_LEN"])
    assert np.allclose(power, p, atol=1e-5, rtol=1e-4)


@pytest.mark.skip
@pytest.mark.parametrize(
    "fname", glob.glob(os.path.join(DATA_PATH, "power_spectra_lightcone_*.h5"))
)
def test_power_spectra_lightcone(fname):
    with h5py.File(fname) as f:
        kwargs = dict(f.attrs)
        power = f["power"]

    user_params, cosmo_params, astro_params, flag_options = _get_all_defaults(kwargs)

    # Now ensure some properties that make the box small
    user_params["HII_DIM"] = 50
    user_params["DIM"] = 150
    user_params["BOX_LEN"] = 100

    redshift = kwargs.pop("redshift")
    lightcone = run_lightcone(
        redshift=redshift,
        max_redshift=redshift + 2,
        user_params=user_params,
        cosmo_params=cosmo_params,
        astro_params=astro_params,
        flag_options=flag_options,
        regenerate=True,
        write=False,
        z_step_factor=kwargs.pop("z_step_factor", 1.02),
        z_heat_max=kwargs.pop("z_heat_max", None),
        use_interp_perturb_field=kwargs.pop("use_interp_pf", False),
        random_seed=12345,
    )

    p, k = get_power(
        lightcone.brightness_temp, boxlength=lightcone.lightcone_dimensions
    )
    assert np.allclose(power, p, atol=1e-5, rtol=1e-4)


def test_global_xHI():
    ...
