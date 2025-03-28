"""
Tests of running lightcones under various flag options.

The aim is to test the C backend for segfaults. These will not test the outputs of the
run past the fact that they are finite, just that the run completes without error.
"""

from timeit import default_timer as timer

import numpy as np
import pytest

import py21cmfast as p21c

from . import produce_integration_test_data as prd

DEFAULT_USER_PARAMS_CTEST = {
    "HII_DIM": 32,
    "DIM": 128,
    "BOX_LEN": 64,
    "NO_RNG": True,
    "SAMPLER_MIN_MASS": 1e9,
}

OPTIONS_CTEST = {
    "defaults": [18, {"USE_MASS_DEPENDENT_ZETA": True}, None],
    "no-mdz": [18, {}, None],
    "mini": [
        18,
        {
            "USE_MINI_HALOS": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
        None,
    ],
    "ts": [18, {"USE_TS_FLUCT": True, "USE_MASS_DEPENDENT_ZETA": True}, None],
    "ts_nomdz": [18, {"USE_TS_FLUCT": True}, None],
    "inhomo": [18, {"INHOMO_RECO": True, "USE_MASS_DEPENDENT_ZETA": True}, None],
    "inhomo_ts": [
        18,
        {"INHOMO_RECO": True, "USE_TS_FLUCT": True, "USE_MASS_DEPENDENT_ZETA": True},
        None,
    ],
    "sampler": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
        None,
    ],
    "fixed_halogrids": [
        18,
        {
            "USE_HALO_FIELD": True,
            "FIXED_HALO_GRIDS": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
        None,
    ],
    "sampler_mini": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_MINI_HALOS": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
        None,
    ],
    "sampler_ts": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_TS_FLUCT": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
        None,
    ],
    "sampler_ir": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "INHOMO_RECO": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
        None,
    ],
    "sampler_ts_ir": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
        None,
    ],
    "photoncons-z": [
        18,
        {"PHOTON_CONS_TYPE": "z-photoncons", "USE_MASS_DEPENDENT_ZETA": True},
        None,
    ],
    "photoncons-a": [
        18,
        {"PHOTON_CONS_TYPE": "alpha-photoncons", "USE_MASS_DEPENDENT_ZETA": True},
        None,
    ],
    "photoncons-f": [
        18,
        {"PHOTON_CONS_TYPE": "f-photoncons", "USE_MASS_DEPENDENT_ZETA": True},
        None,
    ],
    "minimize_mem": [
        18,
        {
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "MINIMIZE_MEMORY": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
        None,
    ],
}


@pytest.mark.parametrize("name", list(OPTIONS_CTEST.keys()))
def test_lc_runs(name, max_redshift, cache):
    redshift, kwargs, expected_time_s = OPTIONS_CTEST[name]
    options = prd.get_all_options_struct(redshift, lc=True, **kwargs)

    node_maxz = max_redshift
    if (
        options["inputs"].flag_options.USE_TS_FLUCT
        or options["inputs"].flag_options.INHOMO_RECO
    ):
        node_maxz = options["inputs"].user_params.Z_HEAT_MAX

    options["inputs"] = options["inputs"].clone(
        user_params=p21c.UserParams.new(DEFAULT_USER_PARAMS_CTEST),
        node_redshifts=p21c.get_logspaced_redshifts(
            min_redshift=redshift,
            max_redshift=node_maxz,
            z_step_factor=options["inputs"].user_params.ZPRIME_STEP_FACTOR,
        ),
    )

    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=redshift,
        max_redshift=max_redshift,
        quantities=[
            "brightness_temp",
        ],
        resolution=options["inputs"].user_params.cell_size,
    )

    start = timer()
    _, _, _, lightcone = p21c.run_lightcone(
        lightconer=lcn,
        write=False,
        cache=cache,
        **options,
    )
    end = timer()

    assert isinstance(lightcone, p21c.LightCone)
    assert np.all(np.isfinite(lightcone.lightcones["brightness_temp"]))
    assert lightcone.user_params == options["inputs"].user_params
    assert lightcone.cosmo_params == options["inputs"].cosmo_params
    assert lightcone.astro_params == options["inputs"].astro_params
    assert lightcone.flag_options == options["inputs"].flag_options
    if expected_time_s:
        assert (end - start) <= (expected_time_s * 1.1)


@pytest.mark.parametrize("name", list(OPTIONS_CTEST.keys()))
def test_cv_runs(name, cache):
    redshift, kwargs, expected_time_s = OPTIONS_CTEST[name]
    options = prd.get_all_options_struct(redshift, lc=False, **kwargs)

    options["inputs"] = options["inputs"].clone(
        user_params=p21c.UserParams.new(DEFAULT_USER_PARAMS_CTEST)
    )

    start = timer()
    cv = p21c.run_coeval(
        write=False,
        cache=cache,
        **options,
    )
    end = timer()

    assert all(isinstance(x, p21c.Coeval) for x in cv)
    cv = cv[0]
    assert np.all(np.isfinite(cv.brightness_temp))
    assert cv.user_params == options["inputs"].user_params
    assert cv.cosmo_params == options["inputs"].cosmo_params
    assert cv.astro_params == options["inputs"].astro_params
    assert cv.flag_options == options["inputs"].flag_options
    if expected_time_s:
        assert (end - start) <= (expected_time_s * 1.1)
