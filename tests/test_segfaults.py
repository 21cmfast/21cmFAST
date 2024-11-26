"""
This file contains tests which run a lightcone under various flag options
to test the C backend for segfaults.
These will not test the outputs of the run past the fact that they are finite,
just that the run completes without error
"""

import pytest

import numpy as np

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
    "defaults": [20, {"USE_MASS_DEPENDENT_ZETA": True}],
    "no-mdz": [20, {}],
    "mini": [
        20,
        {
            "USE_MINI_HALOS": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
    ],
    "ts": [20, {"USE_TS_FLUCT": True, "USE_MASS_DEPENDENT_ZETA": True}],
    "ts_nomdz": [20, {"USE_TS_FLUCT": True}],
    "inhomo": [20, {"INHOMO_RECO": True, "USE_MASS_DEPENDENT_ZETA": True}],
    "inhomo_ts": [
        20,
        {"INHOMO_RECO": True, "USE_TS_FLUCT": True, "USE_MASS_DEPENDENT_ZETA": True},
    ],
    "sampler": [
        20,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
    ],
    "sampler_mini": [
        20,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_MINI_HALOS": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
    ],
    "sampler_ts": [
        20,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_TS_FLUCT": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
    ],
    "sampler_ir": [
        20,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "INHOMO_RECO": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
    ],
    "sampler_ts_ir": [
        20,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
    ],
    "photoncons-z": [
        20,
        {"PHOTON_CONS_TYPE": "z-photoncons", "USE_MASS_DEPENDENT_ZETA": True},
    ],
    "photoncons-a": [
        20,
        {"PHOTON_CONS_TYPE": "alpha-photoncons", "USE_MASS_DEPENDENT_ZETA": True},
    ],
    "photoncons-f": [
        20,
        {"PHOTON_CONS_TYPE": "f-photoncons", "USE_MASS_DEPENDENT_ZETA": True},
    ],
    "minimize_mem": [
        20,
        {
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "MINIMIZE_MEMORY": True,
            "USE_MASS_DEPENDENT_ZETA": True,
        },
    ],
}


@pytest.mark.parametrize("name", list(OPTIONS_CTEST.keys()))
def test_lc_runs(name, max_redshift, default_seed):
    redshift, kwargs = OPTIONS_CTEST[name]
    options = prd.get_all_options(redshift, **kwargs)

    options["user_params"] = p21c.UserParams.new(DEFAULT_USER_PARAMS_CTEST)

    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=redshift,
        max_redshift=max_redshift,
        quantities=[
            "brightness_temp",
        ],
        resolution=options["user_params"].cell_size,
    )

    inputs = p21c.InputParameters(
        cosmo_params=options.pop("cosmo_params"),
        user_params=options.pop("user_params"),
        astro_params=options.pop("astro_params"),
        flag_options=options.pop("flag_options"),
        random_seed=default_seed,
    )

    with p21c.config.use(ignore_R_BUBBLE_MAX_error=True):
        _, _, _, lightcone = p21c.exhaust_lightcone(
            lightconer=lcn,
            write=False,
            inputs=inputs,
        )

    assert isinstance(lightcone, p21c.LightCone)
    assert np.all(np.isfinite(lightcone.brightness_temp))
