"""
Tests of running lightcones under various flag options.

The aim is to test the C backend for segfaults. These will not test the outputs of the
run past the fact that they are finite, just that the run completes without error.
"""

import numpy as np
import pytest

import py21cmfast as p21c

from . import produce_integration_test_data as prd

COMMON_INPUTS_CTEST = {
    "HII_DIM": 32,
    "DIM": 128,
    "BOX_LEN": 64,
    "SAMPLER_MIN_MASS": 1e9,
    "USE_MASS_DEPENDENT_ZETA": True,
    "M_MIN_in_Mass": True,
    "HII_FILTER": "spherical-tophat",
    "NO_RNG": False,
}

OPTIONS_CTEST = {
    "defaults": [18, {}],
    "no-mdz": [
        18,
        {
            "USE_MASS_DEPENDENT_ZETA": False,
        },
    ],
    "mini": [
        18,
        {
            "USE_MINI_HALOS": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
            "USE_TS_FLUCT": True,
            "M_TURN": 5.0,
        },
    ],
    "ts": [
        18,
        {"USE_TS_FLUCT": True},
    ],
    "ts_nomdz": [18, {"USE_TS_FLUCT": True}],
    "inhomo": [
        18,
        {"INHOMO_RECO": True, "R_BUBBLE_MAX": 50.0},
    ],
    "inhomo_ts": [
        18,
        {
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "R_BUBBLE_MAX": 50.0,
        },
    ],
    "sampler": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
        },
    ],
    "fixed_halogrids": [
        18,
        {
            "USE_HALO_FIELD": True,
            "FIXED_HALO_GRIDS": True,
        },
    ],
    "sampler_mini": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_MINI_HALOS": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
            "M_TURN": 5.0,
        },
    ],
    "sampler_ts": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_TS_FLUCT": True,
        },
    ],
    "sampler_ir": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
        },
    ],
    "sampler_ts_ir": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
        },
    ],
    "photoncons-z": [
        18,
        {
            "PHOTON_CONS_TYPE": "z-photoncons",
        },
    ],
    "minimize_mem": [
        18,
        {
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "R_BUBBLE_MAX": 50.0,
            "MINIMIZE_MEMORY": True,
        },
    ],
}


@pytest.mark.parametrize("name", list(OPTIONS_CTEST.keys()))
def test_lc_runs(name, max_redshift, cache, benchmark):
    redshift, kwargs = OPTIONS_CTEST[name]
    options = prd.get_all_options_struct(
        redshift, lc=True, **{**COMMON_INPUTS_CTEST, **kwargs}
    )

    node_maxz = max_redshift
    if (
        options["inputs"].astro_options.USE_TS_FLUCT
        or options["inputs"].astro_options.INHOMO_RECO
    ):
        node_maxz = options["inputs"].simulation_options.Z_HEAT_MAX

    options["inputs"] = options["inputs"].clone(
        node_redshifts=p21c.get_logspaced_redshifts(
            min_redshift=redshift,
            max_redshift=node_maxz,
            z_step_factor=options["inputs"].simulation_options.ZPRIME_STEP_FACTOR,
        ),
    )

    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=redshift,
        max_redshift=max_redshift,
        quantities=[
            "brightness_temp",
        ],
        resolution=options["inputs"].simulation_options.cell_size,
    )

    with p21c.config.use(ignore_R_BUBBLE_MAX_error=True):
        _, _, _, lightcone = benchmark.pedantic(
            p21c.run_lightcone,
            kwargs=dict(
                lightconer=lcn,
                write=False,
                cache=cache,
                **options,
            ),
            iterations=1,  # these tests can be slow
            rounds=1,
        )

    assert isinstance(lightcone, p21c.LightCone)
    assert np.all(np.isfinite(lightcone.lightcones["brightness_temp"]))
    assert lightcone.simulation_options == options["inputs"].simulation_options
    assert lightcone.cosmo_params == options["inputs"].cosmo_params
    assert lightcone.astro_params == options["inputs"].astro_params
    assert lightcone.astro_options == options["inputs"].astro_options


@pytest.mark.parametrize("name", list(OPTIONS_CTEST.keys()))
def test_cv_runs(name, cache):
    redshift, kwargs = OPTIONS_CTEST[name]
    options = prd.get_all_options_struct(
        redshift, lc=False, **{**COMMON_INPUTS_CTEST, **kwargs}
    )

    with p21c.config.use(ignore_R_BUBBLE_MAX_error=True):
        cv = p21c.run_coeval(
            write=False,
            cache=cache,
            **options,
        )

    assert all(isinstance(x, p21c.Coeval) for x in cv)
    cv = cv[0]
    assert np.all(np.isfinite(cv.brightness_temp))
    assert cv.simulation_options == options["inputs"].simulation_options
    assert cv.cosmo_params == options["inputs"].cosmo_params
    assert cv.astro_params == options["inputs"].astro_params
    assert cv.astro_options == options["inputs"].astro_options
