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

DEFAULT_SIMULATION_OPTIONS_CTEST = {
    "HII_DIM": 32,
    "DIM": 128,
    "BOX_LEN": 64,
    "SAMPLER_MIN_MASS": 1e9,
}

OPTIONS_CTEST = {
    "defaults": [18, {"USE_MASS_DEPENDENT_ZETA": True, "M_MIN_in_Mass": True}],
    "no-mdz": [18, {}],
    "mini": [
        18,
        {
            "USE_MINI_HALOS": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
        },
    ],
    "ts": [
        18,
        {"USE_TS_FLUCT": True, "USE_MASS_DEPENDENT_ZETA": True, "M_MIN_in_Mass": True},
    ],
    "ts_nomdz": [18, {"USE_TS_FLUCT": True}],
    "inhomo": [
        18,
        {"INHOMO_RECO": True, "USE_MASS_DEPENDENT_ZETA": True, "M_MIN_in_Mass": True},
    ],
    "inhomo_ts": [
        18,
        {
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
        },
    ],
    "sampler": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
        },
    ],
    "fixed_halogrids": [
        18,
        {
            "USE_HALO_FIELD": True,
            "FIXED_HALO_GRIDS": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
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
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
        },
    ],
    "sampler_ts": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_TS_FLUCT": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
        },
    ],
    "sampler_ir": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "INHOMO_RECO": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
        },
    ],
    "sampler_ts_ir": [
        18,
        {
            "USE_HALO_FIELD": True,
            "HALO_STOCHASTICITY": True,
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
        },
    ],
    "photoncons-z": [
        18,
        {
            "PHOTON_CONS_TYPE": "z-photoncons",
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
        },
    ],
    "photoncons-a": [
        18,
        {
            "PHOTON_CONS_TYPE": "alpha-photoncons",
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
        },
    ],
    "photoncons-f": [
        18,
        {
            "PHOTON_CONS_TYPE": "f-photoncons",
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
        },
    ],
    "minimize_mem": [
        18,
        {
            "USE_TS_FLUCT": True,
            "INHOMO_RECO": True,
            "MINIMIZE_MEMORY": True,
            "USE_MASS_DEPENDENT_ZETA": True,
            "M_MIN_in_Mass": True,
        },
    ],
}


@pytest.mark.parametrize("name", list(OPTIONS_CTEST.keys()))
def test_lc_runs(name, max_redshift, cache, benchmark):
    redshift, kwargs = OPTIONS_CTEST[name]
    options = prd.get_all_options_struct(redshift, lc=True, **kwargs)

    node_maxz = max_redshift
    if (
        options["inputs"].astro_options.USE_TS_FLUCT
        or options["inputs"].astro_options.INHOMO_RECO
    ):
        node_maxz = options["inputs"].simulation_options.Z_HEAT_MAX

    options["inputs"] = options["inputs"].clone(
        simulation_options=p21c.SimulationOptions.new(DEFAULT_SIMULATION_OPTIONS_CTEST),
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
def test_cv_runs(name, cache, benchmark):
    redshift, kwargs = OPTIONS_CTEST[name]
    options = prd.get_all_options_struct(redshift, lc=False, **kwargs)

    options["inputs"] = options["inputs"].clone(
        simulation_options=p21c.SimulationOptions.new(DEFAULT_SIMULATION_OPTIONS_CTEST)
    )

    with p21c.config.use(ignore_R_BUBBLE_MAX_error=True):
        cv = benchmark.pedantic(
            p21c.run_coeval,
            kwargs=dict(
                write=False,
                cache=cache,
                **options,
            ),
            iterations=1,  # these tests can be slow
            rounds=1,
        )

    assert all(isinstance(x, p21c.Coeval) for x in cv)
    cv = cv[0]
    assert np.all(np.isfinite(cv.brightness_temp))
    assert cv.simulation_options == options["inputs"].simulation_options
    assert cv.cosmo_params == options["inputs"].cosmo_params
    assert cv.astro_params == options["inputs"].astro_params
    assert cv.astro_options == options["inputs"].astro_options
