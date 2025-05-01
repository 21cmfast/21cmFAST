"""
Unit-tests of the lightcone drivers.

They do not test for correctness of simulations, but whether different parameter
options work/don't work as intended.
"""

import numpy as np
import pytest

import py21cmfast as p21c


def test_lightcone(
    lc, default_simulation_options, lightcone_min_redshift, max_redshift
):
    assert lc.lightcone_redshifts[-1] >= max_redshift
    assert np.isclose(lc.lightcone_redshifts[0], lightcone_min_redshift, atol=1e-4)
    assert (
        lc.cell_size
        == default_simulation_options.BOX_LEN / default_simulation_options.HII_DIM
    )


def test_lightcone_quantities(
    ic, default_input_struct_lc, lightcone_min_redshift, max_redshift, cache
):
    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=lightcone_min_redshift,
        max_redshift=max_redshift,
        resolution=ic.simulation_options.cell_size,
        cosmo=ic.cosmo_params.cosmo,
        quantities=(
            "density",
            "brightness_temp",
            "ionisation_rate_G12",
        ),
    )

    with pytest.raises(ValueError, match="You asked for"):
        _, _, _, lc = p21c.run_lightcone(
            lightconer=lcn,
            initial_conditions=ic,
            inputs=default_input_struct_lc,
            global_quantities=("cumulative_recombinations"),
            cache=cache,
        )
    _, _, _, lc = p21c.run_lightcone(
        lightconer=lcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        global_quantities=(
            "density",
            "ionisation_rate_G12",
            "log10_mturn_acg",
            "log10_mturn_mcg",
        ),
        cache=cache,
    )

    assert "density" in lc.lightcones
    assert "brightness_temp" in lc.lightcones
    assert "ionisation_rate_G12" in lc.lightcones
    assert "ionisation_rate_G12" in lc.global_quantities
    assert "density" in lc.global_quantities
    assert "log10_mturn_acg" in lc.global_quantities
    assert "log10_mturn_mcg" in lc.global_quantities

    # density should be filled with not zeros.
    assert lc.lightcones["density"].min() != lc.lightcones["density"].max() != 0

    # Simply ensure that different quantities are not getting crossed/referred to each other.
    assert (
        lc.lightcones["density"].min()
        != lc.lightcones["brightness_temp"].min()
        != lc.lightcones["brightness_temp"].max()
    )

    lcn_ts = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=lightcone_min_redshift,
        max_redshift=max_redshift,
        resolution=ic.simulation_options.cell_size,
        cosmo=ic.cosmo_params.cosmo,
        quantities=("spin_temperature", "density"),
    )

    # Raise an error since we're not doing spin temp.
    with pytest.raises(AttributeError):
        p21c.run_lightcone(
            lightconer=lcn_ts,
            initial_conditions=ic,
            inputs=default_input_struct_lc,
            cache=cache,
        )

    # And also raise an error for global quantities.
    with pytest.raises(AttributeError):
        p21c.run_lightcone(
            lightconer=lcn_ts,
            initial_conditions=ic,
            inputs=default_input_struct_lc,
            global_quantities=("spin_temperature",),
            cache=cache,
        )


def test_lightcone_coords(lc):
    assert lc.lightcone_coords.shape == (lc.lightcone_distances.shape[0],)
    assert lc.lightcone_coords[0] == 0.0
    np.testing.assert_allclose(
        np.diff(lc.lightcone_coords.to_value("Mpc")),
        lc.simulation_options.BOX_LEN / lc.simulation_options.HII_DIM,
    )


def test_run_lc_bad_inputs(
    rectlcn,
    default_input_struct_lc: p21c.InputParameters,
    cache,
):
    with pytest.raises(
        ValueError,
        match="You are attempting to run a lightcone with no node_redshifts.",
    ):
        p21c.run_lightcone(
            lightconer=rectlcn,
            inputs=default_input_struct_lc.clone(node_redshifts=[]),
        )


def test_lc_with_lightcone_filename(
    ic, rectlcn, default_input_struct_lc, tmpdirec, cache
):
    fname = tmpdirec / "lightcone.h5"
    _, _, _, lc = p21c.run_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
        cache=cache,
    )
    assert fname.exists()

    lc_loaded = p21c.LightCone.from_file(fname)
    assert lc_loaded == lc
    del lc_loaded

    # This one should NOT run anything.
    _, _, _, lc2 = p21c.run_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
        cache=cache,
    )

    assert lc2 == lc
    del lc2

    fname.unlink()


def test_lc_partial_eval(rectlcn, default_input_struct_lc, tmpdirec, lc, cache):
    fname = tmpdirec / "lightcone_partial.h5"

    z = rectlcn.lc_redshifts.max()
    lc_gen = p21c.generate_lightcone(
        lightconer=rectlcn,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
        write=True,
        cache=cache,
    )
    while z > 20.0:
        iz_1, z, _, partial = next(lc_gen)

    assert 0 < partial._last_completed_node < len(rectlcn.lc_redshifts) - 1

    lc_gen_cont = p21c.generate_lightcone(
        lightconer=rectlcn,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
        cache=cache,
        write=False,
        regenerate=False,
    )
    for iz_2, z, _, finished in lc_gen_cont:  # noqa: B007
        assert z <= 20.0
        assert iz_2 > iz_1

    # this only checks the lightcone object, not the array fields
    assert finished == lc
    # make sure we wrote to the correct indices
    np.testing.assert_allclose(
        finished.global_quantities["brightness_temp"],
        lc.global_quantities["brightness_temp"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        finished.lightcones["brightness_temp"],
        lc.lightcones["brightness_temp"],
        rtol=1e-4,
    )


def test_lc_lowerz_than_photon_cons(
    ic, default_input_struct_lc, default_astro_options, max_redshift, cache
):
    inputs = default_input_struct_lc.clone(
        node_redshifts=p21c.get_logspaced_redshifts(
            min_redshift=1.9,
            max_redshift=max(default_input_struct_lc.node_redshifts),
            z_step_factor=default_input_struct_lc.simulation_options.ZPRIME_STEP_FACTOR,
        ),
        astro_options=default_astro_options.clone(PHOTON_CONS_TYPE="z-photoncons"),
    )
    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=2.0,
        max_redshift=max_redshift,
        resolution=ic.simulation_options.cell_size,
        cosmo=ic.cosmo_params.cosmo,
    )

    with pytest.raises(ValueError, match="You have passed a redshift"):
        p21c.run_lightcone(
            lightconer=lcn, initial_conditions=ic, inputs=inputs, cache=cache
        )
