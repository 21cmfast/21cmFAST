"""
These are designed to be unit-tests of the lightcone drivers. They do not test for
correctness of simulations,
but whether different parameter options work/don't work as intended.
"""

import pytest

import h5py
import numpy as np

import py21cmfast as p21c


def test_lightcone(lc, default_user_params, lightcone_min_redshift, max_redshift):
    assert lc.lightcone_redshifts[-1] >= max_redshift
    assert np.isclose(lc.lightcone_redshifts[0], lightcone_min_redshift, atol=1e-4)
    assert lc.cell_size == default_user_params.BOX_LEN / default_user_params.HII_DIM


def test_lightcone_quantities(
    ic, default_input_struct_lc, lightcone_min_redshift, max_redshift, cache
):
    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=lightcone_min_redshift,
        max_redshift=max_redshift,
        resolution=ic.user_params.cell_size,
        cosmo=ic.cosmo_params.cosmo,
        quantities=("dNrec_box", "density", "brightness_temp", "Gamma12_box"),
    )

    _, _, _, lc = p21c.run_lightcone(
        lightconer=lcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        global_quantities=("density", "Gamma12_box"),
        cache=cache,
    )

    assert "dNrec_box" in lc.lightcones
    assert "density" in lc.lightcones
    assert "brightness_temp" in lc.lightcones
    assert "Gamma12_box" in lc.lightcones
    assert "Gamma12_box" in lc.global_quantities
    assert "density" in lc.global_quantities

    # dNrec is not filled because we're not doing INHOMO_RECO
    assert lc.lightcones["dNrec_box"].max() == lc.lightcones["dNrec_box"].min() == 0

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
        resolution=ic.user_params.cell_size,
        cosmo=ic.cosmo_params.cosmo,
        quantities=("Ts_box", "density"),
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
            global_quantities=("Ts_box",),
            cache=cache,
        )


def test_lightcone_coords(lc):
    assert lc.lightcone_coords.shape == (lc.lightcone_distances.shape[0],)
    assert lc.lightcone_coords[0] == 0.0
    np.testing.assert_allclose(
        np.diff(lc.lightcone_coords.to_value("Mpc")),
        lc.user_params.BOX_LEN / lc.user_params.HII_DIM,
    )


def test_run_lc_bad_inputs(
    rectlcn,
    perturbed_field_lc: p21c.PerturbedField,
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

    with pytest.raises(
        ValueError,
        match="If perturbed_fields are provided, initial_conditions must be provided",
    ):
        # The perturbed_field here has no node redshifts (because it doesn't
        # require any since USE_TS_FLUCT is False). This
        p21c.run_lightcone(
            inputs=default_input_struct_lc,
            lightconer=rectlcn,
            perturbed_fields=[
                perturbed_field_lc,
            ],
            cache=cache,
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


def test_lc_partial_eval(rectlcn, ic, default_input_struct_lc, tmpdirec, lc, cache):
    fname = tmpdirec / "lightcone_partial.h5"

    z = rectlcn.lc_redshifts.max()
    lc_gen = p21c.generate_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
        write=True,
        cache=cache,
    )
    while z > 20.0:
        iz, z, _, partial = next(lc_gen)

    assert 0 < partial._last_completed_node < len(rectlcn.lc_redshifts) - 1

    _, _, _, finished = p21c.run_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
        cache=cache,
    )

    assert finished == lc


def test_lc_lowerz_than_photon_cons(
    ic, default_input_struct_lc, default_flag_options, max_redshift, cache
):
    with pytest.raises(ValueError, match="You have passed a redshift"):
        inputs = default_input_struct_lc.clone(
            node_redshifts=p21c.get_logspaced_redshifts(
                min_redshift=1.9,
                max_redshift=max(default_input_struct_lc.node_redshifts),
                z_step_factor=default_input_struct_lc.user_params.ZPRIME_STEP_FACTOR,
            ),
            flag_options=default_flag_options.clone(PHOTON_CONS_TYPE="z-photoncons"),
        )
        lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
            min_redshift=2.0,
            max_redshift=max_redshift,
            resolution=ic.user_params.cell_size,
            cosmo=ic.cosmo_params.cosmo,
        )

        p21c.run_lightcone(
            lightconer=lcn, initial_conditions=ic, inputs=inputs, cache=cache
        )
