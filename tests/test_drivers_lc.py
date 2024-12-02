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
    ic, default_input_struct_lc, lightcone_min_redshift, max_redshift
):
    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=lightcone_min_redshift,
        max_redshift=max_redshift,
        resolution=ic.user_params.cell_size,
        cosmo=ic.cosmo_params.cosmo,
        quantities=("dNrec_box", "density", "brightness_temp", "Gamma12_box"),
    )

    _, _, _, lc = p21c.exhaust_lightcone(
        lightconer=lcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        global_quantities=("density", "Gamma12_box"),
    )

    print(lc.lightcones.keys())
    print(lc.global_quantities.keys())

    assert hasattr(lc, "dNrec_box")
    assert hasattr(lc, "density")
    assert hasattr(lc, "global_density")
    assert hasattr(lc, "global_Gamma12")

    # dNrec is not filled because we're not doing INHOMO_RECO
    assert lc.dNrec_box.max() == lc.dNrec_box.min() == 0

    # density should be filled with not zeros.
    assert lc.density.min() != lc.density.max() != 0

    # Simply ensure that different quantities are not getting crossed/referred to each other.
    assert lc.density.min() != lc.brightness_temp.min() != lc.brightness_temp.max()

    lcn_ts = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=lightcone_min_redshift,
        max_redshift=max_redshift,
        resolution=ic.user_params.cell_size,
        cosmo=ic.cosmo_params.cosmo,
        quantities=("Ts_box", "density"),
    )

    # Raise an error since we're not doing spin temp.
    with pytest.raises(AttributeError):
        p21c.exhaust_lightcone(
            lightconer=lcn_ts,
            initial_conditions=ic,
            inputs=default_input_struct_lc,
        )

    # And also raise an error for global quantities.
    with pytest.raises(AttributeError):
        p21c.exhaust_lightcone(
            lightconer=lcn_ts,
            initial_conditions=ic,
            inputs=default_input_struct_lc,
            global_quantities=("Ts_box",),
        )


# TODO: decide if we want callbacks at all, since run_lightcone is now a generator
# def _global_Tb(coeval_box):
#     assert isinstance(coeval_box, p21c.Coeval)
#     global_Tb = coeval_box.brightness_temp.mean(dtype=np.float64).astype(np.float32)
#     assert np.isclose(global_Tb, coeval_box.brightness_temp_struct.global_Tb)
#     return global_Tb

# def test_coeval_callback(
#     rectlcn, ic, max_redshift, perturbed_field, default_flag_options
# ):
#     iz, z, coeval_output, lc = p21c.exhaust_lightcone(
#         lightconer=rectlcn,
#         initial_conditions=ic,
#         flag_options=default_flag_options,
#         lightcone_quantities=("brightness_temp",),
#         global_quantities=("brightness_temp",),
#         coeval_callback=_global_Tb,
#     )
#     assert isinstance(lc, p21c.LightCone)
#     assert isinstance(coeval_output, list)
#     assert len(lc.node_redshifts) == len(coeval_output)
#     assert np.allclose(
#         lc.global_brightness_temp, np.array(coeval_output, dtype=np.float32)
#     )


# def test_coeval_callback_redshifts(
#     rectlcn, ic, redshift, max_redshift, perturbed_field, default_flag_options
# ):
#     coeval_callback_redshifts = np.array(
#         [max_redshift, max_redshift, (redshift + max_redshift) / 2, redshift],
#         dtype=np.float32,
#     )
#     iz, z, coeval_output, lc = p21c.exhaust_lightcone(
#         lightconer=rectlcn,
#         initial_conditions=ic,
#         flag_options=default_flag_options,
#         coeval_callback=lambda x: x.redshift,
#         coeval_callback_redshifts=coeval_callback_redshifts,
#     )
#     assert len(coeval_callback_redshifts) - 1 == len(coeval_output)
#     computed_redshifts = [
#         lc.node_redshifts[np.argmin(np.abs(i - lc.node_redshifts))]
#         for i in coeval_callback_redshifts[1:]
#     ]
#     assert np.allclose(coeval_output, computed_redshifts)


# def Heaviside(x):
#     return 1 if x > 0 else 0


# def test_coeval_callback_exceptions(
#     rectlcn, ic, redshift, max_redshift, perturbed_field, default_flag_options
# ):
#     # should output warning in logs and not raise an error
#     iz, z, coeval_output, lc = p21c.exhaust_lightcone(
#         lightconer=rectlcn,
#         initial_conditions=ic,
#         flag_options=default_flag_options,
#         coeval_callback=lambda x: 1
#         / Heaviside(x.redshift - (redshift + max_redshift) / 2),
#         coeval_callback_redshifts=[max_redshift, redshift],
#     )
#     # should raise an error
#     with pytest.raises(RuntimeError) as excinfo:
#         iz, z, coeval_output, lc = p21c.exhaust_lightcone(
#             lightconer=rectlcn,
#             initial_conditions=ic,
#             max_redshift=max_redshift,
#             coeval_callback=lambda x: 1 / 0,
#             coeval_callback_redshifts=[max_redshift, redshift],
#         )
#     assert "coeval_callback computation failed on first trial" in str(excinfo.value)


def test_lightcone_coords(lc):
    assert lc.lightcone_coords.shape == (lc.lightcone_distances.shape[0],)
    assert lc.lightcone_coords[0] == 0.0
    np.testing.assert_allclose(
        np.diff(lc.lightcone_coords.to_value("Mpc")),
        lc.user_params.BOX_LEN / lc.user_params.HII_DIM,
    )


def test_run_lc_bad_inputs(rectlcn, perturbed_field, default_input_struct_lc):
    with pytest.raises(
        ValueError,
        match="You are attempting to run a lightcone with no node_redshifts.",
    ):
        p21c.exhaust_lightcone(
            lightconer=rectlcn,
            inputs=default_input_struct_lc.clone(node_redshifts=[]),
        )
    with pytest.raises(
        ValueError,
        match="If perturbed_fields are provided, initial_conditions must be provided",
    ):
        p21c.exhaust_lightcone(
            inputs=default_input_struct_lc,
            lightconer=rectlcn,
            perturbed_fields=[
                perturbed_field,
            ],
        )


def test_lc_with_lightcone_filename(ic, rectlcn, default_input_struct_lc, tmpdirec):
    fname = tmpdirec / "lightcone.h5"
    _, _, _, lc = p21c.exhaust_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
    )
    assert fname.exists()

    lc_loaded = p21c.LightCone.read(fname)
    assert lc_loaded == lc
    del lc_loaded

    # This one should NOT run anything.
    _, _, _, lc2 = p21c.exhaust_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
    )

    assert lc2 == lc
    del lc2

    fname.unlink()


def test_lc_partial_eval(rectlcn, ic, default_input_struct_lc, tmpdirec, lc):
    fname = tmpdirec / "lightcone_partial.h5"

    z = rectlcn.lc_redshifts.max()
    lc_gen = p21c.run_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
        write=True,
    )
    while z > 20.0:
        iz, z, _, partial = next(lc_gen)

    assert partial._current_index < len(rectlcn.lc_redshifts)
    assert partial._current_index > 0
    assert partial._current_redshift <= 20.0
    assert partial._current_redshift > 15.0

    _, _, _, finished = p21c.exhaust_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
    )

    assert finished == lc

    # Test that if _current redshift is not calculated, a good error is
    # raised
    with h5py.File(fname, "a") as fl:
        fl.attrs["current_redshift"] = 2 * partial._current_redshift

    with pytest.raises(IOError, match="No component boxes found at z"):
        p21c.exhaust_lightcone(
            lightconer=rectlcn,
            initial_conditions=ic,
            inputs=default_input_struct_lc,
            lightcone_filename=fname,
        )


def test_lc_lowerz_than_photon_cons(
    ic, default_input_struct_lc, default_flag_options, max_redshift
):
    with pytest.raises(ValueError, match="You have passed a redshift"):
        inputs = default_input_struct_lc.clone(
            node_redshifts=p21c.get_logspaced_redshifts(
                min_redshift=1.9,
                max_redshift=default_input_struct_lc.node_redshifts.max(),
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

        p21c.exhaust_lightcone(
            lightconer=lcn,
            initial_conditions=ic,
            inputs=inputs,
        )
