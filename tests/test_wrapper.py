"""
These are designed to be unit-tests of the wrapper functionality. They do not test for
correctness of simulations,
but whether different parameter options work/don't work as intended.
"""

import pytest

import h5py
import numpy as np
from astropy import units as un

import py21cmfast as p21c


@pytest.fixture(scope="module")
def ic_newseed(default_user_params, tmpdirec):
    return p21c.compute_initial_conditions(
        user_params=default_user_params, write=True, direc=tmpdirec, random_seed=33
    )


@pytest.fixture(scope="module")
def perturb_field_lowz(ic, low_redshift):
    """A default perturb_field"""
    return p21c.perturb_field(redshift=low_redshift, initial_conditions=ic, write=True)


@pytest.fixture(scope="module")
def ionize_box(ic, perturbed_field, default_astro_params, default_flag_options):
    """A default ionize_box"""
    return p21c.compute_ionization_field(
        initial_conditions=ic,
        astro_params=default_astro_params,
        flag_options=default_flag_options,
        perturbed_field=perturbed_field,
        write=True,
    )


@pytest.fixture(scope="module")
def ionize_box_lowz(ic, perturb_field_lowz, default_astro_params, default_flag_options):
    """A default ionize_box at lower redshift."""
    return p21c.compute_ionization_field(
        initial_conditions=ic,
        astro_params=default_astro_params,
        flag_options=default_flag_options,
        perturbed_field=perturb_field_lowz,
        write=True,
    )


@pytest.fixture(scope="module")
def spin_temp(ic, perturbed_field, default_astro_params, default_flag_options):
    """A default perturb_field"""
    return p21c.spin_temperature(
        initial_conditions=ic,
        astro_params=default_astro_params,
        flag_options=default_flag_options,
        perturbed_field=perturbed_field,
        write=True,
    )


def test_perturb_field_no_ic(default_user_params, redshift, perturbed_field):
    """Run a perturb field without passing an init box"""
    with pytest.raises(TypeError):
        p21c.perturb_field(redshift=redshift, user_params=default_user_params)


def test_ib_no_z(ic):
    with pytest.raises(TypeError):
        p21c.compute_ionization_field(initial_conditions=ic)


def test_pf_unnamed_param():
    """Try using an un-named parameter."""
    with pytest.raises(TypeError):
        p21c.perturb_field(7)


def test_perturb_field_ic(perturbed_field, ic):
    # this will run perturb_field again, since by default regenerate=True for tests.
    # BUT it should produce exactly the same as the default perturb_field since it has
    # the same seed.
    pf = p21c.perturb_field(redshift=perturbed_field.redshift, initial_conditions=ic)

    assert len(pf.density) == len(ic.lowres_density)
    assert pf.cosmo_params == ic.cosmo_params
    assert pf.user_params == ic.user_params
    assert not np.all(pf.density == 0)

    assert pf.user_params == perturbed_field.user_params
    assert pf.cosmo_params == perturbed_field.cosmo_params

    assert pf == perturbed_field


def test_cache_exists(default_input_struct, perturbed_field, tmpdirec):
    pf = p21c.PerturbedField(
        inputs=default_input_struct,
    )

    assert pf.exists(tmpdirec)

    pf.read(tmpdirec)
    assert np.all(pf.density == perturbed_field.density)
    assert pf == perturbed_field


def test_new_seeds(
    ic_newseed,
    perturb_field_lowz,
    ionize_box_lowz,
    default_astro_params,
    default_flag_options,
    tmpdirec,
):
    # Perturbed Field
    pf = p21c.perturb_field(
        redshift=perturb_field_lowz.redshift,
        initial_conditions=ic_newseed,
    )

    # we didn't write it, and this has a different seed
    assert not pf.exists(direc=tmpdirec)
    assert pf.random_seed != perturb_field_lowz.random_seed
    assert not np.all(pf.density == perturb_field_lowz.density)

    # Ionization Box
    with pytest.raises(ValueError):
        p21c.compute_ionization_field(
            initial_conditions=ic_newseed,
            perturbed_field=perturb_field_lowz,
            astro_params=default_astro_params,
            flag_options=default_flag_options,
        )

    ib = p21c.compute_ionization_field(
        initial_conditions=ic_newseed,
        perturbed_field=pf,
        astro_params=default_astro_params,
        flag_options=default_flag_options,
    )

    # we didn't write it, and this has a different seed
    assert not ib.exists(direc=tmpdirec)
    assert ib.random_seed != ionize_box_lowz.random_seed
    assert not np.all(ib.xH_box == ionize_box_lowz.xH_box)


def test_ib_from_pf(perturbed_field, ic, default_astro_params, default_flag_options):
    ib = p21c.compute_ionization_field(
        initial_conditions=ic,
        perturbed_field=perturbed_field,
        astro_params=default_astro_params,
        flag_options=default_flag_options,
    )
    assert ib.redshift == perturbed_field.redshift
    assert ib.user_params == perturbed_field.user_params
    assert ib.cosmo_params == perturbed_field.cosmo_params


def test_ib_override_z_heat_max(
    ic, perturbed_field, default_astro_params, default_flag_options
):
    # save previous z_heat_max
    zheatmax = p21c.global_params.Z_HEAT_MAX

    p21c.compute_ionization_field(
        initial_conditions=ic,
        perturbed_field=perturbed_field,
        astro_params=default_astro_params,
        flag_options=default_flag_options,
        z_heat_max=12.0,
    )

    assert p21c.global_params.Z_HEAT_MAX == zheatmax


def test_ib_bad_st(ic, perturbed_field, redshift):
    with pytest.raises(ValueError):
        p21c.compute_ionization_field(
            initial_conditions=ic,
            perturbed_field=perturbed_field,
            spin_temp=ic,
        )


def test_bt(ionize_box, spin_temp, perturbed_field):
    with pytest.raises(TypeError):  # have to specify param names
        p21c.brightness_temperature(ionize_box, spin_temp, perturbed_field)

    # this will fail because ionized_box was not created with spin temperature.
    with pytest.raises(ValueError):
        p21c.brightness_temperature(
            ionized_box=ionize_box, perturbed_field=perturbed_field, spin_temp=spin_temp
        )

    bt = p21c.brightness_temperature(
        ionized_box=ionize_box, perturbed_field=perturbed_field
    )

    assert bt.cosmo_params == perturbed_field.cosmo_params
    assert bt.user_params == perturbed_field.user_params
    assert bt.flag_options == ionize_box.flag_options
    assert bt.astro_params == ionize_box.astro_params


def test_coeval_against_direct(ic, perturbed_field, ionize_box):
    coeval = p21c.run_coeval(perturbed_field=perturbed_field, initial_conditions=ic)

    assert coeval.init_struct == ic
    assert coeval.perturb_struct == perturbed_field
    assert coeval.ionization_struct == ionize_box


def test_lightcone(lc, default_user_params, redshift, max_redshift):
    assert lc.lightcone_redshifts[-1] >= max_redshift
    assert np.isclose(lc.lightcone_redshifts[0], redshift, atol=1e-4)
    assert lc.cell_size == default_user_params.BOX_LEN / default_user_params.HII_DIM


def test_lightcone_quantities(ic, rectlcn, max_redshift):
    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=rectlcn.min_redshift,
        max_redshift=rectlcn.max_redshift,
        resolution=ic.user_params.cell_size,
        cosmo=ic.cosmo_params.cosmo,
        quantities=("dNrec_box", "density", "brightness_temp", "Gamma12"),
    )

    lc = p21c.exhaust_lightcone(
        lightconer=lcn,
        initial_conditions=ic,
        max_redshift=max_redshift,
    )

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

    # Raise an error since we're not doing spin temp.
    with pytest.raises(AttributeError):
        p21c.exhaust_lightcone(
            lightconer=rectlcn,
            initial_conditions=ic,
            max_redshift=20.0,
            lightcone_quantities=("Ts_box", "density"),
        )

    # And also raise an error for global quantities.
    with pytest.raises(AttributeError):
        p21c.exhaust_lightcone(
            lightconer=rectlcn,
            initial_conditions=ic,
            max_redshift=20.0,
            global_quantities=("Ts_box",),
        )


def test_run_lf():
    muv, mhalo, lf = p21c.compute_luminosity_function(redshifts=[7, 8, 9], nbins=100)
    assert np.all(lf[~np.isnan(lf)] > -30)
    assert lf.shape == (3, 100)

    # Check that memory is in-tact and a second run also works:
    muv, mhalo, lf2 = p21c.compute_luminosity_function(redshifts=[7, 8, 9], nbins=100)
    assert lf2.shape == (3, 100)
    assert np.allclose(lf2[~np.isnan(lf2)], lf[~np.isnan(lf)])

    muv_minih, mhalo_minih, lf_minih = p21c.compute_luminosity_function(
        redshifts=[7, 8, 9],
        nbins=100,
        component="both",
        flag_options={
            "USE_MINI_HALOS": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
        },
        mturnovers=[7.0, 7.0, 7.0],
        mturnovers_mini=[5.0, 5.0, 5.0],
    )
    assert np.all(lf_minih[~np.isnan(lf_minih)] > -30)
    assert lf_minih.shape == (3, 100)


def test_coeval_st(ic, perturbed_field):
    coeval = p21c.run_coeval(
        initial_conditions=ic,
        perturbed_field=perturbed_field,
        flag_options={"USE_TS_FLUCT": True},
    )

    assert isinstance(coeval.spin_temp_struct, p21c.TsBox)


def _global_Tb(coeval_box):
    assert isinstance(coeval_box, p21c.Coeval)
    global_Tb = coeval_box.brightness_temp.mean(dtype=np.float64).astype(np.float32)
    assert np.isclose(global_Tb, coeval_box.brightness_temp_struct.global_Tb)
    return global_Tb


def test_coeval_callback(
    rectlcn, ic, max_redshift, perturbed_field, default_flag_options
):
    iz, z, coeval_output, lc = p21c.exhaust_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        flag_options=default_flag_options,
        lightcone_quantities=("brightness_temp",),
        global_quantities=("brightness_temp",),
        coeval_callback=_global_Tb,
    )
    assert isinstance(lc, p21c.LightCone)
    assert isinstance(coeval_output, list)
    assert len(lc.node_redshifts) == len(coeval_output)
    assert np.allclose(
        lc.global_brightness_temp, np.array(coeval_output, dtype=np.float32)
    )


def test_coeval_callback_redshifts(
    rectlcn, ic, redshift, max_redshift, perturbed_field, default_flag_options
):
    coeval_callback_redshifts = np.array(
        [max_redshift, max_redshift, (redshift + max_redshift) / 2, redshift],
        dtype=np.float32,
    )
    iz, z, coeval_output, lc = p21c.exhaust_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        flag_options=default_flag_options,
        coeval_callback=lambda x: x.redshift,
        coeval_callback_redshifts=coeval_callback_redshifts,
    )
    assert len(coeval_callback_redshifts) - 1 == len(coeval_output)
    computed_redshifts = [
        lc.node_redshifts[np.argmin(np.abs(i - lc.node_redshifts))]
        for i in coeval_callback_redshifts[1:]
    ]
    assert np.allclose(coeval_output, computed_redshifts)


def Heaviside(x):
    return 1 if x > 0 else 0


def test_coeval_callback_exceptions(
    rectlcn, ic, redshift, max_redshift, perturbed_field, default_flag_options
):
    # should output warning in logs and not raise an error
    iz, z, coeval_output, lc = p21c.exhaust_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        flag_options=default_flag_options,
        coeval_callback=lambda x: 1
        / Heaviside(x.redshift - (redshift + max_redshift) / 2),
        coeval_callback_redshifts=[max_redshift, redshift],
    )
    # should raise an error
    with pytest.raises(RuntimeError) as excinfo:
        iz, z, coeval_output, lc = p21c.exhaust_lightcone(
            lightconer=rectlcn,
            initial_conditions=ic,
            max_redshift=max_redshift,
            coeval_callback=lambda x: 1 / 0,
            coeval_callback_redshifts=[max_redshift, redshift],
        )
    assert "coeval_callback computation failed on first trial" in str(excinfo.value)


# TODO: Remove / entirely rewrite this test w/o recursion
# def test_coeval_vs_low_level(ic):
#     coeval = p21c.run_coeval(
#         out_redshifts=20,
#         initial_conditions=ic,
#         zprime_step_factor=1.1,
#         regenerate=True,
#         flag_options={"USE_TS_FLUCT": True},
#         write=False,
#     )

#     st = p21c.spin_temperature(
#         redshift=20,
#         initial_conditions=ic,
#         zprime_step_factor=1.1,
#         regenerate=True,
#         flag_options={"USE_TS_FLUCT": True},
#         write=False,
#     )

#     np.testing.assert_allclose(coeval.Tk_box, st.Tk_box, rtol=1e-4, atol=1e-4)
#     np.testing.assert_allclose(coeval.Ts_box, st.Ts_box, rtol=1e-4, atol=1e-4)
#     np.testing.assert_allclose(coeval.x_e_box, st.x_e_box, rtol=1e-4, atol=1e-4)


def test_using_cached_halo_field(
    ic, test_direc, default_astro_params, default_flag_options
):
    """Test whether the C-based memory in halo fields is cached correctly.

    Prior to v3.1 this was segfaulting, so this test ensure that this behaviour does
    not regress.
    """
    f = default_flag_options.clone(USE_HALO_FIELD=True)
    halo_field = p21c.determine_halo_list(
        redshift=10.0,
        initial_conditions=ic,
        astro_params=default_astro_params,
        flag_options=f,
        write=True,
        direc=test_direc,
    )

    pt_halos = p21c.perturb_halo_list(
        redshift=10.0,
        initial_conditions=ic,
        astro_params=default_astro_params,
        flag_options=f,
        halo_field=halo_field,
        write=True,
        direc=test_direc,
    )

    print("DONE WITH FIRST BOXES!")
    # Now get the halo field again at the same redshift -- should be cached
    new_halo_field = p21c.determine_halo_list(
        redshift=10.0,
        initial_conditions=ic,
        astro_params=default_astro_params,
        flag_options=f,
        write=False,
        regenerate=False,
    )

    new_pt_halos = p21c.perturb_halo_list(
        redshift=10.0,
        initial_conditions=ic,
        halo_field=new_halo_field,
        write=False,
        regenerate=False,
    )

    np.testing.assert_allclose(new_halo_field.halo_masses, halo_field.halo_masses)
    np.testing.assert_allclose(pt_halos.halo_coords, new_pt_halos.halo_coords)


def test_lightcone_coords(lc):
    assert lc.lightcone_coords.shape == (lc.lightcone_distances.shape[0],)
    assert lc.lightcone_coords[0] == 0.0
    np.testing.assert_allclose(
        np.diff(lc.lightcone_coords.to_value("Mpc")),
        lc.user_params.BOX_LEN / lc.user_params.HII_DIM,
    )


def test_run_coeval_bad_inputs():
    with pytest.raises(
        ValueError, match="Cannot use an interpolated perturb field with minihalos"
    ):
        p21c.run_coeval(
            redshift=6.0,
            flag_options={
                "USE_MINI_HALOS": True,
                "INHOMO_RECO": True,
                "USE_TS_FLUCT": True,
            },
            use_interp_perturb_field=True,
        )


def test_run_lc_bad_inputs(rectlcn, default_user_params):
    with pytest.raises(
        ValueError, match="You must provide either redshift, perturb or lightconer"
    ):
        p21c.exhaust_lightcone(lightconer=rectlcn)

    with pytest.raises(
        ValueError,
        match="If trying to minimize memory usage, you must be caching. Set write=True",
    ):
        p21c.exhaust_lightcone(
            lightconer=rectlcn,
            redshift=6.0,
            user_params={"MINIMIZE_MEMORY": True},
            write=False,
        )


def test_lc_with_lightcone_filename(rectlcn, perturbed_field, tmpdirec):
    fname = tmpdirec / "lightcone.h5"
    _, _, _, lc = p21c.exhaust_lightcone(lightconer=rectlcn, lightcone_filename=fname)
    assert fname.exists()

    lc_loaded = p21c.LightCone.read(fname)
    assert lc_loaded == lc
    del lc_loaded

    # This one should NOT run anything.
    _, _, _, lc2 = p21c.exhaust_lightcone(
        lightconer=rectlcn,
        lightcone_filename=fname,
    )
    assert lc2 == lc
    del lc2

    fname.unlink()


def test_lc_partial_eval(rectlcn, perturbed_field, tmpdirec, lc):
    fname = tmpdirec / "lightcone_partial.h5"

    z = rectlcn.max_redshift
    while z > 20.0:
        z, _, _, partial = p21c.run_lightcone(
            lightconer=rectlcn,
            lightcone_filename=fname,
            write=True,
        )

    assert partial._current_index < len(rectlcn.lc_redshifts)
    assert partial._current_index > 0
    assert partial._current_redshift <= 20.0
    assert partial._current_redshift > 15.0

    _, _, _, finished = p21c.exhaust_lightcone(
        lightconer=rectlcn,
        lightcone_filename=fname,
    )

    assert finished == lc

    # Test that if _current redshift is not calculated, a good error is
    # raised
    with h5py.File(fname, "a") as fl:
        fl.attrs["current_redshift"] = 2 * partial._current_redshift

    with pytest.raises(IOError, match="No component boxes found at z"):
        p21c.run_lightcone(
            lightconer=rectlcn,
            lightcone_filename=fname,
        )


def test_coeval_lowerz_than_photon_cons(ic):
    with pytest.raises(ValueError, match="You have passed a redshift"):
        p21c.run_coeval(
            initial_conditions=ic,
            out_redshift=2.0,
            flag_options={
                "PHOTON_CONS_TYPE": 1,
                "USE_HALO_FIELD": False,
                "HALO_STOCHASTICITY": False,
            },
        )


def test_lc_lowerz_than_photon_cons(rectlcn, ic):
    with pytest.raises(ValueError, match="You have passed a redshift"):
        p21c.run_lightcone(
            initial_conditions=ic,
            out_redshift=2.0,
            flag_options={
                "PHOTON_CONS_TYPE": 1,
                "USE_HALO_FIELD": False,
                "HALO_STOCHASTICITY": False,
            },
        )
