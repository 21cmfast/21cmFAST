"""
These are designed to be unit-tests of the wrapper functionality. They do not test for
correctness of simulations,
but whether different parameter options work/don't work as intended.
"""
import pytest

import numpy as np

from py21cmfast import wrapper


@pytest.fixture(scope="module")
def perturb_field_lowz(ic, low_redshift):
    """A default perturb_field"""
    return wrapper.perturb_field(redshift=low_redshift, init_boxes=ic, write=True)


@pytest.fixture(scope="module")
def ionize_box(perturb_field):
    """A default ionize_box"""
    return wrapper.ionize_box(perturbed_field=perturb_field, write=True)


@pytest.fixture(scope="module")
def ionize_box_lowz(perturb_field_lowz):
    """A default ionize_box at lower redshift."""
    return wrapper.ionize_box(perturbed_field=perturb_field_lowz, write=True)


@pytest.fixture(scope="module")
def spin_temp(perturb_field):
    """A default perturb_field"""
    return wrapper.spin_temperature(perturbed_field=perturb_field, write=True)


def test_perturb_field_no_ic(default_user_params, redshift, perturb_field):
    """Run a perturb field without passing an init box"""
    pf = wrapper.perturb_field(redshift=redshift, user_params=default_user_params)
    assert len(pf.density) == pf.user_params.HII_DIM == default_user_params.HII_DIM
    assert pf.redshift == redshift
    assert pf.random_seed != perturb_field.random_seed
    assert not np.all(pf.density == 0)
    assert pf != perturb_field
    assert pf._seedless_repr() == perturb_field._seedless_repr()


def test_ib_no_z(ic):
    with pytest.raises(ValueError):
        wrapper.ionize_box(init_boxes=ic)


def test_pf_unnamed_param():
    """Try using an un-named parameter."""
    with pytest.raises(TypeError):
        wrapper.perturb_field(7)


def test_perturb_field_ic(perturb_field, ic):
    # this will run perturb_field again, since by default regenerate=True for tests.
    # BUT it should produce exactly the same as the default perturb_field since it has
    # the same seed.
    pf = wrapper.perturb_field(redshift=perturb_field.redshift, init_boxes=ic)

    assert len(pf.density) == len(ic.lowres_density)
    assert pf.cosmo_params == ic.cosmo_params
    assert pf.user_params == ic.user_params
    assert not np.all(pf.density == 0)

    assert pf.user_params == perturb_field.user_params
    assert pf.cosmo_params == perturb_field.cosmo_params

    assert pf == perturb_field


def test_cache_exists(default_user_params, perturb_field, tmpdirec):
    pf = wrapper.PerturbedField(
        redshift=perturb_field.redshift,
        cosmo_params=perturb_field.cosmo_params,
        user_params=default_user_params,
    )

    assert pf.exists(tmpdirec)

    pf.read(tmpdirec)
    assert np.all(pf.density == perturb_field.density)
    assert pf == perturb_field


def test_pf_new_seed(perturb_field, tmpdirec):
    pf = wrapper.perturb_field(
        redshift=perturb_field.redshift,
        user_params=perturb_field.user_params,
        random_seed=1,
    )

    # we didn't write it, and this has a different seed
    assert not pf.exists(direc=tmpdirec)
    assert pf.random_seed != perturb_field.random_seed

    assert not np.all(pf.density == perturb_field.density)


def test_ib_new_seed(ionize_box_lowz, perturb_field_lowz, tmpdirec):
    # this should fail because perturb_field has a seed set already, which isn't 1.
    with pytest.raises(ValueError):
        wrapper.ionize_box(
            perturbed_field=perturb_field_lowz,
            random_seed=1,
        )

    ib = wrapper.ionize_box(
        cosmo_params=perturb_field_lowz.cosmo_params,
        redshift=perturb_field_lowz.redshift,
        user_params=perturb_field_lowz.user_params,
        random_seed=1,
    )

    # we didn't write it, and this has a different seed
    assert not ib.exists(direc=tmpdirec)
    assert ib.random_seed != ionize_box_lowz.random_seed
    assert not np.all(ib.xH_box == ionize_box_lowz.xH_box)


def test_st_new_seed(spin_temp, perturb_field, tmpdirec):
    # this should fail because perturb_field has a seed set already, which isn't 1.
    with pytest.raises(ValueError):
        wrapper.spin_temperature(
            perturbed_field=perturb_field,
            random_seed=1,
        )

    st = wrapper.spin_temperature(
        cosmo_params=spin_temp.cosmo_params,
        user_params=spin_temp.user_params,
        astro_params=spin_temp.astro_params,
        flag_options=spin_temp.flag_options,
        redshift=spin_temp.redshift,
        random_seed=1,
    )

    # we didn't write it, and this has a different seed
    assert not st.exists(direc=tmpdirec)
    assert st.random_seed != spin_temp.random_seed
    assert not np.all(st.Ts_box == spin_temp.Ts_box)


def test_st_from_z(perturb_field_lowz, spin_temp):

    # This one has all the same parameters as the nominal spin_temp, but is evaluated
    # with an interpolated perturb_field
    st = wrapper.spin_temperature(
        perturbed_field=perturb_field_lowz,
        astro_params=spin_temp.astro_params,
        flag_options=spin_temp.flag_options,
        redshift=spin_temp.redshift,  # Higher redshift
    )

    assert st != spin_temp
    assert not np.all(st.Ts_box == spin_temp.Ts_box)


def test_ib_from_pf(perturb_field):
    ib = wrapper.ionize_box(perturbed_field=perturb_field)
    assert ib.redshift == perturb_field.redshift
    assert ib.user_params == perturb_field.user_params
    assert ib.cosmo_params == perturb_field.cosmo_params


def test_ib_from_z(default_user_params, perturb_field):
    ib = wrapper.ionize_box(
        redshift=perturb_field.redshift,
        user_params=default_user_params,
        regenerate=False,
    )
    assert ib.redshift == perturb_field.redshift
    assert ib.user_params == perturb_field.user_params
    assert ib.cosmo_params == perturb_field.cosmo_params
    assert ib.cosmo_params is not perturb_field.cosmo_params


def test_ib_override_z(perturb_field):
    with pytest.raises(ValueError):
        wrapper.ionize_box(
            redshift=perturb_field.redshift + 1,
            perturbed_field=perturb_field,
        )


def test_ib_override_z_heat_max(perturb_field):
    # save previous z_heat_max
    zheatmax = wrapper.global_params.Z_HEAT_MAX

    wrapper.ionize_box(
        redshift=perturb_field.redshift,
        perturbed_field=perturb_field,
        z_heat_max=12.0,
    )

    assert wrapper.global_params.Z_HEAT_MAX == zheatmax


def test_ib_bad_st(ic, redshift):
    with pytest.raises(ValueError):
        wrapper.ionize_box(redshift=redshift, spin_temp=ic)


def test_bt(ionize_box, spin_temp, perturb_field):
    with pytest.raises(TypeError):  # have to specify param names
        wrapper.brightness_temperature(ionize_box, spin_temp, perturb_field)

    # this will fail because ionized_box was not created with spin temperature.
    with pytest.raises(ValueError):
        wrapper.brightness_temperature(
            ionized_box=ionize_box, perturbed_field=perturb_field, spin_temp=spin_temp
        )

    bt = wrapper.brightness_temperature(
        ionized_box=ionize_box, perturbed_field=perturb_field
    )

    assert bt.cosmo_params == perturb_field.cosmo_params
    assert bt.user_params == perturb_field.user_params
    assert bt.flag_options == ionize_box.flag_options
    assert bt.astro_params == ionize_box.astro_params


def test_coeval_against_direct(ic, perturb_field, ionize_box):
    coeval = wrapper.run_coeval(perturb=perturb_field, init_box=ic)

    assert coeval.init_struct == ic
    assert coeval.perturb_struct == perturb_field
    assert coeval.ionization_struct == ionize_box


def test_lightcone(lc, default_user_params, redshift, max_redshift):
    assert lc.lightcone_redshifts[-1] >= max_redshift
    assert np.isclose(lc.lightcone_redshifts[0], redshift, atol=1e-4)
    assert lc.cell_size == default_user_params.BOX_LEN / default_user_params.HII_DIM


def test_lightcone_quantities(ic, max_redshift, perturb_field):
    lc = wrapper.run_lightcone(
        init_box=ic,
        perturb=perturb_field,
        max_redshift=max_redshift,
        lightcone_quantities=("dNrec_box", "density", "brightness_temp"),
        global_quantities=("density", "Gamma12_box"),
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
    with pytest.raises(ValueError):
        wrapper.run_lightcone(
            init_box=ic,
            perturb=perturb_field,
            max_redshift=20.0,
            lightcone_quantities=("Ts_box", "density"),
        )

    # And also raise an error for global quantities.
    with pytest.raises(ValueError):
        wrapper.run_lightcone(
            init_box=ic,
            perturb=perturb_field,
            max_redshift=20.0,
            global_quantities=("Ts_box",),
        )


def test_run_lf():
    muv, mhalo, lf = wrapper.compute_luminosity_function(redshifts=[7, 8, 9], nbins=100)
    assert np.all(lf[~np.isnan(lf)] > -30)
    assert lf.shape == (3, 100)

    # Check that memory is in-tact and a second run also works:
    muv, mhalo, lf2 = wrapper.compute_luminosity_function(
        redshifts=[7, 8, 9], nbins=100
    )
    assert lf2.shape == (3, 100)
    assert np.allclose(lf2[~np.isnan(lf2)], lf[~np.isnan(lf)])

    muv_minih, mhalo_minih, lf_minih = wrapper.compute_luminosity_function(
        redshifts=[7, 8, 9],
        nbins=100,
        component=0,
        flag_options={"USE_MINI_HALOS": True},
        mturnovers=[7.0, 7.0, 7.0],
        mturnovers_mini=[5.0, 5.0, 5.0],
    )
    assert np.all(lf_minih[~np.isnan(lf_minih)] > -30)
    assert lf_minih.shape == (3, 100)


def test_coeval_st(ic, perturb_field):
    coeval = wrapper.run_coeval(
        init_box=ic,
        perturb=perturb_field,
        flag_options={"USE_TS_FLUCT": True},
    )

    assert isinstance(coeval.spin_temp_struct, wrapper.TsBox)


def _global_Tb(coeval_box):
    assert isinstance(coeval_box, wrapper.Coeval)
    global_Tb = coeval_box.brightness_temp.mean(dtype=np.float128).astype(np.float32)
    assert np.isclose(global_Tb, coeval_box.brightness_temp_struct.global_Tb)
    return global_Tb


def test_coeval_callback(ic, max_redshift, perturb_field):
    lc, coeval_output = wrapper.run_lightcone(
        init_box=ic,
        perturb=perturb_field,
        max_redshift=max_redshift,
        lightcone_quantities=("brightness_temp",),
        global_quantities=("brightness_temp",),
        coeval_callback=_global_Tb,
    )
    assert isinstance(lc, wrapper.LightCone)
    assert isinstance(coeval_output, list)
    assert len(lc.node_redshifts) == len(coeval_output)
    assert np.allclose(
        lc.global_brightness_temp, np.array(coeval_output, dtype=np.float32)
    )


def test_coeval_callback_redshifts(ic, redshift, max_redshift, perturb_field):
    coeval_callback_redshifts = np.array(
        [max_redshift, max_redshift, (redshift + max_redshift) / 2, redshift],
        dtype=np.float32,
    )
    lc, coeval_output = wrapper.run_lightcone(
        init_box=ic,
        perturb=perturb_field,
        max_redshift=max_redshift,
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


def test_coeval_callback_exceptions(ic, redshift, max_redshift, perturb_field):
    # should output warning in logs and not raise an error
    lc, coeval_output = wrapper.run_lightcone(
        init_box=ic,
        perturb=perturb_field,
        max_redshift=max_redshift,
        coeval_callback=lambda x: 1
        / Heaviside(x.redshift - (redshift + max_redshift) / 2),
        coeval_callback_redshifts=[max_redshift, redshift],
    )
    # should raise an error
    with pytest.raises(RuntimeError) as excinfo:
        lc, coeval_output = wrapper.run_lightcone(
            init_box=ic,
            perturb=perturb_field,
            max_redshift=max_redshift,
            coeval_callback=lambda x: 1 / 0,
            coeval_callback_redshifts=[max_redshift, redshift],
        )
    assert "coeval_callback computation failed on first trial" in str(excinfo.value)


def test_coeval_vs_low_level(ic):

    coeval = wrapper.run_coeval(
        redshift=20,
        init_box=ic,
        zprime_step_factor=1.1,
        regenerate=True,
        flag_options={"USE_TS_FLUCT": True},
        write=False,
    )

    st = wrapper.spin_temperature(
        redshift=20,
        init_boxes=ic,
        zprime_step_factor=1.1,
        regenerate=True,
        flag_options={"USE_TS_FLUCT": True},
        write=False,
    )

    assert np.allclose(coeval.Tk_box, st.Tk_box)
    assert np.allclose(coeval.Ts_box, st.Ts_box)
    assert np.allclose(coeval.x_e_box, st.x_e_box)


def test_using_cached_halo_field(ic, test_direc):
    """Test whether the C-based memory in halo fields is cached correctly.

    Prior to v3.1 this was segfaulting, so this test ensure that this behaviour does
    not regress.
    """
    halo_field = wrapper.determine_halo_list(
        redshift=10.0,
        init_boxes=ic,
        write=True,
        direc=test_direc,
    )

    pt_halos = wrapper.perturb_halo_list(
        redshift=10.0,
        init_boxes=ic,
        halo_field=halo_field,
        write=True,
        direc=test_direc,
    )

    print("DONE WITH FIRST BOXES!")
    # Now get the halo field again at the same redshift -- should be cached
    new_halo_field = wrapper.determine_halo_list(
        redshift=10.0, init_boxes=ic, write=False, regenerate=False
    )

    new_pt_halos = wrapper.perturb_halo_list(
        redshift=10.0,
        init_boxes=ic,
        halo_field=new_halo_field,
        write=False,
        regenerate=False,
    )

    np.testing.assert_allclose(new_halo_field.halo_masses, halo_field.halo_masses)
    np.testing.assert_allclose(pt_halos.halo_coords, new_pt_halos.halo_coords)
