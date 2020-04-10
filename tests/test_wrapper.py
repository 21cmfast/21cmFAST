"""
These are designed to be unit-tests of the wrapper functionality. They do not test for
correctness of simulations,
but whether different parameter options work/don't work as intended.
"""
import pytest

import numpy as np

from py21cmfast import wrapper

REDSHIFT = 15
LOW_REDSHIFT = 8


@pytest.fixture(scope="module")
def user_params():
    # Do a small box, for testing
    return wrapper.UserParams(HII_DIM=35, DIM=70)


@pytest.fixture(scope="module")
def init_box(user_params, tmpdirec):
    return wrapper.initial_conditions(
        user_params=user_params, random_seed=12, write=True
    )


@pytest.fixture(scope="module")
def perturb_field(init_box, tmpdirec):
    "A default perturb_field"
    return wrapper.perturb_field(redshift=REDSHIFT, init_boxes=init_box, write=True)


@pytest.fixture(scope="module")
def perturb_field_lowz(init_box, tmpdirec):
    "A default perturb_field"
    return wrapper.perturb_field(redshift=LOW_REDSHIFT, init_boxes=init_box, write=True)


@pytest.fixture(scope="module")
def ionize_box(perturb_field, tmpdirec):
    """A default ionize_box"""
    return wrapper.ionize_box(perturbed_field=perturb_field, write=True)


@pytest.fixture(scope="module")
def ionize_box_lowz(perturb_field_lowz, tmpdirec):
    """A default ionize_box at lower redshift."""
    return wrapper.ionize_box(perturbed_field=perturb_field_lowz, write=True)


@pytest.fixture(scope="module")
def spin_temp(perturb_field, tmpdirec):
    "A default perturb_field"
    return wrapper.spin_temperature(perturbed_field=perturb_field, write=True)


def test_perturb_field_no_ic(user_params, perturb_field):
    """Run a perturb field without passing an init box"""
    pf = wrapper.perturb_field(redshift=REDSHIFT, user_params=user_params)
    assert len(pf.density) == pf.user_params.HII_DIM == user_params.HII_DIM
    assert pf.redshift == REDSHIFT
    assert pf.random_seed != perturb_field.random_seed
    assert not np.all(pf.density == 0)
    assert pf != perturb_field
    assert pf._seedless_repr() == perturb_field._seedless_repr()


def test_ib_no_z(init_box):
    with pytest.raises(ValueError):
        wrapper.ionize_box(init_boxes=init_box)


def test_pf_unnamed_param():
    "Try using an un-named parameter"
    with pytest.raises(TypeError):
        wrapper.perturb_field(7)


def test_perturb_field_ic(perturb_field, init_box):
    pf = wrapper.perturb_field(redshift=REDSHIFT, init_boxes=init_box,)

    assert len(pf.density) == len(init_box.lowres_density)
    assert pf.cosmo_params == init_box.cosmo_params
    assert pf.user_params == init_box.user_params
    assert not np.all(pf.density == 0)

    assert pf.user_params == perturb_field.user_params
    assert pf.cosmo_params == perturb_field.cosmo_params

    assert pf == perturb_field


def test_cache_exists(user_params, perturb_field, tmpdirec):
    pf = wrapper.PerturbedField(
        redshift=perturb_field.redshift,
        cosmo_params=perturb_field.cosmo_params,
        user_params=user_params,
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
            perturbed_field=perturb_field_lowz, random_seed=1,
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
            perturbed_field=perturb_field, random_seed=1,
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


def test_st_from_z(init_box, spin_temp):
    pf = wrapper.perturb_field(redshift=12, init_boxes=init_box)

    # This one has all the same parameters as the nominal spin_temp, but is evaluated with
    # perturb field exactly matching it, rather than interpolated
    st = wrapper.spin_temperature(
        perturbed_field=pf,
        astro_params=spin_temp.astro_params,
        flag_options=spin_temp.flag_options,
        redshift=spin_temp.redshift,
    )

    assert st == spin_temp
    assert not np.all(st.Ts_box == spin_temp.Ts_box)


def test_pf_regenerate(perturb_field):
    pf = wrapper.perturb_field(
        redshift=perturb_field.redshift, user_params=perturb_field.user_params,
    )

    assert not np.all(pf.density == perturb_field.density)
    assert pf.random_seed != perturb_field.random_seed


def test_ib_from_pf(perturb_field):
    ib = wrapper.ionize_box(perturbed_field=perturb_field)
    assert ib.redshift == perturb_field.redshift
    assert ib.user_params == perturb_field.user_params
    assert ib.cosmo_params == perturb_field.cosmo_params


def test_ib_from_z(user_params, perturb_field):
    ib = wrapper.ionize_box(
        redshift=perturb_field.redshift, user_params=user_params, regenerate=False
    )
    assert ib.redshift == perturb_field.redshift
    assert ib.user_params == perturb_field.user_params
    assert ib.cosmo_params == perturb_field.cosmo_params
    assert ib.cosmo_params is not perturb_field.cosmo_params


def test_ib_override_z(perturb_field):
    with pytest.raises(ValueError):
        wrapper.ionize_box(
            redshift=perturb_field.redshift + 1, perturbed_field=perturb_field,
        )


def test_ib_override_z_heat_max(perturb_field):
    # save previous z_heat_max
    zheatmax = wrapper.global_params.Z_HEAT_MAX

    wrapper.ionize_box(
        redshift=perturb_field.redshift, perturbed_field=perturb_field, z_heat_max=12.0,
    )

    assert wrapper.global_params.Z_HEAT_MAX == zheatmax


def test_ib_bad_st(init_box):
    with pytest.raises(ValueError):
        wrapper.ionize_box(redshift=REDSHIFT, spin_temp=init_box)


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


def test_coeval_against_direct(init_box, perturb_field, ionize_box):
    coeval = wrapper.run_coeval(perturb=perturb_field, init_box=init_box, write=False)

    assert coeval[0].init_struct == init_box
    assert coeval[0].perturb_struct == perturb_field
    assert coeval[0].ionization_struct == ionize_box


def test_lightcone(init_box, perturb_field):
    lc = wrapper.run_lightcone(
        init_box=init_box, perturb=perturb_field, max_redshift=10.0
    )

    assert lc.lightcone_redshifts[-1] >= 10.0
    assert np.isclose(lc.lightcone_redshifts[0], perturb_field.redshift, atol=1e-4)
    assert lc.cell_size == init_box.user_params.BOX_LEN / init_box.user_params.HII_DIM


def test_lightcone_quantities(init_box, perturb_field):
    lc = wrapper.run_lightcone(
        init_box=init_box,
        perturb=perturb_field,
        max_redshift=20.0,
        lightcone_quantities=("dNrec_box", "density", "brightness_temp"),
        global_quantities=("density", "Gamma12_box"),
    )

    assert hasattr(lc, "dNrec_box")
    assert hasattr(lc, "density")
    assert hasattr(lc, "global_density")
    assert hasattr(lc, "global_Gamma12")

    print(perturb_field.density.min(), perturb_field.density.max())
    # dNrec is not filled because we're not doing INHOMO_RECO
    assert lc.dNrec_box.max() == lc.dNrec_box.min() == 0

    # density should be filled with not zeros.
    assert lc.density.min() != lc.density.max() != 0

    # Simply ensure that different quantities are not getting crossed/referred to each other.
    assert lc.density.min() != lc.brightness_temp.min() != lc.brightness_temp.max()

    # Raise an error since we're not doing spin temp.
    with pytest.raises(ValueError):
        wrapper.run_lightcone(
            init_box=init_box,
            perturb=perturb_field,
            max_redshift=20.0,
            lightcone_quantities=("Ts_box", "density"),
        )

    # And also raise an error for global quantities.
    with pytest.raises(ValueError):
        wrapper.run_lightcone(
            init_box=init_box,
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


def test_coeval_st(init_box, perturb_field):
    coeval = wrapper.run_coeval(
        init_box=init_box, perturb=perturb_field, flag_options={"USE_TS_FLUCT": True},
    )

    assert isinstance(coeval.spin_temp_struct, wrapper.TsBox)
