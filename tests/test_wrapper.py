"""
These are designed to be unit-tests of the wrapper functionality. They do not test for correctness of simulations,
but whether different parameter options work/don't work as intended.
"""
import numpy as np
import pytest

from py21cmmc import wrapper

REDSHIFT = 15


@pytest.fixture(scope="module")
def user_params():
    # Do a small box, for testing
    return wrapper.UserParams(HII_DIM=35, DIM=70)


@pytest.fixture(scope="module")
def init_box(user_params, tmpdirec):
    return wrapper.initial_conditions(
        user_params=user_params,
        regenerate=True,
        direc=tmpdirec.strpath,
        random_seed=12
    )


@pytest.fixture(scope="module")
def perturb_field(init_box, tmpdirec):
    "A default perturb_field"
    return wrapper.perturb_field(
        redshift=REDSHIFT,
        regenerate=True,  # i.e. make sure we don't read it in.
        init_boxes=init_box,
        direc=tmpdirec.strpath,
    )


@pytest.fixture(scope="module")
def ionize_box(perturb_field, tmpdirec):
    "A default perturb_field"
    return wrapper.ionize_box(
        perturbed_field=perturb_field,
        regenerate=True,  # i.e. make sure we don't read it in.
        direc=tmpdirec.strpath,
        z_step_factor=1.2
    )


@pytest.fixture(scope="module")
def spin_temp(perturb_field, tmpdirec):
    "A default perturb_field"
    return wrapper.spin_temperature(
        perturbed_field=perturb_field,
        regenerate=True,  # i.e. make sure we don't read it in.
        direc=tmpdirec.strpath,
        z_step_factor=1.2,
    )


def test_perturb_field_no_ic(user_params, perturb_field):
    "Run a perturb field without passing an init box"

    assert len(perturb_field.density) == perturb_field.user_params.HII_DIM == user_params.HII_DIM
    assert perturb_field.redshift == REDSHIFT
    assert not np.all(perturb_field.density == 0)


def test_ib_no_z(init_box):
    with pytest.raises(ValueError):
        wrapper.ionize_box(init_boxes=init_box)


def test_pf_unnamed_param():
    "Try using an un-named parameter"
    with pytest.raises(TypeError):
        wrapper.perturb_field(7)


def test_perturb_field_ic(user_params, perturb_field, tmpdirec):
    ic = wrapper.initial_conditions(user_params=user_params, regenerate=True, direc=tmpdirec.strpath, write=False)
    pf = wrapper.perturb_field(redshift=REDSHIFT, init_boxes=ic, regenerate=True, direc=tmpdirec.strpath, write=False)

    assert len(pf.density) == len(ic.lowres_density)
    assert pf.cosmo_params == ic.cosmo_params
    assert pf.user_params == ic.user_params
    assert not np.all(pf.density == 0)

    assert pf.user_params == perturb_field.user_params
    assert pf.random_seed != perturb_field.random_seed
    assert pf.cosmo_params == perturb_field.cosmo_params

    # they shouldn't be the same, as they have different seeds
    assert pf != perturb_field

    # but they are the same in every other way
    assert pf._seedless_repr() == perturb_field._seedless_repr()


def test_cache_exists(user_params, perturb_field, tmpdirec):
    pf = wrapper.PerturbedField(redshift=perturb_field.redshift, cosmo_params=perturb_field.cosmo_params,
                                user_params=user_params)

    assert pf.exists(tmpdirec.strpath)

    pf.read(tmpdirec.strpath)
    assert np.all(pf.density == perturb_field.density)
    assert pf == perturb_field


def test_pf_new_seed(perturb_field, tmpdirec):
    pf = wrapper.perturb_field(
        redshift=perturb_field.redshift,
        user_params=perturb_field.user_params,
        direc=tmpdirec.strpath,
        random_seed=1,
        write=False
    )

    assert not pf.exists(direc=tmpdirec.strpath)  # we didn't write it, and this has a different seed (presumably)
    assert pf.random_seed != perturb_field.random_seed

    assert not np.all(pf.density == perturb_field.density)


def test_ib_new_seed(ionize_box, perturb_field, tmpdirec):
    # this should fail because perturb_field has a seed set already, which isn't 1.
    with pytest.raises(ValueError):
        ib = wrapper.ionize_box(
            perturbed_field=perturb_field,
            direc=tmpdirec.strpath,
            random_seed=1,
            write=False
        )

    ib = wrapper.ionize_box(
        cosmo_params=perturb_field.cosmo_params,
        redshift=perturb_field.redshift,
        user_params=perturb_field.user_params,
        direc=tmpdirec.strpath,
        random_seed=1,
        write=False
    )

    assert not ib.exists(direc=tmpdirec.strpath)  # we didn't write it, and this has a different seed (presumably)
    assert ib.random_seed != ionize_box.random_seed
    assert not np.all(ib.xH_box == ionize_box.xH_box)


def test_st_new_seed(spin_temp, perturb_field, tmpdirec):
    # this should fail because perturb_field has a seed set already, which isn't 1.
    with pytest.raises(ValueError):
        st = wrapper.spin_temperature(
            perturbed_field=perturb_field,
            direc=tmpdirec.strpath,
            random_seed=1,
            write=False,
        )

    st = wrapper.spin_temperature(
        cosmo_params=spin_temp.cosmo_params,
        user_params=spin_temp.user_params,
        astro_params=spin_temp.astro_params,
        flag_options=spin_temp.flag_options,
        redshift=spin_temp.redshift,
        direc=tmpdirec.strpath,
        random_seed=1,
        write=False
    )

    assert not st.exists(direc=tmpdirec.strpath)  # we didn't write it, and this has a different seed (presumably)
    assert st.random_seed != spin_temp.random_seed
    assert not np.all(st.Ts_box == spin_temp.Ts_box)


def test_st_from_z(init_box, tmpdirec, spin_temp):
    pf = wrapper.perturb_field(
        redshift=12,
        init_boxes=init_box,
        write=False,
        regenerate=True
    )

    # This one has all the same parameters as the nominal spin_temp, but is evaluated with
    # perturb field exactly matching it, rather than interpolated
    st = wrapper.spin_temperature(
        perturbed_field=pf,
        astro_params=spin_temp.astro_params,
        flag_options=spin_temp.flag_options,
        direc=tmpdirec.strpath,
        redshift=spin_temp.redshift,
        write=False
    )

    # TODO: This REALLY SHOULD NOT BE TRUE!!!!!
    assert st == spin_temp
    assert not np.all(st.Ts_box == spin_temp.Ts_box)


def test_pf_regenerate(perturb_field, tmpdirec):
    pf = wrapper.perturb_field(
        redshift=perturb_field.redshift,
        user_params=perturb_field.user_params,
        direc=tmpdirec.strpath,
        regenerate=True
    )

    assert not np.all(pf.density == perturb_field.density)
    assert pf.random_seed != perturb_field.random_seed


def test_ib_from_pf(perturb_field, tmpdirec):
    ib = wrapper.ionize_box(perturbed_field=perturb_field, direc=tmpdirec.strpath, write=False)
    assert ib.redshift == perturb_field.redshift
    assert ib.user_params == perturb_field.user_params
    assert ib.cosmo_params == perturb_field.cosmo_params


def test_ib_from_z(user_params, perturb_field, tmpdirec):
    ib = wrapper.ionize_box(redshift=perturb_field.redshift, user_params=user_params, direc=tmpdirec.strpath,
                            write=False)
    assert ib.redshift == perturb_field.redshift
    assert ib.user_params == perturb_field.user_params
    assert ib.cosmo_params == perturb_field.cosmo_params
    assert ib.cosmo_params is not perturb_field.cosmo_params


def test_ib_override_z(perturb_field, tmpdirec):
    with pytest.raises(ValueError):
        wrapper.ionize_box(redshift=perturb_field.redshift + 1,
                           perturbed_field=perturb_field, direc=tmpdirec.strpath,
                           write=False)


def test_ib_override_z_heat_max(perturb_field, tmpdirec):
    # save previous z_heat_max
    zheatmax = wrapper.global_params.Z_HEAT_MAX

    wrapper.ionize_box(redshift=perturb_field.redshift, perturbed_field=perturb_field, direc=tmpdirec.strpath,
                       write=False, z_heat_max=12.0)

    assert wrapper.global_params.Z_HEAT_MAX == 12.0

    # set it back so that "nothing changes"
    wrapper.global_params.Z_HEAT_MAX = zheatmax


def test_ib_bad_st(init_box):
    with pytest.raises(ValueError):
        wrapper.ionize_box(redshift=REDSHIFT, spin_temp=init_box)


def test_bt(ionize_box, spin_temp, perturb_field):
    with pytest.raises(TypeError):  # have to specify param names
        wrapper.brightness_temperature(ionize_box, spin_temp, perturb_field)

    # this will fail because ionized_box was not created with spin temperature.
    with pytest.raises(ValueError):
        wrapper.brightness_temperature(
            ionized_box=ionize_box,
            perturbed_field=perturb_field,
            spin_temp=spin_temp
        )

    bt = wrapper.brightness_temperature(
        ionized_box=ionize_box,
        perturbed_field=perturb_field
    )

    assert bt.cosmo_params == perturb_field.cosmo_params
    assert bt.user_params == perturb_field.user_params
    assert bt.flag_options == ionize_box.flag_options
    assert bt.astro_params == ionize_box.astro_params


def test_coeval_against_direct(init_box, perturb_field, ionize_box):
    init, pf, ib, bt = wrapper.run_coeval(
        perturb=perturb_field,
        init_box=init_box,
        write=False
    )

    assert init == init_box
    assert pf[0] == perturb_field
    assert ib[0] == ionize_box


def test_lightcone(init_box, perturb_field):
    lc = wrapper.run_lightcone(
        init_box=init_box,
        perturb=perturb_field,
        max_redshift=10.0
    )

    assert lc.lightcone_redshifts[-1] >= 10.0
    assert np.isclose(lc.lightcone_redshifts[0], perturb_field.redshift, atol=1e-4)
    assert lc.cell_size == init_box.user_params.BOX_LEN / init_box.user_params.HII_DIM


def test_run_lf():
    muv, mhalo, lf = wrapper.compute_luminosity_function(redshifts=[7,8,9], nbins=100)
    assert np.all(lf[~np.isnan(lf)] > -30)
    assert lf.shape == (3, 100)

    # Check that memory is in-tact and a second run also works:
    muv, mhalo, lf2 = wrapper.compute_luminosity_function(redshifts=[7, 8, 9], nbins=100)
    assert lf2.shape == (3, 100)
    assert np.allclose(lf2[~np.isnan(lf2)], lf[~np.isnan(lf)])