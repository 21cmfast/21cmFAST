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
from py21cmfast import InitialConditions, OutputCache, TsBox


@pytest.fixture(scope="module")
def ic_newseed(default_input_struct, cache: p21c.OutputCache):
    return p21c.compute_initial_conditions(
        inputs=default_input_struct.clone(random_seed=33), write=True, cache=cache
    )


@pytest.fixture(scope="module")
def perturb_field_lowz(ic: InitialConditions, low_redshift: float, cache: OutputCache):
    """A default perturb_field"""
    return p21c.perturb_field(
        redshift=low_redshift,
        initial_conditions=ic,
        write=True,
        cache=cache,
    )


@pytest.fixture(scope="module")
def ionize_box(
    ic: InitialConditions,
    perturbed_field: p21c.PerturbedField,
    cache: OutputCache,
):
    """A default ionize_box"""
    return p21c.compute_ionization_field(
        initial_conditions=ic,
        perturbed_field=perturbed_field,
        write=True,
        cache=cache,
    )


@pytest.fixture(scope="module")
def ionize_box_lowz(
    ic: InitialConditions,
    perturb_field_lowz: p21c.PerturbedField,
    cache: OutputCache,
):
    """A default ionize_box at lower redshift."""
    return p21c.compute_ionization_field(
        initial_conditions=ic,
        perturbed_field=perturb_field_lowz,
        write=True,
        cache=cache,
    )


@pytest.fixture(scope="module")
def spin_temp_evolution(ic: InitialConditions, default_input_struct_ts: TsBox, cache):
    """An example spin temperature evolution"""
    scrollz = default_input_struct_ts.node_redshifts
    st_prev = None
    outputs = []
    for z in scrollz:
        pt = p21c.perturb_field(
            redshift=z,
            initial_conditions=ic,
            inputs=default_input_struct_ts,
            cache=cache,
        )
        st = p21c.compute_spin_temperature(
            initial_conditions=ic,
            perturbed_field=pt,
            previous_spin_temp=st_prev,
            inputs=default_input_struct_ts,
            cache=cache,
        )
        outputs.append(
            {
                "redshift": z,
                "perturbed_field": pt,
                "spin_temp": st,
            }
        )
        st_prev = st

    return outputs


def test_pf_unnamed_param():
    """Try using an un-named parameter."""
    with pytest.raises(TypeError):
        p21c.perturb_field(7)


def test_perturb_field_ic(perturbed_field, default_input_struct, ic, cache):
    # this will run perturb_field again, since by default regenerate=True for tests.
    # BUT it should produce exactly the same as the default perturb_field since it has
    # the same seed.
    pf = p21c.perturb_field(
        redshift=perturbed_field.redshift, initial_conditions=ic, cache=cache
    )

    assert pf.density.shape == ic.lowres_density.shape
    assert pf.cosmo_params == ic.cosmo_params
    assert pf.user_params == ic.user_params
    assert not np.all(pf.density == 0)

    assert pf.user_params == perturbed_field.user_params
    assert pf.cosmo_params == perturbed_field.cosmo_params

    assert pf == perturbed_field


def test_cache_exists(default_input_struct, perturbed_field, cache):
    pf = p21c.PerturbedField.new(
        redshift=perturbed_field.redshift,
        inputs=default_input_struct,
    )

    assert cache.find_existing(pf) is not None

    pf = cache.load(pf)
    pf.load_all()
    np.testing.assert_allclose(pf.density.value, perturbed_field.density.value)
    assert pf == perturbed_field


def test_new_seeds(
    ic_newseed,
    perturb_field_lowz,
    ionize_box_lowz,
    default_input_struct,
    cache,
):
    # Perturbed Field
    pf = p21c.perturb_field(
        redshift=perturb_field_lowz.redshift, initial_conditions=ic_newseed, cache=cache
    )

    # we didn't write it, and this has a different seed
    assert cache.find_existing(pf) is None
    assert pf.random_seed != perturb_field_lowz.random_seed
    assert not np.all(pf.density.value == perturb_field_lowz.density.value)

    # Ionization Box
    with pytest.raises(ValueError):
        p21c.compute_ionization_field(
            initial_conditions=ic_newseed,
            perturbed_field=perturb_field_lowz,
            cache=cache,
        )

    ib = p21c.compute_ionization_field(
        initial_conditions=ic_newseed, perturbed_field=pf, cache=cache
    )

    # we didn't write it, and this has a different seed
    assert cache.find_existing(ib) is None
    assert ib.random_seed != ionize_box_lowz.random_seed
    assert not np.all(ib.xH_box.value == ionize_box_lowz.xH_box.value)


def test_ib_from_pf(perturbed_field, ic, cache):
    ib = p21c.compute_ionization_field(
        initial_conditions=ic, perturbed_field=perturbed_field, cache=cache
    )
    assert ib.redshift == perturbed_field.redshift
    assert ib.inputs == perturbed_field.inputs


def test_ib_bad_st(ic, default_input_struct, perturbed_field, redshift, cache):
    with pytest.raises(TypeError, match="spin_temp should be of type TsBox"):
        p21c.compute_ionization_field(
            inputs=default_input_struct,
            initial_conditions=ic,
            perturbed_field=perturbed_field,
            spin_temp=ic,
            cache=cache,
        )


def test_bt(
    ionize_box, default_input_struct, spin_temp_evolution, perturbed_field, cache
):
    curr_st = spin_temp_evolution[-1]["spin_temp"]
    # with pytest.raises(TypeError):  # have to specify param names
    #     p21c.brightness_temperature(
    #         default_input_struct, ionize_box, curr_st, perturbed_field
    #     )

    # this will fail because ionized_box was not created with spin temperature.
    with pytest.raises(ValueError):
        p21c.brightness_temperature(
            #            inputs=default_input_struct,
            ionized_box=ionize_box,
            perturbed_field=perturbed_field,
            spin_temp=curr_st,
            cache=cache,
        )

    bt = p21c.brightness_temperature(
        ionized_box=ionize_box, perturbed_field=perturbed_field, cache=cache
    )

    assert bt.inputs == perturbed_field.inputs


def test_coeval_against_direct(
    redshift: float,
    ic: p21c.InitialConditions,
    perturbed_field: p21c.PerturbedField,
    ionize_box: p21c.IonizedBox,
    cache,
):
    [coeval] = p21c.run_coeval(
        out_redshifts=redshift, initial_conditions=ic, cache=cache
    )

    assert coeval.initial_conditions == ic
    assert coeval.perturbed_field == perturbed_field
    assert coeval.ionized_box == ionize_box


def test_using_cached_halo_field(ic, test_direc):
    """Test whether the C-based memory in halo fields is cached correctly.

    Prior to v3.1 this was segfaulting, so this test ensure that this behaviour does
    not regress.
    """
    cache = OutputCache(test_direc)
    halo_field = p21c.determine_halo_list(
        redshift=10.0, initial_conditions=ic, write=True, cache=cache
    )

    pt_halos = p21c.perturb_halo_list(
        initial_conditions=ic,
        halo_field=halo_field,
        write=True,
        cache=cache,
    )

    # Now get the halo field again at the same redshift -- should be cached
    new_halo_field = p21c.determine_halo_list(
        redshift=10.0,
        initial_conditions=ic,
        write=False,
        regenerate=False,
    )

    new_pt_halos = p21c.perturb_halo_list(
        initial_conditions=ic,
        halo_field=new_halo_field,
        write=False,
        regenerate=False,
    )

    np.testing.assert_allclose(
        new_halo_field.halo_masses.value, halo_field.halo_masses.value
    )
    np.testing.assert_allclose(
        pt_halos.halo_coords.value, new_pt_halos.halo_coords.value
    )
