"""
Tests of single-field low-level functions.

These are designed to be unit-tests of the wrapper functionality. They do not test for
correctness of simulations, but whether different parameter options work/don't work as
intended.
"""

import numpy as np
import pytest

import py21cmfast as p21c
from py21cmfast import BrightnessTemp, InitialConditions, IonizedBox, OutputCache, TsBox


@pytest.fixture(scope="module")
def ic_newseed(default_input_struct, cache: p21c.OutputCache):
    return p21c.compute_initial_conditions(
        inputs=default_input_struct.clone(random_seed=33), write=True, cache=cache
    )


@pytest.fixture(scope="module")
def perturb_field_lowz(ic: InitialConditions, low_redshift: float, cache: OutputCache):
    """A default perturb_field."""
    return p21c.perturb_field(
        redshift=low_redshift,
        initial_conditions=ic,
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
    """An example spin temperature evolution."""
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


def test_perturb_field_ic(perturbed_field, ic):
    # this will run perturb_field again,
    # it should produce exactly the same as the default perturb_field since it has
    # the same seed.
    pf = p21c.perturb_field(
        redshift=perturbed_field.redshift,
        initial_conditions=ic,
        regenerate=True,
    )

    assert pf.density.shape == ic.lowres_density.shape
    assert pf.cosmo_params == ic.cosmo_params
    assert pf.simulation_options == ic.simulation_options
    assert not np.all(pf.density == 0)

    assert pf.simulation_options == perturbed_field.simulation_options
    assert pf.cosmo_params == perturbed_field.cosmo_params

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
    with pytest.raises(
        ValueError,
        match="InputParameters in perturbed_field do not match those in initial_conditions",
    ):
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
    assert not np.all(
        ib.neutral_fraction.value == ionize_box_lowz.neutral_fraction.value
    )


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

    # this will fail because ionized_box was not created with spin temperature.
    with pytest.raises(
        ValueError,
        match="InputParameters in spin_temp do not match those in ionized_box",
    ):
        p21c.brightness_temperature(
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


def test_parameter_override(
    ic: p21c.InitialConditions,
    default_input_struct: p21c.InputParameters,
    ionize_box: p21c.IonizedBox,
    perturbed_field: p21c.PerturbedField,
):
    """Ensure that we use the correct parameters for all calculations.

    Tests compatible but unequal parameter sets
    """
    inputs_changenodes = default_input_struct.clone(node_redshifts=(12, 10, 8))

    pf = p21c.perturb_field(
        redshift=12.0, initial_conditions=ic, inputs=inputs_changenodes
    )

    assert isinstance(pf, p21c.PerturbedField)
    assert pf.inputs == inputs_changenodes
    assert pf.inputs != ic.inputs
    assert pf != perturbed_field

    inputs_changeastro = inputs_changenodes.evolve_input_structs(F_STAR10=-3.0)

    ib = p21c.compute_ionization_field(
        initial_conditions=ic,
        perturbed_field=pf,
        inputs=inputs_changeastro,
        write=False,
    )

    assert isinstance(ib, p21c.IonizedBox)
    assert ib.inputs == inputs_changeastro
    assert ib.inputs != pf.inputs
    assert ib != ionize_box


def test_incompatible_parameters(
    ic: p21c.InitialConditions,
    default_input_struct: p21c.InputParameters,
    ionize_box: p21c.IonizedBox,
    perturbed_field: p21c.PerturbedField,
):
    """Ensure that we throw errors when incompatible parameters are given."""
    df_dim = default_input_struct.simulation_options.DIM
    inputs_changedim = default_input_struct.evolve_input_structs(DIM=df_dim + 1)
    with pytest.raises(
        ValueError, match="InputParameters in InitialConditions do not match those in"
    ):
        p21c.perturb_field(
            redshift=10.0, initial_conditions=ic, inputs=inputs_changedim
        )

    inputs_changenodes = default_input_struct.clone(node_redshifts=(12, 10, 8))
    with pytest.raises(
        ValueError, match="InputParameters in PerturbedField do not match those in"
    ):
        p21c.compute_ionization_field(
            initial_conditions=ic,
            perturbed_field=perturbed_field,
            inputs=inputs_changenodes,
            write=False,
        )


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


def test_incompatible_redshifts(default_input_struct, ic):
    """Test whether bad redshifts are handled correctly."""
    inputs = default_input_struct.clone(node_redshifts=np.array([16.0, 14.0, 12.0]))

    # generate some PetrurbedFields to use
    # NOTE: the checks still happen even if the field is not used due to flags
    ptb_list = [
        p21c.perturb_field(
            initial_conditions=ic, redshift=z, inputs=inputs, write=False
        )
        for z in inputs.node_redshifts
    ]

    kw = {
        "initial_conditions": ic,
        "inputs": inputs,
    }
    # try passing the current redshift == previous
    with pytest.raises(ValueError, match="Incompatible redshifts with inputs"):
        p21c.compute_ionization_field(
            perturbed_field=ptb_list[1],
            previous_perturbed_field=ptb_list[1],
            **kw,
        )

    # try passing the previous redshift < current
    with pytest.raises(ValueError, match="Incompatible redshifts with inputs"):
        p21c.compute_ionization_field(
            perturbed_field=ptb_list[1],
            previous_ionize_box=ptb_list[2],
            **kw,
        )

    # try skipping a step
    with pytest.raises(ValueError, match="Incompatible redshifts with inputs"):
        p21c.compute_ionization_field(
            perturbed_field=ptb_list[2],
            previous_perturbed_field=ptb_list[0],
            **kw,
        )


def test_photoncons_backend_error(redshift, default_input_struct, ic):
    """Test whether the error is raised when you try a photoncons run without proper setup."""
    inputs = default_input_struct.evolve_input_structs(PHOTON_CONS_TYPE="z-photoncons")

    # first test if the error occurs with no inputs
    with pytest.raises(
        ValueError, match="Photon conservation is needed but not initialised."
    ):
        p21c.perturb_field(redshift=redshift, initial_conditions=ic, inputs=inputs)

    # test if the error occurs with the wrong inputs
    p21c.setup_photon_cons(initial_conditions=ic, inputs=inputs)
    with pytest.raises(
        ValueError, match="Photon conservation is needed but not initialised."
    ):
        p21c.perturb_field(
            redshift=redshift,
            initial_conditions=ic,
            inputs=inputs.evolve_input_structs(PHOTON_CONS_TYPE="f-photoncons"),
        )

    # finally test if the calibration works when set up correctly
    p21c.perturb_field(
        redshift=redshift,
        initial_conditions=ic,
        inputs=inputs,
    )


def test_global_properties(
    default_input_struct, ionize_box, spin_temp_evolution, perturbed_field
):
    """Test the global properties which exist in some OutputStruct subclasses."""
    st_new = TsBox.new(inputs=default_input_struct, redshift=10.0)
    with pytest.raises(AttributeError, match="global_Ts is not defined"):
        _ = st_new.global_Ts

    with pytest.raises(AttributeError, match="global_Tk is not defined"):
        _ = st_new.global_Tk

    with pytest.raises(AttributeError, match="global_x_e is not defined"):
        _ = st_new.global_x_e

    ion_new = IonizedBox.new(inputs=default_input_struct, redshift=10.0)
    with pytest.raises(AttributeError, match="global_xH is not defined"):
        _ = ion_new.global_xH

    bt_new = BrightnessTemp.new(inputs=default_input_struct, redshift=10.0)
    with pytest.raises(AttributeError, match="global_Tb is not defined"):
        _ = bt_new.global_Tb

    curr_st = spin_temp_evolution[-1]["spin_temp"]
    assert curr_st.global_Ts == np.mean(curr_st.get("spin_temperature"))
    assert curr_st.global_Tk == np.mean(curr_st.get("kinetic_temp_neutral"))
    assert curr_st.global_x_e == np.mean(curr_st.get("xray_ionised_fraction"))

    assert ionize_box.global_xH == np.mean(ionize_box.get("neutral_fraction"))

    bt = p21c.brightness_temperature(
        ionized_box=ionize_box, perturbed_field=perturbed_field
    )
    assert bt.global_Tb == np.mean(bt.get("brightness_temp"))
