"""Unit tests for output structures."""

import pickle

import attrs
import numpy as np
import pytest

from py21cmfast import (
    InitialConditions,  # An example of an output struct
    InputParameters,
    OutputCache,
    compute_initial_conditions,
    config,
    determine_halo_catalog,
    perturb_halo_catalog,
)
from py21cmfast.wrapper import outputs as ox
from py21cmfast.wrapper.arrays import Array


@pytest.fixture
def init(default_input_struct: InputParameters):
    return InitialConditions.new(inputs=default_input_struct)


@pytest.fixture(scope="module")
def ic_with_halos(default_input_struct, cache: OutputCache):
    return compute_initial_conditions(
        inputs=default_input_struct.evolve_input_structs(SOURCE_MODEL="CHMF-SAMPLER"),
        write=True,
        cache=cache,
    )


@pytest.fixture
def halo_cat(ic_with_halos: InitialConditions, default_input_struct: InputParameters):
    return determine_halo_catalog(
        redshift=10.0,
        initial_conditions=ic_with_halos,
        inputs=default_input_struct.evolve_input_structs(SOURCE_MODEL="CHMF-SAMPLER"),
    )


@pytest.fixture
def pert_halo_cat(ic_with_halos: InitialConditions, halo_cat: ox.HaloCatalog):
    return perturb_halo_catalog(
        initial_conditions=ic_with_halos,
        halo_catalog=halo_cat,
    )


def test_different_seeds(
    init: InitialConditions,
    default_input_struct: InputParameters,
):
    ic2 = InitialConditions.new(
        inputs=default_input_struct.clone(
            random_seed=default_input_struct.random_seed + 1
        )
    )

    assert init is not ic2
    assert init != ic2

    # make sure we didn't inadvertantly set the random seed while doing any of this
    assert init.random_seed == default_input_struct.random_seed


def test_pickleability(default_input_struct: InputParameters):
    ic_ = InitialConditions.new(inputs=default_input_struct)
    s = pickle.dumps(ic_)

    ic2 = pickle.loads(s)
    assert repr(ic_) == repr(ic2)


def test_reading_purged(ic: InitialConditions):
    lowres_density = ic.get(ic.lowres_density)

    # Remove it from memory
    ic.purge()

    assert not ic.lowres_density.state.computed_in_mem
    assert ic.lowres_density.state.on_disk

    # But we can still get it.
    lowres_density_2 = ic.get(ic.lowres_density)

    assert ic.lowres_density.state.on_disk
    assert ic.lowres_density.state.computed_in_mem

    assert np.allclose(lowres_density_2, lowres_density)

    ic.load_all()


def test_direct_array_access_after_purge_loads_transparently(ic: InitialConditions):
    """Plain attribute access should transparently resolve a purged array.

    Not via `.get()` - and cache it back onto the struct by default (see #565).
    """
    expected = ic.get(ic.lowres_density)

    ic.purge()
    assert not ic.lowres_density.state.computed_in_mem

    assert ic.lowres_density.mean() == pytest.approx(expected.mean())
    assert np.allclose(np.asarray(ic.lowres_density), expected)

    # Cached back onto the struct as a side effect of the access above.
    assert ic.lowres_density.state.computed_in_mem

    ic.load_all()


def test_direct_array_access_after_purge_respects_no_cache_config(
    ic: InitialConditions,
):
    """Direct access still resolves the value with auto-caching disabled.

    But it must not repopulate memory on the struct.
    """
    expected = ic.get(ic.lowres_density)
    ic.purge()

    with config.use(CACHE_ARRAYS_ON_ACCESS=False):
        assert ic.lowres_density.mean() == pytest.approx(expected.mean())
        assert not ic.lowres_density.state.computed_in_mem

    ic.load_all()


def test_repr_of_purged_struct_array_does_not_load_it(ic: InitialConditions):
    """Merely inspecting a purged array (e.g. in a REPL) must stay cheap."""
    ic.purge()
    assert not ic.lowres_density.state.computed_in_mem

    _ = repr(ic.lowres_density)

    assert not ic.lowres_density.state.computed_in_mem
    assert ic.lowres_density.value is None

    ic.load_all()


@pytest.mark.parametrize("struct", list(ox._ALL_OUTPUT_STRUCTS.values()))
def test_all_fields_exist(struct: ox.OutputStruct):
    cstruct = ox.StructWrapper(struct.__name__)

    this = attrs.fields_dict(struct)

    # Ensure that all fields in the cstruct are also defined on this class.
    for name in cstruct.pointer_fields:
        assert name in this
        assert this[name].type == ox.Array

    for name in cstruct.primitive_fields:
        assert name in this


def test_halocatalogs(default_input_struct_lc: InputParameters):
    """Ensure that the halo catalogs can be made."""
    # First let's define buffer_size
    inputs = default_input_struct_lc.evolve_input_structs(SOURCE_MODEL="CHMF-SAMPLER")
    halo_cat = ox.HaloCatalog.new(redshift=0.0, inputs=inputs, buffer_size=1)
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert isinstance(halo_cat, ox.HaloCatalog)
    assert isinstance(pert_halo_cat, ox.PerturbedHaloCatalog)

    # Now let's not define buffer_size, it should default to None
    halo_cat = ox.HaloCatalog.new(
        redshift=0.0,
        inputs=inputs,  # buffer_size = None
    )
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0,
        inputs=inputs,  # buffer_size = None
    )
    assert isinstance(halo_cat, ox.HaloCatalog)
    assert isinstance(pert_halo_cat, ox.PerturbedHaloCatalog)

    # Now let's define dummy=True
    halo_cat = ox.HaloCatalog.new(redshift=0.0, inputs=inputs, dummy=True)
    pert_halo_cat = ox.PerturbedHaloCatalog.new(redshift=0.0, inputs=inputs, dummy=True)
    assert isinstance(halo_cat, ox.HaloCatalog)
    assert isinstance(pert_halo_cat, ox.PerturbedHaloCatalog)
    assert halo_cat.buffer_size == 0
    assert pert_halo_cat.buffer_size == 0


# NOTE: These do not test every field, but does test every conditional in the
#   OutputStruct constructors, a better approach would probably be to have a
#   comprehensive list of {"field_name": {"flag": value}} conditions for the fields
#   in the output module which is checked in the constructors
def test_optional_field_ic(default_input_struct_lc: InputParameters):
    """Ensure that the correct InitialConditions fields are set based on the parameters."""
    ic = ox.InitialConditions.new(inputs=default_input_struct_lc)
    assert isinstance(ic.lowres_vx, Array)
    assert isinstance(ic.lowres_vx_2LPT, Array)
    assert ic.hires_vx is None
    assert isinstance(ic.hires_vx_2LPT, Array)  # Python requires it, check the C
    assert ic.lowres_vcb is None

    ic = ox.InitialConditions.new(
        inputs=default_input_struct_lc.evolve_input_structs(
            PERTURB_ALGORITHM="ZELDOVICH"
        )
    )
    assert isinstance(ic.lowres_vy, Array)
    assert ic.lowres_vy_2LPT is None
    assert ic.hires_vy is None
    assert ic.hires_vy_2LPT is None

    ic = ox.InitialConditions.new(
        inputs=default_input_struct_lc.evolve_input_structs(PERTURB_ON_HIGH_RES=True)
    )
    assert ic.lowres_vz is None
    assert ic.lowres_vz_2LPT is None
    assert isinstance(ic.hires_vz, Array)
    assert isinstance(ic.hires_vz_2LPT, Array)

    ic = ox.InitialConditions.new(
        inputs=default_input_struct_lc.evolve_input_structs(
            V_CB_MODEL="FLUCTS",
            POWER_SPECTRUM="CLASS",
        )
    )
    assert isinstance(ic.lowres_vx, Array)
    assert isinstance(ic.lowres_vx_2LPT, Array)
    assert ic.hires_vx is None
    assert isinstance(ic.hires_vx_2LPT, Array)
    assert isinstance(ic.lowres_vcb, Array)


def test_optional_field_perturb(default_input_struct_lc: InputParameters):
    """Ensure that the correct PerturbedField fields are set based on the parameters."""
    pt = ox.PerturbedField.new(redshift=0.0, inputs=default_input_struct_lc)
    assert isinstance(pt.density, Array)
    assert isinstance(pt.velocity_z, Array)
    assert isinstance(pt.velocity_x, Array)
    assert isinstance(pt.velocity_y, Array)

    pt = ox.PerturbedField.new(
        redshift=0.0,
        inputs=default_input_struct_lc.evolve_input_structs(KEEP_3D_VELOCITIES=False),
    )
    assert isinstance(pt.density, Array)
    assert isinstance(pt.velocity_z, Array)
    assert pt.velocity_x is None
    assert pt.velocity_y is None


def test_optional_field_perturbed_halocat(default_input_struct_lc: InputParameters):
    """Ensure that the correct EmissivityFields fields are set based on the parameters."""
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=default_input_struct_lc, buffer_size=1
    )
    assert isinstance(pert_halo_cat.halo_masses, Array)
    assert isinstance(pert_halo_cat.halo_coords, Array)
    assert isinstance(pert_halo_cat.halo_masses, Array)
    assert isinstance(pert_halo_cat.halo_coords, Array)
    assert isinstance(pert_halo_cat.stellar_masses_acg, Array)
    assert isinstance(pert_halo_cat.n_ion, Array)
    assert pert_halo_cat.xray_luminosity is None
    assert pert_halo_cat.fesc_weighted_sfr is None
    assert pert_halo_cat.stellar_masses_mcg is None
    assert pert_halo_cat.sfr_mcg is None

    inputs = default_input_struct_lc.evolve_input_structs(USE_TS_FLUCT=True)
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert isinstance(pert_halo_cat.xray_luminosity, Array)
    inputs = inputs.evolve_input_structs(RECOMB_MODEL="inhomogeneous")
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert isinstance(pert_halo_cat.fesc_weighted_sfr, Array)
    inputs = inputs.evolve_input_structs(USE_MCGS=True)
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert isinstance(pert_halo_cat.stellar_masses_mcg, Array)
    assert isinstance(pert_halo_cat.sfr_mcg, Array)


def test_optional_emissivity_fields(default_input_struct_lc: InputParameters):
    """Ensure that the correct EmissivityFields fields are set based on the parameters."""
    emissivity_fields = ox.EmissivityFields.new(
        redshift=0.0, inputs=default_input_struct_lc
    )
    assert emissivity_fields.halo_number is None
    assert emissivity_fields.halo_mass_density is None
    assert emissivity_fields.stellar_mass_density_acg is None
    assert emissivity_fields.stellar_mass_density_mcg is None
    assert emissivity_fields.sfrd_acg is None
    assert emissivity_fields.sfrd_mcg is None
    assert emissivity_fields.xray_emissivity is None
    assert emissivity_fields.fesc_weighted_sfrd is None
    assert isinstance(emissivity_fields.n_ion, Array)

    with config.use(EXTRA_EMISSIVITY_FIELDS=True):
        emissivity_fields = ox.EmissivityFields.new(
            redshift=0.0, inputs=default_input_struct_lc
        )
        assert isinstance(emissivity_fields.halo_mass_density, Array)
        assert isinstance(emissivity_fields.halo_number, Array)
        assert isinstance(emissivity_fields.stellar_mass_density_acg, Array)
        assert emissivity_fields.stellar_mass_density_mcg is None

        emissivity_fields = ox.EmissivityFields.new(
            redshift=0.0,
            inputs=default_input_struct_lc.evolve_input_structs(
                USE_TS_FLUCT=True, RECOMB_MODEL="inhomogeneous", USE_MCGS=True
            ),
        )
        assert isinstance(emissivity_fields.stellar_mass_density_mcg, Array)

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous", SOURCE_MODEL="L-INTEGRAL"
    )
    emissivity_fields = ox.EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert isinstance(emissivity_fields.fesc_weighted_sfrd, Array)

    inputs = inputs.evolve_input_structs(USE_TS_FLUCT=True)
    emissivity_fields = ox.EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert isinstance(emissivity_fields.sfrd_acg, Array)
    assert isinstance(emissivity_fields.xray_emissivity, Array)

    inputs = inputs.evolve_input_structs(USE_MCGS=True)
    emissivity_fields = ox.EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert isinstance(emissivity_fields.sfrd_mcg, Array)


def test_optional_setup_radiation_fields(default_input_struct_lc: InputParameters):
    """Ensure that the correct fields of RadiationFieldsSetup are set based on the parameters."""
    rfs = ox.RadiationFieldsSetup.new(redshift=0.0, inputs=default_input_struct_lc)
    assert isinstance(rfs.filtered_sfrd_acg_for_lya, Array)
    assert isinstance(rfs.filtered_xray_emissivity, Array)
    assert rfs.filtered_sfrd_mcg_for_lya is None

    inputs = default_input_struct_lc.evolve_input_structs(
        USE_TS_FLUCT=True,
        USE_MCGS=True,
        RECOMB_MODEL="inhomogeneous",
    )
    rfs = ox.RadiationFieldsSetup.new(redshift=0.0, inputs=inputs)
    assert isinstance(rfs.filtered_sfrd_mcg_for_lya, Array)


def test_optional_field_ts(default_input_struct_lc: InputParameters):
    """Ensure that the correct TsBox fields are set based on the parameters."""
    ts = ox.TsBox.new(redshift=0.0, inputs=default_input_struct_lc)
    assert isinstance(ts.spin_temperature, Array)
    assert isinstance(ts.xray_ionised_fraction, Array)
    assert isinstance(ts.kinetic_temp_neutral, Array)
    assert ts.J_21_LW is None

    inputs = default_input_struct_lc.evolve_input_structs(
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
        USE_MCGS=True,
    )
    ts = ox.TsBox.new(redshift=0.0, inputs=inputs)
    assert isinstance(ts.J_21_LW, Array)


def test_optional_field_ion(default_input_struct_lc: InputParameters):
    """Ensure that the correct IonizedBox fields are set based on the parameters."""
    ion = ox.IonizedBox.new(redshift=0.0, inputs=default_input_struct_lc)
    assert isinstance(ion.neutral_fraction, Array)
    assert ion.nion_conditional_filtered_mcg is None
    assert ion.cumulative_recombinations is None

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous",
    )
    ion = ox.IonizedBox.new(redshift=0.0, inputs=inputs)
    assert isinstance(ion.cumulative_recombinations, Array)

    inputs = inputs.evolve_input_structs(
        USE_TS_FLUCT=True,
        USE_MCGS=True,
    )
    ion = ox.IonizedBox.new(redshift=0.0, inputs=inputs)
    assert isinstance(ion.nion_conditional_filtered_mcg, Array)


def test_optional_field_bt(default_input_struct_lc: InputParameters):
    """Ensure that the correct BrightnessTemp fields are set based on the parameters."""
    bt = ox.BrightnessTemp.new(redshift=0.0, inputs=default_input_struct_lc)
    assert isinstance(bt.brightness_temp, Array)
    assert bt.tau_21 is None

    inputs = default_input_struct_lc.evolve_input_structs(USE_TS_FLUCT=True)
    bt = ox.BrightnessTemp.new(redshift=0.0, inputs=inputs)
    assert isinstance(bt.tau_21, Array)


@pytest.mark.parametrize("struct", list(ox.OutputStructZ.__subclasses__()))
def test_bad_required_array(default_input_struct, struct):
    # no struct takes this input
    inputs = default_input_struct.evolve_input_structs(SOURCE_MODEL="CHMF-SAMPLER")
    bt = ox.BrightnessTemp.new(redshift=10.0, inputs=inputs)
    kwargs = {"inputs": inputs, "redshift": 10.0}
    if struct is ox.PerturbedHaloCatalog:
        kwargs["buffer_size"] = 1
    output = struct.new(**kwargs)

    with pytest.raises(ValueError, match="is not an input required for"):
        _ = output.get_required_input_arrays(bt)


def test_halocatalog_iteration(halo_cat: ox.HaloCatalog):
    """Test HaloCatalog iteration, len, and indexing."""
    # Test len
    assert len(halo_cat) == halo_cat.n_halos

    # Test iteration and indexing
    halo_list = []
    for halo in halo_cat:
        halo_list.append(halo)
        assert isinstance(halo, ox.Halo)
        assert halo.mass is not None
        assert halo.coords is not None
        assert halo.star_rng is not None
        assert halo.sfr_rng is not None
        assert halo.xray_rng is not None
        assert halo.redshift is not None

    assert len(halo_list) == halo_cat.n_halos

    # Test indexing
    first_halo = halo_cat[0]
    assert isinstance(first_halo, ox.Halo)
    assert first_halo.mass == halo_list[0].mass
    assert np.all(first_halo.coords == halo_list[0].coords)


def test_perturbed_halocatalog_iteration(pert_halo_cat: ox.PerturbedHaloCatalog):
    """Test PerturbedHaloCatalog iteration, len, and indexing."""
    # Test len
    assert len(pert_halo_cat) == pert_halo_cat.n_halos

    # Test iteration and indexing
    halo_list = []
    for halo in pert_halo_cat:
        halo_list.append(halo)
        assert isinstance(halo, ox.Halo)
        assert halo.mass is not None
        assert halo.coords is not None
        assert halo.redshift is not None

    assert len(halo_list) == pert_halo_cat.n_halos

    # Test indexing
    first_halo = pert_halo_cat[0]
    assert isinstance(first_halo, ox.Halo)
    assert first_halo.mass == halo_list[0].mass
    assert np.all(first_halo.coords == halo_list[0].coords)


def test_bad_indices_for_halocatalog(
    halo_cat: ox.HaloCatalog, pert_halo_cat: ox.PerturbedHaloCatalog
):
    """Test bad indices for halo catalog and perturbed halo catalog."""
    with pytest.raises(IndexError, match=f"Halo index {halo_cat.n_halos} out of range"):
        halo_cat[halo_cat.n_halos]
    with pytest.raises(IndexError, match="Halo index -1 out of range"):
        halo_cat[-1]
    with pytest.raises(
        IndexError, match=f"Halo index {pert_halo_cat.n_halos} out of range"
    ):
        pert_halo_cat[pert_halo_cat.n_halos]
    with pytest.raises(IndexError, match="Halo index -1 out of range"):
        pert_halo_cat[-1]
