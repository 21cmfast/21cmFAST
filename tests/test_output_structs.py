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
            USE_RELATIVE_VELOCITIES=True,
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
    """Ensure that the correct HaloBox fields are set based on the parameters."""
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=default_input_struct_lc, buffer_size=1
    )
    assert isinstance(pert_halo_cat.halo_masses, Array)
    assert isinstance(pert_halo_cat.halo_coords, Array)
    assert isinstance(pert_halo_cat.halo_masses, Array)
    assert isinstance(pert_halo_cat.halo_coords, Array)
    assert isinstance(pert_halo_cat.stellar_masses, Array)
    assert isinstance(pert_halo_cat.ion_emissivity, Array)
    assert pert_halo_cat.xray_emissivity is None
    assert pert_halo_cat.fesc_sfr is None
    assert pert_halo_cat.stellar_mini is None
    assert pert_halo_cat.sfr_mini is None

    inputs = default_input_struct_lc.evolve_input_structs(USE_TS_FLUCT=True)
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert isinstance(pert_halo_cat.xray_emissivity, Array)
    inputs = inputs.evolve_input_structs(RECOMB_MODEL="inhomogeneous")
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert isinstance(pert_halo_cat.fesc_sfr, Array)
    inputs = inputs.evolve_input_structs(USE_MINI_HALOS=True)
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert isinstance(pert_halo_cat.stellar_mini, Array)
    assert isinstance(pert_halo_cat.sfr_mini, Array)


def test_optional_field_halobox(default_input_struct_lc: InputParameters):
    """Ensure that the correct HaloBox fields are set based on the parameters."""
    hb = ox.HaloBox.new(redshift=0.0, inputs=default_input_struct_lc)
    assert hb.halo_mass is None
    assert isinstance(hb.halo_sfr, Array)
    assert isinstance(hb.n_ion, Array)
    assert hb.halo_sfr_mini is None
    assert hb.halo_xray is None
    assert hb.whalo_sfr is None

    with config.use(EXTRA_HALOBOX_FIELDS=True):
        hb = ox.HaloBox.new(redshift=0.0, inputs=default_input_struct_lc)
        assert isinstance(hb.halo_mass, Array)
        assert isinstance(hb.count, Array)

        inputs = default_input_struct_lc.evolve_input_structs(
            RECOMB_MODEL="inhomogeneous"
        )
        hb = ox.HaloBox.new(redshift=0.0, inputs=inputs)
        assert isinstance(hb.whalo_sfr, Array)

        inputs = inputs.evolve_input_structs(USE_TS_FLUCT=True)
        hb = ox.HaloBox.new(redshift=0.0, inputs=inputs)
        assert isinstance(hb.halo_xray, Array)

        inputs = inputs.evolve_input_structs(USE_MINI_HALOS=True)
        hb = ox.HaloBox.new(redshift=0.0, inputs=inputs)
        assert isinstance(hb.halo_sfr_mini, Array)


def test_optional_field_xrs(default_input_struct_lc: InputParameters):
    """Ensure that the correct XraySourceBox fields are set based on the parameters."""
    xr = ox.XraySourceBox.new(redshift=0.0, inputs=default_input_struct_lc)
    assert isinstance(xr.filtered_sfr, Array)
    assert isinstance(xr.filtered_xray, Array)
    assert xr.filtered_sfr_mini is None

    inputs = default_input_struct_lc.evolve_input_structs(
        USE_TS_FLUCT=True,
        USE_MINI_HALOS=True,
        RECOMB_MODEL="inhomogeneous",
    )
    xr = ox.XraySourceBox.new(redshift=0.0, inputs=inputs)
    assert isinstance(xr.filtered_sfr_mini, Array)


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
        USE_MINI_HALOS=True,
    )
    ts = ox.TsBox.new(redshift=0.0, inputs=inputs)
    assert isinstance(ts.J_21_LW, Array)


def test_optional_field_ion(default_input_struct_lc: InputParameters):
    """Ensure that the correct IonizedBox fields are set based on the parameters."""
    ion = ox.IonizedBox.new(redshift=0.0, inputs=default_input_struct_lc)
    assert isinstance(ion.neutral_fraction, Array)
    assert ion.unnormalised_nion_mini is None
    assert ion.cumulative_recombinations is None

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous",
    )
    ion = ox.IonizedBox.new(redshift=0.0, inputs=inputs)
    assert isinstance(ion.cumulative_recombinations, Array)

    inputs = inputs.evolve_input_structs(
        USE_TS_FLUCT=True,
        USE_MINI_HALOS=True,
    )
    ion = ox.IonizedBox.new(redshift=0.0, inputs=inputs)
    assert isinstance(ion.unnormalised_nion_mini, Array)


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
