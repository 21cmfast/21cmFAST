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
    lowres_density = ic.get("lowres_density")

    # Remove it from memory
    ic.purge()

    assert not ic.arrays["lowres_density"].state.computed_in_mem
    assert ic.arrays["lowres_density"].state.on_disk

    # But we can still get it.
    lowres_density_2 = ic.get("lowres_density")

    assert ic.arrays["lowres_density"].state.on_disk
    assert ic.arrays["lowres_density"].state.computed_in_mem

    assert np.allclose(lowres_density_2, lowres_density)

    ic.load_all()


def test_direct_array_access_after_purge_loads_transparently(ic: InitialConditions):
    """Plain attribute access resolves a purged array, and caches it back (see #565)."""
    expected = ic.get("lowres_density")

    ic.purge()
    assert not ic.arrays["lowres_density"].state.computed_in_mem

    assert ic.lowres_density.mean() == pytest.approx(expected.mean())
    assert np.allclose(ic.lowres_density, expected)

    # Cached back onto the struct as a side effect of the access above.
    assert ic.arrays["lowres_density"].state.computed_in_mem

    ic.load_all()


def test_direct_array_access_gives_a_real_ndarray(ic: InitialConditions):
    """The attribute must behave as an array in *every* respect, not just some.

    Half duck-typing was the original problem: methods worked but operators didn't.
    """
    density = ic.lowres_density

    assert isinstance(density, np.ndarray)
    assert np.all((density - density) == 0.0)
    assert (density**2).mean() >= 0.0
    # `Array == <scalar>` used to silently evaluate to False.
    assert np.all(np.zeros_like(density) == 0.0)
    assert density[0].shape == density.shape[1:]
    assert len(density) == density.shape[0]


def test_direct_array_access_after_purge_respects_no_cache_config(
    ic: InitialConditions,
):
    """Direct access still resolves the value with auto-caching disabled.

    But it must not repopulate memory on the struct.
    """
    expected = ic.get("lowres_density")
    ic.purge()

    with config.use(CACHE_ARRAYS_ON_ACCESS=False):
        assert ic.lowres_density.mean() == pytest.approx(expected.mean())
        assert not ic.arrays["lowres_density"].state.computed_in_mem

    ic.load_all()


def test_repr_of_purged_struct_does_not_load_it(ic: InitialConditions):
    """Merely inspecting a purged struct (e.g. in a REPL) must stay cheap.

    The `Array` lives in a private attrs field, so the generated `__repr__` never
    goes through the descriptor and so never touches the disk.
    """
    ic.purge()
    assert not ic.arrays["lowres_density"].state.computed_in_mem

    _ = repr(ic)
    _ = repr(ic.arrays["lowres_density"])

    assert not ic.arrays["lowres_density"].state.computed_in_mem
    assert ic.arrays["lowres_density"]._value is None

    ic.load_all()


def test_has(ic: InitialConditions):
    """`has()` distinguishes "no data yet" from "this field does not exist"."""
    assert ic.has("lowres_density")
    assert not ic.has("lowres_vcb")  # not produced by these inputs

    ic.purge()
    # Still available: it is on disk, and reading the attribute will load it.
    assert ic.has("lowres_density")
    ic.load_all()

    fresh = ox.InitialConditions.new(inputs=ic.inputs)
    assert not fresh.has("lowres_density")
    with pytest.raises(ValueError, match="not on disk and not initialized"):
        _ = fresh.lowres_density


@pytest.mark.parametrize("struct", list(ox._ALL_OUTPUT_STRUCTS.values()))
def test_all_fields_exist(struct: ox.OutputStruct):
    cstruct = ox.StructWrapper(struct.__name__)

    this = attrs.fields_dict(struct)

    # Ensure that all fields in the cstruct are also defined on this class. Array
    # fields are declared privately and exposed under the public name by a descriptor.
    for name in cstruct.pointer_fields:
        assert name in struct._array_field_names
        assert this[f"_{name}"].type == ox.Array

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
    assert "lowres_vx" in ic.arrays
    assert "lowres_vx_2LPT" in ic.arrays
    assert "hires_vx" not in ic.arrays
    assert "hires_vx_2LPT" in ic.arrays  # Python requires it, check the C
    assert "lowres_vcb" not in ic.arrays

    ic = ox.InitialConditions.new(
        inputs=default_input_struct_lc.evolve_input_structs(
            PERTURB_ALGORITHM="ZELDOVICH"
        )
    )
    assert "lowres_vy" in ic.arrays
    assert "lowres_vy_2LPT" not in ic.arrays
    assert "hires_vy" not in ic.arrays
    assert "hires_vy_2LPT" not in ic.arrays

    ic = ox.InitialConditions.new(
        inputs=default_input_struct_lc.evolve_input_structs(PERTURB_ON_HIGH_RES=True)
    )
    assert "lowres_vz" not in ic.arrays
    assert "lowres_vz_2LPT" not in ic.arrays
    assert "hires_vz" in ic.arrays
    assert "hires_vz_2LPT" in ic.arrays

    with pytest.warns(UserWarning, match="USE_MCGS is False but V_CB_MODEL"):
        ic = ox.InitialConditions.new(
            inputs=default_input_struct_lc.evolve_input_structs(
                V_CB_MODEL="FLUCTS",
                POWER_SPECTRUM="CLASS",
            )
        )
    )
    assert "lowres_vx" in ic.arrays
    assert "lowres_vx_2LPT" in ic.arrays
    assert "hires_vx" not in ic.arrays
    assert "hires_vx_2LPT" in ic.arrays
    assert "lowres_vcb" in ic.arrays


def test_optional_field_perturb(default_input_struct_lc: InputParameters):
    """Ensure that the correct PerturbedField fields are set based on the parameters."""
    pt = ox.PerturbedField.new(redshift=0.0, inputs=default_input_struct_lc)
    assert "density" in pt.arrays
    assert "velocity_z" in pt.arrays
    assert "velocity_x" in pt.arrays
    assert "velocity_y" in pt.arrays

    pt = ox.PerturbedField.new(
        redshift=0.0,
        inputs=default_input_struct_lc.evolve_input_structs(KEEP_3D_VELOCITIES=False),
    )
    assert "density" in pt.arrays
    assert "velocity_z" in pt.arrays
    assert "velocity_x" not in pt.arrays
    assert "velocity_y" not in pt.arrays


@pytest.mark.filterwarnings(
    "ignore:^You are setting R_BUBBLE_MAX != 50 when RECOMB_MODEL:UserWarning"
)
def test_optional_field_perturbed_halocat(default_input_struct_lc: InputParameters):
    """Ensure that the correct EmissivityFields fields are set based on the parameters."""
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=default_input_struct_lc, buffer_size=1
    )
    assert "halo_masses" in pert_halo_cat.arrays
    assert "halo_coords" in pert_halo_cat.arrays
    assert "halo_masses" in pert_halo_cat.arrays
    assert "halo_coords" in pert_halo_cat.arrays
    assert "stellar_masses_acg" in pert_halo_cat.arrays
    assert "n_ion" in pert_halo_cat.arrays
    assert "xray_luminosity" not in pert_halo_cat.arrays
    assert "fesc_weighted_sfr" not in pert_halo_cat.arrays
    assert "stellar_masses_mcg" not in pert_halo_cat.arrays
    assert "sfr_mcg" not in pert_halo_cat.arrays

    inputs = default_input_struct_lc.evolve_input_structs(USE_TS_FLUCT=True)
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert "xray_luminosity" in pert_halo_cat.arrays
    inputs = inputs.evolve_input_structs(RECOMB_MODEL="inhomogeneous")
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert "fesc_weighted_sfr" in pert_halo_cat.arrays
    inputs = inputs.evolve_input_structs(
      USE_MCGS=True,
      V_CB_MODEL="FLUCTS",
      POWER_SPECTRUM="CLASS",
      M_TURN_STELLAR_FEEDBACK=5.0,
    )
    pert_halo_cat = ox.PerturbedHaloCatalog.new(
        redshift=0.0, inputs=inputs, buffer_size=1
    )
    assert "stellar_masses_mcg" in pert_halo_cat.arrays
    assert "sfr_mcg" in pert_halo_cat.arrays


def test_optional_emissivity_fields(default_input_struct_lc: InputParameters):
    """Ensure that the correct EmissivityFields fields are set based on the parameters."""
    emissivity_fields = ox.EmissivityFields.new(
        redshift=0.0, inputs=default_input_struct_lc
    )
    assert "halo_number" not in emissivity_fields.arrays
    assert "halo_mass_density" not in emissivity_fields.arrays
    assert "stellar_mass_density_acg" not in emissivity_fields.arrays
    assert "stellar_mass_density_mcg" not in emissivity_fields.arrays
    assert "sfrd_acg" not in emissivity_fields.arrays
    assert "sfrd_mcg" not in emissivity_fields.arrays
    assert "xray_emissivity" not in emissivity_fields.arrays
    assert "fesc_weighted_sfrd" not in emissivity_fields.arrays
    assert "n_ion" in emissivity_fields.arrays

    with config.use(EXTRA_EMISSIVITY_FIELDS=True):
        emissivity_fields = ox.EmissivityFields.new(
            redshift=0.0, inputs=default_input_struct_lc
        )
        assert "halo_mass_density" in emissivity_fields.arrays
        assert "halo_number" in emissivity_fields.arrays
        assert "stellar_mass_density_acg" in emissivity_fields.arrays
        assert "stellar_mass_density_mcg" not in emissivity_fields.arrays

        emissivity_fields = ox.EmissivityFields.new(
            redshift=0.0,
            inputs=default_input_struct_lc.evolve_input_structs(
                USE_TS_FLUCT=True,
                RECOMB_MODEL="inhomogeneous",
                R_BUBBLE_MAX=50.0,
                USE_MCGS=True,
                V_CB_MODEL="FLUCTS",
                POWER_SPECTRUM="CLASS",
                M_TURN_STELLAR_FEEDBACK=5.0,
            ),
        )
        assert "stellar_mass_density_mcg" in emissivity_fields.arrays

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous", R_BUBBLE_MAX=50.0, SOURCE_MODEL="L-INTEGRAL"
    )
    emissivity_fields = ox.EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert "fesc_weighted_sfrd" in emissivity_fields.arrays

    inputs = inputs.evolve_input_structs(USE_TS_FLUCT=True)
    emissivity_fields = ox.EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert "sfrd_acg" in emissivity_fields.arrays
    assert "xray_emissivity" in emissivity_fields.arrays

    inputs = inputs.evolve_input_structs(
        USE_MCGS=True,
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        M_TURN_STELLAR_FEEDBACK=5.0,
    )
    emissivity_fields = ox.EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert "sfrd_mcg" in emissivity_fields.arrays


@pytest.mark.filterwarnings(
    "ignore:^You are setting R_BUBBLE_MAX != 50 when RECOMB_MODEL:UserWarning"
)
def test_optional_setup_radiation_fields(default_input_struct_lc: InputParameters):
    """Ensure that the correct fields of RadiationFieldsSetup are set based on the parameters."""
    rfs = ox.RadiationFieldsSetup.new(redshift=0.0, inputs=default_input_struct_lc)
    assert "filtered_sfrd_acg_for_lya" in rfs.arrays
    assert "filtered_xray_emissivity" in rfs.arrays
    assert "filtered_sfrd_mcg_for_lya" not in rfs.arrays

    inputs = default_input_struct_lc.evolve_input_structs(
        USE_TS_FLUCT=True,
        USE_MCGS=True,
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        RECOMB_MODEL="inhomogeneous",
        M_TURN_STELLAR_FEEDBACK=5.0,
    )
    rfs = ox.RadiationFieldsSetup.new(redshift=0.0, inputs=inputs)
    assert "filtered_sfrd_mcg_for_lya" in rfs.arrays


@pytest.mark.filterwarnings(
    "ignore:^You are setting R_BUBBLE_MAX != 50 when RECOMB_MODEL:UserWarning"
)
def test_optional_field_ts(default_input_struct_lc: InputParameters):
    """Ensure that the correct TsBox fields are set based on the parameters."""
    ts = ox.TsBox.new(redshift=0.0, inputs=default_input_struct_lc)
    assert "spin_temperature" in ts.arrays
    assert "xray_ionised_fraction" in ts.arrays
    assert "kinetic_temp_neutral" in ts.arrays
    assert "J_21_LW" not in ts.arrays

    inputs = default_input_struct_lc.evolve_input_structs(
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
        USE_MCGS=True,
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        M_TURN_STELLAR_FEEDBACK=5.0,
    )
    ts = ox.TsBox.new(redshift=0.0, inputs=inputs)
    assert "J_21_LW" in ts.arrays


@pytest.mark.filterwarnings(
    "ignore:^You are setting R_BUBBLE_MAX != 50 when RECOMB_MODEL:UserWarning"
)
def test_optional_field_ion(default_input_struct_lc: InputParameters):
    """Ensure that the correct IonizedBox fields are set based on the parameters."""
    ion = ox.IonizedBox.new(redshift=0.0, inputs=default_input_struct_lc)
    assert "neutral_fraction" in ion.arrays
    assert "nion_conditional_filtered_mcg" not in ion.arrays
    assert "cumulative_recombinations" not in ion.arrays

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous",
    )
    ion = ox.IonizedBox.new(redshift=0.0, inputs=inputs)
    assert "cumulative_recombinations" in ion.arrays

    inputs = inputs.evolve_input_structs(
        USE_TS_FLUCT=True,
        USE_MCGS=True,
        V_CB_MODEL="FLUCTS",
        POWER_SPECTRUM="CLASS",
        M_TURN_STELLAR_FEEDBACK=5.0,
    )
    ion = ox.IonizedBox.new(redshift=0.0, inputs=inputs)
    assert "nion_conditional_filtered_mcg" in ion.arrays


def test_optional_field_bt(default_input_struct_lc: InputParameters):
    """Ensure that the correct BrightnessTemp fields are set based on the parameters."""
    bt = ox.BrightnessTemp.new(redshift=0.0, inputs=default_input_struct_lc)
    assert "brightness_temp" in bt.arrays
    assert "tau_21" not in bt.arrays

    inputs = default_input_struct_lc.evolve_input_structs(USE_TS_FLUCT=True)
    bt = ox.BrightnessTemp.new(redshift=0.0, inputs=inputs)
    assert "tau_21" in bt.arrays


@pytest.mark.parametrize("struct", list(ox.OutputStructZ.__subclasses__()))
def test_bad_required_array(default_input_struct, struct):
    # no struct takes this input
    inputs = default_input_struct.evolve_input_structs(SOURCE_MODEL="CHMF-SAMPLER")
    bt = ox.BrightnessTemp.new(redshift=10.0, inputs=inputs)
    kwargs = {"inputs": inputs, "redshift": 10.0}
    if struct is ox.PerturbedHaloCatalog:
        kwargs["buffer_size"] = 1
    output = struct.new(**kwargs)

    with pytest.raises((ValueError, TypeError), match="is not an input required for"):
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
