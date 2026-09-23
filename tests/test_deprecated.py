"""
Tests for deprecated parameters and APIs.

This module consolidates all deprecation warning tests in one place.
Each deprecated parameter should have:
1. A test verifying the deprecation warning fires correctly.
2. A test decorated with @deprecation.fail_if_not_removed that will
   fail when the removed_in version is reached, reminding developers
   to clean up the deprecated code.

When a parameter is removed in v5, remove its tests from this module.
"""

from pathlib import Path

import attrs
import deprecation
import numpy as np
import pytest

from py21cmfast import (
    AstroOptions,
    AstroParams,
    BrightnessTemp,
    Coeval,
    EmissivityFields,
    HaloBox,
    InitialConditions,
    InputParameters,
    IonizedBox,
    PerturbedField,
    compute_emissivity_fields,
    compute_halo_grid,
    compute_initial_conditions,
    compute_ionization_field,
    compute_radiation_fields,
    compute_spin_temperature,
    config,
    perturb_field,
)
from py21cmfast.io import caching, h5
from py21cmfast.wrapper.arrays import Array


@pytest.fixture(scope="module")
def redshift_test(default_input_struct_lc):
    """The first redshift in the default input struct."""
    return default_input_struct_lc.node_redshifts[0]


@pytest.fixture(scope="module")
def pt(ic, redshift_test, default_input_struct_lc, cache):
    """A real, computed PerturbedField for testing the driver functions."""
    return perturb_field(
        redshift=redshift_test,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        cache=cache,
    )


@pytest.fixture(scope="module")
def computed_emissivity_fields(ic, pt, redshift_test, default_input_struct_lc, cache):
    """A real, computed EmissivityFields + its perturbed field, for testing the driver functions."""
    with config.use(EXTRA_EMISSIVITY_FIELDS=True):
        ef = compute_emissivity_fields(
            redshift=redshift_test,
            initial_conditions=ic,
            perturbed_field=pt,
            inputs=default_input_struct_lc,
            cache=cache,
        )
    return ic, pt, ef


@pytest.fixture(scope="module")
def computed_emissivity_fields_with_mcgs(redshift_test, default_input_struct_lc, cache):
    """A real, computed EmissivityFields with MCGs."""
    inputs = default_input_struct_lc.evolve_input_structs(
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
        USE_MCGS=True,
        V_CB_MODEL="AVG-DEBUG",
        M_TURN_STELLAR_FEEDBACK=5.0,
        SOURCE_MODEL="L-INTEGRAL",
    )
    ic = compute_initial_conditions(
        inputs=inputs,
        cache=cache,
    )
    pt = perturb_field(
        redshift=redshift_test,
        initial_conditions=ic,
        inputs=inputs,
        cache=cache,
    )
    with config.use(EXTRA_EMISSIVITY_FIELDS=True):
        ef = compute_emissivity_fields(
            redshift=redshift_test,
            initial_conditions=ic,
            perturbed_field=pt,
            inputs=inputs,
            cache=cache,
        )
    return ic, pt, ef


@pytest.fixture(scope="module")
def computed_ionization_field(
    computed_emissivity_fields_with_mcgs, default_input_struct_lc, cache
):
    inputs = default_input_struct_lc.evolve_input_structs(
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
        USE_MCGS=True,
        V_CB_MODEL="AVG-DEBUG",
        M_TURN_STELLAR_FEEDBACK=5.0,
        SOURCE_MODEL="L-INTEGRAL",
    )
    ic, pt, ef = computed_emissivity_fields_with_mcgs

    rf = compute_radiation_fields(
        hboxes=[ef],
        redshift=ef.redshift,
        cache=cache,
    )
    st = compute_spin_temperature(
        initial_conditions=ic,
        perturbed_field=pt,
        radiation_fields=rf,
        inputs=inputs,
        cache=cache,
    )

    return compute_ionization_field(
        initial_conditions=ic,
        perturbed_field=pt,
        emissivity_fields=ef,
        spin_temp=st,
        inputs=inputs,
        cache=cache,
    )


def _make_run_cache_with_emissivity_fields(
    tmp_path: Path,
) -> tuple[caching.RunCache, float]:
    """Build a RunCache with a single, real (mocked) EmissivityFields file on disk."""
    inputs = InputParameters.from_template(
        "latest", random_seed=12345, node_redshifts=np.arange(12, 38, 3.0)[::-1]
    ).evolve_input_structs(HII_DIM=10, DIM=20, BOX_LEN=75.0, ZPRIME_STEP_FACTOR=1.3)
    cache = caching.RunCache.from_inputs(inputs, caching.OutputCache(tmp_path))

    z = inputs.node_redshifts[0]
    o = EmissivityFields.new(redshift=z, inputs=inputs)
    o._init_arrays()
    # Go through each array and set it to be "computed" so we can trick the writer
    # into writing it out to file, without needing to run any C computation.
    for k, v in o.arrays.items():
        setattr(o, k, v.with_value(v._value))
    for fld in o._struct.primitive_fields:
        setattr(o, fld, 0.0)
    h5.write_output_to_hdf5(o, cache.EmissivityFields[z])

    return cache, z


def test_halobox_deprecated_warning(default_input_struct):
    """Test that the HaloBox class is deprecated."""
    with pytest.warns(deprecation.DeprecatedWarning, match="HaloBox is deprecated"):
        halo_box = HaloBox.new(inputs=default_input_struct, redshift=10)
    assert isinstance(halo_box, HaloBox)


@deprecation.fail_if_not_removed
def test_halobox_is_removed(default_input_struct):
    """Fails when removed_in version is reached, reminding you to delete HaloBox."""
    HaloBox.new(inputs=default_input_struct, redshift=10)


def test_extra_halobox_fields_deprecated_warning(default_input_struct_lc):
    """Test that the EXTRA_HALOBOX_FIELDS config option is deprecated."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="EXTRA_HALOBOX_FIELDS is deprecated"
    ):
        config._translate_deprecated({"EXTRA_HALOBOX_FIELDS": True})

    with config.use(EXTRA_HALOBOX_FIELDS=True):
        emissivity_fields = EmissivityFields.new(
            redshift=0.0, inputs=default_input_struct_lc
        )
        assert isinstance(emissivity_fields.halo_mass_density, Array)
        assert isinstance(emissivity_fields.halo_number, Array)
        assert isinstance(emissivity_fields.stellar_mass_density_acg, Array)
        assert emissivity_fields.stellar_mass_density_mcg is None

        emissivity_fields = EmissivityFields.new(
            redshift=0.0,
            inputs=default_input_struct_lc.evolve_input_structs(
                USE_TS_FLUCT=True,
                RECOMB_MODEL="inhomogeneous",
                USE_MCGS=True,
                V_CB_MODEL="AVG-DEBUG",
                M_TURN_STELLAR_FEEDBACK=5.0,
            ),
        )
        assert isinstance(emissivity_fields.stellar_mass_density_mcg, Array)

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous", SOURCE_MODEL="L-INTEGRAL"
    )
    emissivity_fields = EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert isinstance(emissivity_fields.fesc_weighted_sfrd, Array)


@deprecation.fail_if_not_removed
def test_extra_halobox_fields_is_removed(default_input_struct_lc):
    """Fails when removed_in version is reached, reminding you to delete EXTRA_HALOBOX_FIELDS."""
    with config.use(EXTRA_HALOBOX_FIELDS=True):
        emissivity_fields = EmissivityFields.new(
            redshift=0.0, inputs=default_input_struct_lc
        )
        assert isinstance(emissivity_fields.halo_mass_density, Array)
        assert isinstance(emissivity_fields.halo_number, Array)
        assert isinstance(emissivity_fields.stellar_mass_density_acg, Array)
        assert emissivity_fields.stellar_mass_density_mcg is None

        emissivity_fields = EmissivityFields.new(
            redshift=0.0,
            inputs=default_input_struct_lc.evolve_input_structs(
                USE_TS_FLUCT=True,
                RECOMB_MODEL="inhomogeneous",
                USE_MCGS=True,
                V_CB_MODEL="AVG-DEBUG",
                M_TURN_STELLAR_FEEDBACK=5.0,
            ),
        )
        assert isinstance(emissivity_fields.stellar_mass_density_mcg, Array)

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous", SOURCE_MODEL="L-INTEGRAL"
    )
    emissivity_fields = EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert isinstance(emissivity_fields.fesc_weighted_sfrd, Array)


def test_coeval_with_halobox_deprecated_warning(default_input_struct_lc):
    """Test that accessing Coeval.halobox is deprecated."""
    coeval = Coeval(
        initial_conditions=InitialConditions.new(inputs=default_input_struct_lc),
        perturbed_field=PerturbedField.new(
            redshift=0.0, inputs=default_input_struct_lc
        ),
        ionized_box=IonizedBox.new(redshift=0.0, inputs=default_input_struct_lc),
        brightness_temperature=BrightnessTemp.new(
            redshift=0.0, inputs=default_input_struct_lc
        ),
        emissivity_fields=EmissivityFields.new(
            redshift=0.0, inputs=default_input_struct_lc
        ),
    )
    with pytest.warns(deprecation.DeprecatedWarning, match="halobox is deprecated"):
        assert coeval.halobox is coeval.emissivity_fields


@deprecation.fail_if_not_removed
def test_coeval_with_halobox_is_removed(default_input_struct_lc):
    """Fails when removed_in version is reached, reminding you to delete halobox from Coeval."""
    coeval = Coeval(
        initial_conditions=InitialConditions.new(inputs=default_input_struct_lc),
        perturbed_field=PerturbedField.new(
            redshift=0.0, inputs=default_input_struct_lc
        ),
        ionized_box=IonizedBox.new(redshift=0.0, inputs=default_input_struct_lc),
        brightness_temperature=BrightnessTemp.new(
            redshift=0.0, inputs=default_input_struct_lc
        ),
        emissivity_fields=EmissivityFields.new(
            redshift=0.0, inputs=default_input_struct_lc
        ),
    )
    assert coeval.halobox is coeval.emissivity_fields


def test_run_cache_with_halobox_deprecated_warning(tmp_path: Path):
    """Test that running the cache with halobox is deprecated."""
    inputs = InputParameters.from_template("latest-dhalos", random_seed=12345)
    cache = caching.RunCache.from_inputs(inputs, caching.OutputCache(tmp_path))
    with pytest.warns(deprecation.DeprecatedWarning, match="HaloBox is deprecated"):
        assert isinstance(cache.HaloBox, dict)


@deprecation.fail_if_not_removed
def test_run_cache_with_halobox_is_removed(tmp_path: Path):
    """Fails when removed_in version is reached, reminding you to delete halobox from RunCache."""
    inputs = InputParameters.from_template("latest-dhalos", random_seed=12345)
    cache = caching.RunCache.from_inputs(inputs, caching.OutputCache(tmp_path))
    assert isinstance(cache.HaloBox, dict)


@pytest.mark.parametrize("config_method", ["on", "off", "noloop", "last_step_only"])
def test_update_cache_with_halobox_deprecated_warning(config_method):
    """Test that updating Cacheconfig with halobox is deprecated."""
    config = getattr(caching.CacheConfig, config_method)()
    fields = attrs.fields(caching.CacheConfig)

    # Annoying fields that we don't want to cache
    deprecated_kwargs = {"halobox": False}
    changed_fields = ["emissivity_fields"]

    # First check that the update method works as expected
    with pytest.warns(deprecation.DeprecatedWarning, match="halobox is deprecated"):
        updated_config = config.update(**deprecated_kwargs)

    for field in fields:
        if field.name in changed_fields:
            assert not getattr(updated_config, field.name)
        else:
            assert getattr(updated_config, field.name) == getattr(config, field.name)

    # Then check that the classmethod versions also work as expected
    with pytest.warns(deprecation.DeprecatedWarning, match="halobox is deprecated"):
        updated_config = getattr(caching.CacheConfig, config_method)(
            **deprecated_kwargs
        )
    for field in fields:
        if field.name in changed_fields:
            assert not getattr(updated_config, field.name)
        else:
            assert getattr(updated_config, field.name) == getattr(config, field.name)


@deprecation.fail_if_not_removed
@pytest.mark.parametrize("config_method", ["on", "off", "noloop", "last_step_only"])
def test_update_cache_with_halobox_is_removed(config_method):
    """Fails when removed_in version is reached, reminding you to delete halobox from CacheConfig."""
    config = getattr(caching.CacheConfig, config_method)()
    fields = attrs.fields(caching.CacheConfig)

    # Annoying fields that we don't want to cache
    deprecated_kwargs = {"halobox": False}
    changed_fields = ["emissivity_fields"]

    # First check that the update method works as expected
    updated_config = config.update(**deprecated_kwargs)
    for field in fields:
        if field.name in changed_fields:
            assert not getattr(updated_config, field.name)
        else:
            assert getattr(updated_config, field.name) == getattr(config, field.name)

    # Then check that the classmethod versions also work as expected
    updated_config = getattr(caching.CacheConfig, config_method)(**deprecated_kwargs)
    for field in fields:
        if field.name in changed_fields:
            assert not getattr(updated_config, field.name)
        else:
            assert getattr(updated_config, field.name) == getattr(config, field.name)


def test_compute_ionization_field_with_halobox_deprecated_warning(
    computed_emissivity_fields,
):
    """Test that compute_ionization_field's halobox kwarg is deprecated."""
    ic, pt, ef = computed_emissivity_fields
    with pytest.warns(deprecation.DeprecatedWarning, match="halobox is deprecated"):
        compute_ionization_field(initial_conditions=ic, perturbed_field=pt, halobox=ef)


@deprecation.fail_if_not_removed
def test_compute_ionization_field_with_halobox_is_removed(computed_emissivity_fields):
    """Fails when removed_in version is reached, reminding you to delete halobox from compute_ionization_field."""
    ic, pt, ef = computed_emissivity_fields
    compute_ionization_field(initial_conditions=ic, perturbed_field=pt, halobox=ef)


def test_compute_radiation_fields_with_hboxes_deprecated_warning(
    computed_emissivity_fields,
):
    """Test that compute_radiation_fields's hboxes kwarg is deprecated."""
    _, _, ef = computed_emissivity_fields
    with pytest.warns(deprecation.DeprecatedWarning, match="hboxes is deprecated"):
        compute_radiation_fields(hboxes=[ef], redshift=ef.redshift)


@deprecation.fail_if_not_removed
def test_compute_radiation_fields_with_hboxes_is_removed(computed_emissivity_fields):
    """Fails when removed_in version is reached, reminding you to delete hboxes from compute_radiation_fields."""
    _, _, ef = computed_emissivity_fields
    compute_radiation_fields(hboxes=[ef], redshift=ef.redshift)


def test_get_output_struct_at_z_with_halobox_deprecated_warning(tmp_path: Path):
    """Test that get_output_struct_at_z with kind='HaloBox' is deprecated."""
    cache, z = _make_run_cache_with_emissivity_fields(tmp_path)
    with pytest.warns(deprecation.DeprecatedWarning, match="HaloBox is deprecated"):
        output = cache.get_output_struct_at_z(kind="HaloBox", z=z)
    assert isinstance(output, EmissivityFields)


@deprecation.fail_if_not_removed
def test_get_output_struct_at_z_with_halobox_is_removed(tmp_path: Path):
    """Fails when removed_in version is reached, reminding you to delete the 'HaloBox' kind alias from get_output_struct_at_z."""
    cache, z = _make_run_cache_with_emissivity_fields(tmp_path)
    output = cache.get_output_struct_at_z(kind="HaloBox", z=z)
    assert isinstance(output, EmissivityFields)


def test_compute_halo_grid_deprecated_warning(
    ic, pt, redshift_test, default_input_struct_lc, cache
):
    """Test that compute_halo_grid is deprecated."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="compute_halo_grid is deprecated"
    ):
        ef = compute_halo_grid(
            redshift=redshift_test,
            initial_conditions=ic,
            perturbed_field=pt,
            inputs=default_input_struct_lc,
            cache=cache,
        )
    assert isinstance(ef, EmissivityFields)


@deprecation.fail_if_not_removed
def test_compute_halo_grid_is_removed(
    ic, pt, redshift_test, default_input_struct_lc, cache
):
    """Fails when removed_in version is reached, reminding you to delete compute_halo_grid."""
    ef = compute_halo_grid(
        redshift=redshift_test,
        initial_conditions=ic,
        perturbed_field=pt,
        inputs=default_input_struct_lc,
        cache=cache,
    )
    assert isinstance(ef, EmissivityFields)


def test_use_mini_halos_deprecated_warning():
    """Test that using USE_MINI_HALOS=True shows deprecation warning."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="USE_MINI_HALOS is deprecated"
    ):
        astro_params = AstroOptions(
            USE_MINI_HALOS=True,
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
        )
    assert astro_params.USE_MCGS == astro_params.USE_MINI_HALOS


@deprecation.fail_if_not_removed
def test_use_mini_halos_is_removed():
    """Fails when removed_in version is reached, reminding you to delete USE_MINI_HALOS."""
    AstroOptions(
        USE_MINI_HALOS=True,
        USE_TS_FLUCT=True,
        RECOMB_MODEL="inhomogeneous",
    )


def test_integration_method_atomic_deprecated_warning():
    """Test that using INTEGRATION_METHOD_ATOMIC shows deprecation warning."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="INTEGRATION_METHOD_ATOMIC is deprecated"
    ):
        astro_params = AstroOptions(INTEGRATION_METHOD_ATOMIC="GAUSS-LEGENDRE")
    assert (
        astro_params.INTEGRATION_METHOD_ACGS == astro_params._INTEGRATION_METHOD_ATOMIC
    )


@deprecation.fail_if_not_removed
def test_integration_method_atomic_is_removed():
    """Fails when removed_in version is reached, reminding you to delete INTEGRATION_METHOD_ATOMIC."""
    AstroOptions(INTEGRATION_METHOD_ATOMIC="GAUSS-LEGENDRE")


def test_integration_method_mini_deprecated_warning():
    """Test that using INTEGRATION_METHOD_MINI shows deprecation warning."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="INTEGRATION_METHOD_MINI is deprecated"
    ):
        astro_params = AstroOptions(INTEGRATION_METHOD_MINI="GAUSS-LEGENDRE")
    assert astro_params.INTEGRATION_METHOD_MCGS == astro_params._INTEGRATION_METHOD_MINI


@deprecation.fail_if_not_removed
def test_integration_method_mini_is_removed():
    """Fails when removed_in version is reached, reminding you to delete INTEGRATION_METHOD_MINI."""
    AstroOptions(INTEGRATION_METHOD_MINI="GAUSS-LEGENDRE")


def test_f_star10_deprecated_warning():
    """Test that using F_STAR10 shows deprecation warning."""
    with pytest.warns(deprecation.DeprecatedWarning, match="F_STAR10 is deprecated"):
        astro_params = AstroParams(F_STAR10=-1.5)
    assert astro_params.F_STAR10_ACG == astro_params.F_STAR10


@deprecation.fail_if_not_removed
def test_f_star10_is_removed():
    """Fails when removed_in version is reached, reminding you to delete F_STAR10."""
    AstroParams(F_STAR10=-1.5)


def test_f_star7_mini_deprecated_warning():
    """Test that using F_STAR7_MINI shows deprecation warning."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="F_STAR7_MINI is deprecated"
    ):
        astro_params = AstroParams(F_STAR7_MINI=-3.5)
    assert astro_params.F_STAR7_MCG == astro_params.F_STAR7_MINI


@deprecation.fail_if_not_removed
def test_f_star7_mini_is_removed():
    """Fails when removed_in version is reached, reminding you to delete F_STAR7_MINI."""
    AstroParams(F_STAR7_MINI=-3.5)


def test_f_esc10_deprecated_warning():
    """Test that using F_ESC10 shows deprecation warning."""
    with pytest.warns(deprecation.DeprecatedWarning, match="F_ESC10 is deprecated"):
        astro_params = AstroParams(F_ESC10=-1.5)
    assert astro_params.F_ESC10_ACG == astro_params.F_ESC10


@deprecation.fail_if_not_removed
def test_f_esc10_is_removed():
    """Fails when removed_in version is reached, reminding you to delete F_ESC10."""
    AstroParams(F_ESC10=-1.5)


def test_f_esc7_mini_deprecated_warning():
    """Test that using F_ESC7_MINI shows deprecation warning."""
    with pytest.warns(deprecation.DeprecatedWarning, match="F_ESC7_MINI is deprecated"):
        astro_params = AstroParams(F_ESC7_MINI=-3.5)
    assert astro_params.F_ESC7_MCG == astro_params.F_ESC7_MINI


@deprecation.fail_if_not_removed
def test_f_esc7_mini_is_removed():
    """Fails when removed_in version is reached, reminding you to delete F_ESC7_MINI."""
    AstroParams(F_ESC7_MINI=-3.5)


def test_alpha_star_deprecated_warning():
    """Test that using ALPHA_STAR shows deprecation warning."""
    with pytest.warns(deprecation.DeprecatedWarning, match="ALPHA_STAR is deprecated"):
        astro_params = AstroParams(ALPHA_STAR=0.5)
    assert astro_params.ALPHA_STAR_ACG == astro_params.ALPHA_STAR


@deprecation.fail_if_not_removed
def test_alpha_star_is_removed():
    """Fails when removed_in version is reached, reminding you to delete ALPHA_STAR."""
    AstroParams(ALPHA_STAR=0.5)


def test_alpha_star_mini_deprecated_warning():
    """Test that using ALPHA_STAR_MINI shows deprecation warning."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="ALPHA_STAR_MINI is deprecated"
    ):
        astro_params = AstroParams(ALPHA_STAR_MINI=0.5)
    assert astro_params.ALPHA_STAR_MCG == astro_params.ALPHA_STAR_MINI


@deprecation.fail_if_not_removed
def test_alpha_star_mini_is_removed():
    """Fails when removed_in version is reached, reminding you to delete ALPHA_STAR_MINI."""
    AstroParams(ALPHA_STAR_MINI=0.5)


def test_l_x_deprecated_warning():
    """Test that using L_X shows deprecation warning."""
    with pytest.warns(deprecation.DeprecatedWarning, match="L_X is deprecated"):
        astro_params = AstroParams(L_X=40.0)
    assert astro_params.LX_OVER_SFR_ACG == astro_params.L_X


@deprecation.fail_if_not_removed
def test_l_x_is_removed():
    """Fails when removed_in version is reached, reminding you to delete L_X."""
    AstroParams(L_X=40.0)


def test_l_x_mini_deprecated_warning():
    """Test that using L_X_MINI shows deprecation warning."""
    with pytest.warns(deprecation.DeprecatedWarning, match="L_X_MINI is deprecated"):
        astro_params = AstroParams(L_X_MINI=40.5)
    assert astro_params.LX_OVER_SFR_MCG == astro_params.L_X_MINI


@deprecation.fail_if_not_removed
def test_l_x_mini_is_removed():
    """Fails when removed_in version is reached, reminding you to delete L_X_MINI."""
    AstroParams(L_X_MINI=40.5)


def test_count_deprecated_warning(computed_emissivity_fields):
    """Test that using count shows deprecation warning."""
    _, _, ef = computed_emissivity_fields
    with pytest.warns(deprecation.DeprecatedWarning, match="count is deprecated"):
        assert isinstance(ef.count, Array)
    with pytest.warns(deprecation.DeprecatedWarning, match="count is deprecated"):
        assert np.all(ef.count == np.asarray(ef.halo_number))


@deprecation.fail_if_not_removed
def test_count_is_removed(computed_emissivity_fields):
    """Fails when removed_in version is reached, reminding you to delete count."""
    _, _, ef = computed_emissivity_fields
    assert isinstance(ef.count, Array)
    assert np.all(ef.count == np.asarray(ef.halo_number))


def test_halo_mass_deprecated_warning(computed_emissivity_fields):
    """Test that using halo_mass shows deprecation warning."""
    _, _, ef = computed_emissivity_fields
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_mass is deprecated"):
        assert isinstance(ef.halo_mass, Array)
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_mass is deprecated"):
        assert np.all(ef.halo_mass == np.asarray(ef.halo_mass_density))


@deprecation.fail_if_not_removed
def test_halo_mass_is_removed(computed_emissivity_fields):
    """Fails when removed_in version is reached, reminding you to delete halo_mass."""
    _, _, ef = computed_emissivity_fields
    assert isinstance(ef.halo_mass, Array)
    assert np.all(ef.halo_mass == np.asarray(ef.halo_mass_density))


def test_halo_stars_deprecated_warning(computed_emissivity_fields):
    """Test that using halo_stars shows deprecation warning."""
    _, _, ef = computed_emissivity_fields
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_stars is deprecated"):
        assert isinstance(ef.halo_stars, Array)
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_stars is deprecated"):
        assert np.all(ef.halo_stars == np.asarray(ef.stellar_mass_density_acg))


@deprecation.fail_if_not_removed
def test_halo_stars_is_removed(computed_emissivity_fields):
    """Fails when removed_in version is reached, reminding you to delete halo_stars."""
    _, _, ef = computed_emissivity_fields
    assert isinstance(ef.halo_stars, Array)
    assert np.all(ef.halo_stars == np.asarray(ef.stellar_mass_density_acg))


def test_halo_stars_mini_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using halo_stars_mini shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(
        deprecation.DeprecatedWarning, match="halo_stars_mini is deprecated"
    ):
        assert isinstance(ef.halo_stars_mini, Array)
    with pytest.warns(
        deprecation.DeprecatedWarning, match="halo_stars_mini is deprecated"
    ):
        assert np.all(ef.halo_stars_mini == np.asarray(ef.stellar_mass_density_mcg))


@deprecation.fail_if_not_removed
def test_halo_stars_mini_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete halo_stars_mini."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert isinstance(ef.halo_stars_mini, Array)
    assert np.all(ef.halo_stars_mini == np.asarray(ef.stellar_mass_density_mcg))


def test_halo_sfr_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using halo_sfr shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_sfr is deprecated"):
        assert isinstance(ef.halo_sfr, Array)
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_sfr is deprecated"):
        assert np.all(ef.halo_sfr == np.asarray(ef.sfrd_acg))


@deprecation.fail_if_not_removed
def test_halo_sfr_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete halo_sfr."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert isinstance(ef.halo_sfr, Array)
    assert np.all(ef.halo_sfr == np.asarray(ef.sfrd_acg))


def test_halo_sfr_mini_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using halo_sfr_mini shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(
        deprecation.DeprecatedWarning, match="halo_sfr_mini is deprecated"
    ):
        assert isinstance(ef.halo_sfr_mini, Array)
    with pytest.warns(
        deprecation.DeprecatedWarning, match="halo_sfr_mini is deprecated"
    ):
        assert np.all(ef.halo_sfr_mini == np.asarray(ef.sfrd_mcg))


@deprecation.fail_if_not_removed
def test_halo_sfr_mini_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete halo_sfr_mini."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert isinstance(ef.halo_sfr_mini, Array)
    assert np.all(ef.halo_sfr_mini == np.asarray(ef.sfrd_mcg))


def test_halo_xray_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using halo_xray shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_xray is deprecated"):
        assert isinstance(ef.halo_xray, Array)
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_xray is deprecated"):
        assert np.all(ef.halo_xray == np.asarray(ef.xray_emissivity))


@deprecation.fail_if_not_removed
def test_halo_xray_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete halo_xray."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert isinstance(ef.halo_xray, Array)
    assert np.all(ef.halo_xray == np.asarray(ef.xray_emissivity))


def test_whalo_sfr_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using whalo_sfr shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(deprecation.DeprecatedWarning, match="whalo_sfr is deprecated"):
        assert isinstance(ef.whalo_sfr, Array)
    with pytest.warns(deprecation.DeprecatedWarning, match="whalo_sfr is deprecated"):
        assert np.all(ef.whalo_sfr == np.asarray(ef.fesc_weighted_sfrd))


@deprecation.fail_if_not_removed
def test_whalo_sfr_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete whalo_sfr."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert isinstance(ef.whalo_sfr, Array)
    assert np.all(ef.whalo_sfr == np.asarray(ef.fesc_weighted_sfrd))


def test_log10_mcrit_acg_ave_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using log10_mcrit_acg_ave shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(
        deprecation.DeprecatedWarning, match="log10_Mcrit_ACG_ave is deprecated"
    ):
        assert np.all(ef.log10_Mcrit_ACG_ave == ef.log10_mturn_acg_ave)


@deprecation.fail_if_not_removed
def test_log10_mcrit_acg_ave_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete log10_mcrit_acg_ave."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert np.all(ef.log10_Mcrit_ACG_ave == ef.log10_mturn_acg_ave)


def test_log10_mcrit_mcg_ave_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using log10_mcrit_mcg_ave shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(
        deprecation.DeprecatedWarning, match="log10_Mcrit_MCG_ave is deprecated"
    ):
        assert np.all(ef.log10_Mcrit_MCG_ave == ef.log10_mturn_mcg_ave)


@deprecation.fail_if_not_removed
def test_log10_mcrit_mcg_ave_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete log10_mcrit_mcg_ave."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert np.all(ef.log10_Mcrit_MCG_ave == ef.log10_mturn_mcg_ave)


def test_log10_mturnover_ave_deprecated_warning(computed_ionization_field):
    """Test that using log10_Mturnover_ave shows deprecation warning."""
    ionized_box = computed_ionization_field
    with pytest.warns(
        deprecation.DeprecatedWarning, match="log10_Mturnover_ave is deprecated"
    ):
        assert np.all(
            ionized_box.log10_Mturnover_ave == ionized_box.log10_mturn_ave_acg
        )


@deprecation.fail_if_not_removed
def test_log10_mturnover_ave_is_removed(computed_ionization_field):
    """Fails when removed_in version is reached, reminding you to delete log10_Mturnover_ave."""
    ionized_box = computed_ionization_field
    assert np.all(ionized_box.log10_Mturnover_ave == ionized_box.log10_mturn_ave_acg)


def test_log10_mturnover_mini_ave_deprecated_warning(computed_ionization_field):
    """Test that using log10_Mturnover_MINI_ave shows deprecation warning."""
    ionized_box = computed_ionization_field
    with pytest.warns(
        deprecation.DeprecatedWarning, match="log10_Mturnover_MINI_ave is deprecated"
    ):
        assert np.all(
            ionized_box.log10_Mturnover_MINI_ave == ionized_box.log10_mturn_ave_mcg
        )


@deprecation.fail_if_not_removed
def test_log10_mturnover_mini_ave_is_removed(computed_ionization_field):
    """Fails when removed_in version is reached, reminding you to delete log10_Mturnover_MINI_ave."""
    ionized_box = computed_ionization_field
    assert np.all(
        ionized_box.log10_Mturnover_MINI_ave == ionized_box.log10_mturn_ave_mcg
    )


def test_bad_deprecated_inputs():
    """Test that bad deprecated inputs raise ValueError."""
    with pytest.raises(ValueError, match="USE_MCGS is set to"):
        AstroOptions(
            USE_MINI_HALOS=False,
            USE_MCGS=True,
            USE_TS_FLUCT=True,
            RECOMB_MODEL="inhomogeneous",
        )

    with pytest.raises(ValueError, match="INTEGRATION_METHOD_ACGS is set to"):
        AstroOptions(
            INTEGRATION_METHOD_ATOMIC="GAUSS-LEGENDRE",
            INTEGRATION_METHOD_ACGS="GSL-QAG",
        )

    with pytest.raises(ValueError, match="INTEGRATION_METHOD_MCGS is set to"):
        AstroOptions(
            INTEGRATION_METHOD_MINI="GAUSS-LEGENDRE",
            INTEGRATION_METHOD_MCGS="GSL-QAG",
        )

    with pytest.raises(ValueError, match="F_STAR10_ACG is set to"):
        AstroParams(F_STAR10=-1.5, F_STAR10_ACG=-1.0)

    with pytest.raises(ValueError, match="F_STAR7_MCG is set to"):
        AstroParams(F_STAR7_MINI=-3.5, F_STAR7_MCG=-3.0)

    with pytest.raises(ValueError, match="F_ESC10_ACG is set to"):
        AstroParams(F_ESC10=-1.5, F_ESC10_ACG=-1.0)

    with pytest.raises(ValueError, match="F_ESC7_MCG is set to"):
        AstroParams(F_ESC7_MINI=-3.5, F_ESC7_MCG=-3.0)

    with pytest.raises(ValueError, match="ALPHA_STAR_ACG is set to"):
        AstroParams(ALPHA_STAR=0.5, ALPHA_STAR_ACG=0.6)

    with pytest.raises(ValueError, match="ALPHA_STAR_MCG is set to"):
        AstroParams(ALPHA_STAR_MINI=0.5, ALPHA_STAR_MCG=0.6)

    with pytest.raises(ValueError, match="LX_OVER_SFR_ACG is set to"):
        AstroParams(L_X=40.5, LX_OVER_SFR_ACG=41.0)

    with pytest.raises(ValueError, match="LX_OVER_SFR_MCG is set to"):
        AstroParams(L_X_MINI=40.5, LX_OVER_SFR_MCG=41.0)

    with (
        pytest.warns(
            deprecation.DeprecatedWarning,
            match="INTEGRATION_METHOD_ATOMIC is deprecated",
        ),
        pytest.raises(ValueError, match="INTEGRATION_METHOD_ACGS must be one of"),
    ):
        AstroOptions(INTEGRATION_METHOD_ATOMIC="INVALID")

    with (
        pytest.warns(
            deprecation.DeprecatedWarning, match="INTEGRATION_METHOD_MINI is deprecated"
        ),
        pytest.raises(ValueError, match="INTEGRATION_METHOD_MCGS must be one of"),
    ):
        AstroOptions(INTEGRATION_METHOD_MINI="INVALID")


def test_array_value_deprecated_warning(ic: InitialConditions):
    """Test that using Array.value shows a deprecation warning."""
    expected = ic.get("hires_density")

    with pytest.warns(deprecation.DeprecatedWarning, match="Array.value is deprecated"):
        assert np.allclose(ic.hires_density.value, expected)

    # It must keep working for a purged array, where it used to return None -- that
    # silent None is what made tests reading `.value` off the shared session-scoped
    # `ic` fixture fail depending on whether a driver had purged it first.
    ic.purge()
    assert ic.hires_density._value is None

    with pytest.warns(deprecation.DeprecatedWarning, match="Array.value is deprecated"):
        assert np.allclose(ic.hires_density.value, expected)

    ic.load_all()


@deprecation.fail_if_not_removed
def test_array_value_is_removed(ic: InitialConditions):
    """Fails when removed_in version is reached, reminding you to delete Array.value."""
    assert np.allclose(ic.hires_density.value, ic.get("hires_density"))
