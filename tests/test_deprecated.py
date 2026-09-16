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
    compute_ionization_field,
    compute_radiation_fields,
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
    ef = compute_emissivity_fields(
        redshift=redshift_test,
        initial_conditions=ic,
        perturbed_field=pt,
        inputs=default_input_struct_lc,
        cache=cache,
    )
    return ic, pt, ef


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
        setattr(o, k, v.with_value(v.value))
    for fld in o._struct.primitive_fields:
        setattr(o, fld, 0.0)
    h5.write_output_to_hdf5(o, cache.EmissivityFields[z])

    return cache, z


def test_halobox_deprecated_warning(default_input_struct):
    """Test that the HaloBox class is deprecated."""
    with pytest.warns(deprecation.DeprecatedWarning):
        halo_box = HaloBox.new(inputs=default_input_struct, redshift=10)
    assert isinstance(halo_box, HaloBox)


@deprecation.fail_if_not_removed
def test_halobox_is_removed(default_input_struct):
    """Fails when removed_in version is reached, reminding you to delete HaloBox."""
    HaloBox.new(inputs=default_input_struct, redshift=10)


def test_extra_halobox_fields_deprecated_warning(default_input_struct_lc):
    """Test that the EXTRA_HALOBOX_FIELDS config option is deprecated."""
    with pytest.warns(deprecation.DeprecatedWarning):
        config._translate_deprecated({"EXTRA_HALOBOX_FIELDS": True})

    with config.use(EXTRA_HALOBOX_FIELDS=True):
        emissivity_fields = EmissivityFields.new(
            redshift=0.0, inputs=default_input_struct_lc
        )
        assert isinstance(emissivity_fields.halo_mass, Array)
        assert isinstance(emissivity_fields.count, Array)
        assert isinstance(emissivity_fields.halo_stars, Array)
        assert emissivity_fields.halo_stars_mini is None

        emissivity_fields = EmissivityFields.new(
            redshift=0.0,
            inputs=default_input_struct_lc.evolve_input_structs(
                USE_TS_FLUCT=True,
                RECOMB_MODEL="inhomogeneous",
                USE_MINI_HALOS=True,
                V_CB_MODEL="AVG-DEBUG",
                M_TURN_STELLAR_FEEDBACK=5.0,
            ),
        )
        assert isinstance(emissivity_fields.halo_stars_mini, Array)

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous", SOURCE_MODEL="L-INTEGRAL"
    )
    emissivity_fields = EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert isinstance(emissivity_fields.whalo_sfr, Array)


@deprecation.fail_if_not_removed
def test_extra_halobox_fields_is_removed(default_input_struct_lc):
    """Fails when removed_in version is reached, reminding you to delete EXTRA_HALOBOX_FIELDS."""
    with config.use(EXTRA_HALOBOX_FIELDS=True):
        emissivity_fields = EmissivityFields.new(
            redshift=0.0, inputs=default_input_struct_lc
        )
        assert isinstance(emissivity_fields.halo_mass, Array)
        assert isinstance(emissivity_fields.count, Array)
        assert isinstance(emissivity_fields.halo_stars, Array)
        assert emissivity_fields.halo_stars_mini is None

        emissivity_fields = EmissivityFields.new(
            redshift=0.0,
            inputs=default_input_struct_lc.evolve_input_structs(
                USE_TS_FLUCT=True,
                RECOMB_MODEL="inhomogeneous",
                USE_MINI_HALOS=True,
                V_CB_MODEL="AVG-DEBUG",
                M_TURN_STELLAR_FEEDBACK=5.0,
            ),
        )
        assert isinstance(emissivity_fields.halo_stars_mini, Array)

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous", SOURCE_MODEL="L-INTEGRAL"
    )
    emissivity_fields = EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert isinstance(emissivity_fields.whalo_sfr, Array)


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
    with pytest.warns(deprecation.DeprecatedWarning):
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
    with pytest.warns(deprecation.DeprecatedWarning):
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
    with pytest.warns(deprecation.DeprecatedWarning):
        updated_config = config.update(**deprecated_kwargs)

    for field in fields:
        if field.name in changed_fields:
            assert not getattr(updated_config, field.name)
        else:
            assert getattr(updated_config, field.name) == getattr(config, field.name)

    # Then check that the classmethod versions also work as expected
    with pytest.warns(deprecation.DeprecatedWarning):
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
    with pytest.warns(deprecation.DeprecatedWarning):
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
    with pytest.warns(deprecation.DeprecatedWarning):
        compute_radiation_fields(hboxes=[ef], redshift=ef.redshift)


@deprecation.fail_if_not_removed
def test_compute_radiation_fields_with_hboxes_is_removed(computed_emissivity_fields):
    """Fails when removed_in version is reached, reminding you to delete hboxes from compute_radiation_fields."""
    _, _, ef = computed_emissivity_fields
    compute_radiation_fields(hboxes=[ef], redshift=ef.redshift)


def test_get_output_struct_at_z_with_halobox_deprecated_warning(tmp_path: Path):
    """Test that get_output_struct_at_z with kind='HaloBox' is deprecated."""
    cache, z = _make_run_cache_with_emissivity_fields(tmp_path)
    with pytest.warns(deprecation.DeprecatedWarning):
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
    with pytest.warns(deprecation.DeprecatedWarning):
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
