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

from contextlib import ExitStack
from pathlib import Path

import attrs
import deprecation
import numpy as np
import pytest
from astropy import units as un

import py21cmfast as p21c
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
from py21cmfast import lightconers as lcn
from py21cmfast.io import caching, h5
from py21cmfast.wrapper import cfuncs as cf
from py21cmfast.wrapper.inputs import MatterOptions

_MAJOR_VERSION = int(p21c.__version__.split(".")[0])


def test_fixed_vavg_deprecated_warning():
    """Test that using FIXED_VAVG=True shows deprecation warning."""
    fixed_vavg = 1.0
    with pytest.warns(deprecation.DeprecatedWarning, match="FIXED_VAVG is deprecated"):
        astro_params = AstroParams(FIXED_VAVG=fixed_vavg)
    assert fixed_vavg == astro_params.FIXED_VAVG
    assert fixed_vavg == astro_params.V_CB_AVG_DEBUG


@pytest.mark.skipif(
    _MAJOR_VERSION < 5, reason="FIXED_VAVG removal scheduled for v5.0.0"
)
def test_fixed_vavg_is_removed():
    """Confirms FIXED_VAVG was removed in v5.0.0; runs only at v5+."""
    with pytest.raises(TypeError, match="FIXED_VAVG"):
        AstroParams(FIXED_VAVG=1.0)


def test_mturn_deprecated_warning():
    """Test that using a non-None value for M_TURN shows deprecation warning."""
    mturn = 8.7
    with pytest.warns(deprecation.DeprecatedWarning, match="M_TURN is deprecated"):
        astro_params = AstroParams(M_TURN=mturn)
    assert mturn == astro_params.M_TURN
    assert mturn == astro_params.M_TURN_STELLAR_FEEDBACK


@pytest.mark.skipif(_MAJOR_VERSION < 5, reason="M_TURN removal scheduled for v5.0.0")
def test_mturn_is_removed():
    """Confirms M_TURN was removed in v5.0.0; runs only at v5+."""
    with pytest.raises(TypeError, match="M_TURN"):
        AstroParams(M_TURN=8.7)


def test_inhomo_reco_deprecated_warning():
    """Test that using INHOMO_RECO=True shows deprecation warning."""
    with pytest.warns(deprecation.DeprecatedWarning, match="INHOMO_RECO is deprecated"):
        opts = AstroOptions(INHOMO_RECO=True)
    assert opts.RECOMB_MODEL == "inhomogeneous"
    assert opts.INHOMO_RECO is True


@pytest.mark.skipif(
    _MAJOR_VERSION < 5, reason="INHOMO_RECO removal scheduled for v5.0.0"
)
def test_inhomo_reco_is_removed():
    """Confirms INHOMO_RECO was removed in v5.0.0; runs only at v5+."""
    with pytest.raises(TypeError, match="INHOMO_RECO"):
        AstroOptions(INHOMO_RECO=True)


def test_use_relative_velocities_deprecated_warning():
    """Test that using USE_RELATIVE_VELOCITIES=True shows deprecation warning."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="USE_RELATIVE_VELOCITIES is deprecated"
    ):
        opts = MatterOptions(USE_RELATIVE_VELOCITIES=True)
    assert opts.V_CB_MODEL == "FLUCTS"
    assert opts.USE_RELATIVE_VELOCITIES is True


@pytest.mark.skipif(
    _MAJOR_VERSION < 5, reason="USE_RELATIVE_VELOCITIES removal scheduled for v5.0.0"
)
def test_use_relative_velocities_is_removed():
    """Confirms USE_RELATIVE_VELOCITIES was removed in v5.0.0; runs only at v5+."""
    with pytest.raises(TypeError, match="USE_RELATIVE_VELOCITIES"):
        MatterOptions(USE_RELATIVE_VELOCITIES=True)


@pytest.mark.parametrize("kwargs", [{}, {"INHOMO_RECO": False}])
def test_inhomo_reco_not_provided_sets_none(kwargs):
    """Test that INHOMO_RECO=False (or not provided) sets RECOMB_MODEL='none'."""
    if kwargs:
        with pytest.warns(
            deprecation.DeprecatedWarning, match="INHOMO_RECO is deprecated"
        ):
            opts = AstroOptions(**kwargs)
    else:
        opts = AstroOptions(**kwargs)
    assert opts.RECOMB_MODEL == "none"
    assert opts.INHOMO_RECO is False


@pytest.mark.parametrize("kwargs", [{}, {"USE_RELATIVE_VELOCITIES": False}])
def test_use_relative_velocities_not_provided_sets_none(kwargs):
    """Test that USE_RELATIVE_VELOCITIES=False (or not provided) sets V_CB_MODEL='NONE'."""
    if kwargs:
        with pytest.warns(
            deprecation.DeprecatedWarning, match="USE_RELATIVE_VELOCITIES is deprecated"
        ):
            opts = MatterOptions(**kwargs)
    else:
        opts = MatterOptions(**kwargs)
    assert opts.V_CB_MODEL == "NONE"
    assert opts.USE_RELATIVE_VELOCITIES is False


@pytest.mark.parametrize("recomb_model", ["none", "homogeneous", "inhomogeneous"])
def test_recomb_model_basic(recomb_model):
    """Test basic RECOMB_MODEL usage without INHOMO_RECO."""
    opts_none = AstroOptions(RECOMB_MODEL=recomb_model)
    assert recomb_model == opts_none.RECOMB_MODEL
    expected = recomb_model != "none"
    assert opts_none.INHOMO_RECO is expected


@pytest.mark.parametrize("recomb_model", ["none", "homogeneous", "inhomogeneous"])
def test_recomb_model_conflict(recomb_model):
    """Test error when INHOMO_RECO=False conflicts with RECOMB_MODEL!='none'."""
    inhomo_reco_wrong = recomb_model == "none"
    with pytest.raises(
        ValueError,
        match=f"RECOMB_MODEL is set to '{recomb_model}' but INHOMO_RECO is {inhomo_reco_wrong}",
    ):
        AstroOptions(INHOMO_RECO=inhomo_reco_wrong, RECOMB_MODEL=recomb_model)


@pytest.mark.parametrize("v_cb_model", ["NONE", "AVG-AUTO", "FLUCTS", "AVG-DEBUG"])
def test_v_cb_model_basic(v_cb_model):
    """Test basic V_CB_MODEL usage without USE_RELATIVE_VELOCITIES."""
    opts_none = MatterOptions(V_CB_MODEL=v_cb_model)
    assert v_cb_model == opts_none.V_CB_MODEL
    expected = v_cb_model == "FLUCTS"
    assert opts_none.USE_RELATIVE_VELOCITIES is expected


@pytest.mark.parametrize("v_cb_model", ["NONE", "AVG-AUTO", "FLUCTS", "AVG-DEBUG"])
def test_v_cb_model_conflict(v_cb_model):
    """Test error when USE_RELATIVE_VELOCITIES=False conflicts with V_CB_MODEL!='NONE'."""
    use_relative_veclocities_wrong = v_cb_model == "NONE"
    with pytest.raises(
        ValueError,
        match=f"V_CB_MODEL is set to '{v_cb_model}' but USE_RELATIVE_VELOCITIES is {use_relative_veclocities_wrong}",
    ):
        MatterOptions(
            USE_RELATIVE_VELOCITIES=use_relative_veclocities_wrong,
            V_CB_MODEL=v_cb_model,
        )


@pytest.mark.filterwarnings("ignore:^USE_MCGS is False but V_CB_MODEL:UserWarning")
def test_fix_vcb_avg_conflict():
    """Test error when FIX_VCB_AVG conflicts with V_CB_MODEL."""
    for fix_vcb_avg in [True, False]:
        v_cb_model_wrong = "NONE" if fix_vcb_avg else "AVG-DEBUG"
        with (
            pytest.warns(
                deprecation.DeprecatedWarning, match="FIX_VCB_AVG is deprecated"
            ),
            pytest.raises(
                ValueError,
                match=f"FIX_VCB_AVG={fix_vcb_avg} is not compatible with ",
            ),
        ):
            InputParameters(
                random_seed=1,
                astro_options=AstroOptions(FIX_VCB_AVG=fix_vcb_avg),
                matter_options=MatterOptions(V_CB_MODEL=v_cb_model_wrong),
            )


# ── AstroParams ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("fix_vcb_avg", [True, False])
@pytest.mark.filterwarnings("ignore:^USE_MCGS is False but V_CB_MODEL:UserWarning")
def test_fix_vcb_avg_deprecated_warning(fix_vcb_avg):
    """Test that using FIX_VCB_AVG shows deprecation warning."""
    v_cb_model = "AVG-DEBUG" if fix_vcb_avg else "NONE"
    with pytest.warns(deprecation.DeprecatedWarning, match="FIX_VCB_AVG is deprecated"):
        inputs = InputParameters(
            random_seed=1,
            astro_options=AstroOptions(FIX_VCB_AVG=fix_vcb_avg),
            matter_options=MatterOptions(V_CB_MODEL=v_cb_model),
        )
    assert v_cb_model == inputs.matter_options.V_CB_MODEL
    assert fix_vcb_avg == inputs.astro_options.FIX_VCB_AVG


@pytest.mark.skipif(
    _MAJOR_VERSION < 5, reason="FIX_VCB_AVG removal scheduled for v5.0.0"
)
def test_fix_vcb_avg_is_removed():
    """Confirms FIX_VCB_AVG was removed in v5.0.0; runs only at v5+."""
    with pytest.raises(TypeError, match="FIX_VCB_AVG"):
        AstroOptions(FIX_VCB_AVG=True)


# ── Lightconers ───────────────────────────────────────────────────────────────


def test_with_equal_cdist_slices_deprecated_warning():
    """Test that with_equal_cdist_slices raises a deprecation warning."""
    with pytest.warns(
        deprecation.DeprecatedWarning, match="with_equal_cdist_slices is deprecated"
    ):
        lcn.RectilinearLightconer.with_equal_cdist_slices(
            min_redshift=6.0,
            max_redshift=7.0,
            resolution=2 * un.Mpc,
        )


@pytest.mark.skipif(
    _MAJOR_VERSION < 5, reason="with_equal_cdist_slices removal scheduled for v5.0.0"
)
def test_with_equal_cdist_slices_is_removed():
    """Confirms with_equal_cdist_slices was removed in v5.0.0; runs only at v5+."""
    with pytest.raises(AttributeError, match="with_equal_cdist_slices"):
        _ = lcn.RectilinearLightconer.with_equal_cdist_slices


# ── InputParameters ───────────────────────────────────────────────────────────


def test_zstep_factor_raises_warning():
    """Test that using zstep_factor raises a warning."""
    with pytest.warns(
        DeprecationWarning,
        match=r"The `zstep_factor` argument is deprecated and will be removed in a future version. Please use `step` instead.",
    ):
        InputParameters(random_seed=1).with_logspaced_redshifts(
            zstep_factor=0.5, zmin=5, zmax=15
        )


# ── cfuncs removed arguments ──────────────────────────────────────────────────


def test_removed_log10mturns_argument(default_input_struct):
    """Test that removed `log10mturns` arguments raise a TypeError with a message."""
    with pytest.raises(
        TypeError, match="`mturnovers` and `mturnovers_mini` have been removed"
    ):
        cf.compute_luminosity_function(
            inputs=default_input_struct,
            redshifts=[7, 8, 9],
            nbins=100,
            mturnovers=np.array([1e7, 1e8, 1e9]),
            mturnovers_mini=np.array([1e4, 1e5, 1e6]),
        )

    with pytest.raises(TypeError, match="`log10mturns` has been removed"):
        cf.evaluate_SFRD_z(
            inputs=default_input_struct,
            redshifts=[7, 8, 9],
            log10mturns=np.array([8.0, 8.5, 9.0]),
        )

    with pytest.raises(TypeError, match="`log10mturns` has been removed"):
        cf.evaluate_Nion_z(
            inputs=default_input_struct,
            redshifts=[7, 8, 9],
            log10mturns=np.array([8.0, 8.5, 9.0]),
        )

    with pytest.raises(TypeError, match="`log10mturns` has been removed"):
        cf.evaluate_SFRD_cond(
            inputs=default_input_struct,
            redshift=8.0,
            radius=5,
            densities=np.linspace(-0.98, 1.7, num=800),
            log10mturns=np.linspace(8.0, 9.0, num=800),
        )

    with pytest.raises(
        TypeError, match="`l10mturns_acg` and `l10mturns_mcg` have been removed"
    ):
        cf.evaluate_Nion_cond(
            inputs=default_input_struct,
            redshift=8.0,
            radius=5,
            densities=np.linspace(-0.98, 1.7, num=800),
            l10mturns_acg=np.linspace(8.0, 9.0, num=800),
            l10mturns_mcg=np.linspace(7.0, 8.0, num=800),
        )

    with pytest.raises(TypeError, match="`log10mturns` has been removed"):
        cf.evaluate_Xray_cond(
            inputs=default_input_struct,
            redshift=8.0,
            radius=5,
            densities=np.linspace(-0.98, 1.7, num=800),
            log10mturns=np.linspace(8.0, 9.0, num=800),
        )


def test_removed_arguments_are_cleaned_up_in_v5():
    """Reminder to remove the TypeError checks for log10mturns etc. in v5."""
    version = tuple(int(x) for x in p21c.__version__.split(".")[:2])
    if version >= (5, 0):
        pytest.fail(
            "Version is now >= 5.0 — please remove the deprecated `mturnovers`, "
            "`log10mturns`, `l10mturns_acg`, and `l10mturns_mcg` arguments and this test."
        )


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
        R_BUBBLE_MAX=50.0,
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
        R_BUBBLE_MAX=50.0,
        USE_MCGS=True,
        V_CB_MODEL="AVG-DEBUG",
        M_TURN_STELLAR_FEEDBACK=5.0,
        SOURCE_MODEL="L-INTEGRAL",
    )
    ic, pt, ef = computed_emissivity_fields_with_mcgs

    rf = compute_radiation_fields(
        emissivity_fields_list=[ef],
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

    with ExitStack() as stack:
        with pytest.warns(
            deprecation.DeprecatedWarning, match="EXTRA_HALOBOX_FIELDS is deprecated"
        ):
            stack.enter_context(config.use(EXTRA_HALOBOX_FIELDS=True))
        emissivity_fields = EmissivityFields.new(
            redshift=0.0, inputs=default_input_struct_lc
        )
        assert "halo_mass_density" in emissivity_fields.arrays
        assert "halo_number" in emissivity_fields.arrays
        assert "stellar_mass_density_acg" in emissivity_fields.arrays
        assert "stellar_mass_density_mcg" not in emissivity_fields.arrays

        emissivity_fields = EmissivityFields.new(
            redshift=0.0,
            inputs=default_input_struct_lc.evolve_input_structs(
                USE_TS_FLUCT=True,
                RECOMB_MODEL="inhomogeneous",
                R_BUBBLE_MAX=50.0,
                USE_MCGS=True,
                V_CB_MODEL="AVG-DEBUG",
                M_TURN_STELLAR_FEEDBACK=5.0,
            ),
        )
        assert "stellar_mass_density_mcg" in emissivity_fields.arrays

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous", R_BUBBLE_MAX=50.0, SOURCE_MODEL="L-INTEGRAL"
    )
    emissivity_fields = EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert "fesc_weighted_sfrd" in emissivity_fields.arrays


@deprecation.fail_if_not_removed
def test_extra_halobox_fields_is_removed(default_input_struct_lc):
    """Fails when removed_in version is reached, reminding you to delete EXTRA_HALOBOX_FIELDS."""
    with config.use(EXTRA_HALOBOX_FIELDS=True):
        emissivity_fields = EmissivityFields.new(
            redshift=0.0, inputs=default_input_struct_lc
        )
        assert "halo_mass_density" in emissivity_fields.arrays
        assert "halo_number" in emissivity_fields.arrays
        assert "stellar_mass_density_acg" in emissivity_fields.arrays
        assert "stellar_mass_density_mcg" not in emissivity_fields.arrays

        emissivity_fields = EmissivityFields.new(
            redshift=0.0,
            inputs=default_input_struct_lc.evolve_input_structs(
                USE_TS_FLUCT=True,
                RECOMB_MODEL="inhomogeneous",
                R_BUBBLE_MAX=50.0,
                USE_MCGS=True,
                V_CB_MODEL="AVG-DEBUG",
                M_TURN_STELLAR_FEEDBACK=5.0,
            ),
        )
        assert "stellar_mass_density_mcg" in emissivity_fields.arrays

    inputs = default_input_struct_lc.evolve_input_structs(
        RECOMB_MODEL="inhomogeneous", R_BUBBLE_MAX=50.0, SOURCE_MODEL="L-INTEGRAL"
    )
    emissivity_fields = EmissivityFields.new(redshift=0.0, inputs=inputs)
    assert "fesc_weighted_sfrd" in emissivity_fields.arrays


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
        assert isinstance(ef.count, np.ndarray)
    with pytest.warns(deprecation.DeprecatedWarning, match="count is deprecated"):
        assert np.all(ef.count == ef.halo_number)


@deprecation.fail_if_not_removed
def test_count_is_removed(computed_emissivity_fields):
    """Fails when removed_in version is reached, reminding you to delete count."""
    _, _, ef = computed_emissivity_fields
    assert isinstance(ef.count, np.ndarray)
    assert np.all(ef.count == ef.halo_number)


def test_halo_mass_deprecated_warning(computed_emissivity_fields):
    """Test that using halo_mass shows deprecation warning."""
    _, _, ef = computed_emissivity_fields
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_mass is deprecated"):
        assert isinstance(ef.halo_mass, np.ndarray)
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_mass is deprecated"):
        assert np.all(ef.halo_mass == ef.halo_mass_density)


@deprecation.fail_if_not_removed
def test_halo_mass_is_removed(computed_emissivity_fields):
    """Fails when removed_in version is reached, reminding you to delete halo_mass."""
    _, _, ef = computed_emissivity_fields
    assert isinstance(ef.halo_mass, np.ndarray)
    assert np.all(ef.halo_mass == ef.halo_mass_density)


def test_halo_stars_deprecated_warning(computed_emissivity_fields):
    """Test that using halo_stars shows deprecation warning."""
    _, _, ef = computed_emissivity_fields
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_stars is deprecated"):
        assert isinstance(ef.halo_stars, np.ndarray)
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_stars is deprecated"):
        assert np.all(ef.halo_stars == ef.stellar_mass_density_acg)


@deprecation.fail_if_not_removed
def test_halo_stars_is_removed(computed_emissivity_fields):
    """Fails when removed_in version is reached, reminding you to delete halo_stars."""
    _, _, ef = computed_emissivity_fields
    assert isinstance(ef.halo_stars, np.ndarray)
    assert np.all(ef.halo_stars == ef.stellar_mass_density_acg)


def test_halo_stars_mini_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using halo_stars_mini shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(
        deprecation.DeprecatedWarning, match="halo_stars_mini is deprecated"
    ):
        assert isinstance(ef.halo_stars_mini, np.ndarray)
    with pytest.warns(
        deprecation.DeprecatedWarning, match="halo_stars_mini is deprecated"
    ):
        assert np.all(ef.halo_stars_mini == ef.stellar_mass_density_mcg)


@deprecation.fail_if_not_removed
def test_halo_stars_mini_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete halo_stars_mini."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert isinstance(ef.halo_stars_mini, np.ndarray)
    assert np.all(ef.halo_stars_mini == ef.stellar_mass_density_mcg)


def test_halo_sfr_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using halo_sfr shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_sfr is deprecated"):
        assert isinstance(ef.halo_sfr, np.ndarray)
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_sfr is deprecated"):
        assert np.all(ef.halo_sfr == ef.sfrd_acg)


@deprecation.fail_if_not_removed
def test_halo_sfr_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete halo_sfr."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert isinstance(ef.halo_sfr, np.ndarray)
    assert np.all(ef.halo_sfr == ef.sfrd_acg)


def test_halo_sfr_mini_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using halo_sfr_mini shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(
        deprecation.DeprecatedWarning, match="halo_sfr_mini is deprecated"
    ):
        assert isinstance(ef.halo_sfr_mini, np.ndarray)
    with pytest.warns(
        deprecation.DeprecatedWarning, match="halo_sfr_mini is deprecated"
    ):
        assert np.all(ef.halo_sfr_mini == ef.sfrd_mcg)


@deprecation.fail_if_not_removed
def test_halo_sfr_mini_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete halo_sfr_mini."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert isinstance(ef.halo_sfr_mini, np.ndarray)
    assert np.all(ef.halo_sfr_mini == ef.sfrd_mcg)


def test_halo_xray_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using halo_xray shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_xray is deprecated"):
        assert isinstance(ef.halo_xray, np.ndarray)
    with pytest.warns(deprecation.DeprecatedWarning, match="halo_xray is deprecated"):
        assert np.all(ef.halo_xray == ef.xray_emissivity)


@deprecation.fail_if_not_removed
def test_halo_xray_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete halo_xray."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert isinstance(ef.halo_xray, np.ndarray)
    assert np.all(ef.halo_xray == ef.xray_emissivity)


def test_whalo_sfr_deprecated_warning(computed_emissivity_fields_with_mcgs):
    """Test that using whalo_sfr shows deprecation warning."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    with pytest.warns(deprecation.DeprecatedWarning, match="whalo_sfr is deprecated"):
        assert isinstance(ef.whalo_sfr, np.ndarray)
    with pytest.warns(deprecation.DeprecatedWarning, match="whalo_sfr is deprecated"):
        assert np.all(ef.whalo_sfr == ef.fesc_weighted_sfrd)


@deprecation.fail_if_not_removed
def test_whalo_sfr_is_removed(computed_emissivity_fields_with_mcgs):
    """Fails when removed_in version is reached, reminding you to delete whalo_sfr."""
    _, _, ef = computed_emissivity_fields_with_mcgs
    assert isinstance(ef.whalo_sfr, np.ndarray)
    assert np.all(ef.whalo_sfr == ef.fesc_weighted_sfrd)


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
        assert np.allclose(ic.arrays["hires_density"].value, expected)

    # It must keep working for a purged array, where it used to return None -- that
    # silent None is what made tests reading `.value` off the shared session-scoped
    # `ic` fixture fail depending on whether a driver had purged it first.
    ic.purge()
    assert ic.arrays["hires_density"]._value is None

    with pytest.warns(deprecation.DeprecatedWarning, match="Array.value is deprecated"):
        assert np.allclose(ic.arrays["hires_density"].value, expected)

    ic.load_all()


@pytest.mark.parametrize("name", ["value", "state", "purged_to_disk"])
def test_struct_attribute_array_api_deprecated_warning(
    ic: InitialConditions, name: str
):
    """`OutputStruct` array attributes are numpy arrays now, not `Array`s.

    The transitional view keeps the old `Array`-only API working for one release so
    that existing user code doesn't break outright, but warns when it is used.
    """
    density = ic.hires_density
    assert isinstance(density, np.ndarray)

    with pytest.warns(deprecation.DeprecatedWarning, match=f"{name} is deprecated"):
        assert getattr(density, name) is not None

    # Genuine numpy attributes are untouched, and raise normally when absent.
    assert density.dtype is not None
    with pytest.raises(AttributeError):
        _ = density.not_a_real_attribute


@deprecation.fail_if_not_removed
def test_array_value_is_removed(ic: InitialConditions):
    """Fails when removed_in version is reached, reminding you to delete Array.value."""
    assert np.allclose(ic.arrays["hires_density"].value, ic.get("hires_density"))
    # ... and the transitional _LegacyArrayView along with it.
    assert np.allclose(ic.hires_density.value, ic.get("hires_density"))
