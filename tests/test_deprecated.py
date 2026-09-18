"""
Tests for deprecated parameters and APIs.

This module consolidates all deprecation warning tests in one place.
Each deprecated parameter should have:
1. A test verifying the deprecation warning fires correctly.
2. A test decorated with @pytest.mark.skipif(_MAJOR_VERSION < 5, ...) that
   will run at v5+ and confirm the parameter was actually removed. These
   tests are skipped at v4.x since the parameter is still present.

Note: @deprecation.fail_if_not_removed is not used here because our
warnings are emitted via warnings.warn(DeprecatedWarning(...)) rather
than the @deprecation.deprecated decorator. The latter automatically
upgrades to UnsupportedWarning when removed_in is reached, which is
what fail_if_not_removed catches. Our manual approach requires explicit
version guards instead.

In v5, remove the obsolete deprecation-warning tests and review the
removal checks and their version guards.
"""

import deprecation
import numpy as np
import pytest
from astropy import units as un

import py21cmfast as p21c
from py21cmfast import lightconers as lcn
from py21cmfast.wrapper import cfuncs as cf
from py21cmfast.wrapper.inputs import (
    AstroOptions,
    AstroParams,
    InputParameters,
    MatterOptions,
)

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


# When fix_vcb_avg=False, the test sets V_CB_MODEL="AVG-DEBUG" to trigger the
# FIX_VCB_AVG conflict error. This incidentally fires the USE_MINI_HALOS/V_CB_MODEL
# advisory since USE_MINI_HALOS defaults to False. The warning is suppressed here
# because fixing the configuration (adding USE_MINI_HALOS=True) would require
# RECOMB_MODEL and USE_TS_FLUCT changes that obscure what the test is verifying.
@pytest.mark.filterwarnings(
    "ignore:^USE_MINI_HALOS is False but V_CB_MODEL:UserWarning"
)
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
# When FIX_VCB_AVG=False, USE_MINI_HALOS is False but V_CB_MODEL is non-trivial,
# triggering this parameter mismatch advisory. The advisory is unrelated to the
# FIX_VCB_AVG deprecation behavior under test.
@pytest.mark.filterwarnings(
    "ignore:^USE_MINI_HALOS is False but V_CB_MODEL:UserWarning"
)
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
