"""Test the C interpolation tables."""

import matplotlib as mpl
import numpy as np
import pytest
from astropy import constants as c
from astropy import units as u

import py21cmfast.c_21cmfast as lib
from py21cmfast import AstroOptions, AstroParams, CosmoParams, SimulationOptions
from py21cmfast.wrapper import cfuncs as cf

from . import produce_integration_test_data as prd

# NOTE: The relative tolerance is set to cover the inaccuracy in interpolaton
#       Whereas absolute tolerances are set to avoid issues with minima
#       i.e the SFRD table has a forced minima of exp(-50)
#       Currently (with the #defined bin numbers) the minihalo tables have a ~1.5 % error maximum
#       With >1% error in less than 4% of bins (at the high turnover mass where fcoll is tiny)
#       The rest of the tables (apart from the xfail inverse tables) are well under 1% error

# The tables are much better than 2% but there are some problem bins (very high density, very small mass ranges)
RELATIVE_TOLERANCE = 2e-2

OPTIONS_PS = {
    "EH": [10, {"POWER_SPECTRUM": "EH"}],
    "BBKS": [10, {"POWER_SPECTRUM": "BBKS"}],
    "BE": [10, {"POWER_SPECTRUM": "EFSTATHIOU"}],
    "Peebles": [10, {"POWER_SPECTRUM": "PEEBLES"}],
    "White": [10, {"POWER_SPECTRUM": "WHITE"}],
    "CLASS": [10, {"POWER_SPECTRUM": "CLASS"}],
}

OPTIONS_HMF = {
    "PS": [10, {"HMF": "PS", "USE_MASS_DEPENDENT_ZETA": True, "M_MIN_in_Mass": True}],
    "ST": [10, {"HMF": "ST", "USE_MASS_DEPENDENT_ZETA": True, "M_MIN_in_Mass": True}],
}

OPTIONS_INTMETHOD = {
    "QAG": "GSL-QAG",
    "GL": "GAUSS-LEGENDRE",
    "FFCOLL": "GAMMA-APPROX",
}

R_PARAM_LIST = [1.5, 5, 10, 30, 60]

options_ps = list(OPTIONS_PS.keys())
options_hmf = list(OPTIONS_HMF.keys())

# This is confusing and we should change the dict to a list
options_intmethod = list(OPTIONS_INTMETHOD.keys())
# the minihalo ffcoll tables have some bins (when Mturn -> M_turn_upper) which go above 10% error compared to their "integrals"
#    they can pass by doubling the number of M_turn bins and setting relative error to 5% but I think this
#    is better left for later


# Test delta range for CMF integrals over cells
@pytest.fixture(scope="module")
def delta_range():
    return np.linspace(-0.98, 1.7, num=100)


# Mass condition range, and integral bound range for testing CMF integrals
@pytest.fixture(scope="module")
def mass_range():
    return np.logspace(7, 13, num=200)


# Mass turnover range for testing CMF/UMF integrals
@pytest.fixture(scope="module")
def log10_mturn_range():
    return np.linspace(5, 8.5, num=40)


# redshift range for testing UMF integrals
@pytest.fixture(scope="module")
def z_range():
    return np.linspace(6, 35, num=100)


@pytest.mark.parametrize("name", options_ps)
def test_sigma_table(name, mass_range, plt):
    abs_tol = 0
    redshift, kwargs = OPTIONS_PS[name]
    inputs = prd.get_all_options_struct(redshift, **kwargs)["inputs"]

    sigma_tables, dsigma_tables = cf.evaluate_sigma(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="sigma-interpolation"
        ),
        masses=mass_range,
    )

    sigma_integrals, dsigma_integrals = cf.evaluate_sigma(
        inputs=inputs.evolve_input_structs(USE_INTERPOLATION_TABLES="no-interpolation"),
        masses=mass_range,
    )

    if plt == mpl.pyplot:
        make_table_comparison_plot(
            [mass_range, mass_range],
            [None, None],
            [sigma_tables, -dsigma_tables],
            [sigma_integrals, -dsigma_integrals],
            plt,
            xlabels=["Mass", "Mass"],
            ylabels=["sigma", "dsigmasqdM"],
        )

    np.testing.assert_allclose(
        sigma_integrals, sigma_tables, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        dsigma_integrals, dsigma_tables, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("cond_type", ["cat", "grid"])
def test_massfunc_conditional_tables(name, cond_type, mass_range, delta_range, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    inputs = prd.get_all_options_struct(redshift, **kwargs)["inputs"]
    from_cat = "cat" in cond_type

    inputs_cond = mass_range if from_cat else delta_range
    z_desc = (
        (redshift + 1) / inputs.simulation_options.ZPRIME_STEP_FACTOR - 1
        if from_cat
        else None
    )

    # get the tables
    nhalo_tbl, mcoll_tbl = cf.evaluate_condition_integrals(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="hmf-interpolation"
        ),
        redshift=redshift,
        redshift_prev=z_desc,
        cond_array=inputs_cond,
    )
    # get the integrals
    nhalo_exp, mcoll_exp = cf.evaluate_condition_integrals(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="sigma-interpolation"
        ),
        redshift=redshift,
        redshift_prev=z_desc,
        cond_array=inputs_cond,
    )

    if plt == mpl.pyplot:
        make_table_comparison_plot(
            [inputs_cond, inputs_cond],
            [None, None],
            [nhalo_tbl, mcoll_tbl],
            [nhalo_exp, mcoll_exp],
            plt,
            logx=from_cat,
            xlabels=["Mass" if from_cat else "delta"] * 2,
            ylabels=["Nhalo", "Mcoll"],
        )

    print_failure_stats(
        nhalo_tbl,
        nhalo_exp,
        [inputs_cond],
        0.0,
        RELATIVE_TOLERANCE,
        "expected N halo",
    )
    print_failure_stats(
        mcoll_tbl,
        mcoll_exp,
        [inputs_cond],
        0.0,
        RELATIVE_TOLERANCE,
        "expected M halo",
    )

    np.testing.assert_allclose(nhalo_exp, nhalo_tbl, rtol=RELATIVE_TOLERANCE)
    np.testing.assert_allclose(mcoll_exp, mcoll_tbl, rtol=RELATIVE_TOLERANCE)


@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("cond_type", ["cat", "grid"])
def test_inverse_cmf_tables(name, cond_type, delta_range, mass_range, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    inputs = prd.get_all_options_struct(redshift, **kwargs)["inputs"]

    from_cat = "cat" in cond_type
    lnMmin_range = np.log(mass_range)
    inputs_cond = mass_range if from_cat else delta_range
    z_desc = (
        (redshift + 1) / inputs.simulation_options.ZPRIME_STEP_FACTOR - 1
        if from_cat
        else None
    )

    # Get the full integrals
    nhalo, _ = cf.evaluate_condition_integrals(
        inputs=inputs,
        redshift=redshift,
        redshift_prev=z_desc,
        cond_array=inputs_cond,
    )
    # Get the Integrals on mass limits
    cmf_integral = cf.integrate_chmf_interval(
        inputs=inputs,
        lnm_lower=lnMmin_range,
        lnm_upper=np.full_like(lnMmin_range, 16 * np.log(10)),
        cond_values=inputs_cond,
        redshift=redshift,
        redshift_prev=z_desc,
    )
    probabilities = cmf_integral / nhalo[:, None]

    inputs_cond_bc, inputs_mmin_bc = np.meshgrid(
        inputs_cond, lnMmin_range, indexing="ij"
    )
    # Take those probabilites to the inverse table
    icmf_table = cf.evaluate_inverse_table(
        inputs=inputs,
        probabilities=probabilities,
        cond_array=inputs_cond_bc,
        redshift=redshift,
        redshift_prev=z_desc,
    )

    # ignore condition out-of-bounds (returned as -1 by evaluate_inverse_table)
    sel_oob = icmf_table < 0.0
    # ignore probability out-of-bounds (extrapolated by the backend)
    sel_lowprob = cmf_integral < np.exp(inputs.simulation_options.MIN_LOGPROB)
    if plt == mpl.pyplot:
        last_cond = np.amax(np.where(np.any(~sel_oob, axis=-1)))
        first_cond = np.amin(np.where(np.any(~sel_oob, axis=-1)))
        sel_plot = np.linspace(first_cond, last_cond, num=6).astype(int)
        make_table_comparison_plot(
            [probabilities[sel_plot, :].T],
            [inputs_cond[sel_plot]],
            [icmf_table[sel_plot, :].T],
            [np.exp(lnMmin_range)],
            plt,
            zlabels=[r"$\delta =$" if not from_cat else r"$M=$"],
            logx=True,
            logy=True,
            label_test=[False, False],
            xlabels=["Probability"],
            ylabels=["Mass"],
            xlim=[np.exp(inputs.simulation_options.MIN_LOGPROB) / 10, 1.0],
            reltol=RELATIVE_TOLERANCE,
        )

    # easiest way to ignore in testing is to fix values to nan
    inputs_mmin_bc[sel_lowprob | sel_oob] = np.nan
    icmf_table[sel_lowprob | sel_oob] = np.nan

    # NOTE: very dense cells can have ~3.5% errors at low mass, other bins much more accurate
    # Previously I didn't test cells near delta_crit, but I think a 5% tolerance test is better
    rtol = 5e-2
    print_failure_stats(
        np.log(icmf_table),
        inputs_mmin_bc,
        [inputs_cond, probabilities],
        0.0,
        rtol,
        "Inverse CMF",
    )

    np.testing.assert_allclose(
        inputs_mmin_bc,
        np.log(icmf_table),
        rtol=rtol,
    )


@pytest.mark.parametrize("R", R_PARAM_LIST)
def test_FgtrM_conditional_tables(R, delta_range, plt):
    redshift, kwargs = OPTIONS_HMF["PS"]  # always erfc
    inputs = prd.get_all_options_struct(redshift, **kwargs)["inputs"]

    fcoll_tables, dfcoll_tables = cf.evaluate_FgtrM_cond(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="hmf-interpolation"
        ),
        redshift=redshift,
        densities=delta_range,
        R=R,
    )
    fcoll_integrals, dfcoll_integrals = cf.evaluate_FgtrM_cond(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="sigma-interpolation"
        ),
        redshift=redshift,
        densities=delta_range,
        R=R,
    )

    if plt == mpl.pyplot:
        make_table_comparison_plot(
            [delta_range, delta_range],
            [None, None],
            [fcoll_tables, np.fabs(dfcoll_tables)],
            [fcoll_integrals, np.fabs(dfcoll_integrals)],
            plt,
            xlabels=["delta", "delta"],
            ylabels=["fcoll", "dfolldz"],
        )

    abs_tol = 5e-6
    print_failure_stats(
        fcoll_tables,
        fcoll_integrals,
        [delta_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "fcoll",
    )

    print_failure_stats(
        dfcoll_tables,
        dfcoll_integrals,
        [delta_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "dfcoll",
    )

    np.testing.assert_allclose(
        fcoll_tables, fcoll_integrals, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        dfcoll_tables, dfcoll_integrals, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("name", options_hmf)
def test_SFRD_z_tables(name, z_range, log10_mturn_range, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    inputs = prd.get_all_options_struct(
        redshift,
        USE_MINI_HALOS=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
        **kwargs,
    )["inputs"]

    z_input, mt_input = np.meshgrid(z_range, log10_mturn_range, indexing="ij")
    SFRD_tables, SFRD_tables_mini = cf.evaluate_SFRD_z(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="hmf-interpolation"
        ),
        redshifts=z_input,
        log10mturns=mt_input,
    )
    SFRD_integrals, SFRD_integrals_mini = cf.evaluate_SFRD_z(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="sigma-interpolation"
        ),
        redshifts=z_input,
        log10mturns=mt_input,
    )

    if plt == mpl.pyplot:
        xl = log10_mturn_range.size - 1
        sel_m = np.linspace(0, xl, num=5).astype(int)
        make_table_comparison_plot(
            [z_range, z_range],
            [np.array([0]), 10 ** log10_mturn_range[sel_m]],
            [SFRD_tables[..., 0], SFRD_tables_mini[..., sel_m]],
            [SFRD_integrals[..., 0], SFRD_integrals_mini[..., sel_m]],
            plt,
            label_test=[True, False],
            xlabels=["redshift", "redshift"],
            ylabels=["SFRD", "SFRD_mini"],
        )

    abs_tol = 1e-5
    print_failure_stats(
        SFRD_tables,
        SFRD_integrals,
        [z_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "SFRD_z",
    )
    print_failure_stats(
        SFRD_tables_mini,
        SFRD_integrals_mini,
        [z_range, 10**log10_mturn_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "SFRD_z_mini",
    )

    np.testing.assert_allclose(
        SFRD_tables, SFRD_integrals, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        SFRD_tables_mini, SFRD_integrals_mini, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("name", options_hmf)
def test_Nion_z_tables(name, z_range, log10_mturn_range, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    inputs = prd.get_all_options_struct(
        redshift,
        USE_MINI_HALOS=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
        **kwargs,
    )["inputs"]

    z_input, mt_input = np.meshgrid(z_range, log10_mturn_range, indexing="ij")
    nion_tables, nion_tables_mini = cf.evaluate_Nion_z(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="hmf-interpolation"
        ),
        redshifts=z_input,
        log10mturns=mt_input,
    )
    nion_integrals, nion_integrals_mini = cf.evaluate_Nion_z(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="sigma-interpolation"
        ),
        redshifts=z_input,
        log10mturns=mt_input,
    )

    if plt == mpl.pyplot:
        xl = log10_mturn_range.size - 1
        sel_m = np.linspace(0, xl, num=5).astype(int)
        make_table_comparison_plot(
            [z_range, z_range],
            [np.array([0]), 10 ** log10_mturn_range[sel_m]],
            [nion_tables[..., 0], nion_tables_mini[..., sel_m]],
            [nion_integrals[..., 0], nion_integrals_mini[..., sel_m]],
            plt,
            label_test=[True, False],
            xlabels=["redshift", "redshift"],
            ylabels=["Nion", "Nion_mini"],
        )

    abs_tol = 2e-6
    print_failure_stats(
        nion_tables,
        nion_integrals,
        [z_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "SFRD_z",
    )
    print_failure_stats(
        nion_tables_mini,
        nion_integrals_mini,
        [z_range, 10**log10_mturn_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "SFRD_z_mini",
    )

    np.testing.assert_allclose(
        nion_tables, nion_integrals, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        nion_tables_mini, nion_integrals_mini, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )


# A few notes on this test function:
#   Mass limits are set explicitly using the param values
#       and #defines are copied to hard code (not ideal)
#   Density and Mturn limits are set to their maxima since we don't have cubes.
#       Hence this is a worst case scenario
#   While the EvaluateX() functions are useful in the main code to be agnostic to USE_INTERPOLATION_TABLES
#       I do not use them here fully, instead calling the integrals directly to avoid parameter changes
#       Mostly since if we set simulation_options.USE_INTERPOLATION_TABLES=False then the sigma tables aren't used
#       and it takes forever
@pytest.mark.parametrize("mini", ["mini", "acg"])
@pytest.mark.parametrize("R", R_PARAM_LIST)
@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("intmethod", options_intmethod)
def test_Nion_conditional_tables(
    name, log10_mturn_range, delta_range, R, mini, intmethod, plt
):
    if intmethod == "FFCOLL":
        if name != "PS":
            pytest.skip("FAST FFCOLL INTEGRALS WORK ONLY WITH EPS")
        else:
            pytest.xfail(
                "FFCOLL TABLES drop sharply at high Mturn, causing failure at 0.1 levels"
            )

    mini_flag = mini == "mini"
    redshift, kwargs = OPTIONS_HMF[name]
    inputs = prd.get_all_options_struct(
        redshift,
        USE_MINI_HALOS=mini_flag,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
        **kwargs,
    )["inputs"]

    # NOTE: we still pass a 2D array when minihalos are off.
    #   part of the test is that passing different mturns *doesn't* change
    #   the integral without minihalos
    d_input, mt_input = np.meshgrid(delta_range, log10_mturn_range, indexing="ij")

    Nion_tables, Nion_tables_mini = cf.evaluate_Nion_cond(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="hmf-interpolation"
        ),
        redshift=redshift,
        radius=R,
        densities=d_input,
        l10mturns_mcg=mt_input,
        l10mturns_acg=mt_input,
    )

    Nion_integrals, Nion_integrals_mini = cf.evaluate_Nion_cond(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="sigma-interpolation"
        ),
        redshift=redshift,
        radius=R,
        densities=d_input,
        l10mturns_mcg=mt_input,
        l10mturns_acg=mt_input,
    )

    # The bilinear interpolation we use underperforms at high mturn due to the sharp
    # dropoff. Setting an absolute tolerance to a level we care about for reionisation
    # rather than a level we expect from interp tables remedies this for now.
    # TODO: In future we should investigate cubic splines etc.
    abs_tol = 1e-8

    if plt == mpl.pyplot:
        xl = log10_mturn_range.size - 1
        sel_m = np.linspace(0, xl, num=5).astype(int)
        Nion_tb_plot = Nion_tables[..., sel_m]
        Nion_il_plot = Nion_integrals[..., sel_m]

        make_table_comparison_plot(
            [delta_range, delta_range],
            [np.array([0]), 10 ** log10_mturn_range[sel_m]],
            [Nion_tb_plot[..., 0], Nion_tables_mini[..., sel_m]],
            [Nion_il_plot[..., 0], Nion_integrals_mini[..., sel_m]],
            plt,
            label_test=[True, False],
            xlabels=["delta", "delta"],
            ylabels=["Nion", "Nion_mini"],
        )

    print_failure_stats(
        Nion_tables,
        Nion_integrals,
        [delta_range, 10**log10_mturn_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "Nion_c",
    )

    if mini_flag:
        print_failure_stats(
            Nion_tables_mini,
            Nion_integrals_mini,
            [delta_range, 10**log10_mturn_range],
            abs_tol,
            RELATIVE_TOLERANCE,
            "Nion_c_mini",
        )

    # We don't want to include values close to delta crit, since
    # interpolating across the sharp gap results in errors
    # TODO: the bound should be over MAX_DELTAC_FRAC*delta_crit, and we should interpolate
    # instead of setting the integral to its limit at delta crit.
    delta_crit = cf.get_delta_crit(
        inputs=inputs,
        mass=cf.get_condition_mass(inputs, R),
        redshift=redshift,
    )
    sel_delta = (delta_crit - delta_range) > 0.05

    np.testing.assert_allclose(
        Nion_tables[sel_delta],
        Nion_integrals[sel_delta],
        atol=abs_tol,
        rtol=RELATIVE_TOLERANCE,
    )

    np.testing.assert_allclose(
        Nion_tables_mini[sel_delta],
        Nion_integrals_mini[sel_delta],
        atol=abs_tol,
        rtol=RELATIVE_TOLERANCE,
    )


@pytest.mark.parametrize("mini", ["mini", "acg"])
@pytest.mark.parametrize("R", R_PARAM_LIST)
@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("intmethod", options_intmethod)
def test_Xray_conditional_tables(
    name, log10_mturn_range, delta_range, R, mini, intmethod, plt
):
    if intmethod == "FFCOLL":
        if name != "PS":
            pytest.skip("FAST FFCOLL INTEGRALS WORK ONLY WITH EPS")
        else:
            pytest.xfail(
                "FFCOLL TABLES drop sharply at high Mturn, causing failure at 0.1 levels"
            )

    mini_flag = mini == "mini"

    redshift, kwargs = OPTIONS_HMF[name]
    inputs = prd.get_all_options_struct(
        redshift,
        USE_MINI_HALOS=mini_flag,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
        **kwargs,
    )["inputs"]

    d_input, mt_input = np.meshgrid(delta_range, log10_mturn_range, indexing="ij")

    Xray_tables = cf.evaluate_Xray_cond(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="hmf-interpolation"
        ),
        redshift=redshift,
        radius=R,
        densities=d_input,
        log10mturns=mt_input,
    )

    Xray_integrals = cf.evaluate_Xray_cond(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="sigma-interpolation"
        ),
        redshift=redshift,
        radius=R,
        densities=d_input,
        log10mturns=mt_input,
    )

    if plt == mpl.pyplot:
        xl = log10_mturn_range.size - 1
        sel_m = np.linspace(0, xl, num=5).astype(int)
        Xray_tb_plot = Xray_tables[..., sel_m]
        Xray_il_plot = Xray_integrals[..., sel_m]
        make_table_comparison_plot(
            [delta_range],
            [10 ** log10_mturn_range[sel_m]],
            [Xray_tb_plot],
            [Xray_il_plot],
            plt,
            label_test=[
                True,
            ],
            xlabels=["delta"],
            ylabels=["Lx"],
        )

    abs_tol = 0.0
    print_failure_stats(
        Xray_tables,
        Xray_integrals,
        [delta_range, 10**log10_mturn_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "Xray_c",
    )

    delta_crit = cf.get_delta_crit(
        inputs=inputs,
        mass=cf.get_condition_mass(inputs, R),
        redshift=redshift,
    )
    sel_delta = (delta_crit - delta_range) > 0.05
    np.testing.assert_allclose(
        Xray_tables[sel_delta],
        Xray_integrals[sel_delta],
        atol=abs_tol,
        rtol=RELATIVE_TOLERANCE,
    )


@pytest.mark.parametrize("R", R_PARAM_LIST)
@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("intmethod", options_intmethod)
def test_SFRD_conditional_table(
    name, log10_mturn_range, delta_range, R, intmethod, plt
):
    if intmethod == "FFCOLL":
        if name != "PS":
            pytest.skip("FAST FFCOLL INTEGRALS WORK ONLY WITH EPS")
        else:
            pytest.xfail("FFCOLL TABLES drop sharply at high Mturn, causing failure")

    redshift, kwargs = OPTIONS_HMF[name]
    inputs = prd.get_all_options_struct(
        redshift,
        USE_MINI_HALOS=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
        **kwargs,
    )["inputs"]

    d_input, mt_input = np.meshgrid(delta_range, log10_mturn_range, indexing="ij")

    SFRD_tables, SFRD_tables_mini = cf.evaluate_SFRD_cond(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="hmf-interpolation"
        ),
        radius=R,
        redshift=redshift,
        densities=d_input,
        log10mturns=mt_input,
    )

    SFRD_integrals, SFRD_integrals_mini = cf.evaluate_SFRD_cond(
        inputs=inputs.evolve_input_structs(
            USE_INTERPOLATION_TABLES="sigma-interpolation"
        ),
        radius=R,
        redshift=redshift,
        densities=d_input,
        log10mturns=mt_input,
    )

    # The bilinear interpolation we use underperforms at high mturn due to the sharp
    # dropoff. Setting an absolute tolerance to a level we care about for reionisation
    # rather than a level we expect from interp tables remedies this for now.
    # TODO: In future we should investigate cubic splines etc.
    abs_tol = 1e-8
    if plt == mpl.pyplot:
        xl = log10_mturn_range.size - 1
        sel_m = np.linspace(0, xl, num=5).astype(int)
        make_table_comparison_plot(
            [delta_range, delta_range],
            [np.array([0]), 10 ** log10_mturn_range[sel_m]],
            [SFRD_tables[:, 0], SFRD_tables_mini[..., sel_m]],
            [SFRD_integrals[:, 0], SFRD_integrals_mini[..., sel_m]],
            plt,
            label_test=[True, False],
            xlabels=["delta", "delta"],
            ylabels=["SFRD", "SFRD_mini"],
        )

    print_failure_stats(
        SFRD_tables,
        SFRD_integrals,
        [delta_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "SFRD_c",
    )
    print_failure_stats(
        SFRD_tables_mini,
        SFRD_integrals_mini,
        [delta_range, 10**log10_mturn_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "SFRD_c_mini",
    )

    delta_crit = cf.get_delta_crit(
        inputs=inputs,
        mass=cf.get_condition_mass(inputs, R),
        redshift=redshift,
    )
    sel_delta = (delta_crit - delta_range) > 0.05
    np.testing.assert_allclose(
        SFRD_tables[sel_delta],
        SFRD_integrals[sel_delta],
        atol=abs_tol,
        rtol=RELATIVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        SFRD_tables_mini[sel_delta],
        SFRD_integrals_mini[sel_delta],
        atol=abs_tol,
        rtol=RELATIVE_TOLERANCE,
    )


INTEGRAND_OPTIONS = ["sfrd", "n_ion"]


@pytest.mark.parametrize("R", R_PARAM_LIST)
@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("integrand", INTEGRAND_OPTIONS)
def test_conditional_integral_methods(
    R, log10_mturn_range, delta_range, name, integrand, plt
):
    redshift, kwargs = OPTIONS_HMF[name]
    inputs = prd.get_all_options_struct(
        redshift,
        USE_MINI_HALOS=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
        **kwargs,
    )["inputs"]

    d_input, mt_input = np.meshgrid(delta_range, log10_mturn_range, indexing="ij")

    integrals = []
    integrals_mini = []
    for method in ["GSL-QAG", "GAUSS-LEGENDRE", "GAMMA-APPROX"]:
        print(f"Starting method {method}", flush=True)
        if name != "PS" and method == "GAMMA-APPROX":
            continue

        inputs = inputs.evolve_input_structs(
            INTEGRATION_METHOD_ATOMIC=method,
            INTEGRATION_METHOD_MINI=method,
        )

        if "sfr" in integrand:
            buf, buf_mini = cf.evaluate_SFRD_cond(
                inputs=inputs,
                redshift=redshift,
                radius=R,
                densities=d_input,
                log10mturns=mt_input,
            )
        else:
            buf, buf_mini = cf.evaluate_Nion_cond(
                inputs=inputs,
                redshift=redshift,
                radius=R,
                densities=d_input,
                l10mturns_acg=mt_input,
                l10mturns_mcg=mt_input,
            )
        integrals.append(buf)
        integrals_mini.append(buf_mini)

    abs_tol = 1e-6  # minimum = exp(-40) ~1e-18
    if plt == mpl.pyplot:
        xl = log10_mturn_range.size - 1
        sel_m = np.linspace(0, xl, num=5).astype(int)
        iplot = [i[..., sel_m] if i.ndim == 2 else i for i in integrals]
        iplot_mini = [i[..., sel_m] for i in integrals_mini]
        make_integral_comparison_plot(
            delta_range,
            10 ** log10_mturn_range[sel_m],
            iplot,
            iplot_mini,
            plt,
        )

    np.testing.assert_allclose(
        integrals[1], integrals[0], atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        integrals_mini[1], integrals_mini[0], atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )


def make_table_comparison_plot(
    x,
    tb_z,
    tables,
    integrals,
    plt,
    **kwargs,
):
    # rows = values,fracitonal diff, cols = 1d table, 2d table
    fig, axs = plt.subplots(
        nrows=2, ncols=len(x), figsize=(12 * len(x) / 2, 9), squeeze=False
    )
    xlabels = kwargs.pop("xlabels", ["delta"] * len(x))
    ylabels = kwargs.pop("ylabels", ["MF_integral"] * len(x))
    zlabels = kwargs.pop("zlabels", ["Mturn"] * len(x))
    label_flags = kwargs.pop("label_test", [True] * len(tb_z))
    for j, z in enumerate(tb_z):
        n_lines = z.size if z is not None else 1
        for i in range(n_lines):
            zlab = zlabels[j] + f" = {z[i]:.2e}" if z is not None else ""
            # allow single arrays
            x_plot = x[j][:, i] if len(x[j].shape) > 1 else x[j]
            i_plot = integrals[j][:, i] if len(integrals[j].shape) > 1 else integrals[j]
            t_plot = tables[j][:, i] if len(tables[j].shape) > 1 else tables[j]
            make_comparison_plot(
                x_plot,
                i_plot,
                t_plot,
                ax=axs[:, j],
                xlab=xlabels[j],
                ylab=ylabels[j],
                label_base=zlab,
                label_test=label_flags[j],
                logx=kwargs.pop("logx", False),
                xlim=kwargs.pop("xlim", None),
                reltol=kwargs.pop("reltol", None),
                color=f"C{i:d}",
            )


# slightly different from comparison plot since each integral shares a "truth"
def make_integral_comparison_plot(x1, x2, integral_list, integral_list_second, plt):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))

    styles = ["-", ":", "--"]
    for i, (i_first, i_second) in enumerate(
        zip(integral_list, integral_list_second, strict=True)
    ):
        comparison = integral_list[0]
        if len(i_first.shape) == 1:
            i_first = i_first[:, None]
            comparison = integral_list[0][:, None]
        for j in range(i_first.shape[1]):
            axs[0, 0].semilogy(
                x1, i_first[:, j], color=f"C{j:d}", linestyle=styles[i], linewidth=2
            )
            axs[1, 0].semilogy(
                x1,
                i_first[:, j] / comparison[:, j],
                color=f"C{j:d}",
                linestyle=styles[i],
                linewidth=2,
            )

        for j in range(i_second.shape[1]):
            axs[0, 1].semilogy(x1, i_second[:, j], color=f"C{j:d}", linestyle=styles[i])
            axs[1, 1].semilogy(
                x1,
                i_second[:, j] / integral_list_second[0][:, j],
                color=f"C{j:d}",
                linestyle=styles[i],
            )

    axs[1, 0].set_xlabel("delta")
    axs[1, 1].set_xlabel("delta")
    axs[1, 0].set_ylabel("Integral")
    axs[0, 0].set_ylabel("Integral")
    axs[0, 0].grid()
    axs[0, 1].grid()
    axs[1, 0].grid()
    axs[1, 1].grid()


# copied and expanded from test_integration_features.py
def make_comparison_plot(
    x,
    true,
    test,
    ax,
    logx=True,
    logy=True,
    xlab=None,
    ylab=None,
    xlim=None,
    reltol=None,
    label_base="",
    label_test=True,
    **kwargs,
):
    true_label = label_base + " True" if label_test else label_base
    test_label = label_base + " Test" if label_test else None
    if logx:
        ax[0].set_xscale("log")
        ax[1].set_xscale("log")
    if logy:
        ax[0].set_yscale("log")
    if xlab:
        ax[0].set_xlabel(xlab)
        ax[1].set_xlabel(xlab)
    if ylab:
        ax[0].set_ylabel(ylab)
    if xlim:
        ax[0].set_xlim(xlim)
        ax[1].set_xlim(xlim)

    ax[0].grid()
    ax[1].grid()

    ax[0].plot(x, true, label=true_label, linestyle="-", **kwargs)
    ax[0].plot(x, test, label=test_label, linestyle=":", linewidth=3, **kwargs)
    ax[0].legend()

    ax[1].plot(x, (test - true) / true, **kwargs)
    ax[1].set_ylabel("Fractional Difference")
    if reltol:
        ax[1].set_ylim([-2 * reltol, 2 * reltol])


def print_failure_stats(test, truth, inputs, abs_tol, rel_tol, name):
    sel_failed = np.fabs(truth - test) > (abs_tol + np.fabs(truth) * rel_tol)
    if np.any(sel_failed):
        failed_idx = np.where(sel_failed)
        print(
            f"{name}: atol {abs_tol} rtol {rel_tol} failed {sel_failed.sum()} of {sel_failed.size} {sel_failed.sum() / sel_failed.size * 100:.4f}%"
        )
        print(
            f"subcube of failures [min] [max] {[f.min() for f in failed_idx]} {[f.max() for f in failed_idx]}"
        )
        print(
            f"failure range truth ({truth[sel_failed].min():.3e},{truth[sel_failed].max():.3e}) test ({test[sel_failed].min():.3e},{test[sel_failed].max():.3e})"
        )
        print(
            f"max abs diff of failures {np.fabs(truth - test)[sel_failed].max():.4e} relative {(np.fabs(truth - test) / truth)[sel_failed].max():.4e}"
        )

        failed_inp = [
            inp[sel_failed if inp.shape == test.shape else failed_idx[i]]
            for i, inp in enumerate(inputs)
        ]
        for i, _inp in enumerate(inputs):
            print(
                f"failure range of inputs axis {i} {failed_inp[i].min():.2e} {failed_inp[i].max():.2e}"
            )

        print("----- First 10 -----")
        for j in range(min(10, sel_failed.sum())):
            input_arr = [f"{failed_inp[i][j]:.2e}" for i, finp in enumerate(failed_inp)]
            print(
                f"CRD {input_arr}"
                + f"  {truth[sel_failed].flatten()[j]:.4e} {test[sel_failed].flatten()[j]:.4e}"
            )
