"""Test the C interpolation tables."""

import matplotlib as mpl
import numpy as np
import pytest
from astropy import constants as c
from astropy import units as u

from py21cmfast import AstroParams, CosmoParams, FlagOptions, UserParams
from py21cmfast.c_21cmfast import ffi, lib
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
    "PS": [10, {"HMF": "PS", "USE_MASS_DEPENDENT_ZETA": True}],
    "ST": [10, {"HMF": "ST", "USE_MASS_DEPENDENT_ZETA": True}],
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
    return np.linspace(-0.98, 1.7, num=200)


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
    return np.linspace(6, 35, num=200)


@pytest.mark.parametrize("name", options_ps)
def test_sigma_table(name, mass_range, plt):
    abs_tol = 0

    redshift, kwargs = OPTIONS_PS[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = opts["user_params"]
    cp = opts["cosmo_params"]

    sigma_tables, dsigma_tables = cf.evaluate_sigma(
        user_params=up,
        cosmo_params=cp,
        masses=mass_range,
    )

    sigma_integrals, dsigma_integrals = cf.evaluate_sigma(
        user_params=up.clone(USE_INTERPOLATION_TABLES=False),
        cosmo_params=cp,
        masses=mass_range,
    )

    if plt == mpl.pyplot:
        make_table_comparison_plot(
            [mass_range, mass_range],
            [None, None],
            [sigma_tables, dsigma_tables],
            [sigma_integrals, dsigma_integrals],
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
@pytest.mark.parametrize("from_cat", ["cat", "grid"])
def test_inverse_cmf_tables(name, from_cat, delta_range, mass_range, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    from_cat = "cat" in from_cat

    # we need the buffer for small conditions, but want the full upper range
    mmin_range = np.logspace(
        np.log10(mass_range.min() / up.SAMPLER_BUFFER_FACTOR),
        np.log10(mass_range.max()),
        num=mass_range.size,
    )

    if not from_cat:
        M_cond = (
            (
                cp.cosmo.critical_density(0)
                * cp.OMm
                * u.Mpc**3
                * (up.BOX_LEN / up.HII_DIM) ** 3
            )
            .to("M_sun")
            .value
        )
        inputs_cond, inputs_mass = np.meshgrid(delta_range, mmin_range, indexing="ij")
        z_desc = None
        inputs_delta = inputs_cond
    else:
        inputs_cond, inputs_mass = np.meshgrid(
            np.log(mass_range), mmin_range, indexing="ij"
        )
        inputs_delta = None
        z_desc = (1 + redshift) / up.ZPRIME_STEP_FACTOR - 1
        M_cond = np.exp(inputs_cond)

    # ----CELLS-----
    # Get the Integrals
    cmf_integral = cf.get_cmf_integral(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=inputs_mass,
        M_max=mmin_range.max(),
        M_cond=M_cond,
        redshift=redshift,
        delta=inputs_delta,
        z_desc=z_desc,
    ).squeeze()  # (cond, minmass)

    # Normalize by max value to get CDF
    max_p_cond = np.copy(cmf_integral[:, :1])
    max_p_cond[max_p_cond == 0] = 1.0
    cmf_integral /= max_p_cond

    # Take those probabilites to the inverse table
    icmf_table = cf.evaluate_inv_massfunc_cond(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=mmin_range.min(),
        redshift=redshift,
        cond_param=z_desc if from_cat else M_cond,
        cond_array=inputs_cond,
        probabilities=cmf_integral,
        from_catalog=from_cat,
    )

    # for plotting and printing, we use mass instead of logmass
    inputs_cond = np.exp(inputs_cond) if from_cat else inputs_cond
    if plt == mpl.pyplot:
        sel = ((inputs_cond.shape[0] - 1) * np.arange(5) / 4).astype(int)
        make_table_comparison_plot(
            [cmf_integral[sel, :].T],
            [inputs_cond[sel, 0]],
            [icmf_table[sel, :].T],
            [inputs_mass[sel, :].T],
            plt,
            zlabels=[r"$\delta =$" if not from_cat else r"$M=$"],
            logx=True,
            logy=True,
            label_test=[False, False],
            xlabels=["Probability"],
            ylabels=["Mass"],
            xlim=[np.exp(up.MIN_LOGPROB) / 10, 1.0],
            reltol=RELATIVE_TOLERANCE,
        )

    # We only want to compare at reasonable probabilities
    # easiest way to ignore is to fix values to nan
    sel_lowprob = cmf_integral < np.exp(up.MIN_LOGPROB)
    cmf_integral[sel_lowprob] = np.nan
    inputs_mass[sel_lowprob] = np.nan
    icmf_table[sel_lowprob] = np.nan

    # We don't want to include values close to delta crit, since the integrals struggle there,
    # and interpolating across the sharp gap results in errors
    # TODO: the bound should be over MAX_DELTAC_FRAC*delta_crit, and we should interpolate
    # instead of setting the integral to its limit at delta crit.
    if not from_cat:
        delta_crit = float(cf.get_delta_crit(up, cp, np.array([M_cond]), redshift))
        sel_delta = delta_range < 0.98 * delta_crit
        delta_range = delta_range[sel_delta]
        cmf_integral = cmf_integral[sel_delta, ...]
        inputs_mass = inputs_mass[sel_delta, ...]
        icmf_table = icmf_table[sel_delta, ...]
        inputs_cond = inputs_cond[sel_delta, ...]

    print_failure_stats(
        np.log(icmf_table),
        np.log(inputs_mass),  # TODO: decide whether to compare logmass or mass
        [inputs_cond, cmf_integral],
        0.0,
        RELATIVE_TOLERANCE,
        "Inverse CMF",
    )

    np.testing.assert_allclose(
        np.log(inputs_mass),
        np.log(icmf_table),
        rtol=RELATIVE_TOLERANCE,
    )


# NOTE: This test currently fails (~10% differences in mass in <1% of bins)
#   I don't want to relax the tolerance yet since it can be improved, but
#   for now this is acceptable
@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("from_cat", ["cat", "grid"])
def test_massfunc_conditional_tables(name, from_cat, mass_range, delta_range, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    M_min = mass_range.min()
    M_max = mass_range.max()
    from_cat = "cat" in from_cat

    if from_cat:
        # condition array is halo mass, parameter is descendant redshift
        cond_arr = np.log(mass_range)
        cond_param = (1 + redshift) / up.ZPRIME_STEP_FACTOR - 1
    else:
        # condition array is density, parameter is cell mass
        cond_param = (
            (
                cp.cosmo.critical_density(0)
                * cp.OMm
                * u.Mpc**3
                * (up.BOX_LEN / up.HII_DIM) ** 3
            )
            .to("M_sun")
            .value
        )
        cond_arr = delta_range

    nhalo_tbl, mcoll_tbl = cf.evaluate_massfunc_cond(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshift=redshift,
        cond_param=cond_param,
        cond_array=cond_arr,
        from_catalog=from_cat,
    )
    nhalo_exp, mcoll_exp = cf.evaluate_massfunc_cond(
        user_params=up.clone(USE_INTERPOLATION_TABLES=False),
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshift=redshift,
        cond_param=cond_param,
        cond_array=cond_arr,
        from_catalog=from_cat,
    )

    if plt == mpl.pyplot:
        cond_arr = np.exp(cond_arr) if from_cat else cond_arr
        make_table_comparison_plot(
            [cond_arr, cond_arr],
            [None, None],
            [nhalo_tbl, mcoll_tbl],
            [nhalo_exp, mcoll_exp],
            plt,
            xlabels=["delta" if from_cat else "Mass"] * 2,
            ylabels=["Nhalo", "Mcoll"],
        )

    print_failure_stats(
        nhalo_tbl,
        nhalo_exp,
        [cond_arr],
        0.0,
        RELATIVE_TOLERANCE,
        "expected N halo",
    )
    print_failure_stats(
        mcoll_tbl,
        mcoll_exp,
        [cond_arr],
        0.0,
        RELATIVE_TOLERANCE,
        "expected M halo",
    )

    np.testing.assert_allclose(nhalo_exp, nhalo_tbl, rtol=RELATIVE_TOLERANCE)
    np.testing.assert_allclose(mcoll_exp, mcoll_tbl, rtol=RELATIVE_TOLERANCE)


@pytest.mark.parametrize("R", R_PARAM_LIST)
def test_FgtrM_conditional_tables(R, delta_range, plt):
    redshift, kwargs = OPTIONS_HMF["PS"]  # always erfc
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    M_min = 1e5
    M_max = 1e20

    cond_mass = (
        (4.0 / 3.0 * np.pi * (R * u.Mpc) ** 3 * cp.cosmo.critical_density(0) * cp.OMm)
        .to("M_sun")
        .value
    )

    fcoll_tables, dfcoll_tables = cf.evaluate_FgtrM_cond(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshift=redshift,
        cond_mass=cond_mass,
        densities=delta_range,
    )
    fcoll_integrals, dfcoll_integrals = cf.evaluate_FgtrM_cond(
        user_params=up.clone(USE_INTERPOLATION_TABLES=False),
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshift=redshift,
        cond_mass=cond_mass,
        densities=delta_range,
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
    # We don't want to include values close to delta crit, since the integrals struggle there,
    # and interpolating across the sharp gap results in errors
    # TODO: the bound should be over MAX_DELTAC_FRAC*delta_crit, and we should interpolate
    # instead of setting the integral to its limit at delta crit.
    delta_crit = float(cf.get_delta_crit(up, cp, np.array([cond_mass]), redshift))
    sel_delta = np.fabs((delta_range - delta_crit) / delta_crit) > 0.02
    delta_range = delta_range[sel_delta]
    fcoll_integrals = fcoll_integrals[sel_delta, ...]
    fcoll_tables = fcoll_tables[sel_delta, ...]
    dfcoll_integrals = dfcoll_integrals[sel_delta, ...]
    dfcoll_tables = dfcoll_tables[sel_delta, ...]

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
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"].clone(
        USE_MINI_HALOS=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
    )

    M_min = 1e5
    M_max = 1e20

    SFRD_tables, SFRD_tables_mini = cf.evaluate_SFRD_z(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshifts=z_range,
        log10mturnovers=log10_mturn_range,
    )
    SFRD_integrals, SFRD_integrals_mini = cf.evaluate_SFRD_z(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshifts=z_range,
        log10mturnovers=log10_mturn_range,
        return_integral=True,
    )

    if plt == mpl.pyplot:
        xl = log10_mturn_range.size - 1
        sel_m = (xl * np.arange(5) / 4).astype(int)
        make_table_comparison_plot(
            [z_range, z_range],
            [np.array([0]), 10 ** log10_mturn_range[sel_m]],
            [SFRD_tables, SFRD_tables_mini[..., sel_m]],
            [SFRD_integrals, SFRD_integrals_mini[..., sel_m]],
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
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"].clone(
        USE_MINI_HALOS=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
    )

    M_min = 1e5
    M_max = 1e20

    Nion_tables, Nion_tables_mini = cf.evaluate_Nion_z(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshifts=z_range,
        log10mturnovers=log10_mturn_range,
    )
    Nion_integrals, Nion_integrals_mini = cf.evaluate_Nion_z(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshifts=z_range,
        log10mturnovers=log10_mturn_range,
        return_integral=True,
    )

    if plt == mpl.pyplot:
        xl = log10_mturn_range.size - 1
        sel_m = (xl * np.arange(5) / 4).astype(int)
        make_table_comparison_plot(
            [z_range, z_range],
            [np.array([0]), log10_mturn_range[sel_m]],
            [Nion_tables[:, None], Nion_tables_mini[..., sel_m]],
            [Nion_integrals[:, None], Nion_integrals_mini[..., sel_m]],
            plt,
            label_test=[True, False],
            xlabels=["redshift", "redshift"],
            ylabels=["Nion", "Nion_mini"],
        )

    abs_tol = 2e-6
    print_failure_stats(
        Nion_tables,
        Nion_integrals,
        [z_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "Nion_z",
    )
    print_failure_stats(
        Nion_tables_mini,
        Nion_integrals_mini,
        [z_range, 10**log10_mturn_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "Nion_z_mini",
    )

    np.testing.assert_allclose(
        Nion_tables, Nion_integrals, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        Nion_tables_mini, Nion_integrals_mini, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )


# A few notes on this test function:
#   Mass limits are set explicitly using the param values
#       and #defines are copied to hard code (not ideal)
#   Density and Mturn limits are set to their maxima since we don't have cubes.
#       Hence this is a worst case scenario
#   While the EvaluateX() functions are useful in the main code to be agnostic to USE_INTERPOLATION_TABLES
#       I do not use them here fully, instead calling the integrals directly to avoid parameter changes
#       Mostly since if we set user_params.USE_INTERPOLATION_TABLES=False then the sigma tables aren't used
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
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"].clone(
        INTEGRATION_METHOD_ATOMIC=OPTIONS_INTMETHOD[intmethod],
        INTEGRATION_METHOD_MINI=OPTIONS_INTMETHOD[intmethod],
    )
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"].clone(
        USE_MINI_HALOS=mini_flag,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
    )

    M_min = 1e5
    M_max = 1e20

    cond_mass = (
        (4.0 / 3.0 * np.pi * (R * u.Mpc) ** 3 * cp.cosmo.critical_density(0) * cp.OMm)
        .to("M_sun")
        .value
    )

    Nion_tables, Nion_tables_mini = cf.evaluate_Nion_cond(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshift=redshift,
        cond_mass=cond_mass,
        densities=delta_range,
        l10mturns=log10_mturn_range,
    )

    Nion_integrals, Nion_integrals_mini = cf.evaluate_Nion_cond(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshift=redshift,
        cond_mass=cond_mass,
        densities=delta_range,
        l10mturns=log10_mturn_range,
        return_integral=True,
    )

    # The bilinear interpolation we use underperforms at high mturn due to the sharp
    # dropoff. Setting an absolute tolerance to a level we care about for reionisation
    # rather than a level we expect from interp tables remedies this for now.
    # TODO: In future we should investigate cubic splines etc.
    abs_tol = 1e-8

    if plt == mpl.pyplot:
        if mini_flag:
            xl = log10_mturn_range.size - 1
            sel_m = (xl * np.arange(5) / 4).astype(int)
            Nion_tb_plot = Nion_tables[..., sel_m]
            Nion_il_plot = Nion_integrals[..., sel_m]
        else:
            Nion_tb_plot = Nion_tables[:, None]
            Nion_il_plot = Nion_integrals[:, None]
            sel_m = np.array([0]).astype(int)

        make_table_comparison_plot(
            [delta_range, delta_range],
            [10 ** log10_mturn_range[sel_m], 10 ** log10_mturn_range[sel_m]],
            [Nion_tb_plot, Nion_tables_mini[..., sel_m]],
            [Nion_il_plot, Nion_integrals_mini[..., sel_m]],
            plt,
            label_test=[True, False],
            xlabels=["delta", "delta"],
            ylabels=["Nion", "Nion_mini"],
        )

    # We don't want to include values close to delta crit, since the integrals struggle there,
    # and interpolating across the sharp gap results in errors
    # TODO: the bound should be over MAX_DELTAC_FRAC*delta_crit, and we should interpolate
    # instead of setting the integral to its limit at delta crit.
    delta_crit = float(cf.get_delta_crit(up, cp, np.array([cond_mass]), redshift))
    sel_delta = np.fabs((delta_range - delta_crit) / delta_crit) > 0.02
    delta_range = delta_range[sel_delta]
    Nion_integrals = Nion_integrals[sel_delta, ...]
    Nion_tables = Nion_tables[sel_delta, ...]
    Nion_integrals_mini = Nion_integrals_mini[sel_delta, ...]
    Nion_tables_mini = Nion_tables_mini[sel_delta, ...]

    print_failure_stats(
        Nion_tables,
        Nion_integrals,
        [delta_range, 10**log10_mturn_range] if mini_flag else [delta_range],
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

    np.testing.assert_allclose(
        Nion_tables, Nion_integrals, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )

    np.testing.assert_allclose(
        Nion_tables_mini, Nion_integrals_mini, atol=abs_tol, rtol=RELATIVE_TOLERANCE
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
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"].clone(
        INTEGRATION_METHOD_ATOMIC=OPTIONS_INTMETHOD[intmethod],
        INTEGRATION_METHOD_MINI=OPTIONS_INTMETHOD[intmethod],
    )
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"].clone(
        USE_MINI_HALOS=mini_flag,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
    )

    M_min = 1e5
    M_max = 1e20

    cond_mass = (
        (4.0 / 3.0 * np.pi * (R * u.Mpc) ** 3 * cp.cosmo.critical_density(0) * cp.OMm)
        .to("M_sun")
        .value
    )

    Xray_tables = cf.evaluate_Xray_cond(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshift=redshift,
        cond_mass=cond_mass,
        densities=delta_range,
        l10mturns=log10_mturn_range,
    )

    Xray_integrals = cf.evaluate_Xray_cond(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshift=redshift,
        cond_mass=cond_mass,
        densities=delta_range,
        l10mturns=log10_mturn_range,
        return_integral=True,
    )

    # The bilinear interpolation we use underperforms at high mturn due to the sharp
    # dropoff. Setting an absolute tolerance to a level we care about for reionisation
    # rather than a level we expect from interp tables remedies this for now.
    # TODO: In future we should investigate cubic splines etc.
    abs_tol = 0.0

    if plt == mpl.pyplot:
        if mini_flag:
            xl = log10_mturn_range.size - 1
            sel_m = (xl * np.arange(5) / 4).astype(int)
            Xray_tb_plot = Xray_tables[..., sel_m]
            Xray_il_plot = Xray_integrals[..., sel_m]
        else:
            Xray_tb_plot = Xray_tables[:, None]
            Xray_il_plot = Xray_integrals[:, None]
            sel_m = np.array([0]).astype(int)
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

    # We don't want to include values close to delta crit, since the integrals struggle there,
    # and interpolating across the sharp gap results in errors
    # TODO: the bound should be over MAX_DELTAC_FRAC*delta_crit, and we should interpolate
    # instead of setting the integral to its limit at delta crit.
    delta_crit = float(cf.get_delta_crit(up, cp, np.array([cond_mass]), redshift))
    sel_delta = np.fabs((delta_range - delta_crit) / delta_crit) > 0.02
    delta_range = delta_range[sel_delta]
    Xray_integrals = Xray_integrals[sel_delta, ...]
    Xray_tables = Xray_tables[sel_delta, ...]
    print_failure_stats(
        Xray_tables,
        Xray_integrals,
        [delta_range, 10**log10_mturn_range] if mini_flag else [delta_range],
        abs_tol,
        RELATIVE_TOLERANCE,
        "Xray_c",
    )

    np.testing.assert_allclose(
        Xray_tables, Xray_integrals, atol=abs_tol, rtol=RELATIVE_TOLERANCE
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
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"].clone(
        INTEGRATION_METHOD_ATOMIC=OPTIONS_INTMETHOD[intmethod],
        INTEGRATION_METHOD_MINI=OPTIONS_INTMETHOD[intmethod],
    )
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"].clone(
        USE_MINI_HALOS=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
    )

    M_min = 1e5
    M_max = 1e20

    cond_mass = (
        (4.0 / 3.0 * np.pi * (R * u.Mpc) ** 3 * cp.cosmo.critical_density(0) * cp.OMm)
        .to("M_sun")
        .value
    )

    SFRD_tables, SFRD_tables_mini = cf.evaluate_SFRD_cond(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshift=redshift,
        cond_mass=cond_mass,
        densities=delta_range,
        l10mturns=log10_mturn_range,
    )

    SFRD_integrals, SFRD_integrals_mini = cf.evaluate_SFRD_cond(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=M_min,
        M_max=M_max,
        redshift=redshift,
        cond_mass=cond_mass,
        densities=delta_range,
        l10mturns=log10_mturn_range,
        return_integral=True,
    )

    # The bilinear interpolation we use underperforms at high mturn due to the sharp
    # dropoff. Setting an absolute tolerance to a level we care about for reionisation
    # rather than a level we expect from interp tables remedies this for now.
    # TODO: In future we should investigate cubic splines etc.
    abs_tol = 1e-8
    if plt == mpl.pyplot:
        xl = log10_mturn_range.size - 1
        sel_m = (xl * np.arange(5) / 4).astype(int)
        make_table_comparison_plot(
            [delta_range, delta_range],
            [np.array([0]), 10 ** log10_mturn_range[sel_m]],
            [SFRD_tables[:, None], SFRD_tables_mini[..., sel_m]],
            [SFRD_integrals[:, None], SFRD_integrals_mini[..., sel_m]],
            plt,
            label_test=[True, False],
            xlabels=["delta", "delta"],
            ylabels=["SFRD", "SFRD_mini"],
        )

    # We don't want to include values close to delta crit, since the integrals struggle there,
    # and interpolating across the sharp gap results in errors
    # TODO: the bound should be over MAX_DELTAC_FRAC*delta_crit, and we should interpolate
    # instead of setting the integral to its limit at delta crit.
    delta_crit = float(cf.get_delta_crit(up, cp, np.array([cond_mass]), redshift))
    sel_delta = np.fabs((delta_range - delta_crit) / delta_crit) > 0.02
    delta_range = delta_range[sel_delta]
    SFRD_integrals = SFRD_integrals[sel_delta, ...]
    SFRD_tables = SFRD_tables[sel_delta, ...]
    SFRD_integrals_mini = SFRD_integrals_mini[sel_delta, ...]
    SFRD_tables_mini = SFRD_tables_mini[sel_delta, ...]

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

    np.testing.assert_allclose(
        SFRD_tables, SFRD_integrals, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        SFRD_tables_mini, SFRD_integrals_mini, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )


INTEGRAND_OPTIONS = ["sfrd", "n_ion"]


@pytest.mark.parametrize("R", R_PARAM_LIST)
@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("integrand", INTEGRAND_OPTIONS)
def test_conditional_integral_methods(
    R, log10_mturn_range, delta_range, name, integrand, plt
):
    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"].clone(
        USE_MINI_HALOS=True,
        USE_MASS_DEPENDENT_ZETA=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
    )

    intgrl_func = cf.evaluate_SFRD_cond if "sfr" in integrand else cf.evaluate_Nion_cond

    M_min = 1e5
    M_max = 1e20
    cond_mass = (
        (4.0 / 3.0 * np.pi * (R * u.Mpc) ** 3 * cp.cosmo.critical_density(0) * cp.OMm)
        .to("M_sun")
        .value
    )

    integrals = []
    integrals_mini = []
    for method in ["GSL-QAG", "GAUSS-LEGENDRE", "GAMMA-APPROX"]:
        print(f"Starting method {method}", flush=True)
        if name != "PS" and method == "GAMMA-APPROX":
            continue

        up = up.clone(
            INTEGRATION_METHOD_ATOMIC=method,
            INTEGRATION_METHOD_MINI=method,
        )

        buf, buf_mini = intgrl_func(
            user_params=up,
            cosmo_params=cp,
            astro_params=ap,
            flag_options=fo,
            M_min=M_min,
            M_max=M_max,
            redshift=redshift,
            cond_mass=cond_mass,
            densities=delta_range,
            l10mturns=log10_mturn_range,
            return_integral=True,
        )
        integrals.append(buf)
        integrals_mini.append(buf_mini)

    abs_tol = 1e-6  # minimum = exp(-40) ~1e-18
    if plt == mpl.pyplot:
        xl = log10_mturn_range.size - 1
        sel_m = (xl * np.arange(5) / 4).astype(int)
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
        zip(integral_list, integral_list_second, strict=False)
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
