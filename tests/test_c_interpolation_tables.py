import pytest

import attrs
import matplotlib as mpl
import numpy as np
from astropy import constants as c
from astropy import units as u

from py21cmfast import AstroParams, CosmoParams, FlagOptions, UserParams, global_params
from py21cmfast.c_21cmfast import ffi, lib

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
    "PS": [10, {"HMF": "PS"}],
    "ST": [10, {"HMF": "ST"}],
    # "Watson": [10, {"HMF": "WATSON"}],
    # "Watsonz": [10, {"HMF": "WATSON-Z"}],
    # "Delos": [10, {"HMF": "DELOS"}],
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
# options_intmethod[2] = pytest.param("FFCOLL", marks=pytest.mark.xfail)


# TODO: write tests for the redshift interpolation tables (global Nion, SFRD, FgtrM)
@pytest.mark.parametrize("name", options_ps)
def test_sigma_table(name, plt):
    abs_tol = 0

    redshift, kwargs = OPTIONS_PS[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = opts["user_params"]
    cp = opts["cosmo_params"]
    lib.Broadcast_struct_global_noastro(up.cstruct, cp.cstruct)

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    mass_range = np.logspace(7, 14, num=100)

    sigma_ref = np.vectorize(lib.sigma_z0)(mass_range)
    dsigmasq_ref = np.vectorize(lib.dsigmasqdm_z0)(mass_range)

    sigma_table = np.vectorize(lib.EvaluateSigma)(np.log(mass_range))
    dsigmasq_table = np.vectorize(lib.EvaluatedSigmasqdm)(np.log(mass_range))

    if plt == mpl.pyplot:
        make_table_comparison_plot(
            [mass_range, mass_range],
            [np.array([0]), np.array([0])],
            [sigma_table[:, None], dsigmasq_table[:, None]],
            [sigma_ref[:, None], dsigmasq_ref[:, None]],
            plt,
        )

    np.testing.assert_allclose(
        sigma_ref, sigma_table, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        dsigmasq_ref, dsigmasq_table, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("name", options_hmf)
def test_inverse_cmf_tables(name, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    hist_size = 1000
    edges = np.logspace(7, 12, num=hist_size).astype("f4")
    edges_ln = np.log(edges)

    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    growth_out = lib.dicke(redshift)
    growth_in = lib.dicke(redshift / global_params.ZPRIME_STEP_FACTOR)
    cell_mass = (
        (
            cp.cosmo.critical_density(0)
            * cp.OMm
            * u.Mpc**3
            * (up.BOX_LEN / up.HII_DIM) ** 3
        )
        .to("M_sun")
        .value
    )

    sigma_cond_cell = lib.sigma_z0(cell_mass)
    sigma_cond_halo = np.vectorize(lib.sigma_z0)(edges)
    delta_crit = lib.get_delta_crit(up.cdict["HMF"], sigma_cond_cell, growth_in)
    delta_update = (
        np.vectorize(lib.get_delta_crit)(up.cdict["HMF"], sigma_cond_halo, growth_in)
        * growth_out
        / growth_in
    )
    edges_d = np.linspace(-1, delta_crit * 1.1, num=hist_size).astype("f4")

    # Cell Integrals
    arg_list_inv_d = np.meshgrid(edges_d[:-1], edges_ln[:-1], indexing="ij")
    N_cmfi_cell = np.vectorize(lib.Nhalo_Conditional)(
        growth_out,
        arg_list_inv_d[1],  # lnM
        edges_ln[-1],  # integrate to max mass
        cell_mass,
        sigma_cond_cell,
        arg_list_inv_d[0],  # density
        0,
    )

    max_in_d = N_cmfi_cell[:, :1]
    max_in_d[edges_d[:-1] == -1, :] = 1.0  # fix delta=-1 where all entries are zero
    N_cmfi_cell = (
        N_cmfi_cell / max_in_d
    )  # to get P(>M) since the y-axis is the lower integral limit

    lib.initialise_dNdM_inverse_table(
        edges_d[0],
        edges_d[-1],
        edges_ln[0],
        growth_out,
        np.log(cell_mass),
        False,
    )

    N_inverse_cell = (
        np.vectorize(lib.EvaluateNhaloInv)(arg_list_inv_d[0], N_cmfi_cell) * cell_mass
    )  # Mass evaluated at the probabilities given by the integral

    # Halo Integrals
    arg_list_inv_m = np.meshgrid(edges_ln[:-1], edges_ln[:-1], indexing="ij")
    N_cmfi_halo = np.vectorize(lib.Nhalo_Conditional)(
        growth_out,
        arg_list_inv_m[1],
        edges_ln[-1],
        edges[:-1, None],
        sigma_cond_halo[:-1, None],  # (condition,masslimit)
        delta_update[:-1, None],
        0,
    )

    # To get P(>M), NOTE that some conditions have no integral
    N_cmfi_halo = N_cmfi_halo / (
        N_cmfi_halo[:, :1] + np.all(N_cmfi_halo == 0, axis=1)[:, None]
    )  # if all entries are zero, do not nan the row, just divide by 1

    lib.initialise_dNdM_inverse_table(
        edges_ln[0],
        edges_ln[-1],
        edges_ln[0],
        growth_out,
        growth_in,
        True,
    )

    N_inverse_halo = (
        np.vectorize(lib.EvaluateNhaloInv)(arg_list_inv_m[0], N_cmfi_halo)
        * edges[:-1, None]
    )  # LOG MASS, evaluated at the probabilities given by the integral

    # NOTE: The tables get inaccurate in the smallest halo bin where the condition mass approaches the minimum
    #       We set the absolute tolerance to be insiginificant in sampler terms (~1% of the smallest halo)
    abs_tol_halo = 1e-2

    if plt == mpl.pyplot:
        xl = edges_d[:-1].size
        sel = (xl * np.arange(6) / 6).astype(int)
        massfunc_table_comparison_plot(
            edges[:-1],
            edges[sel],
            N_cmfi_halo[sel, :],
            N_inverse_halo[sel, :],
            edges_d[sel],
            N_cmfi_cell[sel, :],
            N_inverse_cell[sel, :],
            plt,
        )

    mask_halo_compare = arg_list_inv_m[1] < arg_list_inv_m[0]  # condtition > halo
    mask_cell_compare = arg_list_inv_d[0] < delta_crit  # delta < delta_crit

    N_inverse_halo[mask_halo_compare] = np.exp(arg_list_inv_m[1][mask_halo_compare])
    N_inverse_cell[mask_cell_compare] = np.exp(arg_list_inv_d[1][mask_cell_compare])

    print_failure_stats(
        N_inverse_halo,
        np.exp(arg_list_inv_m[1]),
        arg_list_inv_m,
        0.0,
        RELATIVE_TOLERANCE,
        "Inverse Halo",
    )
    print_failure_stats(
        N_inverse_cell,
        np.exp(arg_list_inv_d[1]),
        arg_list_inv_d,
        0.0,
        RELATIVE_TOLERANCE,
        "Inverse Cell",
    )

    np.testing.assert_allclose(
        np.exp(arg_list_inv_d[1]),
        N_inverse_cell,
        atol=edges[0] * abs_tol_halo,
        rtol=RELATIVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        np.exp(arg_list_inv_m[1]),
        N_inverse_halo,
        atol=edges[0] * abs_tol_halo,
        rtol=RELATIVE_TOLERANCE,
    )


# NOTE: This test currently fails (~10% differences in mass in <1% of bins)
#   I don't want to relax the tolerance yet since it can be improved, but
#   for now this is acceptable
# @pytest.mark.xfail
@pytest.mark.parametrize("name", options_hmf)
def test_Massfunc_conditional_tables(name, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]
    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    hist_size = 1000
    edges = np.logspace(7, 12, num=hist_size).astype("f4")
    edges_ln = np.log(edges)

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    growth_out = lib.dicke(redshift)
    growth_in = lib.dicke(redshift / global_params.ZPRIME_STEP_FACTOR)
    cell_mass = (
        (
            cp.cosmo.critical_density(0)
            * cp.OMm
            * u.Mpc**3
            * (up.BOX_LEN / up.HII_DIM) ** 3
        )
        .to("M_sun")
        .value
    )

    sigma_cond_cell = lib.sigma_z0(cell_mass)
    sigma_cond_halo = np.vectorize(lib.sigma_z0)(edges)
    delta_crit = lib.get_delta_crit(up.cdict["HMF"], sigma_cond_cell, growth_in)
    delta_update = (
        np.vectorize(lib.get_delta_crit)(up.cdict["HMF"], sigma_cond_halo, growth_in)
        * growth_out
        / growth_in
    )
    edges_d = np.linspace(-1, delta_crit * 1.1, num=hist_size).astype("f4")

    M_cmf_cell = (
        np.vectorize(lib.Mcoll_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            cell_mass,
            sigma_cond_cell,
            edges_d[:-1],
            0,
        )
        * cell_mass
    )
    N_cmf_cell = (
        np.vectorize(lib.Nhalo_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            cell_mass,
            sigma_cond_cell,
            edges_d[:-1],
            0,
        )
        * cell_mass
    )

    # Cell Tables
    lib.initialise_dNdM_tables(
        edges_d[0],
        edges_d[-1],
        edges_ln[0],
        edges_ln[-1],
        growth_out,
        np.log(cell_mass),
        False,
    )

    M_exp_cell = (
        np.vectorize(lib.EvaluateMcoll)(edges_d[:-1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        * cell_mass
    )
    N_exp_cell = (
        np.vectorize(lib.EvaluateNhalo)(edges_d[:-1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        * cell_mass
    )

    M_cmf_halo = (
        np.vectorize(lib.Mcoll_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            edges[:-1],
            sigma_cond_halo[:-1],
            delta_update[:-1],
            0,
        )
        * edges[:-1]
    )
    N_cmf_halo = (
        np.vectorize(lib.Nhalo_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            edges[:-1],
            sigma_cond_halo[:-1],
            delta_update[:-1],
            0,
        )
        * edges[:-1]
    )

    # Halo Tables
    lib.initialise_dNdM_tables(
        edges_ln[0],
        edges_ln[-1],
        edges_ln[0],
        edges_ln[-1],
        growth_out,
        growth_in,
        True,
    )
    M_exp_halo = (
        np.vectorize(lib.EvaluateMcoll)(edges_ln[:-1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        * edges[:-1]
    )
    N_exp_halo = (
        np.vectorize(lib.EvaluateNhalo)(edges_ln[:-1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        * edges[:-1]
    )

    # NOTE: The tables get inaccurate in the smallest halo bin where the condition mass approaches the minimum
    #       We set the absolute tolerance to be insiginificant in sampler terms (~1% of the smallest halo)
    abs_tol_halo = 1e-2

    if plt == mpl.pyplot:
        make_table_comparison_plot(
            [edges[:-1], edges_d[:-1], edges[:-1], edges_d[:-1]],
            [np.array([0]), np.array([0]), np.array([0]), np.array([0])],
            [
                N_exp_halo[:, None],
                N_exp_cell[:, None],
                M_exp_halo[:, None],
                M_exp_cell[:, None],
            ],
            [
                N_cmf_halo[:, None],
                N_cmf_cell[:, None],
                M_cmf_halo[:, None],
                M_cmf_cell[:, None],
            ],
            plt,
        )

    print_failure_stats(
        N_cmf_halo,
        N_exp_halo,
        [edges[:-1]],
        abs_tol_halo,
        RELATIVE_TOLERANCE,
        "expected N halo",
    )
    print_failure_stats(
        M_cmf_halo,
        M_exp_halo,
        [edges[:-1]],
        abs_tol_halo,
        RELATIVE_TOLERANCE,
        "expected M halo",
    )

    print_failure_stats(
        N_cmf_cell,
        N_exp_cell,
        [edges_d[:-1]],
        abs_tol_halo,
        RELATIVE_TOLERANCE,
        "expected N cell",
    )
    print_failure_stats(
        M_cmf_cell,
        M_exp_cell,
        [edges_d[:-1]],
        abs_tol_halo,
        RELATIVE_TOLERANCE,
        "expected M cell",
    )

    np.testing.assert_allclose(
        N_cmf_halo, N_exp_halo, atol=abs_tol_halo, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        N_cmf_cell, N_exp_cell, atol=abs_tol_halo, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        M_cmf_halo, M_exp_halo, atol=edges[0] * abs_tol_halo, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        M_cmf_cell, M_exp_cell, atol=edges[0] * abs_tol_halo, rtol=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("R", R_PARAM_LIST)
@pytest.mark.parametrize("name", options_hmf)
def test_FgtrM_conditional_tables(name, R, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]
    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(M_min, M_max)

    growth_out = lib.dicke(redshift)
    sigma_min = lib.sigma_z0(M_min)

    cond_mass = (
        (4.0 / 3.0 * np.pi * (R * u.Mpc) ** 3 * cp.cosmo.critical_density(0) * cp.OMm)
        .to("M_sun")
        .value
    )
    sigma_cond = lib.sigma_z0(cond_mass)
    delta_crit = lib.get_delta_crit(up.cdict["HMF"], sigma_cond, growth_out)

    edges_d = np.linspace(-1, delta_crit * 1.1, num=hist_size).astype(
        "f4"
    )  # EPS is forced with FgtrM due to the erfc functions

    # NOTE: Rather than keeping zp constant we keep zpp constant
    lib.initialise_FgtrM_delta_table(
        edges_d[0], edges_d[-1], redshift, growth_out, sigma_min, sigma_cond
    )

    fcoll_tables = np.vectorize(lib.EvaluateFcoll_delta)(
        edges_d[:-1], growth_out, sigma_min, sigma_cond
    )
    dfcoll_tables = np.vectorize(lib.EvaluatedFcolldz)(
        edges_d[:-1], redshift, sigma_min, sigma_cond
    )

    up = attrs.evolve(up, USE_INTERPOLATION_TABLES=False)
    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    fcoll_integrals = np.vectorize(lib.EvaluateFcoll_delta)(
        edges_d[:-1], growth_out, sigma_min, sigma_cond
    )
    dfcoll_integrals = np.vectorize(lib.EvaluatedFcolldz)(
        edges_d[:-1], redshift, sigma_min, sigma_cond
    )

    if plt == mpl.pyplot:
        make_table_comparison_plot(
            [edges_d[:-1], edges_d[:-1]],
            [np.array([0]), np.array([0])],
            [fcoll_tables[:, None], dfcoll_tables[:, None]],
            [fcoll_integrals[:, None], dfcoll_integrals[:, None]],
            plt,
        )

    abs_tol = 0.0
    print_failure_stats(
        fcoll_tables,
        fcoll_integrals,
        [
            edges_d[:-1],
        ],
        abs_tol,
        RELATIVE_TOLERANCE,
        "fcoll",
    )

    print_failure_stats(
        dfcoll_tables,
        fcoll_integrals,
        [
            edges_d[:-1],
        ],
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
def test_SFRD_z_tables(name, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    fo = attrs.evolve(
        fo,
        USE_MINI_HALOS=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
    )
    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL
    z_array = np.linspace(6, 35, num=hist_size)
    edges_m = np.logspace(5, 8, num=int(hist_size / 10)).astype("f4")
    f10s = 10**ap.F_STAR10
    f7s = 10**ap.F_STAR7_MINI

    lib.init_ps()

    if up.INTEGRATION_METHOD_ATOMIC == 1 or up.INTEGRATION_METHOD_MINI == 1:
        lib.initialise_GL(np.log(M_min), np.log(M_max))

    Mlim_Fstar = 1e10 * (10**ap.F_STAR10) ** (-1.0 / ap.ALPHA_STAR)
    Mlim_Fstar_MINI = 1e7 * (10**ap.F_STAR7_MINI) ** (-1.0 / ap.ALPHA_STAR_MINI)

    lib.initialiseSigmaMInterpTable(M_min, M_max)

    lib.initialise_SFRD_spline(
        400,
        z_array[0],
        z_array[-1],
        ap.ALPHA_STAR,
        ap.ALPHA_STAR_MINI,
        f10s,
        f7s,
        ap.M_TURN,
        True,
    )

    # for the atomic cooling threshold
    # omz = cp.cosmo.Om(z_array)
    # d_nl = 18*np.pi*np.pi + 82*(omz-1) - 39*(omz-1)**2
    # M_turn_a = 7030.97 / (cp.hlittle) * np.sqrt(omz / (cp.OMm*d_nl)) * (1e4/(0.59*(1+z_array)))**1.5
    M_turn_a = np.vectorize(lib.atomic_cooling_threshold)(z_array)

    input_arr = np.meshgrid(z_array[:-1], np.log10(edges_m[:-1]), indexing="ij")

    SFRD_tables = np.vectorize(lib.EvaluateSFRD)(z_array[:-1], Mlim_Fstar)
    SFRD_tables_mini = np.vectorize(lib.EvaluateSFRD_MINI)(
        input_arr[0], input_arr[1], Mlim_Fstar_MINI
    )

    SFRD_integrals = np.vectorize(lib.Nion_General)(
        z_array[:-1],
        np.log(M_min),
        np.log(M_max),
        M_turn_a[:-1],
        ap.ALPHA_STAR,
        0.0,
        f10s,
        1.0,
        Mlim_Fstar,
        0.0,
    )
    SFRD_integrals_mini = np.vectorize(lib.Nion_General_MINI)(
        input_arr[0],
        np.log(M_min),
        np.log(M_max),
        10 ** input_arr[1],
        M_turn_a[:-1][:, None],
        ap.ALPHA_STAR_MINI,
        0.0,
        f7s,
        1.0,
        Mlim_Fstar_MINI,
        0.0,
    )

    if plt == mpl.pyplot:
        xl = input_arr[1].shape[1]
        sel_m = (xl * np.arange(6) / 6).astype(int)
        make_table_comparison_plot(
            [z_array[:-1], z_array[:-1]],
            [np.array([0]), edges_m[sel_m]],
            [SFRD_tables[:, None], SFRD_tables_mini[..., sel_m]],
            [SFRD_integrals[:, None], SFRD_integrals_mini[..., sel_m]],
            plt,
        )

    abs_tol = 1e-7
    print_failure_stats(
        SFRD_tables,
        SFRD_integrals,
        [
            z_array[:-1],
        ],
        abs_tol,
        RELATIVE_TOLERANCE,
        "SFRD_z",
    )
    print_failure_stats(
        SFRD_tables_mini,
        SFRD_integrals_mini,
        input_arr,
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
def test_Nion_z_tables(name, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    fo = attrs.evolve(
        fo,
        USE_MINI_HALOS=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
    )
    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    f10s = 10**ap.F_STAR10
    f7s = 10**ap.F_STAR7_MINI
    f10e = 10**ap.F_ESC10
    f7e = 10**ap.F_ESC7_MINI

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL
    z_array = np.linspace(6, 40, num=hist_size)
    edges_m = np.logspace(5, 8, num=int(hist_size / 10)).astype("f4")

    lib.init_ps()

    if up.INTEGRATION_METHOD_ATOMIC == 1 or up.INTEGRATION_METHOD_MINI == 1:
        lib.initialise_GL(np.log(M_min), np.log(M_max))

    Mlim_Fstar = 1e10 * (10**ap.F_STAR10) ** (-1.0 / ap.ALPHA_STAR)
    Mlim_Fesc = 1e10 * (10**ap.F_ESC10) ** (-1.0 / ap.ALPHA_ESC)
    Mlim_Fstar_MINI = 1e7 * (10**ap.F_STAR7_MINI) ** (-1.0 / ap.ALPHA_STAR_MINI)
    Mlim_Fesc_MINI = 1e7 * (10**ap.F_ESC7_MINI) ** (-1.0 / ap.ALPHA_ESC)

    lib.initialiseSigmaMInterpTable(M_min, M_max)

    lib.initialise_Nion_Ts_spline(
        400,
        z_array[0],
        z_array[-1],
        ap.ALPHA_STAR,
        ap.ALPHA_STAR_MINI,
        ap.ALPHA_ESC,
        f10s,
        f10e,
        f7s,
        f7e,
        ap.M_TURN,
        True,
    )

    # for the atomic cooling threshold
    # omz = cp.cosmo.Om(z_array)
    # d_nl = 18*np.pi*np.pi + 82*(omz-1) - 39*(omz-1)**2
    # M_turn_a = 7030.97 / (cp.hlittle) * np.sqrt(omz / (cp.OMm*d_nl)) * (1e4/(0.59*(1+z_array)))**1.5
    M_turn_a = np.vectorize(lib.atomic_cooling_threshold)(z_array)

    input_arr = np.meshgrid(z_array[:-1], np.log10(edges_m[:-1]), indexing="ij")

    Nion_tables = np.vectorize(lib.EvaluateNionTs)(z_array[:-1], Mlim_Fstar, Mlim_Fesc)
    Nion_tables_mini = np.vectorize(lib.EvaluateNionTs_MINI)(
        input_arr[0], input_arr[1], Mlim_Fstar_MINI, Mlim_Fesc_MINI
    )

    Nion_integrals = np.vectorize(lib.Nion_General)(
        z_array[:-1],
        np.log(M_min),
        np.log(M_max),
        M_turn_a[:-1],
        ap.ALPHA_STAR,
        ap.ALPHA_ESC,
        f10s,
        f10e,
        Mlim_Fstar,
        Mlim_Fesc,
    )
    Nion_integrals_mini = np.vectorize(lib.Nion_General_MINI)(
        input_arr[0],
        np.log(M_min),
        np.log(M_max),
        10 ** input_arr[1],
        M_turn_a[:-1][:, None],
        ap.ALPHA_STAR_MINI,
        ap.ALPHA_ESC,
        f7s,
        f7e,
        Mlim_Fstar_MINI,
        Mlim_Fesc_MINI,
    )

    if plt == mpl.pyplot:
        xl = input_arr[1].shape[1]
        sel_m = (xl * np.arange(6) / 6).astype(int)
        make_table_comparison_plot(
            [z_array[:-1], z_array[:-1]],
            [np.array([0]), edges_m[sel_m]],
            [Nion_tables[:, None], Nion_tables_mini[..., sel_m]],
            [Nion_integrals[:, None], Nion_integrals_mini[..., sel_m]],
            plt,
        )

    abs_tol = 5e-6
    print_failure_stats(
        Nion_tables,
        Nion_integrals,
        [
            z_array[:-1],
        ],
        abs_tol,
        RELATIVE_TOLERANCE,
        "Nion_z",
    )
    print_failure_stats(
        Nion_tables_mini,
        Nion_integrals_mini,
        input_arr,
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
def test_Nion_conditional_tables(name, R, mini, intmethod, plt):
    if name != "PS" and intmethod == "FFCOLL":
        pytest.skip("FAST FFCOLL INTEGRALS WORK ONLY WITH EPS")

    mini_flag = mini == "mini"

    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    up = attrs.evolve(
        up,
        INTEGRATION_METHOD_ATOMIC=OPTIONS_INTMETHOD[intmethod],
        INTEGRATION_METHOD_MINI=OPTIONS_INTMETHOD[intmethod],
    )
    fo = attrs.evolve(
        fo,
        USE_MINI_HALOS=mini_flag,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
    )
    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL

    lib.init_ps()

    if "GAUSS-LEGENDRE" in (up.INTEGRATION_METHOD_ATOMIC, up.INTEGRATION_METHOD_MINI):
        lib.initialise_GL(np.log(M_min), np.log(M_max))

    growth_out = lib.dicke(redshift)
    cond_mass = (
        (4.0 / 3.0 * np.pi * (R * u.Mpc) ** 3 * cp.cosmo.critical_density(0) * cp.OMm)
        .to("M_sun")
        .value
    )
    sigma_cond = lib.sigma_z0(cond_mass)
    delta_crit = lib.get_delta_crit(up.cdict["HMF"], sigma_cond, growth_out)

    edges_d = np.linspace(-1, delta_crit * 1.1, num=hist_size).astype("f4")
    edges_m = np.logspace(5, 10, num=int(hist_size / 10)).astype("f4")

    Mlim_Fstar = 1e10 * (10**ap.F_STAR10) ** (-1.0 / ap.ALPHA_STAR)
    Mlim_Fesc = 1e10 * (10**ap.F_ESC10) ** (-1.0 / ap.ALPHA_ESC)
    Mlim_Fstar_MINI = 1e7 * (10**ap.F_STAR7_MINI) ** (-1.0 / ap.ALPHA_STAR_MINI)
    Mlim_Fesc_MINI = 1e7 * (10**ap.F_ESC7_MINI) ** (-1.0 / ap.ALPHA_ESC)

    lib.initialiseSigmaMInterpTable(M_min, max(cond_mass, M_max))

    lib.initialise_Nion_Conditional_spline(
        redshift,
        10**ap.M_TURN,  # not the redshift dependent version in this test
        edges_d[0],
        edges_d[-1],
        M_min,
        M_max,
        cond_mass,
        np.log10(edges_m[0]),
        np.log10(edges_m[-1]),
        np.log10(edges_m[0]),
        np.log10(edges_m[-1]),
        ap.ALPHA_STAR,
        ap.ALPHA_STAR_MINI,
        ap.ALPHA_ESC,
        10**ap.F_STAR10,
        10**ap.F_ESC10,
        Mlim_Fstar,
        Mlim_Fesc,
        10**ap.F_STAR7_MINI,
        10**ap.F_ESC7_MINI,
        Mlim_Fstar_MINI,
        Mlim_Fesc_MINI,
        up.cdict["INTEGRATION_METHOD_ATOMIC"],
        up.cdict["INTEGRATION_METHOD_MINI"],
        mini_flag,
        False,
    )

    if mini_flag:
        input_arr = np.meshgrid(edges_d[:-1], np.log10(edges_m[:-1]), indexing="ij")
    else:
        input_arr = [
            edges_d[:-1],
            np.full_like(edges_d[:-1], ap.M_TURN),
        ]  # mturn already in log10

    Nion_tables = np.vectorize(lib.EvaluateNion_Conditional)(
        input_arr[0], input_arr[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
    )

    Nion_integrals = np.vectorize(lib.Nion_ConditionalM)(
        growth_out,
        np.log(M_min),
        np.log(M_max),
        cond_mass,
        sigma_cond,
        input_arr[0],
        10 ** input_arr[1],
        ap.ALPHA_STAR,
        ap.ALPHA_ESC,
        10**ap.F_STAR10,
        10**ap.F_ESC10,
        Mlim_Fstar,
        Mlim_Fesc,
        up.cdict["INTEGRATION_METHOD_ATOMIC"],
    )

    #### FIRST ASSERT ####
    abs_tol = 5e-18  # min = exp(-40) ~4e-18
    print_failure_stats(
        Nion_tables,
        Nion_integrals,
        input_arr if mini_flag else input_arr[:1],
        abs_tol,
        RELATIVE_TOLERANCE,
        "Nion_c",
    )

    if mini_flag:
        Nion_tables_mini = np.vectorize(lib.EvaluateNion_Conditional_MINI)(
            input_arr[0], input_arr[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        )

        Nion_integrals_mini = np.vectorize(lib.Nion_ConditionalM_MINI)(
            growth_out,
            np.log(M_min),
            np.log(M_max),
            cond_mass,
            sigma_cond,
            input_arr[0],
            10 ** input_arr[1],
            10**ap.M_TURN,
            ap.ALPHA_STAR_MINI,
            ap.ALPHA_ESC,
            10**ap.F_STAR7_MINI,
            10**ap.F_ESC7_MINI,
            Mlim_Fstar_MINI,
            Mlim_Fesc_MINI,
            up.cdict["INTEGRATION_METHOD_MINI"],
        )
        print_failure_stats(
            Nion_tables_mini,
            Nion_integrals_mini,
            input_arr,
            abs_tol,
            RELATIVE_TOLERANCE,
            "Nion_c_mini",
        )
    else:
        Nion_tables_mini = np.zeros((hist_size - 1, int(hist_size / 10)))
        Nion_integrals_mini = np.zeros((hist_size - 1, int(hist_size / 10)))

    if plt == mpl.pyplot:
        if mini_flag:
            xl = input_arr[1].shape[1]
            sel_m = (xl * np.arange(6) / 6).astype(int)
            Nion_tb_plot = Nion_tables[..., sel_m]
            Nion_il_plot = Nion_integrals[..., sel_m]
        else:
            Nion_tb_plot = Nion_tables[:, None]
            Nion_il_plot = Nion_integrals[:, None]
            sel_m = np.array([0]).astype(int)

        make_table_comparison_plot(
            [edges_d[:-1], edges_d[:-1]],
            [edges_m[sel_m], edges_m[sel_m]],
            [Nion_tb_plot, Nion_tables_mini[..., sel_m]],
            [Nion_il_plot, Nion_integrals_mini[..., sel_m]],
            plt,
        )

    np.testing.assert_allclose(
        Nion_tables, Nion_integrals, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )

    np.testing.assert_allclose(
        Nion_tables_mini, Nion_integrals_mini, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("R", R_PARAM_LIST)
@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("intmethod", options_intmethod)
def test_SFRD_conditional_table(name, R, intmethod, plt):
    if name != "PS" and intmethod == "FFCOLL":
        pytest.skip("FAST FFCOLL INTEGRALS WORK ONLY WITH EPS")

    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    up = attrs.evolve(
        up,
        INTEGRATION_METHOD_ATOMIC=OPTIONS_INTMETHOD[intmethod],
        INTEGRATION_METHOD_MINI=OPTIONS_INTMETHOD[intmethod],
    )
    fo = attrs.evolve(
        fo,
        USE_MINI_HALOS=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
    )
    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL

    lib.init_ps()

    if "GAUSS-LEGENDRE" in (up.INTEGRATION_METHOD_ATOMIC, up.INTEGRATION_METHOD_MINI):
        lib.initialise_GL(np.log(M_min), np.log(M_max))

    growth_out = lib.dicke(redshift)
    cond_mass = (
        (4.0 / 3.0 * np.pi * (R * u.Mpc) ** 3 * cp.cosmo.critical_density(0) * cp.OMm)
        .to("M_sun")
        .value
    )
    sigma_cond = lib.sigma_z0(cond_mass)
    delta_crit = lib.get_delta_crit(up.cdict["HMF"], sigma_cond, growth_out)

    edges_d = np.linspace(-1, delta_crit * 1.1, num=hist_size).astype("f4")
    edges_m = np.logspace(5, 10, num=int(hist_size / 10)).astype("f4")

    Mlim_Fstar = 1e10 * (10**ap.F_STAR10) ** (-1.0 / ap.ALPHA_STAR)
    Mlim_Fstar_MINI = 1e7 * (10**ap.F_STAR7_MINI) ** (-1.0 / ap.ALPHA_STAR_MINI)

    lib.initialiseSigmaMInterpTable(M_min, max(cond_mass, M_max))

    lib.initialise_SFRD_Conditional_table(
        edges_d[0],
        edges_d[-1],
        growth_out,
        10**ap.M_TURN,
        M_min,
        M_max,
        cond_mass,
        ap.ALPHA_STAR,
        ap.ALPHA_STAR_MINI,
        10**ap.F_STAR10,
        10**ap.F_STAR7_MINI,
        up.cdict["INTEGRATION_METHOD_ATOMIC"],
        up.cdict["INTEGRATION_METHOD_MINI"],
        fo.USE_MINI_HALOS,
    )
    # since the turnover mass table edges are hardcoded, we make sure we are within those limits
    SFRD_tables = np.vectorize(lib.EvaluateSFRD_Conditional)(
        edges_d[:-1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    )
    input_arr = np.meshgrid(edges_d[:-1], np.log10(edges_m[:-1]), indexing="ij")
    SFRD_tables_mini = np.vectorize(lib.EvaluateSFRD_Conditional_MINI)(
        input_arr[0],
        input_arr[1],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    SFRD_integrals = np.vectorize(lib.Nion_ConditionalM)(
        growth_out,
        np.log(M_min),
        np.log(M_max),
        cond_mass,
        sigma_cond,
        edges_d[:-1],
        10**ap.M_TURN,
        ap.ALPHA_STAR,
        0.0,
        10**ap.F_STAR10,
        1.0,
        Mlim_Fstar,
        0.0,
        up.cdict["INTEGRATION_METHOD_ATOMIC"],
    )

    SFRD_integrals_mini = np.vectorize(lib.Nion_ConditionalM_MINI)(
        growth_out,
        np.log(M_min),
        np.log(M_max),
        cond_mass,
        sigma_cond,
        input_arr[0],
        10 ** input_arr[1],
        10**ap.M_TURN,
        ap.ALPHA_STAR_MINI,
        0.0,
        10**ap.F_STAR7_MINI,
        1.0,
        Mlim_Fstar_MINI,
        0.0,
        up.cdict["INTEGRATION_METHOD_MINI"],
    )

    abs_tol = 5e-18  # minimum = exp(-40) ~1e-18
    print_failure_stats(
        SFRD_tables,
        SFRD_integrals,
        [
            edges_d[:-1],
        ],
        abs_tol,
        RELATIVE_TOLERANCE,
        "SFRD_c",
    )
    print_failure_stats(
        SFRD_tables_mini,
        SFRD_integrals_mini,
        input_arr,
        abs_tol,
        RELATIVE_TOLERANCE,
        "SFRD_c_mini",
    )

    if plt == mpl.pyplot:
        xl = input_arr[1].shape[1]
        sel_m = (xl * np.arange(6) / 6).astype(int)
        make_table_comparison_plot(
            [edges_d[:-1], edges_d[:-1]],
            [np.array([0]), edges_m[sel_m]],
            [SFRD_tables[:, None], SFRD_tables_mini[..., sel_m]],
            [SFRD_integrals[:, None], SFRD_integrals_mini[..., sel_m]],
            plt,
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
# @pytest.mark.xfail
def test_conditional_integral_methods(R, name, integrand, plt):
    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    up = attrs.evolve(up, USE_INTERPOLATION_TABLES=True)
    fo = attrs.evolve(
        fo,
        USE_MINI_HALOS=True,
        USE_MASS_DEPENDENT_ZETA=True,
        INHOMO_RECO=True,
        USE_TS_FLUCT=True,
    )
    if "sfr" in integrand:
        ap = attrs.evolve(
            ap, F_ESC10=0.0, F_ESC7_MINI=0.0, ALPHA_ESC=0.0
        )  # F_ESCX is in log10

    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL

    lib.init_ps()
    if "GAUSS-LEGENDRE" in (up.INTEGRATION_METHOD_ATOMIC, up.INTEGRATION_METHOD_MINI):
        lib.initialise_GL(np.log(M_min), np.log(M_max))

    growth_out = lib.dicke(redshift)
    cond_mass = (
        (4.0 / 3.0 * np.pi * (R * u.Mpc) ** 3 * cp.cosmo.critical_density(0) * cp.OMm)
        .to("M_sun")
        .value
    )
    sigma_cond = lib.sigma_z0(cond_mass)
    delta_crit = lib.get_delta_crit(up.cdict["HMF"], sigma_cond, growth_out)

    edges_d = np.linspace(-1, delta_crit * 1.1, num=hist_size).astype("f4")
    edges_m = np.logspace(5, 10, num=int(hist_size / 10)).astype("f4")

    Mlim_Fstar = 1e10 * (10**ap.F_STAR10) ** (-1.0 / ap.ALPHA_STAR)
    Mlim_Fstar_MINI = 1e7 * (10**ap.F_STAR7_MINI) ** (-1.0 / ap.ALPHA_STAR_MINI)
    if ap.ALPHA_ESC != 0.0:
        Mlim_Fesc = 1e10 * (10**ap.F_ESC10) ** (-1.0 / ap.ALPHA_ESC)
        Mlim_Fesc_MINI = 1e7 * (10**ap.F_ESC7_MINI) ** (-1.0 / ap.ALPHA_ESC)
    else:
        Mlim_Fesc = 0.0
        Mlim_Fesc_MINI = 0.0

    lib.initialiseSigmaMInterpTable(M_min, max(cond_mass, M_max))

    integrals = []
    integrals_mini = []
    input_arr = np.meshgrid(edges_d[:-1], np.log10(edges_m[:-1]), indexing="ij")
    for method in ["GSL-QAG", "GAUSS-LEGENDRE", "GAMMA-APPROX"]:
        print(f"Starting method {method}", flush=True)
        if name != "PS" and method == "GAMMA-APPROX":
            continue

        up = attrs.evolve(
            up, INTEGRATION_METHOD_ATOMIC=method, INTEGRATION_METHOD_MINI=method
        )
        lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

        integrals.append(
            np.vectorize(lib.Nion_ConditionalM)(
                growth_out,
                np.log(M_min),
                np.log(M_max),
                cond_mass,
                sigma_cond,
                edges_d[:-1],
                10**ap.M_TURN,
                ap.ALPHA_STAR,
                ap.ALPHA_ESC,
                10**ap.F_STAR10,
                10**ap.F_ESC10,
                Mlim_Fstar,
                Mlim_Fesc,
                up.cdict["INTEGRATION_METHOD_ATOMIC"],
            )
        )
        integrals_mini.append(
            np.vectorize(lib.Nion_ConditionalM_MINI)(
                growth_out,
                np.log(M_min),
                np.log(M_max),
                cond_mass,
                sigma_cond,
                input_arr[0],
                10 ** input_arr[1],
                10**ap.M_TURN,
                ap.ALPHA_STAR_MINI,
                ap.ALPHA_ESC,
                10**ap.F_STAR7_MINI,
                10**ap.F_ESC7_MINI,
                Mlim_Fstar_MINI,
                Mlim_Fesc_MINI,
                up.cdict["INTEGRATION_METHOD_MINI"],
            )
        )

    abs_tol = 5e-18  # minimum = exp(-40) ~1e-18
    if plt == mpl.pyplot:
        xl = input_arr[1].shape[1]
        sel_m = (xl * np.arange(6) / 6).astype(int)
        iplot_mini = [i[..., sel_m] for i in integrals_mini]
        print(sel_m, flush=True)
        make_integral_comparison_plot(
            edges_d[:-1],
            edges_m[sel_m],
            integrals,
            iplot_mini,
            plt,
        )

    np.testing.assert_allclose(
        integrals[1], integrals[0], atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        integrals_mini[1], integrals_mini[0], atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )

    # for the FAST_FFCOLL integrals, only the delta-Mturn behaviour matters (because of the mean fixing), so we divide by
    # the value at delta=0 (mturn ~ 5e7 for minihalos) and set a wider tolerance
    if name == "PS":
        sel_deltazero = np.argmin(np.fabs(edges_d))
        sel_mturn = np.argmin(np.fabs(edges_m - 5e7))
        ffcoll_deltazero = integrals[2][sel_deltazero]
        ffcoll_deltazero_mini = integrals_mini[2][sel_deltazero, sel_mturn]
        qag_deltazero = integrals[0][sel_deltazero]
        qag_deltazero_mini = integrals_mini[0][sel_deltazero, sel_mturn]
        np.testing.assert_allclose(
            integrals[2] / ffcoll_deltazero,
            integrals[0] / qag_deltazero,
            atol=abs_tol,
            rtol=1e-1,
        )
        np.testing.assert_allclose(
            integrals_mini[2] / ffcoll_deltazero_mini[None, :],
            integrals_mini[0] / qag_deltazero_mini[None, :],
            atol=abs_tol,
            rtol=1e-1,
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
    fig, axs = plt.subplots(nrows=2, ncols=len(x), figsize=(16, 16 / len(x) * 2))
    xlabels = kwargs.pop("xlabels", ["delta"] * len(x))
    ylabels = kwargs.pop("ylabels", ["MF_integral"] * len(x))
    zlabels = kwargs.pop("zlabels", ["Mturn"] * len(x))
    for j, z in enumerate(tb_z):
        for i in range(z.size):
            zlab = zlabels[j] + f" = {z[i]:.2e}"
            make_comparison_plot(
                x[j],
                integrals[j][:, i],
                tables[j][:, i],
                ax=axs[:, j],
                xlab=xlabels[j],
                ylab=ylabels[j],
                label_base=zlab,
                logx=False,
                color=f"C{i:d}",
            )


# slightly different from comparison plot since each integral shares a "truth"
def make_integral_comparison_plot(x1, x2, integral_list, integral_list_second, plt):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))

    styles = ["-", ":", "--"]
    for i, (i_first, i_second) in enumerate(zip(integral_list, integral_list_second)):
        axs[0, 0].semilogy(
            x1, i_first, color=f"C{i:d}", linewidth=2, label="Method {i}"
        )
        axs[1, 0].semilogy(x1, i_first / integral_list[0], color=f"C{i:d}", linewidth=2)

        for j in range(x2.size):
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
    label_base="",
    **kwargs,
):
    ax[0].plot(x, true, label=label_base + " True", linestyle="-", **kwargs)
    ax[0].plot(
        x, test, label=label_base + " Test", linestyle=":", linewidth=3, **kwargs
    )
    if logx:
        ax[0].set_xscale("log")
    if logy:
        ax[0].set_yscale("log")
    if xlab:
        ax[0].set_xlabel(xlab)
    if ylab:
        ax[0].set_ylabel(ylab)

    ax[0].legend()

    ax[1].plot(x, (test - true) / true, **kwargs)
    ax[1].set_ylabel("Fractional Difference")


def print_failure_stats(test, truth, input_arr, abs_tol, rel_tol, name):
    sel_failed = np.fabs(truth - test) > (abs_tol + np.fabs(truth) * rel_tol)
    if sel_failed.sum() > 0:
        print(
            f"{name}: atol {abs_tol} rtol {rel_tol} failed {sel_failed.sum()} of {sel_failed.size} {sel_failed.sum() / sel_failed.size * 100:.4f}%"
        )
        print(
            f"subcube of failures [min] [max] {np.argwhere(sel_failed).min(axis=0)} {np.argwhere(sel_failed).max(axis=0)}"
        )
        for i, row in enumerate(input_arr):
            print(
                f"failure range of inputs axis {i} {row[sel_failed].min():.2e} {row[sel_failed].max():.2e}"
            )
        print(
            f"failure range truth ({truth[sel_failed].min():.3e},{truth[sel_failed].max():.3e}) test ({test[sel_failed].min():.3e},{test[sel_failed].max():.3e})"
        )
        print(
            f"max abs diff of failures {np.fabs(truth - test)[sel_failed].max():.4e} relative {(np.fabs(truth - test) / truth)[sel_failed].max():.4e}"
        )

        print(
            f"first 10 = {truth[sel_failed].flatten()[:10]} {test[sel_failed].flatten()[:10]}"
        )


def massfunc_table_comparison_plot(
    massbins,
    conds_m,
    N_cmfi_halo,
    M_inverse_halo,
    conds_d,
    N_cmfi_cell,
    M_inverse_cell,
    plt,
):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    for ax in axs:
        ax.grid()
        ax.set_xlabel("M_halo")
        ax.set_xlim([1e7, 5e11])
        ax.set_xscale("log")
        ax.set_ylabel("P(>M)")
        ax.set_ylim([1e-8, 1.2])
        ax.set_yscale("log")

    [
        axs[0].plot(
            massbins,
            N_cmfi_halo[i, :],
            color=f"C{i:d}",
            linestyle="-",
            label=f"M = {m:.3e}",
        )
        for i, m in enumerate(conds_m)
    ]
    [
        axs[0].plot(
            M_inverse_halo[i, :],
            N_cmfi_halo[i, :],
            color=f"C{i:d}",
            linestyle=":",
            linewidth=3,
        )
        for i, m in enumerate(conds_m)
    ]
    axs[0].legend()

    [
        axs[1].plot(
            massbins,
            N_cmfi_cell[i, :],
            color=f"C{i:d}",
            linestyle="-",
            label=f"d = {d:.3f}",
        )
        for i, d in enumerate(conds_d)
    ]
    [
        axs[1].plot(
            M_inverse_cell[i, :],
            N_cmfi_cell[i, :],
            color=f"C{i:d}",
            linestyle=":",
            linewidth=3,
        )
        for i, d in enumerate(conds_d)
    ]
    axs[1].legend()
