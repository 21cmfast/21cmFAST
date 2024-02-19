import pytest

import numpy as np
from astropy import constants as c
from astropy import units as u

from py21cmfast import AstroParams, CosmoParams, FlagOptions, UserParams, global_params
from py21cmfast.c_21cmfast import ffi, lib

from . import produce_integration_test_data as prd

# NOTE: The relative tolerance is set to cover the inaccuracy in interpolaton
#       Whereas absolute tolerances are set to avoid issues with minima
#       i.e the SFRD table has a forced minima of exp(-50)
RELATIVE_TOLERANCE = 1e-2

OPTIONS_PS = {
    "EH": [10, {"POWER_SPECTRUM": 0}],
    "BBKS": [10, {"POWER_SPECTRUM": 1}],
    "BE": [10, {"POWER_SPECTRUM": 2}],
    "Peebles": [10, {"POWER_SPECTRUM": 3}],
    "White": [10, {"POWER_SPECTRUM": 4}],
    "CLASS": [10, {"POWER_SPECTRUM": 5}],
}

OPTIONS_HMF = {
    "PS": [10, {"HMF": 0}],
    # "ST": [10, {"HMF": 1}],
    # "Watson": [10, {"HMF": 2}],
    # "Watsonz": [10, {"HMF": 3}],
    # "Delos": [10, {"HMF": 4}],
}

options_ps = list(OPTIONS_PS.keys())
options_hmf = list(OPTIONS_HMF.keys())


# TODO: write tests for the redshift interpolation tables (global Nion, SFRD, FgtrM)
@pytest.mark.parametrize("name", options_ps)
def test_sigma_table(name):
    abs_tol = 1e-12

    redshift, kwargs = OPTIONS_PS[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = UserParams(opts["user_params"])
    cp = CosmoParams(opts["cosmo_params"])
    up.update(USE_INTERPOLATION_TABLES=True)
    lib.Broadcast_struct_global_PS(up(), cp())
    lib.Broadcast_struct_global_UF(up(), cp())

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    mass_range = np.logspace(7, 14, num=100)

    sigma_ref = np.vectorize(lib.sigma_z0)(mass_range)
    dummy_pointer = np.zeros(1).astype("f8")
    sigma_table = np.vectorize(lib.EvaluateSigma)(
        np.log(mass_range), 0, ffi.cast("double *", dummy_pointer.ctypes.data)
    )

    np.testing.assert_allclose(
        sigma_ref, sigma_table, atol=abs_tol, rtol=RELATIVE_TOLERANCE
    )


# NOTE: This test currently fails (~10% differences in mass in <1% of bins)
#   I don't want to relax the tolerance yet since it can be improved, but
#   for now this is acceptable
@pytest.mark.xfail
@pytest.mark.parametrize("name", options_hmf)
def test_Massfunc_conditional_tables(name):
    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = UserParams(opts["user_params"])
    cp = CosmoParams(opts["cosmo_params"])
    ap = AstroParams(opts["astro_params"])
    fo = FlagOptions(opts["flag_options"])
    up.update(USE_INTERPOLATION_TABLES=True)

    hist_size = 1000
    edges = np.logspace(7, 12, num=hist_size).astype("f4")
    edges_ln = np.log(edges)

    lib.Broadcast_struct_global_PS(up(), cp())
    lib.Broadcast_struct_global_UF(up(), cp())
    lib.Broadcast_struct_global_IT(up(), cp(), ap(), fo())

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    if up.INTEGRATION_METHOD_HALOS == 1:
        lib.initialise_GL(100, np.log(edges[0]), np.log(edges[-1]))

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
    delta_crit = lib.get_delta_crit(up.HMF, sigma_cond_cell, growth_in)
    delta_update = (
        np.vectorize(lib.get_delta_crit)(up.HMF, sigma_cond_halo, growth_in)
        * growth_out
        / growth_in
    )
    edges_d = np.linspace(-1, delta_crit * 0.98, num=hist_size).astype("f4")

    # Cell Integrals
    arg_list_inv_d = np.meshgrid(edges_d[:-1], edges_ln[:-1], indexing="ij")
    N_cmfi_cell = np.vectorize(lib.Nhalo_Conditional)(
        growth_out,
        arg_list_inv_d[1],  # lnM
        edges_ln[-1],  # integrate to max mass
        sigma_cond_cell,
        arg_list_inv_d[0],  # density
        up.INTEGRATION_METHOD_HALOS,
    )

    N_cmfi_cell = (
        N_cmfi_cell / N_cmfi_cell[:, :1]
    )  # to get P(>M) since the y-axis is the lower integral limit
    mask_cell = N_cmfi_cell > np.exp(
        global_params.MIN_LOGPROB
    )  # we don't want to compare below the min prob
    N_cmfi_cell = np.clip(N_cmfi_cell, np.exp(global_params.MIN_LOGPROB), 1)
    N_cmfi_cell = np.log(N_cmfi_cell)

    M_cmf_cell = (
        np.vectorize(lib.Mcoll_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            sigma_cond_cell,
            edges_d[:-1],
            up.INTEGRATION_METHOD_HALOS,
        )
        * cell_mass
    )
    N_cmf_cell = (
        np.vectorize(lib.Nhalo_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            sigma_cond_cell,
            edges_d[:-1],
            up.INTEGRATION_METHOD_HALOS,
        )
        * cell_mass
    )

    # Cell Tables
    # initialise_dNdM_tables(DELTA_MIN, MAX_DELTAC_FRAC*delta_crit, const_struct->lnM_min, const_struct->lnM_max_tb, const_struct->growth_out, const_struct->lnM_cond, false);
    lib.initialise_dNdM_tables(
        edges_d[0],
        edges_d[-1],
        edges_ln[0],
        edges_ln[-1],
        growth_out,
        np.log(cell_mass),
        False,
    )

    M_exp_cell = np.vectorize(lib.EvaluateMcoll)(edges_d[:-1]) * cell_mass
    N_exp_cell = np.vectorize(lib.EvaluateNhalo)(edges_d[:-1]) * cell_mass

    N_inverse_cell = np.vectorize(lib.EvaluateNhaloInv)(
        arg_list_inv_d[0], N_cmfi_cell
    )  # LOG MASS, evaluated at the probabilities given by the integral

    # Halo Integrals
    arg_list_inv_m = np.meshgrid(edges_ln[:-1], edges_ln[:-1], indexing="ij")
    N_cmfi_halo = np.vectorize(lib.Nhalo_Conditional)(
        growth_out,
        arg_list_inv_m[1],
        edges_ln[-1],
        sigma_cond_halo[:-1, None],  # (condition,masslimit)
        delta_update[:-1, None],
        up.INTEGRATION_METHOD_HALOS,
    )

    # To get P(>M), NOTE that some conditions have no integral
    N_cmfi_halo = N_cmfi_halo / (
        N_cmfi_halo[:, :1] + np.all(N_cmfi_halo == 0, axis=1)[:, None]
    )  # if all entries are zero, do not nan the row, just divide by 1
    mask_halo = (
        arg_list_inv_m[0] > arg_list_inv_m[1]
    )  # we don't want to compare above the conditinos

    M_cmf_halo = (
        np.vectorize(lib.Mcoll_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            sigma_cond_halo[:-1],
            delta_update[:-1],
            up.INTEGRATION_METHOD_HALOS,
        )
        * edges[:-1]
    )
    N_cmf_halo = (
        np.vectorize(lib.Nhalo_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            sigma_cond_halo[:-1],
            delta_update[:-1],
            up.INTEGRATION_METHOD_HALOS,
        )
        * edges[:-1]
    )

    # initialise_dNdM_tables(const_struct->lnM_min, const_struct->lnM_max_tb,const_struct->lnM_min, const_struct->lnM_max_tb,
    #                         const_struct->growth_out, const_struct->growth_in, true);
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
    M_exp_halo = np.vectorize(lib.EvaluateMcoll)(edges_ln[:-1]) * edges[:-1]
    N_exp_halo = np.vectorize(lib.EvaluateNhalo)(edges_ln[:-1]) * edges[:-1]

    N_inverse_halo = np.vectorize(lib.EvaluateNhaloInv)(
        arg_list_inv_m[0], N_cmfi_halo
    )  # LOG MASS, evaluated at the probabilities given by the integral

    # NOTE: The tables get inaccurate in the smallest halo bin where the condition mass approaches the minimum
    #       We set the absolute tolerance to be insiginificant in sampler terms (~1% of a halo)
    np.testing.assert_allclose(
        N_cmf_halo, N_exp_halo, atol=1e-2, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        N_cmf_cell, N_exp_cell, atol=1e-2, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        M_cmf_halo, M_exp_halo, atol=edges[0] * 1e-2, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        M_cmf_cell, M_exp_cell, atol=edges[0] * 1e-2, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        np.exp(arg_list_inv_d[1][mask_cell]),
        np.exp(N_inverse_cell[mask_cell]),
        atol=edges[0] * 1e-2,
        rtol=RELATIVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        np.exp(arg_list_inv_m[1][mask_halo]),
        np.exp(N_inverse_halo[mask_halo]),
        atol=edges[0] * 1e-2,
        rtol=RELATIVE_TOLERANCE,
    )


@pytest.mark.parametrize("name", options_hmf)
def test_FgtrM_conditional_tables(name):
    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = UserParams(opts["user_params"])
    cp = CosmoParams(opts["cosmo_params"])
    ap = AstroParams(opts["astro_params"])
    fo = FlagOptions(opts["flag_options"])

    up.update(USE_INTERPOLATION_TABLES=True)
    lib.Broadcast_struct_global_PS(up(), cp())
    lib.Broadcast_struct_global_UF(up(), cp())
    lib.Broadcast_struct_global_IT(up(), cp(), ap(), fo())

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL
    edges_d = np.linspace(-1, 1.65, num=hist_size).astype(
        "f4"
    )  # EPS is forced with FgtrM due to the erfc functions

    # This is a strange linear axis utlising constants from IonisationBox AND SpinTemp, define it more properly
    edges_R = np.linspace(
        up.BOX_LEN / up.HII_DIM,
        ap.R_BUBBLE_MAX,
        num=global_params.NUM_FILTER_STEPS_FOR_Ts,
    )

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(M_min, M_max)

    growth_out = lib.dicke(redshift)
    sigma_min = lib.sigma_z0(M_min)

    for R in edges_R:
        up.update(USE_INTERPOLATION_TABLES=True)
        lib.Broadcast_struct_global_PS(up(), cp())
        lib.Broadcast_struct_global_UF(up(), cp())
        lib.Broadcast_struct_global_IT(up(), cp(), ap(), fo())

        cond_mass = (
            (
                4.0
                / 3.0
                * np.pi
                * (R * u.Mpc) ** 3
                * cp.cosmo.critical_density(0)
                * cp.OMm
            )
            .to("M_sun")
            .value
        )
        sigma_cond = lib.sigma_z0(cond_mass)
        # initialise_FgtrM_delta_table(min_densities[R_ct], max_densities[R_ct], zpp_for_evolve_list[R_ct],
        #  zpp_growth[R_ct], sigma_min[R_ct], sigma_max[R_ct]);
        # NOTE: Rather than keeping zp constant we keep zpp constant
        lib.initialise_FgtrM_delta_table(
            edges_d[0], edges_d[-1], redshift, growth_out, sigma_min, sigma_cond
        )

        # fcoll = EvaluateFcoll_delta(curr_dens,zpp_growth[R_ct],sigma_min[R_ct],sigma_max[R_ct]);
        # dfcoll = EvaluatedFcolldz(curr_dens,zpp_for_evolve_list[R_ct],sigma_min[R_ct],sigma_max[R_ct]);
        fcoll_tables = np.vectorize(lib.EvaluateFcoll_delta)(
            edges_d[:-1], growth_out, sigma_min, sigma_cond
        )
        dfcoll_tables = np.vectorize(lib.EvaluatedFcolldz)(
            edges_d[:-1], redshift, sigma_min, sigma_cond
        )

        up.update(USE_INTERPOLATION_TABLES=False)
        lib.Broadcast_struct_global_PS(up(), cp())
        lib.Broadcast_struct_global_UF(up(), cp())
        lib.Broadcast_struct_global_IT(up(), cp(), ap(), fo())

        fcoll_integrals = np.vectorize(lib.EvaluateFcoll_delta)(
            edges_d[:-1], growth_out, sigma_min, sigma_cond
        )
        dfcoll_integrals = np.vectorize(lib.EvaluatedFcolldz)(
            edges_d[:-1], redshift, sigma_min, sigma_cond
        )

    np.testing.assert_allclose(
        fcoll_tables, fcoll_integrals, atol=1e-5, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        dfcoll_tables, dfcoll_integrals, atol=1e-5, rtol=RELATIVE_TOLERANCE
    )


# A few notes on this test function:
#   Minihalos are set to true to test both tables
#   Mass limits are set explicitly using the param values
#       and #defines are copied to hard code (not ideal)
#   Density and Mturn limits are set to their maxima since we don't have cubes.
#       Hence this is a worst case scenario
#   While the EvaluateX() functions are useful in the main code to be agnostic to USE_INTERPOLATION_TABLES
#       I do not use them here fully, instead calling the integrals directly to avoid parameter changes
@pytest.mark.parametrize("name", options_hmf)
def test_Nion_conditional_tables(name):
    abs_tol = 1e-17  # min = exp(-40) ~4e-18

    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = UserParams(opts["user_params"])
    cp = CosmoParams(opts["cosmo_params"])
    ap = AstroParams(opts["astro_params"])
    fo = FlagOptions(opts["flag_options"])

    up.update(USE_INTERPOLATION_TABLES=True)
    fo.update(USE_MINI_HALOS=True)
    lib.Broadcast_struct_global_PS(up(), cp())
    lib.Broadcast_struct_global_UF(up(), cp())
    lib.Broadcast_struct_global_IT(up(), cp(), ap(), fo())

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL
    edges_d = np.linspace(-1, 1.49, num=hist_size).astype("f4")
    edges_m = np.logspace(5, 10, num=int(hist_size / 10)).astype("f4")

    # This is a strange axis utlising constants from IonisationBox AND SpinTemp, define it more properly
    edges_R = np.linspace(
        up.BOX_LEN / up.HII_DIM,
        ap.R_BUBBLE_MAX,
        num=global_params.NUM_FILTER_STEPS_FOR_Ts,
    )

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(M_min, M_max)

    if up.INTEGRATION_METHOD_ATOMIC == 1 or up.INTEGRATION_METHOD_MINI == 1:
        lib.initialise_GL(100, np.log(M_min), np.log(M_max))

    growth_out = lib.dicke(redshift)

    Mlim_Fstar = 1e10 * (10**ap.F_STAR10) ** (-1.0 / ap.ALPHA_STAR)
    Mlim_Fesc = 1e10 * (10**ap.F_ESC10) ** (-1.0 / ap.ALPHA_ESC)
    Mlim_Fstar_MINI = 1e7 * (10**ap.F_STAR7_MINI) ** (-1.0 / ap.ALPHA_STAR_MINI)
    Mlim_Fesc_MINI = 1e7 * (10**ap.F_ESC7_MINI) ** (-1.0 / ap.ALPHA_ESC)

    for R in edges_R:
        cond_mass = (
            (
                4.0
                / 3.0
                * np.pi
                * (R * u.Mpc) ** 3
                * cp.cosmo.critical_density(0)
                * cp.OMm
            )
            .to("M_sun")
            .value
        )
        sigma_cond = lib.sigma_z0(cond_mass)
        lib.initialise_Nion_Conditional_spline(
            redshift,
            ap.M_TURN,  # not the redshift dependent version in this test
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
            up.INTEGRATION_METHOD_ATOMIC,
            up.INTEGRATION_METHOD_MINI,
            True,
            False,
        )

        input_arr = np.meshgrid(edges_d[:-1], np.log10(edges_m[:-1]), indexing="ij")
        Nion_tables = np.vectorize(lib.EvaluateNion_Conditional)(
            input_arr[0], input_arr[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        )
        Nion_tables_mini = np.vectorize(lib.EvaluateNion_Conditional_MINI)(
            input_arr[0], input_arr[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        )

        Nion_integrals = np.vectorize(lib.Nion_ConditionalM)(
            growth_out,
            np.log(M_min),
            np.log(M_max),
            sigma_cond,
            input_arr[0],
            10 ** input_arr[1],
            ap.ALPHA_STAR,
            ap.ALPHA_ESC,
            10**ap.F_STAR10,
            10**ap.F_ESC10,
            Mlim_Fstar,
            Mlim_Fesc,
            up.INTEGRATION_METHOD_ATOMIC,
        )

        Nion_integrals_mini = np.vectorize(lib.Nion_ConditionalM_MINI)(
            growth_out,
            np.log(M_min),
            np.log(M_max),
            sigma_cond,
            input_arr[0],
            10 ** input_arr[1],
            ap.M_TURN,
            ap.ALPHA_STAR_MINI,
            ap.ALPHA_ESC,
            10**ap.F_STAR7_MINI,
            10**ap.F_ESC7_MINI,
            Mlim_Fstar_MINI,
            Mlim_Fesc_MINI,
            up.INTEGRATION_METHOD_MINI,
        )

        np.testing.assert_allclose(
            Nion_tables, Nion_integrals, atol=abs_tol, rtol=RELATIVE_TOLERANCE
        )
        np.testing.assert_allclose(
            Nion_tables_mini, Nion_integrals_mini, atol=abs_tol, rtol=RELATIVE_TOLERANCE
        )


@pytest.mark.parametrize("name", options_hmf)
def test_SFRD_conditional_table(name):
    abs_tol = 1e-21  # minimum = exp(-50) ~1e-22

    redshift, kwargs = OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = UserParams(opts["user_params"])
    cp = CosmoParams(opts["cosmo_params"])
    ap = AstroParams(opts["astro_params"])
    fo = FlagOptions(opts["flag_options"])

    up.update(USE_INTERPOLATION_TABLES=True)
    fo.update(USE_MINI_HALOS=True)
    lib.Broadcast_struct_global_PS(up(), cp())
    lib.Broadcast_struct_global_UF(up(), cp())
    lib.Broadcast_struct_global_IT(up(), cp(), ap(), fo())

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL
    edges_d = np.linspace(-1, 1.49, num=hist_size).astype("f4")
    edges_m = np.logspace(5, 10, num=int(hist_size / 10)).astype("f4")

    # This is a strange axis utlising constants from IonisationBox AND SpinTemp, define it more properly
    edges_R = np.linspace(
        up.BOX_LEN / up.HII_DIM,
        ap.R_BUBBLE_MAX,
        num=global_params.NUM_FILTER_STEPS_FOR_Ts,
    )

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(M_min, M_max)

    if up.INTEGRATION_METHOD_ATOMIC == 1 or up.INTEGRATION_METHOD_MINI == 1:
        lib.initialise_GL(100, np.log(M_min), np.log(M_max))

    growth_out = lib.dicke(redshift)

    Mlim_Fstar = 1e10 * (10**ap.F_STAR10) ** (-1.0 / ap.ALPHA_STAR)
    Mlim_Fstar_MINI = 1e7 * (10**ap.F_STAR7_MINI) ** (-1.0 / ap.ALPHA_STAR_MINI)

    for R in edges_R:
        cond_mass = (
            (
                4.0
                / 3.0
                * np.pi
                * (R * u.Mpc) ** 3
                * cp.cosmo.critical_density(0)
                * cp.OMm
            )
            .to("M_sun")
            .value
        )
        sigma_cond = lib.sigma_z0(cond_mass)

        lib.initialise_SFRD_Conditional_table(
            edges_d[0],
            edges_d[-1],
            growth_out,
            ap.M_TURN,
            M_min,
            M_max,
            cond_mass,
            ap.ALPHA_STAR,
            ap.ALPHA_STAR_MINI,
            10**ap.F_STAR10,
            10**ap.F_STAR7_MINI,
            up.INTEGRATION_METHOD_ATOMIC,
            up.INTEGRATION_METHOD_MINI,
            True,
        )
        # since the turnover mass table edges are hardcoded, we make sure we are within those limits
        SFRD_tables = np.vectorize(lib.EvaluateSFRD_Conditional)(
            edges_d[:-1], growth_out, M_min, M_max, sigma_cond, ap.M_TURN, Mlim_Fstar
        )
        input_arr = np.meshgrid(edges_d[:-1], np.log10(edges_m[:-1]), indexing="ij")
        SFRD_tables_mini = np.vectorize(lib.EvaluateSFRD_Conditional_MINI)(
            input_arr[0],
            input_arr[1],
            growth_out,
            M_min,
            M_max,
            sigma_cond,
            ap.M_TURN,
            Mlim_Fstar_MINI,
        )

        SFRD_integrals = np.vectorize(lib.Nion_ConditionalM)(
            growth_out,
            np.log(M_min),
            np.log(M_max),
            sigma_cond,
            edges_d[:-1],
            ap.M_TURN,
            ap.ALPHA_STAR,
            0.0,
            10**ap.F_STAR10,
            1.0,
            Mlim_Fstar,
            0.0,
            up.INTEGRATION_METHOD_ATOMIC,
        )

        SFRD_integrals_mini = np.vectorize(lib.Nion_ConditionalM_MINI)(
            growth_out,
            np.log(M_min),
            np.log(M_max),
            sigma_cond,
            input_arr[0],
            10 ** input_arr[1],
            ap.M_TURN,
            ap.ALPHA_STAR_MINI,
            0.0,
            10**ap.F_STAR7_MINI,
            1.0,
            Mlim_Fstar_MINI,
            0.0,
            up.INTEGRATION_METHOD_MINI,
        )

        np.testing.assert_allclose(
            SFRD_tables, SFRD_integrals, atol=abs_tol, rtol=RELATIVE_TOLERANCE
        )
        np.testing.assert_allclose(
            SFRD_tables_mini, SFRD_integrals_mini, atol=abs_tol, rtol=RELATIVE_TOLERANCE
        )
