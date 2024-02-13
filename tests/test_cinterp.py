import pytest

import numpy as np
from astropy import constants as c
from astropy import units as u

from py21cmfast import global_params
from py21cmfast.c_21cmfast import lib

from . import produce_integration_test_data as prd

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
    "ST": [10, {"HMF": 1}],
    "Watson": [10, {"HMF": 2}],
    "Watsonz": [10, {"HMF": 3}],
    "Delos": [10, {"HMF": 4}],
}

options_ps = list(OPTIONS_PS.keys())
options_hmf = list(OPTIONS_HMF.keys())

# TODO: write tests for the redshift interpolation tables (global Nion, SFRD, FgtrM)


@pytest.mark.parametrize("name", options_ps)
def test_sigma_table(name):
    redshift, kwargs = OPTIONS_PS[name]
    opts = prd.get_all_options(redshift, kwargs)

    up = opts["user_params"]
    up.update(USE_INTERPOLATION_TABLES=True)
    cp = opts["cosmo_params"]
    lib.Broadcast_struct_global_PS(up(), cp())

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    mass_range = np.logspace(7, 14, num=100)

    sigma_ref = map(lib.sigma_z0, mass_range)
    sigma_table = map(lib.EvaluateSigma, np.log(mass_range))

    np.testing.assert_allclose(sigma_ref, sigma_table, atol=0, rtol=RELATIVE_TOLERANCE)


@pytest.mark.parametrize("name", options_hmf)
def test_Massfunc_conditional_tables(name):
    redshift, kwargs = OPTIONS_PS[name]
    opts = prd.get_all_options(redshift, kwargs)

    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]
    up.update(USE_INTERPOLATION_TABLES=True)

    hist_size = 1000
    edges = np.logspace(7, 12, num=hist_size).astype("f4")
    edges_ln = np.log(edges)
    edges_p_halo = np.linspace(0, 1, num=hist_size).astype("f4")
    edges_p_cell = np.linspace(global_params.MIN_LOGPROB, 0, num=hist_size).astype("f4")
    edges_d = np.linspace(-1, 1.49, num=hist_size).astype("f4")

    lib.Broadcast_struct_global_PS(up(), cp())
    lib.Broadcast_struct_global_UF(up(), cp())
    lib.Broadcast_struct_global_STOC(up(), cp(), ap(), fo())

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    growth_out = lib.dicke(redshift)
    growth_in = lib.dicke(redshift * global_params.ZPRIME_STEP_FACTOR)
    if up.HMF != 1:
        delta_update = lib.get_delta_crit(up.HMF, 0.0, 0.0) * growth_out / growth_in
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
    M_exp_cell = np.vectorize(lib.EvaluateMcoll)(edges_d) * cell_mass
    N_exp_cell = np.vectorize(lib.EvaluateNhalo)(edges_d) * cell_mass

    arg_list_inv_d = np.meshgrid(edges_d, edges_p_cell, indexing="ij")
    N_inverse_cell = np.vectorize(lib.EvaluateNhaloInv)(
        arg_list_inv_d[0], arg_list_inv_d[1]
    )  # LOG MASS

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
    M_exp_halo = np.vectorize(lib.EvaluateMcoll)(edges_ln) * edges
    N_exp_halo = np.vectorize(lib.EvaluateNhalo)(edges_ln) * edges

    arg_list_inv_m = np.meshgrid(np.log(edges), edges_p_halo, indexing="ij")
    N_inverse_halo = np.vectorize(lib.EvaluateNhaloInv)(
        arg_list_inv_m[0], arg_list_inv_m[1]
    )  # LOG MASS

    # Now get the Integrals for comparison at the masses obtained by the inverse tables
    sigma_cond_cell = lib.sigma_z0(cell_mass)
    N_cmfi_cell = np.vectorize(lib.Nhalo_Conditional)(
        growth_out,
        N_inverse_cell,
        np.log(cell_mass),
        sigma_cond_cell,
        arg_list_inv_d[0],
        0,
    )
    N_cmfi_cell = N_cmfi_cell / N_cmfi_cell[:, -1:]  # to get P(>M)
    M_cmf_cell = (
        np.vectorize(lib.Mcoll_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            sigma_cond_cell,
            edges_d,
            fo.INTEGRATION_METHOD_HALOS,
        )
        * cell_mass
    )
    N_cmf_cell = (
        np.vectorize(lib.Nhalo_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            sigma_cond_cell,
            edges_d,
            fo.INTEGRATION_METHOD_HALOS,
        )
        * cell_mass
    )

    sigma_cond_halo = np.vectorize(lib.sigma_z0)(edges)
    if up.HMF == 1:
        delta_update = (
            np.vectorize(lib.get_delta_crit)(up.HMF, sigma_cond_halo, growth_in)
            * growth_out
            / growth_in
        )

    N_cmfi_halo = np.vectorize(lib.Nhalo_Conditional)(
        growth_out,
        N_inverse_halo,
        edges_ln[-1],
        sigma_cond_halo,
        delta_update,
        fo.INTEGRATION_METHOD_HALOS,
    )
    N_cmfi_halo = N_cmfi_halo / N_cmfi_halo[:, -1:]  # to get P(>M)
    M_cmf_halo = (
        np.vectorize(lib.Mcoll_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            sigma_cond_halo,
            delta_update,
            fo.INTEGRATION_METHOD_HALOS,
        )
        * edges
    )
    N_cmf_halo = (
        np.vectorize(lib.Nhalo_Conditional)(
            growth_out,
            edges_ln[0],
            edges_ln[-1],
            sigma_cond_halo,
            delta_update,
            fo.INTEGRATION_METHOD_HALOS,
        )
        * edges
    )

    np.testing.assert_allclose(N_cmf_halo, N_exp_halo, atol=0, rtol=RELATIVE_TOLERANCE)
    np.testing.assert_allclose(N_cmf_cell, N_exp_cell, atol=0, rtol=RELATIVE_TOLERANCE)
    np.testing.assert_allclose(M_cmf_halo, M_exp_halo, atol=0, rtol=RELATIVE_TOLERANCE)
    np.testing.assert_allclose(M_cmf_cell, M_exp_cell, atol=0, rtol=RELATIVE_TOLERANCE)
    np.testing.assert_allclose(
        N_cmfi_halo, N_inverse_halo, atol=0, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        N_cmfi_cell, N_inverse_cell, atol=0, rtol=RELATIVE_TOLERANCE
    )


@pytest.mark.parametrize("name", options_hmf)
def test_FgtrM_conditional_tables(name):
    redshift, kwargs = OPTIONS_PS[name]
    opts = prd.get_all_options(redshift, kwargs)

    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    up.update(USE_INTERPOLATION_TABLES=True)
    lib.Broadcast_struct_global_PS(up(), cp())
    lib.Broadcast_struct_global_UF(up(), cp())
    lib.Broadcast_struct_global_STOC(up(), cp(), ap(), fo())

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL
    edges_d = np.linspace(-1, 1.49, num=hist_size).astype("f4")

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
        lib.Broadcast_struct_global_STOC(up(), cp(), ap(), fo())

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
        lib.initialise_FgtrM_delta_table(
            edges_d[0], edges_d[-1], redshift, growth_out, sigma_min, sigma_cond
        )

        fcoll_tables = np.vectorize(lib.EvaluateFcoll_delta)(
            edges_d, growth_out, sigma_min, sigma_cond
        )
        dfcoll_tables = np.vectorize(lib.EvaluatedFcolldz)(
            edges_d, growth_out, sigma_min, sigma_cond
        )

        up.update(USE_INTERPOLATION_TABLES=False)
        lib.Broadcast_struct_global_PS(up(), cp())
        lib.Broadcast_struct_global_UF(up(), cp())
        lib.Broadcast_struct_global_STOC(up(), cp(), ap(), fo())

        fcoll_integrals = np.vectorize(lib.EvaluateFcoll_delta)(
            edges_d, growth_out, sigma_min, sigma_cond
        )
        dfcoll_integrals = np.vectorize(lib.EvaluatedFcolldz)(
            edges_d, growth_out, sigma_min, sigma_cond
        )

    np.testing.assert_allclose(
        fcoll_tables, fcoll_integrals, atol=0, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        dfcoll_tables, dfcoll_integrals, atol=0, rtol=RELATIVE_TOLERANCE
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
    redshift, kwargs = OPTIONS_PS[name]
    opts = prd.get_all_options(redshift, kwargs)

    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    up.update(USE_INTERPOLATION_TABLES=True)
    lib.Broadcast_struct_global_PS(up(), cp())
    lib.Broadcast_struct_global_UF(up(), cp())
    lib.Broadcast_struct_global_STOC(up(), cp(), ap(), fo())

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL
    edges_d = np.linspace(-1, 1.49, num=hist_size).astype("f4")
    edges_m = np.logspace(5, 10, num=hist_size).astype("f4")

    # This is a strange axis utlising constants from IonisationBox AND SpinTemp, define it more properly
    edges_R = np.linspace(
        up.BOX_LEN / up.HII_DIM,
        ap.R_BUBBLE_MAX,
        num=global_params.NUM_FILTER_STEPS_FOR_Ts,
    )

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(M_min, M_max)

    growth_out = lib.dicke(redshift)

    Mlim_Fstar = 1e10 * ap.F_STAR10 ** (1 / ap.ALPHA_STAR)
    Mlim_Fesc = 1e10 * ap.F_ESC10 ** (1 / ap.ALPHA_ESC)
    Mlim_Fstar_MINI = 1e7 * ap.F_STAR7_MINI ** (1 / ap.ALPHA_STAR_MINI)
    Mlim_Fesc_MINI = 1e7 * ap.F_ESC7_MINI ** (1 / ap.ALPHA_ESC)

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
            ap.M_TURN,
            edges_d[0],
            edges_d[-1],
            M_min,
            M_max,
            cond_mass,
            0.99e5,
            1e10,
            0.99e5,
            1e10,
            ap.ALPHA_STAR,
            ap.ALPHA_STAR_MINI,
            ap.ALPHA_ESC,
            ap.F_STAR10,
            ap.F_ESC10,
            Mlim_Fstar,
            Mlim_Fesc,
            ap.F_STAR7_MINI,
            ap.F_ESC7_MINI,
            Mlim_Fstar_MINI,
            Mlim_Fesc_MINI,
            fo.INTEGRATION_METHOD_ATOMIC,
            fo.INTEGRATION_METHOD_MINI,
            True,
            False,
        )

        input_arr = np.meshgrid(edges_d, edges_m, indexing="ij")
        Nion_tables = np.vectorize(lib.EvaluateNion_Conditional)(
            input_arr[0], input_arr[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        )
        Nion_tables_mini = np.vectorize(lib.EvaluateNion_Conditional)(
            input_arr[0], input_arr[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        )

        Nion_integrals = np.vectorize(lib.NionConditionalM)(
            growth_out,
            M_min,
            M_max,
            sigma_cond,
            input_arr[0],
            input_arr[1],
            ap.ALPHA_STAR,
            ap.ALPHA_ESC,
            ap.F_STAR10,
            ap.F_ESC10,
            Mlim_Fstar,
            Mlim_Fesc,
            fo.INTEGRATION_METHOD_ATOMIC,
        )

        Nion_integrals_mini = np.vectorize(lib.Nion_ConditionalM_MINI)(
            growth_out,
            M_min,
            M_max,
            sigma_cond,
            input_arr[0],
            input_arr[1],
            ap.M_TURN,
            ap.ALPHA_STAR_MINI,
            ap.ALPHA_ESC,
            ap.F_STAR7_MINI,
            ap.F_ESC7_MINI,
            Mlim_Fstar_MINI,
            Mlim_Fesc_MINI,
            fo.INTEGRATION_METHOD_MINI,
        )

        np.testing.assert_allclose(
            Nion_tables, Nion_integrals, atol=0, rtol=RELATIVE_TOLERANCE
        )
        np.testing.assert_allclose(
            Nion_tables_mini, Nion_integrals_mini, atol=0, rtol=RELATIVE_TOLERANCE
        )


@pytest.mark.parametrize("name", options_hmf)
def test_SFRD_conditional_table(name):
    redshift, kwargs = OPTIONS_PS[name]
    opts = prd.get_all_options(redshift, kwargs)

    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    up.update(USE_INTERPOLATION_TABLES=True)
    lib.Broadcast_struct_global_PS(up(), cp())
    lib.Broadcast_struct_global_UF(up(), cp())
    lib.Broadcast_struct_global_STOC(up(), cp(), ap(), fo())

    hist_size = 1000
    M_min = global_params.M_MIN_INTEGRAL
    M_max = global_params.M_MAX_INTEGRAL
    edges_d = np.linspace(-1, 1.49, num=hist_size).astype("f4")
    edges_m = np.logspace(5, 10, num=hist_size).astype("f4")

    # This is a strange axis utlising constants from IonisationBox AND SpinTemp, define it more properly
    edges_R = np.linspace(
        up.BOX_LEN / up.HII_DIM,
        ap.R_BUBBLE_MAX,
        num=global_params.NUM_FILTER_STEPS_FOR_Ts,
    )

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(M_min, M_max)

    growth_out = lib.dicke(redshift)

    Mlim_Fstar = 1e10 * ap.F_STAR10 ** (1 / ap.ALPHA_STAR)
    Mlim_Fstar_MINI = 1e7 * ap.F_STAR7_MINI ** (1 / ap.ALPHA_STAR_MINI)

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
        lib.initialise_SFRD_Conditional_spline(
            edges_d[0],
            edges_d[-1],
            growth_out,
            ap.M_TURN,
            M_min,
            M_max,
            cond_mass,
            ap.ALPHA_STAR,
            ap.ALPHA_STAR_MINI,
            ap.F_STAR10,
            ap.F_STAR7_MINI,
            Mlim_Fstar,
            Mlim_Fstar_MINI,
            fo.INTEGRATION_METHOD_ATOMIC,
            fo.INTEGRATION_METHOD_MINI,
            True,
        )

        input_arr = np.meshgrid(edges_d, edges_m, indexing="ij")
        SFRD_tables = np.vectorize(lib.EvaluateNion_Conditional)(
            input_arr[0], input_arr[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        )
        SFRD_tables_mini = np.vectorize(lib.EvaluateNion_Conditional)(
            input_arr[0], input_arr[1], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        )

        SFRD_integrals = np.vectorize(lib.NionConditionalM)(
            growth_out,
            M_min,
            M_max,
            sigma_cond,
            input_arr[0],
            input_arr[1],
            ap.ALPHA_STAR,
            0.0,
            ap.F_STAR10,
            1.0,
            Mlim_Fstar,
            0.0,
            fo.INTEGRATION_METHOD_ATOMIC,
        )

        SFRD_integrals_mini = np.vectorize(lib.Nion_ConditionalM_MINI)(
            growth_out,
            M_min,
            M_max,
            sigma_cond,
            input_arr[0],
            input_arr[1],
            ap.M_TURN,
            ap.ALPHA_STAR_MINI,
            0.0,
            ap.F_STAR7_MINI,
            1.0,
            Mlim_Fstar_MINI,
            0.0,
            fo.INTEGRATION_METHOD_MINI,
        )

        np.testing.assert_allclose(
            SFRD_tables, SFRD_integrals, atol=0, rtol=RELATIVE_TOLERANCE
        )
        np.testing.assert_allclose(
            SFRD_tables_mini, SFRD_integrals_mini, atol=0, rtol=RELATIVE_TOLERANCE
        )
