import pytest

import matplotlib as mpl
import numpy as np

from py21cmfast import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    PerturbHaloField,
    UserParams,
    global_params,
)
from py21cmfast.c_21cmfast import ffi, lib

from . import produce_integration_test_data as prd
from . import test_c_interpolation_tables as cint

RELATIVE_TOLERANCE = 1e-1

options_hmf = list(cint.OPTIONS_HMF.keys())

options_delta = [-0.9, 0, 1, 1.6]  # cell densities to draw samples from
options_mass = [1e9, 1e10, 1e11, 1e12]  # halo masses to draw samples from


@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("mass", options_mass)
def test_sampler_from_catalog(name, mass, plt):
    redshift, kwargs = cint.OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = UserParams(opts["user_params"])
    cp = CosmoParams(opts["cosmo_params"])
    ap = AstroParams(opts["astro_params"])
    fo = FlagOptions(opts["flag_options"])
    up.update(USE_INTERPOLATION_TABLES=True)
    lib.Broadcast_struct_global_all(up(), cp(), ap(), fo())

    l10min = np.log10(up.SAMPLER_MIN_MASS)
    l10max = np.log10(mass)
    edges = np.logspace(l10min, l10max, num=int(10 * (l10max - l10min)))
    bin_minima = edges[:-1]
    bin_maxima = edges[1:]
    dlnm = np.log(edges[1:]) - np.log(edges[:-1])

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    n_cond = 30000

    z = 6.0
    z_prev = 5.8
    growth_prev = lib.dicke(z_prev)
    growthf = lib.dicke(z)

    sigma_cond_m = lib.sigma_z0(mass)
    delta_cond_m = (
        lib.get_delta_crit(up.HMF, sigma_cond_m, growth_prev) * growthf / growth_prev
    )
    mass_dens = cp.cosmo.Om0 * cp.cosmo.critical_density(0).to("Mpc-3 M_sun").value
    volume_total_m = mass * n_cond / mass_dens

    crd_in = np.zeros(3 * n_cond).astype("i4")
    # HALO MASS CONDITIONS WITH FIXED z-step
    cond_in = np.full(n_cond, fill_value=mass).astype("f4")  # mass at z6

    nhalo_out = np.zeros(1).astype("i4")
    N_out = np.zeros(n_cond).astype("i4")
    M_out = np.zeros(n_cond).astype("f8")
    exp_M = np.zeros(n_cond).astype("f8")
    exp_N = np.zeros(n_cond).astype("f8")
    halomass_out = np.zeros(int(1e8)).astype("f4")
    halocrd_out = np.zeros(int(3e8)).astype("i4")

    lib.single_test_sample(
        up(),
        cp(),
        ap(),
        fo(),
        12345,
        n_cond,
        ffi.cast("float *", cond_in.ctypes.data),
        ffi.cast("int *", crd_in.ctypes.data),
        z,
        z_prev,
        ffi.cast("int *", nhalo_out.ctypes.data),
        ffi.cast("int *", N_out.ctypes.data),
        ffi.cast("double *", exp_N.ctypes.data),
        ffi.cast("double *", M_out.ctypes.data),
        ffi.cast("double *", exp_M.ctypes.data),
        ffi.cast("float *", halomass_out.ctypes.data),
        ffi.cast("int *", halocrd_out.ctypes.data),
    )

    # since the tables are reallocated in the test sample function, we redo them here
    lib.initialiseSigmaMInterpTable(edges[0] / 2, edges[-1])

    # get CMF integrals in the same bins
    bin_minima = edges[:-1]
    bin_maxima = edges[1:]
    binned_cmf = np.vectorize(lib.Nhalo_Conditional)(
        growthf,
        np.log(bin_minima),
        np.log(bin_maxima),
        mass,
        sigma_cond_m,
        delta_cond_m,
        0,
    )

    hist, _ = np.histogram(halomass_out, edges)
    mf_out = hist / volume_total_m / dlnm
    binned_cmf = binned_cmf * n_cond / volume_total_m / dlnm * mass

    if plt == mpl.pyplot:
        plot_sampler_comparison(
            edges,
            exp_N,
            exp_M,
            N_out,
            M_out,
            binned_cmf,
            mf_out,
            f"mass = {mass:.2e}",
            plt,
        )

    np.testing.assert_allclose(N_out.mean(), exp_N[0], rtol=RELATIVE_TOLERANCE)
    np.testing.assert_allclose(M_out.mean(), exp_M[0], rtol=RELATIVE_TOLERANCE)
    np.testing.assert_allclose(mf_out, binned_cmf, rtol=RELATIVE_TOLERANCE)


@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("delta", options_delta)
def test_sampler_from_grid(name, delta, plt):
    redshift, kwargs = cint.OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = UserParams(opts["user_params"])
    cp = CosmoParams(opts["cosmo_params"])
    ap = AstroParams(opts["astro_params"])
    fo = FlagOptions(opts["flag_options"])
    up.update(USE_INTERPOLATION_TABLES=True)
    lib.Broadcast_struct_global_all(up(), cp(), ap(), fo())

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    n_cond = 30000

    z = 6.0
    growthf = lib.dicke(z)

    mass_dens = cp.cosmo.Om0 * cp.cosmo.critical_density(0).to("Mpc-3 M_sun").value
    cellvol = (up.BOX_LEN / up.HII_DIM) ** 3
    cell_mass = cellvol * mass_dens

    l10min = np.log10(up.SAMPLER_MIN_MASS)
    l10max = np.log10(cell_mass)
    edges = np.logspace(l10min, l10max, num=int(10 * (l10max - l10min)))
    bin_minima = edges[:-1]
    bin_maxima = edges[1:]
    dlnm = np.log(edges[1:]) - np.log(edges[:-1])

    sigma_cond = lib.sigma_z0(cell_mass)
    volume_total = cellvol * n_cond

    crd_in = np.zeros(3 * n_cond).astype("i4")

    cond_in = np.full(n_cond, fill_value=delta).astype("f4")  # mass at z6

    nhalo_out = np.zeros(1).astype("i4")
    N_out = np.zeros(n_cond).astype("i4")
    M_out = np.zeros(n_cond).astype("f8")
    exp_M = np.zeros(n_cond).astype("f8")
    exp_N = np.zeros(n_cond).astype("f8")
    halomass_out = np.zeros(int(1e8)).astype("f4")
    halocrd_out = np.zeros(int(3e8)).astype("i4")

    lib.single_test_sample(
        up(),
        cp(),
        ap(),
        fo(),
        12345,  # TODO: homogenize
        n_cond,
        ffi.cast("float *", cond_in.ctypes.data),
        ffi.cast("int *", crd_in.ctypes.data),
        z,
        -1,
        ffi.cast("int *", nhalo_out.ctypes.data),
        ffi.cast("int *", N_out.ctypes.data),
        ffi.cast("double *", exp_N.ctypes.data),
        ffi.cast("double *", M_out.ctypes.data),
        ffi.cast("double *", exp_M.ctypes.data),
        ffi.cast("float *", halomass_out.ctypes.data),
        ffi.cast("int *", halocrd_out.ctypes.data),
    )

    # since the tables are reallocated in the test sample function, we redo them here
    lib.initialiseSigmaMInterpTable(edges[0] / 2, edges[-1])

    # get CMF integrals in the same bins
    bin_minima = edges[:-1]
    bin_maxima = edges[1:]
    binned_cmf = np.vectorize(lib.Nhalo_Conditional)(
        growthf,
        np.log(bin_minima),
        np.log(bin_maxima),
        cell_mass,
        sigma_cond,
        delta,
        0,
    )

    hist, _ = np.histogram(halomass_out, edges)
    mf_out = hist / volume_total / dlnm
    binned_cmf = binned_cmf * n_cond / volume_total / dlnm * cell_mass

    if plt == mpl.pyplot:
        plot_sampler_comparison(
            edges,
            exp_N,
            exp_M,
            N_out,
            M_out,
            binned_cmf,
            mf_out,
            f"delta = {delta:.2f}",
            plt,
        )

    np.testing.assert_allclose(N_out.mean(), exp_N[0], rtol=RELATIVE_TOLERANCE)
    np.testing.assert_allclose(M_out.mean(), exp_M[0], rtol=RELATIVE_TOLERANCE)
    np.testing.assert_allclose(mf_out, binned_cmf, rtol=RELATIVE_TOLERANCE)


# NOTE: this test is pretty circular. The only way I think I can test the scaling relations are to
#   calculate them in the backend and re-write them in the test for a few masses. This means that
#   changes to any scaling relation model will result in a test fail
def test_halo_scaling_relations():
    # specify parameters to use for this test
    f_star10 = -1.0
    f_star7 = -2.0
    a_star = 1.0
    a_star_mini = 1.0
    t_star = 0.5
    f_esc10 = -1.0
    f_esc7 = -1.0
    a_esc = -0.5  # for the test we don't want a_esc = -a_star
    lx = 40.0
    lx_mini = 40.0
    sigma_star = 0.3
    sigma_sfr_lim = 0.2
    sigma_sfr_index = -0.12
    sigma_lx = 0.5

    redshift = 10.0

    # setup specific parameters that so we know what the outcome should be
    up = UserParams()
    cp = CosmoParams()
    ap = AstroParams(
        F_STAR10=f_star10,
        F_STAR7_MINI=f_star7,
        ALPHA_STAR=a_star,
        ALPHA_STAR_MINI=a_star_mini,
        F_ESC10=f_esc10,
        F_ESC7_MINI=f_esc7,
        ALPHA_ESC=a_esc,
        L_X=lx,
        L_X_MINI=lx_mini,
        SIGMA_STAR=sigma_star,
        SIGMA_SFR_LIM=sigma_sfr_lim,
        SIGMA_SFR_INDEX=sigma_sfr_index,
        SIGMA_LX=sigma_lx,
        t_STAR=0.5,
        M_TURN=6.0,
    )
    # NOTE: Not using upper turnover, this test should be extended
    fo = FlagOptions(
        USE_MINI_HALOS=True,
        USE_HALO_FIELD=True,
        FIXED_HALO_GRIDS=False,
        HALO_STOCHASTICITY=True,
        USE_UPPER_STELLAR_TURNOVER=False,
    )

    lib.Broadcast_struct_global_all(up(), cp(), ap(), fo())
    mturn_acg = np.maximum(lib.atomic_cooling_threshold(redshift), 10**ap.M_TURN)
    mturn_mcg = (
        10**ap.M_TURN
    )  # I don't want to test the LW or reionisation feedback here

    print(f"turnovers [{mturn_acg},{mturn_mcg}]")
    print(f"z={redshift} th = {1/cp.cosmo.H(redshift).to('s-1').value}")

    # setup the halo masses to test
    halo_masses = np.array([1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
    halo_rng = np.ones_like(
        halo_masses
    )  # we set the RNG to one sigma above for the test

    # independently calculate properties for the halos from our scaling relations
    exp_fstar = (10**f_star10) * (halo_masses / 1e10) ** a_star
    exp_fesc = np.minimum((10**f_esc10) * (halo_masses / 1e10) ** a_esc, 1)
    exp_fstar_mini = (10**f_star7) * (halo_masses / 1e7) ** a_star_mini
    exp_fesc_mini = np.minimum((10**f_esc7) * (halo_masses / 1e7) ** a_esc, 1)
    b_r = cp.OMb / cp.OMm
    acg_turnover = np.exp(-mturn_acg / halo_masses)
    mcg_turnovers = np.exp(-halo_masses / mturn_acg) * np.exp(-mturn_mcg / halo_masses)

    expected_hm = halo_masses
    expected_sm = (
        np.minimum(exp_fstar * np.exp(halo_rng * sigma_star) * acg_turnover, 1)
        * halo_masses
        * b_r
    )
    expected_sm_mini = (
        np.minimum(exp_fstar_mini * np.exp(halo_rng * sigma_star) * mcg_turnovers, 1)
        * halo_masses
        * b_r
    )

    sigma_sfr = (
        sigma_sfr_index * np.log10((expected_sm + expected_sm_mini) / 1e10)
        + sigma_sfr_lim
    )
    sigma_sfr = np.maximum(sigma_sfr, sigma_sfr_lim)
    expected_sfr = (
        expected_sm
        / t_star
        * cp.cosmo.H(redshift).to("s-1").value
        * np.exp(halo_rng * sigma_sfr)
    )
    expected_sfr_mini = (
        expected_sm_mini
        / t_star
        * cp.cosmo.H(redshift).to("s-1").value
        * np.exp(halo_rng * sigma_sfr)
    )

    expected_nion = (
        expected_sm * exp_fesc * global_params.Pop2_ion
        + expected_sm_mini * exp_fesc_mini * global_params.Pop3_ion
    )
    expected_wsfr = (
        expected_sfr * exp_fesc * global_params.Pop2_ion
        + expected_sfr_mini * exp_fesc_mini * global_params.Pop3_ion
    )

    # NOTE: These are currently hardcoded in the backend, changes will result in this test failing
    s_per_yr = 365.25 * 60 * 60 * 24
    expected_metals = (
        1.28825e10 * ((expected_sfr + expected_sfr_mini) * s_per_yr) ** 0.56
    )  # SM denominator
    expected_metals = (
        0.296
        * (
            (1 + ((expected_sm + expected_sm_mini) / expected_metals) ** (-2.1))
            ** -0.148
        )
        * 10 ** (-0.056 * redshift + 0.064)
    )

    expected_xray = (
        (expected_sfr * s_per_yr) ** 1.03
        * expected_metals**-0.64
        * np.exp(halo_rng * sigma_lx)
        * 10**lx
    )
    expected_xray += (
        (expected_sfr_mini * s_per_yr) ** 1.03
        * expected_metals**-0.64
        * np.exp(halo_rng * sigma_lx)
        * 10**lx_mini
    )

    # HACK: Make the fake halo list
    fake_pthalos = PerturbHaloField(
        redshift=redshift,
        buffer_size=halo_masses.size,
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
    )
    fake_pthalos()  # initialise memory
    fake_pthalos.halo_masses = halo_masses.astype("f4")
    fake_pthalos.halo_corods = np.zeros(halo_masses.size * 3).astype("i4")
    fake_pthalos.star_rng = halo_rng.astype("f4")
    fake_pthalos.sfr_rng = halo_rng.astype("f4")
    fake_pthalos.xray_rng = halo_rng.astype("f4")
    fake_pthalos.n_halos = halo_masses.size

    # single element zero array to act as the grids (vcb, J_21_LW, z_reion, Gamma12)
    zero_array = ffi.cast("float *", np.zeros(1).ctypes.data)

    out_buffer = np.zeros(12 * halo_masses.size).astype("f4")
    lib.test_halo_props(
        redshift,
        up(),
        cp(),
        ap(),
        fo(),
        zero_array,
        zero_array,
        zero_array,
        zero_array,
        fake_pthalos(),
        ffi.cast("float *", out_buffer.ctypes.data),
    )

    np.testing.assert_allclose(expected_hm, out_buffer[0::12], atol=1e0)

    np.testing.assert_allclose(mturn_acg, out_buffer[8::12], atol=1e0)
    np.testing.assert_allclose(mturn_mcg, out_buffer[9::12], atol=1e0)
    np.testing.assert_allclose(0.0, out_buffer[10::12], atol=1e0)  # no reion feedback

    np.testing.assert_allclose(expected_sm, out_buffer[1::12], atol=1e0)
    np.testing.assert_allclose(expected_sm_mini, out_buffer[6::12], atol=1e0)

    # hubble differences between the two codes make % level changes TODO: change hubble to double precision in backend
    np.testing.assert_allclose(expected_sfr, out_buffer[2::12], rtol=5e-2, atol=1e-20)
    np.testing.assert_allclose(
        expected_sfr_mini, out_buffer[7::12], rtol=5e-2, atol=1e-20
    )

    np.testing.assert_allclose(expected_metals, out_buffer[11::12], rtol=1e-3)
    np.testing.assert_allclose(
        expected_xray, out_buffer[3::12].astype(float) * 1e38, rtol=5e-2
    )

    np.testing.assert_allclose(expected_nion, out_buffer[4::12])
    np.testing.assert_allclose(expected_wsfr, out_buffer[5::12], rtol=5e-2)


def plot_sampler_comparison(
    bin_edges, exp_N, exp_M, N_array, M_array, exp_mf, mf_out, title, plt
):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    axst = [axs[0].twinx(), axs[1].twiny()]

    # total M axis
    axs[1].set_xlabel("M")
    # total N axis
    axst[1].set_ylabel("N")

    # mass function axis
    axs[0].set_title(title)
    axs[0].set_ylim([1e-2, 1e4])
    axs[0].set_xlim([bin_edges[0], bin_edges[-1]])
    axs[0].set_yscale("log")
    axs[0].set_ylabel("dn/dlnM")

    # ratio axis
    axst[0].set_ylim([1e-1, 1e1])
    axst[0].set_yscale("log")
    axst[0].set_ylabel("ratio")

    for ax in axs:
        ax.grid()
        ax.set_xscale("log")

    # log-spaced bins
    dlnm = np.log(bin_edges[1:]) - np.log(bin_edges[:-1])
    bin_centres = (bin_edges[:-1] * np.exp(dlnm / 2)).astype("f4")
    edges_n = np.arange(N_array.max() + 1) - 0.5
    centres_n = (edges_n[:-1] + edges_n[1:]) / 2

    hist_n, _ = np.histogram(N_array, edges_n)
    p_n = hist_n / N_array.size
    hist_m, _ = np.histogram(M_array, bin_edges)
    p_m = hist_m / M_array.size / dlnm  # p(lnM)

    axst[0].loglog(bin_centres, mf_out / exp_mf, color="r", linewidth=2)
    axst[0].loglog(
        bin_centres, np.ones_like(exp_mf), color="r", linestyle=":", linewidth=1
    )

    axs[0].loglog(bin_centres, mf_out, color="k", linewidth=2, label="Sample")
    axs[0].loglog(
        bin_centres, exp_mf, color="k", linestyle=":", linewidth=1, label="Expected"
    )

    axs[1].semilogx(bin_centres, p_m / p_m.max(), color="k", linewidth=2)
    axs[1].axvline(exp_M[0], color="k", linestyle=":", linewidth=2)
    axs[1].axvline(M_array.mean(), color="k", linestyle="-", linewidth=1)
    axs[1].set_xlim(M_array.min() / 10, M_array.max() * 10)

    axst[1].plot(centres_n, p_n / p_n.max(), color="r", linewidth=2)
    axst[1].axvline(exp_N[0], color="r", linestyle=":", linewidth=2)
    axst[1].axvline(N_array.mean(), color="r", linestyle="-", linewidth=1)
    axst[1].set_xlim(N_array.min() - 1, N_array.max() + 1)
