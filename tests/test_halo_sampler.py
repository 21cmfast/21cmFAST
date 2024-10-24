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
options_mass = [9, 10, 11, 12]  # halo masses to draw samples from


@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("mass", options_mass)
def test_sampler_from_catalog(name, mass, plt):
    redshift, kwargs = cint.OPTIONS_HMF[name]
    opts = prd.get_all_options(redshift, **kwargs)

    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    mass = 10**mass

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

    n_cond = 2000

    z = 6.0
    z_prev = 5.8
    growth_prev = lib.dicke(z_prev)
    growthf = lib.dicke(z)

    sigma_cond_m = lib.sigma_z0(mass)
    delta_cond_m = (
        lib.get_delta_crit(up.cdict["HMF"], sigma_cond_m, growth_prev)
        * growthf
        / growth_prev
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
        up.cstruct,
        cp.cstruct,
        ap.cstruct,
        fo.cstruct,
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

    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]
    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    lib.init_ps()
    lib.initialiseSigmaMInterpTable(
        global_params.M_MIN_INTEGRAL, global_params.M_MAX_INTEGRAL
    )

    n_cond = 2000

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
        up.cstruct,
        cp.cstruct,
        ap.cstruct,
        fo.cstruct,
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
    redshift = 10.0
    opts = prd.get_all_options(redshift, {})

    up = opts["user_params"]
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]
    lib.Broadcast_struct_global_all(up.cstruct, cp.cstruct, ap.cstruct, fo.cstruct)

    mturn_acg = np.maximum(lib.atomic_cooling_threshold(redshift), 10**ap.M_TURN)
    # mturn_mcg = 10**ap.M_TURN
    print(f"z={redshift} th = {1/cp.cosmo.H(redshift).to('s-1').value}")

    # setup the halo masses to test
    halo_mass_vals = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12]
    n_halo_per_mass = 1000
    halo_masses = np.array(
        [n_halo_per_mass * [val] for val in halo_mass_vals]
    ).flatten()
    halo_rng = np.random.normal(size=n_halo_per_mass * len(halo_mass_vals))

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
        up.cstruct,
        cp.cstruct,
        ap.cstruct,
        fo.cstruct,
        zero_array,
        zero_array,
        zero_array,
        zero_array,
        fake_pthalos(),
        ffi.cast("float *", out_buffer.ctypes.data),
    )

    # (n_halo*n_mass*n_prop) --> (n_prop,n_mass,n_halo)

    # mass,star,sfr,xray,nion,wsfr,starmini,sfrmini,mturna,mturnm,mturnr,Z
    out_buffer = out_buffer.reshape(len(halo_mass_vals), n_halo_per_mass, 12)

    exp_SHMR = (
        (10**ap.F_STAR10)
        * halo_mass_vals**ap.ALPHA_STAR
        * np.exp(-mturn_acg / halo_mass_vals)
    )
    sim_SHMR = out_buffer[:, :, 1] / out_buffer[:, :, 0]
    np.testing.assert_allclose(exp_SHMR, sim_SHMR.mean(axis=1), rtol=1e-1)
    np.testing.assert_allclose(ap.SIGMA_STAR, sim_SHMR.std(axis=1), rtol=1e-1)

    exp_SSFR = cp.cosmo.H(redshift).to("s").value / (ap.t_STAR)
    sim_SSFR = out_buffer[:, :, 2] / out_buffer[:, :, 1]
    np.testing.assert_allclose(exp_SSFR, sim_SSFR.mean(axis=1), rtol=1e-1)
    np.testing.assert_allclose(
        ap.SIGMA_SFR_LIM, sim_SSFR.std(axis=1), rtol=1e-1
    )  # WRONG

    exp_LX = 10 ** (ap.L_X)  # low-z approx
    sim_LX = out_buffer[:, :, 3] / out_buffer[:, :, 2]
    np.testing.assert_allclose(exp_LX, sim_LX.mean(axis=1), rtol=1e-1)
    np.testing.assert_allclose(ap.SIGMA_LX, sim_LX.std(axis=1), rtol=1e-1)


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
