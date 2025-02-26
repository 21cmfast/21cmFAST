import pytest

import matplotlib as mpl
import numpy as np
from astropy import units as u

from py21cmfast import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    PerturbHaloField,
    UserParams,
    global_params,
)
import py21cmfast.c_21cmfast as lib
from py21cmfast.wrapper import cfuncs as cf

from . import produce_integration_test_data as prd
from . import test_c_interpolation_tables as cint
from .test_c_interpolation_tables import print_failure_stats

RELATIVE_TOLERANCE = 1e-1

options_hmf = list(cint.OPTIONS_HMF.keys())

options_delta = [-0.9, -0.5, 0, 1, 1.4]  # cell densities to draw samples from
options_log10mass = [9, 10, 11, 12, 13]  # halo masses to draw samples from


@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("from_cat", ["cat", "grid"])
@pytest.mark.parametrize("cond", range(len(options_delta)))
def test_sampler(name, cond, from_cat, plt):
    redshift, kwargs = cint.OPTIONS_HMF[name]
    redshift = 8
    opts = prd.get_all_options(redshift, **kwargs)
    up = opts["user_params"].clone(SAMPLER_MIN_MASS=2e8)
    cp = opts["cosmo_params"]
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    from_cat = "cat" in from_cat

    n_cond = 15000
    if from_cat:
        mass = 10 ** options_log10mass[cond]
        cond = mass
        z_desc = (1 + redshift) / global_params.ZPRIME_STEP_FACTOR - 1
        delta = None
    else:
        mass = (
            (
                cp.cosmo.critical_density(0)
                * cp.OMm
                * u.Mpc**3
                * (up.BOX_LEN / up.HII_DIM) ** 3
            )
            .to("M_sun")
            .value
        )
        z_desc = None
        cond = options_delta[cond]
        delta = cond

    sample_dict = cf.halo_sample_test(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        redshift=redshift,
        from_cat=from_cat,
        cond_array=np.full(n_cond, cond),
        seed=987,
    )

    # set up histogram
    l10min = np.log10(up.SAMPLER_MIN_MASS)
    l10max = np.log10(mass * 1.01)
    edges = np.logspace(l10min, l10max, num=int(10 * (l10max - l10min)))
    bin_minima = edges[:-1]
    bin_maxima = edges[1:]
    dlnm = np.log(bin_maxima) - np.log(bin_minima)

    # get CMF integrals in the same bins
    binned_cmf = cf.get_cmf_integral(
        user_params=up,
        cosmo_params=cp,
        astro_params=ap,
        flag_options=fo,
        M_min=bin_minima,
        M_max=bin_maxima,
        M_cond=mass,
        redshift=redshift,
        delta=delta,
        z_desc=z_desc,
    )

    hist, _ = np.histogram(sample_dict["halo_masses"], edges)

    mass_dens = cp.cosmo.Om0 * \
        cp.cosmo.critical_density(0).to("Mpc-3 M_sun").value
    volume_total_m = mass * n_cond / mass_dens
    mf_out = hist / volume_total_m / dlnm
    binned_cmf = binned_cmf / dlnm * mass_dens

    one_in_box = 1 / volume_total_m / dlnm

    if plt == mpl.pyplot:
        plot_sampler_comparison(
            edges,
            sample_dict["expected_progenitors"],
            sample_dict["expected_progenitor_mass"],
            sample_dict["n_progenitors"],
            sample_dict["progenitor_mass"],
            binned_cmf,
            mf_out,
            one_in_box,
            f"mass = {mass:.2e}" if from_cat else f"delta = {delta:.2e}",
            plt,
        )

    np.testing.assert_allclose(
        sample_dict["n_progenitors"].mean(),
        sample_dict["expected_progenitors"][0],
        atol=1,
        rtol=RELATIVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        sample_dict["progenitor_mass"].mean(),
        sample_dict["expected_progenitor_mass"][0],
        atol=up.SAMPLER_MIN_MASS,
        rtol=RELATIVE_TOLERANCE,
    )

    # The histograms get inaccurate when the volume is too small
    # so only compare when we expect real halos
    if sample_dict["expected_progenitor_mass"][0] > up.SAMPLER_MIN_MASS:
        sel_compare_bins = edges[1:] < (0.9 * mass)

        print_failure_stats(
            mf_out[sel_compare_bins],
            binned_cmf[sel_compare_bins],
            [edges[:-1][sel_compare_bins]],
            one_in_box.min(),
            5e-1,
            "binned_cmf",
        )
        # this is a wide tolerance since running enough
        # samples to converge is too slow
        np.testing.assert_allclose(
            mf_out[sel_compare_bins],
            binned_cmf[sel_compare_bins],
            atol=2 * one_in_box.min(),  # 2 halo tolerance
            rtol=5e-1,  # 50%
        )


# NOTE: this test is pretty circular. The only way I think I can test the scaling relations are to
#   calculate them in the backend and re-write them in the test for a few masses. This means that
#   changes to any scaling relation model will result in a test fail
@pytest.mark.xfail(reason="robust tests for scaling relations not yet implemented")
def test_halo_scaling_relations(ic, default_input_struct):
    # specify parameters to use for this test
    redshift = 10.0
    opts = prd.get_all_options(redshift)
    ap = opts["astro_params"]
    fo = opts["flag_options"]

    # setup the halo masses to test
    halo_mass_vals = np.array([1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
    n_halo_per_mass = 10000
    halo_masses = np.broadcast_to(
        halo_mass_vals[:, None], (halo_mass_vals.size, n_halo_per_mass)
    ).flatten()
    halo_rng = np.random.normal(size=n_halo_per_mass * halo_mass_vals.size)

    # (n_halo*n_mass*n_prop) --> (n_prop,n_mass,n_halo)

    # mass,star,sfr,xray,nion,wsfr,starmini,sfrmini,mturna,mturnm,mturnr,Z
    out_dict = cf.convert_halo_properties(
        ics=ic,
        redshift=redshift,
        astro_params=ap,
        flag_options=fo,
        halo_masses=halo_masses,
        halo_rng=halo_rng,
    )

    halo_mass_out = out_dict["halo_mass"].reshape(
        (halo_mass_vals.size, n_halo_per_mass)
    )
    halo_stars_out = out_dict["halo_stars"].reshape(
        (halo_mass_vals.size, n_halo_per_mass)
    )
    halo_sfr_out = out_dict["halo_sfr"].reshape(
        (halo_mass_vals.size, n_halo_per_mass))
    halo_xray_out = out_dict["halo_xray"].reshape(
        (halo_mass_vals.size, n_halo_per_mass)
    )

    # assuming same value for all halos
    mturn_acg = out_dict["mturn_a"][0]

    exp_SHMR = (
        (
            (10**ap.F_STAR10)
            * (halo_mass_vals / 1e10) ** ap.ALPHA_STAR
            * np.exp(-mturn_acg / halo_mass_vals)
        )
        * ic.cosmo_params.OMb
        / ic.cosmo_params.OMm
    )
    sim_SHMR = halo_stars_out / halo_mass_out
    sel_stars = exp_SHMR > 1e-10
    np.testing.assert_allclose(
        exp_SHMR, sim_SHMR.mean(axis=1), atol=1e-10, rtol=1e-1)
    np.testing.assert_allclose(
        ap.SIGMA_STAR, np.log10(sim_SHMR).std(axis=1)[sel_stars], rtol=1e-1
    )

    exp_SSFR = ic.cosmo_params.cosmo.H(redshift).to("s-1").value / (ap.t_STAR)
    sim_SSFR = halo_sfr_out / halo_stars_out
    np.testing.assert_allclose(exp_SSFR, sim_SSFR.mean(axis=1)[
                               sel_stars], rtol=1e-1)
    np.testing.assert_allclose(
        ap.SIGMA_SFR_LIM,
        np.log10(sim_SSFR).std(axis=1)[sel_stars],
    )  # WRONG

    exp_LX = 10 ** (ap.L_X)  # low-z approx
    sim_LX = halo_xray_out / halo_sfr_out
    np.testing.assert_allclose(exp_LX, sim_LX.mean(axis=1), rtol=1e-1)
    np.testing.assert_allclose(ap.SIGMA_LX, sim_LX.std(axis=1), rtol=1e-1)


def plot_sampler_comparison(
    bin_edges, exp_N, exp_M, N_array, M_array, exp_mf, mf_out, one_halo, title, plt
):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    axst = [axs[0].twinx(), axs[1].twiny()]

    # total M axis
    axs[1].set_xlabel("M")
    # total N axis
    axst[1].set_ylabel("N")

    # mass function axis
    axs[0].set_title(title)
    axs[0].set_ylim([1e-6, 1e2])
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
    edges_n = np.linspace(0, max(N_array.max(), 1),
                          min(100, max(N_array.max(), 1) + 1))
    centres_n = (edges_n[:-1] + edges_n[1:]) / 2

    hist_n, _ = np.histogram(N_array, edges_n)
    p_n = hist_n / N_array.size
    hist_m, _ = np.histogram(M_array, bin_edges)
    p_m = hist_m / M_array.size / dlnm  # p(lnM)

    axst[0].loglog(bin_centres, mf_out / exp_mf, color="r", linewidth=2)
    axst[0].loglog(
        bin_centres, np.ones_like(exp_mf), color="r", linestyle=":", linewidth=1
    )

    axs[0].loglog(bin_centres, mf_out, color="k", linewidth=3, label="Sample")
    axs[0].loglog(
        bin_centres, exp_mf, color="k", linestyle=":", linewidth=1, label="Expected"
    )
    axs[0].loglog(
        bin_centres, one_halo, color="k", linestyle="--", linewidth=2, label="Expected"
    )

    axs[1].semilogx(bin_centres, p_m / p_m.max(), color="k", linewidth=2)
    axs[1].axvline(exp_M[0], color="k", linestyle=":", linewidth=2)
    axs[1].axvline(M_array.mean(), color="k", linestyle="-", linewidth=1)
    axs[1].set_xlim(M_array.min() / 10, M_array.max() * 10)

    axst[1].plot(centres_n, p_n / p_n.max(), color="r", linewidth=2)
    axst[1].axvline(exp_N[0], color="r", linestyle=":", linewidth=2)
    axst[1].axvline(N_array.mean(), color="r", linestyle="-", linewidth=1)
    axst[1].set_xlim(N_array.min() - 1, N_array.max() + 1)
