"""Tests of the halo sampler."""

import matplotlib as mpl
import numpy as np
import pytest
from astropy import units as u

from py21cmfast import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    PerturbHaloField,
    UserParams,
)
from py21cmfast.c_21cmfast import ffi, lib
from py21cmfast.wrapper import cfuncs as cf

from . import produce_integration_test_data as prd
from . import test_c_interpolation_tables as cint
from .test_c_interpolation_tables import print_failure_stats

RELATIVE_TOLERANCE = 1e-1

options_hmf = list(cint.OPTIONS_HMF.keys())

options_delta = [-0.9, -0.5, 0, 1, 1.4]  # cell densities to draw samples from
options_mass = [1e9, 1e10, 1e11, 1e12, 1e13]  # halo masses to draw samples from


@pytest.mark.parametrize("name", options_hmf)
@pytest.mark.parametrize("cond_type", ["cat", "grid"])
@pytest.mark.parametrize("cond", range(len(options_delta)))
def test_sampler(name, cond, cond_type, plt):
    redshift, kwargs = cint.OPTIONS_HMF[name]
    inputs = prd.get_all_options_struct(redshift, **kwargs)["inputs"]
    inputs = inputs.evolve_input_structs(
        SAMPLER_MIN_MASS=min(options_mass) / 2,
    )

    from_cat = "cat" in cond_type
    z_desc = (
        (redshift + 1) / inputs.user_params.ZPRIME_STEP_FACTOR - 1 if from_cat else None
    )

    n_cond = 15000
    # Testing large conditions or dense cells can get quite a lot of halos
    #   so we have an expanded buffer
    buffer_size = 500 * n_cond
    cond_val = options_mass[cond] if from_cat else options_delta[cond]
    cond_array = np.full(n_cond, cond_val)

    sample_dict = cf.halo_sample_test(
        inputs=inputs,
        redshift=redshift,
        redshift_prev=z_desc,
        cond_array=cond_array,
        buffer_size=buffer_size,
    )

    # set up histogram
    l10min = np.log10(inputs.user_params.SAMPLER_MIN_MASS)
    l10max = np.log10(1e14)
    edges = np.linspace(l10min, l10max, num=int(10 * (l10max - l10min))) * np.log(10)
    bin_minima = edges[:-1]
    bin_maxima = edges[1:]
    dlnm = bin_maxima - bin_minima

    # get CMF integrals in the same bins
    binned_cmf = cf.integrate_chmf_interval(
        inputs=inputs,
        lnM_lower=bin_minima,
        lnM_upper=bin_maxima,
        redshift=redshift,
        cond_values=np.array([cond_val]),
        redshift_prev=z_desc,
    ).squeeze()

    hist, _ = np.histogram(np.log(sample_dict["halo_masses"]), edges)

    mass = cond_val if from_cat else cf.get_condition_mass(inputs, "cell")
    mass_dens = (
        inputs.cosmo_params.cosmo.Om0
        * inputs.cosmo_params.cosmo.critical_density(0).to("Mpc-3 M_sun").value
    )
    volume_total_m = mass * n_cond / mass_dens
    one_in_box = 1 / volume_total_m / dlnm

    mf_out = hist / volume_total_m / dlnm
    binned_cmf = binned_cmf * n_cond / volume_total_m / dlnm

    if plt == mpl.pyplot:
        plot_sampler_comparison(
            np.exp(edges),
            sample_dict["expected_progenitors"],
            sample_dict["expected_progenitor_mass"],
            sample_dict["n_progenitors"],
            sample_dict["progenitor_mass"],
            binned_cmf,
            mf_out,
            one_in_box,
            f"mass = {cond_val:.2e}" if from_cat else f"delta = {cond_val:.2e}",
            plt,
        )

    # test the number of progenitors per condition
    np.testing.assert_allclose(
        sample_dict["n_progenitors"].mean(),
        sample_dict["expected_progenitors"][0],
        atol=1,
        rtol=RELATIVE_TOLERANCE,
    )
    # test the total mass of progenitors per condition
    np.testing.assert_allclose(
        sample_dict["progenitor_mass"].mean(),
        sample_dict["expected_progenitor_mass"][0],
        atol=inputs.user_params.SAMPLER_MIN_MASS,
        rtol=RELATIVE_TOLERANCE,
    )

    print_failure_stats(
        mf_out,
        binned_cmf,
        [edges[:-1]],
        one_in_box.min(),
        5e-1,
        "binned_cmf",
    )

    np.testing.assert_allclose(
        mf_out,
        binned_cmf,
        atol=2 * one_in_box.min(),  # 2 halo tolerance
        rtol=5e-1,  # 50% tolerance for stochasticity in low-n bins
    )


# NOTE: this test is pretty circular. The only way I think I can test the scaling relations are to
#   calculate them in the backend and re-write them in the test for a few masses. This means that
#   changes to any scaling relation model will result in a test fail
def test_halo_prop_sampling(default_input_struct):
    # specify parameters to use for this test
    redshift = 10.0

    # setup the halo masses to test
    halo_mass_vals = np.array([1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12])
    halo_rng = np.array([-3, -2, -1, 0, 1, 2, 3])
    halo_masses = np.broadcast_to(
        halo_mass_vals[:, None], (halo_mass_vals.size, halo_rng.size)
    ).flatten()

    inputs = default_input_struct.evolve_input_structs(
        USE_UPPER_STELLAR_TURNOVER=False,
    )  # evolve if needed
    out_dict = cf.convert_halo_properties(
        redshift=redshift,
        inputs=default_input_struct,
        halo_masses=halo_masses,
        star_rng=halo_rng,  # testing the diagonal of the RNG
        sfr_rng=halo_rng,
        xray_rng=halo_rng,
    )

    halo_mass_out = out_dict["halo_mass"].reshape((halo_mass_vals.size, halo_rng.size))
    halo_stars_out = out_dict["halo_stars"].reshape(
        (halo_mass_vals.size, halo_rng.size)
    )
    halo_sfr_out = out_dict["halo_sfr"].reshape((halo_mass_vals.size, halo_rng.size))
    halo_xray_out = out_dict["halo_xray"].reshape((halo_mass_vals.size, halo_rng.size))

    # assuming same value for all halos
    mturn_acg = out_dict["mturn_a"][0]
    ap_c = inputs.astro_params.cdict

    exp_SHMR = (
        (
            ap_c.F_STAR10
            * (halo_mass_vals[:, None] / 1e10) ** ap_c.ALPHA_STAR
            * np.exp(
                -mturn_acg / halo_mass_vals
                + halo_rng[None, :] * ap_c.SIGMA_STAR
                - ap_c.SIGMA_STAR**2 / 2
            )
        )
        * inputs.cosmo_params.OMb
        / inputs.cosmo_params.OMm
    )
    sim_SHMR = halo_stars_out / halo_mass_out
    np.testing.assert_allclose(exp_SHMR, sim_SHMR)

    sigma_SSFR = ap_c.SIGMA_SFR_LIM + ap_c.SIGMA_SFR_INDEX * (
        exp_SHMR * halo_mass_vals / 1e10
    )
    sigma_SSFR = max(sigma_SSFR, ap_c.SIGMA_SFR_LIM)
    exp_SSFR = (
        inputs.cosmo_params.cosmo.H(redshift).to("s-1").value
        / (ap_c.t_STAR)
        * np.exp(halo_rng[None, :] * sigma_SSFR - sigma_SSFR**2 / 2)
    )
    sim_SSFR = halo_sfr_out / halo_stars_out
    np.testing.assert_allclose(exp_SSFR, sim_SSFR.mean(axis=1))

    exp_LX = ap_c.L_X * np.exp(+halo_rng[None, :] * ap_c.SIGMA_LX - ap_c.LX**2 / 2)
    sim_LX = halo_xray_out / halo_sfr_out
    np.testing.assert_allclose(exp_LX, sim_LX)


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
    axs[0].set_ylim([one_halo[0] / 2, exp_mf.max()])
    axs[0].set_yscale("log")
    axs[0].set_ylabel("dn/dlnM")
    axs[0].set_xlabel("M")

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
    edges_n = np.arange(0, max(N_array.max(), 1), 1)
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
        bin_centres, one_halo, color="k", linestyle="--", linewidth=2, label="One Halo"
    )
    axs[0].legend()

    axs[1].semilogx(bin_centres, p_m / p_m.max(), color="k", linewidth=2)
    axs[1].axvline(exp_M[0], color="k", linestyle=":", linewidth=2)
    axs[1].axvline(M_array.mean(), color="k", linestyle="-", linewidth=1)
    axs[1].set_xlim(M_array.min() / 10, M_array.max() * 10)

    axst[1].plot(centres_n, p_n / p_n.max(), color="r", linewidth=2)
    axst[1].axvline(exp_N[0], color="r", linestyle=":", linewidth=2)
    axst[1].axvline(N_array.mean(), color="r", linestyle="-", linewidth=1)
    axst[1].set_xlim(N_array.min() - 1, N_array.max() + 1)
