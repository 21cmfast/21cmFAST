"""Test the wrapper functions which access the C-backend, but not though an OutputStruct compute() method."""

import matplotlib as mpl
import numpy as np
import pytest
from hmf import MassFunction

import py21cmfast as p21c
from py21cmfast.wrapper import cfuncs as cf


def test_run_lf():
    inputs = p21c.InputParameters(random_seed=9)
    muv, mhalo, lf = p21c.compute_luminosity_function(
        inputs=inputs, redshifts=[7, 8, 9], nbins=100
    )
    assert np.all(lf[~np.isnan(lf)] > -30)
    assert lf.shape == (3, 100)

    # Check that memory is in-tact and a second run also works:
    muv, mhalo, lf2 = p21c.compute_luminosity_function(
        inputs=inputs, redshifts=[7, 8, 9], nbins=100
    )
    assert lf2.shape == (3, 100)
    assert np.allclose(lf2[~np.isnan(lf2)], lf[~np.isnan(lf)])

    inputs = inputs.from_template("mini", random_seed=9)

    muv_minih, mhalo_minih, lf_minih = p21c.compute_luminosity_function(
        redshifts=[7, 8, 9],
        nbins=100,
        component="mcg",
        inputs=inputs,
        mturnovers=[7.0, 7.0, 7.0],
        mturnovers_mini=[5.0, 5.0, 5.0],
    )
    assert np.all(lf_minih[~np.isnan(lf_minih)] > -30)
    assert lf_minih.shape == (3, 100)


def test_run_tau():
    inputs = p21c.InputParameters(random_seed=9)
    tau = p21c.compute_tau(
        redshifts=[7, 8, 9],
        global_xHI=[0.1, 0.2, 0.3],
        inputs=inputs,
        z_re_HeII=3.0,
    )

    assert tau


def test_bad_integral_inputs(default_input_struct):
    # make arrays with different shapes
    redshifts = np.linspace(6, 35, num=20)
    lnM_base = np.linspace(7, 13, num=30)
    densities = np.linspace(-1, 3, num=25)

    with pytest.raises(ValueError, match="the shapes of"):
        cf.integrate_chmf_interval(
            inputs=default_input_struct,
            redshift=redshifts[0],
            lnm_lower=lnM_base,
            lnm_upper=lnM_base[1:],
            cond_values=densities,
        )

    with pytest.raises(ValueError, match="the shapes of"):
        cf.evaluate_inverse_table(
            inputs=default_input_struct,
            redshift=redshifts[0],
            cond_array=densities,
            probabilities=np.ones(len(densities) - 1),
        )

    with pytest.raises(ValueError, match="the shapes of"):
        cf.evaluate_SFRD_z(
            inputs=default_input_struct,
            redshifts=redshifts,
            log10mturns=lnM_base,
        )

    with pytest.raises(ValueError, match="the shapes of"):
        cf.evaluate_Nion_z(
            inputs=default_input_struct,
            redshifts=redshifts,
            log10mturns=lnM_base,
        )

    with pytest.raises(ValueError, match="the shapes of"):
        cf.evaluate_SFRD_cond(
            inputs=default_input_struct,
            redshift=redshifts[0],
            densities=densities,
            log10mturns=lnM_base,
            radius=10.0,
        )

    with pytest.raises(ValueError, match="the shapes of"):
        cf.evaluate_Nion_cond(
            inputs=default_input_struct,
            redshift=redshifts[0],
            densities=densities,
            l10mturns_acg=lnM_base,
            l10mturns_mcg=lnM_base,
            radius=10.0,
        )

    with pytest.raises(ValueError, match="the shapes of"):
        cf.evaluate_Xray_cond(
            inputs=default_input_struct,
            redshift=redshifts[0],
            densities=densities,
            log10mturns=lnM_base,
            radius=10.0,
        )

    with pytest.raises(
        ValueError, match="Halo masses and rng shapes must be identical."
    ):
        cf.convert_halo_properties(
            inputs=default_input_struct,
            redshift=redshifts[0],
            halo_masses=np.zeros(10),
            star_rng=np.zeros(10),
            sfr_rng=np.zeros(10),
            xray_rng=np.zeros(11),
        )


@pytest.mark.parametrize("hmf_model", ["PS", "ST"])
@pytest.mark.parametrize("ps_model", ["EH", "BBKS", "EFSTATHIOU"])
@pytest.mark.xfail(reason="pending proper comparison between 21cmFAST and hmf")
def test_matterfield_statistics(default_input_struct, hmf_model, ps_model, plt):
    redshift = 8.0
    hmf_map = {
        "PS": "PS",
        "ST": "Jenkins",
    }
    transfer_map = {
        "EH": "EH",
        "BBKS": "BBKS",
        "EFSTATHIOU": "BondEfs",
    }

    if ps_model == "BBKS":
        t_params = {"use_sugiyama_baryons": True, "use_liddle_baryons": False}
    elif ps_model == "EFSTATHIOU":
        t_params = {"nu": 1.13}
    else:
        t_params = {}

    comparison_mf = MassFunction(
        z=redshift,
        Mmin=7,
        Mmax=16,
        hmf_model=hmf_map[hmf_model],
        transfer_model=transfer_map[ps_model],
        transfer_params=t_params,
    )

    inputs = default_input_struct.clone(
        cosmo_params=p21c.CosmoParams.from_astropy(
            comparison_mf.cosmo,
            SIGMA_8=comparison_mf.sigma_8,
            POWER_INDEX=comparison_mf.n,
        ),
    ).evolve_input_structs(
        POWER_SPECTRUM=ps_model,
        HMF=hmf_model,
    )
    h = inputs.cosmo_params.cosmo.h

    hmf_vals = cf.return_uhmf_value(
        inputs=inputs,
        redshift=redshift,
        mass_values=comparison_mf.m / h,
    )

    sigma_vals, _ = cf.evaluate_sigma(
        inputs=inputs,
        masses=comparison_mf.m / h,
    )

    power_vals = cf.get_matter_power_values(
        inputs=inputs,
        k_values=comparison_mf.k * h,
    )

    mass_dens = (
        inputs.cosmo_params.cosmo.critical_density(0).to("M_sun Mpc^-3").value
        * inputs.cosmo_params.cosmo.Om0
    )

    if plt == mpl.pyplot:
        make_matterfield_comparison_plot(
            comparison_mf.m / h,
            comparison_mf.k * h,
            [
                comparison_mf.dndlnm * (h**3),
                comparison_mf._sigma_0,
                comparison_mf._power0 / (h**3),
            ],
            [hmf_vals * mass_dens, sigma_vals, power_vals],
            plt,
        )

    np.testing.assert_allclose(power_vals, comparison_mf.power / (h**3), rtol=1e-3)
    np.testing.assert_allclose(sigma_vals, comparison_mf.sigma, rtol=1e-3)
    np.testing.assert_allclose(
        mass_dens * hmf_vals, comparison_mf.dndlnm * (h**3), rtol=1e-3
    )


@pytest.mark.parametrize("hmf_model", ["PS", "ST"])
@pytest.mark.parametrize("ps_model", ["EH", "BBKS", "EFSTATHIOU"])
def test_hmf_runs(default_input_struct, hmf_model, ps_model):
    mass_range = np.logspace(7, 12, num=64)
    redshift = 8.0

    inputs = default_input_struct.evolve_input_structs(
        POWER_SPECTRUM=ps_model,
        HMF=hmf_model,
    )

    hmf_vals = cf.return_uhmf_value(
        inputs=inputs,
        redshift=redshift,
        mass_values=mass_range,
    )

    assert hmf_vals.shape == (len(mass_range),)
    assert np.all(~np.isnan(hmf_vals))


@pytest.mark.parametrize("hmf_model", ["PS", "ST"])
@pytest.mark.parametrize("ps_model", ["EH", "BBKS", "EFSTATHIOU"])
def test_chmf_runs(default_input_struct, hmf_model, ps_model):
    delta_range = np.linspace(-1.0, 1.7, num=32)
    condmass_range = np.logspace(8, 13, num=16)
    mass_range = np.logspace(7, 12, num=64)
    redshift = 8.0

    inputs = default_input_struct.evolve_input_structs(
        POWER_SPECTRUM=ps_model,
        HMF=hmf_model,
    )

    hmf_vals = cf.return_chmf_value(
        inputs=inputs,
        redshift=redshift,
        mass_values=mass_range,
        delta_values=delta_range,
        condmass_values=condmass_range,
    )

    assert hmf_vals.shape == (len(delta_range), len(condmass_range), len(mass_range))
    assert np.all(~np.isnan(hmf_vals))


def test_ps_runs(default_input_struct):
    k_values = np.logspace(-3, 1, num=64)

    ps = cf.get_matter_power_values(
        inputs=default_input_struct,
        k_values=k_values,
    )

    assert ps.shape == (len(k_values),)
    assert np.all(ps >= 0.0)


def make_matterfield_comparison_plot(
    x,
    k,
    true,
    test,
    plt,
    **kwargs,
):
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.set_xscale("log")
            if i == 1:
                ax.set_xlabel("Mass") if j < 2 else ax.set_xlabel("k")
                ax.set_ylim(-1, 1)
                ax.axhline(0, color="k", linestyle="--")
            else:
                ax.set_yscale("log")

            if j == 0:
                ax.set_ylabel("dndlnM")
            elif j == 1:
                ax.set_ylabel("Sigma")
            else:
                ax.set_ylabel("Power Spectrum")

    axs[0, 0].loglog(x, true[0], linestyle="-", label="Truth", **kwargs)
    axs[0, 0].loglog(x, test[0], linestyle=":", linewidth=3, label="Test", **kwargs)
    axs[1, 0].semilogx(x, (test[0] - true[0]) / true[0], **kwargs)
    axs[0, 0].legend()
    axs[0, 0].set_ylim(1e-12, 1e2)

    axs[0, 1].loglog(x, true[1], linestyle="-", label="Truth", **kwargs)
    axs[0, 1].loglog(x, test[1], linestyle=":", linewidth=3, label="Test", **kwargs)
    axs[1, 1].semilogx(x, (test[1] - true[1]) / true[1], **kwargs)

    axs[0, 2].loglog(k, true[2], linestyle="-", label="Truth", **kwargs)
    axs[0, 2].loglog(k, test[2], linestyle=":", linewidth=3, label="Test", **kwargs)
    axs[1, 2].semilogx(k, (test[2] - true[2]) / true[2], **kwargs)
