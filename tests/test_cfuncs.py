"""Test the wrapper functions which access the C-backend, but not though an OutputStruct compute() method."""

import matplotlib as mpl
import numpy as np
import pytest
from hmf import MassFunction
from scipy import optimize

import py21cmfast as p21c
from py21cmfast.wrapper import cfuncs as cf

YUNG24_PHYSICAL_PARAMS = {
    "A_0": 0.13765772,
    "A_1": -0.01003821,
    "A_2": 0.00102964,
    "a_0": 1.06641384,
    "a_1": 0.02475576,
    "a_2": -0.00283342,
    "b_0": 4.86693806,
    "b_1": 0.09212356,
    "b_2": -0.01426283,
    "c_0": 1.19837952,
    "c_1": -0.00142967,
    "c_2": -0.00033074,
}


@pytest.mark.parametrize("use_lightcone", [True, False])
def test_run_lf(default_input_struct_lc, lc, use_lightcone, cache):
    inputs = default_input_struct_lc
    lightcone = lc if use_lightcone else None
    *_, lf = p21c.compute_luminosity_function(
        inputs=inputs,
        redshifts=[7, 8, 9],
        nbins=100,
        lightcone=lightcone,
    )
    assert np.all(lf[~np.isnan(lf)] > -30)
    assert lf.shape == (3, 100)

    # Check that memory is in-tact and a second run also works:
    _muv, _mhalo, lf2 = p21c.compute_luminosity_function(
        inputs=inputs,
        redshifts=[7, 8, 9],
        nbins=100,
        lightcone=lightcone,
    )
    assert lf2.shape == (3, 100)
    assert np.allclose(lf2[~np.isnan(lf2)], lf[~np.isnan(lf)])

    inputs = inputs.from_template(["mini", "tiny"], random_seed=9)
    if use_lightcone:
        pytest.skip("run_lightcone + mini-halo LF too time consuming (about 4 minutes)")
        lightcone_mini = p21c.run_lightcone(
            lightconer=p21c.RectilinearLightconer.between_redshifts(
                min_redshift=inputs.node_redshifts[-1] + 0.5,
                max_redshift=inputs.node_redshifts[0] - 0.5,
                resolution=inputs.simulation_options.cell_size,
                cosmo=inputs.cosmo_params.cosmo,
            ),
            inputs=inputs,
            write=p21c.CacheConfig(),
            cache=cache,
            include_dvdr_in_tau21=False,
        )
    else:
        lightcone_mini = None

    _muv_minih, _mhalo_minih, lf_minih = p21c.compute_luminosity_function(
        redshifts=[7, 8, 9],
        nbins=100,
        lightcone=lightcone_mini,
        component="mcg",
        inputs=inputs,
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


def test_bad_input_for_expected_nhalo(default_input_struct):
    """Test the expected_nhalo cannot be evaluated if the halo model is not discrete."""
    with pytest.raises(
        ValueError, match="SOURCE_MODEL must have a discrete halo model"
    ):
        cf.get_expected_nhalo(redshift=8.0, inputs=default_input_struct)


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
        cf.evaluate_Nion_z(
            inputs=default_input_struct,
            redshifts=redshifts,
            log10mturns=lnM_base,
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
        ValueError, match="Halo masses and rng shapes must be identical"
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
@pytest.mark.parametrize("ps_model", ["EH", "BBKS"])
@pytest.mark.xfail(reason="pending proper comparison between 21cmFAST and hmf")
def test_matterfield_statistics(default_input_struct, hmf_model, ps_model, plt):
    redshift = 8.0
    hmf_map = {
        "PS": "PS",
        "ST": "SMT",
    }
    transfer_map = {
        "EH": "EH_NoBAO",
        "BBKS": "BBKS",
    }

    if ps_model == "BBKS":
        t_params = {"use_sugiyama_baryons": True, "use_liddle_baryons": False}
    else:
        t_params = {}

    if hmf_model == "PS":
        hmf_params = {}
    elif hmf_model == "ST":
        hmf_params = {"a": 0.73, "p": 0.175, "A": 0.353}

    comparison_mf = MassFunction(
        z=redshift,
        Mmin=7,
        Mmax=16,
        hmf_model=hmf_map[hmf_model],
        hmf_params=hmf_params,
        transfer_model=transfer_map[ps_model],
        transfer_params=t_params,
        delta_c=1.68,  # hmf default is 1.686
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
        USE_INTERPOLATION_TABLES="no-interpolation",
    )
    h = inputs.cosmo_params.cosmo.h

    hmf_vals = cf.return_uhmf_value(
        inputs=inputs,
        redshift=redshift,
        mass_values=comparison_mf.m / h,
    )

    sigma_vals, dsigmasq_vals = cf.evaluate_sigma(
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
                -comparison_mf._dlnsdlnm * comparison_mf._sigma_0 / comparison_mf.m * h,
                comparison_mf._power0 / (h**3),
            ],
            [
                hmf_vals * mass_dens,
                sigma_vals,
                -dsigmasq_vals / (2 * sigma_vals),
                power_vals,
            ],
            plt,
        )

    # check matter power spectrum
    np.testing.assert_allclose(power_vals, comparison_mf._power0 / (h**3), rtol=1e-3)
    # check sigma(M)
    np.testing.assert_allclose(sigma_vals, comparison_mf._sigma_0, rtol=1e-3)
    # check dSigma/dM
    np.testing.assert_allclose(
        dsigmasq_vals / (2 * sigma_vals),
        comparison_mf._dlnsdlnm * comparison_mf._sigma_0 / comparison_mf.m * h,
        rtol=1e-3,
    )
    # check mass function
    np.testing.assert_allclose(
        mass_dens * hmf_vals, comparison_mf.dndlnm * (h**3), rtol=1e-3
    )


@pytest.mark.parametrize("hmf_model", ["PS", "ST", "REED07", "YUNG24"])
@pytest.mark.parametrize(
    "ps_model", ["EH", "BBKS", "EFSTATHIOU", "PEEBLES", "WHITE", "CLASS"]
)
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


@pytest.mark.parametrize("hmf_model", ["REED07", "YUNG24"])
@pytest.mark.parametrize("ps_model", ["EH", "BBKS"])
def test_new_hmf_matches_reference(default_input_struct, hmf_model, ps_model):
    redshift = 8.0
    if hmf_model == "REED07":
        transfer_map = {
            "EH": "EH_NoBAO",
            "BBKS": "BBKS",
        }
        if ps_model == "BBKS":
            transfer_params = {
                "use_sugiyama_baryons": True,
                "use_liddle_baryons": False,
            }
        else:
            transfer_params = {}

        comparison_mf = MassFunction(
            z=redshift,
            Mmin=7,
            Mmax=12,
            hmf_model="Reed07",
            transfer_model=transfer_map[ps_model],
            transfer_params=transfer_params,
            growth_model="GenMFGrowth",
            delta_c=1.686,
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
            USE_INTERPOLATION_TABLES="no-interpolation",
        )
        h = inputs.cosmo_params.cosmo.h
        masses = comparison_mf.m / h
        hmf_vals = cf.return_uhmf_value(
            inputs=inputs,
            redshift=redshift,
            mass_values=masses,
        )
        mass_dens = (
            inputs.cosmo_params.cosmo.critical_density(0).to("M_sun Mpc^-3").value
            * inputs.cosmo_params.cosmo.Om0
        )

        np.testing.assert_allclose(
            mass_dens * hmf_vals,
            comparison_mf.dndlnm * (h**3),
            rtol=2e-2,
        )
    else:
        masses = np.logspace(7, 12, num=64)
        inputs = default_input_struct.evolve_input_structs(
            POWER_SPECTRUM=ps_model,
            HMF=hmf_model,
            USE_INTERPOLATION_TABLES="no-interpolation",
        )
        hmf_vals = cf.return_uhmf_value(
            inputs=inputs, redshift=redshift, mass_values=masses
        )
        sigma0, dsigmasqdm = cf.evaluate_sigma(inputs=inputs, masses=masses)

        ps_inputs = inputs.evolve_input_structs(HMF="PS")
        ps_hmf_vals = cf.return_uhmf_value(
            inputs=ps_inputs, redshift=redshift, mass_values=masses
        )
        test_idx = len(masses) // 2
        delta_c = 1.686

        def ps_difference(growth):
            sigma_z = sigma0[test_idx] * growth
            dsigmadm = dsigmasqdm[test_idx] * growth / (2 * sigma0[test_idx])
            expected = (
                -np.sqrt(2 / np.pi)
                * (delta_c / sigma_z**2)
                * dsigmadm
                * np.exp(-(delta_c**2) / (2 * sigma_z**2))
            )
            return expected - ps_hmf_vals[test_idx]

        growth = optimize.brentq(ps_difference, 1e-4, 1.0)
        sigma = sigma0 * growth
        dlnsdlnm = -masses * dsigmasqdm / (2 * sigma0**2)

        # TODO: Switch this branch to using hmf once Yung24 is merged there.
        z = redshift
        a_z = (
            YUNG24_PHYSICAL_PARAMS["a_0"]
            + YUNG24_PHYSICAL_PARAMS["a_1"] * z
            + YUNG24_PHYSICAL_PARAMS["a_2"] * z**2
        )
        b_z = (
            YUNG24_PHYSICAL_PARAMS["b_0"]
            + YUNG24_PHYSICAL_PARAMS["b_1"] * z
            + YUNG24_PHYSICAL_PARAMS["b_2"] * z**2
        )
        c_z = (
            YUNG24_PHYSICAL_PARAMS["c_0"]
            + YUNG24_PHYSICAL_PARAMS["c_1"] * z
            + YUNG24_PHYSICAL_PARAMS["c_2"] * z**2
        )
        A_z = (
            YUNG24_PHYSICAL_PARAMS["A_0"]
            + YUNG24_PHYSICAL_PARAMS["A_1"] * z
            + YUNG24_PHYSICAL_PARAMS["A_2"] * z**2
        )
        f_sigma = A_z * ((sigma / b_z) ** (-a_z) + 1) * np.exp(-c_z / sigma**2)

        expected = f_sigma * dlnsdlnm / masses

        np.testing.assert_allclose(hmf_vals, expected, rtol=1e-6)


@pytest.mark.parametrize("hmf_model", ["PS", "ST"])
@pytest.mark.parametrize(
    "ps_model", ["EH", "BBKS", "EFSTATHIOU", "PEEBLES", "WHITE", "CLASS"]
)
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

    with pytest.raises(
        ValueError,
        match=r"inputs.matter_options.USE_RELATIVE_VELOCITIES must be True in order to compute the v_cb power spectrum\.",
    ):
        cf.get_vcb_power_values(
            inputs=default_input_struct,
            k_values=k_values,
        )

    ps = cf.get_vcb_power_values(
        inputs=default_input_struct.evolve_input_structs(
            POWER_SPECTRUM="CLASS",
            USE_RELATIVE_VELOCITIES=True,
            K_MAX_FOR_CLASS=1.0,
        ),
        k_values=k_values,
    )

    assert ps.shape == (len(k_values),)
    assert np.all(ps >= 0.0)


def test_ps_with_A_s_and_sigma8(default_input_struct):
    """Test that we get the same power spectrum, regadless if we use A_s or sigma_8."""
    k_values = np.logspace(-3, 1, num=64)

    input_CLASS = default_input_struct.evolve_input_structs(
        POWER_SPECTRUM="CLASS", K_MAX_FOR_CLASS=1.0
    )
    A_s = input_CLASS.cosmo_params.A_s

    ps_sigma8 = cf.get_matter_power_values(
        inputs=input_CLASS,
        k_values=k_values,
    )
    ps_A_s = cf.get_matter_power_values(
        inputs=input_CLASS.evolve_input_structs(A_s=A_s, SIGMA_8=None),
        k_values=k_values,
    )
    np.testing.assert_allclose(
        ps_sigma8,
        ps_A_s,
        rtol=5e-4,
    )


def make_matterfield_comparison_plot(
    x,
    k,
    true,
    test,
    plt,
    **kwargs,
):
    _fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            ax.set_xscale("log")
            ax.grid()
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
            elif j == 2:
                ax.set_ylabel("dSigmadM")
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

    axs[0, 2].loglog(x, true[2], linestyle="-", label="Truth", **kwargs)
    axs[0, 2].loglog(x, test[2], linestyle=":", linewidth=3, label="Test", **kwargs)
    axs[1, 2].semilogx(x, (test[2] - true[2]) / true[2], **kwargs)

    axs[0, 3].loglog(k, true[3], linestyle="-", label="Truth", **kwargs)
    axs[0, 3].loglog(k, test[3], linestyle=":", linewidth=3, label="Test", **kwargs)
    axs[1, 3].semilogx(k, (test[3] - true[3]) / true[3], **kwargs)


@pytest.mark.parametrize("use_lightcone", [True, False])
def test_functions_with_and_without_lightcone(
    default_input_struct_lc, lc, use_lightcone
):
    """Test that we can run functions with and without a lightcone as an input."""
    inputs = default_input_struct_lc
    lightcone = lc if use_lightcone else None

    redshifts = [7, 8, 9]
    densities = np.linspace(-0.98, 1.7, num=800)
    radius = 5  # Mpc

    sfrd, sfrd_mini = cf.evaluate_SFRD_z(
        inputs=inputs, redshifts=redshifts, lightcone=lightcone
    )
    assert len(sfrd) == len(redshifts)
    assert len(sfrd_mini) == len(redshifts)

    sfrd, sfrd_mini = cf.evaluate_SFRD_cond(
        inputs=inputs,
        redshift=redshifts[0],
        radius=radius,
        densities=densities,
        lightcone=lightcone,
    )
    assert len(sfrd) == len(densities)
    assert len(sfrd_mini) == len(densities)
