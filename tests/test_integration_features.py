"""
Large-scale tests which test code updates against previously-run "golden" results.

The idea here is that any new updates (except for major versions) should be non-breaking;
firstly, they should not break the API, so that the tests should run without crashing without
being changed.
Secondly, the actual results of running the basic functions should remain the same for
the same input code, except for potential bug-fixes. In these cases, these tests should
pick these changes up. The test data should then be changed to reflect the new gold
standard, and if applicable, a new test should be written that reflects the previous
broken code.
Thirdly, it enforces that new features, where possible, are added in such a way as to
keep the default behaviour constant. That is, the tests here should *not* run the added
feature, and therefore should continue to produce the same test results regardless of
the new feature added. The new feature should be accompanied by its own tests, whether
in this or another test module. If a new feature *must* be included by default, then
it must be implemented in a new major version of the code, at which point the test data
is able to be updated.

Comparison tests here are meant to be as small as possible while attempting to form
a reasonable test: they should be of reduced data such as power spectra or global xHI
measurements, and they should be generated with small simulations.
"""

import logging

import h5py
import matplotlib as mpl
import numpy as np
import pytest

from py21cmfast import Coeval, LightCone, OutputCache

from . import produce_integration_test_data as prd

logger = logging.getLogger("py21cmfast")
logger.setLevel(logging.INFO)

options = list(prd.OPTIONS_TESTRUNS.keys())
options_pt = list(prd.OPTIONS_PT.keys())


@pytest.mark.parametrize("name", options)
def test_power_spectra_coeval(name, module_direc, plt):
    redshift, kwargs = prd.OPTIONS_TESTRUNS[name]
    print(f"Options used for the test {name} at z={redshift}: ", kwargs)

    # First get pre-made data
    true_powers = {}
    with h5py.File(prd.get_filename("power_spectra", name), "r") as fl:
        true_k = fl["coeval"]["k"][...]
        for key, value in fl["coeval"].items():
            if key.startswith("power_"):
                fieldname = "_".join(key.split("_")[1:])
                true_powers[fieldname] = value[...]
                print(f"{key} --> {fieldname} loaded")

    # Now compute the Coeval object
    test_k, test_powers, cv = prd.produce_coeval_power_spectra(
        redshift, cache=OutputCache(module_direc), **kwargs
    )

    assert isinstance(cv, Coeval)
    assert np.all(np.isfinite(cv.brightness_temp))

    assert np.allclose(true_k, test_k)

    any_failed = False
    # We don't assert that all the fields are identical, but print the differences
    for key in prd.COEVAL_FIELDS:
        if key not in true_powers:
            continue
        any_failed |= prd.print_failure_stats(
            test_powers[key],
            true_powers[key],
            [test_k],
            abs_tol=0,
            rel_tol=1e-4,
            name=f"{name} - {key}",
        )

    any_failed = True  # TODO:remove this testing line
    if plt == mpl.pyplot and any_failed:
        make_coeval_comparison_plot(true_k, test_k, true_powers, test_powers, plt)


@pytest.mark.parametrize("name", options)
def test_power_spectra_lightcone(name, module_direc, plt, benchmark):
    redshift, kwargs = prd.OPTIONS_TESTRUNS[name]
    print(f"Options used for the test {name} at z={redshift}: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename("power_spectra", name), "r") as fl:
        true_powers = {}
        true_global = {}
        true_k = fl["lightcone"]["k"][...]
        for key in fl["lightcone"]:
            fieldname = "_".join(key.split("_")[1:])
            if key.startswith("power_"):
                true_powers[fieldname] = fl["lightcone"][key][...]
            elif key.startswith("global_"):
                true_global[fieldname] = fl["lightcone"][key][...]

    # Now compute the lightcone
    test_k, test_powers, lc = benchmark.pedantic(
        prd.produce_lc_power_spectra,
        kwargs=dict(redshift=redshift, cache=OutputCache(module_direc), **kwargs),
        iterations=1,  # these tests can be slow
        rounds=1,
    )

    assert isinstance(lc, LightCone)
    assert np.all(np.isfinite(lc.lightcones["brightness_temp"]))

    test_global = {k: lc.global_quantities[k] for k in true_global}
    assert np.allclose(true_k, test_k)

    # We don't assert that all the fields are identical, but print the differences
    any_failed = False
    for key in prd.LIGHTCONE_FIELDS:
        if key not in true_powers:
            continue
        any_failed |= prd.print_failure_stats(
            test_powers[key],
            true_powers[key],
            [test_k],
            abs_tol=0,
            rel_tol=1e-4,
            name=key,
        )

    any_failed = True  # TODO:remove this testing line
    if plt == mpl.pyplot and any_failed:
        make_lightcone_comparison_plot(
            true_k,
            test_k,
            lc.inputs.node_redshifts,
            true_powers,
            true_global,
            test_powers,
            test_global,
            plt,
        )

    # For globals, we should assert that they are close
    for key, value in true_global.items():
        print(f"Testing Global {key}")
        assert np.allclose(value, lc.global_quantities[key], atol=0, rtol=1e-3)


plot_ylab = {
    "density": r"$\delta$",
    "lowres_density": r"$\delta_{IC}$",
    "lowres_vx_2LPT": r"$v_x$ IC 2LPT",
    "lowres_vx": r"$v_x$ IC",
    "velocity_z": r"$v_z$",
    "spin_temperature": r"$T_S$",
    "xray_ionised_fraction": r"$x_{e}$",
    "J_21_LW": r"$J_{21,LW}$",
    "kinetic_temp_neutral": r"$T_{K,HI}$",
    "ionisation_rate_G12": r"$\Gamma_{12}$",
    "cumulative_recombinations": r"$N_{rec}$",
    "neutral_fraction": r"$x_{HI}$",
    "z_reion": r"$z_{re}$",
    "brightness_temp": r"$dT_b$",
}


def make_lightcone_comparison_plot(
    true_k, k, z, true_powers, true_global, test_powers, test_global, plt
):
    n = len(true_global) + len(true_powers)
    fig, ax = plt.subplots(
        2,
        n,
        figsize=(4 * n, 6),
        constrained_layout=True,
        sharex="col",
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )

    for i, (key, val) in enumerate(test_powers.items()):
        make_comparison_plot(
            true_k,
            k,
            true_powers[key],
            val,
            ax[:, i],
            xlab="k",
            ylab=f"{plot_ylab[key]} Power",
            make_lower_ylab=i == 0,
        )

    for j, (key, val) in enumerate(test_global.items(), start=i + 1):
        make_comparison_plot(
            z,
            z,
            true_global[key],
            val,
            ax[:, j],
            xlab="z",
            ylab=f"{plot_ylab[key]}",
            logx=False,
            logy=False,
            make_lower_ylab=False,
        )


def make_coeval_comparison_plot(true_k, k, true_powers, test_powers, plt):
    fig, ax = plt.subplots(
        2,
        len(true_powers),
        figsize=(4 * len(true_powers), 6),
        sharex=True,
        constrained_layout=True,
    )

    for i, (key, val) in enumerate(test_powers.items()):
        make_comparison_plot(
            true_k,
            k,
            true_powers[key],
            val,
            ax[:, i],
            xlab="k",
            ylab=f"{plot_ylab[key]} Power",
            make_lower_ylab=i == 0,
        )


def make_comparison_plot(
    xtrue,
    x,
    true,
    test,
    ax,
    logx=True,
    logy=True,
    xlab=None,
    ylab=None,
    make_lower_ylab=True,
):
    ax[0].plot(xtrue, true, label="True")
    ax[0].plot(x, test, label="Test")
    if logx:
        ax[0].set_xscale("log")
    if logy:
        ax[0].set_yscale("log")
    if xlab:
        ax[1].set_xlabel(xlab)
    if ylab:
        ax[0].set_title(ylab)

    ax[0].legend()

    ax[1].plot(x, (test - true) / true)

    if make_lower_ylab:
        ax[1].set_ylabel("Fractional Difference")


@pytest.mark.parametrize("name", options_pt)
def test_perturb_field_data(name):
    redshift, kwargs = prd.OPTIONS_PT[name]
    print("Options used for the test: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename("perturb_field_data", name), "r") as f:
        power_dens = f["power_dens"][...]
        power_vel = f["power_vel"][...]
        pdf_dens = f["pdf_dens"][...]
        pdf_vel = f["pdf_vel"][...]

    (
        k_dens,
        p_dens,
        k_vel,
        p_vel,
        x_dens,
        y_dens,
        x_vel,
        y_vel,
        ic,
    ) = prd.produce_perturb_field_data(redshift, **kwargs)

    np.testing.assert_allclose(p_dens, power_dens, atol=5e-3, rtol=1e-3)
    np.testing.assert_allclose(p_vel, power_vel, atol=5e-3, rtol=1e-3)
    np.testing.assert_allclose(y_dens, pdf_dens, atol=5e-3, rtol=1e-3)
    np.testing.assert_allclose(y_vel, pdf_vel, atol=5e-3, rtol=1e-3)
