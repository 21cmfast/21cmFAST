"""
A set of large-scale tests which test code updates against previously-run "golden"
results.

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
import pytest

import h5py
import logging
import matplotlib as mpl
import numpy as np

from py21cmfast import config, global_params

from . import produce_integration_test_data as prd

logger = logging.getLogger("21cmFAST")
logger.setLevel(logging.INFO)


options = list(prd.OPTIONS.keys())
options_pt = list(prd.OPTIONS_PT.keys())
options_halo = list(prd.OPTIONS_HALO.keys())


@pytest.mark.parametrize("name", options)
def test_power_spectra_coeval(name, module_direc, plt):
    redshift, kwargs = prd.OPTIONS[name]
    print(f"Options used for the test at z={redshift}: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename("power_spectra", name), "r") as fl:
        true_powers = {
            "_".join(key.split("_")[1:]): value[...]
            for key, value in fl["coeval"].items()
            if key.startswith("power_")
        }

    # Now compute the Coeval object
    with config.use(direc=module_direc, regenerate=False, write=True):
        with global_params.use(zprime_step_factor=prd.DEFAULT_ZPRIME_STEP_FACTOR):
            # Note that if zprime_step_factor is set in kwargs, it will over-ride this.
            test_k, test_powers, _ = prd.produce_coeval_power_spectra(
                redshift, **kwargs
            )

    if plt == mpl.pyplot:
        make_coeval_comparison_plot(test_k, true_powers, test_powers, plt)

    for key, value in true_powers.items():
        print(f"Testing {key}")
        assert np.sum(~np.isclose(value, test_powers[key], atol=0, rtol=1e-2)) < 10
        np.testing.assert_allclose(value, test_powers[key], atol=0, rtol=1e-1)


@pytest.mark.parametrize("name", options)
def test_power_spectra_lightcone(name, module_direc, plt):
    redshift, kwargs = prd.OPTIONS[name]
    print(f"Options used for the test at z={redshift}: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename("power_spectra", name), "r") as fl:
        true_powers = {}
        true_global = {}
        for key in fl["lightcone"].keys():
            if key.startswith("power_"):
                true_powers["_".join(key.split("_")[1:])] = fl["lightcone"][key][...]
            elif key.startswith("global_"):
                true_global[key] = fl["lightcone"][key][...]

    # Now compute the lightcone
    with config.use(direc=module_direc, regenerate=False, write=True):
        with global_params.use(zprime_step_factor=prd.DEFAULT_ZPRIME_STEP_FACTOR):
            # Note that if zprime_step_factor is set in kwargs, it will over-ride this.
            test_k, test_powers, lc = prd.produce_lc_power_spectra(redshift, **kwargs)

    if plt == mpl.pyplot:
        make_lightcone_comparison_plot(
            test_k, lc.node_redshifts, true_powers, true_global, test_powers, lc, plt
        )

    for key, value in true_powers.items():
        print(f"Testing {key}")
        # Ensure all but 10 of the values is within 1%, and none of the values
        # is outside 10%
        assert np.sum(~np.isclose(value, test_powers[key], atol=0, rtol=1e-2)) < 10
        assert np.allclose(value, test_powers[key], atol=0, rtol=1e-1)

    for key, value in true_global.items():
        print(f"Testing Global {key}")
        assert np.allclose(value, getattr(lc, key), atol=0, rtol=1e-3)


def make_lightcone_comparison_plot(
    k, z, true_powers, true_global, test_powers, lc, plt
):
    n = len(true_global) + len(true_powers)
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 5))

    for i, (key, val) in enumerate(true_powers.items()):
        make_comparison_plot(
            k, val, test_powers[key], ax[:, i], xlab="k", ylab=f"{key} Power"
        )

    for i, (key, val) in enumerate(true_global.items(), start=i + 1):
        make_comparison_plot(
            z, val, getattr(lc, key), ax[:, i], xlab="z", ylab=f"{key}"
        )


def make_coeval_comparison_plot(k, true_powers, test_powers, plt):
    fig, ax = plt.subplots(
        2, len(true_powers), figsize=(3 * len(true_powers), 6), sharex=True
    )

    for i, (key, val) in enumerate(true_powers.items()):
        make_comparison_plot(
            k, val, test_powers[key], ax[:, i], xlab="k", ylab=f"{key} Power"
        )


def make_comparison_plot(x, true, test, ax, logx=True, logy=True, xlab=None, ylab=None):

    ax[0].plot(x, true, label="True")
    ax[0].plot(x, test, label="Test")
    if logx:
        ax[0].set_xscale("log")
    if logy:
        ax[0].set_yscale("log")
    if xlab:
        ax[0].set_xlabel(xlab)
    if ylab:
        ax[0].set_ylabel(ylab)

    ax[0].legend()

    ax[1].plot(x, (test - true) / true)
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

    with global_params.use(zprime_step_factor=prd.DEFAULT_ZPRIME_STEP_FACTOR):
        # Note that if zprime_step_factor is set in kwargs, it will over-ride this.
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

    assert np.allclose(power_dens, p_dens, atol=5e-3, rtol=1e-3)
    assert np.allclose(power_vel, p_vel, atol=5e-3, rtol=1e-3)
    assert np.allclose(pdf_dens, y_dens, atol=5e-3, rtol=1e-3)
    assert np.allclose(pdf_vel, y_vel, atol=5e-3, rtol=1e-3)


@pytest.mark.parametrize("name", options_halo)
def test_halo_field_data(name):
    redshift, kwargs = prd.OPTIONS_HALO[name]
    print("Options used for the test: ", kwargs)

    # First get pre-made data
    with h5py.File(prd.get_filename("halo_field_data", name), "r") as f:
        n_pt_halos = f["n_pt_halos"][...]
        pt_halo_masses = f["pt_halo_masses"][...]

    with global_params.use(zprime_step_factor=prd.DEFAULT_ZPRIME_STEP_FACTOR):
        # Note that if zprime_step_factor is set in kwargs, it will over-ride this.
        pt_halos = prd.produce_halo_field_data(redshift, **kwargs)

    assert np.allclose(n_pt_halos, pt_halos.n_halos, atol=5e-3, rtol=1e-3)
    assert np.allclose(
        np.sum(pt_halo_masses), np.sum(pt_halos.halo_masses), atol=5e-3, rtol=1e-3
    )
