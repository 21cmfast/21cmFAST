"""Test the wrapper functions which access the C-backend, but not though an OutputStruct compute() method."""

import numpy as np
import pytest

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
