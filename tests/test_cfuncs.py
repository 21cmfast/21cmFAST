"""
Test the wrapper functions which access the C-backend, but not though an OutputStruct compute() method
"""

import pytest

import numpy as np

import py21cmfast as p21c


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
