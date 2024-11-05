"""
Test the wrapper functions which access the C-backend, but not though an OutputStruct compute() method
"""

import pytest

import numpy as np

import py21cmfast as p21c


def test_run_lf():
    muv, mhalo, lf = p21c.compute_luminosity_function(redshifts=[7, 8, 9], nbins=100)
    assert np.all(lf[~np.isnan(lf)] > -30)
    assert lf.shape == (3, 100)

    # Check that memory is in-tact and a second run also works:
    muv, mhalo, lf2 = p21c.compute_luminosity_function(redshifts=[7, 8, 9], nbins=100)
    assert lf2.shape == (3, 100)
    assert np.allclose(lf2[~np.isnan(lf2)], lf[~np.isnan(lf)])

    muv_minih, mhalo_minih, lf_minih = p21c.compute_luminosity_function(
        redshifts=[7, 8, 9],
        nbins=100,
        component="mcg",
        flag_options={
            "USE_MINI_HALOS": True,
            "INHOMO_RECO": True,
            "USE_TS_FLUCT": True,
        },
        mturnovers=[7.0, 7.0, 7.0],
        mturnovers_mini=[5.0, 5.0, 5.0],
    )
    assert np.all(lf_minih[~np.isnan(lf_minih)] > -30)
    assert lf_minih.shape == (3, 100)
