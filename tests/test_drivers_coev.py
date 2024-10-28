"""
These are designed to be unit-tests of the lightcone drivers. They do not test for
correctness of simulations,
but whether different parameter options work/don't work as intended.
"""

import pytest

import py21cmfast as p21c
from py21cmfast import run_coeval


def test_coeval_st(ic, perturbed_field):
    coeval = run_coeval(
        initial_conditions=ic,
        perturbed_field=perturbed_field,
        flag_options={"USE_TS_FLUCT": True},
    )

    assert isinstance(coeval.spin_temp_struct, p21c.TsBox)


def test_run_coeval_bad_inputs():
    with pytest.raises(
        ValueError, match="Cannot use an interpolated perturb field with minihalos"
    ):
        run_coeval(
            redshift=6.0,
            flag_options={
                "USE_MINI_HALOS": True,
                "INHOMO_RECO": True,
                "USE_TS_FLUCT": True,
            },
            use_interp_perturb_field=True,
        )


def test_coeval_lowerz_than_photon_cons(ic):
    with pytest.raises(ValueError, match="You have passed a redshift"):
        run_coeval(
            initial_conditions=ic,
            out_redshift=2.0,
            flag_options={
                "PHOTON_CONS_TYPE": 1,
                "USE_HALO_FIELD": False,
                "HALO_STOCHASTICITY": False,
            },
        )
