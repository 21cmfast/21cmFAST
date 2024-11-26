"""
These are designed to be unit-tests of the lightcone drivers. They do not test for
correctness of simulations,
but whether different parameter options work/don't work as intended.
"""

import pytest

import py21cmfast as p21c
from py21cmfast import run_coeval


def test_coeval_st(ic, default_input_struct_ts, perturbed_field):
    coeval = run_coeval(
        initial_conditions=ic,
        perturbed_field=perturbed_field,
        inputs=default_input_struct_ts,
    )
    assert isinstance(coeval.spin_temp_struct, p21c.TsBox)


def test_run_coeval_bad_inputs(ic, default_input_struct, default_flag_options):
    with pytest.raises(
        ValueError, match="Either out_redshifts or perturb must be given"
    ):
        run_coeval(
            initial_conditions=ic,
            inputs=default_input_struct,
        )

    with pytest.raises(
        ValueError, match="An integer seed, or initial conditions must be given"
    ):
        run_coeval(
            out_redshifts=20.0,
            inputs=default_input_struct.clone(
                random_seed=None,
            ),
        )


def test_coeval_lowerz_than_photon_cons(ic, default_input_struct, default_flag_options):
    with pytest.raises(ValueError, match="You have passed a redshift"):
        run_coeval(
            initial_conditions=ic,
            out_redshifts=2.0,
            inputs=default_input_struct.clone(
                flag_options=default_flag_options.clone(
                    PHOTON_CONS_TYPE="z-photoncons",
                )
            ),
        )
