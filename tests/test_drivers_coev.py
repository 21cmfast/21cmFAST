"""
Unit-tests of the coeval driver.

They do not test for correctness of simulations, but whether different parameter options
work/don't work as intended.
"""

import pytest

import py21cmfast as p21c
from py21cmfast import run_coeval


def test_coeval_st(ic, default_input_struct_ts, cache):
    coeval = run_coeval(
        initial_conditions=ic,
        inputs=default_input_struct_ts,
        cache=cache,
    )
    assert isinstance(coeval[0].ts_box, p21c.TsBox)


def test_run_coeval_bad_inputs(ic, perturbed_field, default_input_struct, cache):
    with pytest.raises(
        ValueError, match="out_redshifts must be given if inputs has no node redshifts"
    ):
        run_coeval(initial_conditions=ic, inputs=default_input_struct, cache=cache)


def test_coeval_lowerz_than_photon_cons(
    ic, default_input_struct, default_astro_flags, cache
):
    with pytest.raises(ValueError, match="You have passed a redshift"):
        run_coeval(
            initial_conditions=ic,
            out_redshifts=2.0,
            inputs=default_input_struct.clone(
                astro_flags=default_astro_flags.clone(
                    PHOTON_CONS_TYPE="z-photoncons",
                )
            ),
            cache=cache,
        )


def test_coeval_warnings(default_input_struct_lc, cache):
    # test for no caching with halo fields
    with pytest.warns(UserWarning, match="You have turned off caching"):
        inputs = default_input_struct_lc.evolve_input_structs(
            USE_HALO_FIELD=True,
        )
        run_coeval(
            out_redshifts=8.0,
            inputs=inputs,
            write=False,
            cache=cache,
        )

    # test for minimum node redshift > out_redshifts
    with pytest.warns(UserWarning, match="minimum node redshift"):
        inputs = default_input_struct_lc.evolve_input_structs(
            USE_TS_FLUCT=True,
        )
        run_coeval(
            out_redshifts=8.0,
            inputs=inputs,
            write=False,
            cache=cache,
        )
