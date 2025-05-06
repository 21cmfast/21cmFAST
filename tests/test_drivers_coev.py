"""
Unit-tests of the coeval driver.

They do not test for correctness of simulations, but whether different parameter options
work/don't work as intended.
"""

import attrs
import pytest

import py21cmfast as p21c
from py21cmfast import Coeval, run_coeval
from py21cmfast.wrapper.arrays import Array
from py21cmfast.wrapper.outputs import OutputStruct


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
    ic, default_input_struct, default_astro_options, cache
):
    with pytest.raises(ValueError, match="You have passed a redshift"):
        run_coeval(
            initial_conditions=ic,
            out_redshifts=2.0,
            inputs=default_input_struct.clone(
                astro_options=default_astro_options.clone(
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
            out_redshifts=16.0,
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


def test_coeval_fields():
    """Test that every array ends up in the coeval object."""
    fields = Coeval.get_fields()
    for kls in OutputStruct.__subclasses__():
        for field in attrs.fields(kls):
            if isinstance(field, Array):
                assert field.name in fields
