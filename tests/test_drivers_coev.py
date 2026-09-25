"""
Unit-tests of the coeval driver.

They do not test for correctness of simulations, but whether different parameter options
work/don't work as intended.
"""

import attrs
import numpy as np
import pytest

import py21cmfast as p21c
from py21cmfast import CacheConfig, Coeval, InputParameters, OutputCache, run_coeval
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


@pytest.mark.filterwarnings("ignore:Trying to purge array:UserWarning")
def test_coeval_warnings(default_input_struct_lc, cache):
    # test for no caching with halo fields
    inputs = default_input_struct_lc.evolve_input_structs(
        SOURCE_MODEL="CHMF-SAMPLER",
    )

    with pytest.warns(UserWarning, match="You have turned off caching"):
        run_coeval(
            out_redshifts=16.0,
            inputs=inputs,
            write=False,
            cache=cache,
        )

    inputs = default_input_struct_lc.evolve_input_structs(
        USE_TS_FLUCT=True,
    )

    # test for minimum node redshift > out_redshifts
    with pytest.warns(UserWarning, match="minimum node redshift"):
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


def test_coeval_resume_reconstructs_radiation_fields_history(tmp_path_factory):
    """Test that a coeval run can be resumed from a previous run, and that the resumed run produces the same results as a full run."""
    inputs = InputParameters.from_template(
        "tiny", random_seed=1234, node_redshifts=np.arange(12, 24, 2.0)[::-1]
    ).evolve_input_structs(
        ZPRIME_STEP_FACTOR=1.5,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        USE_UPPER_STELLAR_TURNOVER=False,
        USE_EXP_FILTER=False,
        CELL_RECOMB=False,
    )
    assert inputs.matter_options.lagrangian_source_grid
    assert not inputs.matter_options.has_discrete_halos

    cache_full = OutputCache(tmp_path_factory.mktemp("resume_full"))
    coeval_full = run_coeval(
        inputs=inputs,
        out_redshifts=inputs.node_redshifts[-1],
        cache=cache_full,
        write=True,
        regenerate=True,
    )[0]

    cache_resume = OutputCache(tmp_path_factory.mktemp("resume_partial"))
    write_no_radfields = CacheConfig(radiation_fields=False)
    mid_z = inputs.node_redshifts[len(inputs.node_redshifts) // 2]

    # First run only partway (up to and including a middle node), matching
    # production usage of not writing RadiationFields to disk.
    run_coeval(
        inputs=inputs,
        out_redshifts=mid_z,
        cache=cache_resume,
        write=write_no_radfields,
        regenerate=True,
    )
    # Now request the final redshift; this should trigger a resume from cache.
    coeval_resumed = run_coeval(
        inputs=inputs,
        out_redshifts=inputs.node_redshifts[-1],
        cache=cache_resume,
        write=write_no_radfields,
    )[0]

    np.testing.assert_array_equal(
        coeval_full.brightness_temperature.brightness_temp.value,
        coeval_resumed.brightness_temperature.brightness_temp.value,
    )


def test_obtain_starting_point_carries_cached_emissivity_fields(tmp_path_factory):
    """Test that the _obtain_starting_point_for_scrolling function correctly carries over cached EmissivityFields data when resuming a run."""
    from py21cmfast.drivers.coeval import _obtain_starting_point_for_scrolling
    from py21cmfast.io.caching import RunCache

    inputs = InputParameters.from_template(
        "tiny", random_seed=1234, node_redshifts=np.arange(12, 24, 2.0)[::-1]
    ).evolve_input_structs(
        ZPRIME_STEP_FACTOR=1.5,
        SOURCE_MODEL="L-INTEGRAL",
        USE_TS_FLUCT=True,
        USE_UPPER_STELLAR_TURNOVER=False,
        USE_EXP_FILTER=False,
        CELL_RECOMB=False,
    )
    assert inputs.matter_options.lagrangian_source_grid

    cache = OutputCache(tmp_path_factory.mktemp("resume_emissivity_fields_casing"))
    run_coeval(
        inputs=inputs,
        out_redshifts=inputs.node_redshifts[-1],
        cache=cache,
        write=True,
        regenerate=True,
    )

    rc = RunCache.from_inputs(inputs, cache)
    idx, coeval = _obtain_starting_point_for_scrolling(
        inputs=inputs,
        initial_conditions=rc.get_ics(),
        photon_nonconservation_data={},
        cache=cache,
        regenerate=False,
    )

    assert idx >= 0
    assert coeval is not None
    assert coeval.emissivity_fields is not None
