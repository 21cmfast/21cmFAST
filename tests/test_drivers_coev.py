"""
Unit-tests of the coeval driver.

They do not test for correctness of simulations, but whether different parameter options
work/don't work as intended.
"""

import gc
from collections import Counter

import attrs
import numpy as np
import pytest

import py21cmfast as p21c
from py21cmfast import CacheConfig, Coeval, InputParameters, OutputCache, run_coeval
from py21cmfast.io.caching import RunCache
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
        coeval_full.brightness_temperature.brightness_temp,
        coeval_resumed.brightness_temperature.brightness_temp,
    )


def test_coeval_resume_cached_ts_without_radiation_fields(tmp_path):
    """Resuming with cached TsBox but uncached RadiationFields must not crash (#791)."""
    inputs = InputParameters.from_template(
        ["latest-discrete", "size-tiny"], random_seed=1
    )
    cache = OutputCache(tmp_path)
    write = CacheConfig(radiation_fields=False)
    zs = inputs.node_redshifts

    for out_z in (zs[:2], zs[2:4]):
        # The second call restarts the loop with the first nodes' TsBox cached.
        out = [
            c
            for c, is_output in p21c.generate_coeval(
                inputs=inputs, out_redshifts=out_z, cache=cache, write=write
            )
            if is_output
        ]
        assert len(out) == len(out_z)


def _collect_cyclic_garbage() -> Counter:
    """Collect all garbage reference cycles, returning the types of the objects in them."""
    gc.set_debug(gc.DEBUG_SAVEALL)
    try:
        gc.collect()
        garbage = Counter(type(obj).__name__ for obj in gc.garbage)
    finally:
        gc.set_debug(0)
        gc.garbage.clear()
    # The cycles are still there after clearing gc.garbage; now really free them.
    gc.collect()
    return garbage


@pytest.mark.filterwarnings("ignore:The maximum halo mass:UserWarning")
@pytest.mark.filterwarnings("ignore:You are setting R_BUBBLE_MAX:UserWarning")
def test_coeval_redshift_steps_create_no_reference_cycles(tmp_path):
    """Test that no redshift step of a coeval run leaves reference cycles behind (#796).

    The high-level drivers run with the garbage collector disabled, so anything caught
    in a reference cycle -- along with everything it references, e.g. the boxes held by
    the frames that a cycle keeps alive -- stays in memory until the run ends. A cycle
    created at every redshift therefore makes memory grow with the length of the run,
    whatever creates it. This checks both a run that computes every box and one that
    replays them all from the cache.
    """
    inputs = InputParameters.from_template(
        ["latest-discrete", "size-tiny"], random_seed=1
    )
    cache = OutputCache(tmp_path)
    redshifts = inputs.node_redshifts[:4]

    for run in ("computed", "replayed from cache"):
        if run == "replayed from cache":
            # With discrete halos, the run is replayed from the first node redshift.
            rc = RunCache.from_inputs(inputs, cache)
            assert all(rc.is_complete_at(z=z) for z in redshifts)

        garbage = [
            _collect_cyclic_garbage()
            for _ in p21c.generate_coeval(
                inputs=inputs, out_redshifts=redshifts, cache=cache
            )
        ]

        # The first step also contains the one-off set-up of the run.
        cyclic = {i: g.most_common(5) for i, g in enumerate(garbage) if i and g}
        assert not cyclic, f"Steps of the {run} run left reference cycles: {cyclic}"


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
