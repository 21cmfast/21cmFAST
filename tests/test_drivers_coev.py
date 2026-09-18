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


def test_coeval_resume_reconstructs_hbox_history(tmp_path_factory):
    """Regression test for a resume-correctness bug in ``_redshift_loop_generator``.

    When resuming a coeval run partway through the redshift scroll (e.g. across
    separate ``run_coeval``/job invocations sharing a cache), the generator used to
    skip both computing *and* loading ``HaloBox`` for the already-completed
    redshifts. But ``compute_xray_source_field`` needs the *entire* HaloBox history
    within ``R_MAX_TS`` of each new redshift (via the accumulated ``hbox_arr``) to
    build its filtered source shells. Without reloading the skipped HaloBoxes from
    cache, the X-ray source field -- and therefore spin temperature and brightness
    temperature -- computed after a resume would silently be wrong.

    This is exercised together with ``write=CacheConfig(xray_source_box=False)``,
    matching production usage where XraySourceBox is not cached (it is never read
    back as an input, so this must not affect resumability -- see
    ``RunCache.is_complete_at``).

    We run the same simulation twice against separate caches: once straight
    through, and once split into two separate ``run_coeval`` calls that force a
    resume partway through. The resulting BrightnessTemp at the final redshift
    must be identical.
    """
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
    write_no_xrsb = CacheConfig(xray_source_box=False)
    mid_z = inputs.node_redshifts[len(inputs.node_redshifts) // 2]

    # First run only partway (up to and including a middle node), matching
    # production usage of not writing XraySourceBox to disk.
    run_coeval(
        inputs=inputs,
        out_redshifts=mid_z,
        cache=cache_resume,
        write=write_no_xrsb,
        regenerate=True,
    )
    # Now request the final redshift; this should trigger a resume from cache.
    coeval_resumed = run_coeval(
        inputs=inputs,
        out_redshifts=inputs.node_redshifts[-1],
        cache=cache_resume,
        write=write_no_xrsb,
    )[0]

    np.testing.assert_array_equal(
        coeval_full.brightness_temperature.brightness_temp.value,
        coeval_resumed.brightness_temperature.brightness_temp.value,
    )


def test_obtain_starting_point_carries_cached_halobox(tmp_path_factory):
    """Regression test: the resume starting-point Coeval must carry its cached HaloBox.

    ``_obtain_starting_point_for_scrolling`` builds a ``Coeval`` from
    ``RunCache.get_all_boxes_at_z()``, whose dict keys are the RunCache
    attribute names (i.e. ``"HaloBox"``, not ``"Halobox"``). A casing typo
    (``outputs.get("Halobox", None)``) meant the returned Coeval's ``halobox``
    field was always ``None``, even when a HaloBox was cached on disk.

    This currently has no effect on simulated physics -- ``_redshift_loop_generator``
    never reads ``prev_coeval.halobox`` (the X-ray source integral is instead
    reconstructed from cache into ``hbox_arr``, see
    ``test_coeval_resume_reconstructs_hbox_history`` above) -- but it is still a
    real bug that silently discards cached data, and would reintroduce ``None``
    for any future code relying on ``prev_coeval.halobox``, analogous to how
    ``ionized_box``/``ts_box`` are already relied upon there.
    """
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

    cache = OutputCache(tmp_path_factory.mktemp("resume_halobox_casing"))
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
    assert coeval.halobox is not None
