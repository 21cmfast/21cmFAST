"""Tests of photon conservation in 21cmFAST."""

import pytest

import py21cmfast as p21c


@pytest.mark.parametrize(
    "model",
    [
        "no-photoncons",
        "z-photoncons",
        "alpha-photoncons",
        "f-photoncons",
    ],
)
def test_memory_accesss(model, tiny_inputs, tiny_ics):
    """Simply tests that no segfaults occur."""
    inputs = tiny_inputs.evolve_input_structs(PHOTON_CONS_TYPE=model)

    p21c.run_coeval(
        inputs=inputs, initial_conditions=tiny_ics, write=False, progressbar=True
    )
