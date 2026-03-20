"""Tests of photon conservation in 21cmFAST."""

import pytest

import py21cmfast as p21c


@pytest.fixture(scope="module")
def tiny_inputs():
    return (
        p21c.InputParameters.from_template(["simple", "tiny"], random_seed=1234)
        .evolve_input_structs(
            PHOTON_CONS_TYPE="z-photoncons",
            Z_HEAT_MAX=20,
        )
        .with_logspaced_redshifts(
            zmin=10,
            zmax=20,
        )
    )


@pytest.fixture(scope="module")
def tiny_ics(tiny_inputs):
    return p21c.compute_initial_conditions(inputs=tiny_inputs)


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
