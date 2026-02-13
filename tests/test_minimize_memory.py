"""Tests of switching on minimize memory flag."""

import numpy as np
import pytest

import py21cmfast as p21c


@pytest.mark.parametrize("source_model", ["L-INTEGRAL", "E-INTEGRAL"])
def test_minimize_memory_on_global_evolution(source_model: str):
    """Test that switching minimize memory on/off doesn't change outputs.

    We test this for two source models: one on the lagrangian grid (L-INTEGRAL) and one
    on the Eulerian grid (E-INTEGRAL). The former should not be affected at all by the
    MINIMIZE_MEMORY flag, while the latter should be significantly affected in terms of
    memory usage, but not in terms of results. We test this by running the global
    evolution for both cases and checking that the brightness temperature evolution is
    unchanged.
    """
    inputs = p21c.InputParameters.from_template(
        ["park19", "tiny"],
        random_seed=1234,
        MINIMIZE_MEMORY=False,
        SOURCE_MODEL=source_model,
    )
    inputs_minmem = inputs.evolve_input_structs(MINIMIZE_MEMORY=True)

    global_evolution = p21c.run_global_evolution(inputs=inputs)
    global_evolution_minmem = p21c.run_global_evolution(inputs=inputs_minmem)

    np.testing.assert_allclose(
        global_evolution.quantities["brightness_temp"],
        global_evolution_minmem.quantities["brightness_temp"],
        atol=0.1,  # mK
    )


def test_kinetic_temperature_validator():
    """Test that kinetic_temperature validator raises error when kinetic_temp_neutral is False."""
    with pytest.raises(
        ValueError,
        match="You cannot compute the kinetic temperature of the IGM if you are not",
    ):
        p21c.InputParameters.from_template(
            ["park19", "tiny"],
            random_seed=1234,
            kinetic_temp_neutral=False,
            kinetic_temperature=True,
        )
