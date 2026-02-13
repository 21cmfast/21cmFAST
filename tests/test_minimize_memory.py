"""Tests of switching on minimize memory flag."""

import numpy as np
import pytest

import py21cmfast as p21c


@pytest.mark.parametrize("source_model", ["L-INTEGRAL", "E-INTEGRAL"])
def test_minimize_memory_on_global_evolution(source_model: str):
    """Test that switching minimize memory on/off doesn't change outputs.

    We test this for a number of templates, to make make sure there aren't combinations
    of flags for which it fails. Specifically:

    * "latest-dhalos" -- in this the MINIMIZE_MEMORY flag should NOT have any effect
      at all in SpinTemperatureBox.c, since the source model is discrete halos.
    * "latest" -- in this the MINIMIZE_MEMORY flag should have a significant effect on
      memory, but will not change the results, since the source model is on the Eulerian
      grid
    * "default" -- always good to test the default case!
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
