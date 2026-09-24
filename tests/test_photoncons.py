"""Tests of photon conservation in 21cmFAST."""

import numpy as np
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


def test_alpha_photoncons_scan_varies_alpha(tiny_inputs, tiny_ics):
    """The ALPHA_ESC scan must actually reach the backend.

    ``photoncons_alpha`` builds an analytic reionization history for each of a range of
    ALPHA_ESC values and picks the one matching the calibration simulation. If those
    values never reach C, every history in the scan comes out identical and the
    correction is fitted to a degenerate set of curves.
    """
    inputs = tiny_inputs.evolve_input_structs(PHOTON_CONS_TYPE="alpha-photoncons")
    data = p21c.setup_photon_cons(initial_conditions=tiny_ics, inputs=inputs)

    q_alpha = data["Q_alpha"]
    assert not np.allclose(q_alpha, q_alpha[0])
