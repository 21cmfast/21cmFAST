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


# The "tiny" template sets R_BUBBLE_MAX with recombinations on, which warns.
@pytest.mark.filterwarnings("ignore:^You are setting R_BUBBLE_MAX:UserWarning")
@pytest.mark.parametrize("alias", [False, True])
def test_calibration_with_discrete_halos(alias):
    """f-photoncons calibrates for discrete halos (CHMF-SAMPLER).

    The calibration runs L-INTEGRAL without spin temperature, recombinations, MCGs or
    the upper stellar turnover, from the run's initial conditions. ``alias`` also sets
    the deprecated INHOMO_RECO, which the calibration's RECOMB_MODEL must override.
    """
    inputs = p21c.InputParameters.from_template(
        ["latest-discrete", "tiny"], random_seed=1
    ).evolve_input_structs(PHOTON_CONS_TYPE="f-photoncons")
    if alias:
        inputs = inputs.evolve_input_structs(INHOMO_RECO=True)
        assert inputs.astro_options._INHOMO_RECO is True
    assert inputs.matter_options.SOURCE_MODEL == "CHMF-SAMPLER"
    assert inputs.astro_options.USE_UPPER_STELLAR_TURNOVER

    ics = p21c.compute_initial_conditions(inputs=inputs)
    data = p21c.setup_photon_cons(initial_conditions=ics, inputs=inputs)

    nf = data["nf_calibration"]
    assert np.all((nf >= 0) & (nf <= 1))
    assert nf.max() > 0.9
    assert nf.min() < 0.1
    assert np.isfinite(data["fit_yint"])
    assert np.isfinite(data["fit_slope"])
