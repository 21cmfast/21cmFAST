"""Test the GlobalEvolution class."""

from pathlib import Path

import numpy as np
import pytest

import py21cmfast as p21c
from py21cmfast import GlobalEvolution

DATA_PATH = Path(__file__).parent / "test_data"


@pytest.mark.parametrize("source_model", ["CONST-ION-EFF", "E-INTEGRAL", "L-INTEGRAL"])
def test_global_quantities(default_input_struct_ts, source_model):
    """Test that global quantities behave as expected."""
    global_evolution = p21c.run_global_evolution(
        inputs=default_input_struct_ts.with_logspaced_redshifts(),
        source_model=source_model,
    )

    z = np.array(global_evolution.node_redshifts)
    T_gamma = 2.7255 * (1.0 + z)
    T_k = global_evolution.quantities["kinetic_temp_neutral"]
    T_s = global_evolution.quantities["spin_temperature"]
    x_HI = global_evolution.quantities["neutral_fraction"]
    T_21 = global_evolution.quantities["brightness_temp"]

    # Find where the minimum temperature is
    min_idx_Tk = np.argmin(T_k)
    min_idx_Ts = np.argmin(T_s)
    # Check it's not at the edges (i.e., there's a true minimum)
    assert 0 < min_idx_Tk < len(T_k) - 1
    assert 0 < min_idx_Ts < len(T_s) - 1
    # Test that the gas is cooled down (by adiabatic cooling) before it is heated up (by X-rays)
    # NOTE: we don't check that the spin temperature continues to climb after the minimum,
    #       because the Lya coupling could decrease at some point, thus lowering the spin temperature
    assert np.all(np.diff(T_k[: min_idx_Tk + 1]) <= 0)
    assert np.all(np.diff(T_k[min_idx_Tk:]) >= 0)
    assert np.all(np.diff(T_s[: min_idx_Ts + 1]) <= 0)

    # Check that the spin temperature is contained between the gas temperature and the CMB temperature
    assert np.all(T_s <= np.maximum(T_gamma, T_k))
    assert np.all(np.minimum(T_gamma, T_k) <= T_s)

    # Make sure that the x_HI curve is montonotonously decreasing
    assert np.all(np.diff(x_HI) <= 0)

    # Test that global signal has two local maxima and minima
    sign_changes = np.diff(np.sign(np.diff(T_21)))
    local_maxima_indices = np.where(sign_changes < 0)[0]
    local_minima_indices = np.where(sign_changes > 0)[0]
    assert len(local_maxima_indices) == 2  # Lya coupling + reionization start
    assert len(local_minima_indices) == 2  # X-ray heating + reionization end

    # Check that the brightness temperature vanishes when reionization ended
    assert np.all(T_21[local_minima_indices[1] + 1 :] == 0.0)
    assert np.all(x_HI[local_minima_indices[1] + 1 :] == 0.0)


@pytest.mark.parametrize("source_model", ["CONST-ION-EFF", "E-INTEGRAL", "L-INTEGRAL"])
def test_run_global_evolution_without_source_model(
    default_input_struct_ts, source_model
):
    """Test that run_global_evolution works without specifying source_model."""
    global_evolution = p21c.run_global_evolution(
        inputs=default_input_struct_ts.evolve_input_structs(SOURCE_MODEL=source_model)
    )
    assert isinstance(global_evolution, GlobalEvolution)


def test_run_global_evolution_without_Ts(default_input_struct):
    """Test that run_global_evolution doesn't crash when USE_TS_FLUCT=False."""
    global_evolution = p21c.run_global_evolution(
        inputs=default_input_struct.evolve_input_structs(USE_TS_FLUCT=False)
    )
    assert isinstance(global_evolution, GlobalEvolution)


def test_global_evolution_roundtrip(test_direc, default_input_struct_ts):
    """Test that the save/from_file methods yield the same GlobalEvolution object."""
    global_evolution1 = p21c.run_global_evolution(inputs=default_input_struct_ts)
    fname = test_direc / "global_evolution.h5"
    global_evolution1.save(path=fname)
    global_evolution2 = GlobalEvolution.from_file(fname)

    assert global_evolution1 == global_evolution2


@pytest.mark.parametrize("source_model", ["X-INTEGRAL", "DEXM-ESF", "CHMF-SAMPLER"])
def test_global_evolution_bad_inputs(default_input_struct_ts, source_model):
    """Test run_global_evolution with bad inputs."""
    with pytest.raises(ValueError, match="'source_model' must be one of"):
        p21c.run_global_evolution(
            inputs=default_input_struct_ts, source_model=source_model
        )
    if source_model == "X-INTEGRAL":
        pytest.xfail("X-INTEGRAL is not a valid option for SOURCE_MODEL.")
    else:
        with pytest.raises(
            ValueError,
            match="You did not specify 'source_model', but SOURCE_MODEL in `inputs` has discrete halos!",
        ):
            p21c.run_global_evolution(
                inputs=default_input_struct_ts.evolve_input_structs(
                    SOURCE_MODEL=source_model
                ),
            )


def test_compatability_with_database():
    """
    Test that loading GlobalEvolution from the database succeeds.

    This test is useful for checking compatability with data files that
    were generated with older versions of 21cmFAST.

    If this test fails DO NOT replace the file in the database, unless you
    are absolutely sure of what you are doing.
    """
    fname = DATA_PATH / "global_evolution.h5"
    global_evolution = GlobalEvolution.from_file(fname)
    assert isinstance(global_evolution, GlobalEvolution)
