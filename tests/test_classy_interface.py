"""Tests of the classy module."""

import numpy as np
import pytest
from astropy import units

from py21cmfast.wrapper import classy_interface
from py21cmfast.wrapper.classy_interface import (
    compute_rms,
    get_transfer_function,
    k_transfer,
    run_classy,
)


def test_classy_runs():
    """Test classy runs in non trivial configurations."""
    with pytest.raises(
        KeyError,
        match="You specified m_ncdm, but set N_ncdm=0",
    ):
        classy_output = run_classy(
            N_ncdm=0,
            m_ncdm="0.06",
        )

    classy_output = run_classy(
        N_ncdm=0,
        P_k_max=1.0 / units.Mpc,
    )
    assert classy_output.Om_ncdm(0) == 0.0
    assert max(classy_output.get_transfer(z=0)["k (h/Mpc)"] * classy_output.h()) < 10.0


def test_dm_transfer_extrapolation_is_positive_and_continuous(monkeypatch):
    """Test the d_m transfer stays positive and continuous across CLASS's k range.

    CLASS's d_m transfer is negative, while the in-range interpolation uses its
    absolute value; the EH extrapolation outside the CLASS k range must match it.
    """
    # Extend the default grid below CLASS's k_min so both extrapolation
    # branches are exercised.
    k_grid = np.concatenate((np.logspace(-7, -5.2, 10) / units.Mpc, k_transfer))
    monkeypatch.setattr(classy_interface, "k_transfer", k_grid)

    classy_output = run_classy(output="mTk,vTk", P_k_max=1.0 / units.Mpc)
    k_CLASS = classy_output.get_transfer(z=0)["k (h/Mpc)"] * classy_output.h()
    assert k_grid.value.min() < k_CLASS.min()
    assert k_grid.value.max() > k_CLASS.max()

    transfer = get_transfer_function(classy_output, kind="d_m").value
    assert np.all(transfer > 0)

    # No jump at either edge: the step across the boundary must be no larger
    # than twice the neighbouring in-range step (in log space).
    log_t = np.log(transfer)
    i = np.searchsorted(k_grid.value, k_CLASS.max())
    assert np.abs(log_t[i] - log_t[i - 1]) < 2 * np.abs(log_t[i - 1] - log_t[i - 2])
    i = np.searchsorted(k_grid.value, k_CLASS.min())
    assert np.abs(log_t[i] - log_t[i - 1]) < 2 * np.abs(log_t[i + 1] - log_t[i])


def test_classy_runs_with_sigma8_A_s(default_input_struct):
    """Test classy runs with sigma and A_s."""
    A_s = 3.0e-9
    sigma8 = 1.0

    classy_output = run_classy()
    assert np.isclose(
        classy_output.sigma8(), default_input_struct.cosmo_params.SIGMA_8
    )  # Not sure why they are not exactly the same
    assert np.isclose(
        classy_output.get_current_derived_parameters(["A_s"])["A_s"],
        default_input_struct.cosmo_params.A_s,
    )

    classy_output = run_classy(sigma8=sigma8)
    assert np.isclose(
        classy_output.sigma8(), sigma8
    )  # Not sure why they are not exactly the same

    classy_output = run_classy(A_s=A_s)
    assert classy_output.get_current_derived_parameters(["A_s"])["A_s"] == A_s

    with pytest.raises(
        KeyError,
        match=r"Do not provide both 'sigma8' and 'A_s' as arguments. Only one of them is allowed.",
    ):
        classy_output = run_classy(sigma8=sigma8, A_s=A_s)


def test_compute_rms(default_input_struct):
    """Test the rms function of classy."""
    # Check that CLASS returns sigma8, and that rms of v_cb at recombination is between 25 to 30 km/s
    classy_output = run_classy()
    rms_d = compute_rms(
        classy_output, kind="d_m", redshifts=0, smoothing_radius=8.0 / classy_output.h()
    )
    rms_vcb = compute_rms(
        classy_output, kind="v_cb", redshifts=1020, smoothing_radius=0
    )
    assert np.isclose(rms_d, default_input_struct.cosmo_params.SIGMA_8)
    assert np.isclose(rms_d, classy_output.sigma8())
    assert rms_vcb < 30.0 * units.km / units.s
    assert rms_vcb > 25.0 * units.km / units.s

    # Same tests as above, but with synchronous gauge
    classy_output = run_classy(gauge="synchronous")
    rms_d = compute_rms(
        classy_output,
        kind="d_m",
        redshifts=0,
        smoothing_radius=8.0 / classy_output.h() * units.Mpc,
    )
    rms_vcb = compute_rms(
        classy_output, kind="v_cb", redshifts=1020, smoothing_radius=0
    )
    assert np.isclose(rms_d, default_input_struct.cosmo_params.SIGMA_8)
    assert np.isclose(rms_d, classy_output.sigma8())
    assert rms_vcb < 30.0 * units.km / units.s
    assert rms_vcb > 25.0 * units.km / units.s


def test_compute_rms_bad_inputs():
    """Test the rms function of classy with bad inputs."""
    classy_output = run_classy()

    with pytest.raises(
        ValueError, match="'kind' can only be d_b, d_cdm, d_m, v_b, v_cdm or v_cb"
    ):
        compute_rms(
            classy_output,
            kind="X",
            redshifts=0,
            smoothing_radius=8.0 / classy_output.h() * units.Mpc,
        )

    with pytest.raises(
        ValueError, match="The units of R_smooth are not of type length!"
    ):
        compute_rms(
            classy_output,
            kind="d_m",
            redshifts=0,
            smoothing_radius=8.0 / classy_output.h() * units.s,
        )
