"""Tests of the classy module."""

import pytest
from astropy import units

from py21cmfast.wrapper.classy_interface import compute_rms, run_classy


def test_classy_runs(default_input_struct_ts):
    """Test classy runs in non trivial configurations"""
    classy_output = run_classy(
        inputs=default_input_struct_ts,
        N_ncdm=0,
        m_ncdm="0.06",
        P_k_max=1.0 / units.Mpc,
    )
    assert classy_output.Om_ncdm(0) == 0.0
    assert max(classy_output.get_transfer(z=0)["k (h/Mpc)"] * classy_output.h()) < 10.0

    classy_output = run_classy(
        inputs=default_input_struct_ts,
        N_ncdm=0,
    )
    assert classy_output.Om_ncdm(0) == 0.0


def test_compute_rms(default_input_struct_ts):
    """Test the rms function of classy"""
    # Check that CLASS returns sigma8, and that rms of v_cb at recombination is between 25 to 30 km/s
    classy_output = run_classy(
        inputs=default_input_struct_ts,
    )
    rms_d = compute_rms(
        classy_output, kind="d_m", redshifts=0, smoothing_radius=8.0 / classy_output.h()
    )
    rms_vcb = compute_rms(
        classy_output, kind="v_cb", redshifts=1020, smoothing_radius=0
    )
    assert abs(rms_d / default_input_struct_ts.cosmo_params.SIGMA_8 - 1.0) < 0.01
    assert rms_vcb < 30.0 * units.km / units.s
    assert rms_vcb > 25.0 * units.km / units.s

    # Same tests as above, but with synchronous gaguge
    classy_output = run_classy(inputs=default_input_struct_ts, gague="synchronous")
    rms_d = compute_rms(
        classy_output,
        kind="d_m",
        redshifts=0,
        smoothing_radius=8.0 / classy_output.h() * units.Mpc,
    )
    rms_vcb = compute_rms(
        classy_output, kind="v_cb", redshifts=1020, smoothing_radius=0
    )
    assert abs(rms_d / default_input_struct_ts.cosmo_params.SIGMA_8 - 1.0) < 0.01
    assert rms_vcb < 30.0 * units.km / units.s
    assert rms_vcb > 25.0 * units.km / units.s


def test_compute_rms_bad_inputs(default_input_struct_ts):
    """Test the rms function of classy with bad inputs"""
    classy_output = run_classy(
        inputs=default_input_struct_ts,
    )

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
