"""Tests of the classy module."""

import numpy as np
import pytest
from astropy import units

from py21cmfast.wrapper.classy_interface import (
    classy_params_default,
    compute_rms,
    run_classy,
)


def test_classy_runs():
    """Test classy runs in non trivial configurations."""
    classy_output = run_classy(
        N_ncdm=0,
        m_ncdm="0.06",
        P_k_max=1.0 / units.Mpc,
    )
    assert classy_output.Om_ncdm(0) == 0.0
    assert max(classy_output.get_transfer(z=0)["k (h/Mpc)"] * classy_output.h()) < 10.0

    classy_output = run_classy(
        N_ncdm=0,
    )
    assert classy_output.Om_ncdm(0) == 0.0


def test_classy_runs_with_sigma8_A_s():
    """Test classy runs with sigma and A_s."""
    A_s = 3.0e-9
    sigma8 = 1.0

    classy_output = run_classy()
    assert np.isclose(
        classy_output.sigma8(), classy_params_default["sigma8"]
    )  # Not sure why they are not exactly the same
    assert np.isclose(
        classy_output.get_current_derived_parameters(["A_s"])["A_s"],
        classy_params_default["A_s"],
    )

    classy_output = run_classy(sigma8=sigma8)
    assert np.isclose(
        classy_output.sigma8(), sigma8
    )  # Not sure why they are not exactly the same

    classy_output = run_classy(A_s=A_s)
    assert classy_output.get_current_derived_parameters(["A_s"])["A_s"] == A_s

    with pytest.raises(
        KeyError,
        match="Do not provide both 'sigma8' and 'A_s' as arguments. Only one of them is allowed.",
    ):
        classy_output = run_classy(sigma8=sigma8, A_s=A_s)


def test_classy_runs_with_inputs(default_input_struct):
    """Test classy runs with inputs."""
    classy_output = run_classy(
        inputs=default_input_struct,
    )
    assert classy_output.h() == default_input_struct.cosmo_params.hlittle
    assert classy_output.Omega_b() == default_input_struct.cosmo_params.OMb
    assert np.isclose(
        classy_output.Omega_m(),
        default_input_struct.cosmo_params.OMm,
        rtol=1e-2,
        atol=1e-2,
    )  # They are not the same because we ignore in 21cmFAST the contribution from massive neutrinos!
    assert classy_output.n_s() == default_input_struct.cosmo_params.POWER_INDEX
    assert np.isclose(
        classy_output.sigma8(), default_input_struct.cosmo_params.SIGMA_8
    )  # Not sure why they are not exactly the same

    classy_output = run_classy(inputs=default_input_struct, N_ncdm=0)
    assert (
        classy_output.Omega_m() == default_input_struct.cosmo_params.OMm
    )  # Now they are the same! (because we set all neutrinos to be massless)

    A_s = 2.105e-9
    classy_output = run_classy(
        inputs=default_input_struct,
        A_s=A_s,
    )
    assert classy_output.get_current_derived_parameters(["A_s"])["A_s"] == A_s

    # We check below that the kwargs overwrite the inputs
    classy_output = run_classy(
        inputs=default_input_struct,
        h=0.7,
    )
    assert classy_output.h() != default_input_struct.cosmo_params.hlittle

    classy_output = run_classy(
        inputs=default_input_struct,
        sigma8=1.0,
    )
    assert not np.isclose(
        classy_output.sigma8(), default_input_struct.cosmo_params.SIGMA_8
    )

    classy_output = run_classy(
        inputs=default_input_struct,
        A_s=3.0e-9,
    )
    assert (
        classy_output.get_current_derived_parameters(["A_s"])["A_s"]
        != default_input_struct.cosmo_params.A_s
    )


def test_compute_rms(default_input_struct):
    """Test the rms function of classy."""
    # Check that CLASS returns sigma8, and that rms of v_cb at recombination is between 25 to 30 km/s
    classy_output = run_classy(
        inputs=default_input_struct,
    )
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
    classy_output = run_classy(inputs=default_input_struct, gauge="synchronous")
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
