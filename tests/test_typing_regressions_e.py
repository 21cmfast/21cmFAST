"""Regression tests for bugs found while adding type annotations (cfuncs, photoncons, CLASS)."""

import astropy.units as un
import numpy as np
import pytest

import py21cmfast as p21c
from py21cmfast.wrapper import classy_interface as ci
from py21cmfast.wrapper import photoncons
from py21cmfast.wrapper.exceptions import PHOTONCONSERROR, PhotonConsError


class _LibWithFailing:
    """Forward to the real C library, except for one function that fails."""

    def __init__(self, real, failing_name: str):
        self._real = real

        def failing(*args):
            return PHOTONCONSERROR

        failing.__name__ = failing_name
        setattr(self, failing_name, failing)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_init_photoncons_raises_on_c_error(monkeypatch, default_input_struct):
    """An error status from InitialisePhotonCons must not be silently ignored."""
    monkeypatch.setattr(
        photoncons, "lib", _LibWithFailing(photoncons.lib, "InitialisePhotonCons")
    )
    with pytest.raises(PhotonConsError):
        photoncons._init_photon_conservation_correction(inputs=default_input_struct)


def test_calibrate_photoncons_raises_on_c_error(monkeypatch):
    """An error status from PhotonCons_Calibration must not be silently ignored."""
    monkeypatch.setattr(
        photoncons, "lib", _LibWithFailing(photoncons.lib, "PhotonCons_Calibration")
    )
    with pytest.raises(PhotonConsError):
        photoncons._calibrate_photon_conservation_correction(
            redshifts_estimate=[6.0, 7.0, 8.0],
            nf_estimate=[0.1, 0.5, 0.9],
            NSpline=3,
        )


def test_get_photoncons_dz(tiny_inputs, tiny_ics):
    """get_photoncons_dz used to call the C function with a missing argument."""
    inputs = tiny_inputs.evolve_input_structs(PHOTON_CONS_TYPE="z-photoncons")
    p21c.setup_photon_cons(initial_conditions=tiny_ics, inputs=inputs)

    adjusted_z, stored_z, deltaz = photoncons.get_photoncons_dz(
        inputs=inputs, redshift=12.0
    )
    for val in (adjusted_z, stored_z, deltaz):
        assert isinstance(val, float)
        assert np.isfinite(val)


@pytest.fixture(scope="module")
def classy_output_low_kmax():
    # A low k_max, so that the transfer function has to be extrapolated at high k.
    return ci.run_classy(output="mTk,vTk", P_k_max=1.0 / un.Mpc)


def test_kinematic_decoupling_redshift_is_float(classy_output_low_kmax):
    z_dec = ci.find_redshift_kinematic_decoupling(classy_output_low_kmax)
    assert isinstance(z_dec, float)
    assert 1000 < z_dec < 1200
