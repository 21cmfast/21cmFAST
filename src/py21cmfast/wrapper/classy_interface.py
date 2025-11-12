"""Module for computing quantities with CLASS."""

from collections.abc import Sequence

import numpy as np
import scipy.integrate as intg
from astropy import constants, units
from classy import Class
from scipy.interpolate import interp1d

# Pivot wavenumber for primoridial power spectrum: A_s * (k/k_pivot)^{n_s-1}
k_pivot = 0.05 / units.Mpc

# This array follows the same spacing transitions in the wavenumbers
# listed in Transfers_z0.dat, but I added more samples in order to compute
# more precisely the variance
k_output = (
    np.concatenate(
        (
            np.logspace(-5.15, -1.49, 50),
            np.logspace(-1.45, -0.258, 80),
            np.logspace(-0.2083, 3.049, 100),
        )
    )
    / units.Mpc
)

classy_params_default = {
    "h": 0.6766,
    "Omega_cdm": 0.11933 / 0.6766**2,
    "Omega_b": 0.02242 / 0.6766**2,
    "n_s": 0.9665,
    "sigma8": 0.8102,
    "A_s": 2.105e-9,
    "output": "tCl,pCl,lCl,mTk,vTk,mPk",
    "tau_reio": 0.0554,
    "T_cmb": 2.7255 * units.K,
    "N_ncdm": 1,
    "m_ncdm": "0.06",  # units.eV (astropy > 7.1 doesn't like units referred to str instances)
    "N_ur": 2.0308,
    "lensing": "yes",
    "z_pk": 1087.0,
    "l_max_scalars": 3000,
    "gauge": "Newtonian",
    "P_k_max_1/Mpc": 10.0 / units.Mpc,
}


def run_classy(**kwargs) -> Class:
    """Run CLASS with specified input parameters.

    Parameters
    ----------
    kwargs :
        Optional keywords to pass to CLASS.

    Returns
    -------
    output : :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
    """
    # Set CLASS parameters to be default parameters
    params = classy_params_default.copy()
    # Pop out A_s if not specified, otherwise pop out sigma8
    if "A_s" not in kwargs:
        params.pop("A_s")
    elif "sigma8" not in kwargs:
        params.pop("sigma8")
    # Raise an error if both sigma8 and A_s are specified
    else:
        raise KeyError(
            "Do not provide both 'sigma8' and 'A_s' as arguments. Only one of them is allowed."
        )
    # Raise an error if N_ncdm=0 but m_ncdm is specified
    if ("m_ncdm" in kwargs) and ("N_ncdm" in kwargs) and kwargs["N_ncdm"] == 0:
        raise KeyError("You specified m_ncdm, but set N_ncdm=0.")

    for k in kwargs:
        # "P_k_max_1/Mpc" cannot serve as a kwarg, but this is the input that CLASS expects to receive,
        # so we control this input with "P_k_max" instead
        if k == "P_k_max":
            params["P_k_max_1/Mpc"] = kwargs["P_k_max"]
        else:
            params[k] = kwargs[k]

    # Set N_ur=3.044 and pop out m_ncdm if N_ncdm=0 (no massive neutrinos)
    if params["N_ncdm"] == 0:
        params["N_ur"] = 3.044
        params.pop("m_ncdm")

    # If we don't need to evaluate CMB anisotropies, we don't need these kwargs in params
    if not (
        params["output"].find("tCl") >= 0
        or params["output"].find("pCl") >= 0
        or params["output"].find("lCl") >= 0
    ):
        params.pop("lensing")
        params.pop("l_max_scalars")

    # Set level to highest order, unless it is specified in kwargs
    if "level" not in kwargs:
        kwargs["level"] = ["distortions"]

    # Run CLASS!
    output = Class()
    output.set(params)
    output.compute(level=kwargs["level"])

    return output


def compute_rms(
    classy_output: Class,
    kind: str = "d_m",
    redshifts: Sequence[float] = 0,
    smoothing_radius: float = 0,
) -> Sequence[float]:
    """Compute the root-mean-square of a field at given redshifts.

    Parameters
    ----------
    classy_output : :class:`classy.Class`
        An object containing all the information from the CLASS calculation.
    kind: str, optioanl
        The type of field for which the rms shall be computed.
        Options are:
            - "d_b", "d_cdm", "d_m": density field of baryons, cold dark matter, or all
              matter (including massive neutrinos).
            - "v_b", "v_cdm": magnitude of the velocity vector field of baryons or CDM
              (this is gauge dependent).
            - "v_cb": magnitude of the relative velocity vector field between baryons
              and CDM (this is gauge independent).
        Default is "d_m".
    redshifts: np.array or a float, optional
        The redshifts at which the rms shall be computed. Default is 0.
    smoothing_radius: float, optional
        If non-zero, the field will be smoothed with a top hat filter (in real space) with comoving radius that is set to R_smooth.
        Can also be passed as type 'astropy.units.quantity.Quantity' with length unit.
        Default is 0.

    Returns
    -------
    rms : np.array
        Array of the rms of the desired field at the given redshifts.
    """
    if hasattr(smoothing_radius, "unit"):
        if smoothing_radius.unit.physical_type != "length":
            raise ValueError("The units of R_smooth are not of type length!")
    else:
        smoothing_radius *= units.Mpc

    if isinstance(redshifts, int | float):
        redshifts = [redshifts]

    A_s = classy_output.get_current_derived_parameters(["A_s"])["A_s"]
    priomordial_PS = A_s * pow(k_output / k_pivot, classy_output.n_s() - 1.0)
    rms_list = []
    for z in redshifts:
        transfers = classy_output.get_transfer(z=z)
        k_CLASS = transfers["k (h/Mpc)"] * classy_output.h() / units.Mpc
        if kind in {"d_b", "d_cdm", "d_m"}:
            transfer = transfers[kind] * units.dimensionless_unscaled
        elif kind in {"v_b", "v_cdm"}:
            try:
                kind_v = f"t{kind[1:]}"
                transfer = transfers[kind_v] / units.Mpc * constants.c / k_CLASS
            except KeyError:  # We might get a KeyError if we are in synchronous gauge, in this case, the CDM peculiar velocity is zero
                return 0.0 * units.Mpc / units.s
        elif kind == "v_cb":
            try:
                transfer = (
                    (transfers["t_cdm"] - transfers["t_b"])
                    / units.Mpc
                    * constants.c
                    / k_CLASS
                )
            except KeyError:  # We might get a KeyError if we are in synchronous gauge, in this case, the CDM peculiar velocity is zero
                transfer = -transfers["t_b"] / units.Mpc * constants.c / k_CLASS
        else:
            raise ValueError("'kind' can only be d_b, d_cdm, d_m, v_b, v_cdm or v_cb")
        # Interpolate transfer at more data points
        # Note: we lose phase information here due to the absolute value
        transfer = (
            np.exp(
                interp1d(
                    np.log(k_CLASS / k_CLASS.unit),
                    np.log(np.abs(transfer / transfer.unit)),
                    kind="cubic",
                    bounds_error=False,
                    fill_value="extrapolate",
                )(np.log(k_output / k_output.unit))
            )
            * transfer.unit
        )
        kr = k_output * smoothing_radius
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # Don't show division by 0 warnings
            W_k = 3.0 * (np.sin(kr * units.rad) - kr * np.cos(kr * units.rad)) / kr**3
        # Taylor expansion for small kr
        kr_small = kr[kr < 1.0e-3]
        W_k[kr < 1.0e-3] = 1.0 - 3.0 * (kr_small**2) / 10.0

        integrand = priomordial_PS * (transfer * W_k) ** 2
        var = intg.simpson(integrand, x=np.log(k_output / k_output.unit))
        rms_list.append(np.sqrt(var))
    # NOTE: intg.simpson removes the unit information, which is why we multiply by the unit when we return
    return np.array(rms_list) * transfer.unit
