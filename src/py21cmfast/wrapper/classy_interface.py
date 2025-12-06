"""Module for computing quantities with CLASS."""

from collections.abc import Sequence

import numpy as np
import scipy.integrate as intg
from astropy import constants, units
from classy import Class
from scipy.interpolate import interp1d

_not4_ = 3.9715  # This is the ratio between Helium to Hydrogen mass. It is not 4!

# Pivot wavenumber for primoridial power spectrum: A_s * (k/k_pivot)^{n_s-1}
k_pivot = 0.05 / units.Mpc

# This array follows the same spacing transitions in the wavenumbers
# listed in Transfers_z0.dat, but I added more samples in order to compute
# more precisely the variance
k_transfer = (
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


def get_transfer_function(
    classy_output: Class, kind: str = "d_m", z: float = 0
) -> Sequence[float]:
    """Get the transfer function of a field at a given redshift.

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
    z: float, optional
        The redshift at which the transfer function shall be computed. Default is 0.

    Returns
    -------
    transfer : np.array
        Array of the desired transfer function at the given redshift.
    """
    transfers = classy_output.get_transfer(z=z)
    k_CLASS = transfers["k (h/Mpc)"] * classy_output.h() / units.Mpc
    if kind in {"d_b", "d_cdm", "d_m"}:
        transfer_CLASS = transfers[kind] * units.dimensionless_unscaled
    elif kind in {"v_b", "v_cdm"}:
        try:
            kind_v = f"t{kind[1:]}"
            transfer_CLASS = transfers[kind_v] / units.Mpc * constants.c / k_CLASS
        except KeyError:  # We might get a KeyError if we are in synchronous gauge, in this case, the CDM peculiar velocity is zero
            return 0.0 * units.Mpc / units.s
    elif kind == "v_cb":
        try:
            transfer_CLASS = (
                (transfers["t_cdm"] - transfers["t_b"])
                / units.Mpc
                * constants.c
                / k_CLASS
            )
        except KeyError:  # We might get a KeyError if we are in synchronous gauge, in this case, the CDM peculiar velocity is zero
            transfer_CLASS = -transfers["t_b"] / units.Mpc * constants.c / k_CLASS
    else:
        raise ValueError("'kind' can only be d_b, d_cdm, d_m, v_b, v_cdm or v_cb")

    # Interpolate transfer at more data points
    # Note: we lose phase information here due to the absolute value
    if kind == "d_m":
        needs_low_extrap = k_transfer < k_CLASS.min()
        needs_high_extrap = k_transfer > k_CLASS.max()
        in_range = ~(needs_low_extrap | needs_high_extrap)

        # Create interpolator for in-range values
        interp_func = interp1d(
            np.log(k_CLASS.value),
            np.log(np.abs(transfer_CLASS.value)),
            kind="cubic",
            bounds_error=False,
            fill_value=np.nan,
        )

        # Interpolate in-range values
        transfer_interp = np.zeros_like(k_transfer.value)
        if np.any(in_range):
            transfer_interp[in_range] = np.exp(
                interp_func(np.log(k_transfer.value[in_range]))
            )

        # Extrapolate using EH if needed
        # This matches the logic in transfer_function_CLASS (in cosmology.c) when k > kmax
        # Note that the EH transfer is multiplied by k^2, this is due to the different conventions used
        # in the definitions of the EH and CLASS transfer functions
        if np.any(needs_low_extrap) or np.any(needs_high_extrap):
            eh_transfer = EHTransferFunction(classy_output)

            if np.any(needs_high_extrap):
                # High-k extrapolation: use ratio at k_max
                T_eh_at_kmax = eh_transfer(k_CLASS.max())
                eh_ratio_at_kmax = transfer_CLASS[-1] / (
                    k_CLASS.max() ** 2 * T_eh_at_kmax
                )
                T_eh_high = eh_transfer(k_transfer[needs_high_extrap])
                transfer_interp[needs_high_extrap] = (
                    eh_ratio_at_kmax * T_eh_high * k_transfer[needs_high_extrap] ** 2
                )

            if np.any(needs_low_extrap):
                # Low-k extrapolation: use ratio at k_min
                T_eh_at_kmin = eh_transfer(k_CLASS.min())
                eh_ratio_at_kmin = transfer_CLASS[0] / (
                    k_CLASS.min() ** 2 * T_eh_at_kmin
                )
                T_eh_low = eh_transfer(k_transfer[needs_low_extrap])
                transfer_interp[needs_low_extrap] = (
                    eh_ratio_at_kmin * T_eh_low * k_transfer[needs_low_extrap] ** 2
                )
    else:
        # For non-d_m kind, use standard log-log interpolation/extrapolation
        interp_func = interp1d(
            np.log(k_CLASS.value),
            np.log(np.abs(transfer_CLASS.value)),
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        transfer_interp = np.exp(interp_func(np.log(k_transfer.value)))

    # Restore units
    return transfer_interp * transfer_CLASS.unit


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
    priomordial_PS = A_s * pow(k_transfer / k_pivot, classy_output.n_s() - 1.0)
    rms_list = []
    for z in redshifts:
        transfer = get_transfer_function(classy_output=classy_output, kind=kind, z=z)
        kr = k_transfer * smoothing_radius
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # Don't show division by 0 warnings
            W_k = 3.0 * (np.sin(kr * units.rad) - kr * np.cos(kr * units.rad)) / kr**3
        # Taylor expansion for small kr
        kr_small = kr[kr < 1.0e-3]
        W_k[kr < 1.0e-3] = 1.0 - 3.0 * (kr_small**2) / 10.0

        integrand = priomordial_PS * (transfer * W_k) ** 2
        var = intg.simpson(integrand, x=np.log(k_transfer / k_transfer.unit))
        rms_list.append(np.sqrt(var))
    # NOTE: intg.simpson removes the unit information, which is why we multiply by the unit when we return
    return np.array(rms_list) * transfer.unit


def find_redshift_kinematic_decoupling(classy_output: Class) -> float:
    """
    Find the redshift of kinematic decoupling.

    For simplicity, we approximate the redshift of kinematic decoupling to be the same redshift of recombination,
    which is defined as the moment when x_e = n_e/(n_H + n_He) = 0.1. For LCDM with Planck 2018 parameters, this corresponds
    to z_dec ~ 1070.

    Parameters
    ----------
    classy_output : :class:`classy.Class`
        An object containing all the information from the CLASS calculation.

    Returns
    -------
    z_dec : float
        Redshift of kinematic decoupling.
    """
    YHe = classy_output.get_current_derived_parameters(["YHe"])["YHe"]
    z_array = np.linspace(800, 1200, 400)
    # There is a need to multiply by n_H/(n_H+n_He)=(1-YHe)/(1-(1-1/_not4_)*YHe)
    # because CLASS returns n_e/n_H (but we want n_e/(n_H+n_He))
    x_e_array = (
        np.array([classy_output.ionization_fraction(z) for z in z_array])
        * (1.0 - YHe)
        / (1.0 - (1.0 - 1.0 / _not4_) * YHe)
    )
    z_dec = interp1d(x_e_array, z_array, kind="cubic")(0.1)
    return z_dec


class EHTransferFunction:
    """Eisenstein & Hu (1998) transfer function matching 21cmFAST implementation. This is the MDM (massive neutrino) formulation from their Eq. 29-31."""

    def __init__(self, classy_output):
        """
        Initialize EH transfer function parameters.

        Parameters
        ----------
        classy_output : :class:`classy.Class`
            An object containing all the information from the CLASS calculation.
        """
        self.Omega_m = classy_output.Omega_m()
        self.Omega_b = classy_output.Omega_b()
        self.hlittle = classy_output.h()
        self.T_cmb = classy_output.T_cmb
        self.Omega_nu = classy_output.Omega_nu()
        self.N_nu = classy_output.pars["N_ncdm"]

        self.omhh = self.Omega_m * self.hlittle**2
        self.obhh = self.Omega_b * self.hlittle**2
        self.theta_cmb = self.T_cmb / 2.7

        self.f_nu = max(self.Omega_nu / self.Omega_m, 1e-10)
        self.f_baryon = max(self.Omega_b / self.Omega_m, 1e-10)

        self._set_parameters()

    def _set_parameters(self):
        """Set constants used in the EH transfer function (matches TFset_parameters in cosmology.c)."""
        f_nu = self.f_nu
        f_b = self.f_baryon
        omhh = self.omhh
        obhh = self.obhh
        theta_cmb = self.theta_cmb

        z_equality = 25000 * omhh * pow(theta_cmb, -4) - 1.0
        k_equality = 0.0746 * omhh / (theta_cmb * theta_cmb)

        z_drag = 0.313 * pow(omhh, -0.419) * (1 + 0.607 * pow(omhh, 0.674))
        z_drag = 1 + z_drag * pow(obhh, 0.238 * pow(omhh, 0.223))
        z_drag *= 1291 * pow(omhh, 0.251) / (1 + 0.659 * pow(omhh, 0.828))

        y_d = (1 + z_equality) / (1.0 + z_drag)

        R_drag = 31.5 * obhh * pow(theta_cmb, -4) * 1000 / (1.0 + z_drag)
        R_equality = 31.5 * obhh * pow(theta_cmb, -4) * 1000 / (1.0 + z_equality)

        self.sound_horizon = (
            2.0
            / 3.0
            / k_equality
            * np.sqrt(6.0 / R_equality)
            * np.log(
                (np.sqrt(1 + R_drag) + np.sqrt(R_drag + R_equality))
                / (1.0 + np.sqrt(R_equality))
            )
        )

        p_c = -(5 - np.sqrt(1 + 24 * (1 - f_nu - f_b))) / 4.0
        p_cb = -(5 - np.sqrt(1 + 24 * (1 - f_nu))) / 4.0
        f_c = 1 - f_nu - f_b
        f_cb = 1 - f_nu
        f_nub = f_nu + f_b

        alpha_nu = (f_c / f_cb) * (2 * (p_c + p_cb) + 5) / (4 * p_cb + 5.0)
        alpha_nu *= 1 - 0.553 * f_nub + 0.126 * pow(f_nub, 3)
        alpha_nu /= 1 - 0.193 * np.sqrt(f_nu) + 0.169 * f_nu
        alpha_nu *= pow(1 + y_d, p_c - p_cb)
        alpha_nu *= 1 + (p_cb - p_c) / 2.0 * (
            1.0 + 1.0 / (4.0 * p_c + 3.0) / (4.0 * p_cb + 7.0)
        ) / (1.0 + y_d)

        self.alpha_nu = alpha_nu
        self.beta_c = 1.0 / (1.0 - 0.949 * f_nub)

    def __call__(self, k):
        """
        Compute the transfer function at wavenumber k. This matches transfer_function_EH in cosmology.c.

        Parameters
        ----------
        k : float or array-like
            Wavenumber in Mpc^-1

        Returns
        -------
        T_k : float or array-like
            Transfer function T(k)
        """
        k = np.asarray(k.value)

        q = k * pow(self.theta_cmb, 2) / self.omhh
        gamma_eff = np.sqrt(self.alpha_nu) + (1.0 - np.sqrt(self.alpha_nu)) / (
            1.0 + pow(0.43 * k * self.sound_horizon, 4)
        )
        q_eff = q / gamma_eff
        TF_m = np.log(np.e + 1.84 * self.beta_c * np.sqrt(self.alpha_nu) * q_eff)
        TF_m /= TF_m + pow(q_eff, 2) * (14.4 + 325.0 / (1.0 + 60.5 * pow(q_eff, 1.11)))
        q_nu = 3.92 * q / np.sqrt(self.f_nu / self.N_nu)
        TF_m *= 1.0 + (
            1.2 * pow(self.f_nu, 0.64) * pow(self.N_nu, 0.3 + 0.6 * self.f_nu)
        ) / (pow(q_nu, -1.6) + pow(q_nu, 0.8))

        return TF_m
