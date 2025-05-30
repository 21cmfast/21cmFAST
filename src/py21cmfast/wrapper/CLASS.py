"""Module for computing quantities with CLASS."""

from classy import Class
import numpy as np
import scipy.integrate as intg
from scipy.interpolate import interp1d
from astropy import units, constants
from .inputs import InputParameters
from collections.abc import Sequence


# Pivot wavenumber for primoridial power spectrum: A_s * (k/k_pivot)^{n_s-1}
k_pivot = 0.05 / units.Mpc

# This array follows the same spacing transitions in the wavenumbers
# listed in Transfers_z0.dat, but I added more samples in order to compute
# more precisely the variance
k_output = np.concatenate((
    np.logspace(-5.15,-1.49,50),
    np.logspace(-1.45,-0.258,80),
    np.logspace(-0.2083,3.049,100)
    )) / units.Mpc

CLASS_params_default = {}
CLASS_params_default["output"] = "tCl,pCl,lCl,mTk,vTk,mPk"
CLASS_params_default["tau_reio"] = 0.0554
CLASS_params_default["T_cmb"] = 2.7255 * units.K
CLASS_params_default["N_ncdm"] = 1
CLASS_params_default["m_ncdm"] = "0.06" * units.eV
CLASS_params_default["N_ur"] = 2.0308
CLASS_params_default["lensing"] = "yes"
CLASS_params_default["z_pk"] = 1087.
CLASS_params_default["l_max_scalars"] = 3000
CLASS_params_default["gauge"]= "Newtonian"
CLASS_params_default["P_k_max_1/Mpc"] = 10. / units.Mpc

def run_CLASS(
    inputs: InputParameters, 
    **kwargs
) -> Class:
    """Run CLASS with specified input parameters.

    Parameters
    ----------
    inputs: InputParameters
        The input parameters corresponding to the box.
    kwargs : 
        Optional keywords to pass to CLASS.
    
    Returns
    -------
    CLASS_output : classy.Class
        An object containing all the information from the CLASS calculation.
    """
    
    # Set CLASS parameters
    CLASS_params = {}
    CLASS_params["h"] = inputs.cosmo_params.hlittle
    CLASS_params["Omega_cdm"] = inputs.cosmo_params.OMm - inputs.cosmo_params.OMb
    CLASS_params["Omega_b"] = inputs.cosmo_params.OMb
    CLASS_params["sigma8"] = inputs.cosmo_params.SIGMA_8
    CLASS_params["n_s"] = inputs.cosmo_params.POWER_INDEX
    for k in CLASS_params_default.keys():
        if k in kwargs:
            if k == "m_ncdm" and CLASS_params["N_ncdm"] == 0:
                continue
            else:
                CLASS_params[k] = kwargs[k]
        elif k == "P_k_max_1/Mpc":
            if "P_k_max" in kwargs:
                CLASS_params["P_k_max_1/Mpc"] = kwargs["P_k_max"]
            else:
                CLASS_params[k] = CLASS_params_default[k]
        else:
            if k == "m_ncdm" and CLASS_params["N_ncdm"] == 0:
                continue
            if k == "N_ur" and CLASS_params["N_ncdm"] == 0:
                CLASS_params[k] = 3.044
            elif (k in ["lensing", "l_max_scalars"]):
                if (
                    CLASS_params["output"].find("tCl") >= 0 or 
                    CLASS_params["output"].find("pCl") >= 0 or 
                    CLASS_params["output"].find("lCl") >= 0
                ):
                    CLASS_params[k] = CLASS_params_default[k]
            else:
                CLASS_params[k] = CLASS_params_default[k]

    if not "level" in kwargs:
        kwargs["level"] = "distortions"
    # Run CLASS!
    CLASS_output = Class()
    CLASS_output.set(CLASS_params)
    CLASS_output.compute(level=kwargs["level"])

    return CLASS_output

def compute_RMS(
    CLASS_output: Class,
    kind: str = "d_m",
    redshifts: Sequence[float] = 0,
    R_smooth: float = 0
) -> Sequence[float]:
    """Compute the root-mean-square of a field at given redshifts.

    Parameters
    ----------
    CLASS_output : classy.Class
        An object containing all the information from the CLASS calculation.
    kind: str, optioanl
        The type of field for which the rms shall be computed.
        Options are:
            - "d_b", "d_cdm", "d_m": density field of baryons, cold dark matter, or all matter (including massive neutrinos).
            - "v_b", "v_cdm": magnitude of the velocity vector field of baryons or CDM (this is gauge dependent).
            - "v_cb": magnitude of the relative velocity vector field between baryons and CDM (this is gauge independent).
        Default is "d_m".
    redshifts: np.array or a float, optional
        The redshifts at which the rms shall be computed. Default is 0.
    R_smooth: float, optional
        If non-zero, the field will be smoothed with a top hat filter (in real space) with comoving radius that is set to R_smooth.
        Can also be passed as type 'astropy.units.quantity.Quantity' with length unit.
        Default is 0.
    
    Returns
    -------
    rms : np.array
        Array of the rms of the desired field at the given redshifts.
    """
    
    if hasattr(R_smooth,"unit"):
        if not R_smooth.unit.physical_type == "length":
            raise ValueError("The units of R_smooth are not of type length!")
    else:
        R_smooth *= units.Mpc
    
    if isinstance(redshifts,int) or isinstance(redshifts,float):
        redshifts = [redshifts,]

    A_s = CLASS_output.get_current_derived_parameters(["A_s"])["A_s"]
    priomordial_PS = A_s * pow(k_output/k_pivot,CLASS_output.n_s()-1.)
    rms_list = []
    for z in redshifts:
        transfers = CLASS_output.get_transfer(z=z)
        k_CLASS = transfers["k (h/Mpc)"]*CLASS_output.h() / units.Mpc
        if kind in ["d_b", "d_cdm", "d_m"]:
            transfer = transfers[kind] * units.dimensionless_unscaled
        elif kind in ["v_b", "v_cdm"]:
            try:
                kind_v = "t" + kind[1:]
                transfer = transfers[kind_v] / units.Mpc * constants.c /k_CLASS
            except KeyError: # We might get a KeyError if we are in synchronous gauge, in this case, the CDM peculiar velocity is zero
                return 0. * units.Mpc / units.s
        elif kind == "v_cb":
            try:
                transfer = (transfers["t_cdm"] - transfers["t_b"]) / units.Mpc * constants.c /k_CLASS
            except KeyError: # We might get a KeyError if we are in synchronous gauge, in this case, the CDM peculiar velocity is zero
                transfer =  - transfers["t_b"] / units.Mpc * constants.c /k_CLASS
        else:
            raise ValueError("'kind' can only be d_b, d_cdm, d_m, v_b, v_cdm or v_cb")
        # Interpolate transfer at more data points
        # Note: we lose phase information here due to the absolute value
        transfer = np.exp(interp1d(np.log(k_CLASS/k_CLASS.unit), np.log(np.abs(transfer/transfer.unit)), kind="cubic",
                                    bounds_error=False,fill_value="extrapolate")(np.log(k_output/k_output.unit))) * transfer.unit
        kr = k_output*R_smooth
        with np.errstate(divide="ignore",invalid="ignore"): # Don't show division by 0 warnings
            W_k = 3.*(np.sin(kr * units.rad)-kr*np.cos(kr * units.rad))/kr**3
        # Taylor expansion for small kr
        kr_small = kr[kr < 1.e-3]
        W_k[kr < 1.e-3]= 1.-3.*(kr_small**2)/10.

        integrand =  priomordial_PS * (transfer * W_k)**2
        var = intg.simpson(integrand, x=np.log(k_output/k_output.unit))
        rms_list.append(np.sqrt(var))
    # NOTE: intg.simpson removes the unit information, which is why we multiply by the unit when we return
    return np.array(rms_list) * transfer.unit