"""Module for accounting redshift space distortions."""

from pathlib import Path
import numpy as np
from astropy import units
from scipy import fft
from cosmotile import apply_rsds

from .inputs import InputParameters

def compute_rsds(
        brightness_temp: np.ndarray,
        los_velocity: np.ndarray,
        redshifts: np.ndarray,
        distances: units.Quantity,
        inputs: InputParameters,
        tau_21: np.ndarray = None,
        periodic: bool = None,
        n_subcells: int = None,
    ):
    """Compute redshift-space distortions from the los_velocity lightcone.
    This includes both modification to the optical depth due to a velocity gradient
    and shifting the cells of the apparent brightness temperature (see Mao+ 2012).

    Parameters
    ----------
    brightness_temp: nd-array
        A box of the brightness temperature, without redshift space distortions.
    los_velocity: nd-array
        A box of the los velocity.
    redshifts: nd-array
        An array of the redshifts along the los. Can also be a float (could be useful for coeval boxes)
    distances: nd-array
        An array of the comoving distances along the los.
    inputs: InputParameters
        The input parameters corresponding to the box.
    tau_21: nd-array, optioanl
        A box of the 21cm optical depth. Not required if inputs.astro_options.USE_TS_FLUCT = False.
    periodic: bool, optioanl
        Whether to assume periodic boundary conditions along the line-of-sight.
    n_subcells: int, optional
        The number of sub-cells to interpolate onto, to make the RSDs more accurate. Default is inputs.astro_options.N_RSD_STEPS.
    
    Returns
    -------
    tb_with_rsds : nd-array
        A box of the brightness temperature, with redshift space distortions.
    """
    if tau_21 is None and inputs.astro_options.USE_TS_FLUCT:
        raise ValueError(
            "tau_21 is not provided, but inputs.astro_options.USE_TS_FLUCT is True!"
        )
    
    if periodic is None:
        periodic = brightness_temp.shape[0] == brightness_temp.shape[-1]
    
    # TODO: currently the gradient is applied w.r.t to the z-axis, even if the user specified a different los-axis. Needs to make it more flexible in the future
    if periodic:
        N = inputs.simulation_options.HII_DIM
        L = inputs.simulation_options.BOX_LEN
        k_real = fft.rfftfreq(N,1/N)*2.*np.pi/L
        k_complex = fft.fftfreq(N,1/N)*2.*np.pi/L
        k_vector = np.stack(np.meshgrid(k_complex,k_complex,k_real,indexing="ij"))
        vel_gradient = fft.irfftn(1j*k_vector[-1]*fft.rfftn(los_velocity),s=(N,N,N)) / units.s
    else:
        vel_gradient = np.gradient(los_velocity * units.Mpc / units.s, distances, axis=-1, edge_order=2)
    
    H = inputs.cosmo_params.cosmo.H(redshifts)
    max_v_deriv = inputs.astro_params.MAX_DVDR * H
    dvdx = np.clip(vel_gradient,-max_v_deriv,max_v_deriv)
    gradient_component = 1. + dvdx/H

    if not inputs.astro_options.USE_TS_FLUCT:
        tb_with_rsds = brightness_temp / gradient_component
    else:
        tb_no_rsds = brightness_temp
        tau_21 = np.float64(tau_21)
        gradient_component = np.float64(gradient_component)
        with np.errstate(divide="ignore", invalid="ignore"): # Don't show division by 0 warnings
            rsd_factor = (1.0 - np.exp(-tau_21/gradient_component))/(1.0 - (np.exp(-tau_21)))
        # It doesn't really matter what the rsd_factor is when tau_21=0 because the brightness temperature is zero by definition
        rsd_factor = np.float32(np.where(tau_21 < 1e-10, 1., rsd_factor))
        tb_with_rsds = tb_no_rsds*rsd_factor

    if n_subcells is None:
        if inputs.astro_options.SUBCELL_RSD:
            n_subcells = inputs.astro_params.N_RSD_STEPS
        else:
            n_subcells = 0

    # Compute the local RSDs
    if n_subcells > 0:

        los_displacement = los_velocity * units.Mpc / units.s / H
        equiv = units.pixel_scale(inputs.simulation_options.cell_size / units.pixel)
        los_displacement = los_displacement.to(units.pixel, equivalencies=equiv)

        # We transform rectilinear lightcone to be an angular-like lightcone
        if len(brightness_temp.shape) == 3:
            tb_with_rsds = tb_with_rsds.reshape((tb_with_rsds.shape[0]*tb_with_rsds.shape[1], -1))
            los_displacement = los_displacement.reshape((los_displacement.shape[0]*los_displacement.shape[1], -1))

        # Here we move the cells along the line of sight, regardless the geometry (rectilinear or angular)
        tb_with_rsds = apply_rsds(
            field=tb_with_rsds.T,
            los_displacement=los_displacement.T,
            distance=distances.to(units.pixel, equiv),
            n_subcells=n_subcells,
        ).T
        
        # And now we transform back to a rectilinear-like lightcone
        if len(brightness_temp.shape) == 3:
            tb_with_rsds = tb_with_rsds.reshape((int(np.sqrt(tb_with_rsds.shape[0])), int(np.sqrt(tb_with_rsds.shape[0])), -1))

    return tb_with_rsds
