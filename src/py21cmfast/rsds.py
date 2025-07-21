"""Module for accounting redshift space distortions."""

from collections.abc import Sequence

import numpy as np
from astropy import cosmology, units
from classy import Class
from cosmotile.cic import cloud_in_cell_los
from scipy import fft
from scipy.interpolate import RegularGridInterpolator

from .wrapper.classy_interface import compute_rms
from .wrapper.inputs import InputParameters


def compute_rsds(
    brightness_temp: np.ndarray,
    los_velocity: np.ndarray,
    redshifts: np.ndarray | float,
    inputs: InputParameters,
    tau_21: np.ndarray | None = None,
    periodic: bool | None = None,
    n_subcells: int | None = None,
) -> np.ndarray:
    """Compute redshift-space distortions from the los_velocity lightcone.

    This includes both modification to the optical depth due to a velocity gradient
    and shifting the cells of the apparent brightness temperature (see Mao+ 2012).

    Parameters
    ----------
    brightness_temp
        A box of the brightness temperature, without redshift space distortions.
    los_velocity
        Line-of-sight velocities for each cell of brightness_temp, in Mpc/second.
    redshifts
        An array of the redshifts along the los. Can also be a float (could be useful for coeval boxes)
    inputs
        The input parameters corresponding to the box.
    tau_21
        A box of the 21cm optical depth. Not required if inputs.astro_options.USE_TS_FLUCT = False.
    periodic
        Whether to assume periodic boundary conditions along the line-of-sight.
    n_subcells
        The number of sub-cells to interpolate onto, to make the RSDs more accurate. Default is inputs.astro_options.N_RSD_STEPS.

    Returns
    -------
    tb_with_rsds
        A box of the brightness temperature, with redshift space distortions.
    """
    if tau_21 is None and inputs.astro_options.USE_TS_FLUCT:
        raise ValueError(
            "tau_21 is not provided, but inputs.astro_options.USE_TS_FLUCT is True!"
        )
    if hasattr(redshifts, "__len__") and len(redshifts) != brightness_temp.shape[-1]:
        raise ValueError(
            "Redshifts must be a float or array with the same size as number of LoS slices"
        )

    N = inputs.simulation_options.HII_DIM

    if periodic is None:
        # assume it's periodic if it looks like a coeval box
        periodic = brightness_temp.shape == (N, N, N)

    dx_los = inputs.simulation_options.BOX_LEN / N  # Mpc
    # TODO: currently the gradient is applied w.r.t to the z-axis, even if the user specified a different los-axis. Needs to make it more flexible in the future
    if periodic:
        k_real = fft.rfftfreq(N, dx_los) * 2.0 * np.pi
        k_complex = fft.fftfreq(N, dx_los) * 2.0 * np.pi
        k_vector = np.stack(np.meshgrid(k_complex, k_complex, k_real, indexing="ij"))
        vel_gradient = (
            fft.irfftn(1j * k_vector[-1] * fft.rfftn(los_velocity), s=(N, N, N))
            / units.s
        )
    else:
        vel_gradient = np.gradient(
            los_velocity * units.Mpc / units.s,
            dx_los * units.Mpc,
            axis=-1,
            edge_order=2,
        )

    H = inputs.cosmo_params.cosmo.H(redshifts)

    if not inputs.astro_options.USE_TS_FLUCT:
        # If we don't have spin temperature, we also assume tau_21 << 1 and make the Taylor approximation.
        # Clipping is required so the brightness temperature won't diverge when the gradient term goes to small values
        max_v_deriv = inputs.astro_params.MAX_DVDR * H
        dvdx = np.clip(vel_gradient, -max_v_deriv, max_v_deriv)
        gradient_component = np.abs(1.0 + dvdx / H)
        tb_with_rsds = brightness_temp / gradient_component
    else:
        # We have the spin temperature, and we do *not* make the Taylor approximation
        tb_no_rsds = brightness_temp.copy()
        tau_21 = np.float64(tau_21)
        gradient_component = np.float64(np.abs(1.0 + vel_gradient / H))
        with np.errstate(
            divide="ignore", invalid="ignore"
        ):  # Don't show division by 0 warnings
            rsd_factor = (1.0 - np.exp(-tau_21 / gradient_component)) / (
                1.0 - (np.exp(-tau_21))
            )
        # It doesn't really matter what the rsd_factor is when tau_21=0 because the brightness temperature is zero by definition
        rsd_factor = np.float32(np.where(tau_21 < 1e-10, 1.0, rsd_factor))
        tb_with_rsds = tb_no_rsds * rsd_factor

    if n_subcells is None:
        if inputs.astro_options.SUBCELL_RSD:
            n_subcells = inputs.astro_params.N_RSD_STEPS
        else:
            n_subcells = 0

    # Compute the local RSDs
    if n_subcells > 0:
        los_displacement = los_velocity * units.Mpc / units.s / H
        equiv = units.pixel_scale(dx_los * units.Mpc / units.pixel)
        los_displacement = los_displacement.to(units.pixel, equivalencies=equiv)

        # We transform rectilinear lightcone to be an angular-like lightcone
        if len(brightness_temp.shape) == 3:
            tb_with_rsds = tb_with_rsds.reshape(
                (tb_with_rsds.shape[0] * tb_with_rsds.shape[1], -1)
            )
            los_displacement = los_displacement.reshape(
                (los_displacement.shape[0] * los_displacement.shape[1], -1)
            )

        # Here we move the cells along the line of sight, regardless the geometry (rectilinear or angular)
        tb_with_rsds = apply_rsds(
            field=tb_with_rsds.T,
            los_displacement=los_displacement.T,
            n_subcells=n_subcells,
            periodic=periodic,
        ).T

        # And now we transform back to a rectilinear-like lightcone
        if len(brightness_temp.shape) == 3:
            tb_with_rsds = tb_with_rsds.reshape(
                (
                    int(np.sqrt(tb_with_rsds.shape[0])),
                    int(np.sqrt(tb_with_rsds.shape[0])),
                    -1,
                )
            )

    return tb_with_rsds


def apply_rsds(
    field: np.ndarray,
    los_displacement: np.ndarray,
    #    distance: np.ndarray,
    n_subcells: int = 4,
    periodic: bool = False,
) -> np.ndarray:
    """Apply redshift-space distortions to a field.

    Notes
    -----
    To ensure that we cover all the slices in the field after the velocities have
    been applied, we extrapolate the densities and velocities on either end by the
    maximum velocity offset in the field.
    Then, to ensure we don't pick up cells with zero particles (after displacement),
    we interpolate the slices onto a finer regular grid (in comoving distance) and
    then displace the field on that grid, before interpolating back onto the original
    slices.

    Parameters
    ----------
    field
        The field to apply redshift-space distortions to, shape (nslices, ncoords).
    los_displacement
        The line-of-sight "apparent" displacement of the field, in pixel coordinates.
        Equal to ``v / H(z) / cell_size``.
        Positive values are towards the observer, shape ``(nslices, ncoords)``.
    distance
        The comoving distance to each slice in the field, in units of the cell size.
        shape (nslices,).
    periodic: bool, optioanl
        Whether to assume periodic boundary conditions along the line-of-sight.
    n_subcells: int, optional
        The number of sub-cells to interpolate onto, to make the RSDs more accurate. Default is inputs.astro_options.N_RSD_STEPS.
    """
    if field.shape[0] < 2:
        raise ValueError("field must have at least 2 slices")

    ang_coords = np.arange(field.shape[1])

    distance = np.arange(field.shape[0])

    # We need to extend the grid once as we would like to interpret the displacement values
    # to be associated with the *centers* of the cells, as was previously done in the C code
    distance_plus = np.arange(field.shape[0] + 1)

    if periodic:
        # We extend the grid twice more to account for periodic boundary conditions
        distance_plus_periodic = np.arange(-1, field.shape[0] + 2)
        distance_grid = (distance_plus_periodic[1:] + distance_plus_periodic[:-1]) / 2
        # We also extend the displacement array to have periodic boundary conditions
        first_slice = los_displacement[-1, :].reshape(1, len(ang_coords))
        last_slice = los_displacement[0, :].reshape(1, len(ang_coords))
        los_displacement = np.concatenate(
            (first_slice, los_displacement, last_slice), axis=0
        )

    else:
        distance_grid = (distance_plus[1:] + distance_plus[:-1]) / 2

    fine_field = np.repeat(field, n_subcells, axis=0) / n_subcells

    # This is where we shall evaluate the subcells
    distance_fine = np.linspace(
        distance_plus.min(),
        distance_plus.max(),
        1 + n_subcells * (len(distance_plus) - 1),
    )
    fine_grid = (distance_fine[1:] + distance_fine[:-1]) / 2
    x, y = np.meshgrid(fine_grid, ang_coords, indexing="ij")
    grid = (x.flatten(), y.flatten())
    fine_rsd = RegularGridInterpolator(
        (distance_grid, ang_coords),
        los_displacement * n_subcells,
        bounds_error=False,
        fill_value=None,
        method="linear",
    )(grid).reshape(x.shape)

    fine_field = cloud_in_cell_los(fine_field, fine_rsd, periodic=periodic)
    # Average the subcells back to the original grid
    return np.sum(
        fine_field.T.reshape(len(ang_coords), len(distance), n_subcells), axis=-1
    ).T


def estimate_rsd_displacements(
    classy_output: Class,
    cosmo: cosmology,
    redshifts: Sequence[float],
    factor: float = 1.0,
) -> Sequence[units.Quantity]:
    """Estimate the rms of the redshift space distortions displacement field at given redshifts.

    Parameters
    ----------
    classy_output : classy.Class
        An object containing all the information from the CLASS calculation.
    cosmo: astropy.cosmology
        The assumed cosmology
    redshifts: list
        List of the redshifts at which the rms of the displacement field is computed.
    factor: float, optional
        Factor to multiply the rms velocity from CLASS. Default is 1.

    Returns
    -------
    displacements : np.array
        The rms of the RSD displacement field.
    """
    v_rms = (
        compute_rms(classy_output=classy_output, kind="v_b", redshifts=redshifts)
        * factor
    )
    # The multiplication by (1+z)=1/a is because CLASS returns the *proper* velocity field,
    # while we work in *comoving* coordinates
    return v_rms / cosmo.H(redshifts) * (1.0 + redshifts)
