"""Module for accounting redshift space distortions."""

import numpy as np
from astropy import units
from scipy import fft
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
import warnings
try:
    from numba import njit

    NUMBA = True
except ImportError:
    NUMBA = False

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
            interpolate_to_subcells=False,
            interpolate_to_supercells=False,
            periodic=periodic,
        ).T
        
        # And now we transform back to a rectilinear-like lightcone
        if len(brightness_temp.shape) == 3:
            tb_with_rsds = tb_with_rsds.reshape((int(np.sqrt(tb_with_rsds.shape[0])), int(np.sqrt(tb_with_rsds.shape[0])), -1))

    return tb_with_rsds


def apply_rsds(
    field: np.ndarray,
    los_displacement: np.ndarray,
    distance: np.ndarray,
    n_subcells: int = 4,
    interpolate_to_subcells: bool = True,
    interpolate_to_supercells: bool = True,
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
    """
    if field.shape != los_displacement.shape:
        raise ValueError("field and los_displacement must have the same shape")
    if field.shape[0] < 2:
        raise ValueError("field must have at least 2 slices")
    if field.shape[0] != distance.size:
        raise ValueError("field and distance must have the same number of slices")

    is_regular = np.allclose(np.diff(np.diff(distance)), 0.0)
    interpolator = RegularGridInterpolator if is_regular else RectBivariateSpline

    ang_coords = np.arange(field.shape[1])

    smallest_slice = np.min(np.diff(distance))
    rsd_dx = smallest_slice / n_subcells

    if periodic:
        distance_plus = np.append(distance,2*distance[-1]-distance[-2])
        distance_fine = np.linspace(distance_plus.min(),distance_plus.max(),1+n_subcells*(len(distance_plus)-1))
        fine_grid = (distance_fine[1:]+distance_fine[:-1])/2
        distance = (distance_plus[1:]+distance_plus[:-1])/2
    else:
        # TODO: convert these to distances...
        vmax_towards_observer = max(np.max(los_displacement[0]), 0)
        vmax_away_from_observer = min(0, np.min(los_displacement[-1]))
        fine_grid = np.arange(
            (distance.min() - vmax_towards_observer).to_value(rsd_dx.unit),
            (distance.max() - vmax_away_from_observer).to_value(rsd_dx.unit),
            rsd_dx.value,
        )
        if fine_grid.max() < distance.max().to_value(rsd_dx.unit):
            fine_grid = np.append(fine_grid, distance.max().to_value(rsd_dx.unit))

    if is_regular:
        x, y = np.meshgrid(fine_grid, ang_coords, indexing="ij")
        grid = (x.flatten(), y.flatten())
        if interpolate_to_subcells:
            fine_field = interpolator(
                (distance, ang_coords), field, bounds_error=False, fill_value=None, method="linear",
            )(grid).reshape(x.shape)
        else:
            fine_field = np.repeat(field, n_subcells, axis=0) / n_subcells

        fine_rsd = interpolator(
            (distance, ang_coords),
            los_displacement / rsd_dx,
            bounds_error=False,
            fill_value=None,
            method="linear",
        )(grid).reshape(x.shape)
    else:
        fine_field = interpolator(distance, ang_coords, field)(fine_grid, ang_coords)
        fine_rsd = interpolator(distance, ang_coords, los_displacement / rsd_dx)(
            fine_grid, ang_coords
        )

    fine_field = cloud_in_cell_los(fine_field, fine_rsd,periodic=periodic)
    
    if interpolate_to_supercells:
        x, y = np.meshgrid(distance, ang_coords, indexing="ij")
        return RegularGridInterpolator((fine_grid, ang_coords), fine_field)(
            (
                x.flatten(),
                y.flatten(),
            )
        ).reshape(x.shape)
    else:
        # Average the subcells back to the original grid
        return np.sum(fine_field.T.reshape(len(ang_coords),len(distance),n_subcells),axis=-1).T
    
def cloud_in_cell_los(
    field: np.ndarray,
    delta_los: np.ndarray,
    periodic: bool = False) -> np.ndarray:
    """
    Interpolate in the line-of-sight direction using cloud-in-cell algorithm.

    Note that the implementation was derived largely from
    https://astro.uchicago.edu/~andrey/Talks/PM/pm.pdf, specially slide 13.

    Notes
    -----
    The ``field`` here is assumed to be regularly spaced in comoving distance along the
    line-of-sight (the angular coordinates can be arbitrarily arranged). For each angular
    point, we first displace the regular grid along the line-of-sight by ``delta_los``,
    to create a a new, non-regular grid (which we can consider to be "particles"). We
    then use the regular cloud-in-cell interpolation to interpolate the particles back
    on to the regular grid.

    Parameters
    ----------
    field
        The regularly-spaced (along LoS) field before displacement by delta_los.
        Shape ``(nlos_slices, nangles)``.
    delta_los
        Displacement of each coordinate in the field along the LoS.
        The displacement must be in units of the regular grid size, i.e.
        ``v / H(z) / grid_resolution``. Same shape as ``field``.
    """
    if not NUMBA:  # pragma: no cover
        warnings.warn("Install numba for a speedup of cloud_in_cell", stacklevel=2)

    if field.shape != delta_los.shape:
        raise ValueError("Field and displacement must have the same shape.")

    out = np.zeros_like(field)

    nslice, nangles = delta_los.shape
    for ii in range(nslice):
        weight = field[ii]

        # Get the offset of this grid cell
        ddx = delta_los[ii]
        x = ii + ddx

        i = x.astype(np.int32)
        ip = i + 1

        tx = ip - x
        ddx = 1 - tx

        for jj in range(nangles):
            if not periodic:
                if 0 <= i[jj] < nslice:
                    out[i[jj], jj] += tx[jj] * weight[jj]
                if 0 <= ip[jj] < nslice:
                    out[ip[jj], jj] += ddx[jj] * weight[jj]
            else:
                out[i[jj]%nslice, jj] += tx[jj] * weight[jj]
                out[ip[jj]%nslice, jj] += ddx[jj] * weight[jj]
    return out

if NUMBA:
    cloud_in_cell_los = njit(cloud_in_cell_los)