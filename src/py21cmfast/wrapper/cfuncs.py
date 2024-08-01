"""Low-level python wrappers of C functions."""

from ..c_21cmfast import ffi, lib
from .inputs import AstroParams, CosmoParams, FlagOptions, UserParams


def get_expected_nhalo(
    redshift: float,
    user_params: UserParams,
    cosmo_params: CosmoParams,
) -> int:
    """Get the expected number of halos in a given box.

    Parameters
    ----------
    redshift : float
        The redshift at which to calculate the halo list.
    user_params : :class:`~UserParams`
        User params defining the box size and resolution.
    cosmo_params : :class:`~CosmoParams`
        Cosmological parameters.
    """
    return lib.expected_nhalo(redshift, user_params(), cosmo_params())


def get_halo_list_buffer_size(
    redshift: float,
    user_params: UserParams,
    cosmo_params: CosmoParams,
    min_size: int = 1000000,
) -> int:
    """Compute the required size of the memory buffer to hold a halo list.

    Parameters
    ----------
    redshift : float
        The redshift at which to calculate the halo list.
    user_params : :class:`~UserParams`
        User params defining the box size and resolution.
    cosmo_params : :class:`~CosmoParams`
        Cosmological parameters.
    min_size : int, optional
        A minimum size to be used as the buffer.
    """
    # find the buffer size from expected halos in the box
    hbuffer_size = get_expected_nhalo(redshift, user_params, cosmo_params)
    hbuffer_size = int((hbuffer_size + 1) * user_params.MAXHALO_FACTOR)

    # set a minimum in case of fluctuation at high z
    return int(max(hbuffer_size, min_size))
