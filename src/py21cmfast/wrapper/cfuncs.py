"""Low-level python wrappers of C functions."""

import logging
import numpy as np
from functools import cache
from scipy.interpolate import interp1d
from typing import Literal, Sequence

from ..c_21cmfast import ffi, lib
from ..drivers.param_config import InputParameters
from ._utils import _process_exitcode
from .inputs import AstroParams, CosmoParams, FlagOptions, UserParams

logger = logging.getLogger(__name__)


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
    return lib.expected_nhalo(redshift, user_params.cstruct, cosmo_params.cstruct)


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


def compute_tau(
    *,
    redshifts: Sequence[float],
    global_xHI: Sequence[float],
    user_params: UserParams | dict | None = None,
    cosmo_params: CosmoParams | dict | None = None,
) -> float:
    """Compute the optical depth to reionization under the given model.

    Parameters
    ----------
    redshifts : array-like
        Redshifts defining an evolution of the neutral fraction.
    global_xHI : array-like
        The mean neutral fraction at `redshifts`.
    user_params : :class:`~inputs.UserParams`
        Parameters defining the simulation run.
    cosmo_params : :class:`~inputs.CosmoParams`
        Cosmological parameters.

    Returns
    -------
    tau : float
        The optional depth to reionization

    Raises
    ------
    ValueError :
        If `redshifts` and `global_xHI` have inconsistent length or if redshifts are not
        in ascending order.
    """
    inputs = InputParameters(user_params, cosmo_params)

    if len(redshifts) != len(global_xHI):
        raise ValueError("redshifts and global_xHI must have same length")

    if not np.all(np.diff(redshifts) > 0):
        raise ValueError("redshifts and global_xHI must be in ascending order")

    # Convert the data to the right type
    redshifts = np.array(redshifts, dtype="float32")
    global_xHI = np.array(global_xHI, dtype="float32")

    z = ffi.cast("float *", ffi.from_buffer(redshifts))
    xHI = ffi.cast("float *", ffi.from_buffer(global_xHI))

    # Run the C code
    return lib.ComputeTau(
        inputs.user_params.cstruct, inputs.cosmo_params.cstruct, len(redshifts), z, xHI
    )


def compute_luminosity_function(
    *,
    redshifts: Sequence[float],
    user_params: UserParams | dict | None = None,
    cosmo_params: CosmoParams | dict | None = None,
    astro_params: AstroParams | dict | None = None,
    flag_options: FlagOptions | dict | None = None,
    nbins: int = 100,
    mturnovers: np.ndarray | None = None,
    mturnovers_mini: np.ndarray | None = None,
    component: Literal["both", "acg", "mcg"] = "both",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a the luminosity function over a given number of bins and redshifts.

    Parameters
    ----------
    redshifts : array-like
        The redshifts at which to compute the luminosity function.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    astro_params : :class:`~AstroParams`, optional
        The astrophysical parameters defining the course of reionization.
    flag_options : :class:`~FlagOptions`, optional
        Some options passed to the reionization routine.
    nbins : int, optional
        The number of luminosity bins to produce for the luminosity function.
    mturnovers : array-like, optional
        The turnover mass at each redshift for massive halos (ACGs).
        Only required when USE_MINI_HALOS is True.
    mturnovers_mini : array-like, optional
        The turnover mass at each redshift for minihalos (MCGs).
        Only required when USE_MINI_HALOS is True.
    component : str, {'both', 'acg', 'mcg}
        The component of the LF to be calculated.

    Returns
    -------
    Muvfunc : np.ndarray
        Magnitude array (i.e. brightness). Shape [nredshifts, nbins]
    Mhfunc : np.ndarray
        Halo mass array. Shape [nredshifts, nbins]
    lfunc : np.ndarray
        Number density of haloes corresponding to each bin defined by `Muvfunc`.
        Shape [nredshifts, nbins].
    """
    user_params = UserParams.new(user_params)
    cosmo_params = CosmoParams.new(cosmo_params)
    flag_options = FlagOptions.new(flag_options)
    astro_params = AstroParams.new(astro_params, flag_options=flag_options)

    redshifts = np.array(redshifts, dtype="float32")
    if flag_options.USE_MINI_HALOS:
        if component in ["both", "acg"]:
            if mturnovers is None:
                raise ValueError(
                    "calculating ACG LFs with mini-halo feature requires users to "
                    "specify mturnovers!"
                )

            mturnovers = np.array(mturnovers, dtype=np.float32)
            if len(mturnovers) != len(redshifts):
                raise ValueError(
                    f"mturnovers ({len(mturnovers)}) does not match the length of "
                    f"redshifts ({len(redshifts)})"
                )
        if component in ["both", "mcg"]:
            if mturnovers_mini is None:
                raise ValueError(
                    "calculating MCG LFs with mini-halo feature requires users to "
                    "specify mturnovers_MINI!"
                )

            mturnovers_mini = np.array(mturnovers_mini, dtype="float32")
            if len(mturnovers_mini) != len(redshifts):
                raise ValueError(
                    f"mturnovers_MINI ({len(mturnovers)}) does not match the length of "
                    f"redshifts ({len(redshifts)})"
                )

    else:
        mturnovers = (
            np.zeros(len(redshifts), dtype=np.float32) + 10**astro_params.M_TURN
        )
        component = "acg"

    lfunc = np.zeros(len(redshifts) * nbins)
    Muvfunc = np.zeros(len(redshifts) * nbins)
    Mhfunc = np.zeros(len(redshifts) * nbins)

    lfunc.shape = (len(redshifts), nbins)
    Muvfunc.shape = (len(redshifts), nbins)
    Mhfunc.shape = (len(redshifts), nbins)

    c_Muvfunc = ffi.cast("double *", ffi.from_buffer(Muvfunc))
    c_Mhfunc = ffi.cast("double *", ffi.from_buffer(Mhfunc))
    c_lfunc = ffi.cast("double *", ffi.from_buffer(lfunc))

    lfunc_MINI = np.zeros(len(redshifts) * nbins)
    Muvfunc_MINI = np.zeros(len(redshifts) * nbins)
    Mhfunc_MINI = np.zeros(len(redshifts) * nbins)

    lfunc_MINI.shape = (len(redshifts), nbins)
    Muvfunc_MINI.shape = (len(redshifts), nbins)
    Mhfunc_MINI.shape = (len(redshifts), nbins)

    c_Muvfunc_MINI = ffi.cast("double *", ffi.from_buffer(Muvfunc_MINI))
    c_Mhfunc_MINI = ffi.cast("double *", ffi.from_buffer(Mhfunc_MINI))
    c_lfunc_MINI = ffi.cast("double *", ffi.from_buffer(lfunc_MINI))

    if component in ("both", "acg"):
        # Run the C code
        errcode = lib.ComputeLF(
            nbins,
            user_params.cstruct,
            cosmo_params.cstruct,
            astro_params.cstruct,
            flag_options.cstruct,
            1,
            len(redshifts),
            ffi.cast("float *", ffi.from_buffer(redshifts)),
            ffi.cast("float *", ffi.from_buffer(mturnovers)),
            c_Muvfunc,
            c_Mhfunc,
            c_lfunc,
        )

        _process_exitcode(
            errcode,
            lib.ComputeLF,
            (
                nbins,
                user_params.cstruct,
                cosmo_params.cstruct,
                astro_params.cstruct,
                flag_options.cstruct,
                1,
                len(redshifts),
            ),
        )

    if component in ("both", "mcg"):
        # Run the C code
        errcode = lib.ComputeLF(
            nbins,
            user_params.cstruct,
            cosmo_params.cstruct,
            astro_params.cstruct,
            flag_options.cstruct,
            2,
            len(redshifts),
            ffi.cast("float *", ffi.from_buffer(redshifts)),
            ffi.cast("float *", ffi.from_buffer(mturnovers_mini)),
            c_Muvfunc_MINI,
            c_Mhfunc_MINI,
            c_lfunc_MINI,
        )

        _process_exitcode(
            errcode,
            lib.ComputeLF,
            (
                nbins,
                user_params.cstruct,
                cosmo_params.cstruct,
                astro_params.cstruct,
                flag_options.cstruct,
                2,
                len(redshifts),
            ),
        )

    if component == "both":
        # redo the Muv range using the faintest (most likely MINI) and the brightest (most likely massive)
        lfunc_all = np.zeros(len(redshifts) * nbins)
        Muvfunc_all = np.zeros(len(redshifts) * nbins)
        Mhfunc_all = np.zeros(len(redshifts) * nbins * 2)

        lfunc_all.shape = (len(redshifts), nbins)
        Muvfunc_all.shape = (len(redshifts), nbins)
        Mhfunc_all.shape = (len(redshifts), nbins, 2)

        for iz in range(len(redshifts)):
            Muvfunc_all[iz] = np.linspace(
                np.min([Muvfunc.min(), Muvfunc_MINI.min()]),
                np.max([Muvfunc.max(), Muvfunc_MINI.max()]),
                nbins,
            )
            lfunc_all[iz] = np.log10(
                10
                ** (
                    interp1d(Muvfunc[iz], lfunc[iz], fill_value="extrapolate")(
                        Muvfunc_all[iz]
                    )
                )
                + 10
                ** (
                    interp1d(
                        Muvfunc_MINI[iz], lfunc_MINI[iz], fill_value="extrapolate"
                    )(Muvfunc_all[iz])
                )
            )
            Mhfunc_all[iz] = np.array(
                [
                    interp1d(Muvfunc[iz], Mhfunc[iz], fill_value="extrapolate")(
                        Muvfunc_all[iz]
                    ),
                    interp1d(
                        Muvfunc_MINI[iz], Mhfunc_MINI[iz], fill_value="extrapolate"
                    )(Muvfunc_all[iz]),
                ],
            ).T
        lfunc_all[lfunc_all <= -30] = np.nan
        return Muvfunc_all, Mhfunc_all, lfunc_all
    elif component == "acg":

        lfunc[lfunc <= -30] = np.nan
        return Muvfunc, Mhfunc, lfunc
    elif component == "mcg":
        lfunc_MINI[lfunc_MINI <= -30] = np.nan
        return Muvfunc_MINI, Mhfunc_MINI, lfunc_MINI
    else:
        raise ValueError(
            f"Unknown component '{component}'. Must be 'both', 'acg' or 'mcg'"
        )


@cache
def construct_fftw_wisdoms(
    *,
    user_params: UserParams | dict | None = None,
    cosmo_params: CosmoParams | dict | None = None,
) -> int:
    """Construct all necessary FFTW wisdoms.

    Parameters
    ----------
    user_params : :class:`~inputs.UserParams`
        Parameters defining the simulation run.

    """
    user_params = UserParams.new(user_params)
    cosmo_params = CosmoParams.new(cosmo_params)

    # Run the C code
    if user_params.USE_FFTW_WISDOM:
        return lib.CreateFFTWWisdoms(user_params.cstruct, cosmo_params.cstruct)
    else:
        return 0
