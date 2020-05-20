"""
The main wrapper for the underlying 21cmFAST C-code.

The module provides both low- and high-level wrappers, using the very low-level machinery
in :mod:`~py21cmfast._utils`, and the convenient input and output structures from
:mod:`~py21cmfast.inputs` and :mod:`~py21cmfast.outputs`.

This module provides a number of:

* Low-level functions which simplify calling the background C functions which populate
  these output objects given the input classes.
* High-level functions which provide the most efficient and simplest way to generate the
  most commonly desired outputs.

**Low-level functions**

The low-level functions provided here ease the production of the aforementioned output
objects. Functions exist for each low-level C routine, which have been decoupled as far
as possible. So, functions exist to create :func:`initial_conditions`,
:func:`perturb_field`, :class:`ionize_box` and so on. Creating a brightness temperature
box (often the desired final output) would generally require calling each of these in
turn, as each depends on the result of a previous function. Nevertheless, each function
has the capability of generating the required previous outputs on-the-fly, so one can
instantly call :func:`ionize_box` and get a self-consistent result. Doing so, while
convenient, is sometimes not *efficient*, especially when using inhomogeneous
recombinations or the spin temperature field, which intrinsically require consistent
evolution of the ionization field through redshift. In these cases, for best efficiency
it is recommended to either use a customised manual approach to calling these low-level
functions, or to call a higher-level function which optimizes this process.

Finally, note that :mod:`py21cmfast` attempts to optimize the production of the large amount of
data via on-disk caching. By default, if a previous set of data has been computed using
the current input parameters, it will be read-in from a caching repository and returned
directly. This behaviour can be tuned in any of the low-level (or high-level) functions
by setting the `write`, `direc`, `regenerate` and `match_seed` parameters (see docs for
:func:`initial_conditions` for details). The function :func:`~query_cache` can be used
to search the cache, and return empty datasets corresponding to each (and these can then be
filled with the data merely by calling ``.read()`` on any data set). Conversely, a
specific data set can be read and returned as a proper output object by calling the
:func:`~py21cmfast.cache_tools.readbox` function.


**High-level functions**

As previously mentioned, calling the low-level functions in some cases is non-optimal,
especially when full evolution of the field is required, and thus iteration through a
series of redshift. In addition, while :class:`InitialConditions` and
:class:`PerturbedField` are necessary intermediate data, it is *usually* the resulting
brightness temperature which is of most interest, and it is easier to not have to worry
about the intermediate steps explicitly. For these typical use-cases, two high-level
functions are available: :func:`run_coeval` and :func:`run_lightcone`, whose purpose
should be self-explanatory. These will optimally run all necessary intermediate
steps (using cached results by default if possible) and return all datasets of interest.


Examples
--------
A typical example of using this module would be the following.

>>> import py21cmfast as p21

Get coeval cubes at redshift 7,8 and 9, without spin temperature or inhomogeneous
recombinations:

>>> coeval = p21.run_coeval(
>>>     redshift=[7,8,9],
>>>     cosmo_params=p21.CosmoParams(hlittle=0.7),
>>>     user_params=p21.UserParams(HII_DIM=100)
>>> )

Get coeval cubes at the same redshift, with both spin temperature and inhomogeneous
recombinations, pulled from the natural evolution of the fields:

>>> all_boxes = p21.run_coeval(
>>>                 redshift=[7,8,9],
>>>                 user_params=p21.UserParams(HII_DIM=100),
>>>                 flag_options=p21.FlagOptions(INHOMO_RECO=True),
>>>                 do_spin_temp=True
>>>             )

Get a self-consistent lightcone defined between z1 and z2 (`z_step_factor` changes the
logarithmic steps between redshift that are actually evaluated, which are then
interpolated onto the lightcone cells):

>>> lightcone = p21.run_lightcone(redshift=z2, max_redshift=z2, z_step_factor=1.03)
"""
import logging
import os
import warnings
from copy import deepcopy

import numpy as np
from astropy import units
from astropy.cosmology import z_at_value
from scipy.interpolate import interp1d

from ._cfg import config
from ._utils import StructWrapper
from ._utils import _check_compatible_inputs
from .c_21cmfast import ffi
from .c_21cmfast import lib
from .inputs import AstroParams
from .inputs import CosmoParams
from .inputs import FlagOptions
from .inputs import UserParams
from .inputs import global_params
from .outputs import BrightnessTemp
from .outputs import Coeval
from .outputs import InitialConditions
from .outputs import IonizedBox
from .outputs import LightCone
from .outputs import PerturbedField
from .outputs import TsBox
from .outputs import _OutputStructZ

logger = logging.getLogger("21cmFAST")


def _configure_inputs(
    defaults: list,
    *datasets,
    ignore: list = ["redshift"],
    flag_none: [list, None] = None,
):
    """Configure a set of input parameter structs.

    This is useful for basing parameters on a previous output.
    The logic is this: the struct _cannot_ be present and different in both defaults and
    a dataset. If it is present in _either_ of them, that will be returned. If it is
    present in _neither_, either an error will be raised (if that item is in `flag_none`)
    or it will pass.

    Parameters
    ----------
    defaults : list of 2-tuples
        Each tuple is (key, val). Keys are input struct names, and values are a default
        structure for that input.
    datasets : list of :class:`~_utils.OutputStruct`
        A number of output datasets to cross-check, and draw parameter values from.
    ignore : list of str
        Attributes to ignore when ensuring that parameter inputs are the same.
    flag_none : list
        A list of parameter names for which ``None`` is not an acceptable value.

    Raises
    ------
    ValueError :
        If an input parameter is present in both defaults and the dataset, and is different.
        OR if the parameter is present in neither defaults not the datasets, and it is
        included in `flag_none`.
    """
    # First ensure all inputs are compaible in their parameters
    _check_compatible_inputs(*datasets, ignore=ignore)

    if flag_none is None:
        flag_none = []

    output = [0] * len(defaults)
    for i, (key, val) in enumerate(defaults):
        # Get the value of this input from the datasets
        data_val = None
        for dataset in datasets:
            if dataset is not None and hasattr(dataset, key):
                data_val = getattr(dataset, key)
                break

        # If both data and default have values
        if val is not None and data_val is not None and data_val != val:
            raise ValueError(
                "%s has an inconsistent value with %s"
                % (key, dataset.__class__.__name__)
            )
        else:
            if val is not None:
                output[i] = val
            elif data_val is not None:
                output[i] = data_val
            elif key in flag_none:
                raise ValueError(
                    "For %s, a value must be provided in some manner" % key
                )
            else:
                output[i] = None

    return output


def configure_redshift(redshift, *structs):
    """
    Check and obtain a redshift from given default and structs.

    Parameters
    ----------
    redshift : float
        The default redshift to use
    structs : list of :class:`~_utils.OutputStruct`
        A number of output datasets from which to find the redshift.

    Raises
    ------
    ValueError :
        If both `redshift` and *all* structs have a value of `None`, **or** if any of them
        are different from each other (and not `None`).
    """
    zs = set()
    for s in structs:
        if s is not None and hasattr(s, "redshift"):
            zs.add(s.redshift)

    zs = list(zs)

    if len(zs) > 1 or (len(zs) == 1 and redshift is not None and zs[0] != redshift):
        raise ValueError("Incompatible redshifts in inputs")
    elif len(zs) == 1:
        return zs[0]
    elif redshift is None:
        raise ValueError(
            "Either redshift must be provided, or a data set containing it."
        )
    else:
        return redshift


def _verify_types(**kwargs):
    """Ensure each argument has a type of None or that matching its name."""
    for k, v in kwargs.items():
        for j, kk in enumerate(["init", "perturb", "ionize", "spin_temp"]):
            if kk in k:
                break
        cls = [InitialConditions, PerturbedField, IonizedBox, TsBox][j]

        if v is not None and not isinstance(v, cls):
            raise ValueError("%s must be an instance of %s" % (k, cls.__name__))


class ParameterError(RuntimeError):
    """An exception representing a bad choice of parameters."""

    def __init__(self):
        default_message = "21cmFAST does not support this combination of parameters."
        super().__init__(default_message)


class FatalCError(Exception):
    """An exception representing something going wrong in C."""

    def __init__(self, msg=None):
        default_message = "21cmFAST is exiting."
        super().__init__(msg or default_message)


SUCCESS = 0
IOERROR = 1
GSLERROR = 2
VALUEERROR = 3
PARAMETERERROR = 4
MEMORYALLOCERROR = 5


def _process_exitcode(exitcode, fnc, args):
    """Determine what happens for different values of the (integer) exit code from a C function."""
    if exitcode != SUCCESS:
        logger.error("In function: {}.  Arguments: {}".format(fnc.__name__, args))

        if exitcode in (GSLERROR, PARAMETERERROR):
            raise ParameterError
        elif exitcode in (IOERROR, VALUEERROR, MEMORYALLOCERROR):
            raise FatalCError
        else:  # Unknown C code
            raise FatalCError("Unknown error in C. Please report this error!")


def _call_c_simple(fnc, *args):
    """Call a simple C function that just returns an object.

    Any such function should be defined such that the last argument is an int pointer generating
    the status.
    """
    # Parse the function to get the type of the last argument
    cdata = str(ffi.addressof(lib, fnc.__name__))
    kind = cdata.split("(")[-1].split(")")[0].split(",")[-1]
    result = ffi.new(kind)
    status = fnc(*args, result)
    _process_exitcode(status, fnc, args)
    return result[0]


def _call_c_func(fnc, obj, direc, *args, write=True):
    logger.debug("Calling {} with args: {}".format(fnc.__name__, args))
    try:
        exitcode = fnc(
            *[arg() if isinstance(arg, StructWrapper) else arg for arg in args], obj()
        )
    except TypeError as e:
        logger.error(
            f"Arguments to {fnc.__name__}: "
            f"{[arg() if isinstance(arg, StructWrapper) else arg for arg in args]}"
        )
        raise e

    _process_exitcode(exitcode, fnc, args)
    obj.filled = True
    obj._expose()

    # Optionally do stuff with the result (like writing it)
    if write:
        obj.write(direc)

    return obj


def _get_config_options(direc, regenerate, write):
    return (
        os.path.expanduser(config["direc"] if direc is None else direc),
        config["regenerate"] if regenerate is None else regenerate,
        config["write"] if write is None else write,
    )


def get_all_fieldnames(arrays_only=True, lightcone_only=False, as_dict=False):
    """Return all possible fieldnames in output structs.

    Parameters
    ----------
    arrays_only : bool, optional
        Whether to only return fields that are arrays.
    lightcone_only : bool, optional
        Whether to only return fields from classes that evolve with redshift.
    as_dict : bool, optional
        Whether to return results as a dictionary of ``quantity: class_name``.
        Otherwise returns a set of quantities.
    """

    def get_all_subclasses(cls):
        all_subclasses = []

        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(get_all_subclasses(subclass))

        return all_subclasses

    classes = [cls(redshift=0) for cls in _OutputStructZ._implementations()]

    if not lightcone_only:
        classes.append(InitialConditions())

    attr = "pointer_fields" if arrays_only else "fieldnames"

    if as_dict:
        return {
            name: cls.__class__.__name__
            for cls in classes
            for name in getattr(cls, attr)
        }
    else:
        return {name for cls in classes for name in getattr(cls, attr)}


# ======================================================================================
# WRAPPING FUNCTIONS
# ======================================================================================
def compute_tau(*, redshifts, global_xHI, user_params=None, cosmo_params=None):
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
        If `redshifts` and `global_xHI` have inconsistent length.
    """
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)

    if len(redshifts) != len(global_xHI):
        raise ValueError("redshifts and global_xHI must have same length")

    # Convert the data to the right type
    redshifts = np.array(redshifts, dtype="float32")
    global_xHI = np.array(global_xHI, dtype="float32")

    z = ffi.cast("float *", ffi.from_buffer(redshifts))
    xHI = ffi.cast("float *", ffi.from_buffer(global_xHI))

    # Run the C code
    return lib.ComputeTau(user_params(), cosmo_params(), len(redshifts), z, xHI)


def compute_luminosity_function(
    *,
    redshifts,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
    nbins=100,
    mturnovers=None,
    mturnovers_mini=None,
    component=0,
):
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
    component : int, optional
        The component of the LF to be calculated. 0, 1 an 2 are for the total,
        ACG and MCG LFs respectively, requiring inputs of both mturnovers and
        mturnovers_MINI (0), only mturnovers (1) or mturnovers_MINI (2).

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
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    astro_params = AstroParams(astro_params)
    flag_options = FlagOptions(flag_options)

    redshifts = np.array(redshifts, dtype="float32")
    if flag_options.USE_MINI_HALOS:
        if component in [0, 1]:
            if mturnovers is None:
                logger.warning(
                    "calculating ACG LFs with mini-halo feature requires users to specify mturnovers!"
                )
                return None, None, None

            mturnovers = np.array(mturnovers, dtype="float32")
            if len(mturnovers) != len(redshifts):
                logger.warning(
                    "mturnovers(%d) does not match the length of redshifts (%d)"
                    % (len(mturnovers), len(redshifts))
                )
                return None, None, None
        if component in [0, 2]:
            if mturnovers_mini is None:
                logger.warning(
                    "calculating MCG LFs with mini-halo feature requires users to "
                    "specify mturnovers_MINI!"
                )
                return None, None, None

            mturnovers_mini = np.array(mturnovers, dtype="float32")
            if len(mturnovers_mini) != len(redshifts):
                logger.warning(
                    "mturnovers_MINI(%d) does not match the length of redshifts (%d)"
                    % (len(mturnovers), len(redshifts))
                )
                return None, None, None

    else:
        mturnovers = (
            np.zeros(len(redshifts), dtype="float32") + 10 ** astro_params.M_TURN
        )
        component = 1

    if component == 0:
        lfunc = np.zeros(len(redshifts) * nbins)
        Muvfunc = np.zeros(len(redshifts) * nbins)
        Mhfunc = np.zeros(len(redshifts) * nbins)

        lfunc.shape = (len(redshifts), nbins)
        Muvfunc.shape = (len(redshifts), nbins)
        Mhfunc.shape = (len(redshifts), nbins)

        c_Muvfunc = ffi.cast("double *", ffi.from_buffer(Muvfunc))
        c_Mhfunc = ffi.cast("double *", ffi.from_buffer(Mhfunc))
        c_lfunc = ffi.cast("double *", ffi.from_buffer(lfunc))

        # Run the C code
        errcode = lib.ComputeLF(
            nbins,
            user_params(),
            cosmo_params(),
            astro_params(),
            flag_options(),
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
                user_params,
                cosmo_params,
                astro_params,
                flag_options,
                1,
                len(redshifts),
            ),
        )

        lfunc_MINI = np.zeros(len(redshifts) * nbins)
        Muvfunc_MINI = np.zeros(len(redshifts) * nbins)
        Mhfunc_MINI = np.zeros(len(redshifts) * nbins)

        lfunc_MINI.shape = (len(redshifts), nbins)
        Muvfunc_MINI.shape = (len(redshifts), nbins)
        Mhfunc_MINI.shape = (len(redshifts), nbins)

        c_Muvfunc_MINI = ffi.cast("double *", ffi.from_buffer(Muvfunc_MINI))
        c_Mhfunc_MINI = ffi.cast("double *", ffi.from_buffer(Mhfunc_MINI))
        c_lfunc_MINI = ffi.cast("double *", ffi.from_buffer(lfunc_MINI))

        # Run the C code
        errcode = lib.ComputeLF(
            nbins,
            user_params(),
            cosmo_params(),
            astro_params(),
            flag_options(),
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
                user_params,
                cosmo_params,
                astro_params,
                flag_options,
                2,
                len(redshifts),
            ),
        )

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
                ** (interp1d(Muvfunc, lfunc, fill_value="extrapolate")(Muvfunc_all[iz]))
                + 10
                ** (
                    interp1d(Muvfunc_MINI, lfunc_MINI, fill_value="extrapolate")(
                        Muvfunc_all[iz]
                    )
                )
            )
            Mhfunc_all[iz] = np.array(
                interp1d(Muvfunc, Mhfunc, fill_value="extrapolate")(Muvfunc_all[iz]),
                interp1d(Muvfunc_MINI, Mhfunc_MINI, fill_value="extrapolate")(
                    Muvfunc_all[iz]
                ),
            ).T
        lfunc_all[lfunc_all <= -30] = np.nan
        return Muvfunc_all, Mhfunc_all, lfunc_all
    elif component == 1:
        lfunc = np.zeros(len(redshifts) * nbins)
        Muvfunc = np.zeros(len(redshifts) * nbins)
        Mhfunc = np.zeros(len(redshifts) * nbins)

        lfunc.shape = (len(redshifts), nbins)
        Muvfunc.shape = (len(redshifts), nbins)
        Mhfunc.shape = (len(redshifts), nbins)

        c_Muvfunc = ffi.cast("double *", ffi.from_buffer(Muvfunc))
        c_Mhfunc = ffi.cast("double *", ffi.from_buffer(Mhfunc))
        c_lfunc = ffi.cast("double *", ffi.from_buffer(lfunc))

        # Run the C code
        errcode = lib.ComputeLF(
            nbins,
            user_params(),
            cosmo_params(),
            astro_params(),
            flag_options(),
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
                user_params,
                cosmo_params,
                astro_params,
                flag_options,
                1,
                len(redshifts),
            ),
        )

        lfunc[lfunc <= -30] = np.nan
        return Muvfunc, Mhfunc, lfunc
    elif component == 2:
        lfunc_MINI = np.zeros(len(redshifts) * nbins)
        Muvfunc_MINI = np.zeros(len(redshifts) * nbins)
        Mhfunc_MINI = np.zeros(len(redshifts) * nbins)

        lfunc_MINI.shape = (len(redshifts), nbins)
        Muvfunc_MINI.shape = (len(redshifts), nbins)
        Mhfunc_MINI.shape = (len(redshifts), nbins)

        c_Muvfunc_MINI = ffi.cast("double *", ffi.from_buffer(Muvfunc_MINI))
        c_Mhfunc_MINI = ffi.cast("double *", ffi.from_buffer(Mhfunc_MINI))
        c_lfunc_MINI = ffi.cast("double *", ffi.from_buffer(lfunc_MINI))

        # Run the C code
        errcode = lib.ComputeLF(
            nbins,
            user_params(),
            cosmo_params(),
            astro_params(),
            flag_options(),
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
                user_params,
                cosmo_params,
                astro_params,
                flag_options,
                2,
                len(redshifts),
            ),
        )

        lfunc_MINI[lfunc_MINI <= -30] = np.nan
        return Muvfunc_MINI, Mhfunc_MINI, lfunc_MINI
    else:
        logger.warning("What is component %d ?" % component)
        return None, None, None


def _init_photon_conservation_correction(
    *, user_params=None, cosmo_params=None, astro_params=None, flag_options=None
):
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    astro_params = AstroParams(astro_params)
    flag_options = FlagOptions(flag_options)

    return lib.InitialisePhotonCons(
        user_params(), cosmo_params(), astro_params(), flag_options()
    )


def _calibrate_photon_conservation_correction(
    *, redshifts_estimate, nf_estimate, NSpline
):
    # Convert the data to the right type
    redshifts_estimate = np.array(redshifts_estimate, dtype="float64")
    nf_estimate = np.array(nf_estimate, dtype="float64")

    z = ffi.cast("double *", ffi.from_buffer(redshifts_estimate))
    xHI = ffi.cast("double *", ffi.from_buffer(nf_estimate))

    return lib.PhotonCons_Calibration(z, xHI, NSpline)


def _calc_zstart_photon_cons():
    # Run the C code
    return _call_c_simple(lib.ComputeZstart_PhotonCons)


def _get_photon_nonconservation_data():
    """
    Access C global data representing the photon-nonconservation corrections.

    .. note::  if not using ``PHOTON_CONS`` (in :class:`~FlagOptions`), *or* if the
               initialisation for photon conservation has not been performed yet, this
               will return None.

    Returns
    -------
    dict :
      z_analytic: array of redshifts defining the analytic ionized fraction
      Q_analytic: array of analytic  ionized fractions corresponding to `z_analytic`
      z_calibration: array of redshifts defining the ionized fraction from 21cmFAST without
      recombinations
      nf_calibration: array of calibration ionized fractions corresponding to `z_calibration`
      delta_z_photon_cons: the change in redshift required to calibrate 21cmFAST, as a function
      of z_calibration
      nf_photoncons: the neutral fraction as a function of redshift
    """
    # Check if photon conservation has been initialised at all
    if not lib.photon_cons_allocated:
        return None

    arbitrary_large_size = 2000

    data = np.zeros((6, arbitrary_large_size))

    IntVal1 = np.array(np.zeros(1), dtype="int32")
    IntVal2 = np.array(np.zeros(1), dtype="int32")
    IntVal3 = np.array(np.zeros(1), dtype="int32")

    c_z_at_Q = ffi.cast("double *", ffi.from_buffer(data[0]))
    c_Qval = ffi.cast("double *", ffi.from_buffer(data[1]))
    c_z_cal = ffi.cast("double *", ffi.from_buffer(data[2]))
    c_nf_cal = ffi.cast("double *", ffi.from_buffer(data[3]))
    c_PC_nf = ffi.cast("double *", ffi.from_buffer(data[4]))
    c_PC_deltaz = ffi.cast("double *", ffi.from_buffer(data[5]))

    c_int_NQ = ffi.cast("int *", ffi.from_buffer(IntVal1))
    c_int_NC = ffi.cast("int *", ffi.from_buffer(IntVal2))
    c_int_NP = ffi.cast("int *", ffi.from_buffer(IntVal3))

    # Run the C code
    errcode = lib.ObtainPhotonConsData(
        c_z_at_Q,
        c_Qval,
        c_int_NQ,
        c_z_cal,
        c_nf_cal,
        c_int_NC,
        c_PC_nf,
        c_PC_deltaz,
        c_int_NP,
    )

    _process_exitcode(errcode, lib.ObtainPhotonConsData, ())

    ArrayIndices = [
        IntVal1[0],
        IntVal1[0],
        IntVal2[0],
        IntVal2[0],
        IntVal3[0],
        IntVal3[0],
    ]

    data_list = [
        "z_analytic",
        "Q_analytic",
        "z_calibration",
        "nf_calibration",
        "delta_z_photon_cons",
        "nf_photoncons",
    ]

    photon_nonconservation_data = {
        name: d[:index] for name, d, index in zip(data_list, data, ArrayIndices)
    }

    return photon_nonconservation_data


def initial_conditions(
    *,
    user_params=None,
    cosmo_params=None,
    random_seed=None,
    regenerate=None,
    write=None,
    direc=None,
    **global_kwargs,
):
    r"""
    Compute initial conditions.

    Parameters
    ----------
    user_params : :class:`~UserParams` instance, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams` instance, optional
        Defines the cosmological parameters used to compute initial conditions.
    regenerate : bool, optional
        Whether to force regeneration of data, even if matching cached data is found.
        This is applied recursively to any potential sub-calculations. It is ignored in
        the case of dependent data only if that data is explicitly passed to the function.
    write : bool, optional
        Whether to write results to file (i.e. cache). This is recursively applied to
        any potential sub-calculations.
    direc : str, optional
        The directory in which to search for the boxes and write them. By default, this
        is the directory given by ``boxdir`` in the configuration file,
        ``~/.21cmfast/config.yml``. This is recursively applied to any potential
        sub-calculations.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~InitialConditions`
    """
    direc, regenerate, write = _get_config_options(direc, regenerate, write)

    with global_params.use(**global_kwargs):
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)

        # Initialize memory for the boxes that will be returned.
        boxes = InitialConditions(
            user_params=user_params, cosmo_params=cosmo_params, random_seed=random_seed
        )

        # First check whether the boxes already exist.
        if not regenerate:
            try:
                boxes.read(direc)
                logger.info(
                    "Existing init_boxes found and read in (seed=%s)."
                    % boxes.random_seed
                )
                return boxes
            except IOError:
                pass

        return _call_c_func(
            lib.ComputeInitialConditions,
            boxes,
            direc,
            boxes.random_seed,
            boxes.user_params,
            boxes.cosmo_params,
            write=write,
        )


def perturb_field(
    *,
    redshift,
    init_boxes=None,
    user_params=None,
    cosmo_params=None,
    random_seed=None,
    regenerate=None,
    write=None,
    direc=None,
    **global_kwargs,
):
    r"""
    Compute a perturbed field at a given redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to compute the perturbed field.
    init_boxes : :class:`~InitialConditions`, optional
        If given, these initial conditions boxes will be used, otherwise initial conditions will
        be generated. If given,
        the user and cosmo params will be set from this object.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~PerturbedField`

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed:
        See docs of :func:`initial_conditions` for more information.

    Examples
    --------
    The simplest method is just to give a redshift::

    >>> field = perturb_field(7.0)
    >>> print(field.density)

    Doing so will internally call the :func:`~initial_conditions` function. If initial conditions
    have already been
    calculated, this can be avoided by passing them:

    >>> init_boxes = initial_conditions()
    >>> field7 = perturb_field(7.0, init_boxes)
    >>> field8 = perturb_field(8.0, init_boxes)

    The user and cosmo parameter structures are by default inferred from the ``init_boxes``,
    so that the following is
    consistent::

    >>> init_boxes = initial_conditions(user_params= UserParams(HII_DIM=1000))
    >>> field7 = perturb_field(7.0, init_boxes)

    If ``init_boxes`` is not passed, then these parameters can be directly passed::

    >>> field7 = perturb_field(7.0, user_params=UserParams(HII_DIM=1000))

    """
    direc, regenerate, write = _get_config_options(direc, regenerate, write)
    with global_params.use(**global_kwargs):
        _verify_types(init_boxes=init_boxes)

        # Configure and check input/output parameters/structs
        random_seed, user_params, cosmo_params = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
            ],
            init_boxes,
        )

        # Verify input parameter structs (need to do this after configure_inputs).
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)

        # Initialize perturbed boxes.
        fields = PerturbedField(
            redshift=redshift,
            user_params=user_params,
            cosmo_params=cosmo_params,
            random_seed=random_seed,
        )

        # Check whether the boxes already exist
        if not regenerate:
            try:
                fields.read(direc)
                logger.info(
                    "Existing z=%s perturb_field boxes found and read in (seed=%s)."
                    % (redshift, fields.random_seed)
                )
                return fields
            except IOError:
                pass

        # Make sure we've got computed init boxes.
        if init_boxes is None or not init_boxes.filled:
            init_boxes = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                regenerate=regenerate,
                write=write,
                direc=direc,
                random_seed=random_seed,
            )

            # Need to update fields to have the same seed as init_boxes
            fields._random_seed = init_boxes.random_seed

        # Run the C Code
        return _call_c_func(
            lib.ComputePerturbField,
            fields,
            direc,
            redshift,
            fields.user_params,
            fields.cosmo_params,
            init_boxes,
            write=write,
        )


def ionize_box(
    *,
    astro_params=None,
    flag_options=None,
    redshift=None,
    perturbed_field=None,
    previous_perturbed_field=None,
    previous_ionize_box=None,
    spin_temp=None,
    init_boxes=None,
    cosmo_params=None,
    user_params=None,
    regenerate=None,
    write=None,
    direc=None,
    random_seed=None,
    cleanup=True,
    **global_kwargs,
):
    r"""
    Compute an ionized box at a given redshift.

    This function has various options for how the evolution of the ionization is computed (if at
    all). See the Notes below for details.

    Parameters
    ----------
    astro_params: :class:`~AstroParams` instance, optional
        The astrophysical parameters defining the course of reionization.
    flag_options: :class:`~FlagOptions` instance, optional
        Some options passed to the reionization routine.
    redshift : float, optional
        The redshift at which to compute the ionized box. If `perturbed_field` is given,
        its inherent redshift
        will take precedence over this argument. If not, this argument is mandatory.
    perturbed_field : :class:`~PerturbField`, optional
        If given, this field will be used, otherwise it will be generated. To be generated,
        either `init_boxes` and
        `redshift` must be given, or `user_params`, `cosmo_params` and `redshift`.
    previous_perturbed_field : :class:`~PerturbField`, optional
        An perturbed field at higher redshift. This is only used if mini_halo is included.
    init_boxes : :class:`~InitialConditions` , optional
        If given, and `perturbed_field` *not* given, these initial conditions boxes will be used
        to generate the perturbed field, otherwise initial conditions will be generated on the fly.
        If given, the user and cosmo params will be set from this object.
    previous_ionize_box: :class:`IonizedBox` or None
        An ionized box at higher redshift. This is only used if `INHOMO_RECO` and/or `do_spin_temp`
        are true. If either of these are true, and this is not given, then it will be assumed that
        this is the "first box", i.e. that it can be populated accurately without knowing source
        statistics.
    spin_temp: :class:`TsBox` or None, optional
        A spin-temperature box, only required if `do_spin_temp` is True. If None, will try to read
        in a spin temp box at the current redshift, and failing that will try to automatically
        create one, using the previous ionized box redshift as the previous spin temperature
        redshift.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning. Typically,
        if `spin_temperature` is called directly, you will want this to be true, as if the next box
        to be calculate has different shape, errors will occur if memory is not cleaned. However,
        it can be useful to set it to False if scrolling through parameters for the same box shape.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~IonizedBox` :
        An object containing the ionized box data.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed :
        See docs of :func:`initial_conditions` for more information.

    Notes
    -----
    Typically, the ionization field at any redshift is dependent on the evolution of xHI up until
    that redshift, which necessitates providing a previous ionization field to define the current
    one. This function provides several options for doing so. First, if neither the spin
    temperature field, nor inhomogeneous recombinations (specified in flag options) are used, no
    evolution needs to be done. Otherwise, either (in order of precedence)

    1. a specific previous :class`~IonizedBox` object is provided, which will be used directly,
    2. a previous redshift is provided, for which a cached field on disk will be sought,
    3. a step factor is provided which recursively steps through redshift, calculating previous
       fields up until Z_HEAT_MAX, and returning just the final field at the current redshift, or
    4. the function is instructed to treat the current field as being an initial "high-redshift"
       field such that specific sources need not be found and evolved.

    .. note:: If a previous specific redshift is given, but no cached field is found at that
              redshift, the previous ionization field will be evaluated based on `z_step_factor`.

    Examples
    --------
    By default, no spin temperature is used, and neither are inhomogeneous recombinations,
    so that no evolution is required, thus the following will compute a coeval ionization box:

    >>> xHI = ionize_box(redshift=7.0)

    However, if either of those options are true, then a full evolution will be required:

    >>> xHI = ionize_box(redshift=7.0, flag_options=FlagOptions(INHOMO_RECO=True,USE_TS_FLUCT=True))

    This will by default evolve the field from a redshift of *at least* `Z_HEAT_MAX` (a global
    parameter), in logarithmic steps of `ZPRIME_STEP_FACTOR`. To change these:

    >>> xHI = ionize_box(redshift=7.0, zprime_step_factor=1.2, z_heat_max=15.0,
    >>>                  flag_options={"USE_TS_FLUCT":True})

    Alternatively, one can pass an exact previous redshift, which will be sought in the disk
    cache, or evaluated:

    >>> ts_box = ionize_box(redshift=7.0, previous_ionize_box=8.0, flag_options={
    >>>                     "USE_TS_FLUCT":True})

    Beware that doing this, if the previous box is not found on disk, will continue to evaluate
    prior boxes based on `ZPRIME_STEP_FACTOR`. Alternatively, one can pass a previous
    :class:`~IonizedBox`:

    >>> xHI_0 = ionize_box(redshift=8.0, flag_options={"USE_TS_FLUCT":True})
    >>> xHI = ionize_box(redshift=7.0, previous_ionize_box=xHI_0)

    Again, the first line here will implicitly use ``ZPRIME_STEP_FACTOR`` to evolve the field from
    ``Z_HEAT_MAX``. Note that in the second line, all of the input parameters are taken directly from
    `xHI_0` so that they are consistent, and we need not specify the ``flag_options``.

    As the function recursively evaluates previous redshift, the previous spin temperature fields
    will also be consistently recursively evaluated. Only the final ionized box will actually be
    returned and kept in memory, however intervening results will by default be cached on disk.
    One can also pass an explicit spin temperature object:

    >>> ts = spin_temperature(redshift=7.0)
    >>> xHI = ionize_box(redshift=7.0, spin_temp=ts)

    If automatic recursion is used, then it is done in such a way that no large boxes are kept
    around in memory for longer than they need to be (only two at a time are required).
    """
    direc, regenerate, write = _get_config_options(direc, regenerate, write)

    with global_params.use(**global_kwargs):
        _verify_types(
            init_boxes=init_boxes,
            perturbed_field=perturbed_field,
            previous_perturbed_field=previous_perturbed_field,
            previous_ionize_box=previous_ionize_box,
            spin_temp=spin_temp,
        )

        # Configure and check input/output parameters/structs
        (
            random_seed,
            user_params,
            cosmo_params,
            astro_params,
            flag_options,
        ) = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
                ("astro_params", astro_params),
                ("flag_options", flag_options),
            ],
            init_boxes,
            spin_temp,
            init_boxes,
            perturbed_field,
            previous_perturbed_field,
            previous_ionize_box,
        )
        redshift = configure_redshift(redshift, spin_temp, perturbed_field)

        # Verify input structs
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        flag_options = FlagOptions(flag_options)
        astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

        if spin_temp is not None and not flag_options.USE_TS_FLUCT:
            logger.warning(
                "Changing flag_options.USE_TS_FLUCT to True since spin_temp was passed."
            )
            flag_options.USE_TS_FLUCT = True

        box = IonizedBox(
            first_box=((1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1)
            > global_params.Z_HEAT_MAX
            and (
                not isinstance(previous_ionize_box, IonizedBox)
                or not previous_ionize_box.filled
            ),
            user_params=user_params,
            cosmo_params=cosmo_params,
            redshift=redshift,
            astro_params=astro_params,
            flag_options=flag_options,
            random_seed=random_seed,
        )

        # Check whether the boxes already exist
        if not regenerate:
            try:
                box.read(direc)
                logger.info(
                    "Existing z=%s ionized boxes found and read in (seed=%s)."
                    % (redshift, box.random_seed)
                )
                return box
            except IOError:
                pass

        # EVERYTHING PAST THIS POINT ONLY HAPPENS IF THE BOX DOESN'T ALREADY EXIST
        # ------------------------------------------------------------------------
        # Get the previous redshift
        if flag_options.INHOMO_RECO or flag_options.USE_TS_FLUCT:

            if previous_ionize_box is not None:
                prev_z = previous_ionize_box.redshift
            else:
                prev_z = (1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1

            # Ensure the previous spin temperature has a higher redshift than this one.
            if prev_z and prev_z <= redshift:
                raise ValueError(
                    "Previous ionized box must have a higher redshift than that being evaluated."
                )
        else:
            prev_z = None

        # Get init_box required.
        if init_boxes is None or not init_boxes.filled:
            init_boxes = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                regenerate=regenerate,
                write=write,
                direc=direc,
                random_seed=random_seed,
            )

            # Need to update random seed
            box._random_seed = init_boxes.random_seed

        # Get appropriate previous ionization box
        if previous_ionize_box is None or not previous_ionize_box.filled:
            # If we are beyond Z_HEAT_MAX, just make an empty box
            if prev_z is None or prev_z > global_params.Z_HEAT_MAX:
                previous_ionize_box = IonizedBox(redshift=0)

            # Otherwise recursively create new previous box.
            else:
                previous_ionize_box = ionize_box(
                    astro_params=astro_params,
                    flag_options=flag_options,
                    redshift=prev_z,
                    init_boxes=init_boxes,
                    regenerate=regenerate,
                    write=write,
                    direc=direc,
                    cleanup=False,  # We *know* we're going to need the memory again.
                )

        # Dynamically produce the perturbed field.
        if perturbed_field is None or not perturbed_field.filled:
            perturbed_field = perturb_field(
                init_boxes=init_boxes,
                # NOTE: this is required, rather than using cosmo_ and user_,
                # since init may have a set seed.
                redshift=redshift,
                regenerate=regenerate,
                write=write,
                direc=direc,
            )

        if previous_perturbed_field is None or not previous_perturbed_field.filled:
            # If we are beyond Z_HEAT_MAX, just make an empty box
            if prev_z is None or prev_z > global_params.Z_HEAT_MAX:
                previous_perturbed_field = PerturbedField(redshift=0)
            else:
                previous_perturbed_field = perturb_field(
                    init_boxes=init_boxes,
                    redshift=prev_z,
                    regenerate=regenerate,
                    write=write,
                    direc=direc,
                )
        # Set empty spin temp box if necessary.
        if not flag_options.USE_TS_FLUCT:
            spin_temp = TsBox(redshift=0)
        elif spin_temp is None:
            spin_temp = spin_temperature(
                perturbed_field=perturbed_field,
                flag_options=flag_options,
                init_boxes=init_boxes,
                direc=direc,
                write=write,
                regenerate=regenerate,
                cleanup=cleanup,
            )

        # Run the C Code
        return _call_c_func(
            lib.ComputeIonizedBox,
            box,
            direc,
            redshift,
            previous_ionize_box.redshift,
            box.user_params,
            box.cosmo_params,
            box.astro_params,
            box.flag_options,
            perturbed_field,
            previous_perturbed_field,
            previous_ionize_box,
            spin_temp,
            write=write,
        )


def spin_temperature(
    *,
    astro_params=None,
    flag_options=None,
    redshift=None,
    perturbed_field=None,
    previous_spin_temp=None,
    init_boxes=None,
    cosmo_params=None,
    user_params=None,
    regenerate=None,
    write=None,
    direc=None,
    random_seed=None,
    cleanup=True,
    **global_kwargs,
):
    r"""
    Compute spin temperature boxes at a given redshift.

    See the notes below for how the spin temperature field is evolved through redshift.

    Parameters
    ----------
    astro_params : :class:`~AstroParams`, optional
        The astrophysical parameters defining the course of reionization.
    flag_options : :class:`~FlagOptions`, optional
        Some options passed to the reionization routine.
    redshift : float, optional
        The redshift at which to compute the ionized box. If not given, the redshift from
        `perturbed_field` will be used. Either `redshift`, `perturbed_field`, or
        `previous_spin_temp` must be given. See notes on `perturbed_field` for how it affects the
        given redshift if both are given.
    perturbed_field : :class:`~PerturbField`, optional
        If given, this field will be used, otherwise it will be generated. To be generated,
        either `init_boxes` and `redshift` must be given, or `user_params`, `cosmo_params` and
        `redshift`. By default, this will be generated at the same redshift as the spin temperature
        box. The redshift of perturb field is allowed to be different than `redshift`. If so, it
        will be interpolated to the correct redshift, which can provide a speedup compared to
        actually computing it at the desired redshift.
    previous_spin_temp : :class:`TsBox` or None
        The previous spin temperature box.
    init_boxes : :class:`~InitialConditions`, optional
        If given, and `perturbed_field` *not* given, these initial conditions boxes will be used
        to generate the perturbed field, otherwise initial conditions will be generated on the fly.
        If given, the user and cosmo params will be set from this object.
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculate has different shape, errors will occur
        if memory is not cleaned. However, it can be useful to set it to False if
        scrolling through parameters for the same box shape.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`~TsBox`
        An object containing the spin temperature box data.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed :
        See docs of :func:`initial_conditions` for more information.

    Notes
    -----
    Typically, the spin temperature field at any redshift is dependent on the evolution of spin
    temperature up until that redshift, which necessitates providing a previous spin temperature
    field to define the current one. This function provides several options for doing so. Either
    (in order of precedence):

    1. a specific previous spin temperature object is provided, which will be used directly,
    2. a previous redshift is provided, for which a cached field on disk will be sought,
    3. a step factor is provided which recursively steps through redshift, calculating previous
       fields up until Z_HEAT_MAX, and returning just the final field at the current redshift, or
    4. the function is instructed to treat the current field as being an initial "high-redshift"
       field such that specific sources need not be found and evolved.

    .. note:: If a previous specific redshift is given, but no cached field is found at that
              redshift, the previous spin temperature field will be evaluated based on
              ``z_step_factor``.

    Examples
    --------
    To calculate and return a fully evolved spin temperature field at a given redshift (with
    default input parameters), simply use:

    >>> ts_box = spin_temperature(redshift=7.0)

    This will by default evolve the field from a redshift of *at least* `Z_HEAT_MAX` (a global
    parameter), in logarithmic steps of `z_step_factor`. Thus to change these:

    >>> ts_box = spin_temperature(redshift=7.0, zprime_step_factor=1.2, z_heat_max=15.0)

    Alternatively, one can pass an exact previous redshift, which will be sought in the disk
    cache, or evaluated:

    >>> ts_box = spin_temperature(redshift=7.0, previous_spin_temp=8.0)

    Beware that doing this, if the previous box is not found on disk, will continue to evaluate
    prior boxes based on the ``z_step_factor``. Alternatively, one can pass a previous spin
    temperature box:

    >>> ts_box1 = spin_temperature(redshift=8.0)
    >>> ts_box = spin_temperature(redshift=7.0, previous_spin_temp=ts_box1)

    Again, the first line here will implicitly use ``z_step_factor`` to evolve the field from
    around ``Z_HEAT_MAX``. Note that in the second line, all of the input parameters are taken
    directly from `ts_box1` so that they are consistent. Finally, one can force the function to
    evaluate the current redshift as if it was beyond ``Z_HEAT_MAX`` so that it depends only on
    itself:

    >>> ts_box = spin_temperature(redshift=7.0, zprime_step_factor=None)

    This is usually a bad idea, and will give a warning, but it is possible.
    """
    direc, regenerate, write = _get_config_options(direc, regenerate, write)

    with global_params.use(**global_kwargs):
        _verify_types(
            init_boxes=init_boxes,
            perturbed_field=perturbed_field,
            previous_spin_temp=previous_spin_temp,
        )

        # Configure and check input/output parameters/structs
        (
            random_seed,
            user_params,
            cosmo_params,
            astro_params,
            flag_options,
        ) = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
                ("astro_params", astro_params),
                ("flag_options", flag_options),
            ],
            init_boxes,
            previous_spin_temp,
            init_boxes,
            perturbed_field,
        )

        # Try to determine redshift from other inputs, if required.
        # Note that perturb_field does not need to match redshift here.
        if redshift is None:
            if perturbed_field is not None:
                redshift = perturbed_field.redshift
            elif previous_spin_temp is not None:
                redshift = (
                    previous_spin_temp.redshift + 1
                ) / global_params.ZPRIME_STEP_FACTOR - 1
            else:
                raise ValueError(
                    "Either the redshift, perturbed_field or previous_spin_temp must be given."
                )
        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        flag_options = FlagOptions(flag_options)
        astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

        # Explicitly set this flag to True, though it shouldn't be required!
        flag_options.update(USE_TS_FLUCT=True)

        box = TsBox(
            first_box=((1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1)
            > global_params.Z_HEAT_MAX
            and (
                not isinstance(previous_spin_temp, IonizedBox)
                or not previous_spin_temp.filled
            ),
            user_params=user_params,
            cosmo_params=cosmo_params,
            redshift=redshift,
            astro_params=astro_params,
            flag_options=flag_options,
            random_seed=random_seed,
        )

        # Check whether the boxes already exist on disk.
        if not regenerate:
            try:
                box.read(direc)
                logger.info(
                    "Existing z=%s spin_temp boxes found and read in (seed=%s)."
                    % (redshift, box.random_seed)
                )
                return box
            except IOError:
                pass

        # EVERYTHING PAST THIS POINT ONLY HAPPENS IF THE BOX DOESN'T ALREADY EXIST
        # ------------------------------------------------------------------------

        # Get the previous redshift
        if previous_spin_temp is not None:
            prev_z = previous_spin_temp.redshift
        else:
            prev_z = (1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1

        # Ensure the previous spin temperature has a higher redshift than this one.
        if prev_z and prev_z <= redshift:
            raise ValueError(
                "Previous spin temperature box must have a higher redshift than that being evaluated."
            )

        # Dynamically produce the initial conditions.
        if init_boxes is None or not init_boxes.filled:
            init_boxes = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                regenerate=regenerate,
                write=write,
                direc=direc,
                random_seed=random_seed,
            )

            # Need to update random seed
            box._random_seed = init_boxes.random_seed

        # Create appropriate previous_spin_temp
        if not isinstance(previous_spin_temp, TsBox):
            if prev_z > global_params.Z_HEAT_MAX or prev_z is None:
                previous_spin_temp = TsBox(
                    redshift=global_params.Z_HEAT_MAX,
                    user_params=init_boxes.user_params,
                    cosmo_params=init_boxes.cosmo_params,
                    astro_params=astro_params,
                    flag_options=flag_options,
                )
            else:
                previous_spin_temp = spin_temperature(
                    init_boxes=init_boxes,
                    astro_params=astro_params,
                    flag_options=flag_options,
                    redshift=prev_z,
                    regenerate=regenerate,
                    write=write,
                    direc=direc,
                    cleanup=False,  # we know we'll need the memory again
                )

        # Dynamically produce the perturbed field.
        if perturbed_field is None or not perturbed_field.filled:
            perturbed_field = perturb_field(
                redshift=redshift,
                init_boxes=init_boxes,
                regenerate=regenerate,
                write=write,
                direc=direc,
            )

        # Run the C Code
        return _call_c_func(
            lib.ComputeTsBox,
            box,
            direc,
            redshift,
            previous_spin_temp.redshift,
            box.user_params,
            box.cosmo_params,
            box.astro_params,
            box.flag_options,
            perturbed_field.redshift,
            cleanup,
            perturbed_field,
            previous_spin_temp,
            write=write,
        )


def brightness_temperature(
    *,
    ionized_box,
    perturbed_field,
    spin_temp=None,
    write=None,
    regenerate=None,
    direc=None,
    **global_kwargs,
):
    r"""
    Compute a coeval brightness temperature box.

    Parameters
    ----------
    ionized_box: :class:`IonizedBox`
        A pre-computed ionized box.
    perturbed_field: :class:`PerturbedField`
        A pre-computed perturbed field at the same redshift as `ionized_box`.
    spin_temp: :class:`TsBox`, optional
        A pre-computed spin temperature, at the same redshift as the other boxes.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    :class:`BrightnessTemp` instance.
    """
    direc, regenerate, write = _get_config_options(direc, regenerate, write)

    with global_params.use(**global_kwargs):
        _verify_types(
            perturbed_field=perturbed_field,
            spin_temp=spin_temp,
            ionized_box=ionized_box,
        )

        # don't ignore redshift here
        _check_compatible_inputs(ionized_box, perturbed_field, spin_temp, ignore=[])

        # ensure ionized_box and perturbed_field aren't None, as we don't do
        # any dynamic calculations here.
        if ionized_box is None or perturbed_field is None:
            raise ValueError("both ionized_box and perturbed_field must be specified.")

        if spin_temp is None:
            if ionized_box.flag_options.USE_TS_FLUCT:
                raise ValueError(
                    "You have USE_TS_FLUCT=True, but have not provided a spin_temp!"
                )

            # Make an unused dummy box.
            spin_temp = TsBox(redshift=0, dummy=True)

        box = BrightnessTemp(
            user_params=ionized_box.user_params,
            cosmo_params=ionized_box.cosmo_params,
            astro_params=ionized_box.astro_params,
            flag_options=ionized_box.flag_options,
            redshift=ionized_box.redshift,
            random_seed=ionized_box.random_seed,
        )

        # Check whether the boxes already exist on disk.
        if not regenerate:
            try:
                box.read(direc)
                logger.info(
                    "Existing brightness_temp box found and read in (seed=%s)."
                    % (box.random_seed)
                )
                return box
            except IOError:
                pass

        return _call_c_func(
            lib.ComputeBrightnessTemp,
            box,
            direc,
            ionized_box.redshift,
            ionized_box.user_params,
            ionized_box.cosmo_params,
            ionized_box.astro_params,
            ionized_box.flag_options,
            spin_temp,
            ionized_box,
            perturbed_field,
            write=write,
        )


def _logscroll_redshifts(min_redshift, z_step_factor, zmax):
    redshifts = [min_redshift]  # mult by 1.001 is probably bad...
    while redshifts[-1] < zmax:
        redshifts.append((redshifts[-1] + 1.0) * z_step_factor - 1.0)
    return redshifts[::-1]


def run_coeval(
    *,
    redshift=None,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
    regenerate=None,
    write=None,
    direc=None,
    init_box=None,
    perturb=None,
    use_interp_perturb_field=False,
    random_seed=None,
    cleanup=True,
    **global_kwargs,
):
    r"""
    Evaluate a coeval ionized box at a given redshift, or multiple redshifts.

    This is generally the easiest and most efficient way to generate a set of coeval cubes at a
    given set of redshift. It self-consistently deals with situations in which the field needs to be
    evolved, and does this with the highest memory-efficiency, only returning the desired redshift.
    All other calculations are by default stored in the on-disk cache so they can be re-used at a
    later time.

    .. note:: User-supplied redshift are *not* used as previous redshift in any scrolling,
              so that pristine log-sampling can be maintained.

    Parameters
    ----------
    redshift: array_like
        A single redshift, or multiple redshift, at which to return results. The minimum of these
        will define the log-scrolling behaviour (if necessary).
    user_params : :class:`~inputs.UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~inputs.CosmoParams` , optional
        Defines the cosmological parameters used to compute initial conditions.
    astro_params : :class:`~inputs.AstroParams` , optional
        The astrophysical parameters defining the course of reionization.
    flag_options : :class:`~inputs.FlagOptions` , optional
        Some options passed to the reionization routine.
    init_box : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not
        be re-calculated.
    perturb : list of :class:`~PerturbedField`, optional
        If given, must be compatible with init_box. It will merely negate the necessity
        of re-calculating the perturb fields.
    use_interp_perturb_field : bool, optional
        Whether to use a single perturb field, at the lowest redshift of the lightcone,
        to determine all spin temperature fields. If so, this field is interpolated in
        the underlying C-code to the correct redshift. This is less accurate (and no more
        efficient), but provides compatibility with older versions of 21cmFAST.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculate has different shape, errors will occur
        if memory is not cleaned. Note that internally, this is set to False until the
        last iteration.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    coevals : :class:`~py21cmfast.outputs.Coeval`
        The full data for the Coeval class, with init boxes, perturbed fields, ionized boxes,
        brightness temperature, and potential data from the conservation of photons. If a
        single redshift was specified, it will return such a class. If multiple redshifts
        were passed, it will return a list of such classes.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed :
        See docs of :func:`initial_conditions` for more information.
    """
    with global_params.use(**global_kwargs):
        if redshift is None and perturb is None:
            raise ValueError("Either redshift or perturb must be given")

        direc, regenerate, write = _get_config_options(direc, regenerate, write)

        singleton = False
        # Ensure perturb is a list of boxes, not just one.
        if perturb is not None:
            if not hasattr(perturb, "__len__"):
                perturb = [perturb]
                singleton = True
        else:
            perturb = []

        random_seed, user_params, cosmo_params = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
            ],
            init_box,
            *perturb,
        )

        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        flag_options = FlagOptions(flag_options)
        astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

        if init_box is None:  # no need to get cosmo, user params out of it.
            init_box = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                random_seed=random_seed,
                write=write,
                regenerate=regenerate,
                direc=direc,
            )

        if perturb:
            if redshift is not None:
                if any(p.redshift != z for p, z in zip(perturb, redshift)):
                    raise ValueError(
                        "Input redshifts do not match perturb field redshifts"
                    )

            else:
                redshift = [p.redshift for p in perturb]

        if not hasattr(redshift, "__len__"):
            singleton = True
            redshift = [redshift]

        if not perturb:
            for z in redshift:
                perturb += [
                    perturb_field(
                        redshift=z,
                        init_boxes=init_box,
                        regenerate=regenerate,
                        write=write,
                        direc=direc,
                    )
                ]

        # Get the list of redshift we need to scroll through.
        if flag_options.INHOMO_RECO or flag_options.USE_TS_FLUCT:
            redshifts = _logscroll_redshifts(
                min(redshift),
                global_params.ZPRIME_STEP_FACTOR,
                global_params.Z_HEAT_MAX,
            )
        else:
            redshifts = [min(redshift)]

        # Add in the redshift defined by the user, and sort in order
        # Turn into a set so that exact matching user-set redshift
        # don't double-up with scrolling ones.
        redshifts += redshift
        redshifts = sorted(set(redshifts), reverse=True)

        if (
            flag_options.PHOTON_CONS
            and np.amin(redshifts) < global_params.PhotonConsEndCalibz
        ):
            raise ValueError(
                f"""
                You have passed a redshift (z = {np.amin(redshifts)}) that is lower than the endpoint
                of the photon non-conservation correction
                (global_params.PhotonConsEndCalibz = {global_params.PhotonConsEndCalibz}).
                If this behaviour is desired then set global_params.PhotonConsEndCalibz to a value lower than
                z = {np.amin(redshifts)}.
                """
            )

        if flag_options.PHOTON_CONS:
            calibrate_photon_cons(
                user_params,
                cosmo_params,
                astro_params,
                flag_options,
                init_box,
                regenerate,
                write,
                direc,
            )

        ib_tracker = [0] * len(redshift)
        bt = [0] * len(redshift)
        st, ib = None, None  # At first we don't have any "previous" st or ib.
        logger.debug("redshifts: %s", redshifts)

        minarg = np.argmin(redshift)

        st_tracker = [None] * len(redshift)

        # Iterate through redshift from top to bottom
        for iz, z in enumerate(redshifts):

            if flag_options.USE_TS_FLUCT:
                logger.debug("PID={} doing spin temp for z={}".format(os.getpid(), z))
                st2 = spin_temperature(
                    redshift=z,
                    previous_spin_temp=st,
                    perturbed_field=(
                        perturb[minarg]
                        if use_interp_perturb_field
                        else (perturb[redshift.index(z)] if z in redshift else None)
                    ),
                    # remember that perturb field is interpolated, so no need to provide exact one.
                    astro_params=astro_params,
                    flag_options=flag_options,
                    regenerate=regenerate,
                    init_boxes=init_box,
                    write=write,
                    direc=direc,
                    cleanup=(
                        cleanup and z == redshifts[-1]
                    ),  # cleanup if its the last time through
                )

                if z not in redshift:
                    st = st2

            logger.debug("PID={} doing ionize box for z={}".format(os.getpid(), z))
            ib2 = ionize_box(
                redshift=z,
                previous_ionize_box=ib,
                init_boxes=init_box,
                perturbed_field=perturb[redshift.index(z)] if z in redshift else None,
                # perturb field *not* interpolated here.
                previous_perturbed_field=(
                    None
                    if iz == 0 or not flag_options.USE_MINI_HALOS
                    else (
                        perturb[redshift.index(redshifts[iz - 1])]
                        if redshifts[iz - 1] in redshift
                        else None
                    )
                ),
                astro_params=astro_params,
                flag_options=flag_options,
                spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
                regenerate=regenerate,
                z_heat_max=global_params.Z_HEAT_MAX,
                write=write,
                direc=direc,
                cleanup=(
                    cleanup and z == redshifts[-1]
                ),  # cleanup if its the last time through
            )

            if z not in redshift:
                ib = ib2
            else:
                logger.debug(
                    "PID={} doing brightness temp for z={}".format(os.getpid(), z)
                )
                ib_tracker[redshift.index(z)] = ib2
                st_tracker[redshift.index(z)] = (
                    st2 if flag_options.USE_TS_FLUCT else None
                )

                bt[redshift.index(z)] = brightness_temperature(
                    ionized_box=ib2,
                    perturbed_field=perturb[redshift.index(z)],
                    spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
                    write=write,
                    direc=direc,
                    regenerate=regenerate,
                )

        if flag_options.PHOTON_CONS:
            photon_nonconservation_data = _get_photon_nonconservation_data()
            if photon_nonconservation_data:
                lib.FreePhotonConsMemory()
        else:
            photon_nonconservation_data = None

        coevals = [
            Coeval(z, init_box, p, ib, _bt, st, photon_nonconservation_data)
            for z, p, ib, _bt, st in zip(redshift, perturb, ib_tracker, bt, st_tracker)
        ]

        # If a single redshift was passed, then pass back singletons.
        if singleton:
            coevals = coevals[0]

        logger.debug("Returning from Coeval")

        return coevals


def run_lightcone(
    *,
    redshift=None,
    max_redshift=None,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
    regenerate=None,
    write=None,
    lightcone_quantities=("brightness_temp",),
    global_quantities=("brightness_temp", "xH_box"),
    direc=None,
    init_box=None,
    perturb=None,
    random_seed=None,
    use_interp_perturb_field=False,
    cleanup=True,
    **global_kwargs,
):
    r"""
    Evaluate a full lightcone ending at a given redshift.

    This is generally the easiest and most efficient way to generate a lightcone, though it can
    be done manually by using the lower-level functions which are called by this function.

    Parameters
    ----------
    redshift : float
        The minimum redshift of the lightcone.
    max_redshift : float, optional
        The maximum redshift at which to keep lightcone information. By default, this is equal to
        `z_heat_max`. Note that this is not *exact*, but will be typically slightly exceeded.
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options : :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature
        fluctuations are required.
    lightcone_quantities : tuple of str, optional
        The quantities to form into a lightcone. By default, just the brightness
        temperature. Note that these quantities must exist in one of the output
        structures:

        * :class:`~InitialConditions`
        * :class:`~PerturbField`
        * :class:`~TsBox`
        * :class:`~IonizedBox`
        * :class:`BrightnessTemp`

        To get a full list of possible quantities, run :func:`get_all_fieldnames`.
    global_quantities : tuple of str, optional
        The quantities to save as globally-averaged redshift-dependent functions.
        These may be any of the quantities that can be used in ``lightcone_quantities``.
        The mean is taken over the full 3D cube at each redshift, rather than a 2D
        slice.
    init_box : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be
        re-calculated.
    perturb : list of :class:`~PerturbedField`, optional
        If given, must be compatible with init_box. It will merely negate the necessity of
        re-calculating the
        perturb fields. It will also be used to set the redshift if given.
    use_interp_perturb_field : bool, optional
        Whether to use a single perturb field, at the lowest redshift of the lightcone,
        to determine all spin temperature fields. If so, this field is interpolated in the
        underlying C-code to the correct redshift. This is less accurate (and no more efficient),
        but provides compatibility with older versions of 21cmFAST.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculate has different shape, errors will occur
        if memory is not cleaned. Note that internally, this is set to False until the
        last iteration.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    lightcone : :class:`~py21cmfast.LightCone`
        The lightcone object.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed
        See docs of :func:`initial_conditions` for more information.
    """
    direc, regenerate, write = _get_config_options(direc, regenerate, write)

    with global_params.use(**global_kwargs):
        random_seed, user_params, cosmo_params = _configure_inputs(
            [
                ("random_seed", random_seed),
                ("user_params", user_params),
                ("cosmo_params", cosmo_params),
            ],
            init_box,
            perturb,
        )

        user_params = UserParams(user_params)
        cosmo_params = CosmoParams(cosmo_params)
        flag_options = FlagOptions(flag_options)
        astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

        # Ensure passed quantities are appropriate
        _fld_names = get_all_fieldnames(
            arrays_only=True, lightcone_only=True, as_dict=True
        )
        assert all(
            q in _fld_names.keys() for q in lightcone_quantities
        ), "invalid lightcone_quantity passed."
        assert all(
            q in _fld_names.keys() for q in global_quantities
        ), "invalid global_quantity passed."

        if not flag_options.USE_TS_FLUCT and any(
            _fld_names[q] == "TsBox" for q in lightcone_quantities + global_quantities
        ):
            raise ValueError(
                "TsBox quantity found in lightcone_quantities or global_quantities, but not running spin_temp!"
            )

        redshift = configure_redshift(redshift, perturb)

        max_redshift = (
            global_params.Z_HEAT_MAX
            if (
                flag_options.INHOMO_RECO
                or flag_options.USE_TS_FLUCT
                or max_redshift is None
            )
            else max_redshift
        )

        # Get the redshift through which we scroll and evaluate the ionization field.
        scrollz = _logscroll_redshifts(
            redshift, global_params.ZPRIME_STEP_FACTOR, max_redshift
        )

        if (
            flag_options.PHOTON_CONS
            and np.amin(scrollz) < global_params.PhotonConsEndCalibz
        ):
            raise ValueError(
                f"""
                You have passed a redshift (z = {np.amin(scrollz)}) that is lower than the endpoint
                of the photon non-conservation correction
                (global_params.PhotonConsEndCalibz = {global_params.PhotonConsEndCalibz}).
                If this behaviour is desired then set global_params.PhotonConsEndCalibz to a value lower than
                z = {np.amin(scrollz)}.
                """
            )

        if init_box is None:  # no need to get cosmo, user params out of it.
            init_box = initial_conditions(
                user_params=user_params,
                cosmo_params=cosmo_params,
                write=write,
                regenerate=regenerate,
                direc=direc,
                random_seed=random_seed,
            )

        if perturb is None:
            # The perturb field that we get here is at the *final* redshift,
            # and can be used in TsBox.
            perturb = perturb_field(
                redshift=redshift,
                init_boxes=init_box,
                regenerate=regenerate,
                direc=direc,
                write=write,
            )

        if flag_options.PHOTON_CONS:
            calibrate_photon_cons(
                user_params,
                cosmo_params,
                astro_params,
                flag_options,
                init_box,
                regenerate,
                write,
                direc,
            )

        d_at_redshift, lc_distances, n_lightcone = _setup_lightcone(
            cosmo_params,
            max_redshift,
            redshift,
            scrollz,
            user_params,
            global_params.ZPRIME_STEP_FACTOR,
        )

        scroll_distances = (
            cosmo_params.cosmo.comoving_distance(scrollz).value - d_at_redshift
        )

        # Iterate through redshift from top to bottom
        st, ib, bt, prev_perturb = None, None, None, None
        lc_index = 0
        box_index = 0
        lc = {
            quantity: np.zeros(
                (user_params.HII_DIM, user_params.HII_DIM, n_lightcone),
                dtype=np.float32,
            )
            for quantity in lightcone_quantities
        }

        interp_functions = {
            "z_re_box": "mean_max",
        }

        global_q = {quantity: np.zeros(len(scrollz)) for quantity in global_quantities}
        pf = perturb

        for iz, z in enumerate(scrollz):
            # Best to get a perturb for this redshift, to pass to brightness_temperature
            pf2 = perturb_field(
                redshift=z,
                init_boxes=init_box,
                regenerate=regenerate,
                direc=direc,
                write=write,
            )

            if flag_options.USE_TS_FLUCT:
                st2 = spin_temperature(
                    redshift=z,
                    previous_spin_temp=st,
                    astro_params=astro_params,
                    flag_options=flag_options,
                    perturbed_field=perturb if use_interp_perturb_field else pf2,
                    regenerate=regenerate,
                    init_boxes=init_box,
                    write=write,
                    direc=direc,
                    cleanup=(cleanup and iz == (len(scrollz) - 1)),
                )

            ib2 = ionize_box(
                redshift=z,
                previous_ionize_box=ib,
                init_boxes=init_box,
                perturbed_field=pf2,
                previous_perturbed_field=prev_perturb,
                astro_params=astro_params,
                flag_options=flag_options,
                spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
                regenerate=regenerate,
                write=write,
                direc=direc,
                cleanup=(cleanup and iz == (len(scrollz) - 1)),
            )

            bt2 = brightness_temperature(
                ionized_box=ib2,
                perturbed_field=pf2,
                spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
                write=write,
                direc=direc,
                regenerate=regenerate,
            )

            outs = {
                "PerturbedField": (pf, pf2),
                "IonizedBox": (ib, ib2),
                "BrightnessTemp": (bt, bt2),
            }
            if flag_options.USE_TS_FLUCT:
                outs["TsBox"] = (st, st2)

            # Save mean/global quantities
            for quantity in global_quantities:
                global_q[quantity][iz] = np.mean(
                    getattr(outs[_fld_names[quantity]][1], quantity)
                )

            # Interpolate the lightcone
            if z < max_redshift:
                for quantity in lightcone_quantities:
                    data1, data2 = outs[_fld_names[quantity]]
                    fnc = interp_functions.get(quantity, "mean")

                    n = _interpolate_in_redshift(
                        iz,
                        box_index,
                        lc_index,
                        n_lightcone,
                        scroll_distances,
                        lc_distances,
                        data1,
                        data2,
                        quantity,
                        lc[quantity],
                        fnc,
                    )
                lc_index += n
                box_index += n

            # Save current ones as old ones.
            if flag_options.USE_TS_FLUCT:
                st = st2
            ib = ib2
            bt = bt2
            if flag_options.USE_MINI_HALOS:
                prev_perturb = pf2
            pf = pf2

        if flag_options.PHOTON_CONS:
            photon_nonconservation_data = _get_photon_nonconservation_data()
            if photon_nonconservation_data:
                lib.FreePhotonConsMemory()
        else:
            photon_nonconservation_data = None

        return LightCone(
            redshift,
            user_params,
            cosmo_params,
            astro_params,
            flag_options,
            init_box.random_seed,
            lc,
            node_redshifts=scrollz,
            global_quantities=global_q,
            photon_nonconservation_data=photon_nonconservation_data,
            _globals=dict(global_params.items()),
        )


def _interpolate_in_redshift(
    z_index,
    box_index,
    lc_index,
    n_lightcone,
    scroll_distances,
    lc_distances,
    output_obj,
    output_obj2,
    quantity,
    lc,
    kind="mean",
):
    try:
        array = getattr(output_obj, quantity)
        array2 = getattr(output_obj2, quantity)
    except AttributeError:
        raise AttributeError(
            "{} is not a valid field of {}".format(
                quantity, output_obj.__class__.__name__
            )
        )

    assert array.__class__ == array2.__class__

    # Do linear interpolation only.
    prev_d = scroll_distances[z_index - 1]
    this_d = scroll_distances[z_index]

    # Get the cells that need to be filled on this iteration.
    these_distances = lc_distances[
        np.logical_and(lc_distances < prev_d, lc_distances >= this_d)
    ]

    n = len(these_distances)
    ind = np.arange(-(box_index + n), -box_index)

    sub_array = array.take(ind + n_lightcone, axis=2, mode="wrap")
    sub_array2 = array2.take(ind + n_lightcone, axis=2, mode="wrap")

    out = (
        np.abs(this_d - these_distances) * sub_array
        + np.abs(prev_d - these_distances) * sub_array2
    ) / (np.abs(prev_d - this_d))
    if kind == "mean_max":
        flag = sub_array * sub_array2 < 0
        out[flag] = np.maximum(sub_array, sub_array2)[flag]
    elif kind != "mean":
        raise ValueError("kind must be 'mean' or 'mean_max'")

    lc[:, :, -(lc_index + n) : n_lightcone - lc_index] = out
    return n


def _setup_lightcone(
    cosmo_params, max_redshift, redshift, scrollz, user_params, z_step_factor
):
    # Here set up the lightcone box.
    # Get a length of the lightcone (bigger than it needs to be at first).
    d_at_redshift = cosmo_params.cosmo.comoving_distance(redshift).value
    Ltotal = (
        cosmo_params.cosmo.comoving_distance(scrollz[0] * z_step_factor).value
        - d_at_redshift
    )
    lc_distances = np.arange(0, Ltotal, user_params.BOX_LEN / user_params.HII_DIM)

    # Use max_redshift to get the actual distances we require.
    Lmax = cosmo_params.cosmo.comoving_distance(max_redshift).value - d_at_redshift
    first_greater = np.argwhere(lc_distances > Lmax)[0][0]

    # Get *at least* as far as max_redshift
    lc_distances = lc_distances[: (first_greater + 1)]

    n_lightcone = len(lc_distances)
    return d_at_redshift, lc_distances, n_lightcone


def _get_lightcone_redshifts(
    cosmo_params, max_redshift, redshift, user_params, z_step_factor
):
    scrollz = _logscroll_redshifts(redshift, z_step_factor, max_redshift)
    lc_distances = _setup_lightcone(
        cosmo_params, max_redshift, redshift, scrollz, user_params, z_step_factor
    )[1]
    lc_distances += cosmo_params.cosmo.comoving_distance(redshift).value

    return np.array(
        [
            z_at_value(cosmo_params.cosmo.comoving_distance, d * units.Mpc)
            for d in lc_distances
        ]
    )


def calibrate_photon_cons(
    user_params,
    cosmo_params,
    astro_params,
    flag_options,
    init_box,
    regenerate,
    write,
    direc,
    **global_kwargs,
):
    r"""
    Set up the photon non-conservation correction.

    Scrolls through in redshift, turning off all flag_options to construct a 21cmFAST calibration
    reionisation history to be matched to the analytic expression from solving the filling factor
    ODE.


    Parameters
    ----------
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options: :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature
        fluctuations are required.
    init_box : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be
        re-calculated.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Other Parameters
    ----------------
    regenerate, write
        See docs of :func:`initial_conditions` for more information.
    """
    direc, regenerate, write = _get_config_options(direc, regenerate, write)

    if not flag_options.PHOTON_CONS:
        return

    with global_params.use(**global_kwargs):
        # Create a new astro_params and flag_options just for the photon_cons correction
        astro_params_photoncons = deepcopy(astro_params)
        astro_params_photoncons._R_BUBBLE_MAX = astro_params.R_BUBBLE_MAX

        flag_options_photoncons = FlagOptions(
            USE_MASS_DEPENDENT_ZETA=flag_options.USE_MASS_DEPENDENT_ZETA,
            M_MIN_in_Mass=flag_options.M_MIN_in_Mass,
        )

        ib = None
        prev_perturb = None

        # Arrays for redshift and neutral fraction for the calibration curve
        z_for_photon_cons = []
        neutral_fraction_photon_cons = []

        # Initialise the analytic expression for the reionisation history
        logger.info("About to start photon conservation correction")
        _init_photon_conservation_correction(
            user_params=user_params,
            cosmo_params=cosmo_params,
            astro_params=astro_params,
            flag_options=flag_options,
        )

        # Determine the starting redshift to start scrolling through to create the
        # calibration reionisation history
        logger.info("Calculating photon conservation zstart")
        z = _calc_zstart_photon_cons()

        while z > global_params.PhotonConsEndCalibz:

            # Determine the ionisation box with recombinations, spin temperature etc.
            # turned off.
            this_perturb = perturb_field(
                redshift=z,
                init_boxes=init_box,
                regenerate=regenerate,
                write=write,
                direc=direc,
            )

            ib2 = ionize_box(
                redshift=z,
                previous_ionize_box=ib,
                init_boxes=init_box,
                perturbed_field=this_perturb,
                previous_perturbed_field=prev_perturb,
                astro_params=astro_params_photoncons,
                flag_options=flag_options_photoncons,
                spin_temp=None,
                regenerate=regenerate,
                write=write,
                direc=direc,
            )

            mean_nf = np.mean(ib2.xH_box)

            # Save mean/global quantities
            neutral_fraction_photon_cons.append(mean_nf)
            z_for_photon_cons.append(z)

            # Can speed up sampling in regions where the evolution is slower
            if 0.3 < mean_nf <= 0.9:
                z -= 0.15
            elif 0.01 < mean_nf <= 0.3:
                z -= 0.05
            else:
                z -= 0.5

            ib = ib2
            if flag_options.USE_MINI_HALOS:
                prev_perturb = this_perturb

        z_for_photon_cons = np.array(z_for_photon_cons[::-1])
        neutral_fraction_photon_cons = np.array(neutral_fraction_photon_cons[::-1])

        # Construct the spline for the calibration curve
        logger.info("Calibrating photon conservation correction")
        _calibrate_photon_conservation_correction(
            redshifts_estimate=z_for_photon_cons,
            nf_estimate=neutral_fraction_photon_cons,
            NSpline=len(z_for_photon_cons),
        )
