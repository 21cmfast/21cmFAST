"""
This is the main wrapper for the underlying 21cmFAST C-code, and the module provides a number of:

* Input-parameter classes which wrap various C structs (these are the classes ending with ``*Params`` or ``*Options``)
* Output objects which simplify access to underlying data structures, such as density, velocity and ionization fields
* Low-level functions which simplify calling the background C functions which populate these output objects given the
  input classes.
* High-level functions which provide the most efficient and simplest way to generate the most commonly desired outputs.

Along with these, the module exposes ``global_params``, which is a simple class providing read/write access to a number
of parameters used throughout the computation which are very rarely varied. These parameters can be accessed as
standard instance attributes of ``global_params``, and listed using ``dir(global_params)``. When set, they are used
globally for all proceeding calculations.

**Input Parameter Classes**

There are four input parameter/option classes, not all of which are required for any given function. They are
:class:`UserParams`, :class:`CosmoParams`, :class:`AstroParams` and :class:`FlagOptions`. Each of them defines a number
of variables, and all of these have default values, to minimize the burden on the user. These defaults are accessed via
the ``_defaults_`` class attribute of each class. The available parameters for each are listed in the documentation
for each class below.

**Output Objects**

The remainder of the classes defined in this module are output classes. These exist to simplify access to large datasets
created within C. Fundamentally, ownership of the data belongs to these classes, and the C functions merely accesses
this and fills it. The various boxes and quantities associated with each output are available as instance attributes.
Along with the output data, each output object contains the various input parameter objects necessary to define it.

.. warning:: These should not be instantiated or filled by the user, but always handled as output objects from the
             various functions contained here. Only the data within the objects should be accessed.

**Low-level functions**

The low-level functions provided here ease the production of the aforementioned output objects. Functions exist for
each low-level C routine, which have been decoupled as far as possible. So, functions exist to create
:func:`initial_conditions`, :func:`perturb_field`, :class:`ionize_box` and so on. Creating a brightness temperature
box (often the desired final output) would generally require calling each of these in turn, as each depends on the
result of a previous function. Nevertheless, each function has the capability of generating the required previous
outputs on-the-fly, so one can instantly call :func:`ionize_box` and get a self-consistent result. Doing so, while
convenient, is sometimes not *efficient*, especially when using inhomogeneous recombinations or the spin temperature
field, which intrinsically require consistent evolution of the ionization field through redshift. In these cases, for
best efficiency it is recommended to either use a customised manual approach to calling these low-level functions, or
to call a higher-level function which optimizes this process.

Finally, note that ``21CMMC`` attempts to optimize the production of the large amount of data via on-disk caching.
By default, if a previous set of data has been computed using the current input parameters, it will be read-in from
a caching repository and returned directly. This behaviour can be tuned in any of the low-level (or high-level)
functions by setting the `write`, `direc`, `regenerate` and `match_seed` parameters (see docs for
:func:`initial_conditions` for details). The function :func:`~query_cache` can be used to search the cache, and return
empty datasets corresponding to each (these can the be filled with the data merely by calling ``.read()`` on any
data set). Conversely, a specific data set can be read and returned as a proper output object by calling the
:func:`readbox` function.


**High-level functions**

As previously mentioned, calling the low-level functions in some cases is non-optimal, especially when full evolution
of the field is required, and thus iteration through a series of redshift. In addition, while
:class:`InitialConditions` and :class:`PerturbedField` are necessary intermediate data, it is *usually* the resulting
brightness temperature which is of most interest, and it is easier to not have to worry about the intermediate steps
explicitly. For these typical use-cases, two high-level functions are available: :func:`run_coeval` and
:func:`run_lightcone`, whose purpose should be self-explanatory. These will optimally run all necessary intermediate
steps (using cached results by default if possible) and return all datasets of interest.


Examples
--------
A typical example of using this module would be the following.

>>> import py21cmmc as p21

Get coeval cubes at redshift 7,8 and 9, without spin temperature or inhomogeneous recombinations:

>>> init, perturb, xHI, Tb = p21.run_coeval(
>>>                              redshift=[7,8,9],
>>>                              cosmo_params=p21.CosmoParams(hlittle=0.7),
>>>                              user_params=p21.UserParams(HII_DIM=100)
>>>                          )

Get coeval cubes at the same redshift, with both spin temperature and inhomogeneous recombinations, pulled from the
natural evolution of the fields:

>>> all_boxes = p21.run_coeval(
>>>                 redshift=[7,8,9],
>>>                 user_params=p21.UserParams(HII_DIM=100),
>>>                 flag_options=p21.FlagOptions(INHOMO_RECO=True),
>>>                 do_spin_temp=True
>>>             )

Get a self-consistent lightcone defined between z1 and z2 (`z_step_factor` changes the logarithmic steps between
redshift that are actually evaluated, which are then interpolated onto the lightcone cells):

>>> lightcone = p21.run_lightcone(redshift=z2, max_redshift=z2, z_step_factor=1.03)
"""
import logging
import os
from os import path

import numpy as np
from astropy import units
from astropy.cosmology import Planck15, z_at_value
from cached_property import cached_property

from ._21cmfast import ffi, lib
from ._utils import StructWithDefaults, OutputStruct as _OS, StructInstanceWrapper, StructWrapper
from ..mcmc import yaml

logger = logging.getLogger("21CMMC")

# Global Options
with open(path.expanduser(path.join("~", '.21CMMC', "config.yml"))) as f:
    config = yaml.load(f)

global_params = StructInstanceWrapper(lib.global_params, ffi)
EXTERNALTABLES = ffi.new("char[]", path.join(path.expanduser("~"), ".21CMMC").encode())
global_params.external_table_path = EXTERNALTABLES


# ======================================================================================================================
# PARAMETER STRUCTURES
# ======================================================================================================================
class CosmoParams(StructWithDefaults):
    """
    Cosmological parameters (with defaults) which translates to a C struct.

    To see default values for each parameter, use ``CosmoParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Parameters
    ----------
    SIGMA_8 : float, optional
        RMS mass variance (power spectrum normalisation).

    hlittle : float, optional
        H_0/100.

    OMm : float, optional
        Omega matter.

    OMb : float, optional
        Omega baryon, the baryon component.

    POWER_INDEX : float, optional
        Spectral index of the power spectrum.

    """
    _ffi = ffi

    _defaults_ = dict(
        SIGMA_8=0.82,
        hlittle=Planck15.h,
        OMm=Planck15.Om0,
        OMb=Planck15.Ob0,
        POWER_INDEX=0.97,
    )

    @property
    def OMl(self):
        """
        Omega lambda, dark energy density.
        """
        return 1 - self.OMm

    @property
    def cosmo(self):
        """
        Return an astropy cosmology object for this cosmology.
        """
        return Planck15.clone(H0=self.hlittle * 100, Om0=self.OMm, Ob0=self.OMb)


class UserParams(StructWithDefaults):
    """
    Structure containing user parameters (with defaults).

    To see default values for each parameter, use ``UserParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Parameters
    ----------
    HII_DIM : int, optional
        Number of cells for the low-res box.

    DIM : int,optional
        Number of cells for the high-res box (sampling ICs) along a principal axis. To avoid
        sampling issues, DIM should be at least 3 or 4 times HII_DIM, and an integer multiple.
        By default, it is set to 4*HII_DIM.

    BOX_LEN : float, optional
        Length of the box, in Mpc.

    HMF: int, optional
        Determines which halo mass function to be used for the normalisation of the collapsed fraction:
        0: Press-Schechter
        1: Sheth-Tormen
        2: Watson FOF
        3: Watson FOF-z

    """
    _ffi = ffi

    _defaults_ = dict(
        BOX_LEN=150.0,
        DIM=None,
        HII_DIM=50,
        USE_FFTW_WISDOM=False,
        HMF=1,
    )

    @property
    def DIM(self):
        """
        Number of cells for the high-res box (sampling ICs) along a principal axis. Dynamically generated.
        """
        return self._DIM or 4 * self.HII_DIM

    @property
    def tot_fft_num_pixels(self):
        """Number of pixels in the high-res box."""
        return self.DIM ** 3

    @property
    def HII_tot_num_pixels(self):
        """Number of pixels in the low-res box."""
        return self.HII_DIM ** 3


class FlagOptions(StructWithDefaults):
    """
    Flag-style options for the ionization routines.

    To see default values for each parameter, use ``FlagOptions._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Parameters
    ----------
    USE_MASS_DEPENDENT_ZETA : bool, optional
        Set to True if using new parameterization.

    SUBCELL_RSDS : bool, optional
        Add sub-cell RSDs (currently doesn't work if Ts is not used)

    INHOMO_RECO : bool, optional
        Whether to perform inhomogeneous recombinations

    USE_TS_FLUCT : bool, optional
        Whether to perform IGM spin temperature fluctuations (i.e. X-ray heating)

    M_MIN_in_Mass : bool, optional
        Whether the minimum mass is defined by Mass or Virial Temperature
    """

    _ffi = ffi

    _defaults_ = dict(
        USE_MASS_DEPENDENT_ZETA=False,
        SUBCELL_RSD=False,
        INHOMO_RECO=False,
        USE_TS_FLUCT=False,
        M_MIN_in_Mass=False,
    )

    @property
    def M_MIN_in_Mass(self):
        if self.USE_MASS_DEPENDENT_ZETA:
            return True

        else:
            return self._M_MIN_in_Mass


class AstroParams(StructWithDefaults):
    """
    Astrophysical parameters.

    To see default values for each parameter, use ``AstroParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Parameters
    ----------
    INHOMO_RECO : bool, optional
        Whether inhomogeneous recombinations are being calculated. This is not a part of the astro parameters structure,
        but is required by this class to set some default behaviour.
    HII_EFF_FACTOR : float, optional
    F_STAR10 : float, optional
    ALPHA_STAR : float, optional
    F_ESC10 : float, optional
    ALPHA_ESC : float, optional
    M_TURN : float, optional
    R_BUBBLE_MAX : float, optional
        Default is 50 if `INHOMO_RECO` is True, or 15.0 if not.
    ION_Tvir_MIN : float, optional
    L_X : float, optional
    NU_X_THRESH : float, optional
    X_RAY_SPEC_INDEX : float, optional
    X_RAY_Tvir_MIN : float, optional
        Default is `ION_Tvir_MIN`.
    t_STAR : float, optional
    N_RSD_STEPS : float, optional
    """

    _ffi = ffi

    _defaults_ = dict(
        HII_EFF_FACTOR=30.0,
        F_STAR10=-1.3,
        ALPHA_STAR=0.5,
        F_ESC10=-1.,
        ALPHA_ESC=-0.5,
        M_TURN=8.7,
        R_BUBBLE_MAX=None,
        ION_Tvir_MIN=4.69897,
        L_X=40.0,
        NU_X_THRESH=500.0,
        X_RAY_SPEC_INDEX=1.0,
        X_RAY_Tvir_MIN=None,
        t_STAR=0.5,
        N_RSD_STEPS=20,
    )

    def __init__(self, *args, INHOMO_RECO=FlagOptions._defaults_['INHOMO_RECO'], **kwargs):
        # TODO: should try to get inhomo_reco out of here... just needed for default of R_BUBBLE_MAX.
        self.INHOMO_RECO = INHOMO_RECO
        super().__init__(*args, **kwargs)

    def convert(self, key, val):
        if key in ['F_STAR10', 'F_ESC10', 'M_TURN', 'ION_Tvir_MIN', "L_X", "X_RAY_Tvir_MIN"]:
            return 10 ** val
        else:
            return val

    @property
    def R_BUBBLE_MAX(self):
        """Maximum radius of bubbles to be searched. Set dynamically."""
        if not self._R_BUBBLE_MAX:
            return 50.0 if self.INHOMO_RECO else 15.0
        else:
            if self.INHOMO_RECO and self._R_BUBBLE_MAX != 50:
                logger.warning("You are setting R_BUBBLE_MAX != 50 when INHOMO_RECO=True. " + \
                               "This is non-standard (but allowed), and usually occurs upon manual update of INHOMO_RECO")
            return self._R_BUBBLE_MAX

    @property
    def X_RAY_Tvir_MIN(self):
        """Minimum virial temperature of X-ray emitting sources (unlogged and set dynamically)."""
        return self._X_RAY_Tvir_MIN if self._X_RAY_Tvir_MIN else self.ION_Tvir_MIN


# ======================================================================================================================
# OUTPUT STRUCTURES
# ======================================================================================================================
class _OutputStruct(_OS):
    _global_params = global_params

    def __init__(self, *, user_params=None, cosmo_params=None, **kwargs):
        if cosmo_params is None:
            cosmo_params = CosmoParams()
        if user_params is None:
            user_params = UserParams()

        super().__init__(user_params=user_params, cosmo_params=cosmo_params, **kwargs)

    _ffi = ffi


class _OutputStructZ(_OutputStruct):
    _inputs = _OutputStruct._inputs + ['redshift']


class InitialConditions(_OutputStruct):
    """
    A class containing all initial conditions boxes.
    """
    # The filter params indicates parameters to overlook when deciding if a cached box matches current parameters.
    # It is useful for ignoring certain global parameters which may not apply to this step or its dependents.
    _filter_params = _OutputStruct._filter_params + [
        'ALPHA_UVB',  # ionization
        'EVOLVE_DENSITY_LINEARLY',  # perturb
        'SMOOTH_EVOLVED_DENSITY_FIELD',  # perturb
        'R_smooth_density',  # perturb
        'HII_ROUND_ERR',  # ionization
        'FIND_BUBBLE_ALGORITHM',  # ib
        'N_POISSON',  # ib
        'T_USE_VELOCITIES',  # bt
        'MAX_DVDR',  # bt
        'DELTA_R_HII_FACTOR',  # ib
        'HII_FILTER',  # ib
        'INITIAL_REDSHIFT',  # pf
        'HEAT_FILTER',  # st
        'CLUMPING_FACTOR',  # st
        'Z_HEAT_MAX',  # st
        'R_XLy_MAX',  # st
        'NUM_FILTER_STEPS_FOR_Ts',  # ts
        'ZPRIME_STEP_FACTOR',  # ts
        'TK_at_Z_HEAT_MAX',  # ts
        'XION_at_Z_HEAT_MAX',  # ts
        'Pop',  # ib
        "Pop2_ion",  # ib
        "Pop3_ion",  # ib
        "NU_X_BAND_MAX",  # st
        "NU_X_MAX",  # ib
    ]

    def _init_arrays(self):
        self.lowres_density = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vx = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vy = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vz = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vx_2LPT = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vy_2LPT = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vz_2LPT = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.hires_density = np.zeros(self.user_params.tot_fft_num_pixels, dtype=np.float32)

        shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)
        self.lowres_density.shape = shape
        self.lowres_vx.shape = shape
        self.lowres_vy.shape = shape
        self.lowres_vz.shape = shape
        self.lowres_vx_2LPT.shape = shape
        self.lowres_vy_2LPT.shape = shape
        self.lowres_vz_2LPT.shape = shape
        self.hires_density.shape = (self.user_params.DIM, self.user_params.DIM, self.user_params.DIM)


class PerturbedField(_OutputStructZ):
    """
    A class containing all perturbed field boxes
    """
    _filter_params = _OutputStruct._filter_params + [
        'ALPHA_UVB',  # ionization
        'HII_ROUND_ERR',  # ionization
        'FIND_BUBBLE_ALGORITHM',  # ib
        'N_POISSON',  # ib
        'T_USE_VELOCITIES',  # bt
        'MAX_DVDR',  # bt
        'DELTA_R_HII_FACTOR',  # ib
        'HII_FILTER',  # ib
        'HEAT_FILTER',  # st
        'CLUMPING_FACTOR',  # st
        'Z_HEAT_MAX',  # st
        'R_XLy_MAX',  # st
        'NUM_FILTER_STEPS_FOR_Ts',  # ts
        'ZPRIME_STEP_FACTOR',  # ts
        'TK_at_Z_HEAT_MAX',  # ts
        'XION_at_Z_HEAT_MAX',  # ts
        'Pop',  # ib
        "Pop2_ion",  # ib
        "Pop3_ion",  # ib
        "NU_X_BAND_MAX",  # st
        "NU_X_MAX",  # ib
    ]

    def _init_arrays(self):
        self.density = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.velocity = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)

        self.density.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)
        self.velocity.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)


class IonizedBox(_OutputStructZ):
    """A class containing all ionized boxes"""
    _inputs = _OutputStructZ._inputs + ['flag_options', 'astro_params']

    _filter_params = _OutputStruct._filter_params + [
        'T_USE_VELOCITIES',  # bt
        'MAX_DVDR',  # bt
    ]

    def __init__(self, astro_params=None, flag_options=None, first_box=False, **kwargs):
        if flag_options is None:
            flag_options = FlagOptions()

        if astro_params is None:
            astro_params = AstroParams(INHOMO_RECO=flag_options.INHOMO_RECO)

        self.first_box = first_box

        super().__init__(astro_params=astro_params, flag_options=flag_options, **kwargs)

    def _init_arrays(self):
        # ionized_box is always initialised to be neutral for excursion set algorithm. Hence np.ones instead of np.zeros
        self.xH_box = np.ones(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.Gamma12_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.z_re_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.dNrec_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)

        shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)
        self.xH_box.shape = shape
        self.Gamma12_box.shape = shape
        self.z_re_box.shape = shape
        self.dNrec_box.shape = shape

    @cached_property
    def global_xH(self):
        if not self.filled:
            raise AttributeError("global_xH is not defined until the ionization calculation has been performed")
        else:
            return np.mean(self.xH_box)


class TsBox(IonizedBox):
    """A class containing all spin temperature boxes"""

    def _init_arrays(self):
        self.Ts_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.x_e_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.Tk_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)

        self.Ts_box.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)
        self.x_e_box.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)
        self.Tk_box.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)

    @cached_property
    def global_Ts(self):
        if not self.filled:
            raise AttributeError("global_Ts is not defined until the ionization calculation has been performed")
        else:
            return np.mean(self.Ts_box)

    @cached_property
    def global_Tk(self):
        if not self.filled:
            raise AttributeError("global_Tk is not defined until the ionization calculation has been performed")
        else:
            return np.mean(self.Tk_box)

    @cached_property
    def global_x_e(self):
        if not self.filled:
            raise AttributeError("global_x_e is not defined until the ionization calculation has been performed")
        else:
            return np.mean(self.x_e_box)


class BrightnessTemp(IonizedBox):
    """A class containing the brightness temperature box."""

    def _init_arrays(self):
        self.brightness_temp = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)

        self.brightness_temp.shape = (self.user_params.HII_DIM, self.user_params.HII_DIM, self.user_params.HII_DIM)

    @cached_property
    def global_Tb(self):
        if not self.filled:
            raise AttributeError("global_Tb is not defined until the ionization calculation has been performed")
        else:
            return np.mean(self.brightness_temp)


# ======================================================================================================================
# HELPER FUNCTIONS
# ======================================================================================================================
def _check_compatible_inputs(*datasets, ignore=['redshift']):
    """
    Ensure that all defined input parameters for the provided datasets are equal, save for those listed in ignore.
    """

    done = []  # keeps track of inputs we've checked so we don't double check.

    for i, d in enumerate(datasets):
        # If a dataset is None, just ignore and move on.
        if d is None:
            continue

        # noinspection PyProtectedMember
        for inp in d._inputs:
            # Skip inputs that we want to ignore
            if inp in ignore:
                continue

            if inp not in done:
                for j, d2 in enumerate(datasets[(i + 1):]):
                    if d2 is None:
                        continue

                    # noinspection PyProtectedMember
                    if inp in d2._inputs and getattr(d, inp) != getattr(d2, inp):
                        raise ValueError("%s and %s are incompatible" % (d.__class__.__name__, d2.__class__.__name__))
                done += [inp]


def configure_inputs(defaults, *datasets, ignore=['redshift'], flag_none=None):
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
            raise ValueError("%s has an inconsistent value with %s" % (key, dataset.__class__.__name__))
        else:
            if val is not None:
                output[i] = val
            elif data_val is not None:
                output[i] = data_val
            elif key in flag_none:
                raise ValueError("For %s, a value must be provided in some manner" % key)
            else:
                output[i] = None

    return output


def configure_redshift(redshift, *structs):
    """
    This is a special case to check and obtain redshift from given default and structs.

    It will raise a ValueError if both redshift and all structs (and/or their redshift) have value of None, *or* if
    any of them are different from each other.
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
        raise ValueError("Either redshift must be provided, or a data set containing it.")
    else:
        return redshift


def verify_types(**kwargs):
    "Ensure each argument has a type of None or that matching its name"
    for k, v in kwargs.items():
        for j, kk in enumerate(['init', 'perturb', 'ionize', 'spin_temp']):
            if kk in k:
                break
        cls = [InitialConditions, PerturbedField, IonizedBox, TsBox][j]

        if v is not None and not isinstance(v, cls):
            raise ValueError("%s must be an instance of %s" % (k, cls.__name__))


class ParameterError(RuntimeError):
    def __init__(self):
        default_message = "21CMMC does not support this combination of parameters."
        super().__init__(default_message)


class FatalCError(Exception):
    def __init__(self):
        default_message = "21CMMC is exiting."
        super().__init__(default_message)


def _process_exitcode(exitcode):
    """
    Determine what happens for different values of the (integer) exit code from a C function
    """
    if exitcode == 0:
        pass
    elif exitcode == 1:
        raise ParameterError
    elif exitcode == 2:
        raise FatalCError


def _call_c_func(fnc, obj, direc, *args, write=True):
    exitcode = fnc(*[arg() if isinstance(arg, StructWrapper) else arg for arg in args], obj())

    _process_exitcode(exitcode)
    obj.filled = True
    obj._expose()

    # Optionally do stuff with the result (like writing it)
    if write:
        obj.write(direc)

    return obj


# ======================================================================================================================
# WRAPPING FUNCTIONS
# ======================================================================================================================
def compute_tau(*, redshifts, global_xHI, user_params=None, cosmo_params=None):
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)

    if len(redshifts) != len(global_xHI):
        raise ValueError("redshifts and global_xHI must have same length")

    # Convert the data to the right type
    redshifts = np.array(redshifts, dtype='float32')
    global_xHI = np.array(global_xHI, dtype='float32')

    z = ffi.cast("float *", ffi.from_buffer(redshifts))
    xHI = ffi.cast("float *", ffi.from_buffer(global_xHI))

    # Run the C code
    return lib.ComputeTau(user_params(), cosmo_params(), len(redshifts), z, xHI)


def compute_luminosity_function(*, redshifts, user_params=None, cosmo_params=None, astro_params=None, flag_options=None,
                                nbins=100):
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    astro_params = AstroParams(astro_params)
    flag_options = FlagOptions(flag_options)

    redshifts = np.array(redshifts, dtype='float32')

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
        nbins, user_params(), cosmo_params(), astro_params(),
        flag_options(), len(redshifts), ffi.cast("float *", ffi.from_buffer(redshifts)),
        c_Muvfunc, c_Mhfunc, c_lfunc
    )

    _process_exitcode(errcode)

    lfunc[lfunc <= -30] = np.nan

    return Muvfunc, Mhfunc, lfunc


def initial_conditions(*, user_params=None, cosmo_params=None, random_seed=None, regenerate=False, write=True,
                       direc=None):
    """
    Compute initial conditions.

    Parameters
    ----------
    user_params : :class:`~UserParams` instance, optional
        Defines the overall options and parameters of the run.

    cosmo_params : :class:`~CosmoParams` instance, optional
        Defines the cosmological parameters used to compute initial conditions.

    regenerate : bool, optional
        Whether to force regeneration of data, even if matching cached data is found. This is applied recursively to
        any potential sub-calculations. It is ignored in the case of dependent data only if that data is explicitly
        passed to the function.

    write : bool, optional
        Whether to write results to file (i.e. cache). This is recursively applied to any potential sub-calculations.

    direc : str, optional
        The directory in which to search for the boxes and write them. By default, this is the directory given by
        ``boxdir`` in the configuration file, ``~/.21CMMC/config.yml``. This is recursively applied to any potential
        sub-calculations.

    Returns
    -------
    :class:`~InitialConditions`
    """
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)

    # Initialize memory for the boxes that will be returned.
    boxes = InitialConditions(
        user_params=user_params,
        cosmo_params=cosmo_params,
        random_seed=random_seed
    )

    # First check whether the boxes already exist.
    if not regenerate:
        try:
            boxes.read(direc)
            logger.info("Existing init_boxes found and read in (seed=%s)." % boxes.random_seed)
            return boxes
        except IOError:
            pass

    return _call_c_func(
        lib.ComputeInitialConditions, boxes, direc,
        boxes.random_seed, boxes.user_params, boxes.cosmo_params,
        write=write
    )


def perturb_field(*, redshift, init_boxes=None, user_params=None, cosmo_params=None, random_seed=None,
                  regenerate=False, write=True, direc=None):
    """
    Compute a perturbed field at a given redshift.

    Parameters
    ----------
    redshift : float
        The redshift at which to compute the perturbed field.

    init_boxes : :class:`~InitialConditions`, optional
        If given, these initial conditions boxes will be used, otherwise initial conditions will be generated. If given,
        the user and cosmo params will be set from this object.

    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.

    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.

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

    Doing so will internally call the :func:`~initial_conditions` function. If initial conditions have already been
    calculated, this can be avoided by passing them:

    >>> init_boxes = initial_conditions()
    >>> field7 = perturb_field(7.0, init_boxes)
    >>> field8 = perturb_field(8.0, init_boxes)

    The user and cosmo parameter structures are by default inferred from the ``init_boxes``, so that the following is
    consistent::

    >>> init_boxes = initial_conditions(user_params= UserParams(HII_DIM=1000))
    >>> field7 = perturb_field(7.0, init_boxes)

    If ``init_boxes`` is not passed, then these parameters can be directly passed::

    >>> field7 = perturb_field(7.0, user_params=UserParams(HII_DIM=1000))

    """
    verify_types(init_boxes=init_boxes)

    # Configure and check input/output parameters/structs
    random_seed, user_params, cosmo_params = configure_inputs(
        [("random_seed", random_seed), ("user_params", user_params), ("cosmo_params", cosmo_params)],
        init_boxes
    )

    # Verify input parameter structs (need to do this after configure_inputs).
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)

    # Initialize perturbed boxes.
    fields = PerturbedField(redshift=redshift, user_params=user_params, cosmo_params=cosmo_params,
                            random_seed=random_seed)

    # Check whether the boxes already exist
    if not regenerate:
        try:
            fields.read(direc)
            logger.info(
                "Existing z=%s perturb_field boxes found and read in (seed=%s)." % (redshift, fields.random_seed))
            return fields
        except IOError:
            pass

    # Make sure we've got computed init boxes.
    if init_boxes is None or not init_boxes.filled:
        init_boxes = initial_conditions(
            user_params=user_params, cosmo_params=cosmo_params,
            regenerate=regenerate, write=write, direc=direc, random_seed=random_seed
        )

        # Need to update fields to have the same seed as init_boxes
        fields._random_seed = init_boxes.random_seed

    # Run the C Code
    return _call_c_func(
        lib.ComputePerturbField, fields, direc,
        redshift, fields.user_params, fields.cosmo_params, init_boxes,
        write=write
    )


def ionize_box(*, astro_params=None, flag_options=None,
               redshift=None, perturbed_field=None,
               previous_ionize_box=None, z_step_factor=global_params.ZPRIME_STEP_FACTOR, z_heat_max=None,
               spin_temp=None,
               init_boxes=None, cosmo_params=None, user_params=None,
               regenerate=False, write=True, direc=None, random_seed=None):
    """
    Compute an ionized box at a given redshift.

    This function has various options for how the evolution of the ionization is computed (if at all). See the Notes
    below for details.

    Parameters
    ----------
    astro_params: :class:`~AstroParams` instance, optional
        The astrophysical parameters defining the course of reionization.

    flag_options: :class:`~FlagOptions` instance, optional
        Some options passed to the reionization routine.

    redshift : float, optional
        The redshift at which to compute the ionized box. If `perturbed_field` is given, its inherent redshift
        will take precedence over this argument. If not, this argument is mandatory.

    perturbed_field : :class:`~PerturbField`, optional
        If given, this field will be used, otherwise it will be generated. To be generated, either `init_boxes` and
        `redshift` must be given, or `user_params`, `cosmo_params` and `redshift`.

    init_boxes : :class:`~InitialConditions` , optional
        If given, and `perturbed_field` *not* given, these initial conditions boxes will be used to generate the
        perturbed field, otherwise initial conditions will be generated on the fly. If given,
        the user and cosmo params will be set from this object.

    previous_ionize_box: :class:`IonizedBox` or None
        An ionized box at higher redshift. This is only used if `INHOMO_RECO` and/or `do_spin_temp` are true. If either
        of these are true, and this is not given, then it will be assumed that this is the "first box", i.e. that it
        can be populated accurately without knowing source statistics.

    z_step_factor: float, optional
        A factor greater than unity, which specifies the logarithmic steps in redshift with which the spin temperature
        box is evolved.

    z_heat_max: float, optional
        The maximum redshift at which to search for heating sources. Practically, this defines the limit in redshift
        at which the spin temperature can be defined purely from the background perturbed field rather than by evolving
        from a previous spin temperature field. Default is the global parameter `Z_HEAT_MAX`.

    spin_temp: :class:`TsBox` or None, optional
        A spin-temperature box, only required if `do_spin_temp` is True.
        If None, will try to read in a spin temp box at the current redshift, and failing that will try to
        automatically create one, using the previous ionized box redshift as the previous spin temperature redshift.

    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.

    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.

    Returns
    -------
    :class:`~IonizedBox`
        An object containing the ionized box data.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed:
        See docs of :func:`initial_conditions` for more information.

    Notes
    -----

    Typically, the ionization field at any redshift is dependent on the evolution of xHI up until
    that redshift, which necessitates providing a previous ionization field to define the current one. This
    function provides several options for doing so. First, if neither the spin temperature field, nor inhomogeneous
    recombinations (specified in flag options) are used, no evolution needs to be done. Otherwise, either (in order of
    precedence) (i) a specific previous :class`~IonizedBox` object is provided, which will be used directly,
    (ii) a previous redshift is provided, for which a cached field on disk will be sought, (iii) a step factor is
    provided which recursively steps through redshift, calculating previous fields up until Z_HEAT_MAX, and returning
    just the final field at the current redshift, or (iv) the function is instructed to treat the current field as
    being an initial "high-redshift" field such that specific sources need not be found and evolved.

    .. note:: If a previous specific redshift is given, but no cached field is found at that redshift, the previous
              ionization field will be evaluated based on `z_step_factor`.

    Examples
    --------
    By default, no spin temperature is used, and neither are inhomogeneous recombinations, so that no evolution is
    required, thus the following will compute a coeval ionization box:

    >>> xHI = ionize_box(redshift=7.0)

    However, if either of those options are true, then a full evolution will be required:

    >>> xHI = ionize_box(redshift=7.0, flag_options=FlagOptions(INHOMO_RECO=True, USE_TS_FLUCT=True))

    This will by default evolve the field from a redshift of *at least* `Z_HEAT_MAX` (a global parameter), in logarithmic
    steps of `z_step_factor`. Thus to change these:

    >>> xHI = ionize_box(redshift=7.0, z_step_factor=1.2, z_heat_max=15.0, flag_options={"USE_TS_FLUCT":True})

    Alternatively, one can pass an exact previous redshift, which will be sought in the disk cache, or evaluated:

    >>> ts_box = ionize_box(redshift=7.0, previous_ionize_box=8.0, flag_options={"USE_TS_FLUCT":True})

    Beware that doing this, if the previous box is not found on disk, will continue to evaluate prior boxes based on the
    `z_step_factor`. Alternatively, one can pass a previous :class:`~IonizedBox`:

    >>> xHI_0 = ionize_box(redshift=8.0, flag_options={"USE_TS_FLUCT":True})
    >>> xHI = ionize_box(redshift=7.0, previous_ionize_box=xHI_0)

    Again, the first line here will implicitly use `z_step_factor` to evolve the field from ~`Z_HEAT_MAX`. Note that
    in the second line, all of the input parameters are taken directly from `xHI_0` so that they are consistent, and
    we need not specify the ``flag_options``.
    Finally, one can force the function to evaluate the current redshift as if it was beyond Z_HEAT_MAX so that it
    depends only on itself:

    >>> xHI = ionize_box(redshift=7.0, z_step_factor=None, flag_options={"USE_TS_FLUCT":True})

    This is usually a bad idea, and will give a warning, but it is possible.

    As the function recursively evaluates previous redshift, the previous spin temperature fields will also be
    consistently recursively evaluated. Only the final ionized box will actually be returned and kept in memory, however
    intervening results will by default be cached on disk. One can also pass an explicit spin temperature object:

    >>> ts = spin_temperature(redshift=7.0)
    >>> xHI = ionize_box(redshift=7.0, spin_temp=ts)

    If automatic recursion is used, then it is done in such a way that no large boxes are kept around in memory for
    longer than they need to be (only two at a time are required).
    """
    verify_types(init_boxes=init_boxes, perturbed_field=perturbed_field, previous_ionize_box=previous_ionize_box,
                 spin_temp=spin_temp)

    # Configure and check input/output parameters/structs
    random_seed, user_params, cosmo_params, astro_params, flag_options = configure_inputs(
        [("random_seed", random_seed), ("user_params", user_params), ("cosmo_params", cosmo_params),
         ('astro_params', astro_params), ("flag_options", flag_options)],
        init_boxes, spin_temp, init_boxes, perturbed_field, previous_ionize_box
    )
    redshift = configure_redshift(redshift, spin_temp, perturbed_field)

    # Verify input structs
    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    flag_options = FlagOptions(flag_options)
    astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

    if spin_temp is not None and not flag_options.USE_TS_FLUCT:
        logger.warning("Changing flag_options.USE_TS_FLUCT to True since spin_temp was passed.")
        flag_options.USE_TS_FLUCT = True

    # Set the upper limit on redshift at which we require a previous spin temp box.
    if z_heat_max is not None:
        global_params.Z_HEAT_MAX = z_heat_max
    if z_step_factor is not None:
        global_params.ZPRIME_STEP_FACTOR = z_step_factor

    box = IonizedBox(
        first_box=((1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1) > global_params.Z_HEAT_MAX and (
                not isinstance(previous_ionize_box, IonizedBox) or not previous_ionize_box.filled),
        user_params=user_params, cosmo_params=cosmo_params,
        redshift=redshift, astro_params=astro_params, flag_options=flag_options, random_seed=random_seed
    )

    # Check whether the boxes already exist
    if not regenerate:
        try:
            box.read(direc)
            logger.info("Existing z=%s ionized boxes found and read in (seed=%s)." % (redshift, box.random_seed))
            return box
        except IOError:
            pass

    # EVERYTHING PAST THIS POINT ONLY HAPPENS IF THE BOX DOESN'T ALREADY EXIST
    # ------------------------------------------------------------------------
    # Get the previous redshift
    if flag_options.INHOMO_RECO or flag_options.USE_TS_FLUCT:

        if previous_ionize_box is not None:
            prev_z = previous_ionize_box.redshift
        elif z_step_factor is not None:
            prev_z = (1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1
        else:
            prev_z = None
            if redshift < global_params.Z_HEAT_MAX:
                logger.warning(
                    "Attempting to evaluate ionization field at z=%s as if it was beyond Z_HEAT_MAX=%s" % (
                        redshift, global_params.Z_HEAT_MAX))

        # Ensure the previous spin temperature has a higher redshift than this one.
        if prev_z and prev_z <= redshift:
            raise ValueError("Previous ionized box must have a higher redshift than that being evaluated.")
    else:
        prev_z = None

    # Get init_box required.
    if init_boxes is None or not init_boxes.filled:
        init_boxes = initial_conditions(
            user_params=user_params, cosmo_params=cosmo_params,
            regenerate=regenerate, write=write, direc=direc,
            random_seed=random_seed
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
                astro_params=astro_params, flag_options=flag_options, redshift=prev_z,
                z_step_factor=z_step_factor, z_heat_max=z_heat_max,
                init_boxes=init_boxes, regenerate=regenerate, write=write, direc=direc,
            )

    # Dynamically produce the perturbed field.
    if perturbed_field is None or not perturbed_field.filled:
        perturbed_field = perturb_field(
            init_boxes=init_boxes,
            # NOTE: this is required, rather than using cosmo_ and user_, since init may have a set seed.
            redshift=redshift,
            regenerate=regenerate, write=write, direc=direc,
        )

    # Set empty spin temp box if necessary.
    if not flag_options.USE_TS_FLUCT:
        spin_temp = TsBox(redshift=0)
    elif spin_temp is None:
        spin_temp = spin_temperature(
            perturbed_field=perturbed_field,
            z_step_factor=z_step_factor,
            z_heat_max=z_heat_max,
            flag_options=flag_options,
            init_boxes=init_boxes,
            direc=direc, write=write, regenerate=regenerate
        )

    # Run the C Code
    return _call_c_func(
        lib.ComputeIonizedBox, box, direc,
        redshift, previous_ionize_box.redshift, box.user_params, box.cosmo_params, box.astro_params, box.flag_options,
        perturbed_field, previous_ionize_box, spin_temp,
        write=write
    )


def spin_temperature(*, astro_params=None, flag_options=None, redshift=None, perturbed_field=None,
                     previous_spin_temp=None, z_step_factor=1.02, z_heat_max=None,
                     init_boxes=None, cosmo_params=None, user_params=None, regenerate=False,
                     write=True, direc=None, random_seed=None):
    """
    Compute spin temperature boxes at a given redshift.

    See the notes below for how the spin temperature field is evolved through redshift.

    Parameters
    ----------
    astro_params: :class:`~AstroParams`, optional
        The astrophysical parameters defining the course of reionization.

    flag_options: :class:`~FlagOptions`, optional
        Some options passed to the reionization routine.

    redshift : float, optional
        The redshift at which to compute the ionized box. If not given, the redshift from `perturbed_field` will be used.
        Either `redshift`, `perturbed_field`, or `previous_spin_temp` must be given. See notes on `perturbed_field` for
        how it affects the given redshift if both are given.

    perturbed_field : :class:`~PerturbField`, optional
        If given, this field will be used, otherwise it will be generated. To be generated, either `init_boxes` and
        `redshift` must be given, or `user_params`, `cosmo_params` and `redshift`. By default, this will be generated
        at the same redshift as the spin temperature box. The redshift of perturb field is allowed to be different
        than `redshift`. If so, it will be interpolated to the correct redshift, which can provide a speedup compared
        to actually computing it at the desired redshift.

    previous_spin_temp : :class:`TsBox` or None
        The previous spin temperature box.

    z_step_factor: float, optional
        A factor greater than unity, which specifies the logarithmic steps in redshift with which the spin temperature
        box is evolved. If None, the code will assume that this is the first box in the evolution process, and generate
        the spin temp directly from the perturbed field.

    z_heat_max: float, optional
        The maximum redshift at which to search for heating sources. Practically, this defines the limit in redshift
        at which the spin temperature can be defined purely from the background perturbed field rather than by evolving
        from a previous spin temperature field. Default is the global parameter `Z_HEAT_MAX`.

    init_boxes : :class:`~InitialConditions`, optional
        If given, and `perturbed_field` *not* given, these initial conditions boxes will be used to generate the
        perturbed field, otherwise initial conditions will be generated on the fly. If given,
        the user and cosmo params will be set from this object.

    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.

    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.

    use_interp_perturb_field : bool, optional
        Whether to use a single perturb field, at the lowest redshift of the lightcone, to determine all spin
        temperature fields. If so, this field is interpolated in the underlying C-code to the correct redshift.
        This is less accurate, but provides compatibility with older versions of 21cmMC.

    Returns
    -------
    :class:`~TsBox`
        An object containing the spin temperature box data.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed:
        See docs of :func:`initial_conditions` for more information.

    Notes
    -----

    Typically, the spin temperature field at any redshift is dependent on the evolution of spin temperature up until
    that redshift, which necessitates providing a previous spin temperature field to define the current one. This
    function provides several options for doing so. Either (in order of precedence) (i) a specific previous spin
    temperature object is provided, which will be used directly, (ii) a previous redshift is provided, for which a
    cached field on disk will be sought, (iii) a step factor is provided which recursively steps through redshift,
    calculating previous fields up until Z_HEAT_MAX, and returning just the final field at the current redshift, or
    (iv) the function is instructed to treat the current field as being an initial "high-redshift" field such that
    specific sources need not be found and evolved.

    .. note:: If a previous specific redshift is given, but no cached field is found at that redshift, the previous
              spin temperature field will be evaluated based on `z_step_factor`.

    Examples
    --------
    To calculate and return a fully evolved spin temperature field at a given redshift (with default input parameters),
    simply use:

    >>> ts_box = spin_temperature(redshift=7.0)

    This will by default evolve the field from a redshift of *at least* `Z_HEAT_MAX` (a global parameter), in logarithmic
    steps of `z_step_factor`. Thus to change these:

    >>> ts_box = spin_temperature(redshift=7.0, z_step_factor=1.2, z_heat_max=15.0)

    Alternatively, one can pass an exact previous redshift, which will be sought in the disk cache, or evaluated:

    >>> ts_box = spin_temperature(redshift=7.0, previous_spin_temp=8.0)

    Beware that doing this, if the previous box is not found on disk, will continue to evaluate prior boxes based on the
    `z_step_factor`. Alternatively, one can pass a previous spin temperature box:

    >>> ts_box1 = spin_temperature(redshift=8.0)
    >>> ts_box = spin_temperature(redshift=7.0, previous_spin_temp=ts_box1)

    Again, the first line here will implicitly use `z_step_factor` to evolve the field from ~`Z_HEAT_MAX`. Note that
    in the second line, all of the input parameters are taken directly from `ts_box1` so that they are consistent.
    Finally, one can force the function to evaluate the current redshift as if it was beyond Z_HEAT_MAX so that it
    depends only on itself:

    >>> ts_box = spin_temperature(redshift=7.0, z_step_factor=None)

    This is usually a bad idea, and will give a warning, but it is possible.
    """
    verify_types(init_boxes=init_boxes, perturbed_field=perturbed_field, previous_spin_temp=previous_spin_temp)

    # Configure and check input/output parameters/structs
    random_seed, user_params, cosmo_params, astro_params, flag_options = configure_inputs(
        [("random_seed", random_seed), ("user_params", user_params), ("cosmo_params", cosmo_params),
         ('astro_params', astro_params), ("flag_options", flag_options)],
        init_boxes, previous_spin_temp, init_boxes, perturbed_field,
    )

    # Try to determine redshift from other inputs, if required.
    # Note that perturb_field does not need to match redshift here.
    if redshift is None:
        if perturbed_field is not None:
            redshift = perturbed_field.redshift
        elif previous_spin_temp is not None:
            redshift = (previous_spin_temp.redshift + 1) / global_params.ZPRIME_STEP_FACTOR - 1

    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    flag_options = FlagOptions(flag_options)
    astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

    # Explicitly set this flag to True, though it shouldn't be required!
    flag_options.USE_TS_FLUCT = True

    # Set the upper limit on redshift at which we require a previous spin temp box.
    if z_heat_max is not None:
        global_params.Z_HEAT_MAX = z_heat_max
    if z_step_factor is not None:
        global_params.ZPRIME_STEP_FACTOR = z_step_factor

    # If there is still no redshift, raise error.
    if redshift is None:
        raise ValueError("Either the redshift, perturbed_field or previous_spin_temp must be given.")

    box = TsBox(
        first_box=((1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1) > global_params.Z_HEAT_MAX and (
                not isinstance(previous_spin_temp, IonizedBox) or not previous_spin_temp.filled),
        user_params=user_params, cosmo_params=cosmo_params,
        redshift=redshift, astro_params=astro_params, flag_options=flag_options, random_seed=random_seed
    )

    # Check whether the boxes already exist on disk.
    if not regenerate:
        try:
            box.read(direc)
            logger.info("Existing z=%s spin_temp boxes found and read in (seed=%s)." % (redshift, box.random_seed))
            return box
        except IOError:
            pass

    # EVERYTHING PAST THIS POINT ONLY HAPPENS IF THE BOX DOESN'T ALREADY EXIST
    # ------------------------------------------------------------------------

    # Get the previous redshift
    if previous_spin_temp is not None:
        prev_z = previous_spin_temp.redshift
    elif z_step_factor is not None:
        prev_z = (1 + redshift) * global_params.ZPRIME_STEP_FACTOR - 1
    else:
        prev_z = None
        if redshift < global_params.Z_HEAT_MAX:
            logger.warning("Attempting to evaluate spin temperature field at z=%s as if it was beyond Z_HEAT_MAX=%s" % (
                redshift, global_params.Z_HEAT_MAX))

    # Ensure the previous spin temperature has a higher redshift than this one.
    if prev_z and prev_z <= redshift:
        raise ValueError("Previous spin temperature box must have a higher redshift than that being evaluated.")

    # Dynamically produce the initial conditions.
    if init_boxes is None or not init_boxes.filled:
        init_boxes = initial_conditions(
            user_params=user_params, cosmo_params=cosmo_params,
            regenerate=regenerate, write=write, direc=direc, random_seed=random_seed
        )

        # Need to update random seed
        box._random_seed = init_boxes.random_seed

    # Create appropriate previous_spin_temp
    if not isinstance(previous_spin_temp, TsBox):
        if prev_z > global_params.Z_HEAT_MAX or prev_z is None:
            previous_spin_temp = TsBox(redshift=0)
        else:
            previous_spin_temp = spin_temperature(
                init_boxes=init_boxes,
                astro_params=astro_params, flag_options=flag_options, redshift=prev_z,
                z_step_factor=z_step_factor, z_heat_max=z_heat_max,
                regenerate=regenerate, write=write, direc=direc
            )

    # Dynamically produce the perturbed field.
    if perturbed_field is None or not perturbed_field.filled:
        perturbed_field = perturb_field(
            redshift=redshift,
            init_boxes=init_boxes,
            regenerate=regenerate, write=write, direc=direc,
        )

    if previous_spin_temp is None:
        previous_spin_temp = TsBox(redshift=0)

        # Run the C Code
    return _call_c_func(
        lib.ComputeTsBox, box, direc,
        redshift, previous_spin_temp.redshift, box.user_params, box.cosmo_params, box.astro_params, box.flag_options,
        perturbed_field.redshift, perturbed_field, previous_spin_temp,
        write=write
    )


def brightness_temperature(*, ionized_box, perturbed_field, spin_temp=None, write=False, direc=None):
    """
    Compute a coeval brightness temperature box.

    Parameters
    ----------
    ionized_box: :class:`IonizedBox`
        A pre-computed ionized box.

    perturbed_field: :class:`PerturbedField`
        A pre-computed perturbed field at the same redshift as `ionized_box`.

    spin_temp: :class:`TsBox`, optional
        A pre-computed spin temperature, at the same redshift as the other boxes.

    Returns
    -------
    :class:`BrightnessTemp` instance.
    """
    verify_types(perturbed_field=perturbed_field, spin_temp=spin_temp, ionized_box=ionized_box)

    # don't ignore redshift here
    _check_compatible_inputs(ionized_box, perturbed_field, spin_temp, ignore=[])

    # ensure ionized_box and perturbed_field aren't None, as we don't do
    # any dynamic calculations here.
    if ionized_box is None or perturbed_field is None:
        raise ValueError("both ionized_box and perturbed_field must be specified.")

    if spin_temp is None:
        saturated_limit = True
        spin_temp = TsBox(redshift=0)
    else:
        saturated_limit = False

    box = BrightnessTemp(user_params=ionized_box.user_params, cosmo_params=ionized_box.cosmo_params,
                         astro_params=ionized_box.astro_params, flag_options=ionized_box.flag_options,
                         redshift=ionized_box.redshift)

    return _call_c_func(
        lib.ComputeBrightnessTemp, box, direc,
        ionized_box.redshift, saturated_limit, ionized_box.user_params, ionized_box.cosmo_params,
        ionized_box.astro_params,
        ionized_box.flag_options, spin_temp, ionized_box, perturbed_field,
        write=write
    )


def _logscroll_redshifts(min_redshift, z_step_factor, zmax):
    redshifts = [min_redshift]  # mult by 1.001 is probably bad...
    while redshifts[-1] < zmax:
        redshifts.append((redshifts[-1] + 1.) * z_step_factor - 1.)
    return redshifts[::-1]


def run_coeval(*, redshift=None, user_params=None, cosmo_params=None, astro_params=None,
               flag_options=None, regenerate=False, write=True, direc=None,
               z_step_factor=global_params.ZPRIME_STEP_FACTOR, z_heat_max=None, init_box=None, perturb=None,
               use_interp_perturb_field=False,
               random_seed=None):
    """
    Evaluates a coeval ionized box at a given redshift, or multiple redshift.

    This is generally the easiest and most efficient way to generate a set of coeval cubes at a given set of redshift.
    It self-consistently deals with situations in which the field needs to be evolved, and does this with the highest
    memory-efficiency, only returning the desired redshift. All other calculations are by default stored in the
    on-disk cache so they can be re-used at a later time.

    .. note:: User-supplied redshift are *not* used as previous redshift in any scrolling, so that pristine
              log-sampling can be maintained.

    Parameters
    ----------
    redshift: array_like
        A single redshift, or multiple redshift, at which to return results. The minimum of these
        will define the log-scrolling behaviour (if necessary).
    user_params : :class:`~UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    astro_params: :class:`~AstroParams`, optional
        The astrophysical parameters defining the course of reionization.
    flag_options: :class:`~FlagOptions`, optional
        Some options passed to the reionization routine.
    z_step_factor: float, optional
        How large the logarithmic steps between redshift are (if required).
    z_heat_max: float, optional
        Controls the global `Z_HEAT_MAX` parameter, which specifies the maximum redshift up to which heating sources
        are required to specify the ionization field. Beyond this, the ionization field is specified directly from
        the perturbed density field.
    init_box : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be re-calculated.
    perturb : list of :class:`~PerturbedField`s, optional
        If given, must be compatible with init_box. It will merely negate the necessity of re-calculating the
        perturb fields.
    use_interp_perturb_field : bool, optional
        Whether to use a single perturb field, at the lowest redshift of the lightcone, to determine all spin
        temperature fields. If so, this field is interpolated in the underlying C-code to the correct redshift.
        This is less accurate (and no more efficient), but provides compatibility with older versions of 21cmMC.

    Returns
    -------
    init_box: :class:`~InitialConditions`
        The initial conditions data.

    perturb: :class:`~PerturbedField` or list thereof
        The perturbed field at the given redshift(s)

    xHI: :class:`~IonizedBox` or list thereof
        The ionization field(s).

    bt: :class:`~BrightnessTemp` or list thereof
        The brightness temperature box(es)

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed:
        See docs of :func:`initial_conditions` for more information.
    """
    # Ensure perturb is a list of boxes, not just one.
    if perturb is not None:
        if not hasattr(perturb, "__len__"):
            perturb = [perturb]
    else:
        perturb = []

    random_seed, user_params, cosmo_params = configure_inputs(
        [("random_seed", random_seed), ("user_params", user_params), ("cosmo_params", cosmo_params)],
        init_box, *perturb
    )

    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    flag_options = FlagOptions(flag_options)
    astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

    if redshift is None and perturb is None:
        raise ValueError("Either redshift or perturb must be given")

    if z_heat_max:
        global_params.Z_HEAT_MAX = z_heat_max
    if z_step_factor is not None:
        global_params.ZPRIME_STEP_FACTOR = z_step_factor

    if init_box is None:  # no need to get cosmo, user params out of it.
        init_box = initial_conditions(
            user_params=user_params, cosmo_params=cosmo_params, random_seed=random_seed,
            write=write, regenerate=regenerate, direc=direc)

    if perturb:
        if redshift is not None:
            if not all([p.redshift == z for p, z in zip(perturb, redshift)]):
                raise ValueError("Input redshifts do not match perturb field redshifts")

        else:
            redshift = [p.redshift for p in perturb]

    singleton = False
    if not hasattr(redshift, "__len__"):
        singleton = True
        redshift = [redshift]

    if not perturb:
        for z in redshift:
            perturb += [perturb_field(redshift=z, init_boxes=init_box, regenerate=regenerate, write=write, direc=direc)]

    # Get the list of redshift we need to scroll through.
    if flag_options.INHOMO_RECO or flag_options.USE_TS_FLUCT:
        redshifts = _logscroll_redshifts(min(redshift), global_params.ZPRIME_STEP_FACTOR, global_params.Z_HEAT_MAX)
    else:
        redshifts = [min(redshift)]

    # Add in the redshift defined by the user, and sort in order
    # Turn into a set so that exact matching user-set redshift
    # don't double-up with scrolling ones.
    redshifts += redshift
    redshifts = sorted(list(set(redshifts)), reverse=True)

    ib_tracker = [0] * len(redshift)
    bt = [0] * len(redshift)
    st, ib = None, None  # At first we don't have any "previous" st or ib.
    logger.debug("redshifts: %s", redshifts)

    minarg = np.argmin(redshift)

    # Iterate through redshift from top to bottom
    for z in redshifts:

        if flag_options.USE_TS_FLUCT:
            logger.debug("PID={} doing spin temp for z={}".format(os.getpid(), z))
            st2 = spin_temperature(
                redshift=z,
                previous_spin_temp=st,
                perturbed_field=perturb[minarg] if use_interp_perturb_field else (
                    perturb[redshift.index(z)] if z in redshift else None),
                # remember that perturb field is interpolated, so no need to provide exact one.
                astro_params=astro_params, flag_options=flag_options,
                regenerate=regenerate,
                init_boxes=init_box,
                write=write, direc=direc, z_heat_max=global_params.Z_HEAT_MAX, z_step_factor=z_step_factor
            )

            if z not in redshift:
                st = st2

        logger.debug("PID={} doing ionize box for z={}".format(os.getpid(), z))
        ib2 = ionize_box(
            redshift=z, previous_ionize_box=ib,
            init_boxes=init_box,
            perturbed_field=perturb[redshift.index(z)] if z in redshift else None,
            # perturb field *not* interpolated here.
            astro_params=astro_params, flag_options=flag_options,
            spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
            regenerate=regenerate, z_heat_max=global_params.Z_HEAT_MAX,
            write=write, direc=direc,
        )

        if z not in redshift:
            ib = ib2
        else:
            logger.debug("PID={} doing brightness temp for z={}".format(os.getpid(), z))
            ib_tracker[redshift.index(z)] = ib2
            bt[redshift.index(z)] = brightness_temperature(ionized_box=ib2, perturbed_field=perturb[redshift.index(z)],
                                                           spin_temp=st2 if flag_options.USE_TS_FLUCT else None)

    # If a single redshift was passed, then pass back singletons.
    if singleton:
        logger.debug("PID={} making into singleton".format(os.getpid()))
        ib_tracker = ib_tracker[0]
        bt = bt[0]
        perturb = perturb[0]

    logger.debug("PID={} RETURNING FROM COEVAL".format(os.getpid()))
    return init_box, perturb, ib_tracker, bt


class LightCone:
    def __init__(self, redshift, user_params, cosmo_params, astro_params, flag_options, brightness_temp,
                 node_redshifts=None, global_xHI=None, global_brightness_temp=None):
        self.redshift = redshift
        self.user_params = user_params
        self.cosmo_params = cosmo_params
        self.astro_params = astro_params
        self.flag_options = flag_options
        self.brightness_temp = brightness_temp

        self.node_redshifts = node_redshifts
        self.global_xHI = global_xHI
        self.global_brightness_temp = global_brightness_temp

    @property
    def cell_size(self):
        return self.user_params.BOX_LEN / self.user_params.HII_DIM

    @property
    def lightcone_dimensions(self):
        return (self.user_params.BOX_LEN, self.user_params.BOX_LEN,
                self.n_slices * self.cell_size)

    @property
    def shape(self):
        return self.brightness_temp.shape

    @property
    def n_slices(self):
        return self.shape[-1]

    @property
    def lightcone_coords(self):
        return np.linspace(0, self.lightcone_dimensions[-1], self.n_slices)

    @property
    def lightcone_distances(self):
        return self.cosmo_params.cosmo.comoving_distance(self.redshift).value + self.lightcone_coords

    @property
    def lightcone_redshifts(self):
        return np.array(
            [z_at_value(self.cosmo_params.cosmo.comoving_distance, d * units.Mpc) for d in self.lightcone_distances])


def run_lightcone(*, redshift=None, max_redshift=None, user_params=None, cosmo_params=None,
                  astro_params=None, flag_options=None, regenerate=False, write=True,
                  direc=None, z_step_factor=global_params.ZPRIME_STEP_FACTOR, z_heat_max=None, init_box=None,
                  perturb=None, random_seed=None,
                  use_interp_perturb_field=False):
    """
    Evaluates a full lightcone ending at a given redshift.

    This is generally the easiest and most efficient way to generate a lightcone, though it can be done manually by
    using the lower-level functions which are called by this function.

    Parameters
    ----------
    redshift: float
        The minimum redshift of the lightcone.
    max_redshift: float, optional
        The maximum redshift at which to keep lightcone information. By default, this is equal to `z_heat_max`.
        Note that this is not *exact*, but will be typically slightly exceeded.
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options: :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature fluctuations are required.
    z_step_factor: float, optional
        How large the logarithmic steps between redshift are (if required).
    z_heat_max: float, optional
        Controls the global `Z_HEAT_MAX` parameter, which specifies the maximum redshift up to which heating sources
        are required to specify the ionization field. Beyond this, the ionization field is specified directly from
        the perturbed density field.
    init_box : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be re-calculated.
    perturb : list of :class:`~PerturbedField`s, optional
        If given, must be compatible with init_box. It will merely negate the necessity of re-calculating the
        perturb fields. It will also be used to set the redshift if given.
    use_interp_perturb_field : bool, optional
        Whether to use a single perturb field, at the lowest redshift of the lightcone, to determine all spin
        temperature fields. If so, this field is interpolated in the underlying C-code to the correct redshift.
        This is less accurate (and no more efficient), but provides compatibility with older versions of 21cmMC.

    Returns
    -------
    lightcone: :class:`~LightCone`
        The lightcone object.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed
        See docs of :func:`initial_conditions` for more information.
    """

    random_seed, user_params, cosmo_params = configure_inputs(
        [("random_seed", random_seed), ("user_params", user_params), ("cosmo_params", cosmo_params)],
        init_box, perturb
    )

    user_params = UserParams(user_params)
    cosmo_params = CosmoParams(cosmo_params)
    flag_options = FlagOptions(flag_options)
    astro_params = AstroParams(astro_params, INHOMO_RECO=flag_options.INHOMO_RECO)

    if z_heat_max:
        global_params.Z_HEAT_MAX = z_heat_max
    if z_step_factor is not None:
        global_params.ZPRIME_STEP_FACTOR = z_step_factor

    if init_box is None:  # no need to get cosmo, user params out of it.
        init_box = initial_conditions(user_params=user_params, cosmo_params=cosmo_params, write=write,
                                      regenerate=regenerate, direc=direc, random_seed=random_seed)

    redshift = configure_redshift(redshift, perturb)

    if perturb is None:
        # The perturb field that we get here is at the *final* redshift, and can be used in TsBox.
        perturb = perturb_field(redshift=redshift, init_boxes=init_box, regenerate=regenerate, direc=direc)

    max_redshift = global_params.Z_HEAT_MAX if (
            flag_options.INHOMO_RECO or flag_options.USE_TS_FLUCT or max_redshift is None) else max_redshift

    # Get the redshift through which we scroll and evaluate the ionization field.
    scrollz = _logscroll_redshifts(redshift, global_params.ZPRIME_STEP_FACTOR, max_redshift)

    d_at_redshift, lc_distances, n_lightcone = _setup_lightcone(cosmo_params, max_redshift, redshift, scrollz,
                                                                user_params, global_params.ZPRIME_STEP_FACTOR)

    lc = np.zeros((user_params.HII_DIM, user_params.HII_DIM, n_lightcone), dtype=np.float32)

    scroll_distances = cosmo_params.cosmo.comoving_distance(scrollz).value - d_at_redshift

    # Iterate through redshift from top to bottom
    st, ib, bt = None, None, None
    lc_index = 0
    box_index = 0
    neutral_fraction = np.zeros(len(scrollz))
    global_signal = np.zeros(len(scrollz))

    for iz, z in enumerate(scrollz):
        # Best to get a perturb for this redshift, to pass to brightness_temperature

        this_perturb = perturb_field(redshift=z, init_boxes=init_box, regenerate=regenerate,
                                     direc=direc, write=write)

        if flag_options.USE_TS_FLUCT:
            st2 = spin_temperature(
                redshift=z,
                previous_spin_temp=st,
                astro_params=astro_params, flag_options=flag_options,
                perturbed_field=perturb if use_interp_perturb_field else this_perturb,
                regenerate=regenerate,
                init_boxes=init_box,
                z_heat_max=global_params.Z_HEAT_MAX, z_step_factor=z_step_factor,
                write=write, direc=direc
            )

        ib2 = ionize_box(
            redshift=z, previous_ionize_box=ib,
            init_boxes=init_box,
            perturbed_field=this_perturb,
            astro_params=astro_params, flag_options=flag_options,
            spin_temp=st2 if flag_options.USE_TS_FLUCT else None,
            regenerate=regenerate,
            z_heat_max=global_params.Z_HEAT_MAX, z_step_factor=z_step_factor,
            write=write, direc=direc
        )

        bt2 = brightness_temperature(ionized_box=ib2, perturbed_field=this_perturb,
                                     spin_temp=st2 if flag_options.USE_TS_FLUCT else None)

        # Save mean/global quantities
        neutral_fraction[iz] = np.mean(ib2.xH_box)
        global_signal[iz] = np.mean(bt2.brightness_temp)

        # HERE IS WHERE WE NEED TO DO THE INTERPOLATION ONTO THE LIGHTCONE!
        if z < max_redshift:  # i.e. now redshift is in the bit where the user wants to save the lightcone:
            # Do linear interpolation only.
            prev_d = scroll_distances[iz - 1]
            this_d = scroll_distances[iz]

            # Get the cells that need to be filled on this iteration.
            these_distances = lc_distances[np.logical_and(lc_distances < prev_d, lc_distances >= this_d)]

            n = len(these_distances)
            ind = np.arange(-(box_index + n), -box_index)

            lc[:, :, -(lc_index + n):n_lightcone - lc_index] = (np.abs(
                this_d - these_distances) * bt.brightness_temp.take(ind + n_lightcone, axis=2, mode='wrap') +
                                                                np.abs(
                                                                    prev_d - these_distances) * bt2.brightness_temp.take(
                        ind + n_lightcone, axis=2, mode='wrap')) / \
                                                               (np.abs(prev_d - this_d))

            lc_index += n
            box_index += n

        # Save current ones as old ones.
        if flag_options.USE_TS_FLUCT: st = st2
        ib = ib2
        bt = bt2

    return LightCone(
        redshift, user_params, cosmo_params, astro_params, flag_options, lc,
        node_redshifts=scrollz, global_xHI=neutral_fraction, global_brightness_temp=global_signal
    )


def _setup_lightcone(cosmo_params, max_redshift, redshift, scrollz, user_params, z_step_factor):
    # Here set up the lightcone box.
    # Get a length of the lightcone (bigger than it needs to be at first).
    d_at_redshift = cosmo_params.cosmo.comoving_distance(redshift).value
    Ltotal = cosmo_params.cosmo.comoving_distance(scrollz[0] * z_step_factor).value - d_at_redshift
    lc_distances = np.arange(0, Ltotal, user_params.BOX_LEN / user_params.HII_DIM)

    # Use max_redshift to get the actual distances we require.
    Lmax = cosmo_params.cosmo.comoving_distance(max_redshift).value - d_at_redshift
    first_greater = np.argwhere(lc_distances > Lmax)[0][0]

    # Get *at least* as far as max_redshift
    lc_distances = lc_distances[:(first_greater + 1)]

    n_lightcone = len(lc_distances)
    return d_at_redshift, lc_distances, n_lightcone


def _get_lightcone_redshifts(cosmo_params, max_redshift, redshift, user_params, z_step_factor):
    scrollz = _logscroll_redshifts(redshift, z_step_factor, max_redshift)
    lc_distances = _setup_lightcone(cosmo_params, max_redshift, redshift, scrollz, user_params, z_step_factor)[1]
    lc_distances += cosmo_params.cosmo.comoving_distance(redshift).value

    return np.array([z_at_value(cosmo_params.cosmo.comoving_distance, d * units.Mpc) for d in lc_distances])
