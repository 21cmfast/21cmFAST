"""
Input parameter classes.

There are four input parameter/option classes, not all of which are required for any
given function. They are :class:`SimulationOptions`, :class:`CosmoParams`, :class:`AstroParams`
and :class:`AstroOptions`. Each of them defines a number of variables, and all of these
have default values, to minimize the burden on the user. These defaults are accessed via
the ``_defaults_`` class attribute of each class. The available parameters for each are
listed in the documentation for each class below.
"""

# we use a few nested if statments in the validators here

import logging
import warnings
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, get_args

import attrs
import numpy as np
from astropy import constants
from astropy import units as un
from astropy.cosmology import FLRW, Planck15
from attrs import asdict, evolve, validators
from attrs import field as _field
from cyclopts import Parameter

from .._cfg import config
from ..c_21cmfast import ffi
from ._utils import snake_to_camel
from .classy_interface import (
    find_redshift_kinematic_decoupling,
    get_transfer_function,
    k_transfer,
    run_classy,
)
from .structs import StructWrapper

logger = logging.getLogger(__name__)


def field(*, transformer=None, **kw):
    """Define an attrs field with a 'transformer' property.

    The transformer, if given, should be a function of a single variable, which will
    be the attribute's value. It will be used to transform the value before usage in
    C-code (e.g. by transformin from log to linear space).
    """
    return _field(metadata={"transformer": transformer}, **kw)


def choice_field(*, validator=None, **kwargs):
    """Create an attrs.field that is a choice."""
    vld = (choice_validator,)
    if validator is not None:
        vld = (choice_validator, validator)
    return field(validator=vld, transformer=choice_transformer, converter=str, **kwargs)


def logtransformer(x, att: attrs.Attribute):
    """Convert from log to linear space."""
    return 10**x


def dex2exp_transformer(x, att: attrs.Attribute):
    """Convert from dex to exponential space."""
    return 2.3025851 * x


def choice_transformer(choice: str, att: attrs.Attribute) -> int:
    """Produce a transformer that converts a string to int.

    The function must be passed a list of string choices. The resulting int is the
    index of the choice made.
    """
    choices = get_args(att.type)
    return choices.index(choice)


def choice_validator(inst, att: attrs.Attribute, val):
    """Validate that a value is one of the choices."""
    choices = get_args(att.type)
    if val not in choices:
        raise ValueError(f"{att.name} must be one of {choices}, got {val} instead.")


def between(mn, mx):
    """Validate that a value is between two values."""

    def vld(inst, att, val):
        if val < mn or val > mx:
            raise ValueError(f"{att.name} must be between {mn} and {mx}")

    return vld


FilterOptions = Literal["spherical-tophat", "sharp-k", "gaussian"]
IntegralMethods = Literal["GSL-QAG", "GAUSS-LEGENDRE", "GAMMA-APPROX"]

# Cosmology is from https://arxiv.org/pdf/1807.06209.pdf
# Table 2, last column. [TT,TE,EE+lowE+lensing+BAO]
Planck18 = Planck15.clone(
    Om0=(0.02242 + 0.11933) / 0.6766**2,
    Ob0=0.02242 / 0.6766**2,
    H0=67.66,
    Neff=3.044,
    name="Planck18",
)


@attrs.define(frozen=True, kw_only=True)
class InputStruct:
    """
    A convenient interface to create a C structure with defaults specified.

    It is provided for the purpose of *creating* C structures in Python to be passed to
    C functions, where sensible defaults are available. Structures which are created
    within C and passed back do not need to be wrapped.

    This provides a *fully initialised* structure, and will fail if not all fields are
    specified with defaults.

    .. note:: The actual C structure is gotten by calling an instance. This is
              auto-generated when called, based on the parameters in the class.

    .. warning:: This class will *not* deal well with parameters of the struct which are
                 pointers. All parameters should be primitive types, except for strings,
                 which are dealt with specially.

    Parameters
    ----------
    ffi : cffi object
        The ffi object from any cffi-wrapped library.
    """

    _subclasses: ClassVar = {}
    _write_exclude_fields = ()

    @classmethod
    def new(cls, x: dict | Self | None = None, **kwargs):
        """
        Create a new instance of the struct.

        Parameters
        ----------
        x : dict | InputStruct | None
            Initial values for the struct. If `x` is a dictionary, it should map field
            names to their corresponding values. If `x` is an instance of this class,
            its attributes will be used as initial values. If `x` is None, the
            struct will be initialised with default values.

        Other Parameters
        ----------------
        All other parameters should be passed as if directly to the class constructor
        (i.e. as parameter names).

        Examples
        --------
        >>> up = SimulationOptions({'HII_DIM': 250})
        >>> up.HII_DIM
        250
        >>> up = SimulationOptions(up)
        >>> up.HII_DIM
        250
        >>> up = SimulationOptions()
        >>> up.HII_DIM
        200
        >>> up = SimulationOptions(HII_DIM=256)
        >>> up.HII_DIM
        256
        """
        if isinstance(x, dict):
            return cls(**x, **kwargs)
        elif isinstance(x, InputStruct):
            return x.clone(**kwargs)
        elif x is None:
            return cls(**kwargs)
        else:
            raise ValueError(
                f"Cannot instantiate {cls.__name__} with type {x.__class__}"
            )

    def __init_subclass__(cls) -> None:
        """Store each subclass for easy access."""
        InputStruct._subclasses[cls.__name__] = cls

    @cached_property
    def struct(self) -> StructWrapper:
        """The python-wrapped struct associated with this input object."""
        return StructWrapper(self.__class__.__name__)

    @cached_property
    def cstruct(self) -> StructWrapper:
        """The object pointing to the memory accessed by C-code for this struct."""
        cdict = self.cdict
        for k in self.struct.fieldnames:
            val = cdict[k]

            # TODO: is this really required here? (I don't think the wrapper can satisfy this condition)
            if isinstance(val, str):
                # If it is a string, need to convert it to C string ourselves.
                val = self.ffi.new("char[]", val.encode())

            setattr(self.struct.cstruct, k, val)

        return self.struct.cstruct

    def clone(self, **kwargs):
        """Make a fresh copy of the instance with arbitrary parameters updated."""
        return evolve(self, **kwargs)

    def asdict(self) -> dict:
        """Return a dict representation of the instance.

        Examples
        --------
        This dict should be such that doing the following should work, i.e. it can be
        used exactly to construct a new instance of the same object::

        >>> inp = InputStruct(**params)
        >>> newinp =InputStruct(**inp.asdict())
        >>> inp == newinp
        """
        return asdict(self)

    @cached_property
    def cdict(self) -> dict:
        """A python dictionary containing the properties of the wrapped C-struct.

        The memory pointed to by this dictionary is *not* owned by the wrapped C-struct,
        but is rather just a python dict. However, in contrast to :meth:`asdict`, this
        method transforms the properties to what they should be in C (e.g. linear space
        vs. log-space) before putting them into the dict.

        This dict also contains *only* the properties of the wrapped C-struct, rather
        than all properties of the :class:`InputStruct` instance (some attributes of the
        python instance are there only to guide setting of defaults, and don't appear
        in the C-struct at all).
        """
        fields = attrs.fields(self.__class__)
        transformers = {
            field.name: field.metadata.get("transformer", None) for field in fields
        }

        out = {}
        for k in self.struct.fieldnames:
            val = getattr(self, k)
            att = attrs.fields_dict(self.__class__).get(k, None)
            # we assume properties (as opposed to attributes) are already converted
            trns = transformers.get(k)
            out[k] = val if trns is None else trns(val, att)
        return out

    def __str__(self) -> str:
        """Human-readable string representation of the object."""
        d = self.asdict()
        biggest_k = max(len(k) for k in d)
        params = "\n    ".join(sorted(f"{k:<{biggest_k}}: {v}" for k, v in d.items()))
        return f"""{self.__class__.__name__}:{params} """

    @classmethod
    def from_subdict(cls, dct, safe=True):
        """Construct an instance of a parameter structure from a dictionary."""
        fieldnames = [
            field.name
            for field in attrs.fields(cls)
            if field.eq  # and field.default is not None
        ]
        if set(fieldnames) != set(dct.keys()):
            missing_items = [
                (field.name, field.default)
                for field in attrs.fields(cls)
                if field.name not in dct and field.name in fieldnames
            ]
            extra_items = [(k, v) for k, v in dct.items() if k not in fieldnames]
            message = (
                f"There are extra or missing {cls.__name__} in the file to be read.\n"
                f"EXTRAS: {extra_items}\n"
                f"MISSING: {missing_items}\n"
            )
            if safe:
                raise ValueError(
                    message
                    + "set `safe=False` to load structures from previous versions"
                )
            else:
                warnings.warn(
                    message
                    + "\nExtras are ignored and missing are set to default (shown) values."
                    + "\nUsing these parameter structures in further computation will give inconsistent results.",
                    stacklevel=2,
                )
            dct = {k: v for k, v in dct.items() if k in fieldnames}

        # Strip leading underscores from items, because attrs accepts non-underscore
        # versions of attributes.
        dct = {k.strip("_"): v for k, v in dct.items()}
        return cls.new(dct)


@attrs.define(frozen=True, kw_only=True)
class Table1D:
    """Class for setting 1D interpolation table."""

    size: int = field(converter=int, validator=validators.gt(0))
    x_values: np.ndarray = field(
        converter=lambda v: np.asarray(v, dtype=np.float64),
        validator=validators.instance_of(np.ndarray),
        eq=attrs.cmp_using(eq=np.array_equal),
    )
    y_values: np.ndarray = field(
        converter=lambda v: np.asarray(v, dtype=np.float64),
        validator=validators.instance_of(np.ndarray),
        eq=attrs.cmp_using(eq=np.array_equal),
    )

    @cached_property
    def cstruct(self):
        """Cached pointer to the memory of the object in C."""
        ctab = ffi.new("Table1D *")
        ctab.size = self.size
        ctab.x_values = ffi.cast("double *", ffi.from_buffer(self.x_values))
        ctab.y_values = ffi.cast("double *", ffi.from_buffer(self.y_values))
        return ctab


@attrs.define(frozen=True, kw_only=True)
class CosmoTables:
    """Class for storing interpolation tables of cosmological functions (e.g. transfer functions, growth factor)."""

    transfer_density: Table1D = field(default=None)
    transfer_vcb: Table1D = field(default=None)
    ps_norm: float = field(default=None)
    USE_SIGMA_8: bool = field(default=None)

    @classmethod
    def new(cls, x: dict | Self | None = None, **kwargs):
        """
        Create a new instance of the struct.

        Parameters
        ----------
        x : dict | CosmoTables | None
            Initial values for the struct. If `x` is a dictionary, it should map field
            names to their corresponding values. If `x` is an instance of this class,
            its attributes will be used as initial values. If `x` is None, the
            struct will be initialised with default values.

        Other Parameters
        ----------------
        All other parameters should be passed as if directly to the class constructor
        (i.e. as parameter names).
        """
        if isinstance(x, dict):
            return cls(**x, **kwargs)
        elif isinstance(x, CosmoTables):
            return x.clone(**kwargs)
        elif x is None:
            return cls(**kwargs)
        else:
            raise ValueError(
                f"Cannot instantiate {cls.__name__} with type {x.__class__}"
            )

    @cached_property
    def struct(self):
        """The python-wrapped struct associated with this input object."""
        return StructWrapper("CosmoTables")

    @cached_property
    def cstruct(self) -> StructWrapper:
        """The object pointing to the memory accessed by C-code for this struct."""
        for k in self.struct.fieldnames:
            val = getattr(self, k)
            if isinstance(val, Table1D):
                setattr(self.struct.cstruct, k, val.cstruct)
            elif isinstance(val, (float | bool)):
                setattr(self.struct.cstruct, k, val)

        return self.struct.cstruct

    def clone(self, **kwargs):
        """Make a fresh copy of the instance with arbitrary parameters updated."""
        return evolve(self, **kwargs)


@attrs.define(frozen=True, kw_only=True)
class CosmoParams(InputStruct):
    """
    Cosmological parameters (with defaults) which translates to a C struct.

    To see default values for each parameter, use ``CosmoParams._defaults_``.
    All parameters passed in the constructor are also saved as instance attributes which should
    be considered read-only. This is true of all input-parameter classes.

    Default parameters are based on Plank18, https://arxiv.org/pdf/1807.06209.pdf,
    Table 2, last column. [TT,TE,EE+lowE+lensing+BAO]

    Parameters
    ----------
    SIGMA_8 : float, optional
        RMS mass variance (power spectrum normalisation).
    hlittle : float, optional
        The hubble parameter, H_0/100.
    OMm : float, optional
        Omega matter.
    OMb : float, optional
        Omega baryon, the baryon component.
    POWER_INDEX : float, optional
        Spectral index of the power spectrum.
    A_s: float, optional
        Amplitude of primordial curvature power spectrum, at k_pivot = 0.05 Mpc^-1.
    """

    _DEFAULT_SIGMA_8: ClassVar[float] = 0.8102
    _DEFAULT_A_s: ClassVar[float] = 2.105e-9

    _base_cosmo: Annotated[FLRW, Parameter(show=False, parse=False)] = field(
        default=Planck18, validator=validators.instance_of(FLRW), eq=False, repr=False
    )
    _SIGMA_8: float = field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=validators.optional(validators.gt(0)),
    )
    hlittle: float = field(
        default=Planck18.h, converter=float, validator=validators.gt(0)
    )
    OMm: float = field(
        default=Planck18.Om0, converter=float, validator=validators.gt(0)
    )
    OMb: float = field(
        default=Planck18.Ob0, converter=float, validator=validators.gt(0)
    )
    POWER_INDEX: float = field(
        default=0.9665, converter=float, validator=validators.gt(0)
    )
    _A_s: float = field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=validators.optional(validators.gt(0)),
    )

    OMn: float = field(default=0.0, converter=float, validator=validators.ge(0))
    OMk: float = field(default=0.0, converter=float, validator=validators.ge(0))
    OMr: float = field(default=8.6e-5, converter=float, validator=validators.ge(0))
    OMtot: float = field(
        default=1.0, converter=float, validator=validators.ge(0)
    )  # TODO: force this to be the sum of the others
    Y_He: float = field(default=0.24, converter=float, validator=validators.ge(0))
    wl: float = field(default=-1.0, converter=float)

    # TODO: Combined validation via Astropy?

    @_SIGMA_8.validator
    def _sigma_8_vld(self, att, val):
        if self._A_s is not None and val is not None:
            raise ValueError(
                "Cannot set both SIGMA_8 and A_s! "
                "If this error arose when loading from template/file or evolving an "
                "existing object, then explicitly set either SIGMA_8 or A_s "
                "to None while setting the other to the desired value."
            )

    @cached_property
    def SIGMA_8(self) -> float:
        """RMS mass variance (power spectrum normalisation).

        If not given explicitly, it is auto-calculated via A_s
        and the other cosmological parameters.
        """
        if self._SIGMA_8 is not None:
            return self._SIGMA_8
        elif self._A_s is not None:
            classy_output = run_classy(
                h=self.hlittle,
                Omega_cdm=self.OMm - self.OMb,
                Omega_b=self.OMb,
                A_s=self._A_s,
                n_s=self.POWER_INDEX,
                output="mPk",
                level="fourier",
            )
            return classy_output.sigma8()
        else:
            return self._DEFAULT_SIGMA_8

    @cached_property
    def A_s(self) -> float:
        """Amplitude of primordial curvature power spectrum, at k_pivot = 0.05 Mpc^-1.

        If not given explicitly, it is auto-calculated via sigma_8
        and the other cosmological parameters.
        """
        if self._A_s is not None:
            return self._A_s
        elif self._SIGMA_8 is not None:
            classy_output = run_classy(
                h=self.hlittle,
                Omega_cdm=self.OMm - self.OMb,
                Omega_b=self.OMb,
                sigma8=self.SIGMA_8,
                n_s=self.POWER_INDEX,
                output="mTk",
                level="thermodynamics",
            )
            return classy_output.get_current_derived_parameters(["A_s"])["A_s"]
        else:
            return self._DEFAULT_A_s

    @property
    def OMl(self):
        """Omega lambda, dark energy density."""
        return 1 - self.OMm

    @cached_property
    def cosmo(self):
        """An astropy cosmology object for this cosmology."""
        return self._base_cosmo.clone(
            name=self._base_cosmo.name,
            H0=self.hlittle * 100,
            Om0=self.OMm,
            Ob0=self.OMb,
        )

    @classmethod
    def from_astropy(cls, cosmo: FLRW, **kwargs):
        """Create a CosmoParams object from an astropy cosmology object.

        Pass SIGMA_8 and POWER_INDEX as kwargs if you want to override the default
        values.
        """
        return cls(
            hlittle=cosmo.h, OMm=cosmo.Om0, OMb=cosmo.Ob0, base_cosmo=cosmo, **kwargs
        )

    def asdict(self) -> dict:
        """Return a dict representation of the instance.

        Examples
        --------
        This dict is such that doing the following should work, i.e. it can be
        used exactly to construct a new instance of the same object::

        >>> inp = InputStruct(**params)
        >>> newinp =InputStruct(**inp.asdict())
        >>> inp == newinp
        """
        d = super().asdict()
        del d["_base_cosmo"]
        return d


@attrs.define(frozen=True, kw_only=True)
class MatterOptions(InputStruct):
    """
    Structure containing options which affect the matter field (ICs, perturbedfield, halos).

    Parameters
    ----------
    HMF: str, optional
        Determines which halo mass function to be used for the normalisation of the
        collapsed fraction (default Sheth-Tormen). Should be one of the
        following codes:
        PS (Press-Schechter)
        ST (Sheth-Tormen)
        Watson (Watson FOF)
        Watson-z (Watson FOF-z)
        Delos (Delos+23)
    USE_RELATIVE_VELOCITIES: int, optional
        Flag to decide whether to use relative velocities.
        If True, POWER_SPECTRUM is automatically set to 5. Default False.
    POWER_SPECTRUM: str, optional
        Determines which power spectrum to use, default EH (unless `USE_RELATIVE_VELOCITIES`
        is True). Use the following codes:
        EH : Eisenstein & Hu 1999
        BBKS: Bardeen et al. 1986
        EFSTATHIOU: Efstathiou et al. 1992
        PEEBLES: Peebles 1980
        WHITE: White 1985
        CLASS: Uses fits from the CLASS code
    PERTURB_ON_HIGH_RES : bool, optional
        Whether to perform the Zel'Dovich or 2LPT perturbation on the low or high
        resolution grid.
    USE_FFTW_WISDOM : bool, optional
        Whether or not to use stored FFTW_WISDOMs for improving performance of FFTs
    USE_INTERPOLATION_TABLES: str, optional
        Defines the interpolation tables used in the code. Default is 'hmf-interpolation'.
        There are three levels available:
        'no-interpolation': No interpolation tables are used.
        'sigma-interpolation': Interpolation tables are used for sigma(M) only.
        'hmf-interpolation': Interpolation tables are used for sigma(M) the halo mass function.
    PERTURB_ALGORITHM: str, optional
        Whether to use second-order Lagrangian perturbation theory (2LPT), Zel'dovich (ZELDOVICH),
        or linear evolution (LINEAR).
        Set this to 2LPT if the density field or the halo positions are extrapolated to
        low redshifts. The current implementation is very naive and adds a factor ~6 to
        the memory requirements. Reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118
        Appendix D.
    MINIMIZE_MEMORY: bool, optional
        If set, the code will run in a mode that minimizes memory usage, at the expense
        of some CPU/disk-IO. Good for large boxes / small computers.
    SAMPLE_METHOD: str, optional
        The sampling method to use in the halo sampler when calculating progenitor populations:
        MASS-LIMITED : Mass-limited CMF sampling, where samples are drawn until the expected mass is reached
        NUMBER-LIMITED : Number-limited CMF sampling, where we select a number of halos from the Poisson distribution
        and then sample the CMF that many times
        PARTITION : Sheth et al 1999 Partition sampling, where the EPS collapsed fraction is sampled (gaussian tail)
        and then the condition is updated using the conservation of mass.
        BINARY-SPLIT : Parkinsson et al 2008 Binary split model as in DarkForest (Qiu et al 2021) where the EPS merger rate
        is sampled on small internal timesteps such that only binary splits can occur.
        NOTE: The initial sampling from the density grid will ALWAYS use number-limited sampling (method 1)
    FILTER : string, optional
        Filter to use for sigma (matter field variance) and radius to mass conversions.
        available options are: `spherical-tophat` and `gaussian`
    HALO_FILTER : string, optional
        Filter to use for the DexM halo finder.
        available options are: `spherical-tophat`, `sharp-k` and `gaussian`
    SMOOTH_EVOLVED_DENSITY_FIELD: bool, optional
        Smooth the evolved density field after perturbation.
    DEXM_OPTMIZE: bool, optional
        Use a faster version of the DexM halo finder which excludes halos from forming within a certain distance of larger halos.
    KEEP_3D_VELOCITIES: bool, optional
        Whether to keep the 3D velocities in the ICs.
        If False, only the z velocity is kept.
    SOURCE_MODEL: str, optional
        The source model to use in the simulation. Options are:
        E-INTEGRAL : The traditional excursion-set formalism, where source properties are
            defined on the Eulerian grid after 2LPT in regions of filter scale R (see the X_FILTER options for filter shapes).
            This integrates over the CHMF using the smoothed density grids, then multiplies the result.
            by (1 + delta) to get the source properties in each cell.
        CONST-ION-EFF: Similar to E-INTEGRAL, but ionizing efficiency is constant and does not depend on the halo mass
            (see Mesinger+ 2010).
        L-INTEGRAL : Analagous to the 'ESF-L' model described in Trac+22, where source properties
            are defined on the Lagrangian (IC) grid by integrating the CHMF prior to the IGM physics
            and then mapping properties to the Eulerian grid using 2LPT.
        DEXM-ESF : The DexM excursion-set formalism, where discrete halo catalogues are generated
            on the Lagrangian (IC) grid using an excursion-set halo finder. Source properties
            are defined on the Lagrangian grid and then mapped to the Eulerian grid using 2LPT.
            This model utilised the 'L-INTEGRAL' method for halos below the DexM mass resolution,
            which is the mass of the high-resolution (DIM^3) cells.
        CHMF-SAMPLER : The CHMF sampler, where discrete halo catalogues are generated by sampling
            the CHMF on the IC grid, between the low-resolution (HII_DIM^3) cell mass and a minimum
            mass defined by the user (SAMPLER_MIN_MASS). This model uses the 'L-INTEGRAL' method for
            halos below the SAMPLER_MIN_MASS, and the 'DEXM-ESF' method for halos above the HII_DIM
            cell mass.
    """

    HMF: Literal["PS", "ST", "WATSON", "WATSON-Z", "DELOS"] = choice_field(default="ST")
    USE_RELATIVE_VELOCITIES: bool = field(default=False, converter=bool)
    POWER_SPECTRUM: Literal["EH", "BBKS", "EFSTATHIOU", "PEEBLES", "WHITE", "CLASS"] = (
        choice_field()
    )
    PERTURB_ON_HIGH_RES: bool = field(default=False, converter=bool)
    USE_INTERPOLATION_TABLES: Literal[
        "no-interpolation", "sigma-interpolation", "hmf-interpolation"
    ] = choice_field(default="hmf-interpolation")
    MINIMIZE_MEMORY: bool = field(default=False, converter=bool)
    KEEP_3D_VELOCITIES: bool = field(default=False, converter=bool)
    SAMPLE_METHOD: Literal[
        "MASS-LIMITED", "NUMBER-LIMITED", "PARTITION", "BINARY-SPLIT"
    ] = choice_field(
        default="MASS-LIMITED",
    )
    FILTER: FilterOptions = choice_field(
        default="spherical-tophat",
        validator=validators.not_(validators.in_(["sharp-k"])),
    )
    HALO_FILTER: FilterOptions = choice_field(default="spherical-tophat")
    SMOOTH_EVOLVED_DENSITY_FIELD: bool = field(default=False, converter=bool)
    DEXM_OPTIMIZE: bool = field(default=False, converter=bool)
    PERTURB_ALGORITHM: Literal["LINEAR", "ZELDOVICH", "2LPT"] = choice_field(
        default="2LPT",
    )
    USE_FFTW_WISDOM: bool = field(default=False, converter=bool)

    SOURCE_MODEL: Literal[
        "CONST-ION-EFF", "E-INTEGRAL", "L-INTEGRAL", "DEXM-ESF", "CHMF-SAMPLER"
    ] = choice_field(default="CHMF-SAMPLER")

    @POWER_SPECTRUM.default
    def _ps_default(self):
        return "CLASS" if self.USE_RELATIVE_VELOCITIES else "EH"

    @POWER_SPECTRUM.validator
    def _POWER_SPECTRUM_vld(self, att, val):
        if self.USE_RELATIVE_VELOCITIES and val != "CLASS":
            raise ValueError(
                "Can only use 'CLASS' power spectrum with relative velocities"
            )

    @SOURCE_MODEL.validator
    def _SOURCE_MODEL_vld(self, att, val):
        """Validate SOURCE_MODEL dependencies."""
        if val in ["DEXM-ESF", "CHMF-SAMPLER"] and self.HMF not in ["ST", "PS"]:
            msg = (
                "The conditional mass functions requied for the discrete halo field are only currently"
                "available for the Sheth-Tormen and Press-Schechter mass functions., use HMF='ST' or 'PS'"
            )
            raise NotImplementedError(msg)
        if (
            val in ["DEXM-ESF", "CHMF-SAMPLER"]
            and self.USE_INTERPOLATION_TABLES != "hmf-interpolation"
        ):
            msg = (
                "SOURCE_MODEL settings using the halo sampler require the use of HMF interpolation tables."
                "Switch USE_INTERPOLATION_TABLES to 'hmf-interpolation'"
            )
            raise ValueError(msg)

    @property
    def has_discrete_halos(self) -> bool:
        """Whether the current options will produce discrete halo catalogues."""
        return self.SOURCE_MODEL in ["DEXM-ESF", "CHMF-SAMPLER"]

    @property
    def lagrangian_source_grid(self) -> bool:
        """Whether the current source model is Lagrangian."""
        return self.SOURCE_MODEL in ["L-INTEGRAL", "DEXM-ESF", "CHMF-SAMPLER"]

    @property
    def mass_dependent_zeta(self) -> bool:
        """Whether the current source model uses mass-dependent zeta."""
        return self.SOURCE_MODEL in [
            "E-INTEGRAL",
            "L-INTEGRAL",
            "DEXM-ESF",
            "CHMF-SAMPLER",
        ]


@attrs.define(frozen=True, kw_only=True)
class SimulationOptions(InputStruct):
    """
    Structure containing broad simulation options.

    Parameters
    ----------
    HII_DIM : int, optional
        Number of cells for the low-res box (after smoothing the high-resolution matter
        field). Default 256.
    HIRES_TO_LOWRES_FACTOR : float, optional
        The ratio of the high-resolution box dimensionality to the low-resolution
        box dimensionality (i.e. DIM/HII_DIM). Use this parameter to define the size
        as a fixed ratio, instead of specifying DIM explicitly. This is useful if
        the parameters will be later evolved, so that specifying a new HII_DIM keeps
        the fixed resolution. By default, this is None, and a default of DIM=3*HII_DIM
        is used. This should be at least 3, and generally an integer (though this is not
        enforced).
    DIM : int, optional
        Number of cells for the high-res box (sampling ICs) along a principal axis.
        In general, prefer setting HIRES_TO_LOWRES_FACTOR instead of DIM directly.
        Setting both will raise an error.
    LOWRES_CELL_SIZE_MPC : float, optional
        The cell size of the low-resolution boxes (i.e. after smoothing the high-resolution
        matter field). Use either this parameter or BOX_LEN, setting both will raise
        an error. This parameter is generally preferrable, as it allows you to evolve
        the HII_DIM later, and keep the same resolution (automatically scaling up
        BOX_LEN). Default is None, falling back on a cell size of 1.5 Mpc.
    BOX_LEN : float, optional
        Length of the box, in Mpc. Prefer setting LOWRES_CELL_SIZE_MPC, which automatically
        defines this setting. Specifying both will result in an error. By default,
        the BOX_LEN will be calculated as 1.5 * HII_DIM.
    NON_CUBIC_FACTOR : float, optional
        Factor which allows the creation of non-cubic boxes. It will shorten/lengthen the line
        of sight dimension of all boxes. NON_CUBIC_FACTOR * DIM/HII_DIM must result in an integer.
    N_THREADS : int, optional
        Sets the number of processors (threads) to be used for performing 21cmFAST.
        Default 1.
    SAMPLER_MIN_MASS: float, optional
        The minimum mass to sample in the halo sampler when SOURCE_MODEL is "CHMF-SAMPLER",
        decreasing this can drastically increase both compute time and memory usage.
    SAMPLER_BUFFER_FACTOR: float, optional
        The arrays for the halo sampler will have size of SAMPLER_BUFFER_FACTOR multiplied by the expected
        number of halos in the box. Ideally this should be close to unity but one may wish to increase it to
        test alternative scenarios
    N_COND_INTERP: int, optional
        The number of condition bins in the inverse CMF tables.
    N_PROB_INTERP: int, optional
        The number of probability bins in the inverse CMF tables.
    MIN_LOGPROB: float, optional
        The minimum log-probability of the inverse CMF tables.
    HALOMASS_CORRECTION: float, optional
        This provides a corrective factor to the mass-limited (SAMPLE_METHOD==0) sampling, which multiplies the
        expected mass from a condition by this number. The default value of 0.9 is calibrated to the mass-limited
        sampling on a timestep of ZPRIME_STEP_FACTOR=1.02.
        If ZPRIME_STEP_FACTOR is increased, this value should be set closer to 1.
        This factor is also used in the partition (SAMPLE_METHOD==2) sampler, dividing nu(M) of each sample drawn.
    PARKINSON_G0: float, optional
        Only used when SAMPLE_METHOD==3, sets the normalisation of the correction to the extended press-schecter
        used in Parkinson et al. 2008.
    PARKINSON_y1: float, optional
        Only used when SAMPLE_METHOD==3, sets the index of the sigma power-law term of the correction to the
        extended Press-Schechter mass function used in Parkinson et al. 2008.
    PARKINSON_y2: float, optional
        Only used when SAMPLE_METHOD==3, sets the index of the delta power-law term of the correction to the
        extended Press-Schechter mass function used in Parkinson et al. 2008.
    Z_HEAT_MAX : float, optional
        Maximum redshift used in the Tk and x_e evolution equations.
        Temperature and x_e are assumed to be homogeneous at higher redshifts.
        Lower values will increase performance.
    ZPRIME_STEP_FACTOR : float, optional
        Logarithmic redshift step-size used in the z' integral.  Logarithmic dz.
        Decreasing (closer to unity) increases total simulation time for lightcones,
        and for Ts calculations.
    INITIAL_REDSHIFT : float, optional
        Initial redshift used to perturb field from
    DELTA_R_FACTOR: float, optional
        The factor by which to decrease the size of the filter in DexM when creating halo catalogues.
    DENSITY_SMOOTH_RADIUS: float, optional
        The radius of the smoothing kernel in Mpc.
    DEXM_OPTIMIZE_MINMASS: float, optional
        The minimum mass of a halo for which to use the DexM optimization if DEXM_OPTIMIZE is True.
    DEXM_R_OVERLAP: float, optional
        The factor by which to multiply the halo radius to determine the distance within which smaller halos are excluded.
    CORR_STAR : float, optional
        Self-correlation length used for updating halo properties. To model the
        correlation in the SHMR between timesteps, we sample from a conditional bivariate gaussian
        with correlation factor given by exp(-dz/CORR_STAR). This value is placed in SimulationOptions
        since it is used in the halo sampler, and not in the ionization routines.
    CORR_SFR : float, optional
        Self-correlation length used for updating star formation rate, see "CORR_STAR" for details.
    CORR_LX : float, optional
        Self-correlation length used for updating xray luminosity, see "CORR_STAR" for details.
    K_MAX_FOR_CLASS: float, optional
        Maximum wavenumber to run CLASS, in 1/Mpc. Becomes relevant only if matter_options.POWER_SPECTRUM = "CLASS".
    MIN_XE_FOR_FCOLL_IN_TAUX: float, optional
        Minimum global x_e value for which the collapsed fraction (f_coll) is evaluated in the tau_X integral (X-ray optical depth).
        When x_e is above this threshold value, it is assumed that f_coll=0, in order to speed up the calculations.
        For now, this parameter becomes relevant only when run_global_evolution is called, as it controls the runtime of this
        function (higher values reduce the runtime, in expense of degraded precision).
    """

    _DEFAULT_HIRES_TO_LOWRES_FACTOR: ClassVar[float] = 3
    _DEFAULT_LOWRES_CELL_SIZE_MPC: ClassVar[float] = 1.5

    HII_DIM: int = field(default=256, converter=int, validator=validators.gt(0))

    _BOX_LEN: float = field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=validators.optional(validators.gt(0)),
    )
    _DIM: int | None = field(default=None, converter=attrs.converters.optional(int))

    _HIRES_TO_LOWRES_FACTOR: float = field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(validators.gt(1)),
    )
    _LOWRES_CELL_SIZE_MPC: float = field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(validators.gt(0)),
    )

    NON_CUBIC_FACTOR: float = field(
        default=1.0, converter=float, validator=validators.gt(0)
    )
    N_THREADS: int = field(default=1, converter=int, validator=validators.gt(0))
    SAMPLER_MIN_MASS: float = field(
        default=1e8, converter=float, validator=validators.gt(0)
    )
    SAMPLER_BUFFER_FACTOR: float = field(default=2.0, converter=float)
    N_COND_INTERP: int = field(default=200, converter=int)
    N_PROB_INTERP: int = field(default=400, converter=int)
    MIN_LOGPROB: float = field(default=-12, converter=float)
    HALOMASS_CORRECTION: float = field(
        default=0.89, converter=float, validator=validators.gt(0)
    )
    PARKINSON_G0: float = field(
        default=1.0, converter=float, validator=validators.gt(0)
    )
    PARKINSON_y1: float = field(default=0.0, converter=float)
    PARKINSON_y2: float = field(default=0.0, converter=float)
    Z_HEAT_MAX: float = field(default=35.0, converter=float)
    ZPRIME_STEP_FACTOR: float = field(default=1.02, converter=float)
    MIN_XE_FOR_FCOLL_IN_TAUX: float = field(default=1e-3, converter=float)

    INITIAL_REDSHIFT: float = field(default=300.0, converter=float)
    DELTA_R_FACTOR: float = field(
        default=1.1, converter=float, validator=validators.gt(1.0)
    )

    DENSITY_SMOOTH_RADIUS: float = field(
        default=0.2, converter=float, validator=validators.gt(0)
    )
    DEXM_OPTIMIZE_MINMASS: float = field(
        default=1e11, converter=float, validator=validators.gt(0)
    )
    DEXM_R_OVERLAP: float = field(
        default=2, converter=float, validator=validators.gt(0)
    )

    # NOTE: Thematically these should be in AstroParams, However they affect the HaloCatalog
    #   Objects and so need to be in the matter_cosmo hash, they seem a little strange here
    #   but will remain until someone comes up with a better organisation down the line
    CORR_STAR: float = field(default=0.5, converter=float)
    CORR_SFR: float = field(default=0.2, converter=float)
    # NOTE (Jdavies): I do not currently have great priors for this value
    CORR_LX: float = field(default=0.2, converter=float)
    K_MAX_FOR_CLASS: float | None = field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(validators.gt(0)),
    )

    @property
    def DIM(self) -> int:
        """The number of cells on a side of the hi-res box used for ICs.

        If not given explicitly, it is auto-calculated via
        ``HII_DIM * HIRES_TO_LOWRES_FACTOR``.
        """
        if self._DIM is not None:
            return self._DIM
        else:
            return int(self.HII_DIM * self.HIRES_TO_LOWRES_FACTOR)

    @property
    def BOX_LEN(self) -> float:
        """The size of the box along a side, in Mpc.

        If not given explicitly, it is auto-calculated via
        ``HII_DIM * LOWRES_CELL_SIZE_MPC``.
        """
        if self._BOX_LEN is not None:
            return self._BOX_LEN
        else:
            return int(self.HII_DIM * self.LOWRES_CELL_SIZE_MPC)

    @_HIRES_TO_LOWRES_FACTOR.validator
    def _hires_to_lowres_vld(self, att, val):
        if self._DIM is not None and val is not None:
            raise ValueError(
                "Cannot set both DIM and HIRES_TO_LOWRES_FACTOR! "
                "If this error arose when loading from template/file or evolving an "
                "existing object, then explicitly set either DIM or HIRES_TO_LOWRES_FACTOR "
                "to None while setting the other to the desired value."
            )

    @_LOWRES_CELL_SIZE_MPC.validator
    def _lowres_cellsize_vld(self, att, val):
        if self._BOX_LEN is not None and val is not None:
            raise ValueError(
                "Cannot set both BOX_LEN and LOWRES_CELL_SIZE_MPC! "
                "If this error arose when loading from template/file or evolving an "
                "existing object, then explicitly set either BOX_LEN or "
                "LOWRES_CELL_SIZE_MPC to None while setting the other to the desired "
                "value."
            )

    @property
    def HIRES_TO_LOWRES_FACTOR(self) -> float:
        """The downsampling factor from high to low-res."""
        if self._DIM is not None:
            return self._DIM / self.HII_DIM
        elif self._HIRES_TO_LOWRES_FACTOR is not None:
            return self._HIRES_TO_LOWRES_FACTOR
        else:
            return self._DEFAULT_HIRES_TO_LOWRES_FACTOR

    @property
    def LOWRES_CELL_SIZE_MPC(self) -> float:
        """The cell size (in Mpc) of the low-res grids."""
        if self._BOX_LEN is not None:
            return self._BOX_LEN / self.HII_DIM
        elif self._LOWRES_CELL_SIZE_MPC is not None:
            return self._LOWRES_CELL_SIZE_MPC
        else:
            return self._DEFAULT_LOWRES_CELL_SIZE_MPC

    @NON_CUBIC_FACTOR.validator
    def _NON_CUBIC_FACTOR_validator(self, att, val):
        dcf = self.DIM * val
        hdcf = self.HII_DIM * val
        if dcf % int(dcf) or hdcf % int(hdcf):
            raise ValueError(
                "NON_CUBIC_FACTOR * DIM and NON_CUBIC_FACTOR * HII_DIM must be integers"
            )

    @property
    def tot_fft_num_pixels(self):
        """Total number of pixels in the high-res box."""
        return int(self.NON_CUBIC_FACTOR * self.DIM**3)

    @property
    def HII_tot_num_pixels(self):
        """Total number of pixels in the low-res box."""
        return int(self.NON_CUBIC_FACTOR * self.HII_DIM**3)

    @property
    def cell_size(self) -> un.Quantity[un.Mpc]:
        """The resolution of a low-res cell."""
        return (self.BOX_LEN / self.HII_DIM) * un.Mpc

    @property
    def cell_size_hires(self) -> un.Quantity[un.Mpc]:
        """The resolution of a hi-res cell."""
        return (self.BOX_LEN / self.DIM) * un.Mpc


@attrs.define(frozen=True, kw_only=True)
class AstroOptions(InputStruct):
    """
    Options for the ionization routines which enable/disable certain modules.

    Parameters
    ----------
    USE_MINI_HALOS : bool, optional
        Set to True if using mini-halos parameterization.
        If True, USE_TS_FLUCT and INHOMO_RECO must be True.
    USE_X_RAY_HEATING : bool, optional
        Whether to include X-ray heating (useful for debugging).
    USE_CMB_HEATING : bool, optional
        Whether to include CMB heating. (cf Eq.4 of Meiksin 2021, arxiv.org/abs/2105.14516)
    USE_LYA_HEATING : bool, optional
        Whether to use Lyman-alpha heating. (cf Sec. 3 of Reis+2021, doi.org/10.1093/mnras/stab2089)
    INHOMO_RECO : bool, optional
        Whether to perform inhomogeneous recombinations. Increases the computation
        time.
    USE_TS_FLUCT : bool, optional
        Whether to perform IGM spin temperature fluctuations (i.e. X-ray heating).
        Dramatically increases the computation time.
    M_MIN_in_Mass : bool, optional
        Whether the minimum halo mass (for ionization) is defined by
        mass or virial temperature. Only has an effect when SOURCE_MODEL == 'CONST-ION-EFF'
    PHOTON_CONS_TYPE : str, optional
        Whether to perform a small correction to account for the inherent
        photon non-conservation. This can be one of three types of correction:

        no-photoncons: No photon cosnervation correction,
        z-photoncons: Photon conservation correction by adjusting the redshift of the N_ion source field (Park+22)
        alpha-photoncons: Adjustment to the escape fraction power-law slope, based on fiducial results in Park+22, This runs a
        series of global xH evolutions and one calibration simulation to find the adjustment as a function of xH
        f-photoncons: Adjustment to the escape fraction normalisation, runs one calibration simulation to find the
        adjustment as a function of xH where f'/f = xH_global/xH_calibration
    FIX_VCB_AVG: bool, optional
        Determines whether to use a fixed vcb=VAVG (*regardless* of USE_RELATIVE_VELOCITIES). It includes the average effect of velocities but not its fluctuations. See Muñoz+21 (2110.13919).
    USE_EXP_FILTER: bool, optional
        Use the exponential filter (MFP-epsilon(r) from Davies & Furlanetto 2021) when calculating ionising emissivity fields
        NOTE: this does not affect other field filters, and should probably be used with HII_FILTER==0 (real-space top-hat)
    CELL_RECOMB: bool, optional
        An alternate way of counting recombinations based on the local cell rather than the filter region.
        This is part of the perspective shift (see Davies & Furlanetto 2021) from counting photons/atoms in a sphere and flagging a central
        pixel to counting photons which we expect to reach the central pixel, and taking the ratio of atoms in the pixel.
        This flag simply turns off the filtering of N_rec grids, and takes the recombinations in the central cell.
    USE_UPPER_STELLAR_TURNOVER: bool, optional
        Whether to use an additional powerlaw in stellar mass fraction at high halo mass. The pivot mass scale and power-law index are
        controlled by two parameters, UPPER_STELLAR_TURNOVER_MASS and UPPER_STELLAR_TURNOVER_INDEX respectively.
        This is currently only implemented using the discrete halo model, and has no effect otherwise.
    HALO_SCALING_RELATIONS_MEDIAN: bool, optional
        If True, halo scaling relation parameters (F_STAR10,t_STAR etc...) define the median of their conditional distributions
        If False, they describe the mean.
        This becomes important when using non-symmetric dristributions such as the log-normal
    HII_FILTER : string
        Filter for the halo or density field used to generate ionization field
        Available options are: 'spherical-tophat', 'sharp-k', and 'gaussian'
    HEAT_FILTER : int
        Filter for the halo or density field used to generate the spin-temperature field
        Available options are: 'spherical-tophat', 'sharp-k', and 'gaussian'
    IONISE_ENTIRE_SPHERE: bool, optional
        If True, ionises the entire sphere on the filter scale when an ionised region is found
        in the excursion set.
    INTEGRATION_METHOD_ATOMIC: str, optional
        The integration method to use for conditional MF integrals of atomic halos in the grids:
        NOTE: global integrals will use GSL QAG adaptive integration
        'GSL-QAG': GSL QAG adaptive integration,
        'GAUSS-LEGENDRE': Gauss-Legendre integration, previously forced in the interpolation tables,
        'GAMMA-APPROX': Approximate integration, assuming sharp cutoffs and a triple power-law for sigma(M) based on EPS
    INTEGRATION_METHOD_MINI: str, optional
        The integration method to use for conditional MF integrals of minihalos in the grids:
        'GSL-QAG': GSL QAG adaptive integration,
        'GAUSS-LEGENDRE': Gauss-Legendre integration, previously forced in the interpolation tables,
        'GAMMA-APPROX': Approximate integration, assuming sharp cutoffs and a triple power-law for sigma(M) based on EPS
    """

    USE_MINI_HALOS: bool = field(default=False, converter=bool)
    USE_X_RAY_HEATING: bool = field(default=True, converter=bool)
    USE_CMB_HEATING: bool = field(default=True, converter=bool)
    USE_LYA_HEATING: bool = field(default=True, converter=bool)
    INHOMO_RECO: bool = field(default=False, converter=bool)
    USE_TS_FLUCT: bool = field(default=False, converter=bool)
    FIX_VCB_AVG: bool = field(default=False, converter=bool)
    USE_EXP_FILTER: bool = field(default=True, converter=bool)
    CELL_RECOMB: bool = field(default=True, converter=bool)
    PHOTON_CONS_TYPE: Literal[
        "no-photoncons", "z-photoncons", "alpha-photoncons", "f-photoncons"
    ] = choice_field(
        default="no-photoncons",
    )
    USE_UPPER_STELLAR_TURNOVER: bool = field(default=True, converter=bool)
    M_MIN_in_Mass: bool = field(default=True, converter=bool)
    HALO_SCALING_RELATIONS_MEDIAN: bool = field(default=False, converter=bool)
    HII_FILTER: FilterOptions = choice_field(default="spherical-tophat")
    HEAT_FILTER: FilterOptions = choice_field(default="spherical-tophat")
    IONISE_ENTIRE_SPHERE: bool = field(default=False, converter=bool)

    INTEGRATION_METHOD_ATOMIC: IntegralMethods = choice_field(default="GAUSS-LEGENDRE")
    INTEGRATION_METHOD_MINI: IntegralMethods = choice_field(default="GAUSS-LEGENDRE")

    @USE_MINI_HALOS.validator
    def _USE_MINI_HALOS_vald(self, att, val):
        """
        Raise an error USE_MINI_HALOS is True with incompatible flags.

        This happens when INHOMO_RECO or USE_TS_FLUCT is False.
        """
        if val and not self.INHOMO_RECO:
            raise ValueError(
                "You have set USE_MINI_HALOS to True but INHOMO_RECO is False! "
            )
        if val and not self.USE_TS_FLUCT:
            raise ValueError(
                "You have set USE_MINI_HALOS to True but USE_TS_FLUCT is False! "
            )

    @PHOTON_CONS_TYPE.validator
    def _PHOTON_CONS_TYPE_vld(self, att, val):
        """Raise an error if using PHOTON_CONS_TYPE='z_photoncons' and USE_MINI_HALOS is True."""
        if self.USE_MINI_HALOS and val == "z-photoncons":
            raise ValueError(
                "USE_MINI_HALOS is not compatible with the redshift-based"
                " photon conservation corrections (PHOTON_CONS_TYPE=='z_photoncons')! "
            )

    @USE_EXP_FILTER.validator
    def _USE_EXP_FILTER_vld(self, att, val):
        """Raise an error if USE_EXP_FILTER is False and HII_FILTER!=0."""
        if val and self.HII_FILTER != "spherical-tophat":
            raise ValueError(
                "USE_EXP_FILTER can only be used with a real-space tophat HII_FILTER==0"
            )

        if val and not self.CELL_RECOMB:
            raise ValueError("USE_EXP_FILTER is True but CELL_RECOMB is False")


@attrs.define(frozen=True, kw_only=True)
class AstroParams(InputStruct):
    """
    Astrophysical parameters.

    NB: All Mean scaling relations are defined in log-space, such that the lines they produce
    give exp(<log(property)>), this means that increasing the lognormal scatter in these relations
    will increase the <property> but not <log(property)>

    Parameters
    ----------
    INHOMO_RECO : bool, optional
        Whether inhomogeneous recombinations are being calculated. This is not a part of the
        astro parameters structure, but is required by this class to set some default behaviour.
    HII_EFF_FACTOR : float, optional
        The ionizing efficiency of high-z galaxies (zeta, from Eq. 2 of Greig+2015).
        Higher values tend to speed up reionization.
    F_STAR10 : float, optional
        The fraction of galactic gas in stars for 10^10 solar mass haloes.
        Only used in the "new" parameterization,
        This is used along with `F_ESC10` to determine `HII_EFF_FACTOR` (which
        is then unused). See Eq. 11 of Greig+2018 and Sec 2.1 of Park+2018.
        Given in log10 units.
    F_STAR7_MINI : float, optional
        The fraction of galactic gas in stars for 10^7 solar mass minihaloes. Only used
        in the "minihalo" parameterization, i.e. when `USE_MINI_HALOS` is set to True
        (in :class:`AstroOptions`). If so, this is used along with `F_ESC7_MINI` to
        determine `HII_EFF_FACTOR_MINI` (which is then unused). See Eq. 8 of Qin+2020.
        If the MCG scaling relations are not provided explicitly, we extend the ACG
        ones by default. Given in log10 units.
    ALPHA_STAR : float, optional
        Power-law index of fraction of galactic gas in stars as a function of halo mass.
        See Sec 2.1 of Park+2018.
    ALPHA_STAR_MINI : float, optional
        Power-law index of fraction of galactic gas in stars as a function of halo mass,
        for MCGs. See Sec 2 of Muñoz+21 (2110.13919). If the MCG scaling relations are
        not provided explicitly, we extend the ACG ones by default.
    SIGMA_STAR : float, optional
        Lognormal scatter (dex) of the halo mass to stellar mass relation.
        Uniform across all masses and redshifts.
    SIGMA_SFR_LIM : float, optional
        Lognormal scatter (dex) of the stellar mass to SFR relation above a stellar mass of 1e10 solar.
    SIGMA_SFR_INDEX : float, optional
        index of the power-law between SFMS scatter and stellar mass below 1e10 solar.
    F_ESC10 : float, optional
        The "escape fraction", i.e. the fraction of ionizing photons escaping into the
        IGM, for 10^10 solar mass haloes. Only used in the "new" parameterization.
        This is used along with `F_STAR10` to determine `HII_EFF_FACTOR` (which
        is then unused). See Eq. 11 of Greig+2018 and Sec 2.1 of Park+2018.
    F_ESC7_MINI: float, optional
        The "escape fraction for minihalos", i.e. the fraction of ionizing photons escaping
        into the IGM, for 10^7 solar mass minihaloes. Only used in the "minihalo"
        parameterization, i.e. when `USE_MINI_HALOS` is set to True (in
        :class:`AstroOptions`). If so, this is used along with `F_ESC7_MINI` to determine
        `HII_EFF_FACTOR_MINI` (which is then unused). See Eq. 17 of Qin+2020. If the MCG
        scaling relations are not provided explicitly, we extend the ACG ones by default.
        Given in log10 units.
    ALPHA_ESC : float, optional
        Power-law index of escape fraction as a function of halo mass. See Sec 2.1 of
        Park+2018.
    M_TURN : float, optional
        Turnover mass (in log10 solar mass units) for quenching of star formation in
        halos, due to SNe or photo-heating feedback, or inefficient gas accretion.
        See Sec 2.1 of Park+2018.
    R_BUBBLE_MAX : float, optional
        Mean free path in Mpc of ionizing photons within ionizing regions (Sec. 2.1.2 of
        Greig+2015). Default is 50 if `INHOMO_RECO` is True, or 15.0 if not.
    ION_Tvir_MIN : float, optional
        Minimum virial temperature of star-forming haloes (Sec 2.1.3 of Greig+2015).
        Given in log10 units.
    L_X : float, optional
        The specific X-ray luminosity per unit star formation escaping host galaxies.
        Cf. Eq. 6 of Greig+2018. Given in log10 units. For the double power-law used in the Halo Model
        This gives the low-z limite.
    L_X_MINI: float, optional
        The specific X-ray luminosity per unit star formation escaping host galaxies for
        minihalos. Cf. Eq. 23 of Qin+2020. Given in log10 units. For the double
        power-law used in the Halo Model. This gives the low-z limite. If the MCG
        scaling relations are not provided explicitly, we extend the ACG ones by default.
    NU_X_THRESH : float, optional
        X-ray energy threshold for self-absorption by host galaxies (in eV). Also called
        E_0 (cf. Sec 4.1 of Greig+2018). Typical range is (100, 1500).
    X_RAY_SPEC_INDEX : float, optional
        X-ray spectral energy index (cf. Sec 4.1 of Greig+2018). Typical range is
        (-1, 3).
    X_RAY_Tvir_MIN : float, optional
        Minimum halo virial temperature in which X-rays are produced. Given in log10
        units. Default is `ION_Tvir_MIN`.
    F_H2_SHIELD: float, optional
        Self-shielding factor of molecular hydrogen when experiencing LW suppression.
        Cf. Eq. 12 of Qin+2020. Consistently included in A_LW fit from sims.
        If used we recommend going back to Macachek+01 A_LW=22.86.
    t_STAR : float, optional
        Fractional characteristic time-scale (fraction of hubble time) defining the
        star-formation rate of galaxies. See Sec 2.1, Eq. 3 of Park+2018.
    A_LW, BETA_LW: float, optional
        Impact of the LW feedback on Mturn for minihaloes. Default is 22.8685 and 0.47 following Machacek+01, respectively. Latest simulations suggest 2.0 and 0.6. See Sec 2 of Muñoz+21 (2110.13919).
    A_VCB, BETA_VCB: float, optional
        Impact of the DM-baryon relative velocities on Mturn for minihaloes. Default is 1.0 and 1.8, and agrees between different sims. See Sec 2 of Muñoz+21 (2110.13919).
    UPPER_STELLAR_TURNOVER_MASS:
        The pivot mass associated with the optional upper mass power-law of the stellar-halo mass relation
        (see AstroOptions.USE_UPPER_STELLAR_TURNOVER)
    UPPER_STELLAR_TURNOVER_INDEX:
        The power-law index associated with the optional upper mass power-law of the stellar-halo mass relation
        (see AstroOptions.USE_UPPER_STELLAR_TURNOVER)
    SIGMA_LX: float, optional
        Lognormal scatter (dex) of the Xray luminosity relation (a function of stellar mass, star formation rate and redshift).
        This scatter is uniform across all halo properties and redshifts.
    FIXED_VAVG : float, optional
        The fixed value of the average velocity used when AstroOptions.FIX_VCB_AVG is set to True.
    POP2_ION: float, optional
        Number of ionizing photons per baryon produced by Pop II stars.
    POP3_ION: float, optional
        Number of ionizing photons per baryon produced by Pop III stars.
    CLUMPING_FACTOR: float, optional
        Clumping factor of the IGM used ONLY in the x-ray partial ionisations (not the reionsiation model). Default is 2.0.
    ALPHA_UVB: float, optional
        The power-law index of the UVB spectrum. Used for Gamma12 in the recombination model
    DELTA_R_HII_FACTOR: float, optional
        The factor by which to decrease the size of the HII filter when calculating the HII regions.
    R_BUBBLE_MIN: float, optional
        Minimum size of ionized regions in Mpc. Default is 0.620350491.
    MAX_DVDR: float, optional
        Maximum value of the gradient of the velocity field used in the RSD algorithm.
    NU_X_BAND_MAX: float, optional
        The maximum frequency of the X-ray band used to calculate the X-ray Luminosity.
    NU_X_MAX: float, optional
        The maximum frequency of the integrals over nu for the x-ray heating/ionisation rates.
    """

    HII_EFF_FACTOR: float = field(
        default=30.0, converter=float, validator=validators.gt(0)
    )
    F_STAR10: float = field(
        default=-1.3,
        converter=float,
        validator=between(-3.0, 0.0),
        transformer=logtransformer,
    )
    ALPHA_STAR: float = field(
        default=0.5,
        converter=float,
    )
    F_STAR7_MINI: float = field(converter=float, transformer=logtransformer)
    ALPHA_STAR_MINI: float = field(converter=float)
    F_ESC10: float = field(
        default=-1.0,
        converter=float,
        transformer=logtransformer,
    )
    ALPHA_ESC: float = field(
        default=-0.5,
        converter=float,
    )
    F_ESC7_MINI: float = field(
        default=-2.0,
        converter=float,
        transformer=logtransformer,
    )
    M_TURN: float = field(
        default=8.7,
        converter=float,
        validator=validators.gt(0),
        transformer=logtransformer,
    )
    R_BUBBLE_MAX: float = field(
        default=15.0, converter=float, validator=validators.gt(0)
    )
    R_BUBBLE_MIN: float = field(
        default=0.620350491, converter=float, validator=validators.gt(0)
    )
    ION_Tvir_MIN: float = field(
        default=4.69897,
        converter=float,
        validator=validators.gt(0),
        transformer=logtransformer,
    )
    L_X: float = field(
        default=40.5,
        converter=float,
        validator=validators.gt(0),
        transformer=logtransformer,
    )
    L_X_MINI: float = field(
        converter=float, validator=validators.gt(0), transformer=logtransformer
    )
    NU_X_THRESH: float = field(
        default=500.0, converter=float, validator=validators.gt(0)
    )
    X_RAY_SPEC_INDEX: float = field(default=1.0, converter=float)
    X_RAY_Tvir_MIN: float = field(
        converter=float, validator=validators.gt(0), transformer=logtransformer
    )
    F_H2_SHIELD: float = field(default=0.0, converter=float)
    t_STAR: float = field(default=0.5, converter=float, validator=between(0, 1))
    A_LW: float = field(default=2.0, converter=float, validator=validators.gt(0))
    BETA_LW: float = field(default=0.6, converter=float)
    A_VCB: float = field(default=1.0, converter=float)
    BETA_VCB: float = field(default=1.8, converter=float)
    UPPER_STELLAR_TURNOVER_MASS: float = field(
        default=11.447, converter=float, transformer=logtransformer
    )
    UPPER_STELLAR_TURNOVER_INDEX: float = field(default=-0.6, converter=float)
    SIGMA_STAR: float = field(
        default=0.25, converter=float, transformer=dex2exp_transformer
    )
    SIGMA_LX: float = field(
        default=0.5, converter=float, transformer=dex2exp_transformer
    )
    SIGMA_SFR_LIM: float = field(
        default=0.19, converter=float, transformer=dex2exp_transformer
    )
    SIGMA_SFR_INDEX: float = field(default=-0.12, converter=float)

    T_RE: float = field(default=2e4, converter=float)
    FIXED_VAVG: float = field(
        default=25.86, converter=float, validator=validators.gt(0)
    )
    POP2_ION: float = field(default=5000.0, converter=float)
    POP3_ION: float = field(default=44021.0, converter=float)

    PHOTONCONS_CALIBRATION_END: float = field(default=3.5, converter=float)
    CLUMPING_FACTOR: float = field(
        default=2.0, converter=float, validator=validators.gt(0)
    )
    ALPHA_UVB: float = field(default=5.0, converter=float)
    R_MAX_TS: float = field(default=500.0, converter=float, validator=validators.gt(0))
    N_STEP_TS: float = field(default=40, converter=int, validator=validators.gt(0))
    MAX_DVDR: float = field(default=0.2, converter=float, validator=validators.ge(0))

    DELTA_R_HII_FACTOR: float = field(
        default=1.1, converter=float, validator=validators.gt(1.0)
    )

    NU_X_BAND_MAX: float = field(
        default=2000.0, converter=float, validator=validators.gt(0)
    )
    NU_X_MAX: float = field(
        default=10000.0, converter=float, validator=validators.gt(0)
    )

    # set the default of the minihalo scalings to continue the same PL
    @F_STAR7_MINI.default
    def _F_STAR7_MINI_default(self):
        return self.F_STAR10 - 3 * self.ALPHA_STAR  # -3*alpha since 1e7/1e10 = 1e-3

    @ALPHA_STAR_MINI.default
    def _ALPHA_STAR_MINI_default(self):
        return self.ALPHA_STAR

    @L_X_MINI.default
    def _L_X_MINI_default(self):
        return self.L_X

    @X_RAY_Tvir_MIN.default
    def _X_RAY_Tvir_MIN_default(self):
        return self.ION_Tvir_MIN

    @NU_X_THRESH.validator
    def _NU_X_THRESH_vld(self, att, val):
        """Check if the choice of NU_X_THRESH is sensible."""
        if val < 100.0:
            raise ValueError(
                "Chosen NU_X_THRESH is < 100 eV. NU_X_THRESH must be above 100 eV as it describes X-ray photons"
            )
        elif val >= self.NU_X_BAND_MAX:
            raise ValueError(
                f"""
                Chosen NU_X_THRESH > {self.NU_X_BAND_MAX}, which is the upper limit of the adopted X-ray band
                (fiducially the soft band 0.5 - 2.0 keV). If you know what you are doing with this
                choice, please modify the parameter: NU_X_BAND_MAX"""
            )
        elif self.NU_X_BAND_MAX > self.NU_X_MAX:
            raise ValueError(
                f"""
                Chosen NU_X_BAND_MAX > {self.NU_X_MAX}, which is the upper limit of X-ray integrals (fiducially 10 keV)
                If you know what you are doing, please modify the parameter:
                NU_X_MAX
                """
            )


class InputCrossValidationError(ValueError):
    """Error when two parameters from different structs aren't consistent."""


def input_param_field(kls: InputStruct):
    """Create an attrs field that must be an InputStruct.

    Parameters
    ----------
    kls : InputStruct subclass
        The parameter structure which should be returned as an attrs field
    """
    return _field(
        default=kls.new(),
        converter=kls.new,
        validator=validators.instance_of(kls),
    )


def get_logspaced_redshifts(
    min_redshift: float,
    z_step_factor: float,
    max_redshift: float,
) -> tuple[float]:
    """Compute a sequence of redshifts to evolve over that are log-spaced."""
    redshifts = [min_redshift]
    while redshifts[-1] < max_redshift:
        redshifts.append((redshifts[-1] + 1.0) * z_step_factor - 1.0)

    return tuple(redshifts[::-1])


def _node_redshifts_converter(value) -> tuple[float] | None:
    if value is None or len(value) == 0:
        return ()
    if hasattr(value, "__len__"):
        return tuple(sorted([float(v) for v in value], reverse=True))
    return (float(value),)


@attrs.define(kw_only=True, frozen=True)
class InputParameters:
    """A class defining a collection of InputStruct instances.

    This class simplifies combining different InputStruct instances together, performing
    validation checks between them, and being able to cross-check compatibility between
    different sets of instances.

    Parameters
    ----------
    random_seed
        The seed that determines the realization produced by a 21cmFAST run.
    node_redshifts
        The redshifts at which coeval boxes will be computed. By default,
        empty if no evolution is required, and logarithmically spaced in (1+z)
        between z=5.5 and SimulationOptions.Z_HEAT_MAX if evolution is required.
    cosmo_params
        Cosmological parameters of a 21cmFAST run.
    simulation_options
        Parameters controlling the simulation as a whole, e.g. the box size and
        dimensionality.
    matter_options
        Parameters controlling the matter field generated by 21cmFAST.
    astro_options
        Options for which physical processes to include in the simulation.
    astro_params
        Astrophysical parameter values.

    """

    random_seed = _field(converter=int)
    cosmo_params: CosmoParams = input_param_field(CosmoParams)
    matter_options: MatterOptions = input_param_field(MatterOptions)
    simulation_options: SimulationOptions = input_param_field(SimulationOptions)
    astro_options: AstroOptions = input_param_field(AstroOptions)
    astro_params: AstroParams = input_param_field(AstroParams)
    node_redshifts = _field(converter=_node_redshifts_converter)
    cosmo_tables: CosmoTables = field()

    @node_redshifts.default
    def _node_redshifts_default(self):
        return (
            get_logspaced_redshifts(
                min_redshift=5.5,
                max_redshift=self.simulation_options.Z_HEAT_MAX,
                z_step_factor=self.simulation_options.ZPRIME_STEP_FACTOR,
            )
            if self.evolution_required
            else None
        )

    @node_redshifts.validator
    def _node_redshifts_validator(self, att, val):
        if (self.astro_options.INHOMO_RECO or self.astro_options.USE_TS_FLUCT) and (
            (max(val) if val else 0.0) < self.simulation_options.Z_HEAT_MAX
        ):
            raise ValueError(
                "For runs with inhomogeneous recombinations or spin temperature fluctuations,\n"
                + f"your maximum passed node_redshifts {max(val) if hasattr(val, '__len__') else val} must be above Z_HEAT_MAX {self.simulation_options.Z_HEAT_MAX}"
            )

    @cosmo_tables.default
    def _cosmo_tables_default(self):
        if self.matter_options.POWER_SPECTRUM == "CLASS":
            if self.simulation_options.K_MAX_FOR_CLASS is not None:
                k_max = self.simulation_options.K_MAX_FOR_CLASS / un.Mpc
            else:
                if self.astro_options.USE_MINI_HALOS:
                    M_min = 1e5 * un.M_sun
                else:
                    M_min = 1e9 * un.M_sun
                R_min = pow(
                    M_min
                    / (
                        4
                        * np.pi
                        / 3.0
                        * self.cosmo_params.cosmo.critical_density0
                        * self.cosmo_params.OMm
                    ),
                    1 / 3,
                )
                k_max = (2 * np.pi / R_min).to(
                    "1/Mpc"
                ) * 1.5  # Multiply by 1.5 for better precision

            classy_output = run_classy(
                h=self.cosmo_params.hlittle,
                Omega_cdm=self.cosmo_params.OMm - self.cosmo_params.OMb,
                Omega_b=self.cosmo_params.OMb,
                n_s=self.cosmo_params.POWER_INDEX,
                sigma8=self.cosmo_params.SIGMA_8,
                output="mTk,vTk",
                P_k_max=k_max,
            )
            # Linear matter density transfer function at z=0
            transfer_density = get_transfer_function(
                classy_output=classy_output, kind="d_m", z=0.0
            )
            # Linear vcb transfer function at kinematic decoupling
            z_dec = find_redshift_kinematic_decoupling(classy_output)
            transfer_vcb = (
                (
                    get_transfer_function(
                        classy_output=classy_output, kind="v_cb", z=z_dec
                    )
                    / constants.c  # Need to normalize by c, because ComputeInitialConditions() accepts to receive a dimensionless transfer function
                ).to(un.dimensionless_unscaled)
            )

            # Include a sample at k=0
            k_transfer_with_0 = np.concatenate(([0.0], k_transfer))
            transfer_density = np.concatenate(([0.0], transfer_density))
            transfer_vcb = np.concatenate(([0.0], transfer_vcb))

            # we use A_s to normalize the power spectrum only if it was provided
            USE_SIGMA_8 = self.cosmo_params._A_s is None

            cosmo_tables = CosmoTables(
                transfer_density=Table1D(
                    size=k_transfer_with_0.size,
                    x_values=k_transfer_with_0,
                    y_values=transfer_density,
                ),
                transfer_vcb=Table1D(
                    size=k_transfer_with_0.size,
                    x_values=k_transfer_with_0,
                    y_values=transfer_vcb,
                ),
                ps_norm=self.cosmo_params.SIGMA_8
                if USE_SIGMA_8
                else self.cosmo_params.A_s,
                USE_SIGMA_8=USE_SIGMA_8,
            )
        else:
            # we ALWAYS use sigma8 to normalize the power spectrum if we don't use CLASS
            cosmo_tables = CosmoTables(
                ps_norm=self.cosmo_params.SIGMA_8, USE_SIGMA_8=True
            )
        return cosmo_tables

    @astro_options.validator
    def _astro_options_validator(self, att, val):
        if self.matter_options is None:
            return
        if val.USE_MINI_HALOS:
            if not self.matter_options.USE_RELATIVE_VELOCITIES and not val.FIX_VCB_AVG:
                warnings.warn(
                    "USE_MINI_HALOS needs USE_RELATIVE_VELOCITIES to get the right evolution!",
                    stacklevel=2,
                )
            if self.matter_options.SOURCE_MODEL == "CONST-ION-EFF":
                raise ValueError(
                    "SOURCE_MODEL == 'CONST-ION-EFF' is not compatible with USE_MINI_HALOS=True"
                )

        if self.matter_options.lagrangian_source_grid:
            if val.PHOTON_CONS_TYPE == "z-photoncons":
                raise ValueError(
                    f"SOURCE_MODEL={self.matter_options.SOURCE_MODEL} is not compatible with the redshift-based"
                    " photon conservation corrections (PHOTON_CONS_TYPE=='z_photoncons')! Use a "
                    " different PHOTON_CONS_TYPE or set SOURCE_MODEL='E-INTEGRAL' to use the old"
                    " source model"
                )
        else:
            if val.USE_EXP_FILTER:
                raise ValueError(
                    f"USE_EXP_FILTER is not compatible with SOURCE_MODEL == {self.matter_options.SOURCE_MODEL}"
                )
        if (
            not self.matter_options.has_discrete_halos
            and val.USE_UPPER_STELLAR_TURNOVER
        ):
            raise NotImplementedError(
                f"USE_UPPER_STELLAR_TURNOVER is not yet implemented for SOURCE_MODEL = {self.matter_options.SOURCE_MODEL}"
            )
        if self.matter_options.HMF not in ["PS", "ST", "DELOS"]:
            warnings.warn(
                "A selection of a mass function other than Press-Schechter, Sheth-Tormen or Delos will"
                "Result in the use of the EPS conditional mass function, normalised the unconditional"
                "mass function provided by the user as matter_options.HMF",
                stacklevel=2,
            )
        elif (
            val.INTEGRATION_METHOD_ATOMIC == "GAMMA-APPROX"
            or val.INTEGRATION_METHOD_MINI == "GAMMA-APPROX"
            or self.matter_options.SOURCE_MODEL == "CONST-ION-EFF"
        ) and self.matter_options.HMF != "PS":
            warnings.warn(
                "Your model (either SOURCE_MODEL=='CONST-ION-EFF' or INTEGRATION_METHOD_X=='GAMMA-APPROX')"
                "uses the EPS conditional mass function normalised to the unconditional mass"
                "function provided by the user as matter_options.HMF",
                stacklevel=2,
            )

    @astro_params.validator
    def _astro_params_validator(self, att, val):
        if val.R_BUBBLE_MAX > self.simulation_options.BOX_LEN:
            raise InputCrossValidationError(
                f"R_BUBBLE_MAX is larger than BOX_LEN ({val.R_BUBBLE_MAX} > {self.simulation_options.BOX_LEN}). This is not allowed."
            )

        if val.R_BUBBLE_MAX != 50 and self.astro_options.INHOMO_RECO:
            warnings.warn(
                "You are setting R_BUBBLE_MAX != 50 when INHOMO_RECO=True. "
                "This is non-standard (but allowed), and usually occurs upon manual "
                "update of INHOMO_RECO",
                stacklevel=2,
            )

        if val.M_TURN > 8 and self.astro_options.USE_MINI_HALOS:
            warnings.warn(
                "You are setting M_TURN > 8 when USE_MINI_HALOS=True. "
                "This is non-standard (but allowed), and usually occurs upon manual "
                "update of M_TURN",
                stacklevel=2,
            )

        if (
            self.astro_options.HII_FILTER == "sharp-k"
            and val.R_BUBBLE_MAX > self.simulation_options.BOX_LEN / 3
        ):
            msg = (
                "Your R_BUBBLE_MAX is > BOX_LEN/3 "
                f"({val.R_BUBBLE_MAX} > {self.simulation_options.BOX_LEN / 3})."
                f" This can produce strange reionisation topologies"
            )

            if config["ignore_R_BUBBLE_MAX_error"]:
                warnings.warn(msg, stacklevel=2)
            else:
                raise ValueError(
                    msg
                    + " To ignore this error, set `py21cmfast.config['ignore_R_BUBBLE_MAX_error'] = True`"
                )

    @simulation_options.validator
    def _simulation_options_validator(self, att, val):
        # perform a very rudimentary check to see if we are underresolved and not using the linear approx
        if self.matter_options is not None and (
            val.cell_size_hires > 1 * un.Mpc
            and self.matter_options.PERTURB_ALGORITHM != "LINEAR"
        ):
            warnings.warn(
                "Resolution is likely too low for accurate evolved density fields. "
                "It is recommended that you either increase the resolution "
                "(DIM/BOX_LEN) or set the EVOLVE_DENSITY_LINEARLY flag to True. "
                f"Got DIM={val.DIM}, BOX_LEN={val.BOX_LEN}, resolution={val.cell_size_hires} Mpc.",
                stacklevel=2,
            )

        if (
            self.matter_options.POWER_SPECTRUM != "CLASS"
            and self.cosmo_params._A_s is not None
        ):
            warnings.warn(
                f"You have chosen to work with POWER_SPECTRUM={self.matter_options.POWER_SPECTRUM}, "
                "but at the same time you work with A_s (rather than SIGMA_8). "
                "While this is allowed, it is important to realize that it is impossible "
                "to normalize correctly the power spectrum with A_s while using the "
                f"{self.matter_options.POWER_SPECTRUM} transfer function. "
                "CLASS will convert your A_s to SIGMA_8.",
                stacklevel=2,
            )

    def __getitem__(self, key):
        """Get an item from the instance in a dict-like manner."""
        # Also allow using **input_parameters
        return getattr(self, key)

    def is_compatible_with(self, other: Self) -> bool:
        """Check if this object is compatible with another parameter struct.

        Compatibility is slightly different from strict equality. Compatibility requires
        that if a parameter struct *exists* on the other object, then it must be equal
        to this one. That is, if astro_params is None on the other InputParameter object,
        then this one can have astro_params as NOT None, and it will still be
        compatible. However the inverse is not true -- if this one has astro_params as
        None, then all others must have it as None as well.
        """
        if not isinstance(other, InputParameters):
            return False

        return not any(
            other[key] is not None and self[key] is not None and self[key] != other[key]
            for key in self.merge_keys()
        )

    def evolve_input_structs(self, **kwargs):
        """Return an altered clone of the `InputParameters` structs.

        Unlike clone(), this function takes fields from the constituent `InputStruct` classes
        and only overwrites those sub-fields instead of the entire field
        """
        struct_args = {}
        kwargs_copy = kwargs.copy()
        for inp_type in (
            "cosmo_params",
            "simulation_options",
            "matter_options",
            "astro_params",
            "astro_options",
            "cosmo_tables",
        ):
            obj = getattr(self, inp_type)
            struct_args[inp_type] = obj.clone(
                **{k: kwargs_copy.pop(k) for k in kwargs if hasattr(obj, k)}
            )

        if len(kwargs_copy):
            wrong_key = next(iter(kwargs_copy.keys()))
            raise TypeError(f"{wrong_key} is not a valid keyword input.")

        inputs_clone = self.clone(**struct_args)
        if inputs_clone.matter_options.POWER_SPECTRUM == "CLASS":
            if (
                self.matter_options.POWER_SPECTRUM != "CLASS"
                or np.any([hasattr(self.cosmo_params, k) for k in kwargs])
                or (
                    self.simulation_options.K_MAX_FOR_CLASS
                    != inputs_clone.simulation_options.K_MAX_FOR_CLASS
                )
            ):
                # we need to run CLASS again and update cosmo_tables
                struct_args["cosmo_tables"] = inputs_clone._cosmo_tables_default()
                inputs_clone = self.clone(**struct_args)
        else:
            # No need to have the tables from the original inputs, but we do need to change ps_norm and USE_SIGMA_8
            struct_args["cosmo_tables"] = CosmoTables(
                ps_norm=inputs_clone.cosmo_params.SIGMA_8, USE_SIGMA_8=True
            )
            inputs_clone = self.clone(**struct_args)

        return inputs_clone

    @classmethod
    def from_template(
        cls,
        name: str | Path | Sequence[str | Path],
        random_seed: int,
        node_redshifts: tuple[float] | None = None,
        **kwargs,
    ):
        """Construct full InputParameters instance from native or TOML file template.

        Parameters
        ----------
        name
            Either a name of a built-in template, or a path to a parameter TOML,
            or a list of such names/paths. If a list, the final parameters will be
            built from left-to-right so that the parameters at the end of the list
            will have precedence.
        random_seed
            A random seed to use for the run. Must be specified manually.
        node_redshifts
            The redshifts at which to evolve the simulation (if applicable). Default
            detects whether evolution is required and sets the node redshifts according
            to the simulation parameters.

        Other Parameters
        ----------------
        All other parameters are interpreted as elements of :class:`InputStruct`
        subclasses (e.g. :class:`SimulationOptions`) and will over-ride what is in
        the template.
        """
        from .._templates import create_params_from_template

        cls_kw = {"random_seed": random_seed}
        if node_redshifts is not None:
            cls_kw["node_redshifts"] = node_redshifts

        dct = create_params_from_template(name, **kwargs)
        dct.pop("cosmo_tables")
        return cls(**dct, **cls_kw)

    def clone(self, **kwargs):
        """Generate a copy of the InputParameter structure with specified changes."""
        return evolve(self, **kwargs)

    def __repr__(self):
        """
        Return a string representation of the structure.

        Created by combining repr methods from the InputStructs
        which make up this object
        """
        return (
            f"cosmo_params: {self.cosmo_params!r}\n"
            + f"simulation_options: {self.simulation_options!r}\n"
            + f"matter_options: {self.matter_options!r}\n"
            + f"astro_params: {self.astro_params!r}\n"
            + f"astro_options: {self.astro_options!r}\n"
        )

    # NOTE: These hashes are used to compare structs within a run, and so don't need to stay
    #   constant between sessions
    @cached_property
    def _user_cosmo_hash(self):
        """A hash generated from the user and cosmo params as well random seed."""
        return hash(
            (
                self.random_seed,
                self.simulation_options,
                self.matter_options,
                self.cosmo_params,
            )
        )

    @cached_property
    def _zgrid_hash(self):
        return hash((self._user_cosmo_hash, self.node_redshifts))

    @cached_property
    def _full_hash(self):
        return hash(
            (
                self.random_seed,
                self.cosmo_params,
                self.matter_options,
                self.simulation_options,
                self.astro_options,
                self.astro_params,
                self.node_redshifts,
            )
        )

    @property
    def evolution_required(self) -> bool:
        """Whether evolution is required for these parameters."""
        return (
            self.astro_options.USE_TS_FLUCT
            or self.astro_options.INHOMO_RECO
            or self.astro_options.USE_MINI_HALOS
        )

    def with_logspaced_redshifts(
        self,
        zmin: float = 5.5,
        zmax: float | None = None,
        zstep_factor: float | None = None,
    ) -> Self:
        """Create a new InputParameters instance with logspaced redshifts."""
        if zmax is None:
            zmax = self.simulation_options.Z_HEAT_MAX

        if zstep_factor is None:
            zstep_factor = self.simulation_options.ZPRIME_STEP_FACTOR

        return self.clone(
            node_redshifts=get_logspaced_redshifts(
                min_redshift=zmin,
                z_step_factor=zstep_factor,
                max_redshift=zmax,
            )
        )

    def asdict(
        self,
        only_structs: bool = False,
        camel: bool = False,
        remove_base_cosmo: bool = True,
        only_cstruct_params: bool = True,
        use_aliases: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Convert the instance to a recursive dictionary."""
        dct = attrs.asdict(self, recurse=True)

        if remove_base_cosmo:
            del dct["cosmo_params"]["_base_cosmo"]

        inpstructs = [
            k for k in dct if isinstance(getattr(self, k), (InputStruct | CosmoTables))
        ]
        if only_structs:
            dct = {k: v for k, v in dct.items() if k in inpstructs}

        if use_aliases:
            # Change keys to aliases instead of attribute names if desired.
            # i.e. change _DIM to DIM, which is actually what is needed to
            # instantiate a class.
            for k, v in dct.items():
                attribute = getattr(self, k)
                if isinstance(attribute, InputStruct):
                    fields = attrs.fields_dict(attribute.__class__)
                    dct[k] = {fields[kk].alias: vv for kk, vv in v.items()}

        if only_cstruct_params:
            for k, v in dct.items():
                attribute = getattr(self, k)
                if isinstance(attribute, InputStruct):
                    dct[k] = {
                        kk: vv for kk, vv in v.items() if kk in attribute.cdict
                    } | {
                        kk: getattr(attribute, kk)
                        for kk in attribute.cdict
                        if kk not in dct[k]
                    }

        if camel:
            dct = {
                (snake_to_camel(k) if k in inpstructs else k): v for k, v in dct.items()
            }

        return dct

    def __attrs_post_init__(self) -> None:
        """Run a check after initialization.

        Currently just checks that the halo mass ranges are sensible.
        """
        check_halomass_range(self)


def check_halomass_range(inputs: InputParameters) -> None:
    """Check that the halo mass range is sensible given the parameters.

    This function checks that the minimum halo mass set by the various resolutions
    and flags does not have any gaps. We raise an error if there is a gap, and a warning
    if it is above the turnover mass.
    """
    # There are no problems if we are not using halos
    if not inputs.matter_options.lagrangian_source_grid:
        return

    # simplified behaviour of lib.minimum_source_mass()
    if inputs.astro_options.USE_MINI_HALOS:
        min_integral_mass = 1e5 * un.M_sun
    else:
        min_integral_mass = (
            max(inputs.astro_params.cdict["M_TURN"] / 50, 1e5) * un.M_sun
        )
    max_integral_mass = 1e16 * un.M_sun  # define macro in hmf.h

    massdens = inputs.cosmo_params.cosmo.critical_density(0) * inputs.cosmo_params.OMm
    hires_cell_mass = (massdens * inputs.simulation_options.cell_size_hires**3).to(
        un.M_sun
    )
    lores_cell_mass = (massdens * inputs.simulation_options.cell_size**3).to(un.M_sun)
    pt_cell_mass = (
        hires_cell_mass
        if inputs.matter_options.PERTURB_ON_HIGH_RES
        else lores_cell_mass
    )

    has_dexm_halos = inputs.matter_options.SOURCE_MODEL in ["DEXM-ESF", "CHMF-SAMPLER"]
    has_sampled_halos = inputs.matter_options.SOURCE_MODEL == "CHMF-SAMPLER"
    has_integrals = (
        min_integral_mass / un.M_sun < inputs.simulation_options.SAMPLER_MIN_MASS
    )

    min_cellint = min_integral_mass
    if inputs.matter_options.SOURCE_MODEL == "CHMF-SAMPLER":
        max_cellint = inputs.simulation_options.SAMPLER_MIN_MASS * un.M_sun
    elif inputs.matter_options.SOURCE_MODEL == "DEXM-ESF":
        max_cellint = hires_cell_mass
    else:
        max_cellint = max_integral_mass

    max_cellint = min(max_cellint, pt_cell_mass)

    min_sampler = inputs.simulation_options.SAMPLER_MIN_MASS * un.M_sun
    # if the cell is smaller, the sampler won't draw any halos
    max_sampler = max(lores_cell_mass, min_sampler)

    min_dexm = lores_cell_mass if has_sampled_halos else hires_cell_mass
    # not the real maximum, (7 sigma), but sufficient for our checks here
    max_dexm = 1e16 * un.M_sun

    mass_limits = ()
    names = ()
    if has_integrals:
        mass_limits += ((min_cellint, max_cellint),)
        names += ("integrals",)
    if has_sampled_halos:
        mass_limits += ((min_sampler, max_sampler),)
        names += ("sampler",)
    if has_dexm_halos:
        mass_limits += ((min_dexm, max_dexm),)
        names += ("dexm",)

    for i in range(len(mass_limits) - 1):
        if mass_limits[i][1] != mass_limits[i + 1][0]:
            raise ValueError(
                f"There is a gap/overlap in the halo mass ranges of {dict(zip(names, mass_limits, strict=False))}. "
                "This will lead to unphysical results. Please adjust your parameters to remove this gap."
            )

    if min(min(mass_limits)) > min_integral_mass:
        warnings.warn(
            f"The minimum halo mass {min(min(mass_limits)):.2e} is high compared to the turnover {inputs.astro_params.cdict['M_TURN']:.2e}. "
            f"Halos below {min(min(mass_limits)):.2e} will not be accounted for in the simulation.",
            stacklevel=2,
        )

    if max(max(mass_limits)) < max_integral_mass:
        warnings.warn(
            f"The maximum halo mass {max(max(mass_limits)):.2e} is below the integral mass {max_integral_mass:.2e}. "
            f"Halos above {max(max(mass_limits)):.2e} will not be accounted for in the simulation.",
            stacklevel=2,
        )
