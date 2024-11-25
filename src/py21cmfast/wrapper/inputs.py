"""
Input parameter classes.

There are four input parameter/option classes, not all of which are required for any
given function. They are :class:`UserParams`, :class:`CosmoParams`, :class:`AstroParams`
and :class:`FlagOptions`. Each of them defines a number of variables, and all of these
have default values, to minimize the burden on the user. These defaults are accessed via
the ``_defaults_`` class attribute of each class. The available parameters for each are
listed in the documentation for each class below.

Along with these, the module exposes ``global_params``, a singleton object of type
:class:`GlobalParams`, which is a simple class providing read/write access to a number of parameters
used throughout the computation which are very rarely varied.
"""

from __future__ import annotations

import logging
import warnings
from astropy import units as un
from astropy.cosmology import FLRW, Planck15
from attrs import converters, define
from attrs import field as _field
from attrs import validators

from .._cfg import config
from .._data import DATA_PATH
from ..c_21cmfast import ffi, lib
from .globals import global_params
from .structs import InputStruct

logger = logging.getLogger(__name__)


def field(*, transformer=None, **kw):
    """Define an attrs field with a 'transformer' property.

    The transformer, if given, should be a function of a single variable, which will
    be the attribute's value. It will be used to transform the value before usage in
    C-code (e.g. by transformin from log to linear space).
    """
    return _field(metadata={"transformer": transformer}, **kw)


def logtransformer(x):
    """An attrs field transformer that converts from log to linear space."""
    return 10**x


def dex2exp_transformer(x):
    """An attrs transformer that converts from dex to exponential space."""
    return 2.3025851 * x


def choice_transformer(choices):
    """A factory function that produces a transformer that converts a string to int.

    The function must be passed a list of string choices. The resulting int is the
    index of the choice made.
    """

    def transformer(choice) -> int:
        return choices.index(choice)

    return transformer


def between(mn, mx):
    """An attrs validator for validating that a value is between two values."""

    def vld(inst, att, val):
        if val < mn or val > mx:
            raise ValueError(f"{att.name} must be between {mn} and {mx}")


# Cosmology is from https://arxiv.org/pdf/1807.06209.pdf
# Table 2, last column. [TT,TE,EE+lowE+lensing+BAO]
Planck18 = Planck15.clone(
    Om0=(0.02242 + 0.11933) / 0.6766**2,
    Ob0=0.02242 / 0.6766**2,
    H0=67.66,
    name="Planck18",
)


@define(frozen=True, kw_only=True)
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
    """

    _base_cosmo = field(
        default=Planck18, validator=validators.instance_of(FLRW), eq=False, repr=False
    )
    SIGMA_8 = field(default=0.8102, converter=float, validator=validators.gt(0))
    hlittle = field(default=Planck18.h, converter=float, validator=validators.gt(0))
    OMm = field(default=Planck18.Om0, converter=float, validator=validators.gt(0))
    OMb = field(default=Planck18.Ob0, converter=float, validator=validators.gt(0))
    POWER_INDEX = field(default=0.9665, converter=float, validator=validators.gt(0))

    @property
    def OMl(self):
        """Omega lambda, dark energy density."""
        return 1 - self.OMm

    @property
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


@define(frozen=True, kw_only=True)
class UserParams(InputStruct):
    """
    Structure containing user parameters (with defaults).

    Parameters
    ----------
    HII_DIM : int, optional
        Number of cells for the low-res box. Default 200.
    DIM : int,optional
        Number of cells for the high-res box (sampling ICs) along a principal axis. To avoid
        sampling issues, DIM should be at least 3 or 4 times HII_DIM, and an integer multiple.
        By default, it is set to 3*HII_DIM.
    NON_CUBIC_FACTOR : float, optional
        Factor which allows the creation of non-cubic boxes. It will shorten/lengthen the line
        of sight dimension of all boxes. NON_CUBIC_FACTOR * DIM/HII_DIM must result in an integer
    BOX_LEN : float, optional
        Length of the box, in Mpc. Default 300 Mpc.
    HMF: int or str, optional
        Determines which halo mass function to be used for the normalisation of the
        collapsed fraction (default Sheth-Tormen). If string should be one of the
        following codes:
        0: PS (Press-Schechter)
        1: ST (Sheth-Tormen)
        2: Watson (Watson FOF)
        3: Watson-z (Watson FOF-z)
        3: Delos (Delos+23)
    USE_RELATIVE_VELOCITIES: int, optional
        Flag to decide whether to use relative velocities.
        If True, POWER_SPECTRUM is automatically set to 5. Default False.
    POWER_SPECTRUM: int or str, optional
        Determines which power spectrum to use, default EH (unless `USE_RELATIVE_VELOCITIES`
        is True). If string, use the following codes:
        0: EH
        1: BBKS
        2: EFSTATHIOU
        3: PEEBLES
        4: WHITE
        5: CLASS (single cosmology)
    N_THREADS : int, optional
        Sets the number of processors (threads) to be used for performing 21cmFAST.
        Default 1.
    PERTURB_ON_HIGH_RES : bool, optional
        Whether to perform the Zel'Dovich or 2LPT perturbation on the low or high
        resolution grid.
    NO_RNG : bool, optional
        Ability to turn off random number generation for initial conditions. Can be
        useful for debugging and adding in new features
    USE_FFTW_WISDOM : bool, optional
        Whether or not to use stored FFTW_WISDOMs for improving performance of FFTs
    USE_INTERPOLATION_TABLES: bool, optional
        If True, calculates and evaluates quantites using interpolation tables, which
        is considerably faster than when performing integrals explicitly.
    INTEGRATION_METHOD_ATOMIC: int, optional
        The integration method to use for conditional MF integrals of atomic halos in the grids:
        NOTE: global integrals will use GSL QAG adaptive integration
        0: GSL QAG adaptive integration,
        1: Gauss-Legendre integration, previously forced in the interpolation tables,
        2: Approximate integration, assuming sharp cutoffs and a triple power-law for sigma(M) based on EPS
    INTEGRATION_METHOD_MINI: int, optional
        The integration method to use for conditional MF integrals of minihalos in the grids:
        0: GSL QAG adaptive integration,
        1: Gauss-Legendre integration, previously forced in the interpolation tables,
        2: Approximate integration, assuming sharp cutoffs and a triple power-law for sigma(M) based on EPS
    USE_2LPT: bool, optional
        Whether to use second-order Lagrangian perturbation theory (2LPT).
        Set this to True if the density field or the halo positions are extrapolated to
        low redshifts. The current implementation is very naive and adds a factor ~6 to
        the memory requirements. Reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118
        Appendix D.
    MINIMIZE_MEMORY: bool, optional
        If set, the code will run in a mode that minimizes memory usage, at the expense
        of some CPU/disk-IO. Good for large boxes / small computers.
    STOC_MINIMUM_Z: float, optional
        The minimum (first) redshift at which to calculate the halo boxes, will behave as follows:
        If STOC_MINIMUM_Z is set, we step DOWN from the requested redshift by ZPRIME_STEP_FACTOR
        until we get to STOC_MINIMUM_Z, where the last z-step will be shorter to be exactly at
        STOC_MINIMUM_Z, we then build the halo boxes from low to high redshift.
        If STOC_MINIMUM_Z is not provided, we simply sample at the given redshift, unless
        USE_TS_FLUCT is given or we want a lightcone, in which case the minimum redshift is set
        to the minimum redshift of those fields.
    SAMPLER_MIN_MASS: float, optional
        The minimum mass to sample in the halo sampler when USE_HALO_FIELD and HALO_STOCHASTICITY are true,
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
    SAMPLE_METHOD: int, optional
        The sampling method to use in the halo sampler when calculating progenitor populations:
        0: Mass-limited CMF sampling, where samples are drawn until the expected mass is reached
        1: Number-limited CMF sampling, where we select a number of halos from the Poisson distribution
        and then sample the CMF that many times
        2: Sheth et al 1999 Partition sampling, where the EPS collapsed fraction is sampled (gaussian tail)
        and then the condition is updated using the conservation of mass.
        3: Parkinsson et al 2008 Binary split model as in DarkForest (Qiu et al 2021) where the EPS merger rate
        is sampled on small internal timesteps such that only binary splits can occur.
        NOTE: Sampling from the density grid will ALWAYS use number-limited sampling (method 1)
    AVG_BELOW_SAMPLER: bool, optional
        When switched on, an integral is performed in each cell between the minimum source mass and SAMPLER_MIN_MASS,
        effectively placing the average halo population in each HaloBox cell below the sampler resolution.
        When switched off, all halos below SAMPLER_MIN_MASS are ignored. This flag saves memory for larger boxes,
        while still including the effects of smaller sources, albeit without stochasticity.
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
    """

    _hmf_models = ["PS", "ST", "WATSON", "WATSON-Z", "DELOS"]
    _power_models = ["EH", "BBKS", "EFSTATHIOU", "PEEBLES", "WHITE", "CLASS"]
    _sample_methods = ["MASS-LIMITED", "NUMBER-LIMITED", "PARTITION", "BINARY-SPLIT"]
    _integral_methods = ["GSL-QAG", "GAUSS-LEGENDRE", "GAMMA-APPROX"]

    BOX_LEN = field(default=300.0, converter=float, validator=validators.gt(0))
    HII_DIM = field(default=200, converter=int, validator=validators.gt(0))
    DIM = field(converter=int)
    NON_CUBIC_FACTOR = field(default=1.0, converter=float, validator=validators.gt(0))
    USE_FFTW_WISDOM = field(default=False, converter=bool)
    HMF = field(
        default="ST",
        converter=str,
        validator=validators.in_(_hmf_models),
        transformer=choice_transformer(_hmf_models),
    )
    USE_RELATIVE_VELOCITIES = field(default=False, converter=bool)
    POWER_SPECTRUM = field(
        converter=str,
        validator=validators.in_(_power_models),
        transformer=choice_transformer(_power_models),
    )
    N_THREADS = field(default=1, converter=int, validator=validators.gt(0))
    PERTURB_ON_HIGH_RES = field(default=False, converter=bool)
    NO_RNG = field(default=False, converter=bool)
    USE_INTERPOLATION_TABLES = field(default=True, converter=bool)
    INTEGRATION_METHOD_ATOMIC = field(
        default="GAUSS-LEGENDRE",
        converter=str,
        validator=validators.in_(_integral_methods),
        transformer=choice_transformer(_integral_methods),
    )
    INTEGRATION_METHOD_MINI = field(
        default="GAUSS-LEGENDRE",
        converter=str,
        validator=validators.in_(_integral_methods),
        transformer=choice_transformer(_integral_methods),
    )
    USE_2LPT = field(default=True, converter=bool)
    MINIMIZE_MEMORY = field(default=False, converter=bool)
    KEEP_3D_VELOCITIES = field(default=False, converter=bool)
    SAMPLER_MIN_MASS = field(default=1e8, converter=float, validator=validators.gt(0))
    SAMPLER_BUFFER_FACTOR = field(default=2.0, converter=float)
    MAXHALO_FACTOR = field(default=2.0, converter=float)
    N_COND_INTERP = field(default=200, converter=int)
    N_PROB_INTERP = field(default=400, converter=int)
    MIN_LOGPROB = field(default=-12, converter=float)
    SAMPLE_METHOD = field(
        default="MASS-LIMITED",
        validator=validators.in_(_sample_methods),
        transformer=choice_transformer(_sample_methods),
    )
    AVG_BELOW_SAMPLER = field(default=True, converter=bool)
    HALOMASS_CORRECTION = field(
        default=0.9, converter=float, validator=validators.gt(0)
    )
    PARKINSON_G0 = field(default=1.0, converter=float, validator=validators.gt(0))
    PARKINSON_y1 = field(default=0.0, converter=float)
    PARKINSON_y2 = field(default=0.0, converter=float)

    @DIM.default
    def _dim_default(self):
        """Number of cells for the high-res box (sampling ICs) along a principal axis."""
        return 3 * self.HII_DIM

    @NON_CUBIC_FACTOR.validator
    def _NON_CUBIC_FACTOR_validator(self, att, val):
        """Factor to shorten/lengthen the line-of-sight dimension (non-cubic boxes)."""
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

    @POWER_SPECTRUM.default
    def _ps_default(self):
        return "CLASS" if self.USE_RELATIVE_VELOCITIES else "EH"

    @POWER_SPECTRUM.validator
    def _POWER_SPECTRUM_vld(self, att, val):
        """
        The power spectrum generator to use, as an integer.

        See :func:`power_spectrum_model` for a string representation.
        """
        if self.USE_RELATIVE_VELOCITIES and self.POWER_SPECTRUM != "CLASS":
            raise ValueError(
                "Can only use 'CLASS' power spectrum with relative velocities"
            )

    @property
    def cell_size(self) -> un.Quantity[un.Mpc]:
        """The resolution of a low-res cell."""
        return (self.BOX_LEN / self.HII_DIM) * un.Mpc

    @property
    def cell_size_hires(self) -> un.Quantity[un.Mpc]:
        """The resolution of a hi-res cell."""
        return (self.BOX_LEN / self.DIM) * un.Mpc


@define(frozen=True, kw_only=True)
class FlagOptions(InputStruct):
    """
    Flag-style options for the ionization routines.

    Note that all flags are set to False by default, giving the simplest "vanilla"
    version of 21cmFAST.

    Parameters
    ----------
    USE_HALO_FIELD : bool, optional
        Set to True if intending to find and use the halo field. If False, uses
        the mean collapse fraction (which is considerably faster).
    USE_MINI_HALOS : bool, optional
        Set to True if using mini-halos parameterization.
        If True, USE_MASS_DEPENDENT_ZETA and INHOMO_RECO must be True.
    USE_CMB_HEATING : bool, optional
        Whether to include CMB Heating. (cf Eq.4 of Meiksin 2021, arxiv.org/abs/2105.14516)
    USE_LYA_HEATING : bool, optional
        Whether to use Lyman-alpha heating. (cf Sec. 3 of Reis+2021, doi.org/10.1093/mnras/stab2089)
    USE_MASS_DEPENDENT_ZETA : bool, optional
        Set to True if using new parameterization. Setting to True will automatically
        set `M_MIN_in_Mass` to True.
    SUBCELL_RSDS : bool, optional
        Add sub-cell redshift-space-distortions (cf Sec 2.2 of Greig+2018).
        Will only be effective if `USE_TS_FLUCT` is True.
    INHOMO_RECO : bool, optional
        Whether to perform inhomogeneous recombinations. Increases the computation
        time.
    USE_TS_FLUCT : bool, optional
        Whether to perform IGM spin temperature fluctuations (i.e. X-ray heating).
        Dramatically increases the computation time.
    M_MIN_in_Mass : bool, optional
        Whether the minimum halo mass (for ionization) is defined by
        mass or virial temperature. Automatically True if `USE_MASS_DEPENDENT_ZETA`
        is True.
    PHOTON_CONS_TYPE : int, optional
        Whether to perform a small correction to account for the inherent
        photon non-conservation. This can be one of three types of correction:

        0: No photon cosnervation correction,
        1: Photon conservation correction by adjusting the redshift of the N_ion source field (Park+22)
        2: Adjustment to the escape fraction power-law slope, based on fiducial results in Park+22, This runs a
        series of global xH evolutions and one calibration simulation to find the adjustment as a function of xH
        3: Adjustment to the escape fraction normalisation, runs one calibration simulation to find the
        adjustment as a function of xH where f'/f = xH_global/xH_calibration
    FIX_VCB_AVG: bool, optional
        Determines whether to use a fixed vcb=VAVG (*regardless* of USE_RELATIVE_VELOCITIES). It includes the average effect of velocities but not its fluctuations. See Muñoz+21 (2110.13919).
    HALO_STOCHASTICITY: bool, optional
        Sample the Conditional Halo Mass Function and sum over the sample instead of integrating it.
        This allows us to include stochasticity in other properties
    USE_EXP_FILTER: bool, optional
        Use the exponential filter (MFP-epsilon(r) from Davies & Furlanetto 2021) when calculating ionising emissivity fields
        NOTE: this does not affect other field filters, and should probably be used with HII_FILTER==0 (real-space top-hat)
    FIXED_HALO_GRIDS: bool, optional
        When USE_HALO_FIELD is True, this flag bypasses the sampler, and calculates fixed grids of halo mass, stellar mass etc
        analagous to FFRT-P (Davies & Furlanetto 2021) or ESF-E (Trac et al 2021), This flag has no effect is USE_HALO_FIELD is False
        With USE_HALO_FIELD: (FIXED_HALO_GRIDS,HALO_STOCHASTICITY):

        (0,0): DexM only,
        (0,1): Halo Sampler,
        (1,?): FFRT-P fixed halo grids
    CELL_RECOMB: bool, optional
        An alternate way of counting recombinations based on the local cell rather than the filter region.
        This is part of the perspective shift (see Davies & Furlanetto 2021) from counting photons/atoms in a sphere and flagging a central
        pixel to counting photons which we expect to reach the central pixel, and taking the ratio of atoms in the pixel.
        This flag simply turns off the filtering of N_rec grids, and takes the recombinations in the central cell.
    USE_UPPER_STELLAR_TURNOVER: bool, optional
        Whether to use an additional powerlaw in stellar mass fraction at high halo mass. The pivot mass scale and power-law index are
        controlled by two global parameters, UPPER_STELLAR_TURNOVER_MASS and UPPER_STELLAR_TURNOVER_INDEX respectively.
        This is currently only implemented in the halo model (USE_HALO_FIELD=True), and has no effect otherwise.
    HALO_SCALING_RELATIONS_MEDIAN: bool, optional
        If True, halo scaling relation parameters (F_STAR10,t_STAR etc...) define the median of their conditional distributions
        If False, they describe the mean.
        This becomes important when using non-symmetric dristributions such as the log-normal
    """

    _photoncons_models = [
        "no-photoncons",
        "z-photoncons",
        "alpha-photoncons",
        "f-photoncons",
    ]

    USE_MINI_HALOS = field(default=False, converter=bool)
    USE_CMB_HEATING = field(default=True, converter=bool)
    USE_LYA_HEATING = field(default=True, converter=bool)
    USE_MASS_DEPENDENT_ZETA = field(default=True, converter=bool)
    USE_HALO_FIELD = field(default=True, converter=bool)
    APPLY_RSDS = field(default=True, converter=bool)
    SUBCELL_RSD = field(default=False, converter=bool)
    INHOMO_RECO = field(default=False, converter=bool)
    USE_TS_FLUCT = field(default=False, converter=bool)
    FIX_VCB_AVG = field(default=False, converter=bool)
    HALO_STOCHASTICITY = field(default=True, converter=bool)
    USE_EXP_FILTER = field(default=True, converter=bool)
    FIXED_HALO_GRIDS = field(default=False, converter=bool)
    CELL_RECOMB = field(default=True, converter=bool)
    PHOTON_CONS_TYPE = field(
        default="no-photoncons",
        converter=str,
        validator=validators.in_(_photoncons_models),
        transformer=choice_transformer(_photoncons_models),
    )
    USE_UPPER_STELLAR_TURNOVER = field(default=True, converter=bool)
    M_MIN_in_Mass = field(default=True, converter=bool)
    HALO_SCALING_RELATIONS_MEDIAN = field(default=False, converter=bool)

    @M_MIN_in_Mass.validator
    def _M_MIN_in_Mass_vld(self, att, val):
        """M_MIN_in_Mass must be true if USE_MASS_DEPENDENT_ZETA is true."""
        if not val and self.USE_MASS_DEPENDENT_ZETA:
            raise ValueError(
                "M_MIN_in_Mass must be true if USE_MASS_DEPENDENT_ZETA is true."
            )

    @SUBCELL_RSD.validator
    def _SUBCELL_RSD_vld(self, att, val):
        """The SUBCELL_RSD flag is only effective if APPLY_RSDS is True."""
        if val and not self.APPLY_RSDS:
            raise ValueError(
                "The SUBCELL_RSD flag is only effective if APPLY_RSDS is True."
            )

    @USE_HALO_FIELD.validator
    def _USE_HALO_FIELD_vld(self, att, val):
        """Raise an error if USE_HALO_FIELD is True and USE_MASS_DEPENDENT_ZETA is False."""
        if val and not self.USE_MASS_DEPENDENT_ZETA:
            raise ValueError(
                "You have set USE_MASS_DEPENDENT_ZETA to False but USE_HALO_FIELD is True! "
            )

    @USE_MINI_HALOS.validator
    def _USE_MINI_HALOS_vald(self, att, val):
        """
        Raise an error USE_MINI_HALOS is True with incompatible flags.

        This happens when anyof of USE_MASS_DEPENDENT_ZETA, INHOMO_RECO,
        or USE_TS_FLUCT is False.
        """
        if val and not self.USE_MASS_DEPENDENT_ZETA:
            raise ValueError(
                "You have set USE_MINI_HALOS to True but USE_MASS_DEPENDENT_ZETA is False! "
            )
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
        if (self.USE_MINI_HALOS or self.USE_HALO_FIELD) and val == "z-photoncons":
            raise ValueError(
                "USE_MINI_HALOS and USE_HALO_FIELD are not compatible with the redshift-based"
                " photon conservation corrections (PHOTON_CONS_TYPE=='z_photoncons')! "
            )

    @HALO_STOCHASTICITY.validator
    def _HALO_STOCHASTICITY_vld(self, att, val):
        """Raise an error if HALO_STOCHASTICITY is True and USE_HALO_FIELD is False."""
        if val and not self.USE_HALO_FIELD:
            raise ValueError("HALO_STOCHASTICITY is True but USE_HALO_FIELD is False")

    @USE_EXP_FILTER.validator
    def _USE_EXP_FILTER_vld(self, att, val):
        """Raise an error if USE_EXP_FILTER is False and HII_FILTER!=0."""
        if val and global_params.HII_FILTER != 0:
            raise ValueError(
                "USE_EXP_FILTER can only be used with a real-space tophat HII_FILTER==0"
            )

        if val and not self.CELL_RECOMB:
            raise ValueError("USE_EXP_FILTER is True but CELL_RECOMB is False")


@define(frozen=True, kw_only=True)
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
        i.e. when `USE_MASS_DEPENDENT_ZETA` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_ESC10` to determine `HII_EFF_FACTOR` (which
        is then unused). See Eq. 11 of Greig+2018 and Sec 2.1 of Park+2018.
        Given in log10 units.
    F_STAR7_MINI : float, optional
        The fraction of galactic gas in stars for 10^7 solar mass minihaloes.
        Only used in the "minihalo" parameterization,
        i.e. when `USE_MINI_HALOS` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_ESC7_MINI` to determine `HII_EFF_FACTOR_MINI` (which
        is then unused). See Eq. 8 of Qin+2020.
        Given in log10 units.
    ALPHA_STAR : float, optional
        Power-law index of fraction of galactic gas in stars as a function of halo mass.
        See Sec 2.1 of Park+2018.
    ALPHA_STAR_MINI : float, optional
        Power-law index of fraction of galactic gas in stars as a function of halo mass, for MCGs.
        See Sec 2 of Muñoz+21 (2110.13919).
    SIGMA_STAR : float, optional
        Lognormal scatter (dex) of the halo mass to stellar mass relation.
        Uniform across all masses and redshifts.
    CORR_STAR : float, optional
        Self-correlation length used for updating halo properties. To model the correlation in the SHMR
        between timesteps, we take two samples, one completely correlated (at the exact same percentile in the distribution)
        and one completely uncorrelated. We interpolate between these two samples where
        the interpolation point in [0,1] based on this parameter and the redshift difference between timesteps,
        given by exp(-dz/CORR_STAR)
    SIGMA_SFR_LIM : float, optional
        Lognormal scatter (dex) of the stellar mass to SFR relation above a stellar mass of 1e10 solar.
    SIGMA_SFR_INDEX : float, optional
        index of the power-law between SFMS scatter and stellar mass below 1e10 solar.
    CORR_SFR : float, optional
        Self-correlation length used for updating xray luminosity, see "CORR_STAR" for details.
    F_ESC10 : float, optional
        The "escape fraction", i.e. the fraction of ionizing photons escaping into the
        IGM, for 10^10 solar mass haloes. Only used in the "new" parameterization,
        i.e. when `USE_MASS_DEPENDENT_ZETA` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_STAR10` to determine `HII_EFF_FACTOR` (which
        is then unused). See Eq. 11 of Greig+2018 and Sec 2.1 of Park+2018.
    F_ESC7_MINI: float, optional
        The "escape fraction for minihalos", i.e. the fraction of ionizing photons escaping
        into the IGM, for 10^7 solar mass minihaloes. Only used in the "minihalo" parameterization,
        i.e. when `USE_MINI_HALOS` is set to True (in :class:`FlagOptions`).
        If so, this is used along with `F_ESC7_MINI` to determine `HII_EFF_FACTOR_MINI` (which
        is then unused). See Eq. 17 of Qin+2020.
        Given in log10 units.
    ALPHA_ESC : float, optional
        Power-law index of escape fraction as a function of halo mass. See Sec 2.1 of
        Park+2018.
    M_TURN : float, optional
        Turnover mass (in log10 solar mass units) for quenching of star formation in
        halos, due to SNe or photo-heating feedback, or inefficient gas accretion. Only
        used if `USE_MASS_DEPENDENT_ZETA` is set to True in :class:`FlagOptions`.
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
        minihalos. Cf. Eq. 23 of Qin+2020. Given in log10 units. For the double power-law used in the Halo Model
        This gives the low-z limite.
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
        star-formation rate of galaxies. Only used if `USE_MASS_DEPENDENT_ZETA` is set
        to True in :class:`FlagOptions`. See Sec 2.1, Eq. 3 of Park+2018.
    N_RSD_STEPS : int, optional
        Number of steps used in redshift-space-distortion algorithm. NOT A PHYSICAL
        PARAMETER.
    A_LW, BETA_LW: float, optional
        Impact of the LW feedback on Mturn for minihaloes. Default is 22.8685 and 0.47 following Machacek+01, respectively. Latest simulations suggest 2.0 and 0.6. See Sec 2 of Muñoz+21 (2110.13919).
    A_VCB, BETA_VCB: float, optional
        Impact of the DM-baryon relative velocities on Mturn for minihaloes. Default is 1.0 and 1.8, and agrees between different sims. See Sec 2 of Muñoz+21 (2110.13919).
    UPPER_STELLAR_TURNOVER_MASS:
        The pivot mass associated with the optional upper mass power-law of the stellar-halo mass relation
        (see FlagOptions.USE_UPPER_STELLAR_TURNOVER)
    UPPER_STELLAR_TURNOVER_INDEX:
        The power-law index associated with the optional upper mass power-law of the stellar-halo mass relation
        (see FlagOptions.USE_UPPER_STELLAR_TURNOVER)
    SIGMA_LX: float, optional
        Lognormal scatter (dex) of the Xray luminosity relation (a function of stellar mass, star formation rate and redshift).
        This scatter is uniform across all halo properties and redshifts.
    CORR_LX : float, optional
        Self-correlation length used for updating xray luminosity, see "CORR_STAR" for details.
    """

    HII_EFF_FACTOR = field(default=30.0, converter=float, validator=validators.gt(0))
    F_STAR10 = field(
        default=-1.3,
        converter=float,
        validator=between(-3.0, 0.0),
        transformer=logtransformer,
    )
    ALPHA_STAR = field(
        default=0.5,
        converter=float,
    )
    F_STAR7_MINI = field(converter=float, transformer=logtransformer)
    ALPHA_STAR_MINI = field(converter=float)
    F_ESC10 = field(
        default=-1.0,
        converter=float,
        transformer=logtransformer,
    )
    ALPHA_ESC = field(
        default=-0.5,
        converter=float,
    )
    F_ESC7_MINI = field(
        converter=float,
        transformer=logtransformer,
    )
    M_TURN = field(
        default=8.7,
        converter=float,
        validator=validators.gt(0),
        transformer=logtransformer,
    )
    R_BUBBLE_MAX = field(default=15.,converter=float, validator=validators.gt(0))
    ION_Tvir_MIN = field(
        default=4.69897,
        converter=float,
        validator=validators.gt(0),
        transformer=logtransformer,
    )
    L_X = field(
        default=40.5,
        converter=float,
        validator=validators.gt(0),
        transformer=logtransformer,
    )
    L_X_MINI = field(
        converter=float, validator=validators.gt(0), transformer=logtransformer
    )
    NU_X_THRESH = field(default=500.0, converter=float, validator=validators.gt(0))
    X_RAY_SPEC_INDEX = field(default=1.0, converter=float)
    X_RAY_Tvir_MIN = field(
        converter=float, validator=validators.gt(0), transformer=logtransformer
    )
    F_H2_SHIELD = field(default=0.0, converter=float)
    t_STAR = field(default=0.5, converter=float, validator=between(0, 1))
    N_RSD_STEPS = field(default=20, converter=int, validator=validators.gt(0))
    A_LW = field(default=2.0, converter=float, validator=validators.gt(0))
    BETA_LW = field(default=0.6, converter=float)
    A_VCB = field(default=1.0, converter=float)
    BETA_VCB = field(default=1.8, converter=float)
    UPPER_STELLAR_TURNOVER_MASS = field(
        default=11.447, converter=float, transformer=logtransformer
    )
    UPPER_STELLAR_TURNOVER_INDEX = field(default=-0.6, converter=float)
    SIGMA_STAR = field(default=0.25, converter=float, transformer=dex2exp_transformer)
    SIGMA_LX = field(default=0.5, converter=float, transformer=dex2exp_transformer)
    SIGMA_SFR_LIM = field(
        default=0.19, converter=float, transformer=dex2exp_transformer
    )
    SIGMA_SFR_INDEX = field(default=-0.12, converter=float)
    CORR_STAR = field(default=0.5, converter=float)
    CORR_SFR = field(default=0.2, converter=float)
    # NOTE (Jdavies): It's difficult to know what this should be, ASTRID doesn't have
    # the xrays and I don't know which hydros do
    CORR_LX = field(default=0.2, converter=float)

    # set the default of the minihalo scalings to continue the same PL
    @F_STAR7_MINI.default
    def _F_STAR7_MINI_default(self):
        """
        The stellar-to-halo mass ratio at 1e7 Solar Masses for Molecularly cooled galaxies.

        If the MCG scaling relations are not provided, we extend the ACG ones
        """
        return self.F_STAR10 - 3 * self.ALPHA_STAR  # -3*alpha since 1e7/1e10 = 1e-3

    # NOTE: Currently the default is not `None`, so this would normally do nothing.
    #   We need to examine the MCG/ACG connection to popII/popIII stars and
    #   discuss what this model should be.
    @F_ESC7_MINI.default
    def _F_ESC7_MINI_default(self):
        """The stellar-to-halo mass ratio at 1e7 Solar Masses for Molecularly cooled galaxies."""
        return self.F_ESC10 - 3 * self.ALPHA_ESC  # -3*alpha since 1e7/1e10 = 1e-3

    @ALPHA_STAR_MINI.default
    def _ALPHA_STAR_MINI_default(self):
        """
        The power law index of the SHMR for Molecularly cooled galaxies.

        If the MCG scaling relations are not provided, we extend the ACG ones
        """
        return self.ALPHA_STAR

    @L_X_MINI.default
    def _L_X_MINI_default(self):
        """
        The Lx/SFR normalisation for Molecularly cooled galaxies.

        If the MCG scaling relations are not provided, we extend the ACG ones
        """
        return self.L_X

    @X_RAY_Tvir_MIN.default
    def _X_RAY_Tvir_MIN_default(self):
        """Minimum virial temperature of X-ray emitting sources (unlogged and set dynamically)."""
        return self.ION_Tvir_MIN

    @NU_X_THRESH.validator
    def _NU_X_THRESH_vld(self, att, val):
        """Check if the choice of NU_X_THRESH is sensible."""
        if val < 100.0:
            raise ValueError(
                "Chosen NU_X_THRESH is < 100 eV. NU_X_THRESH must be above 100 eV as it describes X-ray photons"
            )
        elif val >= global_params.NU_X_BAND_MAX:
            raise ValueError(
                f"""
                Chosen NU_X_THRESH > {global_params.NU_X_BAND_MAX}, which is the upper limit of the adopted X-ray band
                (fiducially the soft band 0.5 - 2.0 keV). If you know what you are doing with this
                choice, please modify the global parameter: NU_X_BAND_MAX"""
            )
        elif global_params.NU_X_BAND_MAX > global_params.NU_X_MAX:
            raise ValueError(
                f"""
                Chosen NU_X_BAND_MAX > {global_params.NU_X_MAX}, which is the upper limit of X-ray integrals (fiducially 10 keV)
                If you know what you are doing, please modify the global parameter:
                NU_X_MAX
                """
            )
