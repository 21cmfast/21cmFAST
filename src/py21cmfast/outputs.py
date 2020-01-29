"""
Output class objects.

The classes provided by this module exist to simplify access to large datasets created within C.
Fundamentally, ownership of the data belongs to these classes, and the C functions merely accesses
this and fills it. The various boxes and quantities associated with each output are available as
instance attributes. Along with the output data, each output object contains the various input
parameter objects necessary to define it.

.. warning:: These should not be instantiated or filled by the user, but always handled
             as output objects from the various functions contained here. Only the data
             within the objects should be accessed.
"""
import numpy as np
from cached_property import cached_property

from ._utils import OutputStruct as _BaseOutputStruct
from .c_21cmfast import ffi
from .inputs import AstroParams
from .inputs import CosmoParams
from .inputs import FlagOptions
from .inputs import UserParams
from .inputs import global_params


class _OutputStruct(_BaseOutputStruct):
    _global_params = global_params

    def __init__(self, *, user_params=None, cosmo_params=None, **kwargs):
        if cosmo_params is None:
            cosmo_params = CosmoParams()
        if user_params is None:
            user_params = UserParams()

        super().__init__(user_params=user_params, cosmo_params=cosmo_params, **kwargs)

    _ffi = ffi


class _OutputStructZ(_OutputStruct):
    _inputs = _OutputStruct._inputs + ["redshift"]


class InitialConditions(_OutputStruct):
    """A class containing all initial conditions boxes."""

    # The filter params indicates parameters to overlook when deciding if a cached box
    # matches current parameters.
    # It is useful for ignoring certain global parameters which may not apply to this
    # step or its dependents.
    _filter_params = _OutputStruct._filter_params + [
        "ALPHA_UVB",  # ionization
        "EVOLVE_DENSITY_LINEARLY",  # perturb
        "SMOOTH_EVOLVED_DENSITY_FIELD",  # perturb
        "R_smooth_density",  # perturb
        "HII_ROUND_ERR",  # ionization
        "FIND_BUBBLE_ALGORITHM",  # ib
        "N_POISSON",  # ib
        "T_USE_VELOCITIES",  # bt
        "MAX_DVDR",  # bt
        "DELTA_R_HII_FACTOR",  # ib
        "HII_FILTER",  # ib
        "INITIAL_REDSHIFT",  # pf
        "HEAT_FILTER",  # st
        "CLUMPING_FACTOR",  # st
        "Z_HEAT_MAX",  # st
        "R_XLy_MAX",  # st
        "NUM_FILTER_STEPS_FOR_Ts",  # ts
        "ZPRIME_STEP_FACTOR",  # ts
        "TK_at_Z_HEAT_MAX",  # ts
        "XION_at_Z_HEAT_MAX",  # ts
        "Pop",  # ib
        "Pop2_ion",  # ib
        "Pop3_ion",  # ib
        "NU_X_BAND_MAX",  # st
        "NU_X_MAX",  # ib
    ]

    def _init_arrays(self):
        self.lowres_density = np.zeros(
            self.user_params.HII_tot_num_pixels, dtype=np.float32
        )
        self.lowres_vx = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vy = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vz = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.lowres_vx_2LPT = np.zeros(
            self.user_params.HII_tot_num_pixels, dtype=np.float32
        )
        self.lowres_vy_2LPT = np.zeros(
            self.user_params.HII_tot_num_pixels, dtype=np.float32
        )
        self.lowres_vz_2LPT = np.zeros(
            self.user_params.HII_tot_num_pixels, dtype=np.float32
        )
        self.hires_density = np.zeros(
            self.user_params.tot_fft_num_pixels, dtype=np.float32
        )
        self.hires_vcb = np.zeros(self.user_params.tot_fft_num_pixels, dtype=np.float32)
        self.lowres_vcb = np.zeros(
            self.user_params.HII_tot_num_pixels, dtype=np.float32
        )

        shape = (
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
        )
        hires_shape = (self.user_params.DIM, self.user_params.DIM, self.user_params.DIM)

        self.lowres_density.shape = shape
        self.lowres_vx.shape = shape
        self.lowres_vy.shape = shape
        self.lowres_vz.shape = shape
        self.lowres_vx_2LPT.shape = shape
        self.lowres_vy_2LPT.shape = shape
        self.lowres_vz_2LPT.shape = shape
        self.hires_density.shape = hires_shape

        self.lowres_vcb.shape = shape
        self.hires_vcb.shape = hires_shape


class PerturbedField(_OutputStructZ):
    """A class containing all perturbed field boxes."""

    _filter_params = _OutputStruct._filter_params + [
        "ALPHA_UVB",  # ionization
        "HII_ROUND_ERR",  # ionization
        "FIND_BUBBLE_ALGORITHM",  # ib
        "N_POISSON",  # ib
        "T_USE_VELOCITIES",  # bt
        "MAX_DVDR",  # bt
        "DELTA_R_HII_FACTOR",  # ib
        "HII_FILTER",  # ib
        "HEAT_FILTER",  # st
        "CLUMPING_FACTOR",  # st
        "Z_HEAT_MAX",  # st
        "R_XLy_MAX",  # st
        "NUM_FILTER_STEPS_FOR_Ts",  # ts
        "ZPRIME_STEP_FACTOR",  # ts
        "TK_at_Z_HEAT_MAX",  # ts
        "XION_at_Z_HEAT_MAX",  # ts
        "Pop",  # ib
        "Pop2_ion",  # ib
        "Pop3_ion",  # ib
        "NU_X_BAND_MAX",  # st
        "NU_X_MAX",  # ib
    ]

    def _init_arrays(self):
        self.density = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.velocity = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)

        self.density.shape = (
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
        )
        self.velocity.shape = (
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
        )


class _AllParamsBox(_OutputStructZ):
    _inputs = _OutputStructZ._inputs + ["flag_options", "astro_params"]

    _filter_params = _OutputStruct._filter_params + [
        "T_USE_VELOCITIES",  # bt
        "MAX_DVDR",  # bt
    ]

    def __init__(self, astro_params=None, flag_options=None, first_box=False, **kwargs):
        if flag_options is None:
            flag_options = FlagOptions()

        if astro_params is None:
            astro_params = AstroParams(INHOMO_RECO=flag_options.INHOMO_RECO)

        self.first_box = first_box

        super().__init__(astro_params=astro_params, flag_options=flag_options, **kwargs)


class IonizedBox(_AllParamsBox):
    """A class containing all ionized boxes."""

    def _init_arrays(self):
        # ionized_box is always initialised to be neutral for excursion set algorithm.
        # Hence np.ones instead of np.zeros
        self.xH_box = np.ones(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.Gamma12_box = np.zeros(
            self.user_params.HII_tot_num_pixels, dtype=np.float32
        )
        self.z_re_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.dNrec_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)

        shape = (
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
        )
        self.xH_box.shape = shape
        self.Gamma12_box.shape = shape
        self.z_re_box.shape = shape
        self.dNrec_box.shape = shape

    @cached_property
    def global_xH(self):
        """Global (mean) neutral fraction."""
        if not self.filled:
            raise AttributeError(
                "global_xH is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.xH_box)


class TsBox(_AllParamsBox):
    """A class containing all spin temperature boxes."""

    def _init_arrays(self):
        self.Ts_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.x_e_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)
        self.Tk_box = np.zeros(self.user_params.HII_tot_num_pixels, dtype=np.float32)

        self.Ts_box.shape = (
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
        )
        self.x_e_box.shape = (
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
        )
        self.Tk_box.shape = (
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
        )

    @cached_property
    def global_Ts(self):
        """Global (mean) spin temperature."""
        if not self.filled:
            raise AttributeError(
                "global_Ts is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.Ts_box)

    @cached_property
    def global_Tk(self):
        """Global (mean) Tk."""
        if not self.filled:
            raise AttributeError(
                "global_Tk is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.Tk_box)

    @cached_property
    def global_x_e(self):
        """Global (mean) x_e."""
        if not self.filled:
            raise AttributeError(
                "global_x_e is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.x_e_box)


class BrightnessTemp(_AllParamsBox):
    """A class containing the brightness temperature box."""

    _filter_params = _OutputStructZ._filter_params

    def _init_arrays(self):
        self.brightness_temp = np.zeros(
            self.user_params.HII_tot_num_pixels, dtype=np.float32
        )

        self.brightness_temp.shape = (
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
            self.user_params.HII_DIM,
        )

    @cached_property
    def global_Tb(self):
        """Global (mean) brightness temperature."""
        if not self.filled:
            raise AttributeError(
                "global_Tb is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.brightness_temp)
