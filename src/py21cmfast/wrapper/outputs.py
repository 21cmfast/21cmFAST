"""
Output class objects.

The classes provided by this module exist to simplify access to large datasets created within C.
Fundamentally, ownership of the data belongs to these classes, and the C functions merely accesses
this and fills it. The various boxes and lightcones associated with each output are available as
instance attributes. Along with the output data, each output object contains the various input
parameter objects necessary to define it.

.. warning:: These should not be instantiated or filled by the user, but always handled
             as output objects from the various functions contained here. Only the data
             within the objects should be accessed.
"""

from __future__ import annotations

import logging
import numpy as np
from cached_property import cached_property

from .. import __version__
from ._utils import OutputStruct as _BaseOutputStruct
from .c_21cmfast import ffi, lib
from .inputs import AstroParams, CosmoParams, FlagOptions, UserParams, global_params

logger = logging.getLogger(__name__)


class _OutputStruct(_BaseOutputStruct):
    _global_params = global_params

    def __init__(self, *, user_params=None, cosmo_params=None, **kwargs):
        self.cosmo_params = cosmo_params or CosmoParams()
        self.user_params = user_params or UserParams()

        super().__init__(**kwargs)

    _ffi = ffi


class _OutputStructZ(_OutputStruct):
    _inputs = _OutputStruct._inputs + ("redshift",)


class InitialConditions(_OutputStruct):
    """A class containing all initial conditions boxes."""

    _c_compute_function = lib.ComputeInitialConditions

    # The filter params indicates parameters to overlook when deciding if a cached box
    # matches current parameters.
    # It is useful for ignoring certain global parameters which may not apply to this
    # step or its dependents.
    _meta = False
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

    def prepare_for_perturb(self, flag_options: FlagOptions, force: bool = False):
        """Ensure the ICs have all the boxes loaded for perturb, but no extra."""
        keep = ["hires_density"]

        if flag_options.HALO_STOCHASTICITY:
            keep.append("lowres_density")

        if not self.user_params.PERTURB_ON_HIGH_RES:
            keep.append("lowres_density")
            keep.append("lowres_vx")
            keep.append("lowres_vy")
            keep.append("lowres_vz")

            if self.user_params.USE_2LPT:
                keep.append("lowres_vx_2LPT")
                keep.append("lowres_vy_2LPT")
                keep.append("lowres_vz_2LPT")

        else:
            keep.append("hires_vx")
            keep.append("hires_vy")
            keep.append("hires_vz")

            if self.user_params.USE_2LPT:
                keep.append("hires_vx_2LPT")
                keep.append("hires_vy_2LPT")
                keep.append("hires_vz_2LPT")

        if self.user_params.USE_RELATIVE_VELOCITIES:
            keep.append("lowres_vcb")

        self.prepare(keep=keep, force=force)

    def prepare_for_halos(self, flag_options: FlagOptions, force: bool = False):
        """Ensure ICs have all boxes required for the halos, and no more."""
        keep = ["hires_density"]  # for dexm
        if flag_options.HALO_STOCHASTICITY:
            keep.append("lowres_density")  # for the sampler
        if self.user_params.USE_RELATIVE_VELOCITIES:
            keep.append("lowres_vcb")
        self.prepare(keep=keep, force=force)

    def prepare_for_spin_temp(self, flag_options: FlagOptions, force: bool = False):
        """Ensure ICs have all boxes required for spin_temp, and no more."""
        keep = []
        if self.user_params.USE_RELATIVE_VELOCITIES:
            keep.append("lowres_vcb")
        self.prepare(keep=keep, force=force)

    def _get_box_structures(self) -> dict[str, dict | tuple[int]]:
        shape = (self.user_params.HII_DIM,) * 2 + (
            int(self.user_params.NON_CUBIC_FACTOR * self.user_params.HII_DIM),
        )
        hires_shape = (self.user_params.DIM,) * 2 + (
            int(self.user_params.NON_CUBIC_FACTOR * self.user_params.DIM),
        )

        out = {
            "lowres_density": shape,
            "lowres_vx": shape,
            "lowres_vy": shape,
            "lowres_vz": shape,
            "hires_density": hires_shape,
            "hires_vx": hires_shape,
            "hires_vy": hires_shape,
            "hires_vz": hires_shape,
        }

        if self.user_params.USE_2LPT:
            out.update(
                {
                    "lowres_vx_2LPT": shape,
                    "lowres_vy_2LPT": shape,
                    "lowres_vz_2LPT": shape,
                    "hires_vx_2LPT": hires_shape,
                    "hires_vy_2LPT": hires_shape,
                    "hires_vz_2LPT": hires_shape,
                }
            )

        if self.user_params.USE_RELATIVE_VELOCITIES:
            out.update({"lowres_vcb": shape})

        return out

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        return []

    def compute(self, hooks: dict):
        """Compute the function."""
        return self._compute(
            self.random_seed,
            self.user_params,
            self.cosmo_params,
            hooks=hooks,
        )


class PerturbedField(_OutputStructZ):
    """A class containing all perturbed field boxes."""

    _c_compute_function = lib.ComputePerturbField

    _meta = False
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

    def _get_box_structures(self) -> dict[str, dict | tuple[int]]:
        out = {
            "density": (self.user_params.HII_DIM,) * 2
            + (int(self.user_params.NON_CUBIC_FACTOR * self.user_params.HII_DIM),),
            "velocity_z": (self.user_params.HII_DIM,) * 2
            + (int(self.user_params.NON_CUBIC_FACTOR * self.user_params.HII_DIM),),
        }
        if self.user_params.KEEP_3D_VELOCITIES:
            out["velocity_x"] = out["density"]
            out["velocity_y"] = out["density"]

        return out

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []

        if not isinstance(input_box, InitialConditions):
            raise ValueError(
                f"{type(input_box)} is not an input required for PerturbedField!"
            )

        # Always require hires_density
        required += ["hires_density"]

        if self.user_params.PERTURB_ON_HIGH_RES:
            required += ["hires_vx", "hires_vy", "hires_vz"]

            if self.user_params.USE_2LPT:
                required += ["hires_vx_2LPT", "hires_vy_2LPT", "hires_vz_2LPT"]

        else:
            required += ["lowres_density", "lowres_vx", "lowres_vy", "lowres_vz"]

            if self.user_params.USE_2LPT:
                required += [
                    "lowres_vx_2LPT",
                    "lowres_vy_2LPT",
                    "lowres_vz_2LPT",
                ]

        if self.user_params.USE_RELATIVE_VELOCITIES:
            required.append("lowres_vcb")

        return required

    def compute(self, *, ics: InitialConditions, hooks: dict):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.user_params,
            self.cosmo_params,
            ics,
            hooks=hooks,
        )

    @property
    def velocity(self):
        """The velocity of the box in the 3rd dimension (for backwards compat)."""
        return self.velocity_z  # for backwards compatibility


class _AllParamsBox(_OutputStructZ):
    _meta = True
    _inputs = _OutputStructZ._inputs + ("flag_options", "astro_params")

    _filter_params = _OutputStruct._filter_params + [
        "T_USE_VELOCITIES",  # bt
        "MAX_DVDR",  # bt
    ]

    def __init__(
        self,
        *,
        astro_params: AstroParams | None = None,
        flag_options: FlagOptions | None = None,
        **kwargs,
    ):
        self.flag_options = flag_options or FlagOptions()
        self.astro_params = astro_params or AstroParams(
            INHOMO_RECO=self.flag_options.INHOMO_RECO
        )

        self.log10_Mturnover_ave = 0.0
        self.log10_Mturnover_MINI_ave = 0.0

        super().__init__(**kwargs)


class HaloField(_AllParamsBox):
    """A class containing all fields related to halos."""

    _meta = False
    _inputs = _AllParamsBox._inputs + (
        "desc_redshift",
        "buffer_size",
    )
    _c_compute_function = lib.ComputeHaloField

    def __init__(
        self,
        *,
        desc_redshift: float | None = None,
        buffer_size: int = 0,
        **kwargs,
    ):
        self.desc_redshift = desc_redshift
        self.buffer_size = buffer_size

        super().__init__(**kwargs)

    def _get_box_structures(self) -> dict[str, dict | tuple[int]]:
        out = {
            "halo_masses": (self.buffer_size,),
            "star_rng": (self.buffer_size,),
            "sfr_rng": (self.buffer_size,),
            "xray_rng": (self.buffer_size,),
            "halo_coords": (self.buffer_size, 3),
        }

        return out

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if self.flag_options.HALO_STOCHASTICITY:
                # when the sampler is on, the grids are only needed for the first sample
                if self.desc_redshift < 0:
                    required += ["hires_density"]
                    required += ["lowres_density"]
            # without the sampler, dexm needs the hires density at each redshift
            else:
                required += ["hires_density"]
        elif isinstance(input_box, HaloField):
            required += ["halo_masses", "halo_coords", "star_rng", "sfr_rng"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for HaloField!"
            )
        return required

    def compute(
        self, *, descendant_halos: HaloField, ics: InitialConditions, hooks: dict
    ):
        """Compute the function."""
        return self._compute(
            self.desc_redshift,
            self.redshift,
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            ics,
            ics.random_seed,
            descendant_halos,
            hooks=hooks,
        )


class PerturbHaloField(_AllParamsBox):
    """A class containing all fields related to halos."""

    _c_compute_function = lib.ComputePerturbHaloField
    _meta = False
    _inputs = _AllParamsBox._inputs + ("buffer_size",)

    def __init__(
        self,
        buffer_size: int = 0.0,
        **kwargs,
    ):
        self.buffer_size = buffer_size
        super().__init__(**kwargs)

    def _get_box_structures(self) -> dict[str, dict | tuple[int]]:
        out = {
            "halo_masses": (self.buffer_size,),
            "star_rng": (self.buffer_size,),
            "sfr_rng": (self.buffer_size,),
            "xray_rng": (self.buffer_size,),
            "halo_coords": (self.buffer_size, 3),
        }

        return out

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if self.user_params.PERTURB_ON_HIGH_RES:
                required += ["hires_vx", "hires_vy", "hires_vz"]
            else:
                required += ["lowres_vx", "lowres_vy", "lowres_vz"]

            if self.user_params.USE_2LPT:
                required += [k + "_2LPT" for k in required]
        elif isinstance(input_box, HaloField):
            required += ["halo_coords", "halo_masses"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for PerturbHaloField!"
            )

        return required

    def compute(self, *, ics: InitialConditions, halo_field: HaloField, hooks: dict):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            ics,
            halo_field,
            hooks=hooks,
        )


class HaloBox(_AllParamsBox):
    """A class containing all gridded halo properties."""

    _meta = False
    _c_compute_function = lib.ComputeHaloBox
    _inputs = _AllParamsBox._inputs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_box_structures(self) -> dict[str, dict | tuple[int]]:
        shape = (self.user_params.HII_DIM,) * 2 + (
            int(self.user_params.NON_CUBIC_FACTOR * self.user_params.HII_DIM),
        )

        out = {
            "halo_mass": shape,
            "halo_stars": shape,
            "halo_stars_mini": shape,
            "count": shape,
            "halo_sfr": shape,
            "halo_sfr_mini": shape,
            "halo_xray": shape,
            "n_ion": shape,
            "whalo_sfr": shape,
        }

        return out

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, PerturbHaloField):
            if not self.flag_options.FIXED_HALO_GRIDS:
                required += ["halo_coords", "halo_masses", "star_rng", "sfr_rng"]
        elif isinstance(input_box, PerturbedField):
            if self.flag_options.FIXED_HALO_GRIDS or self.user_params.AVG_BELOW_SAMPLER:
                required += ["density"]
        elif isinstance(input_box, TsBox):
            required += ["J_21_LW_box"]
        elif isinstance(input_box, IonizedBox):
            required += ["Gamma12_box", "z_re_box"]
        elif isinstance(input_box, InitialConditions):
            if self.user_params.USE_RELATIVE_VELOCITIES:
                required += ["lowres_vcb"]
        else:
            raise ValueError(f"{type(input_box)} is not an input required for HaloBox!")

        return required

    def compute(
        self,
        *,
        init_boxes: InitialConditions,
        pt_halos: PerturbHaloField,
        perturbed_field: PerturbedField,
        previous_spin_temp: TsBox,
        previous_ionize_box: IonizedBox,
        hooks: dict,
    ):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            init_boxes,
            perturbed_field,
            pt_halos,
            previous_spin_temp,
            previous_ionize_box,
            hooks=hooks,
        )


class XraySourceBox(_AllParamsBox):
    """A class containing the filtered sfr grids."""

    _meta = False
    _c_compute_function = lib.UpdateXraySourceBox
    _inputs = _AllParamsBox._inputs

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_box_structures(self) -> dict[str, dict | tuple[int]]:
        shape = (
            (global_params.NUM_FILTER_STEPS_FOR_Ts,)
            + (self.user_params.HII_DIM,) * 2
            + (int(self.user_params.NON_CUBIC_FACTOR * self.user_params.HII_DIM),)
        )

        out = {
            "filtered_sfr": shape,
            "filtered_sfr_mini": shape,
            "filtered_xray": shape,
            "mean_sfr": (global_params.NUM_FILTER_STEPS_FOR_Ts,),
            "mean_sfr_mini": (global_params.NUM_FILTER_STEPS_FOR_Ts,),
            "mean_log10_Mcrit_LW": (global_params.NUM_FILTER_STEPS_FOR_Ts,),
        }

        return out

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, HaloBox):
            required += ["halo_sfr", "halo_xray"]
            if self.flag_options.USE_MINI_HALOS:
                required += ["halo_sfr_mini"]
        else:
            raise ValueError(f"{type(input_box)} is not an input required for HaloBox!")

        return required

    def compute(
        self,
        *,
        halobox: HaloBox,
        R_inner,
        R_outer,
        R_ct,
        hooks: dict,
    ):
        """Compute the function."""
        return self._compute(
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            halobox,
            R_inner,
            R_outer,
            R_ct,
            hooks=hooks,
        )


class TsBox(_AllParamsBox):
    """A class containing all spin temperature boxes."""

    _c_compute_function = lib.ComputeTsBox
    _meta = False
    _inputs = _AllParamsBox._inputs + ("prev_spin_redshift", "perturbed_field_redshift")

    def __init__(
        self,
        *,
        prev_spin_redshift: float | None = None,
        perturbed_field_redshift: float | None = None,
        **kwargs,
    ):
        self.prev_spin_redshift = prev_spin_redshift
        self.perturbed_field_redshift = perturbed_field_redshift
        super().__init__(**kwargs)

    def _get_box_structures(self) -> dict[str, dict | tuple[int]]:
        shape = (self.user_params.HII_DIM,) * 2 + (
            int(self.user_params.NON_CUBIC_FACTOR * self.user_params.HII_DIM),
        )
        return {
            "Ts_box": shape,
            "x_e_box": shape,
            "Tk_box": shape,
            "J_21_LW_box": shape,
        }

    @cached_property
    def global_Ts(self):
        """Global (mean) spin temperature."""
        if "Ts_box" not in self._computed_arrays:
            raise AttributeError(
                "global_Ts is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.Ts_box)

    @cached_property
    def global_Tk(self):
        """Global (mean) Tk."""
        if "Tk_box" not in self._computed_arrays:
            raise AttributeError(
                "global_Tk is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.Tk_box)

    @cached_property
    def global_x_e(self):
        """Global (mean) x_e."""
        if "x_e_box" not in self._computed_arrays:
            raise AttributeError(
                "global_x_e is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.x_e_box)

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if (
                self.user_params.USE_RELATIVE_VELOCITIES
                and self.flag_options.USE_MINI_HALOS
            ):
                required += ["lowres_vcb"]
        elif isinstance(input_box, PerturbedField):
            required += ["density"]
        elif isinstance(input_box, TsBox):
            required += ["Tk_box", "x_e_box", "Ts_box"]
            if self.flag_options.USE_MINI_HALOS:
                required += ["J_21_LW_box"]
        elif isinstance(input_box, XraySourceBox):
            if self.flag_options.USE_HALO_FIELD:
                required += ["filtered_sfr", "filtered_xray"]
                if self.flag_options.USE_MINI_HALOS:
                    required += ["filtered_sfr_mini"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for PerturbHaloField!"
            )

        return required

    def compute(
        self,
        *,
        cleanup: bool,
        perturbed_field: PerturbedField,
        xray_source_box: XraySourceBox,
        prev_spin_temp,
        ics: InitialConditions,
        hooks: dict,
    ):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.prev_spin_redshift,
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            self.perturbed_field_redshift,
            cleanup,
            perturbed_field,
            xray_source_box,
            prev_spin_temp,
            ics,
            hooks=hooks,
        )


class IonizedBox(_AllParamsBox):
    """A class containing all ionized boxes."""

    _meta = False
    _c_compute_function = lib.ComputeIonizedBox
    _inputs = _AllParamsBox._inputs + ("prev_ionize_redshift",)

    def __init__(self, *, prev_ionize_redshift: float | None = None, **kwargs):
        self.prev_ionize_redshift = prev_ionize_redshift
        super().__init__(**kwargs)

    def _get_box_structures(self) -> dict[str, dict | tuple[int]]:
        if self.flag_options.USE_MINI_HALOS:
            n_filtering = (
                int(
                    np.log(
                        min(
                            self.astro_params.R_BUBBLE_MAX,
                            0.620350491 * self.user_params.BOX_LEN,
                        )
                        / max(
                            global_params.R_BUBBLE_MIN,
                            0.620350491
                            * self.user_params.BOX_LEN
                            / self.user_params.HII_DIM,
                        )
                    )
                    / np.log(global_params.DELTA_R_HII_FACTOR)
                )
                + 1
            )
        else:
            n_filtering = 1

        shape = (self.user_params.HII_DIM,) * 2 + (
            int(self.user_params.NON_CUBIC_FACTOR * self.user_params.HII_DIM),
        )
        filter_shape = (n_filtering,) + shape

        out = {
            "xH_box": {"init": np.ones, "shape": shape},
            "Gamma12_box": shape,
            "MFP_box": shape,
            "z_re_box": shape,
            "dNrec_box": shape,
            "temp_kinetic_all_gas": shape,
            "Fcoll": filter_shape,
        }

        if self.flag_options.USE_MINI_HALOS:
            out["Fcoll_MINI"] = filter_shape

        return out

    @cached_property
    def global_xH(self):
        """Global (mean) neutral fraction."""
        if not self.filled:
            raise AttributeError(
                "global_xH is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.xH_box)

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if (
                self.user_params.USE_RELATIVE_VELOCITIES
                and self.flag_options.USE_MASS_DEPENDENT_ZETA
            ):
                required += ["lowres_vcb"]
        elif isinstance(input_box, PerturbedField):
            required += ["density"]
        elif isinstance(input_box, TsBox):
            required += ["J_21_LW_box", "x_e_box", "Tk_box"]
        elif isinstance(input_box, IonizedBox):
            required += ["z_re_box", "Gamma12_box"]
            if self.flag_options.INHOMO_RECO:
                required += [
                    "dNrec_box",
                ]
            if (
                self.flag_options.USE_MASS_DEPENDENT_ZETA
                and self.flag_options.USE_MINI_HALOS
            ):
                required += ["Fcoll", "Fcoll_MINI"]
        elif isinstance(input_box, HaloBox):
            required += ["n_ion", "whalo_sfr"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for IonizedBox!"
            )

        return required

    def compute(
        self,
        *,
        perturbed_field: PerturbedField,
        prev_perturbed_field: PerturbedField,
        prev_ionize_box,
        spin_temp: TsBox,
        halobox: HaloBox,
        ics: InitialConditions,
        hooks: dict,
    ):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.prev_ionize_redshift,
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            perturbed_field,
            prev_perturbed_field,
            prev_ionize_box,
            spin_temp,
            halobox,
            ics,
            hooks=hooks,
        )


class BrightnessTemp(_AllParamsBox):
    """A class containing the brightness temperature box."""

    _c_compute_function = lib.ComputeBrightnessTemp

    _meta = False
    _filter_params = _OutputStructZ._filter_params

    def _get_box_structures(self) -> dict[str, dict | tuple[int]]:
        return {
            "brightness_temp": (self.user_params.HII_DIM,) * 2
            + (int(self.user_params.NON_CUBIC_FACTOR * self.user_params.HII_DIM),)
        }

    @cached_property
    def global_Tb(self):
        """Global (mean) brightness temperature."""
        if not self.is_computed:
            raise AttributeError(
                "global_Tb is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.brightness_temp)

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, PerturbedField):
            if self.flag_options.APPLY_RSDS:
                required += ["velocity_z"]
        elif isinstance(input_box, TsBox):
            required += ["Ts_box"]
        elif isinstance(input_box, IonizedBox):
            required += ["xH_box"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for BrightnessTemp!"
            )

        return required

    def compute(
        self,
        *,
        spin_temp: TsBox,
        ionized_box: IonizedBox,
        perturbed_field: PerturbedField,
        hooks: dict,
    ):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            spin_temp,
            ionized_box,
            perturbed_field,
            hooks=hooks,
        )
