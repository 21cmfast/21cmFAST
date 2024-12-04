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

import attrs
import logging
import numpy as np
from cached_property import cached_property
from typing import Self

from .. import __version__
from ..c_21cmfast import ffi, lib
from ..drivers.param_config import InputParameters
from ..wrapper.arrays import Array
from .inputs import AstroParams, CosmoParams, FlagOptions, UserParams, global_params
from .structs import OutputStruct

logger = logging.getLogger(__name__)


def arrayfield(optional: bool = False, **kw):
    if optional:
        return attrs.field(
            validator=attrs.validators.optional(attrs.validators.instance_of(Array))
        )
    else:
        return attrs.field(validator=attrs.validators.instance_of(Array))


class InitialConditions(OutputStruct):
    """A class representing an InitialConditions C-struct."""

    _c_compute_function = lib.ComputeInitialConditions
    _meta = False

    lowres_density = arrayfield()
    lowres_vx = arrayfield()
    lowres_vy = arrayfield()
    lowres_vz = arrayfield()
    hires_density = arrayfield()
    hires_vx = arrayfield()
    hires_vy = arrayfield()
    hires_vz = arrayfield()

    lowres_vx_2LPT = arrayfield(optional=True)
    lowres_vy_2LPT = arrayfield(optional=True)
    lowres_vz_2LPT = arrayfield(optional=True)
    hires_vx_2LPT = arrayfield(optional=True)
    hires_vy_2LPT = arrayfield(optional=True)
    hires_vz_2LPT = arrayfield(optional=True)

    lowres_vcb = arrayfield(optional=True)

    @classmethod
    def new(cls, inputs: InputParameters) -> Self:
        """Create a new instance, given a set of input parameters."""

        shape = (inputs.user_params.HII_DIM,) * 2 + (
            int(inputs.user_params.NON_CUBIC_FACTOR * inputs.user_params.HII_DIM),
        )
        hires_shape = (inputs.user_params.DIM,) * 2 + (
            int(inputs.user_params.NON_CUBIC_FACTOR * inputs.user_params.DIM),
        )

        out = {
            "lowres_density": Array(shape),
            "lowres_vx": Array(shape),
            "lowres_vy": Array(shape),
            "lowres_vz": Array(shape),
            "hires_density": Array(hires_shape),
            "hires_vx": Array(hires_shape),
            "hires_vy": Array(hires_shape),
            "hires_vz": Array(hires_shape),
        }

        if inputs.user_params.USE_2LPT:
            out |= {
                "lowres_vx_2LPT": Array(shape),
                "lowres_vy_2LPT": Array(shape),
                "lowres_vz_2LPT": Array(shape),
                "hires_vx_2LPT": Array(hires_shape),
                "hires_vy_2LPT": Array(hires_shape),
                "hires_vz_2LPT": Array(hires_shape),
            }

        if inputs.user_params.USE_RELATIVE_VELOCITIES:
            out["lowres_vcb"] = Array(shape)

        return cls(inputs=inputs, **out)

    def prepare_for_perturb(self, flag_options: FlagOptions, force: bool = False):
        """Ensure the ICs have all the boxes loaded for perturb, but no extra."""
        keep = ["hires_density"]

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

    def prepare_for_spin_temp(self, flag_options: FlagOptions, force: bool = False):
        """Ensure ICs have all boxes required for spin_temp, and no more."""
        keep = []
        if flag_options.USE_HALO_FIELD and self.user_params.AVG_BELOW_SAMPLER:
            keep.append("lowres_density")  # for the cmfs
        if self.user_params.USE_RELATIVE_VELOCITIES:
            keep.append("lowres_vcb")
        self.prepare(keep=keep, force=force)

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
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


class PerturbedField(OutputStruct):
    """A class containing all perturbed field boxes."""

    _c_compute_function = lib.ComputePerturbField
    _meta = False

    density = arrayfield()
    velocity_z = arrayfield()
    velocity_x = arrayfield(optional=True)
    velocity_y = arrayfield(optional=True)

    @classmethod
    def new(cls, inputs: InputParameters) -> Self:
        dim = inputs.user_params.HII_DIM

        shape = (dim, dim, int(inputs.user_params.NON_CUBIC_FACTOR * dim))

        out = {
            "density": Array(shape),
            "velocity_z": Array(shape),
        }
        if inputs.user_params.KEEP_3D_VELOCITIES:
            out["velocity_x"] = Array(shape)
            out["velocity_y"] = Array(shape)

        return out

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
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


class PerturbHaloField(OutputStruct):
    """A class containing all fields related to halos."""

    _c_compute_function = lib.ComputePerturbHaloField
    _meta = False
    buffer_size: int = attrs.field(default=0, converter=int)

    halo_masses = arrayfield()
    star_rng = arrayfield()
    sfr_rng = arrayfield()
    xray_rng = arrayfield()
    halo_coords = arrayfield()

    @classmethod
    def new(cls, inputs: InputParameters, buffer_size: int = 0, **kw) -> Self:
        return cls(
            inputs,
            halo_masses=Array((buffer_size,)),
            star_rng=Array((buffer_size,)),
            sfr_rng=Array((buffer_size,)),
            xray_rng=Array((buffer_size,)),
            halo_coords=Array((buffer_size, 3)),
            **kw,
        )

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if self.user_params.PERTURB_ON_HIGH_RES:
                required += ["hires_vx", "hires_vy", "hires_vz"]
            else:
                required += ["lowres_vx", "lowres_vy", "lowres_vz"]

            if self.user_params.USE_2LPT:
                required += [f"{k}_2LPT" for k in required]
        elif isinstance(input_box, HaloField):
            required += [
                "halo_coords",
                "halo_masses",
                "star_rng",
                "sfr_rng",
                "xray_rng",
            ]
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


class HaloField(PerturbHaloField):
    """A class containing all fields related to halos."""

    desc_redshift: float | None = attrs.field(default=None)

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if self.flag_options.HALO_STOCHASTICITY:
                # when the sampler is on, the grids are only needed for the first sample
                if self.desc_redshift <= 0:
                    required += ["hires_density"]
                    required += ["lowres_density"]
            # without the sampler, dexm needs the hires density at each redshift
            else:
                required += ["hires_density"]
        elif isinstance(input_box, HaloField):
            if self.flag_options.HALO_STOCHASTICITY:
                required += [
                    "halo_masses",
                    "halo_coords",
                    "star_rng",
                    "sfr_rng",
                    "xray_rng",
                ]
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


class HaloBox(OutputStruct):
    """A class containing all gridded halo properties."""

    _meta = False
    _c_compute_function = lib.ComputeHaloBox

    halo_mass = arrayfield()
    halo_stars = arrayfield()
    halo_stars_mini = arrayfield()
    count = arrayfield()
    halo_sfr = arrayfield()
    halo_sfr_mini = arrayfield()
    halo_xray = arrayfield()
    n_ion = arrayfield()
    whalo_sfr = arrayfield()

    @classmethod
    def new(cls, inputs: InputParameters) -> Self:
        dim = inputs.user_params.HII_DIM
        shape = (dim, dim, int(inputs.user_params.NON_CUBIC_FACTOR * dim))

        return cls(
            inputs,
            **{
                "halo_mass": Array(shape),
                "halo_stars": Array(shape),
                "halo_stars_mini": Array(shape),
                "count": Array(shape),
                "halo_sfr": Array(shape),
                "halo_sfr_mini": Array(shape),
                "halo_xray": Array(shape),
                "n_ion": Array(shape),
                "whalo_sfr": Array(shape),
            },
        )

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, PerturbHaloField):
            if not self.flag_options.FIXED_HALO_GRIDS:
                required += [
                    "halo_coords",
                    "halo_masses",
                    "star_rng",
                    "sfr_rng",
                    "xray_rng",
                ]
        elif isinstance(input_box, PerturbedField):
            if self.flag_options.FIXED_HALO_GRIDS:
                required += ["density"]
        elif isinstance(input_box, TsBox):
            required += ["J_21_LW_box"]
        elif isinstance(input_box, IonizedBox):
            required += ["Gamma12_box", "z_re_box"]
        elif isinstance(input_box, InitialConditions):
            if (
                self.flag_options.HALO_STOCHASTICITY
                and self.user_params.AVG_BELOW_SAMPLER
            ):
                required += ["lowres_density"]
            if self.user_params.USE_RELATIVE_VELOCITIES:
                required += ["lowres_vcb"]
        else:
            raise ValueError(f"{type(input_box)} is not an input required for HaloBox!")

        return required

    def compute(
        self,
        *,
        initial_conditions: InitialConditions,
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
            initial_conditions,
            perturbed_field,
            pt_halos,
            previous_spin_temp,
            previous_ionize_box,
            hooks=hooks,
        )


class XraySourceBox(OutputStruct):
    """A class containing the filtered sfr grids."""

    _meta = False
    _c_compute_function = lib.UpdateXraySourceBox

    filtered_sfr = arrayfield()
    filtered_sfr_mini = arrayfield()
    filtered_xray = arrayfield()
    mean_sfr = arrayfield()
    mean_sfr_mini = arrayfield()
    mean_log10_Mcrit_LW = arrayfield()

    @classmethod
    def new(cls, inputs) -> Self:
        shape = (
            (global_params.NUM_FILTER_STEPS_FOR_Ts,)
            + (inputs.user_params.HII_DIM,) * 2
            + (int(inputs.user_params.NON_CUBIC_FACTOR * inputs.user_params.HII_DIM),)
        )

        return cls(
            inputs,
            filtered_sfr=Array(shape),
            filtered_sfr_mini=Array(shape),
            filtered_xray=Array(shape),
            mean_sfr=Array(shape),
            mean_sfr_mini=Array((global_params.NUM_FILTER_STEPS_FOR_Ts,)),
            mean_log10_Mcrit_LW=Array((global_params.NUM_FILTER_STEPS_FOR_Ts,)),
        )

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if not isinstance(input_box, HaloBox):
            raise ValueError(f"{type(input_box)} is not an input required for HaloBox!")

        required += ["halo_sfr", "halo_xray"]
        if self.flag_options.USE_MINI_HALOS:
            required += ["halo_sfr_mini"]
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


class TsBox(OutputStruct):
    """A class containing all spin temperature boxes."""

    _c_compute_function = lib.ComputeTsBox
    _meta = False

    prev_spin_redshift: float | None = attrs.field(default=None)

    Ts_box = arrayfield()
    x_e_box = arrayfield()
    Tk_box = arrayfield()
    J_21_LW_box = arrayfield()

    @classmethod
    def new(cls, inputs, prev_spin_redshift: float | None = None) -> Self:
        shape = (inputs.user_params.HII_DIM,) * 2 + (
            int(inputs.user_params.NON_CUBIC_FACTOR * inputs.user_params.HII_DIM),
        )
        return cls(
            inputs,
            Ts_box=Array(shape),
            x_e_box=Array(shape),
            Tk_box=Array(shape),
            J_21_LW_box=Array(shape),
            prev_spin_redshift=prev_spin_redshift,
        )

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

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
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
            perturbed_field.redshift,
            cleanup,
            perturbed_field,
            xray_source_box,
            prev_spin_temp,
            ics,
            hooks=hooks,
        )


class IonizedBox(OutputStruct):
    """A class containing all ionized boxes."""

    _meta = False
    _c_compute_function = lib.ComputeIonizedBox

    prev_ionize_redshift: float | None = attrs.field(default=None)

    @classmethod
    def new(cls, inputs, prev_ionize_redshift: float | None) -> Self:
        if inputs.flag_options.USE_MINI_HALOS:
            n_filtering = (
                int(
                    np.log(
                        min(
                            inputs.astro_params.R_BUBBLE_MAX,
                            0.620350491 * inputs.user_params.BOX_LEN,
                        )
                        / max(
                            global_params.R_BUBBLE_MIN,
                            0.620350491
                            * inputs.user_params.BOX_LEN
                            / inputs.user_params.HII_DIM,
                        )
                    )
                    / np.log(global_params.DELTA_R_HII_FACTOR)
                )
                + 1
            )
        else:
            n_filtering = 1

        shape = (inputs.user_params.HII_DIM,) * 2 + (
            int(inputs.user_params.NON_CUBIC_FACTOR * inputs.user_params.HII_DIM),
        )
        filter_shape = (n_filtering,) + shape

        out = {
            "xH_box": Array(shape, initfunc=np.ones),
            "Gamma12_box": Array(shape),
            "MFP_box": Array(shape),
            "z_re_box": Array(shape),
            "dNrec_box": Array(shape),
            "temp_kinetic_all_gas": Array(shape),
            "Fcoll": Array(filter_shape),
        }

        if inputs.flag_options.USE_MINI_HALOS:
            out["Fcoll_MINI"] = Array(filter_shape)

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

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
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


class BrightnessTemp(OutputStruct):
    """A class containing the brightness temperature box."""

    _c_compute_function = lib.ComputeBrightnessTemp

    _meta = False
    brightness_temp = arrayfield()

    @classmethod
    def new(cls, inputs) -> Self:
        shape = (inputs.user_params.HII_DIM,) * 2 + (
            int(inputs.user_params.NON_CUBIC_FACTOR * inputs.user_params.HII_DIM),
        )

        return cls(
            inputs,
            brightness_temp=Array(shape),
        )

    @cached_property
    def global_Tb(self):
        """Global (mean) brightness temperature."""
        if not self.is_computed:
            raise AttributeError(
                "global_Tb is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.brightness_temp)

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, PerturbedField):
            required += ["density"]
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
