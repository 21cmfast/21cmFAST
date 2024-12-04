"""Functions for setting up and configuring inputs to driver functions."""

from __future__ import annotations

import attrs
import logging
import numpy as np
import os
import warnings
from functools import cached_property
from typing import Any, Sequence

from .._cfg import config
from ..run_templates import create_params_from_template
from ..wrapper.globals import global_params
from ..wrapper.inputs import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    InputStruct,
    UserParams,
)

logger = logging.getLogger(__name__)


class InputCrossValidationError(ValueError):
    """Error when two parameters from different structs aren't consistent."""

    pass


def input_param_field(kls: InputStruct):
    """An attrs field that must be an InputStruct.

    Parameters
    ----------
    kls : InputStruct subclass
        The parameter structure which should be returned as an attrs field

    """
    return attrs.field(
        default=kls.new(),
        converter=kls.new,
        validator=attrs.validators.instance_of(kls),
    )


def get_logspaced_redshifts(
    min_redshift: float,
    z_step_factor: float,
    max_redshift: float,
):
    """Compute a sequence of redshifts to evolve over that are log-spaced."""
    redshifts = [min_redshift]
    while redshifts[-1] < max_redshift:
        redshifts.append((redshifts[-1] + 1.0) * z_step_factor - 1.0)

    return np.array(redshifts)[::-1]


def _node_redshifts_converter(value, self):
    # we assume an array-like is passed
    if hasattr(value, "__len__"):
        return np.sort(np.array(value, dtype=float).flatten())[::-1]
    if isinstance(value, float):
        return np.array([value])
    return np.array([])


@attrs.define(kw_only=True, frozen=True)
class InputParameters:
    """A class defining a collection of InputStruct instances.

    This class simplifies combining different InputStruct instances together, performing
    validation checks between them, and being able to cross-check compatibility between
    different sets of instances.
    """

    random_seed = attrs.field(converter=int)
    user_params: UserParams = input_param_field(UserParams)
    cosmo_params: CosmoParams = input_param_field(CosmoParams)
    flag_options: FlagOptions = input_param_field(FlagOptions)
    astro_params: AstroParams = input_param_field(AstroParams)

    # passed to the converter, TODO: this can be cleaned up
    node_redshifts = attrs.field(
        converter=attrs.Converter(_node_redshifts_converter, takes_self=True)
    )

    @node_redshifts.default
    def _node_redshifts_default(self):
        return (
            get_logspaced_redshifts(
                min_redshift=5.5,
                max_redshift=self.user_params.Z_HEAT_MAX,
                z_step_factor=self.user_params.ZPRIME_STEP_FACTOR,
            )
            if (self.flag_options.INHOMO_RECO or self.flag_options.USE_TS_FLUCT)
            else None
        )

    @node_redshifts.validator
    def _node_redshifts_validator(self, att, val):
        if (
            self.flag_options.INHOMO_RECO or self.flag_options.USE_TS_FLUCT
        ) and val.max() < self.user_params.Z_HEAT_MAX:
            raise ValueError(
                "For runs with inhomogeneous recombinations or spin temperature fluctuations,\n"
                + f"your maximum passed node_redshifts {val.max()} must be above Z_HEAT_MAX {self.user_params.Z_HEAT_MAX}"
            )

    @flag_options.validator
    def _flag_options_validator(self, att, val):
        if self.user_params is not None:
            if (
                val.USE_MINI_HALOS
                and not self.user_params.USE_RELATIVE_VELOCITIES
                and not val.FIX_VCB_AVG
            ):
                warnings.warn(
                    "USE_MINI_HALOS needs USE_RELATIVE_VELOCITIES to get the right evolution!"
                )

            if val.HALO_STOCHASTICITY and self.user_params.PERTURB_ON_HIGH_RES:
                msg = (
                    "Since the lowres density fields are required for the halo sampler"
                    "We are currently unable to use PERTURB_ON_HIGH_RES and HALO_STOCHASTICITY"
                    "Simultaneously."
                )
                raise NotImplementedError(msg)

        if val.USE_EXP_FILTER and not val.USE_HALO_FIELD:
            warnings.warn("USE_EXP_FILTER has no effect unless USE_HALO_FIELD is true")

    @astro_params.validator
    def _astro_params_validator(self, att, val):
        if val.R_BUBBLE_MAX > self.user_params.BOX_LEN:
            raise InputCrossValidationError(
                f"R_BUBBLE_MAX is larger than BOX_LEN ({val.R_BUBBLE_MAX} > {self.user_params.BOX_LEN}). This is not allowed."
            )

        if val.R_BUBBLE_MAX != 50 and self.flag_options.INHOMO_RECO:
            warnings.warn(
                "You are setting R_BUBBLE_MAX != 50 when INHOMO_RECO=True. "
                "This is non-standard (but allowed), and usually occurs upon manual "
                "update of INHOMO_RECO"
            )

        if val.M_TURN > 8 and self.flag_options.USE_MINI_HALOS:
            warnings.warn(
                "You are setting M_TURN > 8 when USE_MINI_HALOS=True. "
                "This is non-standard (but allowed), and usually occurs upon manual "
                "update of M_TURN"
            )

        if (
            global_params.HII_FILTER == 1
            and val.R_BUBBLE_MAX > self.user_params.BOX_LEN / 3
        ):
            msg = (
                "Your R_BUBBLE_MAX is > BOX_LEN/3 "
                f"({val.R_BUBBLE_MAX} > {self.user_params.BOX_LEN / 3})."
            )

            if config["ignore_R_BUBBLE_MAX_error"]:
                warnings.warn(msg)
            else:
                raise ValueError(msg)

    @user_params.validator
    def _user_params_validator(self, att, val):
        # perform a very rudimentary check to see if we are underresolved and not using the linear approx
        if val.BOX_LEN > val.DIM and not global_params.EVOLVE_DENSITY_LINEARLY:
            warnings.warn(
                "Resolution is likely too low for accurate evolved density fields\n It Is recommended"
                + "that you either increase the resolution (DIM/BOX_LEN) or"
                + "set the EVOLVE_DENSITY_LINEARLY flag to 1"
            )

    def __getitem__(self, key):
        """Get an item from the instance in a dict-like manner."""
        # Also allow using **input_parameters
        return getattr(self, key)

    def is_compatible_with(self, other: InputParameters) -> bool:
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

    def check_output_compatibility(self, output_structs):
        """Generate a new InputParameters instance given a list of OutputStructs.

        In contrast to other construction methods, we do not accept overwriting of
        sub-fields here, since it will no longer be compatible with the output structs.

        All required fields not present in the `OutputStruct` objects need to be provided.
        """
        # get matching fields in each output struct
        fieldnames = [field.name for field in attrs.fields(self.__class__) if field.eq]
        for struct in output_structs:
            if struct is None:
                continue

            input_params = {
                k.lstrip(""): getattr(struct, k, None)
                for k in struct._inputs
                if k.lstrip("") in fieldnames
            }

            # Since self is always complete we can just compare against it
            for field, struct_val in input_params.items():
                input_val = getattr(self, field)
                if struct_val is not None and struct_val != input_val:
                    raise ValueError(
                        f"InputParameters not compatible with {struct} {field}: inputs {input_val} != struct {struct_val}"
                    )

    def evolve_input_structs(self, **kwargs):
        """Return an altered clone of the `InputParameters` structs.

        Unlike clone(), this function takes fields from the constituent `InputStruct` classes
        and only overwrites those sub-fields instead of the entire field
        """
        struct_args = {}
        for inp_type in ("cosmo_params", "user_params", "astro_params", "flag_options"):
            obj = getattr(self, inp_type)
            struct_args[inp_type] = obj.clone(
                **{k: v for k, v in kwargs.items() if hasattr(obj, k)}
            )

        return self.clone(**struct_args)

    @classmethod
    def from_template(cls, name, **kwargs):
        """Construct full InputParameters instance from native or TOML file template.

        Takes `InputStruct` fields as keyword arguments which overwrite the template/default fields
        """
        return cls(
            **create_params_from_template(name),
            **kwargs,
        )

    def clone(self, **kwargs):
        """Generate a copy of the InputParameter structure with specified changes."""
        return attrs.evolve(self, **kwargs)

    def __repr__(self):
        """
        String representation of the structure.

        Created by combining repr methods from the InputStructs
        which make up this object
        """
        return (
            f"cosmo_params: {repr(self.cosmo_params)}\n"
            + f"user_params: {repr(self.user_params)}\n"
            + f"astro_params: {repr(self.astro_params)}\n"
            + f"flag_options: {repr(self.flag_options)}\n"
        )

    # TODO: methods for equality: w/o redshifts, w/o seed


def check_redshift_consistency(redshift, output_structs):
    """Make sure all given OutputStruct objects exist at the same given redshift."""
    for struct in output_structs:
        if struct is not None and struct.redshift != redshift:
            raise ValueError(
                f"Incompatible redshifts with inputs and {struct.__class__.__name__}: {redshift} != {struct.redshift}"
            )


def _get_config_options(
    direc, regenerate, write, hooks
) -> tuple[str, bool, dict[callable, dict[str, Any]]]:
    direc = str(os.path.expanduser(config["direc"] if direc is None else direc))

    if hooks is None or len(hooks) > 0:
        hooks = hooks or {}

        if callable(write) and write not in hooks:
            hooks[write] = {"direc": direc}

        if not hooks:
            if write is None:
                write = config["write"]

            if not callable(write) and write:
                hooks["write"] = {"direc": direc}

    return (
        direc,
        bool(config["regenerate"] if regenerate is None else regenerate),
        hooks,
    )
