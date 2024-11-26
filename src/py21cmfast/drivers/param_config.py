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
        default=None,
        converter=attrs.converters.optional(kls.new),
        validator=attrs.validators.optional(attrs.validators.instance_of(kls)),
    )


@attrs.define(kw_only=True)
class InputParameters:
    """A class defining a collection of InputStruct instances.

    This class simplifies combining different InputStruct instances together, performing
    validation checks between them, and being able to cross-check compatibility between
    different sets of instances.
    """

    random_seed = attrs.field(default=None, converter=attrs.converters.optional(int))
    redshift = attrs.field(default=None, converter=attrs.converters.optional(float))
    user_params: UserParams = input_param_field(UserParams)
    cosmo_params: CosmoParams = input_param_field(CosmoParams)
    flag_options: FlagOptions = input_param_field(FlagOptions)
    astro_params: AstroParams = input_param_field(AstroParams)

    @flag_options.validator
    def _flag_options_validator(self, att, val):
        if val is None:
            return

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
        if val is None:
            return
        if self.user_params is None:
            return

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
        if val is None:
            return
        # perform a very rudimentary check to see if we are underresolved and not using the linear approx
        if val.BOX_LEN > val.DIM and not global_params.EVOLVE_DENSITY_LINEARLY:
            warnings.warn(
                "Resolution is likely too low for accurate evolved density fields\n It Is recommended"
                + "that you either increase the resolution (DIM/BOX_LEN) or"
                + "set the EVOLVE_DENSITY_LINEARLY flag to 1"
            )

    def merge_keys(self):
        """The list of available structs in this instance."""
        # Allow using **input_parameters
        return [
            field.name
            for field in attrs.fields(self.__class__)
            if field.name != "redshift"
        ]

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

    def merge(self, other: InputParameters) -> InputParameters:
        """Merge another InputParameters instance with this one, checking for compatibility."""
        if not self.is_compatible_with(other):
            raise ValueError(
                f"Input parameters are not compatible. \n SELF {self} \n OTHER {other}"
            )
        return InputParameters(**{k: self[k] or other[k] for k in self.merge_keys()})

    @classmethod
    def combine(cls, inputs: Sequence[InputParameters]) -> InputParameters:
        """Combine multiple input parameter structs into one.

        Parameters
        ----------
        inputs : list of :class:`InputParameters`
            The input parameter structs to combine.

        Returns
        -------
        :class:`InputParameters`
            The combined input parameter struct.
        """
        # Create an empty instance of InputParameters
        this = inputs[0]

        for inp in inputs[1:]:
            this = this.merge(inp)

        return this

    @classmethod
    def from_output_structs(
        cls, output_structs, redshift=None, **defaults
    ) -> InputParameters:
        """Generate a new InputParameters instance given a list of OutputStructs."""
        input_params = []
        for struct in output_structs:
            if struct is not None:
                ip_args = {
                    k.lstrip("_"): getattr(struct, k, None)
                    for k in struct._inputs
                    if k.lstrip("_") in [field.name for field in attrs.fields(cls)]
                }
                input_params.append(cls(**ip_args))

        if len(input_params) == 0:
            return cls(**defaults)
        else:
            # Combine all the parameter structs from input boxes
            out = cls.combine(input_params)
            # Now combine with provided kwargs
            return attrs.evolve(out.merge(cls(**defaults)), redshift=redshift)

    @classmethod
    def from_template(cls, name, random_seed):
        """Construct full InputParameters instance from native or TOML file template."""
        return cls(**create_params_from_template(name), random_seed=random_seed)

    @classmethod
    def from_inputstructs(
        cls,
        cosmo_params: CosmoParams,
        user_params: UserParams,
        astro_params: AstroParams,
        flag_options: flag_options,
        random_seed: int,
    ):
        """Construct full InputParameters instance from InputStruct instances."""
        return cls(
            cosmo_params=cosmo_params,
            user_params=user_params,
            astro_params=astro_params,
            flag_options=flag_options,
            random_seed=random_seed,
        )

    @classmethod
    def from_defaults(cls, random_seed, **kwargs):
        """Construct full InputParameters instance from default values."""
        cosmo_params = CosmoParams.new(
            {
                k: v
                for k, v in kwargs.items()
                if k in [f.name for f in attrs.fields(CosmoParams)]
            }
        )
        user_params = UserParams.new(
            {
                k: v
                for k, v in kwargs.items()
                if k in [f.name for f in attrs.fields(UserParams)]
            }
        )
        astro_params = AstroParams.new(
            {
                k: v
                for k, v in kwargs.items()
                if k in [f.name for f in attrs.fields(AstroParams)]
            }
        )
        flag_options = FlagOptions.new(
            {
                k: v
                for k, v in kwargs.items()
                if k in [f.name for f in attrs.fields(FlagOptions)]
            }
        )

        return cls(
            cosmo_params=cosmo_params,
            user_params=user_params,
            astro_params=astro_params,
            flag_options=flag_options,
            random_seed=random_seed,
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

    # TODO: methods for equality: w/o redshift, w/o seed


def check_redshift_consistency(inputs: InputParameters, output_structs):
    """Check the redshifts between provided OutputStruct objects and an InputParameters instance."""
    for struct in output_structs:
        if struct is not None and struct.redshift != inputs.redshift:
            raise ValueError("Incompatible redshifts in inputs")


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
