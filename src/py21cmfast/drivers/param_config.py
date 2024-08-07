"""Functions for setting up and configuring inputs to driver functions."""

from __future__ import annotations

import attrs
import os
import warnings
from functools import cached_property
from typing import Any, Sequence

from .._cfg import config
from ..wrapper.globals import global_params
from ..wrapper.inputs import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    InputStruct,
    UserParams,
)


class InputCrossValidationError(ValueError):
    """Error when two parameters from different structs aren't consistent."""

    pass


def input_param_field(kls: InputStruct, default: bool = True):
    """An attrs field that must be an InputStruct."""
    if default:
        return attrs.field(
            default=kls(),
            converter=kls.new,
            validator=attrs.validators.instance_of(kls),
        )
    else:
        return attrs.field(
            converter=attrs.converters.optional(kls.new),
            validator=attrs.validators.optional(attrs.validators.instance_of(kls)),
        )


@attrs.define
class InputParameters:
    """A class defining a collection of InputStruct instances.

    This class simplifies combining different InputStruct instances together, performing
    validation checks between them, and being able to cross-check compatibility between
    different sets of instances.
    """

    user_params: UserParams = input_param_field(UserParams, default=True)
    cosmo_params: CosmoParams = input_param_field(CosmoParams, default=True)
    flag_options: FlagOptions = input_param_field(FlagOptions, default=False)
    _astro_params: AstroParams = input_param_field(AstroParams, default=False)

    @cached_property
    def astro_params(self) -> AstroParams:
        """The AstroParams instance."""
        return self._astro_params.clone(
            INHOMO_RECO=self.flag_options.INHOMO_RECO,
            USE_MINI_HALOS=self.flag_options.USE_MINI_HALOS,
        )

    @flag_options.validator
    def _flag_options_validator(self, att, val):
        if val is None:
            return

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

        if val.R_BUBBLE_MAX > self.user_params.BOX_LEN:
            raise InputCrossValidationError(
                f"R_BUBBLE_MAX is larger than BOX_LEN ({val.R_BUBBLE_MAX} > {self.user_params.BOX_LEN}). This is not allowed."
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

    def keys(self):
        """The list of available structs in this instance."""
        # Allow using **input_parameters
        return [
            field.name
            for field in attrs.fields(self.__class__)
            if getattr(self, field.name) is not None
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
            other[key] is not None and self[key] != other[key] for key in self.keys()
        )

    def merge(self, other: InputParameters) -> InputParameters:
        """Merge another InputParameters instance with this one, checking for compatibility."""
        if not self.is_compatible_with(other):
            raise ValueError("Input parameters are not compatible.")

        return attrs.evolve(self, **other)

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
    def from_output_structs(cls, output_structs, **defaults) -> InputParameters:
        """Generate a new InputParameters instance given a list of OutputStructs."""
        input_params = []
        for struct in output_structs:
            if struct is not None:
                for k, v in dir(struct).items():
                    if isinstance(v, InputParameters):
                        input_params.append(v)

        if len(input_params) == 0:
            return cls(**defaults)
        else:
            out = cls.combine(input_params)
            return out.merge(cls(**defaults))


# def _configure_inputs(
#     defaults: list,
#     *datasets,
#     ignore: list = ["redshift"],
#     flag_none: list | None = None,
# ):
#     """Configure a set of input parameter structs.

#     This is useful for basing parameters on a previous output.
#     The logic is this: the struct _cannot_ be present and different in both defaults and
#     a dataset. If it is present in _either_ of them, that will be returned. If it is
#     present in _neither_, either an error will be raised (if that item is in `flag_none`)
#     or it will pass.

#     Parameters
#     ----------
#     defaults : list of 2-tuples
#         Each tuple is (key, val). Keys are input struct names, and values are a default
#         structure for that input.
#     datasets : list of :class:`~_utils.OutputStruct`
#         A number of output datasets to cross-check, and draw parameter values from.
#     ignore : list of str
#         Attributes to ignore when ensuring that parameter inputs are the same.
#     flag_none : list
#         A list of parameter names for which ``None`` is not an acceptable value.

#     Raises
#     ------
#     ValueError :
#         If an input parameter is present in both defaults and the dataset, and is different.
#         OR if the parameter is present in neither defaults not the datasets, and it is
#         included in `flag_none`.
#     """
#     # First ensure all inputs are compatible in their parameters
#     _check_compatible_inputs(*datasets, ignore=ignore)

#     if flag_none is None:
#         flag_none = []

#     output = [0] * len(defaults)
#     for i, (key, val) in enumerate(defaults):
#         # Get the value of this input from the datasets
#         data_val = None
#         for dataset in datasets:
#             if dataset is not None and hasattr(dataset, key):
#                 data_val = getattr(dataset, key)
#                 break

#         if val is not None and data_val is not None and data_val != val:
#             raise ValueError(
#                 f"{key} has an inconsistent value with {dataset.__class__.__name__}."
#                 f" Expected:\n\n{val}\n\nGot:\n\n{data_val}."
#             )
#         if val is not None:
#             output[i] = val
#         elif data_val is not None:
#             output[i] = data_val
#         elif key in flag_none:
#             raise ValueError(f"For {key}, a value must be provided in some manner")
#         else:
#             output[i] = None

#     return output


# def configure_redshift(redshift, *structs):
#     """
#     Check and obtain a redshift from given default and structs.

#     Parameters
#     ----------
#     redshift : float
#         The default redshift to use
#     structs : list of :class:`~_utils.OutputStruct`
#         A number of output datasets from which to find the redshift.

#     Raises
#     ------
#     ValueError :
#         If both `redshift` and *all* structs have a value of `None`, **or** if any of them
#         are different from each other (and not `None`).
#     """
#     zs = {s.redshift for s in structs if s is not None and hasattr(s, "redshift")}
#     zs = list(zs)

#     if len(zs) > 1 or (
#         len(zs) == 1
#         and redshift is not None
#         and not np.isclose(zs[0], redshift, atol=1e-5)
#     ):
#         raise ValueError("Incompatible redshifts in inputs")
#     elif len(zs) == 1:
#         return zs[0]
#     elif redshift is None:
#         raise ValueError(
#             "Either redshift must be provided, or a data set containing it."
#         )
#     else:
#         return redshift


# def _setup_inputs(
#     input_params: dict[str, Any],
#     input_boxes: dict[str, OutputStruct] | None = None,
#     redshift=-1,
# ):
#     """
#     Verify and set up input parameters to any function that runs C code.

#     Parameters
#     ----------
#     input_boxes
#         A dictionary of OutputStruct objects that are meant as inputs to the current
#         calculation. These will be verified against each other, and also used to
#         determine redshift, if appropriate.
#     input_params
#         A dictionary of keys and dicts / input structs. This should have the random
#         seed, cosmo/user params and optionally the flag and astro params.
#     redshift
#         Optional value of the redshift. Can be None. If not provided, no redshift is
#         returned.

#     Returns
#     -------
#     random_seed
#         The random seed to use, determined from either explicit input or input boxes.
#     input_params
#         The configured input parameter structs, in the order in which they were given.
#     redshift
#         If redshift is given, it will also be output.
#     """
#     input_boxes = input_boxes or {}

#     if "flag_options" in input_params and "user_params" not in input_params:
#         raise ValueError("To set flag_options requires user_params")
#     if "astro_params" in input_params and "flag_options" not in input_params:
#         raise ValueError("To set astro_params requires flag_options")

#     keys = list(input_params.keys())
#     pkeys = ["user_params", "cosmo_params", "astro_params", "flag_options"]

#     # Convert the input params into the correct classes, unless they are None.
#     outparams = convert_input_dicts(*[input_params.pop(k, None) for k in pkeys])

#     # Get defaults from datasets where available
#     params = _configure_inputs(
#         list(zip(pkeys, outparams)) + list(input_params.items()),
#         *list(input_boxes.values()),
#     )

#     if redshift != -1:
#         redshift = configure_redshift(
#             redshift,
#             *[
#                 v
#                 for k, v in input_boxes.items()
#                 if hasattr(v, "redshift") and "prev" not in k
#             ],
#         )

#     p = convert_input_dicts(*params[:4], defaults=True)

#     # This turns params into a dict with all the input parameters in it.
#     params = dict(zip(pkeys + list(input_params.keys()), list(p) + params[4:]))

#     # Sort the params back into input order and ignore params not in input_params.
#     params = dict(zip(keys, [params[k] for k in keys]))

#     # Perform validation between different sets of inputs.
#     validate_all_inputs(**{k: v for k, v in params.items() if k != "random_seed"})

#     # return as list of values
#     params = list(params.values())

#     out = params
#     if redshift != -1:
#         out.append(redshift)

#     return out


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


# def _check_compatible_inputs(*datasets, ignore=["redshift"]):
#     """Ensure that all defined input parameters for the provided datasets are equal.

#     Parameters
#     ----------
#     datasets : list of :class:`~_utils.OutputStruct`
#         A number of output datasets to cross-check.
#     ignore : list of str
#         Attributes to ignore when ensuring that parameter inputs are the same.

#     Raises
#     ------
#     ValueError :
#         If datasets are not compatible.
#     """
#     done = []  # keeps track of inputs we've checked so we don't double check.

#     for i, d in enumerate(datasets):
#         # If a dataset is None, just ignore and move on.
#         if d is None:
#             continue

#         # noinspection PyProtectedMember
#         for inp in d._inputs:
#             # Skip inputs that we want to ignore
#             if inp in ignore:
#                 continue

#             if inp not in done:
#                 for j, d2 in enumerate(datasets[(i + 1) :]):
#                     if d2 is None:
#                         continue

#                     # noinspection PyProtectedMember
#                     if inp in d2._inputs and getattr(d, inp) != getattr(d2, inp):
#                         raise ValueError(
#                             f"""
#                             {d.__class__.__name__} and {d2.__class__.__name__} are incompatible.
#                             {inp}: {getattr(d, inp)}
#                             vs.
#                             {inp}: {getattr(d2, inp)}
#                             """
#                         )
#                 done += [inp]
