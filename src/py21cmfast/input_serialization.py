"""Functions for handling InputParameters for various purposes, e.g. I/O and unstructuring."""

import warnings
from typing import Any, Literal

import attrs

from .utils import recursive_difference
from .wrapper._utils import camel_to_snake, snake_to_camel
from .wrapper.inputs import CosmoTables, InputParameters, InputStruct


def convert_inputs_to_dict(
    inputs: InputParameters,
    mode: Literal["full", "minimal"] = "full",
    only_structs: bool = True,
    camel: bool = True,
    only_cstruct_params: bool = False,
    use_aliases: bool = True,
) -> dict[str, dict[str, Any]]:
    """Convert an InputParameters object to a dictionary, with various options.

    While the :class:`~py21cmfast.wrapper.inputs.InputParameters.asdict` method allows
    several options for "unstructuring" the inputs to a dictionary, this function is
    much more flexible, having options that cover a number of use-cases.

    Parameters
    ----------
    inputs
        The input parameters to convert to a dict.
    mode
        Either 'minimal' to get the minimal dictionary required to specify the
        parameters (on top of defaults) or 'full' to keep all parameters.
    only_structs
        Whether to only return InputStruct objects (unstructured to dicts),
        otherwise also return other attributes (e.g. random_seed).
    camel
        Whether the keys of the returned dict should be camel-case, e.g.
        SimulationOptions. Otherwise, return as snake_case. Only applies to
        InputStruct attributes of the inputs (i.e. not node_redshifts or random_seed).
    only_cstruct_params
        Only return parameters that are part of the Cstruct, rather than all
        fields of the class. This is useful for pretty-printing.
    use_aliases
        If True, use correct aliases for parameters, which allows the dictionary
        to be passed back into the InputStruct constructors (e.g. use DIM instead
        of _DIM).
    """
    kw = {
        "only_structs": only_structs,
        "camel": camel,
        "only_cstruct_params": only_cstruct_params,
        "use_aliases": use_aliases,
    }
    all_inputs = inputs.asdict(**kw)

    if mode == "minimal":
        defaults = InputParameters(random_seed=0)
        default_dct = defaults.asdict(**kw)
        all_inputs = recursive_difference(all_inputs, default_dct)

    return all_inputs


def prepare_inputs_for_serialization(
    inputs: InputParameters,
    mode: Literal["full", "minimal"] = "full",
    only_structs: bool = True,
    camel: bool = True,
) -> dict[str, dict[str, Any]]:
    """Prepare an inputs class for serialization (to e.g. TOML, YAML or HDF5).

    This is a thin wrapper around :func:`~convert_inputs_to_dict` that also
    ensures that 'None' values are removed (so long as their default is also None)
    and that the parameter names map back to aliases of InputStruct attributes.
    """
    dct = convert_inputs_to_dict(
        inputs,
        mode=mode,
        only_structs=only_structs,
        camel=camel,
        use_aliases=False,  # convert to aliases below instead.
    )

    # dct is a dict of dicts, where each subdict represents a
    # single input parameter struct (SimulationOptions, AstroParams etc).
    # The rules are that _all_ of the input structs are in this dict, even if they are
    # empty. The keys allowed in each struct are such that:
    #
    #    1. They are valid keys to pass to the class constructor, not necessarily
    #       all the fields required for the Cstruct (since sometimes these are
    #       auto-calcualted).
    #    2. They are not necessarily exhaustive, especially if mode=="minimal".
    #    3. The values are those that are required to *build the isntance*, not their
    #       final transformed values for use in C.
    #
    # However, some attributes of the InputStruct classes are not necessarily
    # required for building the Cstruct. This can occur if there are two pathways of
    # setting a particular necessary parameter (e.g. DIM can be set either by setting
    # DIM directly, or by setting HII_DIM and HIRES_TO_LOWRES_FACTOR). In this case,
    # at least one of the attributes will have a default value of None. If something
    # has an actual value of None, we require that its default is None. This way,
    # we can simply leave it out of the written TOML (or HDF5) when its value is None,
    # and it will anyway be set to its own default if read back in.
    out = {}
    for structname, structvals in dct.items():
        this = {}
        clsname = snake_to_camel(structname)
        if clsname in InputStruct._subclasses:
            fields = attrs.fields_dict(InputStruct._subclasses[clsname])
        elif clsname == "CosmoTables":
            fields = attrs.fields_dict(CosmoTables)

        for key, val in structvals.items():
            if val is None:
                if fields[key].default is not None:  # pragma: nocover
                    # This should not be reachable because setting a required parameter
                    # to None should error on validation, rather than reaching here.
                    raise RuntimeError(
                        f"Detected that {structname} has {key}=None but it is not an optional parameter!"
                    )
            else:
                # Convert to use the alias, rather than the name (i.e. DIM instead of _DIM)
                this[fields[key].alias] = val
        if this:
            out[structname] = this

    # Furthermore, some
    return out


def deserialize_inputs(
    dict_of_structdicts: dict[str, Any], safe: bool = True, **loose_params
) -> dict[str, InputStruct]:
    """Construct a dictionary of InputStructs ready to be converted to InputParameters.

    Parameters
    ----------
    dict_of_structdicts
        A dictionary whose keys are names of InputStruct attributes of
        InputParameters (e.g. CosmoParams, SimulationOptions), and whose values
        are dictionaries of parameters specific to each struct. Not every parameter
        of every struct is required.

    Other Parameters
    ----------------
    All other parameters are considered "loose" parameters of one of the InputStructs,
    and will *override* the parameter matching their name if it is found in one of the
    structs.

    Returns
    -------
    dict_of_structs
        A dictionary whose keys are attribute names of the InputParameters class,
        and whose values are InputStruct instances, ready to instantiate an
        InputParameters class, e.g. ``InputParameters(**dict_of_structs, random_seed=1)``.
    """
    # It's possible that names in dict_of_structdicts are in camel case or snake
    # case. We normalise to camel case here. This also de-references the input
    # so when we modify it in-place (via .pop()) later, we don't mess with the user's
    # input.
    dict_of_structdicts = {
        snake_to_camel(name): dct for name, dct in dict_of_structdicts.items()
    }

    input_dict = {}
    extra_params = {}
    for structname, kls in (
        InputStruct._subclasses | {"CosmoTables": CosmoTables}
    ).items():
        # Use field.alias instead of field.name because the alias is what needs
        # to be passed to the class constructor (e.g. "DIM" instead of "_DIM")
        fieldnames = [field.alias for field in attrs.fields(kls)]
        kw_dict = {kk: loose_params.pop(kk) for kk in fieldnames if kk in loose_params}

        these_all = dict_of_structdicts.pop(structname, {})

        these = {kk: these_all[kk] for kk in these_all if kk in fieldnames}
        extra = {kk: these_all[kk] for kk in these_all if kk not in fieldnames}

        # Here, if structname is not in the input dict_of_structdicts, it is OK,
        # because we just assume an empty set of parameters, potentially added to
        # by the loose params.
        arg_dict = {**these, **kw_dict}
        input_struct = kls.new(arg_dict)
        input_dict[camel_to_snake(structname)] = input_struct

        if extra:
            extra_params[structname] = extra

    if dict_of_structdicts:
        warnings.warn(
            f"The following keys were not recognized for deserializing to InputParameters: {tuple(dict_of_structdicts.keys())}",
            stacklevel=2,
        )

    if extra_params or loose_params:
        all_extra = {**extra_params, **loose_params}
        msg = f"Excess arguments exist: {all_extra}"
        if safe:
            raise ValueError(msg)
        else:
            warnings.warn(msg, stacklevel=2)

    return input_dict
