"""Functions for setting up and configuring inputs to driver functions."""

import numpy as np
import os
from typing import Any

from .._cfg import config
from ..wrapper.inputs import convert_input_dicts, validate_all_inputs
from ..wrapper.outputs import OutputStruct


def _configure_inputs(
    defaults: list,
    *datasets,
    ignore: list = ["redshift"],
    flag_none: list | None = None,
):
    """Configure a set of input parameter structs.

    This is useful for basing parameters on a previous output.
    The logic is this: the struct _cannot_ be present and different in both defaults and
    a dataset. If it is present in _either_ of them, that will be returned. If it is
    present in _neither_, either an error will be raised (if that item is in `flag_none`)
    or it will pass.

    Parameters
    ----------
    defaults : list of 2-tuples
        Each tuple is (key, val). Keys are input struct names, and values are a default
        structure for that input.
    datasets : list of :class:`~_utils.OutputStruct`
        A number of output datasets to cross-check, and draw parameter values from.
    ignore : list of str
        Attributes to ignore when ensuring that parameter inputs are the same.
    flag_none : list
        A list of parameter names for which ``None`` is not an acceptable value.

    Raises
    ------
    ValueError :
        If an input parameter is present in both defaults and the dataset, and is different.
        OR if the parameter is present in neither defaults not the datasets, and it is
        included in `flag_none`.
    """
    # First ensure all inputs are compatible in their parameters
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

        if val is not None and data_val is not None and data_val != val:
            raise ValueError(
                f"{key} has an inconsistent value with {dataset.__class__.__name__}."
                f" Expected:\n\n{val}\n\nGot:\n\n{data_val}."
            )
        if val is not None:
            output[i] = val
        elif data_val is not None:
            output[i] = data_val
        elif key in flag_none:
            raise ValueError(f"For {key}, a value must be provided in some manner")
        else:
            output[i] = None

    return output


def configure_redshift(redshift, *structs):
    """
    Check and obtain a redshift from given default and structs.

    Parameters
    ----------
    redshift : float
        The default redshift to use
    structs : list of :class:`~_utils.OutputStruct`
        A number of output datasets from which to find the redshift.

    Raises
    ------
    ValueError :
        If both `redshift` and *all* structs have a value of `None`, **or** if any of them
        are different from each other (and not `None`).
    """
    zs = {s.redshift for s in structs if s is not None and hasattr(s, "redshift")}
    zs = list(zs)

    if len(zs) > 1 or (
        len(zs) == 1
        and redshift is not None
        and not np.isclose(zs[0], redshift, atol=1e-5)
    ):
        raise ValueError("Incompatible redshifts in inputs")
    elif len(zs) == 1:
        return zs[0]
    elif redshift is None:
        raise ValueError(
            "Either redshift must be provided, or a data set containing it."
        )
    else:
        return redshift


def _setup_inputs(
    input_params: dict[str, Any],
    input_boxes: dict[str, OutputStruct] | None = None,
    redshift=-1,
):
    """
    Verify and set up input parameters to any function that runs C code.

    Parameters
    ----------
    input_boxes
        A dictionary of OutputStruct objects that are meant as inputs to the current
        calculation. These will be verified against each other, and also used to
        determine redshift, if appropriate.
    input_params
        A dictionary of keys and dicts / input structs. This should have the random
        seed, cosmo/user params and optionally the flag and astro params.
    redshift
        Optional value of the redshift. Can be None. If not provided, no redshift is
        returned.

    Returns
    -------
    random_seed
        The random seed to use, determined from either explicit input or input boxes.
    input_params
        The configured input parameter structs, in the order in which they were given.
    redshift
        If redshift is given, it will also be output.
    """
    input_boxes = input_boxes or {}

    if "flag_options" in input_params and "user_params" not in input_params:
        raise ValueError("To set flag_options requires user_params")
    if "astro_params" in input_params and "flag_options" not in input_params:
        raise ValueError("To set astro_params requires flag_options")

    keys = list(input_params.keys())
    pkeys = ["user_params", "cosmo_params", "astro_params", "flag_options"]

    # Convert the input params into the correct classes, unless they are None.
    outparams = convert_input_dicts(*[input_params.pop(k, None) for k in pkeys])

    # Get defaults from datasets where available
    params = _configure_inputs(
        list(zip(pkeys, outparams)) + list(input_params.items()),
        *list(input_boxes.values()),
    )

    if redshift != -1:
        redshift = configure_redshift(
            redshift,
            *[
                v
                for k, v in input_boxes.items()
                if hasattr(v, "redshift") and "prev" not in k
            ],
        )

    p = convert_input_dicts(*params[:4], defaults=True)

    # This turns params into a dict with all the input parameters in it.
    params = dict(zip(pkeys + list(input_params.keys()), list(p) + params[4:]))

    # Sort the params back into input order and ignore params not in input_params.
    params = dict(zip(keys, [params[k] for k in keys]))

    # Perform validation between different sets of inputs.
    validate_all_inputs(**{k: v for k, v in params.items() if k != "random_seed"})

    # return as list of values
    params = list(params.values())

    out = params
    if redshift != -1:
        out.append(redshift)

    return out


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


def _check_compatible_inputs(*datasets, ignore=["redshift"]):
    """Ensure that all defined input parameters for the provided datasets are equal.

    Parameters
    ----------
    datasets : list of :class:`~_utils.OutputStruct`
        A number of output datasets to cross-check.
    ignore : list of str
        Attributes to ignore when ensuring that parameter inputs are the same.

    Raises
    ------
    ValueError :
        If datasets are not compatible.
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
                for j, d2 in enumerate(datasets[(i + 1) :]):
                    if d2 is None:
                        continue

                    # noinspection PyProtectedMember
                    if inp in d2._inputs and getattr(d, inp) != getattr(d2, inp):
                        raise ValueError(
                            f"""
                            {d.__class__.__name__} and {d2.__class__.__name__} are incompatible.
                            {inp}: {getattr(d, inp)}
                            vs.
                            {inp}: {getattr(d2, inp)}
                            """
                        )
                done += [inp]
