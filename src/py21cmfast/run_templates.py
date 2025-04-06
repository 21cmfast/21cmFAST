"""
Parameter Templates.

This file contains parameter default templates for use with sets of parameters which
are set to reproduce a certain published result or which we know work together.

These should make it easier to define a run, choosing the closest template to your
desired parameters and altering as few as possible.
"""

import contextlib
import logging
import warnings
from pathlib import Path

import attrs
import tomllib

from .wrapper._utils import camel_to_snake
from .wrapper.inputs import InputStruct

TEMPLATE_PATH = Path(__file__).parent / "templates/"

logger = logging.getLogger(__name__)


def _construct_param_objects(template_dict, **kwargs):
    input_classes = {c.__name__: c for c in InputStruct.__subclasses__()}

    input_dict = {}
    for k, c in input_classes.items():
        fieldnames = [field.name for field in attrs.fields(c)]
        kw_dict = {kk: kwargs.pop(kk) for kk in fieldnames if kk in kwargs}
        arg_dict = {**template_dict[k], **kw_dict}
        input_struct = c.new(arg_dict)
        input_dict[camel_to_snake(k)] = input_struct

    if kwargs:
        warnings.warn(
            f"Excess arguments to `create_params_from_template` will be ignored: {kwargs}",
            stacklevel=2,
        )

    return input_dict


def list_templates() -> list[dict]:
    """Return a list of the available templates."""
    with (TEMPLATE_PATH / "manifest.toml").open("rb") as f:
        manifest = tomllib.load(f)
    return manifest["templates"]


def create_params_from_template(template_name: str | Path, **kwargs):
    """
    Construct the required InputStruct instances for a run from a given template.

    Parameters
    ----------
    template_name: str,
        defines the name/alias of the native template (see templates/manifest.toml for a list)
        alternatively, is the path to a TOML file containing tables titled [CosmoParams],
        [MatterParams], [AstroParams] and [AstroFlags] with parameter settings

    Other Parameters
    ----------------
    Any other parameter passed is considered to be a name and
    value of a parameter in any of the `InputStruct` subclasses,
    and will be used to over-ride the template values.

    Returns
    -------
    input_dict : dict containing:
        cosmo_params : CosmoParams
            Instance containing cosmological parameters
        matter_params : MatterParams
            Instance containing general run parameters
        astro_params : AstroParams
            Instance containing astrophysical parameters
        astro_flags : AstroFlags
            Instance containing flags which enable optional modules
    """
    # First check if the provided name is a path to an existsing TOML file
    template = None
    if Path(template_name).is_file():
        with Path(template_name).open("rb") as template_file:
            template = tomllib.load(template_file)

    # Next, check if the string matches one of our template aliases
    with (TEMPLATE_PATH / "manifest.toml").open("rb") as f:
        manifest = tomllib.load(f)
    for manf_entry in manifest["templates"]:
        if template_name.casefold() in [x.casefold() for x in manf_entry["aliases"]]:
            with (TEMPLATE_PATH / manf_entry["file"]).open("rb") as f:
                template = tomllib.load(f)
            break

    if template is not None:
        return _construct_param_objects(template, **kwargs)

    # We have not found a template in our templates or on file, raise an error
    message = (
        f"Template {template_name} not found on-disk or in native template aliases\n"
        + "Available native templates are:\n"
    )
    for manf_entry in manifest["templates"]:
        message += f"{manf_entry['name']}: {manf_entry['aliases']}\n"
        message += f"     {manf_entry['description']}:\n\n"

    raise ValueError(message)
