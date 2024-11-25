"""
Parameter Templates.

This file contains parameter default templates for use with sets of parameters which
are set to reproduce a certain published result or which we know work together.

These should make it easier to define a run, choosing the closest template to your
desired parameters and altering as few as possible.
"""

import attrs
import contextlib
import logging
import tomllib
import warnings
from pathlib import Path
from .wrapper.inputs import AstroParams, CosmoParams, FlagOptions, UserParams
from .wrapper.structs import InputStruct
from .wrapper._utils import camel_to_snake

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
            f"Excess arguments to `create_params_from_template` will be ignored: {kwargs}"
        )

    return input_dict


def create_params_from_template(template_name: str, **kwargs):
    """
    Constructs the required InputStruct instances for a run from a given template.

    Parameters
    ----------
    template_name: str,
        defines the name/alias of the native template (see templates/manifest.toml for a list)
        alternatively, is the path to a TOML file containing tables titled [CosmoParams],
        [UserParams], [AstroParams] and [FlagOptions] with parameter settings

    kwargs,
        names and values of parameters in the ``InputStruct`` subclasses to override the template values with.

    Returns
    -------
    input_dict : dict containing:
        cosmo_params : CosmoParams
            Instance containing cosmological parameters
        user_params : UserParams
            Instance containing general run parameters
        astro_params : AstroParams
            Instance containing astrophysical parameters
        flag_options : FlagOptions
            Instance containing flags which enable optional modules
    """
    # First check if the provided name is a path to an existsing TOML file
    template = None
    if Path(template_name).is_file():
        template = tomllib.load(template_name)

    # Next, check if the string matches one of our template aliases
    with open(TEMPLATE_PATH / "manifest.toml",'rb') as f:
        manifest = tomllib.load(f)
    for manf_entry in manifest["templates"]:
        if template_name.casefold() in [x.casefold() for x  in manf_entry["aliases"]]:
            with open(TEMPLATE_PATH / manf_entry["file"],'rb') as f:
                template = tomllib.load(f)
            break

    if template is not None:
        return _construct_param_objects(template, **kwargs)

    # We have not found a template in our templates or on file, raise an error
    logger.error(
        f"Template {template_name} not found on-disk or in native template aliases"
    )
    logger.error("Available native templates are:")
    for manf_entry in manifest['templates']:
        logger.error(f" {manf_entry['name']}: {manf_entry['aliases']}")
        logger.error(f"     {manf_entry['description']}:")
        logger.error("")

    raise ValueError(
        f"template {template_name} not found on-disk or in native template aliases"
    )
