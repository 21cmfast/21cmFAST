"""
Parameter Templates.

This file contains parameter default templates for use with sets of parameters which
are set to reproduce a certain published result or which we know work together.

These should make it easier to define a run, choosing the closest template to your
desired parameters and altering as few as possible.
"""

import contextlib
import logging
import tomllib
import warnings
from os import path
from wrapper.inputs import AstroParams, CosmoParams, FlagOptions, UserParams

logger = logging.getLogger(__name__)


def _verify_template(template: dict):
    """
    Ensure a parameter template has all required keys for setting parameters and no more.

    Uses the park19 template as a comparison
    """
    pass

    # if keys_excess:
    #     message = (
    #         f"There are extra paramters in the given parameter template.\n"
    #         f"These will be ignored: {keys_excess}\n"
    #     )
    #     warnings.warn(message)


def _construct_param_objects(template_dict):
    cosmo_params = CosmoParams.new(template_dict["CosmoParams"])
    user_params = UserParams.new(template_dict["UserParams"])
    astro_params = CosmoParams.new(template_dict["AstroParams"])
    flag_options = CosmoParams.new(template_dict["FlagOptions"])
    return cosmo_params, user_params, astro_params, flag_options


def create_params_from_template(
    template_name: str,
):
    """
    Constructs the required InputStruct instances for a run from a given template.

    Parameters
    ----------
    template_name: str,
        defines the name/alias of the native template (see templates/manifest.toml for a list)
        alternatively, is the path to a TOML file containing tables titled [CosmoParams],
        [UserParams], [AstroParams] and [FlagOptions] with parameter settings

    Returns
    -------
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
    if path.isfile(template_name):
        template = tomllib.load(template_name)

    # Next, check if the string matches one of our template aliases
    manifest = tomllib.load("PATH_TO_21CMFAST_DATA/templates/manifest.toml")
    for manf_entry in manifest:
        if template_name.lower in manf_entry["aliases"]:
            template = tomllib.load(manf_entry)

    if template is not None:
        _verify_template(template)
        return _construct_param_objects(template)

    # We have not found a template in our templates or on file, raise an error
    logger.error(
        f"Template {template_name} not found on-disk or in native template aliases"
    )
    logger.error("Available native templates are:")
    for manf_entry in manifest:
        logger.error(f" {manf_entry['name']}: {manf_entry['aliases']}")
        logger.error(f"     {manf_entry['description']}:")
        logger.error("")

    raise ValueError(
        f"template {template_name} not found on-disk or in native template aliases"
    )
