"""
Parameter Templates.

This file contains parameter default templates for use with sets of parameters which
are set to reproduce a certain published result or which we know work together.

These should make it easier to define a run, choosing the closest template to your
desired parameters and altering as few as possible.
"""

import datetime
import logging
import warnings
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import attrs
import tomlkit

from .wrapper._utils import camel_to_snake
from .wrapper.inputs import InputParameters, InputStruct

TEMPLATE_PATH = Path(__file__).parent / "templates/"
MANIFEST = TEMPLATE_PATH / "manifest.toml"

logger = logging.getLogger(__name__)

TOMLMode = Literal["full", "minimal"]


def _construct_param_objects(
    template_dict: dict[str, Any], **kwargs
) -> dict[str, InputStruct]:
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
    with MANIFEST.open("r") as fl:
        manifest = tomlkit.load(fl)
    return manifest["templates"]


def load_template_file(template_name: str | Path):
    """
    Handle the loading of template TOML files.

    First it checks for a file with the name given,
    then it checks for a native template with that name,
    throwing an error if neither are found.
    """
    if (fname := Path(template_name)).is_file():
        with fname.open("r") as fl:
            return tomlkit.load(fl)

    with MANIFEST.open("r") as fl:
        manifest = tomlkit.load(fl)

    for manf_entry in manifest["templates"]:
        if template_name.casefold() in [x.casefold() for x in manf_entry["aliases"]]:
            with (TEMPLATE_PATH / manf_entry["file"]).open("r") as fl:
                return tomlkit.load(fl)

    message = (
        f"Template {template_name} not found on-disk or in native template aliases\n"
        + "Available native templates are:\n"
    )
    for manf_entry in manifest["templates"]:
        message += f"{manf_entry['name']}: {manf_entry['aliases']}\n"
        message += f"     {manf_entry['description']}:\n\n"

    raise ValueError(message)


def create_params_from_template(
    template_name: str | Path | Sequence[str | Path], **kwargs
):
    """
    Construct the required InputStruct instances for a run from a given template.

    Parameters
    ----------
    template_name: str,
        defines the name/alias of the native template (see templates/manifest.toml for a list)
        alternatively, is the path to a TOML file containing tables titled [CosmoParams],
        [SimulationOptions], [AstroParams] and [AstroOptions] with parameter settings

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
        simulation_options : SimulationOptions
            Instance containing general run parameters
        simulation_options : MatterOptions
            Instance containing general run flags and enums
        astro_params : AstroParams
            Instance containing astrophysical parameters
        astro_options : AstroOptions
            Instance containing astrophysical flags and enums
    """
    if isinstance(template_name, str | Path):
        templates = [template_name]
    else:
        templates = template_name

    full_template = defaultdict(dict)
    for tmpl in templates:
        thist = load_template_file(tmpl)
        for k, v in thist.items():
            full_template[k] |= v

    return _construct_param_objects(full_template, **kwargs)


def _get_inputs_as_dict(inputs: InputParameters, mode: TOMLMode = "full"):
    all_inputs = inputs.asdict(only_structs=True, camel=True, only_cstruct_params=True)

    if mode == "minimal":
        defaults = InputParameters(random_seed=0)
        default_dct = defaults.asdict(only_structs=True, camel=True)

        # Get the minimal set of params (non-default params)
        all_inputs = {
            structname: {
                k: v for k, v in params.items() if default_dct[structname][k] != v
            }
            for structname, params in all_inputs.items()
        }
    return all_inputs


def write_template(
    inputs: InputParameters, template_file: Path | str, mode: TOMLMode = "full"
):
    """Write a set of input parameters to a template file.

    Parameters
    ----------
    inputs
        The inputs to write to file.
    template_file
        The path of the output.
    """
    inputs_dct = _get_inputs_as_dict(inputs, mode=mode)

    template_file = Path(template_file)
    doc = tomlkit.document()
    doc.add(
        tomlkit.comment(
            "This file was generated by py21cmfast.run_templates.write_template"
        )
    )
    doc.add(tomlkit.comment(f"Created on: {datetime.datetime.now().isoformat()}"))
    doc.update(inputs_dct)

    with template_file.open("w") as fl:
        tomlkit.dump(doc, fl)
