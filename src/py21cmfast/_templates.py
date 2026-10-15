"""
Parameter Templates.

This file contains parameter default templates for use with sets of parameters which
are set to reproduce a certain published result or which we know work together.

These should make it easier to define a run, choosing the closest template to your
desired parameters and altering as few as possible.
"""

import datetime
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

import tomlkit

from .input_serialization import deserialize_inputs, prepare_inputs_for_serialization
from .wrapper.inputs import (
    AstroOptions,
    AstroParams,
    CosmoParams,
    CosmoTables,
    InputParameters,
    MatterOptions,
    SimulationOptions,
)

TEMPLATE_PATH = Path(__file__).parent / "templates/"
MANIFEST = TEMPLATE_PATH / "manifest.toml"

logger = logging.getLogger(__name__)

TOMLMode = Literal["full", "minimal"]


class TemplateParams(TypedDict):
    """The parameters defined by a template, as keyword arguments of InputParameters."""

    cosmo_params: CosmoParams
    simulation_options: SimulationOptions
    matter_options: MatterOptions
    astro_params: AstroParams
    astro_options: AstroOptions
    cosmo_tables: NotRequired[CosmoTables]
    random_seed: NotRequired[int]
    node_redshifts: NotRequired[Sequence[float]]


def list_templates() -> list[dict[str, Any]]:
    """Return a list of the available templates."""
    with MANIFEST.open("r") as fl:
        manifest = tomlkit.load(fl)
    return manifest["templates"]


def load_template_file(template_name: str | Path) -> tomlkit.TOMLDocument:
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
        if str(template_name).casefold() in [
            x.casefold() for x in manf_entry["aliases"]
        ]:
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
    template_name: str | Path | Sequence[str | Path], **kwargs: Any
) -> TemplateParams:
    """
    Construct the required InputStruct instances for a run from a given template.

    Parameters
    ----------
    template_name: str, Path, or a sequence of them
        defines the name/alias of the native template (see templates/manifest.toml for a list)
        alternatively, is the path to a TOML file containing tables titled [CosmoParams],
        [SimulationOptions], [AstroParams] and [AstroOptions] with parameter settings.
        If a sequence, the templates are combined from left to right (later ones take
        precedence).

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
        random_seed : int, optional
            The random seed, if the template defines one.
        node_redshifts : list of float, optional
            The node redshifts, if the template defines them.
    """
    if isinstance(template_name, str | Path):
        templates = [template_name]
    else:
        templates = template_name

    full_template: defaultdict[str, dict[str, Any]] = defaultdict(dict)

    random_seed = None
    node_redshifts = None

    for tmpl in templates:
        thist = load_template_file(tmpl)
        for k, v in thist.items():
            if k == "random_seed":
                random_seed = v
            elif k == "node_redshifts":
                node_redshifts = v
            else:
                full_template[k] |= v
    # deserialize_inputs returns an instance of every InputStruct subclass, keyed by
    # its snake-case name as an InputParameters attribute (and the CosmoTables, if
    # asked for), i.e. a TemplateParams.
    out = cast("TemplateParams", deserialize_inputs(full_template, **kwargs))
    if random_seed is not None:
        out["random_seed"] = random_seed
    if node_redshifts is not None:
        out["node_redshifts"] = node_redshifts
    return out


def write_template(
    inputs: InputParameters,
    template_file: Path | str,
    mode: TOMLMode = "full",
    only_structs: bool | None = None,
) -> None:
    """Write a set of input parameters to a template file.

    Parameters
    ----------
    inputs
        The inputs to write to file.
    template_file
        The path of the output.
    """
    assert mode in ("full", "minimal"), "mode must be 'full' or 'minimal'"

    if only_structs is None:
        only_structs = mode == "minimal"

    inputs_dct = prepare_inputs_for_serialization(
        inputs,
        mode=mode,
        only_structs=only_structs,
        # Templates should remain portable and contain only user-settable inputs,
        # not derived cached cosmology tables.
        include_cosmo_tables="never",
    )

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
