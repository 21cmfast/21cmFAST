"""Utilities for interacting with 21cmFAST data structures."""

from operator import eq
from typing import Any

from .wrapper.inputs import InputParameters

# allow print functions and ambiguous unicode characters
# ruff: noqa: RUF001


def recursive_difference(
    dct1: dict | Any, dct2: dict | Any, cmprules: dict[type, callable] | None = None
) -> dict:
    """Return the recursive difference between two dicts.

    This is an *asymmetric* difference, i.e. we only end with values from dct1, and
    only if it is not exactly the same as in dct2. Entries in dct2 that are not in
    dct1 are ignored.

    Only dictionaries (and their subclasses) are recursed -- all other containers are
    treated as primitive types to be compared. If you need to recurse into custom
    structures, first "unstructure" your objects into dictionaries (e.g. with
    ``attrs.asdict(obj)``) then use this function.
    """
    cmprules = cmprules or {}

    out = {}
    for key, val1 in dct1.items():
        if isinstance(val1, dict):
            if not isinstance(dct2.get(key, None), dict):
                diff = val1
            else:
                diff = recursive_difference(val1, dct2[key], cmprules)
            if diff:
                out[key] = diff
        else:
            _eq = cmprules.get(type(val1), eq)
            if not _eq(val1, dct2.get(key, None)):
                out[key] = val1

    return out


def show_references(inputs: InputParameters, print_to_stdout=True):
    """Print out all the relevant references for a simulation based on input parameters."""
    ref_string = (
        "The main reference for 21cmFAST v3+:\n"
        "====================================\n"
        "Murray et al., (2020). 21cmFAST v3: A Python-integrated C code for generating 3D\n"
        "realizations of the cosmic 21cm signal. Journal of Open Source Software, 5(54),\n"
        "2582, https://doi.org/10.21105/joss.02582\n\n"
        "Based on the original 21cmFAST model:\n"
        "=====================================\n"
        'Andrei Mesinger, Steven Furlanetto and Renyue Cen, "21CMFAST: a fast, seminumerical\n'
        'simulation of the high-redshift 21-cm signal", Monthly Notices of the Royal\n'
        "Astronomical Society, Volume 411, Issue 2, pp. 955-972 (2011),\n"
        "https://ui.adsabs.harvard.edu/link_gateway/2011MNRAS.411..955M/doi:10.1111/j.1365-2966.2010.17731.x\n\n"
    )

    if inputs.astro_options.USE_MASS_DEPENDENT_ZETA:
        ref_string += (
            "The mass-dependent ionising efficiency model:\n"
            "=============================================\n"
            "Park, J., Mesinger, A., Greig, B., and Gillet, N.,\n"
            "“Inferring the astrophysics of reionization and cosmic dawn from galaxy luminosity\n"
            "functions and the 21-cm signal”, Monthly Notices of the Royal Astronomical Society,\n"
            "vol. 484, no. 1, pp. 933–949, 2019. https://doi.org/10.1093/mnras/stz032.\n\n"
        )

    if inputs.astro_options.USE_MINI_HALOS:
        ref_string += (
            "The minihalo model was first introduced in:\n"
            "===========================================\n"
            "Qin, Y., Mesinger, A., Park, J., Greig, B., and Muñoz, J. B.,\n"
            "“A tale of two sites - I. Inferring the properties of minihalo-hosted galaxies from\n"
            "current observations”, Monthly Notices of the Royal Astronomical Society, vol. 495,\n"
            "no. 1, pp. 123–140, 2020. https://doi.org/10.1093/mnras/staa1131.\n\n"
            "With improvements and DM-baryon relative velocity model in:\n"
            "===========================================================\n"
            "Muñoz, J.B., Qin, Y., Mesinger, A., Murray, S., Greig, B., and Mason, C.,\n"
            '"The Impact of the First Galaxies on Cosmic Dawn and Reionization",\n'
            "Monthly Notices of the Royal Astronomical Society, vol. 511, no. 3,\n"
            "pp 3657-3681, 2022 https://doi.org/10.1093/mnras/stac185\n"
        )

    if inputs.matter_options.lagrangian_source_grid:
        ref_string += (
            "Version 4 and the discrete halo sampler is described in:\n"
            "========================================================\n"
            "Davies, J. E., Mesinger, A., Murray, S. G.,\n"
            '"Efficient simulation of discrete galaxy populations and associated radiation\n'
            'fields during the first billion years",\n'
            "eprint arXiv:2504.17254, 2025. https://doi.org/10.48550/arXiv.2504.17254\n\n"
        )

    if print_to_stdout:
        print(ref_string)  # noqa: T201

    return ref_string
