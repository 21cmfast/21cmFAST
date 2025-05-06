"""Utilities for interacting with 21cmFAST data structures."""

from .wrapper.inputs import InputParameters
from .wrapper.outputs import InitialConditions, OutputStruct

# allow print functions and ambiguous unicode characters
# ruff: noqa: RUF001


def get_all_fieldnames(
    arrays_only=True, lightcone_only=False, as_dict=False
) -> dict[str, str] | set[str]:
    """Return all possible fieldnames in output structs.

    Parameters
    ----------
    arrays_only : bool, optional
        Whether to only return fields that are arrays.
    lightcone_only : bool, optional
        Whether to only return fields from classes that evolve with redshift.
    as_dict : bool, optional
        Whether to return results as a dictionary of ``quantity: class_name``.
        Otherwise returns a set of quantities.
    """
    classes = [cls.dummy() for cls in OutputStruct._implementations()]

    if not lightcone_only:
        classes.append(InitialConditions())

    attr = "pointer_fields" if arrays_only else "fieldnames"

    if as_dict:
        return {
            name: cls.__class__.__name__
            for cls in classes
            for name in getattr(cls, attr)
        }
    else:
        return {name for cls in classes for name in getattr(cls, attr)}


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

    if inputs.matter_options.USE_HALO_FIELD:
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
