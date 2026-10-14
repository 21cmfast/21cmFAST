"""Changes to the file format between v4.1.0 and v4.2.0."""

from __future__ import annotations

import numpy as np

from .._compat_framework import (
    AddParameter,
    ConvertParameters,
    FormatRelease,
    RawInputs,
)


def _recomb_model(inputs: RawInputs) -> None:
    opts = inputs.structs.setdefault("astro_options", {})
    if "INHOMO_RECO" in opts:
        inhomo = bool(opts.pop("INHOMO_RECO"))
        opts.setdefault("RECOMB_MODEL", "inhomogeneous" if inhomo else "none")


def _sigma_sfr_index(inputs: RawInputs) -> None:
    # SIGMA_SFR_INDEX was given in base e, and is now in dex.
    params = inputs.structs.get("astro_params", {})
    if "SIGMA_SFR_INDEX" in params:
        params["SIGMA_SFR_INDEX"] = params["SIGMA_SFR_INDEX"] / np.log(10)


RELEASE = FormatRelease(
    "4.1.0",
    "4.2.0",
    [
        ConvertParameters(
            _recomb_model,
            "astro_options.INHOMO_RECO replaced by astro_options.RECOMB_MODEL "
            "(True -> 'inhomogeneous', False -> 'none')",
        ),
        ConvertParameters(
            _sigma_sfr_index,
            "astro_params.SIGMA_SFR_INDEX is now in dex rather than base e, so "
            "values in older files are divided by ln(10)",
        ),
        AddParameter(
            "astro_options",
            "LYA_MULTIPLE_SCATTERING",
            explanation="its default (False) reproduces older versions.",
        ),
        AddParameter(
            "astro_options",
            "USE_ADIABATIC_FLUCTUATIONS",
            explanation="its default (True) reproduces older versions.",
        ),
    ],
)
