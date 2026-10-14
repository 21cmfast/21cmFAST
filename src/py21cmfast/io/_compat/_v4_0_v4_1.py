"""Changes to the file format between v4.0.0 and v4.1.0."""

from __future__ import annotations

from .._compat_framework import (
    AddField,
    AddParameter,
    FormatRelease,
    RawInputs,
    RawOutputStruct,
    warn,
)


def _min_xe(inputs: RawInputs) -> float | None:
    # The threshold was only applied for single-cell (global) simulations, so that a
    # threshold of zero reproduces older versions. In all other cases, the default
    # is fine.
    return 0.0 if inputs.get("simulation_options", "HII_DIM") == 1 else None


def _q_hi(struct: RawOutputStruct) -> None:
    if struct.inputs.get("simulation_options", "HII_DIM") == 1:
        warn(
            f"This TsBox (written by 21cmFAST v{struct.version}) has no Q_HI, which "
            "is required by single-cell (global) simulations. It is set to 1."
        )


RELEASE = FormatRelease(
    "4.0.0",
    "4.1.0",
    [
        AddParameter(
            "simulation_options",
            "MIN_XE_FOR_FCOLL_IN_TAUX",
            _min_xe,
            "the mean x_e above which n_ion is set to zero in the X-ray optical "
            "depth (only used when HII_DIM=1). Set to zero for older files with "
            "HII_DIM=1, reproducing their behaviour.",
        ),
        AddField(
            "TsBox",
            "Q_HI",
            _q_hi,
            "it is left at its default value (1) for older files. It is only "
            "used for single-cell (global) simulations.",
        ),
    ],
)
