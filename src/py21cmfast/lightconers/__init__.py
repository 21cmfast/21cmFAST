"""Classes that make lightcones from the coeval boxes of a simulation.

:func:`~py21cmfast.run_lightcone` takes a *lightconer*: an object that is given the
fields of each node redshift in turn, and returns the lightcone piece by piece. What
such an object must provide is defined in :mod:`~py21cmfast.lightconers.protocol`
(:class:`LightconerProtocol`).

21cmFAST itself only comes with :class:`RectilinearLightconer`. Other lightconers,
such as angular (and HEALPix) ones, are in the ``tuesday`` package
(:mod:`tuesday.core.lightcones`). Until v5, :class:`AngularLightconer` can still be
got from here, if ``tuesday`` is installed.
"""

from ._base import (
    CoevalSnapshot,
    LegacyLightconerAPI,
    LightconerSpec,
    context_from_inputs,
    redshift_interpolation,
    slices_between,
)
from ._legacy import _LIGHTCONERS, Lightconer
from ._prepare import prepare_lightconer
from .protocol import (
    Chunk,
    LightconeContext,
    LightconerProtocol,
    LightconeSession,
    Snapshot,
)
from .rectilinear import RectilinearLightconer, RectilinearSession

__all__ = [
    "AngularLightconer",
    "Chunk",
    "CoevalSnapshot",
    "LegacyLightconerAPI",
    "LightconeContext",
    "LightconeSession",
    "Lightconer",
    "LightconerProtocol",
    "LightconerSpec",
    "RectilinearLightconer",
    "RectilinearSession",
    "Snapshot",
    "context_from_inputs",
    "prepare_lightconer",
    "redshift_interpolation",
    "slices_between",
]

# The classes are part of this package, wherever in it they are defined. Their full
# name is also recorded in files (by tuesday), so it should not change.
for _cls in (Lightconer, RectilinearLightconer, RectilinearSession):
    _cls.__module__ = __name__
del _cls

# The lightconers of 21cmFAST count as (virtual) subclasses of the base class of
# v4.3, so that ``isinstance(lightconer, Lightconer)`` remains true for them.
Lightconer.register(RectilinearLightconer)
_LIGHTCONERS["RectilinearLightconer"] = RectilinearLightconer


def __getattr__(name: str):
    """Import the lightconers that need ``tuesday`` only when they are asked for."""
    if name == "AngularLightconer":
        from .angular import AngularLightconer

        return AngularLightconer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
