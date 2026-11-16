"""The angular lightconer, which is now part of ``tuesday``.

Lightcones other than rectilinear ones are made by the lightconers of the ``tuesday``
package (:mod:`tuesday.core.lightcones`), which follow the
:class:`~py21cmfast.lightconers.LightconerProtocol`. Until v5,
:class:`AngularLightconer` remains available from 21cmFAST, with the methods and the
defaults that it had in v4.3, if ``tuesday`` is installed (``pip install tuesday-eor``,
or ``pip install 21cmFAST[lightcones]``).
"""

from __future__ import annotations

from typing import Self

import attrs
from astropy.cosmology import FLRW

from ..wrapper.inputs import Planck18, SimulationOptions
from ._base import LegacyLightconerAPI
from ._legacy import _LIGHTCONERS, Lightconer

_INSTALL = (
    "Install it with `pip install tuesday-eor` (or `pip install 21cmFAST[lightcones]`). "
    "From v5, use tuesday.core.lightcones.AngularLightconer directly."
)

try:
    import tuesday.core.lightcones as _lightcones
except ImportError as e:
    raise ImportError(
        "AngularLightconer is now part of the 'tuesday' package, which is not "
        f"installed. {_INSTALL}"
    ) from e

try:
    _AngularLightconer = _lightcones.AngularLightconer
except AttributeError as e:
    raise ImportError(
        "AngularLightconer is now part of the 'tuesday' package, but the version of "
        f"tuesday that is installed is too old to have it. {_INSTALL}"
    ) from e


@attrs.define(kw_only=True, slots=False)
class AngularLightconer(LegacyLightconerAPI, _AngularLightconer):
    """Angular lightcone slices constructed from rectlinear coevals.

    This is :class:`tuesday.core.lightcones.AngularLightconer` (see there for the
    parameters), with the cosmology of 21cmFAST as the default, and the methods that
    lightconers had in 21cmFAST v4.3. It will be removed from 21cmFAST in v5.
    """

    #: The cosmology that is used if none is given: the Planck18 of 21cmFAST.
    default_cosmo = Planck18

    @classmethod
    def like_rectilinear(
        cls,
        simulation_options: SimulationOptions,
        match_at_z: float,
        cosmo: FLRW = Planck18,
        **kw,
    ) -> Self:
        """Create an angular lightconer with the same pixel size as a rectilinear one.

        This is useful for comparing the two lightconer types.

        Parameters
        ----------
        simulation_options
            The user parameters.
        match_at_z
            The redshift at which the angular lightconer should match the rectilinear
            one.
        cosmo
            The cosmology to use.

        Other Parameters
        ----------------
        All other parameters passed through to the constructor.

        Returns
        -------
        AngularLightconer
            The angular lightconer.
        """
        return super().like_rectilinear(
            n_cells=simulation_options.HII_DIM,
            cell_size=simulation_options.cell_size,
            match_at_z=match_at_z,
            cosmo=cosmo,
            **kw,
        )


AngularLightconer.__module__ = "py21cmfast.lightconers"
Lightconer.register(AngularLightconer)
_LIGHTCONERS["AngularLightconer"] = AngularLightconer
