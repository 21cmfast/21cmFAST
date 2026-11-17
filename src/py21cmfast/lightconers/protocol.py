"""The protocol that an object must follow to make lightcones in 21cmFAST.

:func:`~py21cmfast.run_lightcone` does not make lightcones itself. It computes the
coeval boxes of each node redshift, hands each one to a *lightconer*, and stores
whatever the lightconer gives back. How a lightcone is made from the boxes (which
boxes are used for which part of the lightcone, how they are interpolated in redshift
and projected on the sky, and what is cached along the way) is up to the lightconer.

This module defines what the driver expects of a lightconer. There are two objects:

:class:`LightconerProtocol`
    What the user makes and passes to :func:`~py21cmfast.run_lightcone`. It describes
    the lightcone (its quantities, its distances, its geometry) and does not change
    during a run.

:class:`LightconeSession`
    The state of one run, made by :meth:`LightconerProtocol.start`. The driver passes
    it each :class:`Snapshot` (the fields of one node redshift) with
    :meth:`~LightconeSession.ingest`, and gets back :class:`Chunk` objects: pieces of
    the lightcone that are now known. The session keeps whatever it needs between
    calls: earlier snapshots, interpolators, anything else.

These are :class:`typing.Protocol` classes, so a lightconer does not need to inherit
from them, or to import 21cmFAST at all: it only needs the attributes and methods
listed here. :class:`~py21cmfast.RectilinearLightconer` is a short, complete example.

The loop that the driver runs is, in essence::

    session = lightconer.start(context)
    lightcones = {q: np.zeros(session.shape) for q in lightconer.quantities}

    for snapshot in snapshots:  # from high to low redshift
        for quantity, index, values in session.ingest(snapshot):
            lightcones[quantity][index] = values

    for quantity, index, values in session.finish():
        lightcones[quantity][index] = values
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, NamedTuple, Protocol, Self, runtime_checkable

import attrs
import numpy as np
from astropy.cosmology import FLRW
from astropy.units import Quantity

__all__ = [
    "Chunk",
    "LightconeContext",
    "LightconeSession",
    "LightconerProtocol",
    "Snapshot",
]


class Chunk(NamedTuple):
    """A piece of a lightcone, returned by a :class:`LightconeSession`.

    The driver stores it with ``lightcones[quantity][index] = values``.
    """

    #: The name of the quantity (one of the lightconer's ``quantities``).
    quantity: str
    #: Where the values go in the lightcone of that quantity: any index that numpy
    #: accepts for an array of shape :attr:`LightconeSession.shape`. For a whole
    #: slice at index ``i`` along the line of sight this is ``(..., i)``.
    index: Any
    #: The values, of the shape that ``index`` selects.
    values: np.ndarray


@attrs.frozen(kw_only=True)
class LightconeContext:
    """What a lightconer is told about the simulation, when a run starts.

    It holds plain values only (no 21cmFAST objects), so that a lightconer can be
    written, and used, without 21cmFAST.
    """

    #: The cosmology of the simulation.
    cosmo: FLRW = attrs.field()
    #: The number of cells of the coeval boxes along each axis.
    box_shape: tuple[int, int, int] = attrs.field(
        converter=lambda x: tuple(int(n) for n in x)
    )
    #: The comoving size of a cell of the coeval boxes.
    cell_size: Quantity = attrs.field()
    #: The redshifts of the snapshots that will be passed to the session, in order.
    node_redshifts: tuple[float, ...] = attrs.field(
        converter=lambda x: tuple(float(z) for z in x)
    )


@runtime_checkable
class Snapshot(Protocol):
    """The fields of the simulation at one node redshift.

    The driver passes one of these to :meth:`LightconeSession.ingest` for each node
    redshift, from high to low redshift.
    """

    #: The redshift of the snapshot.
    redshift: float

    def __getitem__(self, field: str) -> np.ndarray:
        """Return the box of a field, an array of shape ``context.box_shape``.

        Any of the lightconer's :meth:`~LightconerProtocol.required_fields` can be
        asked for. A box may only be computed, or read from disk, when it is first
        asked for, so a session should only ask for the boxes it uses. A session can
        keep a snapshot (or its boxes) for as long as it needs it, but should ask
        for the boxes it will need later during the ``ingest`` call that passes the
        snapshot: the driver may release the memory of the simulation afterwards.
        """


@runtime_checkable
class LightconeSession(Protocol):
    """The state of the making of a lightcone during one run.

    It is made by :meth:`LightconerProtocol.start`, and is the place to keep anything
    that is needed between snapshots: earlier snapshots, interpolators, and so on.
    """

    #: The shape of the lightcone of each quantity. The last axis is the line of
    #: sight, and has one element for each of the lightconer's ``lc_distances``.
    shape: tuple[int, ...]

    #: How many of the latest snapshots must be passed again to a new session for it
    #: to carry on after them (1 if each part of the lightcone is made from two
    #: consecutive snapshots). Used to resume a run from a checkpoint.
    resume_depth: int

    def ingest(self, snapshot: Snapshot) -> Iterable[Chunk]:
        """Take the next snapshot, and return the parts of the lightcone now known.

        Snapshots are passed in order of decreasing redshift. Nothing needs to be
        returned for a given snapshot: a session can return a part of the lightcone
        whenever it has what it needs to make it.
        """

    def finish(self) -> Iterable[Chunk]:
        """Return the parts of the lightcone that were not returned yet.

        Called once, after the last snapshot.
        """


@runtime_checkable
class LightconerProtocol(Protocol):
    """What :func:`~py21cmfast.run_lightcone` needs of a lightconer.

    A lightconer may also have the attribute ``los_velocity_rms_factor``, the ratio
    of the rms of its ``"los_velocity"`` quantity to the rms of the full (3D)
    velocity. It is only used to choose how far the lightcone is extended when
    redshift-space distortions are applied, and is taken to be 1 if absent.
    """

    #: The names of the quantities that lightcones are made of. Usually these are
    #: names of fields of the simulation, but a lightconer may define others. The
    #: driver may ask for ``"los_velocity"`` (the velocity along the line of sight,
    #: used for redshift-space distortions) and ``"tau_21"``.
    quantities: tuple[str, ...]

    #: The comoving distances of the slices of the lightcone (its last axis).
    lc_distances: Quantity

    def evolve(self, **changes: Any) -> Self:
        """Return a copy of the lightconer with some attributes changed.

        The driver uses this to add quantities (``quantities=...``) and to extend
        the lightcone along the line of sight (``lc_distances=...``), when it applies
        redshift-space distortions.
        """

    def required_fields(self) -> tuple[str, ...]:
        """Return the names of the fields that are needed to make the quantities."""

    def start(self, context: LightconeContext) -> LightconeSession:
        """Start a run, and return the session that will make the lightcone."""
