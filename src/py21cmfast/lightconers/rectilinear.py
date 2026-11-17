"""A lightconer for rectilinear lightcones.

This is the lightconer that comes with 21cmFAST. It is also meant as an example of
how to write one: everything that :func:`~py21cmfast.run_lightcone` needs of a
lightconer is in :class:`RectilinearLightconer` (``quantities``, ``lc_distances``,
``evolve``, ``required_fields`` and ``start``) and in :class:`RectilinearSession`
(``shape``, ``resume_depth``, ``ingest`` and ``finish``). See
:mod:`py21cmfast.lightconers.protocol`.
"""

from __future__ import annotations

from collections.abc import Iterator

import attrs
import numpy as np
from astropy.units import Quantity, pixel, pixel_scale

from ._base import (
    _LENGTH,
    LegacyLightconerAPI,
    LightconerSpec,
    _warn_deprecated,
    redshift_interpolation,
    slices_between,
)
from .protocol import Chunk, LightconeContext, Snapshot


@attrs.define(kw_only=True, slots=False)
class RectilinearLightconer(LegacyLightconerAPI, LightconerSpec):
    """A lightconer whose lightcones are boxes: two transverse axes, and distance.

    Each slice of the lightcone is a slice of the (periodic) coeval box,
    perpendicular to its last axis, interpolated linearly in comoving distance
    between the two snapshots around it. From one slice of the lightcone to the
    next, the slice of the box also moves by one cell, so that structures are
    continuous along the line of sight.

    Parameters
    ----------
    lc_distances
        The comoving distances to the lightcone slices, in Mpc. Either this or
        the ``lc_redshifts`` must be provided. The distances should be one cell
        apart (see :meth:`between_redshifts`).
    lc_redshifts
        The redshifts of the lightcone slices. Either this or the ``lc_distances``
        must be provided.
    cosmo
        The cosmology to use. Defaults to Planck18.
    quantities
        The quantities to make lightcones of. These are fields of the simulation
        (attributes of :class:`~py21cmfast.Coeval`), or the special value
        ``"los_velocity"``, the velocity along the line of sight. Defaults to
        ``("brightness_temp",)``.
    interp_kinds
        The kind of interpolation in distance to use for each quantity (see
        :func:`~py21cmfast.lightconers.redshift_interpolation`).
    index_offset
        The index of the slice of the coeval box that is used for the last (most
        distant) slice of the lightcone, plus one. By default, the number of slices
        of the lightcone.
    line_of_sight_axis
        The axis of the coeval boxes that is the line of sight. Only the last axis
        is supported at the moment.
    """

    index_offset: int = attrs.field()
    line_of_sight_axis: int = attrs.field(
        default=-1, converter=int, validator=attrs.validators.in_([-1, 0, 1, 2])
    )

    #: The ratio of the rms of the velocity along the line of sight to that of the
    #: full velocity: one component of an isotropic field.
    los_velocity_rms_factor = 1.0 / np.sqrt(3.0)

    @index_offset.default
    def _index_offset_default(self) -> int:
        # While it probably makes more sense to use zero as the default offset,
        # we use n_lightcone to maintain default backwards compatibility.
        return len(self.lc_distances)

    # ------------------------------------------------------------------
    # The lightconer protocol
    # ------------------------------------------------------------------
    def required_fields(self) -> tuple[str, ...]:
        """Return the names of the fields that are needed to make the quantities.

        These are the quantities themselves, except that ``"los_velocity"`` is made
        from the field ``"velocity_z"``.
        """
        return tuple(dict.fromkeys(self._field_of(q) for q in self.quantities))

    def start(self, context: LightconeContext) -> RectilinearSession:
        """Start a run, and return the session that will make the lightcone."""
        return RectilinearSession(self, context)

    # ------------------------------------------------------------------
    # How a slice is made
    # ------------------------------------------------------------------
    @staticmethod
    def _field_of(quantity: str) -> str:
        """Return the name of the field that a quantity is made from."""
        # TODO: get the correct component according to self.line_of_sight_axis
        return "velocity_z" if quantity == "los_velocity" else quantity

    def box_indices(self, slice_distances: np.ndarray) -> np.ndarray:
        """Return the index of the slice of the coeval box used for each lightcone slice.

        The back of the lightcone lines up with the back of the coeval box (shifted
        by ``index_offset``), and a slice that is one cell closer uses the slice of
        the box that is one cell closer. The indices may be outside the box, which
        is periodic.

        Parameters
        ----------
        slice_distances
            The comoving distances of the slices of the lightcone, in cells.
        """
        from_back = (slice_distances.max() - slice_distances + 1).astype(int)
        return self.index_offset - from_back

    def coeval_subselect(
        self, lcd: Quantity[pixel], coeval: np.ndarray, coeval_res: Quantity[_LENGTH]
    ) -> np.ndarray:
        """Sub-select the coeval slice corresponding to this coeval distance.

        .. deprecated:: 4.4.0
            Use :meth:`box_indices`.
        """
        _warn_deprecated("coeval_subselect", "Use box_indices instead.")
        # This makes the back of the lightcone exactly line up with the back of the
        # coeval box at that redshift, modulo the index_offset.
        lcpix = self.get_lc_distances_in_pixels(coeval_res)
        lcidx = int((lcpix.max() - lcd + 1 * pixel).to_value(pixel))
        return coeval.take(-lcidx + self.index_offset, axis=2, mode="wrap")


class RectilinearSession:
    """The making of one rectilinear lightcone.

    Each slice of the lightcone is made from two consecutive snapshots, so the only
    thing that is kept between snapshots is the previous one.
    """

    #: A new session only needs the latest snapshot to carry on after it.
    resume_depth = 1

    def __init__(self, lightconer: RectilinearLightconer, context: LightconeContext):
        self.lightconer = lightconer
        self.context = context

        #: The shape of the lightcone of each quantity.
        self.shape = (*context.box_shape[:2], len(lightconer.lc_distances))

        # The distances of the slices of the lightcone, in cells.
        self._to_cells = pixel_scale(context.cell_size / pixel)
        self._slice_distances = lightconer.lc_distances.to_value(pixel, self._to_cells)

        self._box_indices = lightconer.box_indices(self._slice_distances)

        self._previous: Snapshot | None = None
        self._previous_distance: float | None = None

    def _distance(self, redshift: float) -> np.float64:
        """Return the comoving distance to a redshift, in cells."""
        distance = self.context.cosmo.comoving_distance(redshift)
        return np.float64(distance.to_value(pixel, self._to_cells))

    def ingest(self, snapshot: Snapshot) -> Iterator[Chunk]:
        """Make the slices that lie between this snapshot and the one before it."""
        distance = self._distance(snapshot.redshift)
        previous, previous_distance = self._previous, self._previous_distance
        self._previous, self._previous_distance = snapshot, distance

        if previous is None:
            # Nothing can be made from the first snapshot alone.
            return

        lightconer = self.lightconer
        for i in slices_between(self._slice_distances, distance, previous_distance):
            j = self._box_indices[i]
            for quantity in lightconer.quantities:
                field = lightconer._field_of(quantity)
                values = redshift_interpolation(
                    self._slice_distances[i],
                    snapshot[field].take(j, axis=2, mode="wrap"),
                    previous[field].take(j, axis=2, mode="wrap"),
                    distance,
                    previous_distance,
                    kind=lightconer.interp_kinds.get(field, "mean"),
                )
                yield Chunk(quantity, (..., int(i)), values)

    def finish(self) -> tuple[Chunk, ...]:
        """Return nothing: every slice was returned by ``ingest``."""
        self._previous = None
        return ()
