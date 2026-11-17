"""Code shared by the lightconers of 21cmFAST."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from functools import cached_property, partial
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Self

import attrs
import deprecation
import numpy as np
from astropy.cosmology import FLRW, z_at_value
from astropy.units import Mpc, Quantity, pixel, pixel_scale

from ..wrapper.inputs import Planck18  # Not *quite* the same as astropy's Planck18
from .protocol import LightconeContext, Snapshot

if TYPE_CHECKING:
    from classy import Class

    from ..wrapper.inputs import InputParameters, SimulationOptions

_LENGTH = "length"

#: The version in which the methods of lightconers that the driver used to call
#: (before the lightconer protocol) were deprecated.
_DEPRECATED_IN = "4.4.0"


def _warn_deprecated(name: str, instead: str) -> None:
    """Warn that a method of the lightconers of 21cmFAST v4.3 is deprecated."""
    warnings.warn(
        deprecation.DeprecatedWarning(
            name,
            deprecated_in=_DEPRECATED_IN,
            removed_in="5.0.0",
            details=f"{name} is deprecated. {instead}",
        ),
        stacklevel=3,
    )


def redshift_interpolation(
    dc: float,
    coeval_a: np.ndarray,
    coeval_b: np.ndarray,
    dc_a: float,
    dc_b: float,
    kind: str = "mean",
) -> np.ndarray:
    """Interpolate two boxes of a field linearly in comoving distance.

    Parameters
    ----------
    dc
        The comoving distance to interpolate to.
    coeval_a, coeval_b
        The field at the two distances. They must have the same shape.
    dc_a, dc_b
        The comoving distances of the two boxes, in the same units as ``dc``.
    kind
        Either "mean", for the linear interpolation, or "mean_max", which is the
        same, except where the two boxes have opposite signs, where the larger of
        the two values is taken. This is meant for fields like ``z_reion`` in which
        a negative value means "not yet".

    Returns
    -------
    np.ndarray
        The interpolated field.
    """
    if coeval_a.shape != coeval_b.shape:
        raise ValueError("coeval_a and coeval_b must have the same shape")

    out = (np.abs(dc_b - dc) * coeval_a + np.abs(dc_a - dc) * coeval_b) / np.abs(
        dc_a - dc_b
    )

    if kind == "mean_max":
        flag = coeval_a * coeval_b < 0
        out[flag] = np.maximum(coeval_a, coeval_b)[flag]
    elif kind != "mean":
        raise ValueError("kind must be 'mean' or 'mean_max'")

    return out


def slices_between(
    slice_distances: np.ndarray, d_a: float, d_b: float
) -> np.ndarray[int]:
    """Return the indices of the slices that lie between two distances.

    A slice at the smaller of the two distances is included, one at the larger is
    not, so that every slice belongs to one pair of consecutive snapshots.
    """
    dmin, dmax = min(d_a, d_b), max(d_a, d_b)
    # At the lower redshift, we include some tolerance. This is because the very
    # last slice (lowest redshift) may correspond *exactly* to the lowest coeval
    # box, and due to rounding error in the `z_at_value` call, they might be
    # slightly off.
    return np.nonzero(
        (slice_distances >= dmin * (1 - 1e-6)) & (slice_distances < dmax)
    )[0]


class CoevalSnapshot:
    """The fields of a coeval box, as a :class:`~.protocol.Snapshot`.

    Parameters
    ----------
    coeval
        An object with the attribute ``redshift``, and with each field as an
        attribute, such as a :class:`~py21cmfast.Coeval`.

    Notes
    -----
    A field is only got from the coeval box when it is first asked for (which may
    read it from disk), and a reference to it is kept, so that it remains available
    after the memory of the coeval box is released.
    """

    def __init__(self, coeval: Any):
        self.coeval = coeval
        self.redshift = float(coeval.redshift)
        self._boxes = {}

    def __getitem__(self, field: str) -> np.ndarray:
        """Return the box of a field."""
        if field not in self._boxes:
            try:
                self._boxes[field] = getattr(self.coeval, field)
            except AttributeError as e:
                raise KeyError(
                    f"The field '{field}' is not available at z={self.redshift}"
                ) from e
        return self._boxes[field]


class _SnapshotAsCoeval:
    """A snapshot that looks like a Coeval, for lightconers written for v4.3."""

    def __init__(
        self,
        snapshot: Snapshot,
        simulation_options: Any,
        cosmo_params: Any,
    ):
        self._snapshot = snapshot
        self.redshift = snapshot.redshift
        self.simulation_options = simulation_options
        self.cosmo_params = cosmo_params

    def __getattr__(self, name: str) -> np.ndarray:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._snapshot[name]
        except KeyError as e:
            raise AttributeError(name) from e


def box_shape_of(simulation_options: SimulationOptions) -> tuple[int, int, int]:
    """Return the shape of the (low-resolution) coeval boxes of a simulation."""
    n = int(simulation_options.HII_DIM)
    factor = getattr(simulation_options, "NON_CUBIC_FACTOR", 1.0)
    return (n, n, round(factor * n))


def context_from_inputs(
    inputs: InputParameters, node_redshifts: tuple[float, ...] | None = None
) -> LightconeContext:
    """Return what a lightconer is told about a simulation with these inputs.

    Parameters
    ----------
    inputs
        The input parameters of the simulation.
    node_redshifts
        The redshifts of the snapshots that will be passed to the lightconer, if
        they are not ``inputs.node_redshifts``.
    """
    if node_redshifts is None:
        node_redshifts = inputs.node_redshifts
    return LightconeContext(
        cosmo=inputs.cosmo_params.cosmo,
        box_shape=box_shape_of(inputs.simulation_options),
        cell_size=inputs.simulation_options.cell_size,
        node_redshifts=node_redshifts,
    )


def _options_from_context(context: LightconeContext) -> SimpleNamespace:
    """Return the parts of a SimulationOptions that a v4.3 lightconer may use."""
    n = context.box_shape[0]
    return SimpleNamespace(
        HII_DIM=n,
        BOX_LEN=(n * context.cell_size).to_value(Mpc),
        NON_CUBIC_FACTOR=context.box_shape[2] / n,
        cell_size=context.cell_size,
    )


@attrs.define(kw_only=True, slots=False)
class LightconerSpec:
    """The description of a lightcone that the lightconers of 21cmFAST share.

    Parameters
    ----------
    lc_distances
        The comoving distances to the lightcone slices, in Mpc. Either this or
        the ``lc_redshifts`` must be provided.
    lc_redshifts
        The redshifts of the lightcone slices. Either this or the ``lc_distances``
        must be provided.
    cosmo
        The cosmology to use. Defaults to Planck18.
    quantities
        An iteratable of quantities to include in the lightcone slices. These should
        be fields of the simulation (attributes of :class:`~py21cmfast.Coeval`), or
        the special value ``"los_velocity"``, the velocity along the line of sight.
        Defaults to ``("brightness_temp",)``.
    interp_kinds
        The kind of interpolation in distance to use for each quantity (see
        :func:`redshift_interpolation`). Quantities that are not given use "mean".
        Defaults to ``{"z_reion": "mean_max"}``.
    """

    cosmo: FLRW = attrs.field(
        default=Planck18, validator=attrs.validators.instance_of(FLRW)
    )

    _lc_redshifts: np.ndarray = attrs.field(default=None, eq=False)
    lc_distances: Quantity[_LENGTH] = attrs.field(
        eq=attrs.cmp_using(eq=partial(np.allclose, rtol=1e-5, atol=0))
    )

    quantities: tuple[str] = attrs.field(
        default=("brightness_temp",),
        converter=tuple,
        validator=attrs.validators.deep_iterable(attrs.validators.instance_of(str)),
    )
    interp_kinds: dict[str, str] = attrs.field()

    @lc_distances.default
    def _lcd_default(self):
        if self._lc_redshifts is None:
            raise ValueError("Either lc_distances or lc_redshifts must be provided")
        return self.cosmo.comoving_distance(self._lc_redshifts)

    @lc_distances.validator
    def _lcd_vld(self, attribute, value):
        if np.any(value < 0):
            raise ValueError("lc_distances must be non-negative")

    @_lc_redshifts.validator
    def _lcz_vld(self, attribute, value):
        if value is None:
            return

        if np.any(value < 0):
            raise ValueError("lc_redshifts must be non-negative")

    @cached_property
    def lc_redshifts(self) -> np.ndarray:
        """The redshifts of the lightcone slices."""
        if self._lc_redshifts is not None:
            return self._lc_redshifts

        d = self.lc_distances
        zmin = z_at_value(self.cosmo.comoving_distance, d.min()).value
        zmax = z_at_value(self.cosmo.comoving_distance, d.max()).value

        zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 100)
        dgrid = self.cosmo.comoving_distance(zgrid)
        return np.interp(d.value, dgrid.value, zgrid)

    def get_lc_distances_in_pixels(self, resolution: Quantity[_LENGTH]):
        """Get the lightcone distances in pixels, given a resolution."""
        return self.lc_distances.to(pixel, pixel_scale(resolution / pixel))

    @interp_kinds.default
    def _interp_kinds_def(self):
        return {"z_reion": "mean_max"}

    def evolve(self, **changes: Any) -> Self:
        """Return a copy of the lightconer with some attributes changed.

        If ``lc_distances`` is changed, the redshifts are those of the new distances
        (unless ``lc_redshifts`` is given as well).
        """
        if "lc_distances" in changes:
            changes.setdefault("lc_redshifts", None)
        return attrs.evolve(self, **changes)

    @classmethod
    def between_redshifts(
        cls,
        min_redshift: float,
        max_redshift: float,
        resolution: Quantity[_LENGTH],
        cosmo=Planck18,
        **kw,
    ):
        """Construct a Lightconer with regular comoving dist. slices between two z's."""
        d_at_redshift = cosmo.comoving_distance(min_redshift).to_value(Mpc)
        dmax = cosmo.comoving_distance(max_redshift).to_value(Mpc)
        res = resolution.to_value(Mpc)

        lc_distances = np.arange(d_at_redshift, dmax + res, res)

        return cls(lc_distances=lc_distances * Mpc, cosmo=cosmo, **kw)

    @classmethod
    def with_equal_cdist_slices(
        cls,
        min_redshift: float,
        max_redshift: float,
        resolution: Quantity[_LENGTH],
        cosmo=Planck18,
        **kw,
    ):
        """
        Create a lightconer with equally spaced slices in comoving distance.

        This method is deprecated and will be removed in future versions.
        Instead, use `between_redshifts` to create a lightconer with equally spaced
        slices in comoving distance.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "with_equal_cdist_slices",
                deprecated_in="4.2.0",
                removed_in="5.0.0",
                details=(
                    "with_equal_cdist_slices is deprecated and will be removed in a future version. "
                    "Call between_redshifts instead to silence this warning."
                ),
            ),
            stacklevel=2,
        )
        return cls.between_redshifts(
            min_redshift=min_redshift,
            max_redshift=max_redshift,
            resolution=resolution,
            cosmo=cosmo,
            **kw,
        )


def _los_index(index: Any) -> int:
    """Return the index along the line of sight of a chunk that is one slice."""
    if isinstance(index, tuple):
        index = index[-1]
    return int(index)


class LegacyLightconerAPI:
    """The methods of the lightconers of v4.3 that the driver no longer calls.

    They are written with the methods of the
    :class:`~py21cmfast.lightconers.LightconerProtocol`, for lightconers that make
    each slice from the two snapshots around it, and will be removed in v5.
    """

    def make_lightcone_slices(
        self,
        c1: Any,
        c2: Any,
    ) -> Iterator[tuple[str | None, int | np.ndarray, np.ndarray | None]]:
        """
        Make lightcone slices out of two coeval objects.

        .. deprecated:: 4.4.0
            Use :meth:`start` and the session it returns.

        Parameters
        ----------
        c1, c2 : Coeval
            The coeval boxes to interpolate.

        Yields
        ------
        quantity
            The name of the quantity of the slice.
        lcidx
            The index of the lightcone to which the slice belongs.
        slice
            The quantity on the slice.
        """
        _warn_deprecated(
            "make_lightcone_slices",
            "Use lightconer.start(context), and the ingest() method of the session "
            "it returns, instead.",
        )
        if c1.simulation_options != c2.simulation_options:
            raise ValueError("c1 and c2 must have the same user parameters")
        if c1.cosmo_params != c2.cosmo_params:
            raise ValueError("c1 and c2 must have the same cosmological parameters")

        high, low = sorted((c1, c2), key=lambda c: c.redshift, reverse=True)
        context = LightconeContext(
            cosmo=c1.cosmo_params.cosmo,
            box_shape=box_shape_of(c1.simulation_options),
            cell_size=c1.simulation_options.cell_size,
            node_redshifts=(high.redshift, low.redshift),
        )
        session = self.start(context)

        empty = True
        for snapshot in (CoevalSnapshot(high), CoevalSnapshot(low)):
            for chunk in session.ingest(snapshot):
                empty = False
                yield chunk.quantity, _los_index(chunk.index), chunk.values
        for chunk in session.finish():
            empty = False
            yield chunk.quantity, _los_index(chunk.index), chunk.values

        if empty:
            yield None, np.array([], dtype=int), None

    def redshift_interpolation(
        self,
        dc: float,
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: float,
        dc_b: float,
        kind: str = "mean",
    ) -> np.ndarray:
        """Perform redshift interpolation to a new box given two bracketing coevals.

        .. deprecated:: 4.4.0
            Use :func:`py21cmfast.lightconers.redshift_interpolation`.
        """
        _warn_deprecated(
            "redshift_interpolation",
            "Use the function py21cmfast.lightconers.redshift_interpolation instead.",
        )
        return redshift_interpolation(dc, coeval_a, coeval_b, dc_a, dc_b, kind=kind)

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, ...]:
        """Get the shape of the lightcone of each quantity.

        .. deprecated:: 4.4.0
            Use the ``shape`` of the session that :meth:`start` returns.
        """
        _warn_deprecated(
            "get_shape",
            "Use the shape of the session returned by lightconer.start(context) "
            "instead.",
        )
        context = LightconeContext(
            cosmo=self.cosmo,
            box_shape=box_shape_of(simulation_options),
            cell_size=simulation_options.cell_size,
            node_redshifts=(),
        )
        return self.start(context).shape

    def validate_options(
        self,
        inputs: InputParameters,
        include_dvdr_in_tau21: bool,
        apply_rsds: bool,
        classy_output: Class | None = None,
    ) -> Self:
        """Check the lightconer against the inputs, and return the one to use.

        .. deprecated:: 4.4.0
            Use :func:`py21cmfast.lightconers.prepare_lightconer`.
        """
        from ._prepare import prepare_lightconer

        _warn_deprecated(
            "validate_options",
            "Use the function py21cmfast.lightconers.prepare_lightconer instead.",
        )
        return prepare_lightconer(
            self,
            inputs=inputs,
            include_dvdr_in_tau21=include_dvdr_in_tau21,
            apply_rsds=apply_rsds,
            classy_output=classy_output,
        )

    def find_required_lightcone_limits(
        self, classy_output: Class, inputs: InputParameters
    ) -> list[Quantity]:
        """Obtain the distance limits required for the lightcone to include RSDs.

        .. deprecated:: 4.4.0
            Use :func:`py21cmfast.rsds.find_required_lightcone_limits`.
        """
        from ..rsds import find_required_lightcone_limits

        _warn_deprecated(
            "find_required_lightcone_limits",
            "Use the function py21cmfast.rsds.find_required_lightcone_limits instead.",
        )
        return find_required_lightcone_limits(
            self, classy_output=classy_output, inputs=inputs
        )


__all__ = [
    "CoevalSnapshot",
    "LegacyLightconerAPI",
    "LightconerSpec",
    "context_from_inputs",
    "redshift_interpolation",
    "slices_between",
]
