"""The base class of the lightconers of 21cmFAST v4.3, kept until v5."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Self

import attrs
import deprecation
import numpy as np
from astropy.units import Quantity, pixel, pixel_scale

from ._base import (
    _DEPRECATED_IN,
    _LENGTH,
    LightconerSpec,
    _options_from_context,
    _SnapshotAsCoeval,
    redshift_interpolation,
)
from .protocol import Chunk, LightconeContext, Snapshot

if TYPE_CHECKING:
    from classy import Class

    from ..drivers.coeval import Coeval
    from ..wrapper.inputs import InputParameters, SimulationOptions

#: The lightconers of 21cmFAST, by name.
_LIGHTCONERS = {}


@attrs.define(kw_only=True, slots=False)
class Lightconer(LightconerSpec, ABC):
    """A class that creates lightcone slices from two Coeval objects.

    .. deprecated:: 4.4.0
        This is the base class of the lightconers of 21cmFAST v4.3, and will be
        removed in v5. Subclasses of it can still be passed to
        :func:`~py21cmfast.run_lightcone`, but new lightconers should follow the
        :class:`~py21cmfast.lightconers.LightconerProtocol` instead (which does not
        require a base class). :class:`~py21cmfast.RectilinearLightconer` is an
        example.

    A subclass makes each slice of the lightcone from the two coeval boxes around
    it, by defining :meth:`construct_lightcone`, :meth:`construct_los_velocity_lightcone`
    and :meth:`get_shape` (and, optionally, any of the other methods).

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
        be attributes of the :class:~`outputs.Coeval` class that are arrays of
        shape ``HII_DIM^3``. A *special* value here is ``los_velocity``, the velocity
        along the line of sight. Defaults to ``("brightness_temp",)``.
    """

    #: The ratio of the rms of the velocity along the line of sight to that of the
    #: full velocity: one component of an isotropic field.
    los_velocity_rms_factor = 1.0 / np.sqrt(3.0)

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, int, int]:
        """Get the shape of the lightcone slices."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # The lightconer protocol, in terms of the methods of v4.3
    # ------------------------------------------------------------------
    def required_fields(self) -> tuple[str, ...]:
        """Return the names of the fields that are needed to make the quantities."""
        fields = [q for q in self.quantities if q != "los_velocity"]
        if "los_velocity" in self.quantities:
            fields.append("velocity_z")
        return tuple(dict.fromkeys(fields))

    def start(self, context: LightconeContext) -> _TwoCoevalSession:
        """Start a run, and return the session that will make the lightcone."""
        return _TwoCoevalSession(self, context)

    # ------------------------------------------------------------------
    # The methods of v4.3
    # ------------------------------------------------------------------
    def make_lightcone_slices(
        self,
        c1: Coeval,
        c2: Coeval,
    ) -> Iterator[tuple[str | None, Any, np.ndarray | None]]:
        """
        Make lightcone slices out of two coeval objects.

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
            The quantity on the slice, for the slices that exist within the redshift
            range spanned by ``c1`` and ``c2``.
        """
        if c1.simulation_options != c2.simulation_options:
            raise ValueError("c1 and c2 must have the same user parameters")
        if c1.cosmo_params != c2.cosmo_params:
            raise ValueError("c1 and c2 must have the same cosmological parameters")

        cosmo = c1.cosmo_params.cosmo
        pixeleq = pixel_scale(c1.simulation_options.cell_size / pixel)

        dc1 = cosmo.comoving_distance(c1.redshift).to(pixel, equivalencies=pixeleq)
        dc2 = cosmo.comoving_distance(c2.redshift).to(pixel, equivalencies=pixeleq)

        dcmin = min(dc1, dc2)
        dcmax = max(dc1, dc2)

        pixlcdist = self.get_lc_distances_in_pixels(c1.simulation_options.cell_size)

        # At the lower redshift, we include some tolerance. This is because the very
        # last slice (lowest redshift) may correspond *exactly* to the lowest coeval
        # box, and due to rounding error in the `z_at_value` call, they might be
        # slightly off.
        lcidx = np.nonzero((pixlcdist >= dcmin * (1 - 1e-6)) & (pixlcdist < dcmax))[0]

        # Return early if no lightcone indices are between the coeval distances.
        if len(lcidx) == 0:
            yield None, lcidx, None

        lc_distances = pixlcdist[lcidx]

        lc_quantities = self.quantities
        coeval_res = c1.simulation_options.cell_size

        for idx, lcd in zip(lcidx, lc_distances, strict=True):
            for q in lc_quantities:
                if q == "los_velocity":
                    continue

                yield (
                    q,
                    idx,
                    self.interpolate_slice(
                        lcd,
                        getattr(c1, q),
                        getattr(c2, q),
                        dc1,
                        dc2,
                        coeval_res,
                        kind=self.interp_kinds.get(q, "mean"),
                    ),
                )

                if "los_velocity" in self.quantities and q == self.quantities[0]:
                    # While doing the first quantity, also add in the los velocity, if desired.
                    # Doing it now means we can keep whatever cached interpolation setup
                    # is used to do construct_lightcone().
                    yield (
                        "los_velocity",
                        idx,
                        self.interpolate_los_velocity_slice(lcd, c1, c2, dc1, dc2),
                    )

    def interpolate_slice(
        self,
        lcd: Quantity[pixel],
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: Quantity[pixel],
        dc_b: Quantity[pixel],
        coeval_res: Quantity[_LENGTH],
        kind: str = "mean",
    ) -> np.ndarray:
        """Make one lightcone slice of a field from the two coevals that bracket it.

        The coevals are first sub-selected (:meth:`coeval_subselect`), then
        interpolated to the slice's distance (:meth:`redshift_interpolation`), then
        made into a slice (:meth:`construct_lightcone`).

        Parameters
        ----------
        lcd
            The comoving distance of the slice, in pixels.
        coeval_a, coeval_b
            The field in the two coeval boxes.
        dc_a, dc_b
            The comoving distances of the two coeval boxes, in pixels.
        coeval_res
            The cell size of the coeval boxes.
        kind
            The kind of interpolation in distance (see :meth:`redshift_interpolation`).
        """
        box_a = self.coeval_subselect(lcd, coeval_a, coeval_res)
        box_b = self.coeval_subselect(lcd, coeval_b, coeval_res)
        box = self.redshift_interpolation(lcd, box_a, box_b, dc_a, dc_b, kind=kind)
        return self.construct_lightcone(lcd, box)

    def interpolate_los_velocity_slice(
        self,
        lcd: Quantity[pixel],
        c_a: Coeval,
        c_b: Coeval,
        dc_a: Quantity[pixel],
        dc_b: Quantity[pixel],
    ) -> np.ndarray:
        """Make one slice of the line-of-sight velocity from two bracketing coevals.

        By default, this is the z-component of the velocity, interpolated like
        :meth:`interpolate_slice` does, and made into a slice by
        :meth:`construct_los_velocity_lightcone`.
        """
        coeval_res = c_a.simulation_options.cell_size
        box_a = self.coeval_subselect(lcd, c_a.velocity_z, coeval_res)
        box_b = self.coeval_subselect(lcd, c_b.velocity_z, coeval_res)
        box = self.redshift_interpolation(
            lcd,
            box_a,
            box_b,
            dc_a,
            dc_b,
            kind=self.interp_kinds.get("velocity_z", "mean"),
        )
        return self.construct_los_velocity_lightcone(lcd, box)

    def coeval_subselect(
        self, lcd: float, coeval: np.ndarray, coeval_res: Quantity[_LENGTH]
    ) -> np.ndarray:
        """Sub-Select the coeval box required for interpolation at one slice."""
        return coeval

    def redshift_interpolation(
        self,
        dc: float,
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: float,
        dc_b: float,
        kind: str = "mean",
    ) -> np.ndarray:
        """Perform redshift interpolation to a new box given two bracketing coevals."""
        return redshift_interpolation(dc, coeval_a, coeval_b, dc_a, dc_b, kind=kind)

    @abstractmethod
    def construct_lightcone(
        self,
        lc_distances: np.ndarray,
        boxes: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Abstract method for constructing the lightcone slices."""

    @abstractmethod
    def construct_los_velocity_lightcone(
        self,
        lc_distances: np.ndarray,
        velocities: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Abstract method for constructing the LoS velocity lightcone slices."""

    def validate_options(
        self,
        inputs: InputParameters,
        include_dvdr_in_tau21: bool,
        apply_rsds: bool,
        classy_output: Class | None = None,
    ) -> Self:
        """Validate 21cmFAST options, and return the lightconer to use."""
        from ._prepare import _prepare_lightconer

        return _prepare_lightconer(
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

        See :func:`py21cmfast.rsds.find_required_lightcone_limits`.
        """
        from ..rsds import find_required_lightcone_limits

        return find_required_lightcone_limits(
            self, classy_output=classy_output, inputs=inputs
        )

    def __init_subclass__(cls) -> None:
        """Enabe plugin-style behaviour."""
        _LIGHTCONERS[cls.__name__] = cls
        warnings.warn(
            deprecation.DeprecatedWarning(
                "Lightconer",
                deprecated_in=_DEPRECATED_IN,
                removed_in="5.0.0",
                details=(
                    f"{cls.__name__} is a subclass of py21cmfast.lightconers.Lightconer, "
                    "which is deprecated. It still works, but new lightconers should "
                    "follow py21cmfast.lightconers.LightconerProtocol instead."
                ),
            ),
            stacklevel=2,
        )
        return super().__init_subclass__()


class _TwoCoevalSession:
    """The making of a lightcone by a lightconer written for v4.3.

    Each slice is made by ``Lightconer.make_lightcone_slices``, from the two
    snapshots around it.
    """

    resume_depth = 1

    def __init__(self, lightconer: Lightconer, context: LightconeContext):
        self.lightconer = lightconer
        # What v4.3 lightconers read from the coeval boxes they are given.
        self._simulation_options = _options_from_context(context)
        self._cosmo_params = _CosmoParams(context.cosmo)
        self.shape = tuple(lightconer.get_shape(self._simulation_options))
        self._previous = None

    def ingest(self, snapshot: Snapshot) -> Iterable[Chunk]:
        """Make the slices between this snapshot and the one before it."""
        this = _SnapshotAsCoeval(snapshot, self._simulation_options, self._cosmo_params)
        previous, self._previous = self._previous, this
        if previous is None:
            return

        for quantity, idx, values in self.lightconer.make_lightcone_slices(
            this, previous
        ):
            if values is not None:
                yield Chunk(quantity, (..., idx), values)

    def finish(self) -> Iterable[Chunk]:
        """Return nothing: every slice was returned by ``ingest``."""
        self._previous = None
        return ()


@attrs.frozen
class _CosmoParams:
    """The part of a CosmoParams that a v4.3 lightconer may use."""

    cosmo: Any = attrs.field()
