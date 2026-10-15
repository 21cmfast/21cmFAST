"""A module for classes that create lightcone slices from Coeval objects."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict

import attrs
import deprecation
import numpy as np
from astropy import units
from astropy.cosmology import FLRW, z_at_value
from astropy.units import Mpc, pixel, pixel_scale
from classy import Class
from cosmotile import (
    make_lightcone_slice_interpolator,
    make_lightcone_slice_vector_field,
)
from scipy.spatial.transform import Rotation

from .rsds import estimate_rsd_displacements
from .wrapper.classy_interface import run_classy
from .wrapper.inputs import (
    InputParameters,
    Planck18,  # Not *quite* the same as astropy's Planck18
    SimulationOptions,
)

if TYPE_CHECKING:
    from .drivers.coeval import Coeval

_LIGHTCONERS: dict[str, type[Lightconer]] = {}

type InterpKind = Literal["mean", "mean_max"]
"""The kinds of redshift interpolation supported by :meth:`Lightconer.redshift_interpolation`."""

type LightconeSlice = tuple[str, int, np.ndarray] | tuple[None, np.ndarray, None]
"""A slice yielded by :meth:`Lightconer.make_lightcone_slices`.

Either ``(quantity, lightcone_index, slice)``, or ``(None, empty_index_array, None)``
when no lightcone slices lie between the two coeval boxes.
"""


def _lc_distances_default(self: Lightconer) -> units.Quantity:
    if self._lc_redshifts is None:
        raise ValueError("Either lc_distances or lc_redshifts must be provided")
    return self.cosmo.comoving_distance(self._lc_redshifts)


def _lc_distances_validator(
    inst: Lightconer, attribute: attrs.Attribute, value: units.Quantity
) -> None:
    if np.any(value < 0):
        raise ValueError("lc_distances must be non-negative")


def _lc_redshifts_validator(
    inst: Lightconer, attribute: attrs.Attribute, value: np.ndarray | None
) -> None:
    if value is None:
        return

    if np.any(value < 0):
        raise ValueError("lc_redshifts must be non-negative")


def _interp_kinds_default() -> dict[str, InterpKind]:
    return {"z_reion": "mean_max"}


@attrs.define(kw_only=True, slots=False)
class Lightconer(ABC):
    """A class that creates lightcone slices from Coeval objects.

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
        shape ``HII_DIM^3``. A *special* value here is ``los_velocity``, which gives
        the line-of-sight velocity. Defaults to ``("brightness_temp",)``.
    interp_kinds
        The kind of redshift interpolation (see :meth:`redshift_interpolation`) to use
        for each quantity. Quantities not in this dict use ``"mean"``.
    """

    # FLRW is wrapped by astropy's ``dataclass_decorator`` (annotated as returning
    # ``type[Cosmology]``), so ty treats it as a variable rather than a class.
    cosmo: FLRW = attrs.field(  # ty: ignore[invalid-type-form]
        default=Planck18, validator=attrs.validators.instance_of(FLRW)
    )

    _lc_redshifts: np.ndarray | None = attrs.field(
        default=None,
        eq=False,
        alias="lc_redshifts",
        validator=_lc_redshifts_validator,
    )
    lc_distances: units.Quantity = attrs.field(
        default=attrs.Factory(_lc_distances_default, takes_self=True),
        eq=attrs.cmp_using(eq=partial(np.allclose, rtol=1e-5, atol=0)),
        validator=_lc_distances_validator,
    )

    quantities: tuple[str, ...] = attrs.field(
        default=("brightness_temp",),
        converter=tuple,
        validator=attrs.validators.deep_iterable(attrs.validators.instance_of(str)),
    )
    interp_kinds: dict[str, InterpKind] = attrs.field(factory=_interp_kinds_default)

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

    def get_lc_distances_in_pixels(self, resolution: units.Quantity) -> units.Quantity:
        """Get the lightcone distances in pixels, given a resolution (a length)."""
        return self.lc_distances.to(pixel, pixel_scale(resolution / pixel))

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, ...]:
        """Get the shape of the lightcone slices."""
        raise NotImplementedError

    @classmethod
    def between_redshifts(
        cls,
        min_redshift: float,
        max_redshift: float,
        resolution: units.Quantity,
        cosmo: FLRW = Planck18,  # ty: ignore[invalid-type-form]  # see Lightconer.cosmo
        **kw: Any,
    ) -> Self:
        """Construct a Lightconer with regular comoving dist. slices between two z's.

        ``resolution`` is the comoving distance between slices (a length). All other
        keyword arguments are passed through to the constructor.
        """
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
        resolution: units.Quantity,
        cosmo: FLRW = Planck18,  # ty: ignore[invalid-type-form]  # see Lightconer.cosmo
        **kw: Any,
    ) -> Self:
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

    def make_lightcone_slices(
        self,
        c1: Coeval,
        c2: Coeval,
    ) -> Iterator[LightconeSlice]:
        """
        Make lightcone slices out of two coeval objects.

        Parameters
        ----------
        c1, c2 : Coeval
            The coeval boxes to interpolate.

        Yields
        ------
        quantity
            The field name of the quantity of this slice (or None if there are no
            lightcone slices within the redshift range spanned by ``c1`` and ``c2``).
        lcidx
            The index of the lightcone to which this slice belongs (or an empty
            index array if there are no slices).
        slice
            The field evaluated on the "lightcone" slice (or None if there are no
            slices).
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

        for idx, lcd in zip(lcidx, lc_distances, strict=True):
            for q in lc_quantities:
                if q == "los_velocity":
                    continue
                box1 = self.coeval_subselect(
                    lcd, getattr(c1, q), c1.simulation_options.cell_size
                )
                box2 = self.coeval_subselect(
                    lcd, getattr(c2, q), c2.simulation_options.cell_size
                )
                box = self.redshift_interpolation(
                    lcd, box1, box2, dc1, dc2, kind=self.interp_kinds.get(q, "mean")
                )

                yield q, int(idx), self.construct_lightcone(lcd, box)

            if "los_velocity" in lc_quantities:
                # After doing the scalar quantities at this slice, also add in the los
                # velocity, if desired. Doing it at the same slice means we can keep
                # whatever cached interpolation setup is used to do
                # construct_lightcone().
                if isinstance(self, AngularLightconer):
                    boxes1 = [
                        self.coeval_subselect(
                            lcd,
                            getattr(c1, f"velocity_{ax}"),
                            c1.simulation_options.cell_size,
                        )
                        for ax in "xyz"
                    ]
                    boxes2 = [
                        self.coeval_subselect(
                            lcd,
                            getattr(c2, f"velocity_{ax}"),
                            c2.simulation_options.cell_size,
                        )
                        for ax in "xyz"
                    ]

                    interpolated_boxes = [
                        self.redshift_interpolation(
                            lcd,
                            box1,
                            box2,
                            dc1,
                            dc2,
                            kind=self.interp_kinds.get("velocity", "mean"),
                        )
                        for (box1, box2) in zip(boxes1, boxes2, strict=True)
                    ]
                    los_velocity = self.construct_los_velocity_lightcone(
                        lcd, interpolated_boxes
                    )
                else:
                    # TODO: get the correct component according to self.line_of_sight_axis
                    box1 = self.coeval_subselect(
                        lcd, c1.velocity_z, c1.simulation_options.cell_size
                    )
                    box2 = self.coeval_subselect(
                        lcd, c2.velocity_z, c2.simulation_options.cell_size
                    )
                    interpolated_box = self.redshift_interpolation(
                        lcd,
                        box1,
                        box2,
                        dc1,
                        dc2,
                        kind=self.interp_kinds.get("velocity_z", "mean"),
                    )
                    los_velocity = self.construct_los_velocity_lightcone(
                        lcd, interpolated_box
                    )
                yield "los_velocity", int(idx), los_velocity

    def coeval_subselect(
        self, lcd: units.Quantity, coeval: np.ndarray, coeval_res: units.Quantity
    ) -> np.ndarray:
        """Sub-Select the coeval box required for interpolation at one slice.

        ``lcd`` is the comoving distance of the slice (in pixels), and ``coeval_res``
        is the cell size of the coeval box (a length).
        """
        return coeval

    def redshift_interpolation(
        self,
        dc: float | units.Quantity,
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: float | units.Quantity,
        dc_b: float | units.Quantity,
        kind: InterpKind = "mean",
    ) -> np.ndarray:
        """Perform redshift interpolation to a new box given two bracketing coevals."""
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

    @abstractmethod
    def construct_lightcone(
        self,
        lcd: units.Quantity,
        box: np.ndarray,
        /,
    ) -> np.ndarray:
        """Construct the lightcone slice at distance ``lcd`` from an interpolated box.

        Parameters
        ----------
        lcd
            The comoving distance of the slice, in pixels.
        box
            The coeval box, interpolated to the redshift of the slice.
        """

    @abstractmethod
    def construct_los_velocity_lightcone(
        self,
        lcd: units.Quantity,
        velocities: np.ndarray,
        /,
    ) -> np.ndarray:
        """Construct the LoS velocity lightcone slice at distance ``lcd``.

        Parameters
        ----------
        lcd
            The comoving distance of the slice, in pixels.
        velocities
            The velocity box(es), interpolated to the redshift of the slice.
            Subclasses may accept other forms of the velocity field (e.g. the
            :class:`AngularLightconer` accepts the three Cartesian components).
        """

    def validate_options(
        self,
        inputs: InputParameters,
        include_dvdr_in_tau21: bool,
        apply_rsds: bool,
        classy_output: Class | None = None,
    ) -> Self:
        """Validate 21cmFAST options."""
        if len(inputs.node_redshifts) == 0:
            raise ValueError(
                "You are attempting to run a lightcone with no node_redshifts."
                "Please set node_redshifts on the `inputs` parameter."
            )
        if self.cosmo != inputs.cosmo_params.cosmo:
            raise ValueError(
                "The cosmology used to construct the lightcone is not the same as the cosmology in the input parameters. "
                "Please make sure that your lightconer.cosmo is the same as inputs.cosmo_params.cosmo."
            )
        lcd = self.lc_distances
        d_node_min = self.cosmo.comoving_distance(min(inputs.node_redshifts))
        d_node_max = self.cosmo.comoving_distance(max(inputs.node_redshifts))
        if not ((d_node_min <= lcd.min()) and (lcd.max() <= d_node_max)):
            lcz = self.lc_redshifts
            raise ValueError(
                "The lightcone redshifts are not compatible with the given redshift."
                f"The range of computed redshifts is {min(inputs.node_redshifts)} to {max(inputs.node_redshifts)}, "
                f"while the lightcone redshift range is {lcz.min()} to {lcz.max()}. "
                "Extend the limits of node redshifts to avoid this error."
            )

        if (
            include_dvdr_in_tau21
            and inputs.astro_options.USE_TS_FLUCT
            and "tau_21" not in self.quantities
        ):
            lightconer = attrs.evolve(self, quantities=(*self.quantities, "tau_21"))
        else:
            lightconer = self

        if apply_rsds or include_dvdr_in_tau21:
            lightconer = attrs.evolve(
                lightconer, quantities=(*lightconer.quantities, "los_velocity")
            )

        if not apply_rsds:
            return lightconer

        if classy_output is None:
            classy_output = run_classy(
                h=inputs.cosmo_params.hlittle,
                Omega_cdm=inputs.cosmo_params.OMm - inputs.cosmo_params.OMb,
                Omega_b=inputs.cosmo_params.OMb,
                n_s=inputs.cosmo_params.POWER_INDEX,
                sigma8=inputs.cosmo_params.SIGMA_8,
                output="vTk",
            )
        lcd_limits_rsd = self.find_required_lightcone_limits(
            classy_output=classy_output, inputs=inputs
        )
        lcd_rsd = (
            np.arange(
                lcd_limits_rsd[0].value,
                lcd_limits_rsd[1].value + inputs.simulation_options.cell_size.value,
                inputs.simulation_options.cell_size.value,
            )
            * self.lc_distances.unit
        )
        # Make a new lightconer which is identical to self, but extends to further boundaries!
        return attrs.evolve(lightconer, lc_redshifts=None, lc_distances=lcd_rsd)

    def find_required_lightcone_limits(
        self, classy_output: Class, inputs: InputParameters
    ) -> list[units.Quantity]:
        """Obtain the redshift limits required for the lightcone to include RSDs.

        This is a *crude* estimation of the maximum/minimum lightcone limits that are
        required in order to simulate all the "mass" that enters the requested
        ligthcone (due to RSD shift). We use the rms of the velocity field from linear
        perturbation theory in order to determine the required lightcone limits.
        If no limit is found, it means that the limits of node_redshifts are not
        sufficient and an error is raised.

        Parameters
        ----------
        classy_output : classy.Class
            An object containing all the information from the CLASS calculation.
        inputs: InputParameters
            The input parameters corresponding to the box.

        Returns
        -------
        lcd_limits_rsd : list
            List that contains the limits of the required lightcone distances.
        """
        # The velocity rms that CLASS returns corresponds to the magnitude of the velocity vector.
        # In case of rectilinear lightcones, since we shift the cells only along one axis (e.g. z-axis),
        # we need the rms of the correponding component of the velocity vector, which is smaller by sqrt(3)
        # due to isotropy
        factor = 1.0 if isinstance(self, AngularLightconer) else 1.0 / np.sqrt(3.0)
        z_node_limits = [min(inputs.node_redshifts), max(inputs.node_redshifts)]
        lcd_limits = [self.lc_distances.min(), self.lc_distances.max()]
        signs = [-1, 1]
        lcd_limits_rsd = []
        for sign, z_node_limit, lcd_limit in zip(
            signs, z_node_limits, lcd_limits, strict=False
        ):
            distances_out = (
                np.arange(
                    lcd_limit.value,
                    self.cosmo.comoving_distance(z_node_limit).value,
                    inputs.simulation_options.cell_size.value * sign,
                )
                * lcd_limit.unit
            )
            if len(distances_out) > 0:
                displacements = estimate_rsd_displacements(
                    classy_output=classy_output,
                    cosmo=self.cosmo,
                    redshifts=z_at_value(self.cosmo.comoving_distance, distances_out),
                    factor=factor,
                )
                if sign < 0:
                    out_indices = distances_out + displacements < lcd_limit
                else:
                    out_indices = distances_out - displacements > lcd_limit
            else:
                out_indices = False
            if np.any(out_indices):
                # If there are displaced coordinates outside the lightcone box, we return the
                # corresponding distance that is closest to the original lightcone boundaries
                lcd_limits_rsd.append(distances_out[out_indices][0])
            else:
                raise ValueError(
                    f"You have set apply_rsds to True with node redshifts between {min(inputs.node_redshifts)} and {max(inputs.node_redshifts)} "
                    f"and lightcone redshifts between {self.lc_redshifts.min()} and {self.lc_redshifts.max()}. "
                    "However, RSDs are expected to contribute the lightcone from higer/lower rdshifts. "
                    "Extend the limits of node redshifts to avoid this error."
                )
        return lcd_limits_rsd

    def __init_subclass__(cls) -> None:
        """Enabe plugin-style behaviour."""
        _LIGHTCONERS[cls.__name__] = cls
        return super().__init_subclass__()


def _index_offset_default(self: RectilinearLightconer) -> int:
    # While it probably makes more sense to use zero as the default offset,
    # we use n_lightcone to maintain default backwards compatibility.
    return len(self.lc_distances)


@attrs.define(kw_only=True, slots=False)
class RectilinearLightconer(Lightconer):
    """The class rectilinear lightconer."""

    index_offset: int = attrs.field(
        default=attrs.Factory(_index_offset_default, takes_self=True)
    )
    line_of_sight_axis: int = attrs.field(
        default=-1, converter=int, validator=attrs.validators.in_([-1, 0, 1, 2])
    )

    def coeval_subselect(
        self, lcd: units.Quantity, coeval: np.ndarray, coeval_res: units.Quantity
    ) -> np.ndarray:
        """Sub-select the coeval slice corresponding to this coeval distance.

        ``lcd`` is the comoving distance of the slice (in pixels), and ``coeval_res``
        is the cell size of the coeval box (a length).
        """
        # This makes the back of the lightcone exactly line up with the back of the
        # coeval box at that redshift, modulo the index_offset.
        lcpix = self.get_lc_distances_in_pixels(coeval_res)
        lcidx = int((lcpix.max() - lcd + 1 * pixel).to_value(pixel))
        return coeval.take(-lcidx + self.index_offset, axis=2, mode="wrap")

    def construct_lightcone(
        self,
        lcd: units.Quantity,
        box: np.ndarray,
    ) -> np.ndarray:
        """Construct slices of the lightcone between two coevals."""
        return box

    def construct_los_velocity_lightcone(
        self,
        lcd: units.Quantity,
        velocity: np.ndarray,
    ) -> np.ndarray:
        """Construct slices of the lightcone between two coevals."""
        return velocity

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, int, int]:
        """Get the shape of the lightcone."""
        return (
            simulation_options.HII_DIM,
            simulation_options.HII_DIM,
            len(self.lc_distances),
        )


def _rotation_eq(x: Rotation | None, y: Rotation | None) -> bool:
    """Compare two rotations."""
    if x is None or y is None:
        return x is y

    return bool(np.allclose(x.as_matrix(), y.as_matrix()))


def _longitude_validator(
    inst: AngularLightconer, attribute: attrs.Attribute, value: np.ndarray
) -> None:
    if value.ndim != 1:
        raise ValueError("longitude must be 1-dimensional")
    if np.any(value < 0) or np.any(value > 2 * np.pi):
        raise ValueError("longitude must be in the range [0, 2pi]")
    if value.shape != inst.latitude.shape:
        raise ValueError("longitude and latitude must have the same shape")


def _origin_default() -> units.Quantity:
    return np.zeros(3) * pixel


class _InterpolatorCache(TypedDict):
    """Cache of the slice interpolator of an :class:`AngularLightconer`."""

    lcd: units.Quantity | None
    interpolator: partial[np.ndarray] | None


@attrs.define(kw_only=True, slots=False)
class AngularLightconer(Lightconer):
    """Angular lightcone slices constructed from rectlinear coevals."""

    latitude: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    longitude: np.ndarray = attrs.field(
        eq=attrs.cmp_using(eq=np.allclose), validator=_longitude_validator
    )
    interpolation_order: int = attrs.field(
        default=1, converter=int, validator=attrs.validators.in_([0, 1, 3, 5])
    )
    # The position of the observer, in pixel units, shape (3,).
    origin: units.Quantity = attrs.field(
        factory=_origin_default, eq=attrs.cmp_using(eq=np.allclose)
    )
    rotation: Rotation | None = attrs.field(
        default=None,
        eq=attrs.cmp_using(eq=_rotation_eq),
        validator=attrs.validators.optional(attrs.validators.instance_of(Rotation)),
    )

    def __attrs_post_init__(self) -> None:
        """Post-init."""
        self._cache: _InterpolatorCache = {
            "lcd": None,
            "interpolator": None,
        }

    @classmethod
    def like_rectilinear(
        cls,
        simulation_options: SimulationOptions,
        match_at_z: float,
        cosmo: FLRW = Planck18,  # ty: ignore[invalid-type-form]  # see Lightconer.cosmo
        **kw: Any,
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
        box_size_radians = (
            simulation_options.BOX_LEN / cosmo.comoving_distance(match_at_z).value
        )

        lon = np.linspace(0, box_size_radians, simulation_options.HII_DIM)
        # This makes the X-values increasing from 0.
        lat = np.linspace(0, box_size_radians, simulation_options.HII_DIM)[::-1]

        LON, LAT = np.meshgrid(lon, lat)
        LON = LON.flatten()
        LAT = LAT.flatten()

        origin_offset = -cosmo.comoving_distance(match_at_z).to(
            pixel, pixel_scale(simulation_options.cell_size / pixel)
        )
        origin = np.array([0, 0, origin_offset.value]) * origin_offset.unit
        rot = Rotation.from_euler("Y", -np.pi / 2)

        return cls.between_redshifts(
            min_redshift=match_at_z,
            resolution=simulation_options.cell_size,
            cosmo=cosmo,
            latitude=LAT,
            longitude=LON,
            origin=origin,
            rotation=rot,
            **kw,
        )

    def _get_interpolator(self, lcd: units.Quantity) -> partial[np.ndarray]:
        """Get the (cached) slice interpolator for the slice at distance ``lcd``."""
        interpolator = self._cache["interpolator"]
        if interpolator is None or self._cache["lcd"] != lcd:
            interpolator = self._refresh_cache(lcd)
        return interpolator

    def construct_lightcone(
        self,
        lcd: units.Quantity,
        box: np.ndarray,
    ) -> np.ndarray:
        """Construct the lightcone slices from bracketing coevals."""
        return self._get_interpolator(lcd)(box)

    def construct_los_velocity_lightcone(
        self,
        lcd: units.Quantity,
        velocities: np.ndarray | Sequence[np.ndarray],
    ) -> np.ndarray:
        """Construct the LoS velocity lightcone from 3D velocities.

        ``velocities`` are the three Cartesian components of the velocity field
        (either a sequence of three boxes, or an array of shape ``(3, ...)``).
        """
        return next(
            make_lightcone_slice_vector_field(
                [list(velocities)], self._get_interpolator(lcd)
            )
        )

    def _refresh_cache(self, lcd: units.Quantity) -> partial[np.ndarray]:
        # cosmotile works with plain floats in units of the coeval cell size (pixels).
        result = make_lightcone_slice_interpolator(
            latitude=self.latitude,
            longitude=self.longitude,
            distance_to_shell=lcd.to_value(pixel),
            interpolation_order=self.interpolation_order,
            origin=self.origin.to_value(pixel),
            rotation=self.rotation,
        )
        self._cache["lcd"] = lcd
        self._cache["interpolator"] = result
        return result

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, int]:
        """Get the shape of the lightcone slices."""
        return (len(self.longitude), len(self.lc_redshifts))

    def validate_options(
        self,
        inputs: InputParameters,
        include_dvdr_in_tau21: bool,
        apply_rsds: bool,
        classy_output: Class | None = None,
    ) -> Self:
        """Validate 21cmFAST options."""
        lightconer = super().validate_options(
            inputs=inputs,
            include_dvdr_in_tau21=include_dvdr_in_tau21,
            apply_rsds=apply_rsds,
            classy_output=classy_output,
        )

        if (
            include_dvdr_in_tau21 or apply_rsds
        ) and not inputs.matter_options.KEEP_3D_VELOCITIES:
            raise ValueError(
                "To account for RSDs or velocity corrections in an angular lightcone, you need to set "
                "matter_options.KEEP_3D_VELOCITIES=True"
            )

        return lightconer
