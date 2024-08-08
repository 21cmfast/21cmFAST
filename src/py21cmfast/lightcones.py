"""A module for classes that create lightcone slices from Coeval objects."""

from __future__ import annotations

import attr
import numpy as np
from abc import ABC, abstractmethod
from astropy.cosmology import FLRW, z_at_value
from astropy.units import MHz, Mpc, Quantity, pixel, pixel_scale
from cosmotile import (
    make_lightcone_slice_interpolator,
    make_lightcone_slice_vector_field,
)
from functools import cached_property, partial
from scipy.spatial.transform import Rotation
from typing import Sequence

from .inputs import Planck18  # Not *quite* the same as astropy's Planck18
from .inputs import FlagOptions, UserParams
from .outputs import Coeval

_LIGHTCONERS = {}
_LENGTH = "length"


@attr.define(kw_only=True, slots=False)
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
        shape ``HII_DIM^3``. A *special* value here is `velocity_los`, which  Defaults to ``("brightness_temp",)``.
    """

    cosmo: FLRW = attr.field(
        default=Planck18, validator=attr.validators.instance_of(FLRW)
    )

    _lc_redshifts: np.ndarray = attr.field(default=None, eq=False)
    lc_distances: Quantity[_LENGTH] = attr.field(
        eq=attr.cmp_using(eq=partial(np.allclose, rtol=1e-5, atol=0))
    )

    quantities: tuple[str] = attr.field(
        default=("brightness_temp",),
        converter=tuple,
        validator=attr.validators.deep_iterable(attr.validators.instance_of(str)),
    )
    get_los_velocity: bool = attr.field(default=False, converter=bool)
    interp_kinds: dict[str, str] = attr.field()

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

        return np.array(
            [z_at_value(self.cosmo.comoving_distance, d) for d in self.lc_distances]
        )

    def get_lc_distances_in_pixels(self, resolution: Quantity[_LENGTH]):
        """Get the lightcone distances in pixels, given a resolution."""
        return self.lc_distances.to(pixel, pixel_scale(resolution / pixel))

    @interp_kinds.default
    def _interp_kinds_def(self):
        return {"z_re_box": "mean_max"}

    def get_shape(self, user_params: UserParams) -> tuple[int, int, int]:
        """The shape of the lightcone slices."""
        raise NotImplementedError

    @classmethod
    def with_equal_cdist_slices(
        cls,
        min_redshift: float,
        max_redshift: float,
        resolution: Quantity[_LENGTH],
        cosmo=Planck18,
        **kw,
    ):
        """Construct a Lightconer with equally spaced slices in comoving distance."""
        d_at_redshift = cosmo.comoving_distance(min_redshift).to_value(Mpc)
        dmax = cosmo.comoving_distance(max_redshift).to_value(Mpc)
        res = resolution.to_value(Mpc)

        lc_distances = np.arange(d_at_redshift, dmax + res, res)
        # if np.isclose(lc_distances.max() + res, dmax):
        #     lc_distances = np.append(lc_distances, dmax)

        return cls(lc_distances=lc_distances * Mpc, cosmo=cosmo, **kw)

    @classmethod
    def with_equal_redshift_slices(
        cls,
        min_redshift: float,
        max_redshift: float,
        dz: float | None = None,
        resolution: Quantity[_LENGTH] | None = None,
        cosmo=Planck18,
        **kw,
    ):
        """Construct a Lightconer with equally spaced slices in redshift."""
        if dz is None and resolution is None:
            raise ValueError("Either dz or resolution must be provided")

        if dz is None:
            dc = cosmo.comoving_distance(min_redshift) + resolution
            zdash = z_at_value(cosmo.comoving_distance, dc)
            dz = zdash - min_redshift

        zs = np.arange(min_redshift, max_redshift + dz, dz)
        return cls(lc_redshifts=zs, cosmo=cosmo, **kw)

    @classmethod
    def from_frequencies(cls, freqs: Quantity[MHz], cosmo=Planck18, **kw):
        """Construct a Lightconer with slices corresponding to given frequencies."""
        zs = (1420.4 * MHz / freqs - 1).to_value("")
        return cls(lc_redshifts=zs, cosmo=cosmo, **kw)

    def make_lightcone_slices(
        self, c1: Coeval, c2: Coeval
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Make lightcone slices out of two coeval objects.

        Parameters
        ----------
        c1, c2 : Coeval
            The coeval boxes to interpolate.

        Returns
        -------
        quantity
            The field names of the quantities required by the lightcone.
        lcidx
            The indices of the lightcone to which these slices belong.
        scalar_field_slices
            The scalar fields evaluated on the "lightcone" slices that exist within
            the redshift range spanned by ``c1`` and ``c2``.
        """
        if c1.user_params != c2.user_params:
            raise ValueError("c1 and c2 must have the same user parameters")
        if c1.cosmo_params != c2.cosmo_params:
            raise ValueError("c1 and c2 must have the same cosmological parameters")

        cosmo = c1.cosmo_params.cosmo
        pixeleq = pixel_scale(c1.user_params.cell_size / pixel)

        dc1 = cosmo.comoving_distance(c1.redshift).to(pixel, equivalencies=pixeleq)
        dc2 = cosmo.comoving_distance(c2.redshift).to(pixel, equivalencies=pixeleq)

        dcmin = min(dc1, dc2)
        dcmax = max(dc1, dc2)

        pixlcdist = self.get_lc_distances_in_pixels(c1.user_params.cell_size)

        # At the lower redshift, we include some tolerance. This is because the very
        # last slice (lowest redshift) may correspond *exactly* to the lowest coeval
        # box, and due to rounding error in the `z_at_value` call, they might be
        # slightly off.
        lcidx = np.nonzero((pixlcdist >= dcmin * 0.9999) & (pixlcdist < dcmax))[0]

        # Return early if no lightcone indices are between the coeval distances.
        if len(lcidx) == 0:
            yield None, lcidx, None

        lc_distances = pixlcdist[lcidx]

        for idx, lcd in zip(lcidx, lc_distances):
            for q in self.quantities:
                box1 = self.coeval_subselect(
                    lcd, getattr(c1, q), c1.user_params.cell_size
                )
                box2 = self.coeval_subselect(
                    lcd, getattr(c2, q), c2.user_params.cell_size
                )
                box = self.redshift_interpolation(
                    lcd, box1, box2, dc1, dc2, kind=self.interp_kinds.get(q, "mean")
                )

                yield q, idx, self.construct_lightcone(lcd, box)

                if self.get_los_velocity and q == self.quantities[0]:
                    # While doing the first quantity, also add in the los velocity, if desired.
                    # Doing it now means we can keep whatever cached interpolation setup
                    # is used to do construct_lightcone().

                    boxes1 = [
                        self.coeval_subselect(
                            lcd, getattr(c1, f"velocity_{q}"), c1.user_params.cell_size
                        )
                        for q in "xyz"
                    ]
                    boxes2 = [
                        self.coeval_subselect(
                            lcd, getattr(c2, f"velocity_{q}"), c2.user_params.cell_size
                        )
                        for q in "xyz"
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
                        for (box1, box2) in zip(boxes1, boxes2)
                    ]
                    yield (
                        "los_velocity",
                        idx,
                        self.construct_los_velocity_lightcone(
                            lcd,
                            interpolated_boxes,
                        ),
                    )

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
        lc_distances: np.ndarray,
        boxes: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Abstract method for constructing the lightcone slices."""
        pass

    @abstractmethod
    def construct_los_velocity_lightcone(
        self,
        lc_distances: np.ndarray,
        velocities: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """Abstract method for constructing the LoS velocity lightcone slices."""
        pass

    def validate_options(self, user_params: UserParams, flag_options: FlagOptions):
        """Validate 21cmFAST options."""
        pass

    def __init_subclass__(cls) -> None:
        """Enabe plugin-style behaviour."""
        _LIGHTCONERS[cls.__name__] = cls
        return super().__init_subclass__()


@attr.define(kw_only=True, slots=False)
class RectilinearLightconer(Lightconer):
    """The class rectilinear lightconer."""

    index_offset: int = attr.field()
    line_of_sight_axis: int = attr.field(
        default=-1, converter=int, validator=attr.validators.in_([-1, 0, 1, 2])
    )

    @index_offset.default
    def _index_offset_default(self) -> int:
        # While it probably makes more sense to use zero as the default offset,
        # we use n_lightcone to maintain default backwards compatibility.
        return len(self.lc_distances)

    def coeval_subselect(
        self, lcd: Quantity[pixel], coeval: np.ndarray, coeval_res: Quantity[_LENGTH]
    ):
        """Sub-select the coeval slice corresponding to this coeval distance."""
        # This makes the back of the lightcone exactly line up with the back of the
        # coeval box at that redshift, modulo the index_offset.
        lcpix = self.get_lc_distances_in_pixels(coeval_res)
        lcidx = int((lcpix.max() - lcd + 1 * pixel).to_value(pixel))
        return coeval.take(-lcidx + self.index_offset, axis=2, mode="wrap")

    def construct_lightcone(
        self,
        lcd: np.ndarray,
        box: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct slices of the lightcone between two coevals."""
        return box

    def construct_los_velocity_lightcone(
        self,
        lcd: np.ndarray,
        velocities: np.ndarray,
    ) -> np.ndarray:
        """Construct slices of the lightcone between two coevals."""
        return velocities[self.line_of_sight_axis]

    def get_shape(self, user_params: UserParams) -> tuple[int, int, int]:
        """Get the shape of the lightcone."""
        return (user_params.HII_DIM, user_params.HII_DIM, len(self.lc_distances))


def _rotation_eq(x, y):
    """Compare two rotations."""
    if x is None and y is None:
        return True

    return np.allclose(x.as_matrix(), y.as_matrix())


@attr.define(kw_only=True, slots=False)
class AngularLightconer(Lightconer):
    """Angular lightcone slices constructed from rectlinear coevals."""

    latitude: np.ndarray = attr.field(eq=attr.cmp_using(eq=np.allclose))
    longitude: np.ndarray = attr.field(eq=attr.cmp_using(eq=np.allclose))
    interpolation_order: int = attr.field(
        default=1, converter=int, validator=attr.validators.in_([0, 1, 3, 5])
    )
    origin: Quantity[pixel, (3,), float] = attr.field(eq=attr.cmp_using(eq=np.allclose))
    rotation: Rotation = attr.field(
        default=None,
        eq=attr.cmp_using(eq=_rotation_eq),
        validator=attr.validators.optional(attr.validators.instance_of(Rotation)),
    )

    def __attrs_post_init__(self) -> None:
        """Post-init."""
        self._cache = {
            "lcd": None,
            "interpolator": None,
        }

    @longitude.validator
    def _longitude_validator(self, attribute, value):
        if value.ndim != 1:
            raise ValueError("longitude must be 1-dimensional")
        if np.any(value < 0) or np.any(value > 2 * np.pi):
            raise ValueError("longitude must be in the range [0, 2pi]")
        if value.shape != self.latitude.shape:
            raise ValueError("longitude and latitude must have the same shape")

    @origin.default
    def _origin_default(self):
        return np.zeros(3) * pixel

    @classmethod
    def like_rectilinear(
        cls, user_params: UserParams, match_at_z: float, cosmo: FLRW = Planck18, **kw
    ):
        """Create an angular lightconer with the same pixel size as a rectilinear one.

        This is useful for comparing the two lightconer types.

        Parameters
        ----------
        user_params
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
            user_params.BOX_LEN / cosmo.comoving_distance(match_at_z).value
        )

        lon = np.linspace(0, box_size_radians, user_params.HII_DIM)
        # This makes the X-values increasing from 0.
        lat = np.linspace(0, box_size_radians, user_params.HII_DIM)[::-1]

        LON, LAT = np.meshgrid(lon, lat)
        LON = LON.flatten()
        LAT = LAT.flatten()

        origin_offset = -cosmo.comoving_distance(match_at_z).to(
            pixel, pixel_scale(user_params.cell_size / pixel)
        )
        origin = np.array([0, 0, origin_offset.value]) * origin_offset.unit
        rot = Rotation.from_euler("Y", -np.pi / 2)

        return cls.with_equal_cdist_slices(
            min_redshift=match_at_z,
            resolution=user_params.cell_size,
            latitude=LAT,
            longitude=LON,
            origin=origin,
            rotation=rot,
            **kw,
        )

    def construct_lightcone(
        self,
        lcd: Quantity[pixel],
        box: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the lightcone slices from bracketing coevals."""
        if self._cache["lcd"] == lcd:
            interpolator = self._cache["interpolator"]
        else:
            interpolator = self._refresh_cache(lcd)
        return interpolator(box)

    def construct_los_velocity_lightcone(
        self,
        lcd: Quantity[pixel],
        velocities: Sequence[np.ndarray],
    ):
        """Construct the LoS velocity lightcone from 3D velocities."""
        if self._cache["lcd"] == lcd:
            interpolator = self._cache["interpolator"]
        else:
            interpolator = self._refresh_cache(lcd)
        return next(make_lightcone_slice_vector_field([velocities], interpolator))

    def _refresh_cache(self, lcd):
        result = make_lightcone_slice_interpolator(
            latitude=self.latitude,
            longitude=self.longitude,
            distance_to_shell=lcd,
            interpolation_order=self.interpolation_order,
            origin=self.origin,
            rotation=self.rotation,
        )
        self._cache["lcd"] = lcd
        self._cache["interpolator"] = result
        return result

    def get_shape(self, user_params: UserParams) -> tuple[int, int]:
        """The shape of the lightcone slices."""
        return (len(self.longitude), len(self.lc_redshifts))

    def validate_options(self, user_params: UserParams, flag_options: FlagOptions):
        """Validate 21cmFAST options.

        Raises
        ------
        ValueError
            If APPLY_RSDs is True.
        """
        if flag_options.APPLY_RSDS:
            raise ValueError(
                "APPLY_RSDs must be False for angular lightcones, as the RSDs are "
                "applied in the lightcone construction."
            )
        if self.get_los_velocity and not user_params.KEEP_3D_VELOCITIES:
            raise ValueError(
                "To get the LoS velocity, you need to set "
                "user_params.KEEP_3D_VELOCITIES=True"
            )
