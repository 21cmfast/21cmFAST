"""A module for classes that create lightcone slices from Coeval objects."""

from __future__ import annotations

import attr
import numpy as np
from abc import ABC, abstractmethod
from astropy.cosmology import FLRW, z_at_value
from astropy.units import MHz, Mpc, Quantity
from cosmotile import make_lightcone_slice
from functools import cached_property
from scipy.spatial.transform import Rotation

from .inputs import Planck18  # Not *quite* the same as astropy's Planck18
from .inputs import UserParams
from .outputs import Coeval

_LIGHTCONERS = {}


@attr.define(kw_only=True, slots=False)
class Lightconer(ABC):
    """A class that creates lightcone slices from Coeval objects."""

    _lc_distances: np.ndarray = attr.field(default=None, eq=attr.cmp_using(np.allclose))
    _lc_redshifts: np.ndarray = attr.field(default=None, eq=attr.cmp_using(np.allclose))
    cosmo: FLRW = attr.field(
        default=Planck18, validator=attr.validators.instance_of(FLRW)
    )
    quantities: tuple[str] = attr.field(
        default=("brightness_temp",),
        converter=tuple,
        validator=attr.validators.deep_iterable(attr.validators.instance_of(str)),
    )
    interp_kinds: dict[str, str] = attr.field()

    @_lc_redshifts.validator
    def _lcz_vld(self, att, val):
        if val is not None and self._lc_distances is not None:
            assert np.allclose(
                self._lc_distances, self.cosmo.comoving_distance(val).val, atol=1e-4
            )
        if val is None and self._lc_distances is None:
            raise ValueError("Must set either lc_distances or lc_redshifts")

    @interp_kinds.default
    def _interp_kinds_def(self):
        return {"z_re_box": "mean_max"}

    @cached_property
    def lc_distances(self) -> np.ndarray:
        """The comoving distances of the lightcone slices."""
        if self._lc_distances is not None:
            return self._lc_distances
        else:
            return self.cosmo.comoving_distance(self._lc_redshifts).value

    @cached_property
    def lc_redshifts(self):
        """The redshifts of all the lightcone slices."""
        if self._lc_redshifts is not None:
            return self._lc_redshifts
        else:
            return np.array(
                [
                    z_at_value(self.cosmo.comoving_distance, d * Mpc)
                    for d in self.lc_distances
                ]
            )

    def get_shape(self, user_params: UserParams) -> tuple[int, int, int]:
        """The shape of the lightcone slices."""
        raise NotImplementedError

    @classmethod
    def with_equal_cdist_slices(
        cls,
        min_redshift: float,
        max_redshift: float,
        resolution: float | None = None,
        cosmo=Planck18,
        user_params: UserParams | None = None,
        **kw,
    ):
        """Construct a Lightconer with equally spaced slices in comoving distance."""
        if resolution is None and user_params is None:
            raise ValueError("resolution or user_params is required")

        if resolution is None:
            resolution = user_params.BOX_LEN / user_params.HII_DIM

        d_at_redshift = cosmo.comoving_distance(min_redshift).value
        dmax = cosmo.comoving_distance(max_redshift).value

        lc_distances = np.arange(d_at_redshift, dmax + resolution, resolution)
        if np.isclose(lc_distances.max() + resolution, dmax):
            lc_distances = np.append(lc_distances, dmax)
        lcz = np.array(
            [z_at_value(cosmo.comoving_distance, d * Mpc) for d in lc_distances]
        )
        lcz[0] = min_redshift  # Get it a bit more exact.
        return cls(cosmo=cosmo, lc_distances=lc_distances, **kw)

    @classmethod
    def with_equal_redshift_slices(
        cls,
        min_redshift: float,
        max_redshift: float,
        dz: float | None = None,
        user_params: UserParams | None = None,
        cosmo=Planck18,
        **kw,
    ):
        """Construct a Lightconer with equally spaced slices in redshift."""
        if dz is None and user_params is None:
            raise ValueError("dz or user_params is required")

        if dz is None:
            dc = cosmo.comoving_distance(min_redshift)
            dc = dc + dc.unit * user_params.BOX_LEN / user_params.HII_DIM
            zdash = z_at_value(cosmo.comoving_distance, dc)
            dz = zdash - min_redshift

        zs = np.arange(min_redshift, max_redshift + dz, dz)
        return cls(cosmo=cosmo, lc_redshifts=zs, **kw)

    @classmethod
    def from_frequencies(cls, freqs: Quantity[MHz], cosmo=Planck18, **kw):
        """Construct a Lightconer with slices corresponding to given frequencies."""
        zs = (1420 * MHz / freqs - 1).to_value("")
        return cls(cosmo=cosmo, lc_redshifts=zs, **kw)

    def make_lightcone_slices(
        self, c1: Coeval, c2: Coeval
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Make lightcone slices out of two coeval objects.

        Parameters
        ----------
        d
            The comoving distance to the redshift at which the slices should be
            interpolated.
        c1, c2 : Coeval
            The coeval boxes to interpolate.
        """
        if self.lc_distances is None:
            raise ValueError(
                "Run set_slice_comoving_distances before making lightcone slices!"
            )

        dc1 = self.cosmo.comoving_distance(c1.redshift).value
        dc2 = self.cosmo.comoving_distance(c2.redshift).value

        dcmin = min(dc1, dc2)
        dcmax = max(dc1, dc2)

        # At the lower redshift, we include some tolerance. This is because the very
        # last slice (lowest redshift) may correspond *exactly* to the lowest coeval
        # box, and due to rounding error in the `z_at_value` call, they might be
        # slightly off.
        lcidx = (self.lc_distances >= dcmin * 0.9999) & (self.lc_distances < dcmax)

        # Return early if no lightcone indices are between the coeval distances.
        if not np.any(lcidx):
            return None, lcidx

        lc_distances = self.lc_distances[lcidx]
        res = c1.user_params.BOX_LEN / c1.user_params.HII_DIM
        out = {}
        for q in self.quantities:
            box1 = getattr(c1, q)
            box2 = getattr(c2, q)

            out[q] = self.construct_lightcone(
                lc_distances,
                box1,
                box2,
                dc1,
                dc2,
                box_res=res,
                interp_kind=self.interp_kinds.get(q, "mean"),
            )

        return out, lcidx

    def redshift_interpolation(
        self,
        dc: np.ndarray | float,
        coeval_a: np.ndarray,
        coeval_b: np.ndarray,
        dc_a: float,
        dc_b: float,
        kind: str = "mean",
    ) -> np.ndarray:
        """Perform redshift interpolation to a new box given two bracketing coevals."""
        if hasattr(dc, "__len__") and len(dc) != coeval_a.shape[-1]:
            raise ValueError("dc must have the same size as last dimension of coeval")

        if coeval_a.shape != coeval_b.shape:
            raise ValueError("coeval_a and coeva_b must have the same shape")

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
        box1: np.ndarray,
        box2: np.ndarray,
        dc1: float,
        dc2: float,
        box_res: float,
        interp_kind: str = "mean",
    ) -> np.ndarray:
        """Abstract method for constructing the lightcone slices."""
        pass

    def __init_subclass__(cls) -> None:
        """Enabe plugin-style behaviour."""
        _LIGHTCONERS[cls.__name__] = cls
        return super().__init_subclass__()


@attr.define(kw_only=True, slots=False)
class RectilinearLightconer(Lightconer):
    """The class rectilinear lightconer."""

    index_offset: int = attr.field()

    @index_offset.default
    def _index_offset_default(self) -> int:
        # While it probably makes more sense to use zero as the default offset,
        # we use n_lightcone to maintain default backwards compatibility.
        return len(self.lc_distances)

    def construct_lightcone(
        self,
        lc_distances: np.ndarray,
        box1: np.ndarray,
        box2: np.ndarray,
        dc1: float,
        dc2: float,
        box_res: float,
        interp_kind: str = "mean",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct slices of the lightcone between two coevals."""
        # Do linear interpolation only.
        # This makes the back of the lightcone exactly line up with the back of the
        # coeval box at that redshift, modulo the index_offset.

        lcidxs = ((self.lc_distances.max() - lc_distances) // box_res + 1).astype(int)
        box1 = box1.take(-lcidxs + self.index_offset, axis=2, mode="wrap")
        box2 = box2.take(-lcidxs + self.index_offset, axis=2, mode="wrap")

        return self.redshift_interpolation(
            lc_distances, box1, box2, dc1, dc2, kind=interp_kind
        )

    def get_shape(self, user_params: UserParams) -> tuple[int, int, int]:
        """Get the shape of the lightcone."""
        return (user_params.HII_DIM, user_params.HII_DIM, len(self.lc_redshifts))


@attr.define(kw_only=True, slots=False)
class AngularLightconer(Lightconer):
    """Angular lightcone slices constructed from rectlinear coevals."""

    latitude: np.ndarray = attr.field()
    longitude: np.ndarray = attr.field()
    interpolation_order: int = attr.field(default=1)
    origin: tuple[float, float, float] = attr.field(default=(0, 0, 0))
    rotation: Rotation = attr.field(default=None)

    def construct_lightcone(
        self,
        lc_distances: np.ndarray,
        box1: np.ndarray,
        box2: np.ndarray,
        dc1: float,
        dc2: float,
        box_res: float,
        interp_kind: str = "mean",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the lightcone slices from bracketing coevals."""
        slices = np.zeros((len(self.longitude), len(lc_distances)))

        for i, dc in enumerate(lc_distances):
            interp = self.redshift_interpolation(
                dc, box1, box2, dc1, dc2, kind=interp_kind
            )
            slices[:, i] = make_lightcone_slice(
                coeval=interp,
                coeval_res=box_res,
                latitude=self.latitude,
                longitude=self.longitude,
                distance_to_shell=dc,
                cosmo=self.cosmo,
                interpolation_order=self.interpolation_order,
                origin=self.origin,
                rotation=self.rotation,
            )

        return slices

    def get_shape(self, user_params: UserParams) -> tuple[int, int]:
        """The shape of the lightcone slices."""
        return (len(self.longitude), len(self.lc_redshifts))
