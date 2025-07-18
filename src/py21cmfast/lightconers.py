"""A module for classes that create lightcone slices from Coeval objects."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cached_property, partial

import attrs
import numpy as np
from astropy import units
from astropy.cosmology import FLRW, z_at_value
from astropy.units import MHz, Mpc, Quantity, pixel, pixel_scale
from classy import Class
from cosmotile import (
    make_lightcone_slice_interpolator,
    make_lightcone_slice_vector_field,
)
from scipy.spatial.transform import Rotation

from .drivers.coeval import Coeval
from .rsds import estimate_rsd_displacements
from .wrapper.classy_interface import run_classy
from .wrapper.inputs import (
    InputParameters,
    Planck18,  # Not *quite* the same as astropy's Planck18
    SimulationOptions,
)

_LIGHTCONERS = {}
_LENGTH = "length"


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
        shape ``HII_DIM^3``. A *special* value here is `velocity_los`, which  Defaults to ``("brightness_temp",)``.
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

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, int, int]:
        """Get the shape of the lightcone slices."""
        raise NotImplementedError

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
            "with_equal_cdist_slices is deprecated and will be removed in future versions. "
            "Call between_redshifts instead to silence this warning.",
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
        if c1.astro_options.INCLUDE_DVDR_IN_TAU21 and c1.astro_options.USE_TS_FLUCT:
            lc_quantities += ("tau_21",)

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

                yield q, idx, self.construct_lightcone(lcd, box)

                if (c1.astro_options.INCLUDE_DVDR_IN_TAU21 or c1.astro_options.APPLY_RSDS) and q == self.quantities[0]:
                    # While doing the first quantity, also add in the los velocity, if desired.
                    # Doing it now means we can keep whatever cached interpolation setup
                    # is used to do construct_lightcone().
                    if isinstance(self, AngularLightconer):
                        boxes1 = [
                            self.coeval_subselect(
                                lcd,
                                getattr(c1, f"velocity_{q}"),
                                c1.simulation_options.cell_size,
                            )
                            for q in "xyz"
                        ]
                        boxes2 = [
                            self.coeval_subselect(
                                lcd,
                                getattr(c2, f"velocity_{q}"),
                                c2.simulation_options.cell_size,
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
                            for (box1, box2) in zip(boxes1, boxes2, strict=True)
                        ]
                    else:
                        # TODO: get the correct component according to self.line_of_sight_axis
                        box1 = self.coeval_subselect(
                            lcd, c1.velocity_z, c1.simulation_options.cell_size
                        )
                        box2 = self.coeval_subselect(
                            lcd, c2.velocity_z, c2.simulation_options.cell_size
                        )
                        interpolated_boxes = self.redshift_interpolation(
                            lcd,
                            box1,
                            box2,
                            dc1,
                            dc2,
                            kind=self.interp_kinds.get("velocity_z", "mean"),
                        )
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
        classy_output: Class | None = None,
    ) -> Lightconer:
        """Validate 21cmFAST options."""
        if len(inputs.node_redshifts) == 0:
            raise ValueError(
                "You are attempting to run a lightcone with no node_redshifts."
                "Please set node_redshifts on the `inputs` parameter."
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
        if inputs.astro_options.APPLY_RSDS:
            if classy_output is None:
                classy_output = run_classy(inputs=inputs, output="vTk")
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
            lightconer_rsd = attrs.evolve(self, lc_redshifts=None, lc_distances=lcd_rsd)
            return lightconer_rsd
        else:
            return self

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
                    f"You have set APPLY_RSDS to True with node redshifts between {min(inputs.node_redshifts)} and {max(inputs.node_redshifts)} "
                    f"and lightcone redshifts between {self.lc_redshifts.min()} and {self.lc_redshifts.max()}. "
                    "However, RSDs are expected to contribute the lightcone from higer/lower rdshifts. "
                    "Extend the limits of node redshifts to avoid this error."
                )
        return lcd_limits_rsd

    def __init_subclass__(cls) -> None:
        """Enabe plugin-style behaviour."""
        _LIGHTCONERS[cls.__name__] = cls
        return super().__init_subclass__()


@attrs.define(kw_only=True, slots=False)
class RectilinearLightconer(Lightconer):
    """The class rectilinear lightconer."""

    index_offset: int = attrs.field()
    line_of_sight_axis: int = attrs.field(
        default=-1, converter=int, validator=attrs.validators.in_([-1, 0, 1, 2])
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


def _rotation_eq(x, y):
    """Compare two rotations."""
    if x is None and y is None:
        return True

    return np.allclose(x.as_matrix(), y.as_matrix())


@attrs.define(kw_only=True, slots=False)
class AngularLightconer(Lightconer):
    """Angular lightcone slices constructed from rectlinear coevals."""

    latitude: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    longitude: np.ndarray = attrs.field(eq=attrs.cmp_using(eq=np.allclose))
    interpolation_order: int = attrs.field(
        default=1, converter=int, validator=attrs.validators.in_([0, 1, 3, 5])
    )
    origin: Quantity[pixel, (3,), float] = attrs.field(
        eq=attrs.cmp_using(eq=np.allclose)
    )
    rotation: Rotation = attrs.field(
        default=None,
        eq=attrs.cmp_using(eq=_rotation_eq),
        validator=attrs.validators.optional(attrs.validators.instance_of(Rotation)),
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
        cls,
        simulation_options: SimulationOptions,
        match_at_z: float,
        cosmo: FLRW = Planck18,
        **kw,
    ):
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

    def get_shape(self, simulation_options: SimulationOptions) -> tuple[int, int]:
        """Get the shape of the lightcone slices."""
        return (len(self.longitude), len(self.lc_redshifts))

    def validate_options(
        self,
        inputs: InputParameters,
        classy_output: Class | None = None,
    ) -> Lightconer:
        """Validate 21cmFAST options."""
        lightconer = super().validate_options(
            inputs=inputs, classy_output=classy_output
        )

        if (
            (inputs.astro_options.INCLUDE_DVDR_IN_TAU21 or inputs.astro_options.APPLY_RSDS)
            and not inputs.matter_options.KEEP_3D_VELOCITIES
        ):
            raise ValueError(
                "To account for RSDs or velocity corrections in an angular lightcone, you need to set "
                "matter_options.KEEP_3D_VELOCITIES=True"
            )
        return lightconer
