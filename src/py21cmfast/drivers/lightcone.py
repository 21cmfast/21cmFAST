"""Module containing a driver function for creating lightcones."""

import contextlib
import logging
import warnings
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import attrs
import h5py
import numpy as np
from astropy import units
from astropy.cosmology import z_at_value
from cosmotile import apply_rsds

from .. import __version__
from ..c_21cmfast import lib
from ..io import h5
from ..io.caching import CacheConfig, OutputCache, RunCache
from ..lightcones import Lightconer, RectilinearLightconer
from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import InitialConditions, PerturbedField, PerturbHaloField
from ..wrapper.photoncons import _get_photon_nonconservation_data, setup_photon_cons
from . import exhaust
from . import single_field as sf
from ._param_config import high_level_func
from .coeval import (
    _obtain_starting_point_for_scrolling,
    _redshift_loop_generator,
    _setup_ics_and_pfs_for_scrolling,
    evolve_perturb_halos,
)

logger = logging.getLogger(__name__)


_cache = CacheConfig()
_ocache = OutputCache(".")


@attrs.define()
class LightCone:
    """A full Lightcone with all associated evolved data.

    Attributes
    ----------
    lightcone_distances: units.Quantity
        The comoving distance to each cell in the lightcones.
    inputs: InputParameters
        The input parameters corresponding to the lightcones.
    lightcones: dict[str, np.ndarray]
        Lightcone arrays, each of shape `(N, N, Nz)`.
    global_quantities: dict[str, np.ndarray] | None
        Arrays of length `node_redshifts` containing the mean field across redshift.
    photon_nonconservation_data: dict
        Data defining the conservation hack for photons.
    _last_completed_node: int
        Since the lightcone is filled up incrementally, this keeps track of the index
        of the last completed node redshift that has been added to the lightcone.
    _last_completed_lcidx: int
        In conjunction with _last_completed_node, this keeps track of the index that
        has been filled up *in the lightcone* (recalling that the lightcone has
        multiple redshifts in between each node redshift). While in principle this
        can be computed from _last_completed_node, it is much more efficient to keep
        track of it manually.
    """

    lightcone_distances: units.Quantity = attrs.field()
    inputs: InputParameters = attrs.field(
        validator=attrs.validators.instance_of(InputParameters)
    )
    lightcones: dict[str, np.ndarray] = attrs.field(
        validator=attrs.validators.instance_of(dict)
    )
    global_quantities: dict[str, np.ndarray] | None = attrs.field(default=None)
    photon_nonconservation_data: dict = attrs.field(factory=dict)
    _last_completed_node: int = attrs.field(default=-1)
    _last_completed_lcidx: int = attrs.field(default=-1)

    @property
    def cell_size(self) -> float:
        """Cell size [Mpc] of the lightcone voxels."""
        return self.user_params.BOX_LEN / self.user_params.HII_DIM

    @property
    def lightcone_dimensions(self) -> tuple[float, float, float]:
        """Lightcone size over each dimension -- tuple of (x,y,z) in Mpc."""
        return (
            self.user_params.BOX_LEN,
            self.user_params.BOX_LEN,
            self.n_slices * self.cell_size,
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the lightcone as a 3-tuple."""
        return self.lightcones[next(iter(self.lightcones.keys()))].shape

    @property
    def n_slices(self) -> int:
        """Number of redshift slices in the lightcone."""
        return self.shape[-1]

    @property
    def lightcone_coords(self) -> tuple[float, float, float]:
        """Co-ordinates [Mpc] of each slice along the redshift axis."""
        return self.lightcone_distances - self.lightcone_distances[0]

    @property
    def user_params(self):
        """User params shared by all datasets."""
        return self.inputs.user_params

    @property
    def cosmo_params(self):
        """Cosmo params shared by all datasets."""
        return self.inputs.cosmo_params

    @property
    def flag_options(self):
        """Flag Options shared by all datasets."""
        return self.inputs.flag_options

    @property
    def astro_params(self):
        """Astro params shared by all datasets."""
        return self.inputs.astro_params

    @property
    def random_seed(self):
        """Random seed shared by all datasets."""
        return self.inputs.random_seed

    @property
    def lightcone_redshifts(self) -> np.ndarray:
        """Redshift of each cell along the redshift axis."""
        return np.array(
            [
                z_at_value(self.cosmo_params.cosmo.comoving_distance, d)
                for d in self.lightcone_distances
            ]
        )

    def save(self, path: str | Path, clobber=False):
        """Save the lightcone object to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_mode = "w" if clobber else "a"
        with h5py.File(path, file_mode) as fl:
            fl.attrs["lightcone"] = True  # marker identifying this as a lightcone box

            fl.attrs["last_completed_node"] = self._last_completed_node
            fl.attrs["last_completed_lcidx"] = self._last_completed_lcidx

            fl.attrs["__version__"] = __version__

            grp = fl.create_group("photon_nonconservation_data")
            for k, v in self.photon_nonconservation_data.items():
                grp[k] = v

            # Save the boxes to the file
            boxes = fl.create_group("lightcones")
            for k, val in self.lightcones.items():
                boxes[k] = val

            global_q = fl.create_group("global_quantities")
            for k, v in self.global_quantities.items():
                global_q[k] = v

            fl["lightcone_distances"] = self.lightcone_distances.to_value("Mpc")

        h5._write_inputs_to_group(self.inputs, path)

    def make_checkpoint(self, path: str | Path, lcidx: int, node_index: int):
        """Write updated lightcone data to file."""
        with h5py.File(path, "a") as fl:
            last_completed_lcidx = fl.attrs["last_completed_lcidx"]

            for k, v in self.lightcones.items():
                fl["lightcones"][k][
                    ..., -lcidx : v.shape[-1] - last_completed_lcidx
                ] = v[..., -lcidx : v.shape[-1] - last_completed_lcidx]

            global_q = fl["global_quantities"]
            for k, v in self.global_quantities.items():
                global_q[k][-lcidx : v.shape[-1] - last_completed_lcidx] = v[
                    -lcidx : v.shape[-1] - last_completed_lcidx
                ]

            fl.attrs["last_completed_lcidx"] = lcidx
            fl.attrs["last_completed_node"] = node_index

            self._last_completed_lcidx = lcidx
            self._last_completed_node = node_index

    @classmethod
    def from_file(cls, path: str | Path, safe: bool = True) -> Self:
        """Create a new instance from a saved lightcone on disk."""
        kwargs = {}
        with h5py.File(path, "r") as fl:
            if not fl.attrs.get("lightcone", False):
                raise ValueError(f"The file {path} is not a lightcone file!")

            kwargs["inputs"] = h5.read_inputs(fl, safe=safe)
            kwargs["last_completed_node"] = fl.attrs["last_completed_node"]
            kwargs["last_completed_lcidx"] = fl.attrs["last_completed_lcidx"]

            grp = fl["photon_nonconservation_data"]
            kwargs["photon_nonconservation_data"] = {k: v[...] for k, v in grp.items()}

            boxes = fl["lightcones"]
            kwargs["lightcones"] = {k: boxes[k][...] for k in boxes}

            glb = fl["global_quantities"]
            kwargs["global_quantities"] = {k: glb[k][...] for k in glb}
            kwargs["lightcone_distances"] = fl["lightcone_distances"][...] * units.Mpc

        return cls(**kwargs)

    def __eq__(self, other):
        """Determine if this is equal to another object."""
        return (
            isinstance(other, self.__class__)
            and np.all(
                np.isclose(
                    other.lightcone_redshifts, self.lightcone_redshifts, atol=1e-3
                )
            )
            and self.inputs == other.inputs
            and self.global_quantities.keys() == other.global_quantities.keys()
            and self.lightcones.keys() == other.lightcones.keys()
        )


class AngularLightcone(LightCone):
    """An angular lightcone."""

    @property
    def cell_size(self):
        """Cell size [Mpc] of the lightcone voxels."""
        raise AttributeError("This is not an attribute of an AngularLightcone")

    @property
    def lightcone_dimensions(self):
        """Lightcone size over each dimension -- tuple of (x,y,z) in Mpc."""
        raise AttributeError("This is not an attribute of an AngularLightcone")

    def compute_rsds(self, n_subcells: int = 4, fname: str | Path | None = None):
        """Compute redshift-space distortions from the los_velocity lightcone.

        Parameters
        ----------
        n_subcells
            The number of sub-cells to interpolate onto, to make the RSDs more accurate.
        fname
            An output path to write the new RSD-corrected brightness temperature to.
        """
        if "los_velocity" not in self.lightcones:
            raise ValueError(
                "Lightcone does not contain los velocity field, cannot compute_rsds"
            )
        if "brightness_temp_with_rsds" in self.lightcones:
            warnings.warn(
                "Lightcone already contains brightness_temp_with_rsds, returning",
                stacklevel=2,
            )
            return self.lightcones["brightness_temp_with_rsds"]

        H0 = self.cosmo_params.cosmo.H(self.lightcone_redshifts)
        los_displacement = self.lightcones["los_velocity"] * units.Mpc / units.s / H0
        equiv = units.pixel_scale(self.user_params.cell_size / units.pixel)
        los_displacement = -los_displacement.to(units.pixel, equivalencies=equiv)

        lcd = self.lightcone_distances.to(units.pixel, equiv)
        dvdx_on_h = np.gradient(los_displacement, lcd, axis=1)

        if not (self.flag_options.USE_TS_FLUCT and self.flag_options.SUBCELL_RSD):
            # Now, clip dvdx...
            dvdx_on_h = np.clip(
                dvdx_on_h,
                -self.astro_params.MAX_DVDR,
                self.astro_params.MAX_DVDR,
                out=dvdx_on_h,
            )

            tb_with_rsds = self.lightcones["brightness_temp"] / (1 + dvdx_on_h)
        else:
            gradient_component = 1 + dvdx_on_h  # not clipped!
            Tcmb = 2.728
            Trad = Tcmb * (1 + self.lightcone_redshifts)
            tb_with_rsds = np.where(
                gradient_component < 1e-7,
                1000.0 * (self.Ts_box - Trad) / (1.0 + self.lightcone_redshifts),
                (1.0 - np.exp(self.lightcones["brightness_temp"] / gradient_component))
                * 1000.0
                * (self.Ts_box - Trad)
                / (1.0 + self.lightcone_redshifts),
            )

        # Compute the local RSDs
        if n_subcells > 0:
            tb_with_rsds = apply_rsds(
                field=tb_with_rsds.T,
                los_displacement=los_displacement.T,
                distance=self.lightcone_distances.to(units.pixel, equiv),
                n_subcells=n_subcells,
            ).T

        self.lightcones["brightness_temp_with_rsds"] = tb_with_rsds

        if fname:
            if Path(fname).exists():
                with h5py.File(fname, "a") as fl:
                    fl["lightcones"]["brightness_temp_with_rsds"] = tb_with_rsds
            else:
                self.save(fname)

        return tb_with_rsds


def setup_lightcone_instance(
    lightconer: Lightconer,
    scrollz: Sequence[float],
    inputs: InputParameters,
    global_quantities: Sequence[str],
    photon_nonconservation_data: dict,
    lightcone_filename: Path | None = None,
) -> LightCone:
    """Return a LightCone instance given a lightconer as input."""
    if lightcone_filename and Path(lightcone_filename).exists():
        lightcone = LightCone.from_file(lightcone_filename)
        idx = lightcone._last_completed_node
        logger.info("Read in LC file")
        if idx < len(scrollz) - 1:
            logger.info(
                f"starting at z={scrollz[idx + 1]}, step ({idx + 2}/{len(scrollz)}"
            )
    else:
        lcn_cls = (
            LightCone
            if isinstance(lightconer, RectilinearLightconer)
            else AngularLightcone
        )
        lc = {
            quantity: np.zeros(
                lightconer.get_shape(inputs.user_params),
                dtype=np.float32,
            )
            for quantity in lightconer.quantities
        }

        # Special case: AngularLightconer can also save los_velocity
        if getattr(lightconer, "get_los_velocity", False):
            lc["los_velocity"] = np.zeros(
                lightconer.get_shape(inputs.user_params), dtype=np.float32
            )

        lightcone = lcn_cls(
            lightcone_distances=lightconer.lc_distances,
            inputs=inputs,
            lightcones=lc,
            global_quantities={
                quantity: np.zeros(len(scrollz)) for quantity in global_quantities
            },
            photon_nonconservation_data=photon_nonconservation_data,
        )
    return lightcone


def _run_lightcone_from_perturbed_fields(
    *,
    initial_conditions: InitialConditions,
    perturbed_fields: Sequence[PerturbedField],
    lightconer: Lightconer,
    inputs: InputParameters,
    photon_nonconservation_data: dict,
    pt_halos: list[PerturbHaloField],
    regenerate: bool | None = None,
    global_quantities: tuple[str] = ("brightness_temp", "xH_box"),
    cache: OutputCache = _ocache,
    cleanup: bool = True,
    write: CacheConfig = _cache,
    always_purge: bool = False,
    lightcone_filename: str | Path | None = None,
):
    lightconer.validate_options(inputs.user_params, inputs.flag_options)

    # Get the redshift through which we scroll and evaluate the ionization field.
    scrollz = np.array([pf.redshift for pf in perturbed_fields])

    lcz = lightconer.lc_redshifts
    if not np.all(scrollz.min() * 0.99 < lcz) and np.all(lcz < scrollz.max() * 1.01):
        # We have a 1% tolerance on the redshifts, because the lightcone redshifts are
        # computed via inverse fitting the comoving_distance.
        raise ValueError(
            "The lightcone redshifts are not compatible with the given redshift."
            f"The range of computed redshifts is {scrollz.min()} to {scrollz.max()}, "
            f"while the lightcone redshift range is {lcz.min()} to {lcz.max()}."
        )

    iokw = {"regenerate": regenerate, "cache": cache}

    # Create the LightCone instance, loading from file if needed
    lightcone = setup_lightcone_instance(
        lightconer=lightconer,
        inputs=inputs,
        scrollz=scrollz,
        global_quantities=global_quantities,
        lightcone_filename=lightcone_filename,
        photon_nonconservation_data=photon_nonconservation_data,
    )
    if lightcone._last_completed_node == len(scrollz) - 1:
        logger.info("Lightcone already full. Returning.")
        yield None, None, None, lightcone

    idx, prev_coeval = _obtain_starting_point_for_scrolling(
        inputs=inputs,
        cache=cache,
        initial_conditions=initial_conditions,
        photon_nonconservation_data=photon_nonconservation_data,
        minimum_node=lightcone._last_completed_node,
    )

    if idx < lightcone._last_completed_node:
        warnings.warn(
            f"The cache at {cache} only contains complete coeval boxes for {idx + 1} redshift nodes, "
            f"instead of {lightcone._last_completed_node + 1}, which is the current checkpointing "
            f"redshift of the lightcone. Repeating the higher-z calculations...",
            stacklevel=2,
        )

    lightcone._last_completed_node = idx
    lightcone._last_completed_lcidx = (
        np.sum(lightcone.lightcone_redshifts >= scrollz[lightcone._last_completed_node])
        - 1
    )

    if lightcone_filename and not Path(lightcone_filename).exists():
        lightcone.save(lightcone_filename)

    for iz, coeval in enumerate(
        _redshift_loop_generator(
            inputs=inputs,
            initial_conditions=initial_conditions,
            all_redshifts=scrollz,
            perturbed_field=perturbed_fields,
            pt_halos=pt_halos,
            write=write,
            cleanup=cleanup,
            always_purge=always_purge,
            photon_nonconservation_data=photon_nonconservation_data,
            start_idx=lightcone._last_completed_node + 1,
            init_coeval=prev_coeval,
            iokw=iokw,
        )
    ):
        # Save mean/global quantities
        for quantity in global_quantities:
            if "log10_mturn_acg" == quantity:
                lightcone.global_quantities[quantity][
                    iz
                ] = coeval.ionized_box.log10_Mturnover_ave
            elif "log10_mturn_mcg" == quantity:
                lightcone.global_quantities[quantity][
                    iz
                ] = coeval.ionized_box.log10_Mturnover_MINI_ave
            else:
                lightcone.global_quantities[quantity][iz] = np.mean(
                    getattr(coeval, quantity)
                )

        # Update photon conservation data in-place
        lightcone.photon_nonconservation_data |= coeval.photon_nonconservation_data

        # Get lightcone slices
        lc_index = None
        if prev_coeval is not None:
            for quantity, idx, this_lc in lightconer.make_lightcone_slices(
                coeval, prev_coeval
            ):
                if this_lc is not None:
                    lightcone.lightcones[quantity][..., idx] = this_lc
                    lc_index = idx

            # only checkpoint if we have slices
            if lightcone_filename and lc_index is not None:
                lightcone.make_checkpoint(lightcone_filename, lcidx=idx, node_index=iz)

        prev_coeval = coeval

        # last redshift things
        if iz == len(scrollz) - 1:
            if lib.photon_cons_allocated:
                lib.FreePhotonConsMemory()

            if isinstance(lightcone, AngularLightcone) and lightconer.get_los_velocity:
                lightcone.compute_rsds(
                    fname=lightcone_filename, n_subcells=inputs.astro_params.N_RSD_STEPS
                )

        yield iz, coeval.redshift, coeval, lightcone


@high_level_func
def generate_lightcone(
    *,
    lightconer: Lightconer,
    inputs: InputParameters,
    global_quantities=("brightness_temp", "xH_box"),
    initial_conditions: InitialConditions | None = None,
    cleanup: bool = True,
    write: CacheConfig = _cache,
    cache: OutputCache | None = _ocache,
    regenerate: bool = True,
    always_purge: bool = False,
    lightcone_filename: str | Path | None = None,
):
    r"""
    Create a generator function for a lightcone run.

    This is generally the easiest and most efficient way to generate a lightcone, though it can
    be done manually by using the lower-level functions which are called by this function.

    Parameters
    ----------
    lightconer : :class:`~Lightconer`
        This object specifies the dimensions, redshifts, and quantities required by the lightcone run
    inputs: :class:`~InputParameters`
        This object specifies the input parameters for the run, including the random seed
    global_quantities : tuple of str, optional
        The quantities to save as globally-averaged redshift-dependent functions.
        These may be any of the quantities that can be used in ``Lightconer.quantities``.
        The mean is taken over the full 3D cube at each redshift, rather than a 2D
        slice.
    initial_conditions : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be
        re-calculated.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculate has different shape, errors will occur
        if memory is not cleaned. Note that internally, this is set to False until the
        last iteration.
    lightcone_filename
        The filename to which to save the lightcone. The lightcone is returned in
        memory, and can be saved manually later, but including this filename will
        save the lightcone on each iteration, which can be helpful for checkpointing.

    Returns
    -------
    lightcone : :class:`~py21cmfast.LightCone`
        The lightcone object.
    coeval_callback_output : list
        Only if coeval_callback in not None.

    Other Parameters
    ----------------
    regenerate, write, direc, hooks
        See docs of :func:`initial_conditions` for more information.
    """
    if isinstance(write, bool):
        write = CacheConfig() if write else CacheConfig.off()

    if len(inputs.node_redshifts) == 0:
        raise ValueError(
            "You are attempting to run a lightcone with no node_redshifts."
            "Please set node_redshifts on the `inputs` parameter."
        )

    # while we still use the full list for caching etc, we don't need to run below the lightconer instance
    #   So stop one after the lightconer
    scrollz = np.copy(inputs.node_redshifts)
    below_lc_z = inputs.node_redshifts <= min(lightconer.lc_redshifts)
    if np.any(below_lc_z):
        final_node = np.argmax(below_lc_z)
        scrollz = scrollz[: final_node + 1]  # inclusive

    lcz = lightconer.lc_redshifts

    if not np.all(min(scrollz) * 0.99 < lcz) and np.all(lcz < max(scrollz) * 1.01):
        # We have a 1% tolerance on the redshifts, because the lightcone redshifts are
        # computed via inverse fitting the comoving_distance.
        raise ValueError(
            "The lightcone redshifts are not compatible with the given redshift."
            f"The range of computed redshifts is {min(scrollz)} to {max(scrollz)}, "
            f"while the lightcone redshift range is {lcz.min()} to {lcz.max()}."
        )

    iokw = {"cache": cache, "regenerate": regenerate}

    (
        initial_conditions,
        perturbed_fields,
        pt_halos,
        photon_nonconservation_data,
    ) = _setup_ics_and_pfs_for_scrolling(
        all_redshifts=scrollz,
        initial_conditions=initial_conditions,
        inputs=inputs,
        write=write,
        always_purge=always_purge,
        **iokw,
    )

    yield from _run_lightcone_from_perturbed_fields(
        initial_conditions=initial_conditions,
        perturbed_fields=perturbed_fields,
        lightconer=lightconer,
        inputs=inputs,
        regenerate=regenerate,
        pt_halos=pt_halos,
        photon_nonconservation_data=photon_nonconservation_data,
        global_quantities=global_quantities,
        cache=cache,
        write=write,
        cleanup=cleanup,
        always_purge=always_purge,
        lightcone_filename=lightcone_filename,
    )


def run_lightcone(**kwargs):  # noqa: D103
    return exhaust(generate_lightcone(**kwargs))


run_lightcone.__doc__ = generate_lightcone.__doc__
