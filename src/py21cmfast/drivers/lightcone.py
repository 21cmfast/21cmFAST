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

from .. import __version__
from ..c_21cmfast import lib
from ..io import h5
from ..io.caching import CacheConfig, OutputCache, RunCache
from ..lightcones import Lightconer, RectilinearLightconer
from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import (
    BrightnessTemp,
    HaloBox,
    InitialConditions,
    IonizedBox,
    PerturbedField,
    PerturbHaloField,
    TsBox,
)
from ..wrapper.photoncons import _get_photon_nonconservation_data, setup_photon_cons
from ..wrapper.rsd import compute_rsds
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
        return self.simulation_options.BOX_LEN / self.simulation_options.HII_DIM

    @property
    def lightcone_dimensions(self) -> tuple[float, float, float]:
        """Lightcone size over each dimension -- tuple of (x,y,z) in Mpc."""
        return (
            self.simulation_options.BOX_LEN,
            self.simulation_options.BOX_LEN,
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
    def simulation_options(self):
        """Matter params shared by all datasets."""
        return self.inputs.simulation_options

    @property
    def matter_options(self):
        """Matter flags shared by all datasets."""
        return self.inputs.matter_options

    @property
    def cosmo_params(self):
        """Cosmo params shared by all datasets."""
        return self.inputs.cosmo_params

    @property
    def astro_options(self):
        """Flag Options shared by all datasets."""
        return self.inputs.astro_options

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

    def save(
        self,
        path: str | Path,
        clobber=False,
        lowz_buffer_pixels: int = 0,
        highz_buffer_pixels: int = 0,
    ):
        """Save the lightcone object to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_mode = "w" if clobber else "a"
        with h5py.File(path, file_mode) as fl:
            fl.attrs["lightcone"] = True  # marker identifying this as a lightcone box

            fl.attrs["last_completed_node"] = self._last_completed_node
            fl.attrs["last_completed_lcidx"] = self._last_completed_lcidx
            fl.attrs["lowz_buffer_pixels"] = lowz_buffer_pixels
            fl.attrs["highz_buffer_pixels"] = highz_buffer_pixels

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
            last_completed_node = fl.attrs["last_completed_node"]

            save_idx = (
                len(self.lightcone_distances)
                if last_completed_lcidx < 0
                else last_completed_lcidx
            )
            for k, v in self.lightcones.items():
                fl["lightcones"][k][..., lcidx:save_idx] = v[..., lcidx:save_idx]

            global_q = fl["global_quantities"]
            for k, v in self.global_quantities.items():
                global_q[k][last_completed_node + 1 : node_index + 1] = v[
                    last_completed_node + 1 : node_index + 1
                ]

            fl.attrs["last_completed_lcidx"] = lcidx
            fl.attrs["last_completed_node"] = node_index

            self._last_completed_lcidx = lcidx
            self._last_completed_node = node_index

    def trim(self, distances: np.ndarray) -> Self:
        """Create a new lightcone box containing only the desired distances range."""
        inds = np.logical_and(
            self.lightcone_distances >= distances.min(),
            self.lightcone_distances <= distances.max(),
        )
        return attrs.evolve(
            self,
            lightcone_distances=self.lightcone_distances[inds],
            lightcones={k: v[..., inds] for k, v in self.lightcones.items()},
        )

    @classmethod
    def from_file(
        cls, path: str | Path, safe: bool = True, remove_buffer: bool = True
    ) -> Self:
        """Create a new instance from a saved lightcone on disk."""
        kwargs = {}
        with h5py.File(path, "r") as fl:
            if not fl.attrs.get("lightcone", False):
                raise ValueError(f"The file {path} is not a lightcone file!")

            kwargs["inputs"] = h5.read_inputs(fl, safe=safe)
            kwargs["last_completed_node"] = fl.attrs["last_completed_node"]
            kwargs["last_completed_lcidx"] = fl.attrs["last_completed_lcidx"]

            if remove_buffer:
                lowz_buffer_pixels = fl.attrs.get("lowz_buffer_pixels", 0)
                highz_buffer_pixels = fl.attrs.get("highz_buffer_pixels", 0)
            else:
                lowz_buffer_pixels = 0
                highz_buffer_pixels = 0

            highz_buffer_pixels = len(fl["lightcone_distances"]) - highz_buffer_pixels

            grp = fl["photon_nonconservation_data"]
            kwargs["photon_nonconservation_data"] = {k: v[...] for k, v in grp.items()}

            boxes = fl["lightcones"]
            kwargs["lightcones"] = {
                k: boxes[k][..., lowz_buffer_pixels:highz_buffer_pixels] for k in boxes
            }

            glb = fl["global_quantities"]
            kwargs["global_quantities"] = {k: glb[k][...] for k in glb}
            kwargs["lightcone_distances"] = (
                fl["lightcone_distances"][..., lowz_buffer_pixels:highz_buffer_pixels]
                * units.Mpc
            )

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


def _check_desired_arrays_exist(desired_arrays: list[str], inputs: InputParameters):
    possible_outputs = [
        InitialConditions.new(inputs),
        PerturbedField.new(inputs, redshift=0),
        TsBox.new(inputs, redshift=0),
        HaloBox.new(inputs, redshift=0),
        IonizedBox.new(inputs, redshift=0),
        BrightnessTemp.new(inputs, redshift=0),
    ]
    for name in desired_arrays:
        exists = False
        for output in possible_outputs:
            if name in output.arrays or name in ["log10_mturn_acg", "log10_mturn_mcg"]:
                exists = True
                break
        if not exists:
            raise ValueError(
                f"You asked for {name} but it is not computed for the inputs: {inputs}"
            )


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
        lightcone = LightCone.from_file(lightcone_filename, remove_buffer=False)
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
                lightconer.get_shape(inputs.simulation_options),
                dtype=np.float32,
            )
            for quantity in lightconer.quantities
        }

        if inputs.astro_options.APPLY_RSDS:
            lc["los_velocity"] = np.zeros(
                lightconer.get_shape(inputs.simulation_options), dtype=np.float32
            )
            if inputs.astro_options.USE_TS_FLUCT:
                lc["tau_21"] = np.zeros(
                    lightconer.get_shape(inputs.simulation_options), dtype=np.float32
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
    lc_distances: np.array,
    photon_nonconservation_data: dict,
    pt_halos: list[PerturbHaloField],
    regenerate: bool | None = None,
    global_quantities: tuple[str] = ("brightness_temp", "neutral_fraction"),
    cache: OutputCache = _ocache,
    cleanup: bool = True,
    write: CacheConfig = _cache,
    always_purge: bool = False,
    progressbar: bool = False,
    lightcone_filename: str | Path | None = None,
):
    # Get the redshift through which we scroll and evaluate the ionization field.
    scrollz = np.array([pf.redshift for pf in perturbed_fields])

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
        initial_conditions=initial_conditions,
        photon_nonconservation_data=photon_nonconservation_data,
        minimum_node=lightcone._last_completed_node,
        **iokw,
    )

    if idx < lightcone._last_completed_node:
        warnings.warn(
            f"The cache at {cache} only contains complete coeval boxes for {idx + 1} redshift nodes, "
            f"instead of {lightcone._last_completed_node + 1}, which is the current checkpointing "
            f"redshift of the lightcone. Repeating the higher-z calculations...",
            stacklevel=2,
        )

    # Find how many pixels on either end of the lightcone are in the "buffer" region.
    # These are used to generate RSDs, but are then removed from the lightcone before
    # returning.
    lowz_buffer_pixels = np.sum(lc_distances.min() > lightcone.lightcone_distances)
    highz_buffer_pixels = np.sum(lc_distances.max() < lightcone.lightcone_distances)

    if lightcone_filename and not Path(lightcone_filename).exists():
        lightcone.save(
            lightcone_filename,
            lowz_buffer_pixels=lowz_buffer_pixels,
            highz_buffer_pixels=highz_buffer_pixels,
        )

    for iz, coeval in _redshift_loop_generator(
        inputs=inputs,
        initial_conditions=initial_conditions,
        all_redshifts=scrollz,
        perturbed_field=perturbed_fields,
        pt_halos=pt_halos,
        write=write,
        cleanup=cleanup,
        always_purge=always_purge,
        progressbar=progressbar,
        photon_nonconservation_data=photon_nonconservation_data,
        start_idx=lightcone._last_completed_node + 1,
        init_coeval=prev_coeval,
        iokw=iokw,
    ):
        # Save mean/global quantities
        for quantity in global_quantities:
            if quantity == "log10_mturn_acg":
                lightcone.global_quantities[quantity][iz] = (
                    coeval.ionized_box.log10_Mturnover_ave
                )
            elif quantity == "log10_mturn_mcg":
                lightcone.global_quantities[quantity][iz] = (
                    coeval.ionized_box.log10_Mturnover_MINI_ave
                )
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
                    # save the lowest index
                    if lc_index is None:
                        lc_index = idx

            # only checkpoint if we have slices
            if lightcone_filename and lc_index is not None:
                lightcone.make_checkpoint(
                    lightcone_filename, lcidx=lc_index, node_index=iz
                )

        prev_coeval = coeval

        # last redshift things
        if iz == len(scrollz) - 1:
            if lib.photon_cons_allocated:
                lib.FreePhotonConsMemory()

            if inputs.astro_options.APPLY_RSDS:
                tb_with_rsds = compute_rsds(
                    brightness_temp=lightcone.lightcones["brightness_temp"],
                    los_velocity=lightcone.lightcones["los_velocity"],
                    redshifts=lightcone.lightcone_redshifts,
                    distances=lightcone.lightcone_distances,
                    inputs=inputs,
                    tau_21=lightcone.lightcones["tau_21"]
                    if inputs.astro_options.USE_TS_FLUCT
                    else None,
                    periodic=False,
                    n_subcells=inputs.astro_params.N_RSD_STEPS
                    if inputs.astro_options.SUBCELL_RSD
                    else 0,
                )
                lightcone.lightcones["brightness_temp_with_rsds"] = tb_with_rsds

                if lightcone_filename:
                    if Path(lightcone_filename).exists():
                        with h5py.File(lightcone_filename, "a") as fl:
                            fl["lightcones"]["brightness_temp_with_rsds"] = tb_with_rsds
                    else:
                        lightcone.save(
                            lightcone_filename,
                            lowz_buffer_pixels=lowz_buffer_pixels,
                            highz_buffer_pixels=highz_buffer_pixels,
                        )

                if inputs.astro_options.SUBCELL_RSD:
                    lightcone = lightcone.trim(lc_distances)

        yield iz, coeval.redshift, coeval, lightcone


@high_level_func
def generate_lightcone(
    *,
    lightconer: Lightconer,
    inputs: InputParameters,
    global_quantities=("brightness_temp", "neutral_fraction"),
    initial_conditions: InitialConditions | None = None,
    cleanup: bool = True,
    write: CacheConfig = _cache,
    cache: OutputCache | None = _ocache,
    regenerate: bool = True,
    always_purge: bool = False,
    progressbar: bool = False,
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
    progressbar: bool, optional
        If True, a progress bar will be displayed throughout the simulation. Defaults to False.
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
    lc_distances = lightconer.lc_distances.copy()

    # Validate the lightconer options and return a "ghost" lightconer if we require
    # some extra buffer pixels (e.g. for RSDs).
    lightconer = lightconer.validate_options(inputs)

    if isinstance(write, bool):
        write = CacheConfig() if write else CacheConfig.off()

    _check_desired_arrays_exist(global_quantities, inputs)
    _check_desired_arrays_exist(lightconer.quantities, inputs)

    # while we still use the full list for caching etc, we don't need to run below the lightconer instance
    #   So stop one after the lightconer
    scrollz = np.copy(inputs.node_redshifts)
    below_lc_z = scrollz <= min(lightconer.lc_redshifts)
    if np.any(below_lc_z):
        final_node = np.where(below_lc_z)[0][0]  # first node below the lightcone
        scrollz = scrollz[: final_node + 1]  # inclusive to get the last few entries

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
        all_redshifts=inputs.node_redshifts,
        initial_conditions=initial_conditions,
        inputs=inputs,
        write=write,
        always_purge=always_purge,
        progressbar=progressbar,
        **iokw,
    )

    yield from _run_lightcone_from_perturbed_fields(
        initial_conditions=initial_conditions,
        perturbed_fields=perturbed_fields,
        lightconer=lightconer,
        inputs=inputs,
        lc_distances=lc_distances,
        regenerate=regenerate,
        pt_halos=pt_halos,
        photon_nonconservation_data=photon_nonconservation_data,
        global_quantities=global_quantities,
        cache=cache,
        write=write,
        cleanup=cleanup,
        always_purge=always_purge,
        progressbar=progressbar,
        lightcone_filename=lightcone_filename,
    )


def run_lightcone(**kwargs):  # noqa: D103
    return exhaust(generate_lightcone(**kwargs))


run_lightcone.__doc__ = generate_lightcone.__doc__
