"""Compute simulations that evolve over redshift."""

import logging
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Self, get_args

import attrs
import h5py
import numpy as np
from rich import progress as prg
from rich.console import Console
from rich.progress import Progress

from .. import __version__
from ..c_21cmfast import lib
from ..io import h5
from ..io.caching import CacheConfig, OutputCache, RunCache
from ..rsds import apply_rsds, include_dvdr_in_tau21
from ..wrapper.arrays import Array
from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import (
    BrightnessTemp,
    HaloBox,
    HaloCatalog,
    InitialConditions,
    IonizedBox,
    OutputStruct,
    PerturbedField,
    TsBox,
)
from ..wrapper.photoncons import _get_photon_nonconservation_data, setup_photon_cons
from . import single_field as sf
from ._param_config import high_level_func

logger = logging.getLogger(__name__)

_console = Console()


def _progressbar(**kwargs):
    return Progress(
        prg.TextColumn("[progress.description]{task.description:>24}"),
        prg.BarColumn(),
        prg.TaskProgressColumn(),
        prg.TextColumn("•"),
        prg.MofNCompleteColumn(),
        "[green]redshifts",
        prg.TextColumn("•"),
        prg.TimeElapsedColumn(),
        prg.TextColumn("•"),
        prg.TimeRemainingColumn(),
        "remaining",
        console=_console,
        **kwargs,
    )


@attrs.define
class Coeval:
    """A full coeval box with all associated data."""

    initial_conditions: InitialConditions = attrs.field(
        validator=attrs.validators.instance_of(InitialConditions)
    )
    perturbed_field: PerturbedField = attrs.field(
        validator=attrs.validators.instance_of(PerturbedField)
    )
    ionized_box: IonizedBox = attrs.field(
        validator=attrs.validators.instance_of(IonizedBox)
    )
    brightness_temperature: BrightnessTemp = attrs.field(
        validator=attrs.validators.instance_of(BrightnessTemp)
    )
    ts_box: TsBox = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(TsBox)),
    )
    halobox: HaloBox = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(HaloBox)),
    )
    photon_nonconservation_data: dict = attrs.field(factory=dict)

    def __getattr__(self, name):
        """
        Get underlying Array objects as attributes.

        This method allows accessing arrays from OutputStruct objects within the Coeval instance
        as if they were direct attributes of the Coeval object.

        Parameters
        ----------
        name : str
            The name of the attribute being accessed.

        Returns
        -------
        Any
            The value of the requested array from the appropriate OutputStruct object.

        Raises
        ------
        AttributeError
            If the requested attribute is not found in any of the OutputStruct objects.
        """
        # We only want to expose fields that are part of the Coeval object
        for box in attrs.asdict(self, recurse=False).values():
            if isinstance(box, OutputStruct) and name in box.arrays:
                return box.get(name)
        raise AttributeError(f"Coeval has no attribute '{name}'")

    @property
    def output_structs(self) -> dict[str, OutputStruct]:
        """
        Get a dictionary of OutputStruct objects contained in this Coeval instance.

        This property method returns a dictionary containing all the OutputStruct
        objects that are attributes of the Coeval instance. It filters out any
        non-OutputStruct attributes.

        Returns
        -------
        dict[str, OutputStruct]
            A dictionary where the keys are attribute names and the values are
            the corresponding OutputStruct objects.
        """
        return {
            k: v
            for k, v in attrs.asdict(self, recurse=False).items()
            if isinstance(v, OutputStruct)
        }

    @classmethod
    def get_fields(cls, ignore_structs: tuple[str] = ()) -> list[str]:
        """Obtain a list of name of simulation boxes saved in the Coeval object."""
        output_structs = []
        for fld in attrs.fields(cls):
            if fld.name in ignore_structs:
                continue

            if issubclass(fld.type, OutputStruct):
                output_structs.append(fld.type)
            else:
                args = get_args(fld.type)
                for k in args:
                    if issubclass(k, OutputStruct):
                        output_structs.append(k)
                        break

        pointer_fields = []
        for struct in output_structs:
            pointer_fields += [
                k for k, v in attrs.fields_dict(struct).items() if v.type == Array
            ]

        return pointer_fields

    @property
    def redshift(self) -> float:
        """The redshift of the coeval box."""
        return self.perturbed_field.redshift

    @property
    def inputs(self) -> InputParameters:
        """An InputParameters object associated with the coeval box."""
        return self.brightness_temperature.inputs

    @property
    def simulation_options(self):
        """Matter Params shared by all datasets."""
        return self.inputs.simulation_options

    @property
    def matter_options(self):
        """Matter Flags shared by all datasets."""
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

    def save(self, path: str | Path, clobber=False):
        """Save the Coeval object to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        file_mode = "w" if clobber else "a"
        with h5py.File(path, file_mode) as fl:
            fl.attrs["coeval"] = True  # marker identifying this as a coeval box
            fl.attrs["__version__"] = __version__

            grp = fl.create_group("photon_nonconservation_data")
            for k, v in self.photon_nonconservation_data.items():
                grp[k] = v

        output_structs = self.output_structs
        for struct in output_structs.values():
            h5.write_output_to_hdf5(struct, path, mode="a")

    def include_dvdr_in_tau21(self, axis: str = "z", periodic: bool = True):
        """Include velocity gradient corrections to the brightness temperature field.

        Parameters
        ----------
        axis: str, optioanl
            The assumed axis of the line-of-sight. Options are "x", "y", or "z". Default is "z".
        periodic: bool, optioanl
            Whether to assume periodic boundary conditions along the line-of-sight. Default is True.

        Returns
        -------
        tb_with_dvdr
            A box of the brightness temperature, with velocity gradient corrections.
        """
        if not hasattr(self, "velocity_" + axis):
            if axis not in ["x", "y", "z"]:
                raise ValueError("`axis` can only be `x`, `y` or `z`.")
            else:
                raise ValueError(
                    "You asked for axis = '"
                    + axis
                    + "', but the coeval doesn't have velocity_"
                    + axis
                    + "!"
                    " Set matter_options.KEEP_3D_VELOCITIES=True next time you call run_coeval if you "
                    "wish to set axis=`" + axis + "'."
                )

        return include_dvdr_in_tau21(
            brightness_temp=self.brightness_temp,
            los_velocity=getattr(self, "velocity_" + axis),
            redshifts=self.redshift,  # TODO: do we want to use a single redshift? Or a redshift array that is determined from the coeval los?
            inputs=self.inputs,
            tau_21=self.tau_21 if self.inputs.astro_options.USE_TS_FLUCT else None,
            periodic=periodic,
        )

    def apply_rsds(
        self,
        field: str = "brightness_temp",
        axis: str = "z",
        periodic: bool = True,
        n_rsd_subcells: int = 4,
    ):
        """Apply redshift-space distortions to a particular field of the coeval box.

        Parameters
        ----------
        field: str, optional
            The field onto which redshift-space distortions shall be applied. Default is "brightness_temp".
        axis: str, optioanl
            The assumed axis of the line-of-sight. Options are "x", "y", or "z". Default is "z".
        periodic: bool, optioanl
            Whether to assume periodic boundary conditions along the line-of-sight. Default is True.
        n_rsd_subcells: int, optional
            The number of subcells into which each cell is divided when redshift space distortions are applied. Default is 4.

        Returns
        -------
        field_with_rsds
            A box of the field, with redshift space distortions.
        """
        if not hasattr(self, "velocity_" + axis):
            if axis not in ["x", "y", "z"]:
                raise ValueError("`axis` can only be `x`, `y` or `z`.")
            else:
                raise ValueError(
                    "You asked for axis = '"
                    + axis
                    + "', but the coeval doesn't have velocity_"
                    + axis
                    + "!"
                    " Set matter_options.KEEP_3D_VELOCITIES=True next time you call run_coeval if you "
                    "wish to set axis=`" + axis + "'."
                )

        return apply_rsds(
            field=getattr(self, field),
            los_velocity=getattr(self, "velocity_" + axis),
            redshifts=self.redshift,  # TODO: do we want to use a single redshift? Or a redshift array that is determined from the coeval los?
            inputs=self.inputs,
            periodic=periodic,
            n_rsd_subcells=n_rsd_subcells,
        )

    def apply_velocity_corrections(
        self, axis: str = "z", periodic: bool = True, n_rsd_subcells: int = 4
    ):
        """Apply velocity gradient corrections and redshift-space distortions to the brightness temperature field.

        Parameters
        ----------
        axis: str, optioanl
            The assumed axis of the line-of-sight. Options are "x", "y", or "z". Default is "z".
        periodic: bool, optioanl
            Whether to assume periodic boundary conditions along the line-of-sight. Default is True.
        n_rsd_subcells: int, optional
            The number of subcells into which each cell is divided when redshift space distortions are applied. Default is 4.

        Returns
        -------
        field_with_rsds
            A box of the brightness temperature, with velocity gradient corrections and redshift-space distortions.
        """
        if not hasattr(self, "velocity_" + axis):
            if axis not in ["x", "y", "z"]:
                raise ValueError("`axis` can only be `x`, `y` or `z`.")
            else:
                raise ValueError(
                    "You asked for axis = '"
                    + axis
                    + "', but the coeval doesn't have velocity_"
                    + axis
                    + "!"
                    " Set matter_options.KEEP_3D_VELOCITIES=True next time you call run_coeval if you "
                    "wish to set axis=`" + axis + "'."
                )

        tb_with_dvdr = include_dvdr_in_tau21(
            brightness_temp=self.brightness_temp,
            los_velocity=getattr(self, "velocity_" + axis),
            redshifts=self.redshift,  # TODO: do we want to use a single redshift? Or a redshift array that is determined from the coeval los?
            inputs=self.inputs,
            tau_21=self.tau_21 if self.inputs.astro_options.USE_TS_FLUCT else None,
            periodic=periodic,
        )

        return apply_rsds(
            field=tb_with_dvdr,
            los_velocity=getattr(self, "velocity_" + axis),
            redshifts=self.redshift,  # TODO: do we want to use a single redshift? Or a redshift array that is determined from the coeval los?
            inputs=self.inputs,
            periodic=periodic,
            n_rsd_subcells=n_rsd_subcells,
        )

    @classmethod
    def from_file(cls, path: str | Path, safe: bool = True) -> Self:
        """Read the Coeval object from disk and return it."""
        path = Path(path)
        if not path.exists():
            raise FileExistsError(f"The file {path} does not exist!")

        selfdict = attrs.fields_dict(cls)
        type_to_name = {v.type.__name__: k for k, v in selfdict.items()}

        with h5py.File(path, "r") as fl:
            if not fl.attrs.get("coeval", False):
                raise ValueError(f"The file {path} is not a Coeval file!")

            keys = set(fl.keys())

            grp = fl["photon_nonconservation_data"]
            photoncons = {k: v[...] for k, v in grp.items()}
            keys.remove("photon_nonconservation_data")

            kwargs = {
                type_to_name[k]: h5.read_output_struct(path, struct=k, safe=safe)
                for k in keys
            }
            return cls(photon_nonconservation_data=photoncons, **kwargs)

    def __eq__(self, other):
        """Determine if this is equal to another object."""
        return (
            isinstance(other, self.__class__)
            and other.inputs == self.inputs
            and self.redshift == other.redshift
        )


def evolve_halos(
    inputs: InputParameters,
    all_redshifts: list[float],
    write: CacheConfig,
    initial_conditions: InitialConditions,
    cache: OutputCache,
    regenerate: bool,
    progressbar: bool = False,
):
    """
    Evolve and perturb halo fields across multiple redshifts.

    This function computes and evolves halo fields for a given set of redshifts,
    applying perturbations to each halo list. It processes redshifts in reverse order
    to account for descendant halos.

    Parameters
    ----------
    inputs : InputParameters
        Input parameters for the simulation.
    all_redshifts : list[float]
        List of redshifts to process, in descending order.
    write : CacheConfig
        Configuration for writing output to cache.
    initial_conditions : InitialConditions
        Initial conditions for the simulation.
    cache : OutputCache
        Cache object for storing and retrieving computed results.
    regenerate : bool
        Flag to indicate whether to regenerate results or use cached values.
    progressbar: bool, optional
        If True, a progress bar will be displayed throughout the simulation. Defaults to False.

    Returns
    -------
    list
        A list of perturbed halo fields for each redshift, in ascending redshift order.
        Returns an empty list if halo fields are not used or fixed grids are enabled.
    """
    # get the halos (reverse redshift order)
    if not inputs.matter_options.has_discrete_halos:
        return []

    if not write.halo_catalog and len(all_redshifts) > 1:
        warnings.warn(
            "You have turned off caching for the perturbed halo fields, but are"
            " evolving them across multiple redshifts. This will result in very high memory usage",
            stacklevel=2,
        )

    halofield_list = []
    kw = {
        "initial_conditions": initial_conditions,
        "cache": cache,
        "regenerate": regenerate,
    }
    halos_desc = None
    with _progressbar(disable=not progressbar) as _progbar:
        for _i, z in _progbar.track(
            enumerate(all_redshifts[::-1]),
            description="Evolving Halos",
            total=len(all_redshifts),
        ):
            halos = sf.determine_halo_catalog(
                redshift=z,
                inputs=inputs,
                descendant_halos=halos_desc,
                write=write.halo_catalog,
                **kw,
            )

            halofield_list.append(halos)

            if z in inputs.node_redshifts:
                # Only evolve on the node_redshifts, not any redshifts in-between
                # that the user might care about.
                # we never want to store every halofield
                if write.halo_catalog and (halos_desc is not None):
                    halos_desc.purge()
                halos_desc = halos

    # reverse to get the right redshift order
    return halofield_list[::-1]


@high_level_func
def generate_coeval(
    *,
    inputs: InputParameters | None = None,
    out_redshifts: float | tuple[float] = (),
    regenerate: bool | None = None,
    write: CacheConfig | bool = True,
    cache: OutputCache | None = None,
    initial_conditions: InitialConditions | None = None,
    cleanup: bool = True,
    progressbar: bool = False,
):
    r"""
    Perform a full coeval simulation of all fields at given redshifts.

    This is generally the easiest and most efficient way to generate a set of coeval cubes at a
    given set of redshifts. It self-consistently deals with situations in which the field needs to be
    evolved, and does this with the highest memory-efficiency, only returning the desired redshift.
    All other calculations are by default stored in the on-disk cache so they can be re-used at a
    later time.

    Some calculations of the coeval quantities require redshift evolution, i.e. the
    calculation of higher-redshift coeval boxes up to some maximum redshift in order
    to integrate the quantities over cosmic time. The redshifts that define this
    evolution are set by the ``inputs.node_redshifts`` parameter. However, in some
    simple cases, this evolution is not required, and this parameter can be empty.
    Thus there is a distinction between the redshifts required for computing the physics
    (i.e. ``inputs.node_redshifts``) and the redshifts at which the user wants to
    obtain the resulting coeval cubes. The latter is controlled by ``out_redshifts``.
    If not set, ``out_redshifts`` will be set to ``inputs.node_redshifts``, so that
    all computed redshifts are returned as coeval boxes.

    .. note:: User-supplied ``out_redshifts`` are *not* used in the redshift evolution,
              so that the results depend precisely on the ``node_redshifts`` defined
              in the input parameters.

    Parameters
    ----------
    inputs: :class:`~InputParameters`
        This object specifies the input parameters for the run, including the random seed
    out_redshifts: array_like, optional
        A single redshift, or multiple redshifts, at which to return results. By default,
        use all the ``inputs.node_redshifts``. If neither is specified, an error will be
        raised.
    regenerate : bool
        If True, regenerate all fields, even if they are in the cache.
    write : :class:`~py21cmfast.cache.CacheConfig`, optional
        Either a bool specifying whether to write _all_ the boxes to cache (or none of
        them), or a :class:`~py21cmfast.cache.CacheConfig` object specifying which boxes
        to write.
    cache : :class:`~py21cmfast.cache.OutputCache`, optional
        The cache object to use for reading and writing data from the cache. This should
        be an instance of :class:`~py21cmfast.cache.OutputCache`, which depends solely
        on specifying a directory to host the cache.
    initial_conditions : :class:`~InitialConditions`, optional
        If given, use these intial conditions as a basis for computing the other
        fields, instead of re-computing the ICs. If this is defined, the ``inputs`` do
        not need to be defined (but can be, in order to overwrite the ``node_redshifts``).
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculated has different shape, errors will occur
        if memory is not cleaned. Note that internally, this is set to False until the
        last iteration.
    progressbar: bool, optional
        If True, a progress bar will be displayed throughout the simulation. Defaults to False.

    Returns
    -------
    coevals : list of :class:`~py21cmfast.drivers.coeval.Coeval`
        The full data for the Coeval class, with init boxes, perturbed fields, ionized boxes,
        brightness temperature, and potential data from the conservation of photons. A
        list of such objects, one for each redshift in ``out_redshifts``.
    """
    if cache is None:
        cache = OutputCache(".")

    if isinstance(write, bool):
        write = CacheConfig() if write else CacheConfig.off()

    if not out_redshifts:
        out_redshifts = inputs.node_redshifts

    if not out_redshifts and not inputs.node_redshifts:
        raise ValueError("out_redshifts must be given if inputs has no node redshifts")

    iokw = {"regenerate": regenerate, "cache": cache}

    if not hasattr(out_redshifts, "__len__"):
        out_redshifts = [out_redshifts]

    if isinstance(out_redshifts, np.ndarray):
        out_redshifts = out_redshifts.tolist()

    # Get the list of redshifts we need to scroll through.
    all_redshifts = _get_required_redshifts_coeval(inputs, out_redshifts)

    (
        initial_conditions,
        perturbed_field,
        halofield_list,
        photon_nonconservation_data,
    ) = _setup_ics_and_pfs_for_scrolling(
        all_redshifts=all_redshifts,
        inputs=inputs,
        initial_conditions=initial_conditions,
        write=write,
        progressbar=progressbar,
        **iokw,
    )

    # get the first node redshift that is above all output redshifts
    first_out_node = None
    if inputs.node_redshifts:
        # if any output redshift is above all nodes, start from the start
        if max(out_redshifts) > max(inputs.node_redshifts):
            first_out_node = -1
        # otherwise, find the first node above all user redshifts and start there
        elif max(out_redshifts) >= min(inputs.node_redshifts):
            nodes_in_outputs = np.array(inputs.node_redshifts) <= max(out_redshifts)
            first_out_node = np.where(nodes_in_outputs)[0][0] - 1
    idx, coeval = _obtain_starting_point_for_scrolling(
        inputs=inputs,
        initial_conditions=initial_conditions,
        photon_nonconservation_data=photon_nonconservation_data,
        minimum_node=first_out_node,
        **iokw,
    )
    # convert node_redshift index to all_redshift index
    if idx > 0:
        idx = np.argmin(np.fabs(np.array(all_redshifts) - inputs.node_redshifts[idx]))

    for _, coeval in _redshift_loop_generator(  # noqa: B020
        inputs=inputs,
        all_redshifts=all_redshifts,
        initial_conditions=initial_conditions,
        photon_nonconservation_data=photon_nonconservation_data,
        perturbed_field=perturbed_field,
        halofield_list=halofield_list,
        write=write,
        cleanup=cleanup,
        progressbar=progressbar,
        iokw=iokw,
        init_coeval=coeval,
        start_idx=idx + 1,
    ):
        yield coeval, coeval.redshift in out_redshifts

    if lib.photon_cons_allocated:
        lib.FreePhotonConsMemory()


def run_coeval(**kwargs) -> list[Coeval]:  # noqa: D103
    return [coeval for coeval, in_outputs in generate_coeval(**kwargs) if in_outputs]


run_coeval.__doc__ = generate_coeval.__doc__


def _obtain_starting_point_for_scrolling(
    inputs: InputParameters,
    initial_conditions: InitialConditions,
    photon_nonconservation_data: dict,
    cache: OutputCache,
    regenerate: bool,
    minimum_node: int | None = None,
):
    outputs = None

    if minimum_node is None:
        # By default, check for completeness at all nodes, starting at
        # the last one.
        minimum_node = len(inputs.node_redshifts) - 1

    if minimum_node < 0 or inputs.matter_options.has_discrete_halos or regenerate:
        # TODO: (low priority) implement a backward loop for finding first halo files
        #   Noting that we need *all* the halo fields in the cache to run
        return (
            -1,
            None,
        )

    logger.info(f"Determining pre-cached boxes for the run in {cache}")
    rc = RunCache.from_inputs(inputs, cache)

    for idx in range(minimum_node, -1, -1):
        if not rc.is_complete_at(index=idx):
            continue

        _z = inputs.node_redshifts[idx]
        outputs = rc.get_all_boxes_at_z(z=_z)
        break

    # Create a Coeval from the outputs
    if outputs is not None:
        return idx, Coeval(
            initial_conditions=initial_conditions,
            perturbed_field=outputs["PerturbedField"],
            ionized_box=outputs["IonizedBox"],
            brightness_temperature=outputs["BrightnessTemp"],
            ts_box=outputs.get("TsBox", None),
            halobox=outputs.get("Halobox", None),
            photon_nonconservation_data=photon_nonconservation_data,
        )
    else:
        return -1, None


def _redshift_loop_generator(
    inputs: InputParameters,
    initial_conditions: InitialConditions,
    all_redshifts: Sequence[float],
    perturbed_field: list[PerturbedField],
    halofield_list: list[HaloCatalog],
    write: CacheConfig,
    iokw: dict,
    cleanup: bool,
    progressbar: bool,
    photon_nonconservation_data: dict,
    start_idx: int = 0,
    init_coeval: Coeval | None = None,
):
    if isinstance(write, bool):
        write = CacheConfig()

    # Iterate through redshift from top to bottom
    hbox_arr = []

    prev_coeval = init_coeval
    this_coeval = None

    this_halobox = None
    this_spin_temp = None
    this_halofield = None
    this_xraysource = None

    kw = {
        **iokw,
        "initial_conditions": initial_conditions,
    }
    with _progressbar(disable=not progressbar) as _progbar:
        for iz, z in _progbar.track(
            enumerate(all_redshifts),
            description="Evolving Astrophysics",
            total=len(all_redshifts),
        ):
            if not progressbar:
                logger.info(
                    f"Computing Redshift {z} ({iz + 1}/{len(all_redshifts)}) iterations."
                )
            if iz < start_idx:
                continue

            this_perturbed_field = perturbed_field[iz]
            this_perturbed_field.load_all()

            if inputs.matter_options.lagrangian_source_grid:
                if inputs.matter_options.has_discrete_halos:
                    this_halofield = halofield_list[iz]
                    this_halofield.load_all()

                this_halobox = sf.compute_halo_grid(
                    inputs=inputs,
                    halo_catalog=this_halofield,
                    redshift=z,
                    previous_ionize_box=getattr(prev_coeval, "ionized_box", None),
                    previous_spin_temp=getattr(prev_coeval, "ts_box", None),
                    write=write.halobox,
                    **kw,
                )

            if inputs.astro_options.USE_TS_FLUCT:
                if inputs.matter_options.lagrangian_source_grid:
                    # append the halo redshift array so we have all halo boxes [z,zmax]
                    this_xraysource = sf.compute_xray_source_field(
                        redshift=z,
                        hboxes=[*hbox_arr, this_halobox],
                        write=write.xray_source_box,
                        **kw,
                    )

                this_spin_temp = sf.compute_spin_temperature(
                    inputs=inputs,
                    previous_spin_temp=getattr(prev_coeval, "ts_box", None),
                    perturbed_field=this_perturbed_field,
                    xray_source_box=this_xraysource,
                    write=write.spin_temp,
                    **kw,
                    cleanup=(cleanup and z == all_redshifts[-1]),
                )
                # Purge XraySourceBox because it's enormous
                if inputs.matter_options.lagrangian_source_grid:
                    this_xraysource.purge(force=True)

            this_ionized_box = sf.compute_ionization_field(
                inputs=inputs,
                previous_ionized_box=getattr(prev_coeval, "ionized_box", None),
                perturbed_field=this_perturbed_field,
                # perturb field *not* interpolated here.
                previous_perturbed_field=getattr(prev_coeval, "perturbed_field", None),
                halobox=this_halobox,
                spin_temp=this_spin_temp,
                write=write.ionized_box,
                **kw,
            )

            this_bt = sf.brightness_temperature(
                ionized_box=this_ionized_box,
                perturbed_field=this_perturbed_field,
                spin_temp=this_spin_temp,
                write=write.brightness_temp,
                **iokw,
            )

            if inputs.astro_options.PHOTON_CONS_TYPE == "z-photoncons":
                # Updated info at each z.
                photon_nonconservation_data = _get_photon_nonconservation_data()

            this_coeval = Coeval(
                initial_conditions=initial_conditions,
                perturbed_field=this_perturbed_field,
                ionized_box=this_ionized_box,
                brightness_temperature=this_bt,
                ts_box=this_spin_temp,
                halobox=this_halobox,
                photon_nonconservation_data=photon_nonconservation_data,
            )

            # We purge previous fields and those we no longer need
            if prev_coeval is not None:
                prev_coeval.perturbed_field.purge()
                if (
                    inputs.matter_options.lagrangian_source_grid
                    and write.halobox
                    and iz + 1 < len(all_redshifts)
                ):
                    for hbox in hbox_arr:
                        hbox.prepare_for_next_snapshot(next_z=all_redshifts[iz + 1])

            if this_halofield is not None:
                this_halofield.purge()

            if z in inputs.node_redshifts:
                # Only evolve on the node_redshifts, not any redshifts in-between
                # that the user might care about.
                prev_coeval = this_coeval
                hbox_arr += [this_halobox]

            # yield before the cleanup, so we can get at the fields before they are purged
            yield iz, this_coeval


def _setup_ics_and_pfs_for_scrolling(
    all_redshifts: Sequence[float],
    initial_conditions: InitialConditions | None,
    inputs: InputParameters,
    write: CacheConfig,
    progressbar: bool,
    **iokw,
) -> tuple[InitialConditions, list[PerturbedField], list[HaloCatalog], dict]:
    if initial_conditions is None:
        initial_conditions = sf.compute_initial_conditions(
            inputs=inputs, write=write.initial_conditions, **iokw
        )

    # We can go ahead and purge some of the stuff in the initial_conditions, but only if
    # it is cached -- otherwise we could be losing information.
    if write.initial_conditions:
        initial_conditions.prepare_for_perturb()
    kw = {
        "initial_conditions": initial_conditions,
        **iokw,
    }
    photon_nonconservation_data = {}
    if inputs.astro_options.PHOTON_CONS_TYPE != "no-photoncons":
        photon_nonconservation_data = setup_photon_cons(**kw)

    if (
        inputs.astro_options.PHOTON_CONS_TYPE == "z-photoncons"
        and np.amin(all_redshifts) < inputs.astro_params.PHOTONCONS_CALIBRATION_END
    ):
        raise ValueError(
            f"You have passed a redshift (z = {np.amin(all_redshifts)}) that is lower than"
            "the endpoint of the photon non-conservation correction"
            f"(astro_params.PHOTONCONS_CALIBRATION_END = {inputs.astro_params.PHOTONCONS_CALIBRATION_END})."
            "If this behaviour is desired then set astro_params.PHOTONCONS_CALIBRATION_END"
            f"to a value lower than z = {np.amin(all_redshifts)}."
        )

    # Get all the perturb boxes early. We need to get the perturb at every
    # redshift.
    perturbed_field = []

    with _progressbar(disable=not progressbar) as _progbar:
        for z in _progbar.track(all_redshifts, description="Perturbing Matter Fields"):
            p = sf.perturb_field(
                redshift=z,
                inputs=inputs,
                write=write.perturbed_field,
                **kw,
            )

            if inputs.matter_options.MINIMIZE_MEMORY and write.perturbed_field:
                p.purge()
            perturbed_field.append(p)

    halofield_list = evolve_halos(
        inputs=inputs,
        all_redshifts=all_redshifts,
        write=write,
        progressbar=progressbar,
        **kw,
    )
    # Now we can purge initial_conditions further.
    if write.initial_conditions:
        initial_conditions.prepare_for_spin_temp()

    return (
        initial_conditions,
        perturbed_field,
        halofield_list,
        photon_nonconservation_data,
    )


def _get_required_redshifts_coeval(
    inputs: InputParameters, user_redshifts: Sequence
) -> list[float]:
    # Add in the redshift defined by the user, and sort in order
    # Turn into a set so that exact matching user-set redshift
    # don't double-up with scrolling ones.
    if (
        (inputs.astro_options.USE_TS_FLUCT or inputs.astro_options.INHOMO_RECO)
        and user_redshifts
        and min(inputs.node_redshifts) > min(user_redshifts)
    ):
        warnings.warn(
            f"minimum node redshift {min(inputs.node_redshifts)} is above output redshift {min(user_redshifts)},"
            + "This may result in strange evolution",
            stacklevel=2,
        )

    zmin_user = min(user_redshifts) if user_redshifts else 0
    needed_nodes = [z for z in inputs.node_redshifts if z > zmin_user]
    redshifts = np.concatenate((needed_nodes, user_redshifts))
    redshifts = np.sort(np.unique(redshifts))[::-1]
    return redshifts.tolist()
