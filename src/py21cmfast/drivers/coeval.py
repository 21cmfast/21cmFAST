"""Compute simulations that evolve over redshift."""

import attrs
import contextlib
import h5py
import logging
import numpy as np
import os
import warnings
from hashlib import md5
from pathlib import Path
from typing import Any, Self, Sequence, get_args

from .. import __version__
from ..c_21cmfast import lib
from ..io import h5
from ..io.caching import CacheConfig, OutputCache
from ..wrapper.arrays import Array
from ..wrapper.globals import global_params
from ..wrapper.inputs import InputParameters
from ..wrapper.outputs import (
    BrightnessTemp,
    HaloBox,
    InitialConditions,
    IonizedBox,
    OutputStruct,
    PerturbedField,
    PerturbHaloField,
    TsBox,
)
from ..wrapper.photoncons import _get_photon_nonconservation_data, setup_photon_cons
from . import single_field as sf
from ._param_config import high_level_func

logger = logging.getLogger(__name__)


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
        Custom attribute getter for the Coeval class.

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

            if np.issubclass_(fld.type, OutputStruct):
                output_structs.append(fld.type)
            else:
                args = get_args(fld.type)
                for k in args:

                    if np.issubclass_(k, OutputStruct):
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

    def save(self, path: str | Path):
        """Save the Coeval object to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output_structs = self.output_structs
        for struct in output_structs.values():
            h5.write_output_to_hdf5(struct, path, mode="a")

        with h5py.File(path, "a") as fl:
            fl.attrs["coeval"] = True  # marker identifying this as a coeval box
            fl.attrs["__version__"] = __version__

            grp = fl.create_group("photon_nonconservation_data")
            for k, v in self.photon_nonconservation_data.items():
                grp[k] = v

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


def evolve_perturb_halos(
    inputs: InputParameters,
    all_redshifts: list[float],
    write: CacheConfig,
    initial_conditions: InitialConditions,
    cache: OutputCache,
    regenerate: bool,
    always_purge: bool = False,
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
    always_purge : bool, optional
        If True, always purge temporary data. Defaults to False.

    Returns
    -------
    list
        A list of perturbed halo fields for each redshift, in ascending redshift order.
        Returns an empty list if halo fields are not used or fixed grids are enabled.
    """
    # get the halos (reverse redshift order)
    if not inputs.flag_options.USE_HALO_FIELD or inputs.flag_options.FIXED_HALO_GRIDS:
        return []

    pt_halos = []
    kw = {
        "initial_conditions": initial_conditions,
        "cache": cache,
        "regenerate": regenerate,
    }
    halos_desc = None
    for i, z in enumerate(all_redshifts[::-1]):
        halos = sf.determine_halo_list(
            redshift=z,
            inputs=inputs,
            descendant_halos=halos_desc,
            write=write.halo_field,
            **kw,
        )

        pt_halos.append(
            sf.perturb_halo_list(
                halo_field=halos, write=write.perturbed_halo_field, **kw
            )
        )

        # we never want to store every halofield
        with contextlib.suppress(OSError):
            pt_halos[i].purge(force=always_purge)

        if z in inputs.node_redshifts:
            # Only evolve on the node_redshifts, not any redshifts in-between
            # that the user might care about.
            halos_desc = halos

    # reverse to get the right redshift order
    return pt_halos[::-1]


@high_level_func
def generate_coeval(
    *,
    inputs: InputParameters | None = None,
    out_redshifts: float | tuple[float] = (),
    regenerate: bool | None = None,
    write: CacheConfig = CacheConfig(),
    cache: OutputCache = OutputCache("."),
    initial_conditions: InitialConditions | None = None,
    perturbed_field: PerturbedField | None = None,
    cleanup: bool = True,
    always_purge: bool = False,
):
    r"""
    Perform a full coeval simulation of all fields at given redshifts.

    This is generally the easiest and most efficient way to generate a set of coeval cubes at a
    given set of redshifts. It self-consistently deals with situations in which the field needs to be
    evolved, and does this with the highest memory-efficiency, only returning the desired redshift.
    All other calculations are by default stored in the on-disk cache so they can be re-used at a
    later time.

    .. note:: User-supplied redshifts are *not* used as previous redshift in any scrolling,
              so that pristine log-sampling can be maintained.

    Parameters
    ----------
    inputs: :class:`~InputParameters`
        This object specifies the input parameters for the run, including the random seed
    out_redshifts: array_like, optional
        A single redshift, or multiple redshift, at which to return results. The minimum of these
        will define the log-scrolling behaviour (if necessary).
    initial_conditions : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not
        be re-calculated.
    perturbed_field : list of :class:`~PerturbedField`, optional
        If given, must be compatible with initial_conditions. It will merely negate the necessity
        of re-calculating the perturb fields.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculated has different shape, errors will occur
        if memory is not cleaned. Note that internally, this is set to False until the
        last iteration.

    Returns
    -------
    coevals : list of :class:`~py21cmfast.drivers.coeval.Coeval`
        The full data for the Coeval class, with init boxes, perturbed fields, ionized boxes,
        brightness temperature, and potential data from the conservation of photons. If a
        single redshift was specified, it will return such a class. If multiple redshifts
        were passed, it will return a list of such classes.
    """
    if isinstance(write, bool):
        write = CacheConfig() if write else CacheConfig.off()

    # Ensure perturb is a list of boxes, not just one.
    if perturbed_field is None:
        perturbed_field = ()
    elif not hasattr(perturbed_field, "__len__"):
        perturbed_field = (perturbed_field,)

    if not out_redshifts and not perturbed_field and not inputs.node_redshifts:
        raise ValueError(
            "Either out_redshifts or perturb must be given if inputs has no node redshifts"
        )

    iokw = {"regenerate": regenerate, "cache": cache}

    if initial_conditions is None:
        initial_conditions = sf.compute_initial_conditions(
            inputs=inputs, write=write.initial_conditions, **iokw
        )

    # We can go ahead and purge some of the stuff in the initial_conditions, but only if
    # it is cached -- otherwise we could be losing information.
    with contextlib.suppress(OSError):
        initial_conditions.prepare_for_perturb(
            flag_options=inputs.flag_options, force=always_purge
        )

    if out_redshifts is not None and not hasattr(out_redshifts, "__len__"):
        out_redshifts = [out_redshifts]

    if isinstance(out_redshifts, np.ndarray):
        out_redshifts = out_redshifts.tolist()
    if perturbed_field:
        if out_redshifts is not None and any(
            p.redshift != z for p, z in zip(perturbed_field, out_redshifts)
        ):
            raise ValueError(
                f"Input redshifts {out_redshifts} do not match "
                + f"perturb field redshifts {[p.redshift for p in perturbed_field]}"
            )
        else:
            out_redshifts = [p.redshift for p in perturbed_field]

    kw = {
        "initial_conditions": initial_conditions,
        **iokw,
    }
    photon_nonconservation_data = {}
    if inputs.flag_options.PHOTON_CONS_TYPE != "no-photoncons":
        photon_nonconservation_data = setup_photon_cons(**kw)

    # Get the list of redshifts we need to scroll through.
    all_redshifts = _get_required_redshifts_coeval(inputs, out_redshifts)

    # Get all the perturb boxes early. We need to get the perturb at every
    # redshift.
    pz = [p.redshift for p in perturbed_field]
    perturb_ = []
    for z in all_redshifts:
        p = (
            sf.perturb_field(redshift=z, write=write.perturbed_field, **kw)
            if z not in pz
            else perturbed_field[pz.index(z)]
        )

        if inputs.user_params.MINIMIZE_MEMORY:
            with contextlib.suppress(OSError):
                p.purge(force=always_purge)
        perturb_.append(p)

    perturbed_field = perturb_

    pt_halos = evolve_perturb_halos(
        inputs=inputs,
        all_redshifts=all_redshifts,
        write=write,
        always_purge=always_purge,
        **kw,
    )
    # Now we can purge initial_conditions further.
    with contextlib.suppress(OSError):
        initial_conditions.prepare_for_spin_temp(
            flag_options=inputs.flag_options, force=always_purge
        )

    if (
        inputs.flag_options.PHOTON_CONS_TYPE == "z-photoncons"
        and np.amin(all_redshifts) < global_params.PhotonConsEndCalibz
    ):
        raise ValueError(
            f"You have passed a redshift (z = {np.amin(all_redshifts)}) that is lower than"
            "the endpoint of the photon non-conservation correction"
            f"(global_params.PhotonConsEndCalibz = {global_params.PhotonConsEndCalibz})."
            "If this behaviour is desired then set global_params.PhotonConsEndCalibz"
            f"to a value lower than z = {np.amin(all_redshifts)}."
        )

    for coeval in _redshift_loop_generator(
        inputs=inputs,
        initial_conditions=initial_conditions,
        all_redshifts=all_redshifts,
        perturbed_field=perturbed_field,
        pt_halos=pt_halos,
        write=write,
        kw=kw,
        cleanup=cleanup,
        always_purge=always_purge,
        photon_nonconservation_data=photon_nonconservation_data,
        iokw=iokw,
    ):
        yield coeval, coeval.redshift in inputs.node_redshifts

    if lib.photon_cons_allocated:
        lib.FreePhotonConsMemory()


def run_coeval(**kwargs) -> list[Coeval]:  # noqa: D103
    return [coeval for coeval, in_nodes in generate_coeval(**kwargs) if in_nodes]


run_coeval.__doc__ = generate_coeval.__doc__


def _redshift_loop_generator(
    inputs: InputParameters,
    initial_conditions: InitialConditions,
    all_redshifts: Sequence[float],
    perturbed_field: list[PerturbedField],
    pt_halos: list[PerturbHaloField],
    write: CacheConfig,
    kw: dict,
    iokw: dict,
    cleanup: bool,
    always_purge: bool,
    photon_nonconservation_data: dict,
    start_idx: int = 0,
    st: TsBox | None = None,
    ib: IonizedBox | None = None,
    hb=None,
):
    if isinstance(write, bool):
        write = CacheConfig()

    # Iterate through redshift from top to bottom
    hbox_arr = []
    hb2 = None
    st2 = None
    ph2 = None
    xrs = None
    pf = None

    for iz, z in enumerate(all_redshifts):
        if iz > 0:
            pf = perturbed_field[iz - 1]

        if iz < start_idx:
            continue

        logger.info(
            f"Computing Redshift {z} ({iz + 1}/{len(all_redshifts)}) iterations."
        )
        pf2 = perturbed_field[iz]
        pf2.load_all()

        if inputs.flag_options.USE_HALO_FIELD:
            if not inputs.flag_options.FIXED_HALO_GRIDS:
                ph2 = pt_halos[iz]

            hb2 = sf.compute_halo_grid(
                perturbed_halo_list=ph2,
                perturbed_field=pf2,
                previous_ionize_box=ib,
                previous_spin_temp=st,
                write=write.halobox,
                **kw,
            )

        if inputs.flag_options.USE_TS_FLUCT:
            # append the halo redshift array so we have all halo boxes [z,zmax]
            hbox_arr += [hb2]
            if inputs.flag_options.USE_HALO_FIELD:
                xrs = sf.compute_xray_source_field(
                    hboxes=hbox_arr,
                    write=write.xray_source_box,
                    **kw,
                )

            st2 = sf.compute_spin_temperature(
                previous_spin_temp=st,
                perturbed_field=pf2,
                xray_source_box=xrs,
                write=write.spin_temp,
                **kw,
                cleanup=(cleanup and z == all_redshifts[-1]),
            )

        ib2 = sf.compute_ionization_field(
            previous_ionized_box=ib,
            perturbed_field=pf2,
            # perturb field *not* interpolated here.
            previous_perturbed_field=pf,
            halobox=hb2,
            spin_temp=st2,
            write=write.ionized_box,
            **kw,
        )

        if pf is not None:
            with contextlib.suppress(OSError):
                pf.purge(force=always_purge)
        if ph2 is not None:
            with contextlib.suppress(OSError):
                ph2.purge(force=always_purge)
        # we only need the SFR fields at previous redshifts for XraySourceBox
        if hb is not None:
            with contextlib.suppress(OSError):
                hb.prepare(
                    keep=[
                        "halo_sfr",
                        "halo_sfr_mini",
                        "halo_xray",
                        "log10_Mcrit_MCG_ave",
                    ],
                    force=always_purge,
                )

        logger.debug(f"PID={os.getpid()} doing brightness temp for z={z}")
        _bt = sf.brightness_temperature(
            ionized_box=ib2,
            perturbed_field=pf2,
            spin_temp=st2,
            write=write.brightness_temp,
            **iokw,
        )

        if inputs.flag_options.PHOTON_CONS_TYPE == "z-photoncons":
            # Updated info at each z.
            photon_nonconservation_data = _get_photon_nonconservation_data()

        yield Coeval(
            initial_conditions=initial_conditions,
            perturbed_field=pf2,
            ionized_box=ib2,
            brightness_temperature=_bt,
            ts_box=st2,
            halobox=hb,
            photon_nonconservation_data=photon_nonconservation_data,
        )

        if z in inputs.node_redshifts:
            # Only evolve on the node_redshifts, not any redshifts in-between
            # that the user might care about.
            ib = ib2
            pf = pf2
            _bt = None
            hb = hb2
            st = st2


def _get_required_redshifts_coeval(
    inputs: InputParameters, user_redshifts: Sequence
) -> list[float]:
    # Add in the redshift defined by the user, and sort in order
    # Turn into a set so that exact matching user-set redshift
    # don't double-up with scrolling ones.
    if (
        (inputs.flag_options.USE_TS_FLUCT or inputs.flag_options.INHOMO_RECO)
        and user_redshifts
        and min(inputs.node_redshifts) > min(user_redshifts)
    ):
        warnings.warn(
            f"minimum node redshift {inputs.node_redshifts.min()} is above output redshift {min(user_redshifts)},"
            + "This may result in strange evolution"
        )

    zmin_user = min(user_redshifts) if user_redshifts else 0
    needed_nodes = [z for z in inputs.node_redshifts if z > zmin_user]
    redshifts = np.concatenate((needed_nodes, user_redshifts))
    redshifts = np.sort(np.unique(redshifts))[::-1]
    return redshifts.tolist()
