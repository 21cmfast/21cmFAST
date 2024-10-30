"""Module containing a driver function for creating lightcones."""

import contextlib
import h5py
import logging
import numpy as np
import os
import warnings
from astropy import units
from astropy.cosmology import z_at_value
from collections import deque
from cosmotile import apply_rsds
from pathlib import Path
from typing import Sequence

from ..c_21cmfast import lib
from ..cache_tools import get_boxes_at_redshift
from ..lightcones import Lightconer, RectilinearLightconer
from ..wrapper.globals import global_params
from ..wrapper.inputs import AstroParams, CosmoParams, FlagOptions, UserParams
from ..wrapper.outputs import InitialConditions, PerturbedField
from ..wrapper.photoncons import _get_photon_nonconservation_data, setup_photon_cons
from . import single_field as sf
from .coeval import (
    Coeval,
    _get_coeval_callbacks,
    _HighLevelOutput,
    get_logspaced_redshifts,
)
from .param_config import InputParameters, _get_config_options
from .single_field import set_globals

logger = logging.getLogger(__name__)


class LightCone(_HighLevelOutput):
    """A full Lightcone with all associated evolved data."""

    def __init__(
        self,
        distances,
        inputs,
        lightcones,
        node_redshifts=None,
        global_quantities=None,
        photon_nonconservation_data=None,
        cache_files: dict | None = None,
        _globals=None,
        log10_mturnovers=None,
        log10_mturnovers_mini=None,
        current_redshift=None,
        current_index=None,
    ):
        self.random_seed = inputs.random_seed
        self.user_params = inputs.user_params
        self.cosmo_params = inputs.cosmo_params
        self.astro_params = inputs.astro_params
        self.flag_options = inputs.flag_options
        self.node_redshifts = node_redshifts
        self.cache_files = cache_files
        self.log10_mturnovers = log10_mturnovers
        self.log10_mturnovers_mini = log10_mturnovers_mini
        self._current_redshift = current_redshift
        self.lightcone_distances = distances

        if not hasattr(self.lightcone_distances, "unit"):
            self.lightcone_distances <<= units.Mpc

        # A *copy* of the current global parameters.
        self.global_params = _globals or dict(global_params.items())

        if global_quantities:
            for name, data in global_quantities.items():
                if name.endswith("_box"):
                    # Remove the _box because it looks dumb.
                    setattr(self, f"global_{name[:-4]}", data)
                else:
                    setattr(self, f"global_{name}", data)

        self.photon_nonconservation_data = photon_nonconservation_data

        for name, data in lightcones.items():
            setattr(self, name, data)

        # Hold a reference to the global/lightcones in a dict form for easy reference.
        self.global_quantities = global_quantities
        self.lightcones = lightcones
        self._current_index = current_index or self.shape[-1] - 1

    @property
    def global_xHI(self):
        """Global neutral fraction function."""
        warnings.warn(
            "global_xHI is deprecated. From now on, use global_xH. Will be removed in v3.1"
        )
        return self.global_xH

    @property
    def cell_size(self):
        """Cell size [Mpc] of the lightcone voxels."""
        return self.user_params.BOX_LEN / self.user_params.HII_DIM

    @property
    def lightcone_dimensions(self):
        """Lightcone size over each dimension -- tuple of (x,y,z) in Mpc."""
        return (
            self.user_params.BOX_LEN,
            self.user_params.BOX_LEN,
            self.n_slices * self.cell_size,
        )

    @property
    def shape(self):
        """Shape of the lightcone as a 3-tuple."""
        return self.lightcones[list(self.lightcones.keys())[0]].shape

    @property
    def n_slices(self):
        """Number of redshift slices in the lightcone."""
        return self.shape[-1]

    @property
    def lightcone_coords(self):
        """Co-ordinates [Mpc] of each slice along the redshift axis."""
        return self.lightcone_distances - self.lightcone_distances[0]

    @property
    def lightcone_redshifts(self):
        """Redshift of each cell along the redshift axis."""
        return np.array(
            [
                z_at_value(self.cosmo_params.cosmo.comoving_distance, d)
                for d in self.lightcone_distances
            ]
        )

    def _get_prefix(self):
        return "{name}_z{zmin:.4}-{zmax:.4}_{{hash}}_r{seed}.h5".format(
            name=self.__class__.__name__,
            zmin=float(self.lightcone_redshifts.min()),
            zmax=float(self.lightcone_redshifts.max()),
            seed=self.random_seed,
        )

    def _particular_rep(self):
        return (
            str(np.round(self.node_redshifts, 3))
            + str(self.global_quantities.keys())
            + str(self.lightcones.keys())
        )

    def _write_particulars(self, fname):
        with h5py.File(fname, "a") as f:
            # Save the boxes to the file
            boxes = f.create_group("lightcones")

            # Go through all fields in this struct, and save
            for k, val in self.lightcones.items():
                boxes[k] = val

            global_q = f.create_group("global_quantities")
            for k, v in self.global_quantities.items():
                global_q[k] = v

            f["node_redshifts"] = self.node_redshifts
            f["distances"] = self.lightcone_distances
            f["log10_mturnovers"] = self.log10_mturnovers
            f["log10_mturnovers_mini"] = self.log10_mturnovers_mini

    def make_checkpoint(self, fname, index: int, redshift: float):
        """Write updated lightcone data to file."""
        with h5py.File(fname, "a") as fl:
            current_index = fl.attrs.get("current_index", 0)

            for k, v in self.lightcones.items():
                fl["lightcones"][k][..., -index : v.shape[-1] - current_index] = v[
                    ..., -index : v.shape[-1] - current_index
                ]

            global_q = fl["global_quantities"]
            for k, v in self.global_quantities.items():
                global_q[k][-index : v.shape[-1] - current_index] = v[
                    -index : v.shape[-1] - current_index
                ]

            fl.attrs["current_index"] = index
            fl.attrs["current_redshift"] = redshift
            self._current_redshift = redshift
            self._current_index = index

    @classmethod
    def _read_inputs(cls, fname, safe=True):
        kwargs = {}
        parkw = {}
        with h5py.File(fname, "r") as fl:
            for k, kls in [
                ("user_params", UserParams),
                ("cosmo_params", CosmoParams),
                ("flag_options", FlagOptions),
                ("astro_params", AstroParams),
            ]:
                dct = dict(fl[k].attrs)
                parkw[k] = kls.from_subdict(dct, safe=safe)

            parkw["random_seed"] = fl.attrs["random_seed"]
            kwargs["inputs"] = InputParameters(**parkw)
            kwargs["current_redshift"] = fl.attrs.get("current_redshift", None)
            kwargs["current_index"] = fl.attrs.get("current_index", None)

        # Get the standard inputs.
        kw, glbls = _HighLevelOutput._read_inputs(fname, safe=safe)

        return {**kw, **kwargs}, glbls

    @classmethod
    def _read_particular(cls, fname, safe=True):
        kwargs = {}
        with h5py.File(fname, "r") as fl:
            boxes = fl["lightcones"]
            kwargs["lightcones"] = {k: boxes[k][...] for k in boxes.keys()}

            glb = fl["global_quantities"]
            kwargs["global_quantities"] = {k: glb[k][...] for k in glb.keys()}

            kwargs["node_redshifts"] = fl["node_redshifts"][...]
            kwargs["distances"] = fl["distances"][...]

            kwargs["log10_mturnovers"] = fl["log10_mturnovers"][...]
            kwargs["log10_mturnovers_mini"] = fl["log10_mturnovers_mini"][...]

        return kwargs

    def __eq__(self, other):
        """Determine if this is equal to another object."""
        return (
            isinstance(other, self.__class__)
            and other.random_seed == self.random_seed
            and np.all(
                np.isclose(
                    other.lightcone_redshifts, self.lightcone_redshifts, atol=1e-3
                )
            )
            and np.all(np.isclose(other.node_redshifts, self.node_redshifts, atol=1e-3))
            and self.user_params == other.user_params
            and self.cosmo_params == other.cosmo_params
            and self.flag_options == other.flag_options
            and self.astro_params == other.astro_params
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
                "Lightcone already contains brightness_temp_with_rsds, returning"
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
                -global_params.MAX_DVDR,
                global_params.MAX_DVDR,
                out=dvdx_on_h,
            )

            tb_with_rsds = self.brightness_temp / (1 + dvdx_on_h)
        else:
            gradient_component = 1 + dvdx_on_h  # not clipped!
            Tcmb = 2.728
            Trad = Tcmb * (1 + self.lightcone_redshifts)
            tb_with_rsds = np.where(
                gradient_component < 1e-7,
                1000.0 * (self.Ts_box - Trad) / (1.0 + self.lightcone_redshifts),
                (1.0 - np.exp(self.brightness_temp / gradient_component))
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
    inputs: InputParameters,
    scrollz: Sequence[float],
    global_quantities: Sequence[str],
    lightcone_filename: Path | None = None,
):
    """Returns a LightCone instance given a lightconer as input."""
    if lightcone_filename and Path(lightcone_filename).exists():
        lightcone = LightCone.read(lightcone_filename)
        start_idx = np.sum(
            scrollz >= lightcone._current_redshift
        )  # NOTE: assuming descending
        logger.info(f"Read in LC file at z={lightcone._current_redshift}")
        if start_idx < len(scrollz):
            logger.info(
                f"starting at z={scrollz[start_idx]}, step ({start_idx+1}/{len(scrollz)+1}"
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
            lightconer.lc_distances,
            inputs,
            lc,
            node_redshifts=scrollz,
            log10_mturnovers=np.zeros_like(scrollz),
            log10_mturnovers_mini=np.zeros_like(scrollz),
            global_quantities={
                quantity: np.zeros(len(scrollz)) for quantity in global_quantities
            },
            _globals=dict(global_params.items()),
        )
        start_idx = 0
    return lightcone, start_idx


@set_globals
def _run_lightcone_from_perturbed_fields(
    *,
    initial_conditions: InitialConditions,
    perturbed_fields: Sequence[PerturbedField],
    lightconer: Lightconer,
    astro_params: AstroParams,
    flag_options: FlagOptions,
    regenerate: bool | None = None,
    global_quantities: tuple[str] = ("brightness_temp", "xH_box"),
    direc: Path | str | None = None,
    cleanup: bool = True,
    hooks: dict | None = None,
    always_purge: bool = False,
    lightcone_filename: str | Path = None,
    **global_kwargs,
):
    r"""
    Evaluate a full lightcone ending at a given redshift.

    This is generally the easiest and most efficient way to generate a lightcone, though it can
    be done manually by using the lower-level functions which are called by this function.

    Parameters
    ----------
    lightconer : :class:`~Lightconer`
        This object specifies the dimensions, redshifts, and quantities required by the lightcone run
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options : :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature
        fluctuations are required.
    global_quantities : tuple of str, optional
        The quantities to save as globally-averaged redshift-dependent functions.
        These may be any of the quantities that can be used in ``lightcone_quantities``.
        The mean is taken over the full 3D cube at each redshift, rather than a 2D
        slice.
    initial_conditions : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be
        re-calculated.
    perturbed_fields : list of :class:`~PerturbedField`, optional
        If given, must be compatible with initial_conditions. It will merely negate the necessity of
        re-calculating the
        perturb fields. It will also be used to set the redshift if given.
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
    return_at_z
        If given, evaluation of the lightcone will be stopped at the given redshift,
        and the partial lightcone object will be returned. Lightcone evaluation can
        continue if the returned lightcone is saved to file, and this file is passed
        as `lightcone_filename`.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    lightcone : :class:`~py21cmfast.LightCone`
        The lightcone object.
    coeval_callback_output : list
        Only if coeval_callback in not None.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed
        See docs of :func:`initial_conditions` for more information.
    """
    direc = Path(direc)

    inputs = InputParameters.from_output_structs(
        (initial_conditions, *perturbed_fields),
        astro_params=astro_params,
        flag_options=flag_options,
        redshift=None,
    )

    lightconer.validate_options(inputs.user_params, inputs.flag_options)

    # Get the redshift through which we scroll and evaluate the ionization field.
    scrollz = np.array([pf.redshift for pf in perturbed_fields])
    if np.any(np.diff(scrollz) >= 0):
        raise ValueError(
            "The perturb fields must be ordered by redshift in descending order.\n"
            + f"redshifts: {scrollz}\n"
            + f"diffs: {np.diff(scrollz)}"
        )

    lcz = lightconer.lc_redshifts
    if not np.all(scrollz.min() * 0.99 < lcz) and np.all(lcz < scrollz.max() * 1.01):
        # We have a 1% tolerance on the redshifts, because the lightcone redshifts are
        # computed via inverse fitting the comoving_distance.
        raise ValueError(
            "The lightcone redshifts are not compatible with the given redshift."
            f"The range of computed redshifts is {scrollz.min()} to {scrollz.max()}, "
            f"while the lightcone redshift range is {lcz.min()} to {lcz.max()}."
        )

    if (
        inputs.flag_options.PHOTON_CONS_TYPE == "z-photoncons"
        and np.amin(scrollz) < global_params.PhotonConsEndCalibz
    ):
        raise ValueError(
            f"""
            You have passed a redshift (z = {np.amin(scrollz)}) that is lower than the
            endpoint of the photon non-conservation correction
            (global_params.PhotonConsEndCalibz = {global_params.PhotonConsEndCalibz}).
            If this behaviour is desired then set global_params.PhotonConsEndCalibz to a
            value lower than z = {np.amin(scrollz)}.
            """
        )

    iokw = {"hooks": hooks, "regenerate": regenerate, "direc": direc}

    # Create the LightCone instance, loading from file if needed
    lightcone, start_idx = setup_lightcone_instance(
        lightconer=lightconer,
        inputs=inputs,
        scrollz=scrollz,
        global_quantities=global_quantities,
        lightcone_filename=lightcone_filename,
    )
    if start_idx >= len(scrollz):
        logger.info(
            f"Lightcone already full at z={lightcone._current_redshift}. Returning."
        )
        # effectively adds one more iteration, but since start_idx > len(scrollz) it will be the only one
        yield None, None, None, lightcone

    # Remove anything in initial_conditions not required for spin_temp
    with contextlib.suppress(OSError):
        initial_conditions.prepare_for_spin_temp(
            flag_options=flag_options, force=always_purge
        )
    kw = {
        **{
            "initial_conditions": initial_conditions,
            "astro_params": inputs.astro_params,
            "flag_options": inputs.flag_options,
        },
        **iokw,
    }

    photon_nonconservation_data = None
    if inputs.flag_options.PHOTON_CONS_TYPE != "no-photoncons":
        setup_photon_cons(**kw)

    # At first we don't have any "previous" fields.
    st, ib, pf, hbox = None, None, None, None
    # optional fields which remain None if their flags are off
    hbox2, ph2, st2, xrs = None, None, None, None

    if (
        lightcone._current_redshift
        and not np.isclose(lightcone.node_redshifts.min(), lightcone._current_redshift)
        and not inputs.flag_options.USE_HALO_FIELD
    ):
        logger.info(
            f"Finding boxes at z={lightcone._current_redshift} with seed {lightcone.random_seed} and direc={direc}"
        )
        cached_boxes = get_boxes_at_redshift(
            redshift=lightcone._current_redshift,
            seed=lightcone.random_seed,
            direc=direc,
            user_params=inputs.user_params,
            cosmo_params=inputs.cosmo_params,
            flag_options=inputs.flag_options,
            astro_params=inputs.astro_params,
        )
        try:
            st = cached_boxes["TsBox"][0] if inputs.flag_options.USE_TS_FLUCT else None
            pf = cached_boxes["PerturbedField"][0]
            ib = cached_boxes["IonizedBox"][0]
        except (KeyError, IndexError):
            raise OSError(
                f"No component boxes found at z={lightcone._current_redshift} with "
                f"seed {lightcone.random_seed} and direc={direc}. You need to have "
                "run with write=True to continue from a checkpoint."
            )

    # Now we can purge initial_conditions further.
    with contextlib.suppress(OSError):
        initial_conditions.prepare_for_halos(
            flag_options=inputs.flag_options, force=always_purge
        )
    # we explicitly pass the descendant halos here since we have a redshift list prior
    #   this will generate the extra fields if STOC_MINIMUM_Z is given
    pt_halos = []
    if inputs.flag_options.USE_HALO_FIELD and not inputs.flag_options.FIXED_HALO_GRIDS:
        halos_desc = None
        for iz, z in enumerate(scrollz[::-1]):
            halo_field = sf.determine_halo_list(
                redshift=z,
                descendant_halos=halos_desc,
                **kw,
            )
            halos_desc = halo_field
            pt_halos += [sf.perturb_halo_list(halo_field=halo_field, **kw)]

            # we never want to store every halofield
            with contextlib.suppress(OSError):
                pt_halos[iz].purge(force=always_purge)
        # reverse the halo lists to be in line with the redshift lists
        pt_halos = pt_halos[::-1]

    # Now that we've got all the perturb fields, we can purge init more.
    with contextlib.suppress(OSError):
        initial_conditions.prepare_for_spin_temp(
            flag_options=inputs.flag_options, force=always_purge
        )

    # arrays to hold cache filenames
    perturb_files = []
    spin_temp_files = []
    ionize_files = []
    brightness_files = []
    hbox_files = []
    pth_files = []

    # saved global quantities which aren't lightcones
    log10_mturnovers = np.zeros(len(scrollz))
    log10_mturnovers_mini = np.zeros(len(scrollz))

    # structs which need to be kept beyond one snapshot
    hboxes = []

    # coeval objects to interpolate onto the lightcone
    coeval = None
    prev_coeval = None

    if lightcone_filename and not Path(lightcone_filename).exists():
        lightcone.save(lightcone_filename)

    # Iterate through redshift from top to bottom
    for iz, z in enumerate(scrollz):
        if iz < start_idx:
            continue
        logger.info(f"Computing Redshift {z} ({iz + 1}/{len(scrollz)}) iterations.")

        # Best to get a perturb for this redshift, to pass to brightness_temperature
        pf2 = perturbed_fields[iz]
        # This ensures that all the arrays that are required for spin_temp are there,
        # in case we dumped them from memory into file.
        pf2.load_all()
        if inputs.flag_options.USE_HALO_FIELD:
            if not inputs.flag_options.FIXED_HALO_GRIDS:
                ph2 = pt_halos[iz]
                ph2.load_all()

            hbox2 = sf.compute_halo_grid(
                perturbed_halo_list=ph2,
                previous_ionize_box=ib,
                previous_spin_temp=st,
                perturbed_field=pf2,
                **kw,
            )

            if inputs.flag_options.USE_TS_FLUCT:
                hboxes.append(hbox2)
                xrs = sf.compute_xray_source_field(
                    hboxes=hboxes,
                    **kw,
                )

        if inputs.flag_options.USE_TS_FLUCT:
            st2 = sf.spin_temperature(
                previous_spin_temp=st,
                perturbed_field=pf2,
                xray_source_box=xrs,
                cleanup=(cleanup and iz == (len(scrollz) - 1)),
                **kw,
            )

        ib2 = sf.compute_ionization_field(
            previous_ionized_box=ib,
            perturbed_field=pf2,
            previous_perturbed_field=pf,
            spin_temp=st2,
            halobox=hbox2,
            cleanup=(cleanup and iz == (len(scrollz) - 1)),
            **kw,
        )
        log10_mturnovers[iz] = ib2.log10_Mturnover_ave
        log10_mturnovers_mini[iz] = ib2.log10_Mturnover_MINI_ave

        bt2 = sf.brightness_temperature(
            ionized_box=ib2,
            perturbed_field=pf2,
            spin_temp=st2,
            **iokw,
        )

        coeval = Coeval(
            redshift=z,
            initial_conditions=initial_conditions,
            perturbed_field=pf2,
            ionized_box=ib2,
            brightness_temp=bt2,
            ts_box=st2,
            halobox=hbox2,
            photon_nonconservation_data=photon_nonconservation_data,
            _globals=None,
        )

        perturb_files.append((z, direc / pf2.filename))
        if (
            inputs.flag_options.USE_HALO_FIELD
            and not inputs.flag_options.FIXED_HALO_GRIDS
        ):
            hbox_files.append((z, direc / hbox2.filename))
            pth_files.append((z, direc / ph2.filename))
        if inputs.flag_options.USE_TS_FLUCT:
            spin_temp_files.append((z, direc / st2.filename))
        ionize_files.append((z, direc / ib2.filename))
        brightness_files.append((z, direc / bt2.filename))

        # Save mean/global quantities
        for quantity in global_quantities:
            lightcone.global_quantities[quantity][iz] = np.mean(
                getattr(coeval, quantity)
            )

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
                lightcone.make_checkpoint(
                    lightcone_filename, redshift=z, index=lc_index
                )

        # purge arrays we don't need
        if pf is not None:
            with contextlib.suppress(OSError):
                pf.purge(force=always_purge)
        if ph2 is not None:
            with contextlib.suppress(OSError):
                ph2.purge(force=always_purge)
        # we only need the SFR fields at previous redshifts for XraySourceBox
        if hbox is not None:
            with contextlib.suppress(OSError):
                hbox.prepare(
                    keep=[
                        "halo_sfr",
                        "halo_sfr_mini",
                        "halo_xray",
                        "log10_Mcrit_MCG_ave",
                    ],
                    force=always_purge,
                )

        # Save current ones as old ones.
        pf = pf2
        hbox = hbox2
        st = st2
        ib = ib2
        prev_coeval = coeval

        # last redshift things
        if iz == len(scrollz) - 1:
            if inputs.flag_options.PHOTON_CONS_TYPE == "z-photoncons":
                photon_nonconservation_data = _get_photon_nonconservation_data()

            if lib.photon_cons_allocated:
                lib.FreePhotonConsMemory()

            lightcone.photon_nonconservation_data = photon_nonconservation_data
            if isinstance(lightcone, AngularLightcone) and lightconer.get_los_velocity:
                lightcone.compute_rsds(
                    fname=lightcone_filename, n_subcells=inputs.astro_params.N_RSD_STEPS
                )

        # Append some info to the lightcone before we return
        lightcone.cache_files = {
            "init": [(0, direc / initial_conditions.filename)],
            "perturb_field": perturb_files,
            "ionized_box": ionize_files,
            "brightness_temp": brightness_files,
            "spin_temp": spin_temp_files,
            "halobox": hbox_files,
            "pt_halos": pth_files,
        }

        lightcone.log10_mturnovers = log10_mturnovers
        lightcone.log10_mturnovers_mini = log10_mturnovers_mini

        yield iz, z, coeval, lightcone


def run_lightcone(
    *,
    lightconer: Lightconer,
    node_redshifts: np.ndarray | None = None,
    user_params=None,
    cosmo_params=None,
    astro_params=None,
    flag_options=None,
    regenerate=None,
    write=None,
    global_quantities=("brightness_temp", "xH_box"),
    direc=None,
    initial_conditions: InitialConditions | None = None,
    perturbed_fields: Sequence[PerturbedField | None] = (None,),
    random_seed=None,
    cleanup=True,
    hooks=None,
    always_purge: bool = False,
    lightcone_filename: str | Path = None,
    **global_kwargs,
):
    r"""
    Create a generator function for a lightcone run.

    This is generally the easiest and most efficient way to generate a lightcone, though it can
    be done manually by using the lower-level functions which are called by this function.

    Parameters
    ----------
    lightconer : :class:`~Lightconer`
        This object specifies the dimensions, redshifts, and quantities required by the lightcone run
    node_redshifts : array_like, optional
        This array specifies the redshifts at which the simulation snapshots occur.
        By default it is evenly spaced in log(1+z) by a factor set by global_params.ZPRIME_STEP_FACTOR
    user_params : `~UserParams`, optional
        Defines the overall options and parameters of the run.
    astro_params : :class:`~AstroParams`, optional
        Defines the astrophysical parameters of the run.
    cosmo_params : :class:`~CosmoParams`, optional
        Defines the cosmological parameters used to compute initial conditions.
    flag_options : :class:`~FlagOptions`, optional
        Options concerning how the reionization process is run, eg. if spin temperature
        fluctuations are required.
    global_quantities : tuple of str, optional
        The quantities to save as globally-averaged redshift-dependent functions.
        These may be any of the quantities that can be used in ``Lightconer.quantities``.
        The mean is taken over the full 3D cube at each redshift, rather than a 2D
        slice.
    initial_conditions : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not be
        re-calculated.
    perturbed_fields : list of :class:`~PerturbedField`, optional
        If given, must be compatible with initial_conditions. It will merely negate the necessity of
        re-calculating the
        perturb fields. It will also be used to set the redshift if given.
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
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    lightcone : :class:`~py21cmfast.LightCone`
        The lightcone object.
    coeval_callback_output : list
        Only if coeval_callback in not None.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed, hooks
        See docs of :func:`initial_conditions` for more information.
    """
    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    pf_given = any(perturbed_fields)
    if pf_given and initial_conditions is None:
        raise ValueError(
            f"If perturbed_fields {perturbed_fields} are provided, initial_conditions {initial_conditions} must be provided"
        )

    # For the high-level, we need all the InputStruct initialised
    if cosmo_params is None and initial_conditions is None:
        cosmo_params = CosmoParams.from_astropy(lightconer.cosmo)

    cosmo_params = CosmoParams.new(cosmo_params)
    user_params = UserParams.new(user_params)
    flag_options = FlagOptions.new(flag_options)
    astro_params = AstroParams.new(astro_params, flag_options=flag_options)

    if pf_given:
        node_redshifts = [pf.redshift for pf in perturbed_fields]
    elif node_redshifts is None:
        node_redshifts = get_logspaced_redshifts(
            lightconer.lc_redshifts.min(),
            global_params.ZPRIME_STEP_FACTOR,
            max(lightconer.lc_redshifts.max(), global_params.Z_HEAT_MAX),
        )

    lcz = lightconer.lc_redshifts
    if not np.all(min(node_redshifts) * 0.99 < lcz) and np.all(
        lcz < max(node_redshifts) * 1.01
    ):
        # We have a 1% tolerance on the redshifts, because the lightcone redshifts are
        # computed via inverse fitting the comoving_distance.
        raise ValueError(
            "The lightcone redshifts are not compatible with the given redshift."
            f"The range of computed redshifts is {min(node_redshifts)} to {max(node_redshifts)}, "
            f"while the lightcone redshift range is {lcz.min()} to {lcz.max()}."
        )

    iokw = {"hooks": hooks, "regenerate": regenerate, "direc": direc}

    if initial_conditions is None:  # no need to get cosmo, user params out of it.
        initial_conditions = sf.compute_initial_conditions(
            user_params=user_params,
            cosmo_params=cosmo_params,
            random_seed=random_seed,
            **iokw,
        )

    # We can go ahead and purge some of the stuff in the initial_conditions, but only if
    # it is cached -- otherwise we could be losing information.
    try:
        # TODO: should really check that the file at path actually contains a fully
        # working copy of the initial_conditions.
        initial_conditions.prepare_for_perturb(
            flag_options=flag_options, force=always_purge
        )
    except OSError:
        pass

    if not pf_given:
        perturbed_fields = []
        for z in node_redshifts:
            p = sf.perturb_field(
                redshift=z, initial_conditions=initial_conditions, **iokw
            )
            if user_params.MINIMIZE_MEMORY:
                with contextlib.suppress(OSError):
                    p.purge(force=always_purge)
            perturbed_fields.append(p)

    yield from _run_lightcone_from_perturbed_fields(
        initial_conditions=initial_conditions,
        perturbed_fields=perturbed_fields,
        lightconer=lightconer,
        astro_params=astro_params,
        flag_options=flag_options,
        regenerate=regenerate,
        global_quantities=global_quantities,
        direc=direc,
        cleanup=cleanup,
        hooks=hooks,
        always_purge=always_purge,
        lightcone_filename=lightcone_filename,
        **global_kwargs,
    )


def exhaust_lightcone(**kwargs):
    """
    Convenience function to run through an entire lightcone.

    keywords passed are identical to run_lightcone.
    """
    lc_gen = run_lightcone(**kwargs)
    [[iz, z, coev, lc]] = deque(lc_gen, maxlen=1)

    return iz, z, coev, lc
