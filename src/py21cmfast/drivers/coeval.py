"""Compute simulations that evolve over redshift."""

import contextlib
import h5py
import logging
import numpy as np
import os
import warnings
from hashlib import md5
from pathlib import Path
from typing import Any, Sequence

from .. import __version__
from .._cfg import config
from ..c_21cmfast import lib
from ..wrapper._utils import camel_to_snake
from ..wrapper.globals import global_params
from ..wrapper.inputs import AstroParams, CosmoParams, FlagOptions, UserParams
from ..wrapper.outputs import (
    BrightnessTemp,
    HaloBox,
    InitialConditions,
    IonizedBox,
    PerturbedField,
    TsBox,
    _OutputStruct,
)
from ..wrapper.photoncons import _get_photon_nonconservation_data, setup_photon_cons
from . import single_field as sf
from .param_config import (
    InputParameters,
    _get_config_options,
    check_redshift_consistency,
)
from .single_field import set_globals

logger = logging.getLogger(__name__)


class _HighLevelOutput:
    def get_cached_data(
        self, kind: str, redshift: float, load_data: bool = False
    ) -> _OutputStruct:
        """
        Return an OutputStruct object which was cached in creating this Coeval box.

        Parameters
        ----------
        kind
            The kind of object: "init", "perturb", "spin_temp", "ionize" or "brightness"
        redshift
            The (approximate) redshift of the object to return.
        load_data
            Whether to actually read the field data of the object in (call ``obj.read()``
            after this function to do this manually)

        Returns
        -------
        output
            The output struct object.
        """
        if self.cache_files is None:
            raise AttributeError(
                "No cache files were associated with this Coeval object."
            )

        # TODO: also check this file, because it may have been "gather"d.

        if kind not in self.cache_files:
            raise ValueError(
                f"{kind} is not a valid kind for the cache. Valid options: "
                f"{self.cache_files.keys()}"
            )

        files = self.cache_files.get(kind, {})
        # files is a list of tuples of (redshift, filename)

        redshifts = np.array([f[0] for f in files])

        indx = np.argmin(np.abs(redshifts - redshift))
        fname = files[indx][1]

        if not os.path.exists(fname):
            raise OSError(
                "The cached file you requested does not exist (maybe it was removed?)."
            )

        kinds = {
            "init": InitialConditions,
            "perturb_field": PerturbedField,
            "halobox": HaloBox,
            "ionized_box": IonizedBox,
            "spin_temp": TsBox,
            "brightness_temp": BrightnessTemp,
        }
        cls = kinds[kind]

        return cls.from_file(fname, load_data=load_data)

    def gather(
        self,
        fname: str | None | Path = None,
        kinds: Sequence | None = None,
        clean: bool | dict = False,
        direc: str | Path | None = None,
    ) -> Path:
        """Gather the cached data associated with this object into its file."""
        kinds = kinds or [
            "init",
            "perturb_field",
            "halobox",
            "ionized_box",
            "spin_temp",
            "brightness_temp",
        ]

        clean = kinds if clean and not hasattr(clean, "__len__") else clean or []
        if any(c not in kinds for c in clean):
            raise ValueError(
                "You are trying to clean cached items that you will not be gathering."
            )

        direc = Path(direc or config["direc"]).expanduser().absolute()
        fname = Path(fname or self.get_unique_filename()).expanduser()

        if not fname.exists():
            fname = direc / fname

        for kind in kinds:
            redshifts = (f[0] for f in self.cache_files[kind])
            for i, z in enumerate(redshifts):
                cache_fname = self.cache_files[kind][i][1]

                obj = self.get_cached_data(kind, redshift=z, load_data=True)
                with h5py.File(fname, "a") as fl:
                    cache = (
                        fl.create_group("cache") if "cache" not in fl else fl["cache"]
                    )
                    kind_group = (
                        cache.create_group(kind) if kind not in cache else cache[kind]
                    )

                    zstr = f"z{z:.2f}"
                    if zstr not in kind_group:
                        z_group = kind_group.create_group(zstr)
                    else:
                        z_group = kind_group[zstr]

                    obj.write_data_to_hdf5_group(z_group)

                    if kind in clean:
                        os.remove(cache_fname)
        return fname

    def _get_prefix(self):
        pass

    def _input_rep(self):
        return "".join(
            repr(getattr(self, inp))
            for inp in [
                "user_params",
                "cosmo_params",
                "astro_params",
                "flag_options",
                "global_params",
            ]
        )

    def get_unique_filename(self):
        """Generate a unique hash filename for this instance."""
        return self._get_prefix().format(
            hash=md5((self._input_rep() + self._particular_rep()).encode()).hexdigest()
        )

    def _write(self, direc=None, fname=None, clobber=False):
        """
        Write the high level output to file in standard HDF5 format.

        This method is primarily meant for the automatic caching. Its default
        filename is a hash generated based on the input data, and the directory is
        the configured caching directory.

        Parameters
        ----------
        direc : str, optional
            The directory into which to write the file. Default is the configuration
            directory.
        fname : str, optional
            The filename to write, default a unique name produced by the inputs.
        clobber : bool, optional
            Whether to overwrite existing file.

        Returns
        -------
        fname : str
            The absolute path to which the file was written.
        """
        direc = os.path.expanduser(direc or config["direc"])

        if fname is None:
            fname = self.get_unique_filename()

        if not os.path.isabs(fname):
            fname = os.path.abspath(os.path.join(direc, fname))

        if not clobber and os.path.exists(fname):
            raise FileExistsError(
                f"The file {fname} already exists. If you want to overwrite, set clobber=True."
            )

        with h5py.File(fname, "w") as f:
            # Save input parameters as attributes
            for k in [
                "user_params",
                "cosmo_params",
                "flag_options",
                "astro_params",
                "global_params",
            ]:
                q = getattr(self, k)
                kfile = "_globals" if k == "global_params" else k
                grp = f.create_group(kfile)

                try:
                    dct = q.asdict()
                except AttributeError:
                    dct = q

                for kk, v in dct.items():
                    if v is None:
                        continue
                    with contextlib.suppress(TypeError):
                        grp.attrs[kk] = v
            if self.photon_nonconservation_data is not None:
                photon_data = f.create_group("photon_nonconservation_data")
                for k, val in self.photon_nonconservation_data.items():
                    photon_data[k] = val

            f.attrs["random_seed"] = self.random_seed
            f.attrs["version"] = __version__

        self._write_particulars(fname)

        return fname

    def _write_particulars(self, fname):
        pass

    def save(self, fname=None, direc=".", clobber: bool = False):
        """Save to disk.

        This function has defaults that make it easy to save a unique box to
        the current directory.

        Parameters
        ----------
        fname : str, optional
            The filename to write, default a unique name produced by the inputs.
        direc : str, optional
            The directory into which to write the file. Default is the current directory.

        Returns
        -------
        str :
            The filename to which the box was written.
        """
        return self._write(direc=direc, fname=fname, clobber=clobber)

    @classmethod
    def _read_inputs(cls, fname, safe=True):
        kwargs = {}
        with h5py.File(fname, "r") as fl:
            global_req_keys = [
                k for k, v in global_params.items() if "path" not in k and v is not None
            ]
            glbls = dict(fl["_globals"].attrs)
            if set(glbls.keys()) != set(global_req_keys):
                missing_items = [
                    (k, v)
                    for k, v in global_params.items()
                    if k not in glbls.keys() and k in global_req_keys
                ]
                extra_items = [
                    (k, v) for k, v in glbls.items() if k not in global_params.keys()
                ]
                message = (
                    f"There are extra or missing global params in the file to be read.\n"
                    f"EXTRAS: {extra_items}\n"
                    f"MISSING: {missing_items}\n"
                )
                # we don't save None values (we probably should) or paths so ignore these
                # We also only print the warning for these fields if "safe" is turned off
                if safe:
                    raise ValueError(
                        message
                        + "set `safe=False` to load structures from previous versions"
                    )
                else:
                    warnings.warn(
                        message
                        + "\nExtras are ignored and missing are set to default (shown) values"
                    )

            if "photon_nonconservation_data" in fl.keys():
                d = fl["photon_nonconservation_data"]
                kwargs["photon_nonconservation_data"] = {k: d[k][...] for k in d.keys()}

        return kwargs, glbls

    @classmethod
    def read(cls, fname, direc=".", safe=True):
        """Read the HighLevelOutput file from disk, creating a LightCone or Coeval object.

        Parameters
        ----------
        fname : str
            The filename path. Can be absolute or relative.
        direc : str
            If fname, is relative, the directory in which to find the file. By default,
            both the current directory and default cache and the  will be searched, in
            that order.
        safe : bool
            If safe is true, we throw an error if the parameter structures in the file do not
            match the structures in the `inputs.py` module. If false, we allow extra and missing
            items, setting the missing items to the default values and ignoring extra items.

        Returns
        -------
        LightCone :
            A :class:`LightCone` instance created from the file's data.
        """
        if not os.path.isabs(fname):
            fname = os.path.abspath(os.path.join(direc, fname))

        if not os.path.exists(fname):
            raise FileExistsError(f"The file {fname} does not exist!")

        park, glbls = cls._read_inputs(fname, safe=safe)
        boxk = cls._read_particular(fname, safe=safe)

        with global_params.use(**glbls):
            out = cls(**park, **boxk)

        return out

    def _read_particular(self, fname, safe=True):
        pass


class Coeval(_HighLevelOutput):
    """A full coeval box with all associated data."""

    def __init__(
        self,
        redshift: float,
        initial_conditions: InitialConditions,
        perturbed_field: PerturbedField,
        ionized_box: IonizedBox,
        brightness_temp: BrightnessTemp,
        ts_box: TsBox | None = None,
        halobox: HaloBox | None = None,
        cache_files: dict | None = None,
        photon_nonconservation_data=None,
        _globals=None,
    ):

        # Check that all the fields have the same input parameters.
        # TODO: use this instead of all the parameter methods
        input_struct = InputParameters.from_output_structs(
            (
                initial_conditions,
                perturbed_field,
                halobox,
                ionized_box,
                brightness_temp,
                ts_box,
            ),
            redshift=redshift,
        )
        check_redshift_consistency(
            input_struct,
            (
                perturbed_field,
                halobox,
                ionized_box,
                brightness_temp,
                ts_box,
            ),
        )

        self.redshift = redshift
        self.init_struct = initial_conditions
        self.perturb_struct = perturbed_field
        self.ionization_struct = ionized_box
        self.brightness_temp_struct = brightness_temp
        self.halobox_struct = halobox
        self.spin_temp_struct = ts_box

        self.cache_files = cache_files

        self.photon_nonconservation_data = photon_nonconservation_data
        # A *copy* of the current global parameters.
        self.global_params = _globals or dict(global_params.items())

        # Expose all the fields of the structs to the surface of the Coeval object
        for box in [
            initial_conditions,
            perturbed_field,
            halobox,
            ionized_box,
            brightness_temp,
            ts_box,
        ]:
            if box is None:
                continue
            for field in box._get_box_structures():
                setattr(self, field, getattr(box, field))

        # For backwards compatibility
        if hasattr(self, "velocity_z"):
            self.velocity = self.velocity_z

    @classmethod
    def get_fields(cls, spin_temp: bool = True, hbox: bool = True) -> list[str]:
        """Obtain a list of name of simulation boxes saved in the Coeval object."""
        pointer_fields = []
        for cls in [InitialConditions, PerturbedField, IonizedBox, BrightnessTemp]:
            pointer_fields += cls.get_pointer_fields()

        if spin_temp:
            pointer_fields += TsBox.get_pointer_fields()

        if hbox:
            pointer_fields += HaloBox.get_pointer_fields()

        return pointer_fields

    @property
    def user_params(self):
        """User params shared by all datasets."""
        return self.brightness_temp_struct.user_params

    @property
    def cosmo_params(self):
        """Cosmo params shared by all datasets."""
        return self.brightness_temp_struct.cosmo_params

    @property
    def flag_options(self):
        """Flag Options shared by all datasets."""
        return self.brightness_temp_struct.flag_options

    @property
    def astro_params(self):
        """Astro params shared by all datasets."""
        return self.brightness_temp_struct.astro_params

    @property
    def random_seed(self):
        """Random seed shared by all datasets."""
        return self.brightness_temp_struct.random_seed

    def _get_prefix(self):
        return "{name}_z{redshift:.4}_{{hash}}_r{seed}.h5".format(
            name=self.__class__.__name__,
            redshift=float(self.redshift),
            seed=self.random_seed,
        )

    def _particular_rep(self):
        return ""

    def _write_particulars(self, fname):
        for name in ["init", "perturb", "ionization", "brightness_temp", "spin_temp"]:
            struct = getattr(self, f"{name}_struct")
            if struct is not None:
                struct.write(fname=fname, write_inputs=False)

                # Also write any other inputs to any of the constituent boxes
                # to the overarching attrs.
                with h5py.File(fname, "a") as fl:
                    for inp in struct._inputs:
                        if inp not in fl.attrs and inp not in [
                            "user_params",
                            "cosmo_params",
                            "flag_options",
                            "astro_params",
                            "global_params",
                        ]:
                            fl.attrs[inp] = getattr(struct, inp)

    @classmethod
    def _read_particular(cls, fname, safe=True):
        kwargs = {}

        with h5py.File(fname, "r") as fl:
            kwargs["redshift"] = float(fl.attrs["redshift"])
            for output_class in _OutputStruct._implementations():
                if output_class.__name__ in fl:
                    kwargs[camel_to_snake(output_class.__name__)] = (
                        output_class.from_file(fname, safe=safe)
                    )

        return kwargs

    def __eq__(self, other):
        """Determine if this is equal to another object."""
        return (
            isinstance(other, self.__class__)
            and other.random_seed == self.random_seed
            and other.redshift == self.redshift
            and self.user_params == other.user_params
            and self.cosmo_params == other.cosmo_params
            and self.flag_options == other.flag_options
            and self.astro_params == other.astro_params
        )


def get_logspaced_redshifts(min_redshift: float, z_step_factor: float, zmax: float):
    """Compute a sequence of redshifts to evolve over that are log-spaced."""
    redshifts = [min_redshift]
    while redshifts[-1] < zmax:
        redshifts.append((redshifts[-1] + 1.0) * z_step_factor - 1.0)

    return np.array(redshifts)[::-1]


@set_globals
def run_coeval(
    *,
    out_redshifts: float | np.ndarray | None = None,
    inputs: InputParameters | str | None = None,
    regenerate: bool | None = None,
    write: bool | None = None,
    direc: str | Path | None = None,
    initial_conditions: InitialConditions | None = None,
    perturbed_field: PerturbedField | None = None,
    random_seed: int | None = None,
    cleanup: bool = True,
    hooks: dict[callable, dict[str, Any]] | None = None,
    always_purge: bool = False,
    **global_kwargs,
):
    r"""
    Evaluate a coeval ionized box at a given redshift, or multiple redshifts.

    This is generally the easiest and most efficient way to generate a set of coeval cubes at a
    given set of redshift. It self-consistently deals with situations in which the field needs to be
    evolved, and does this with the highest memory-efficiency, only returning the desired redshift.
    All other calculations are by default stored in the on-disk cache so they can be re-used at a
    later time.

    .. note:: User-supplied redshift are *not* used as previous redshift in any scrolling,
              so that pristine log-sampling can be maintained.

    Parameters
    ----------
    out_redshifts: array_like
        A single redshift, or multiple redshift, at which to return results. The minimum of these
        will define the log-scrolling behaviour (if necessary).
    user_params : :class:`~inputs.UserParams`, optional
        Defines the overall options and parameters of the run.
    cosmo_params : :class:`~inputs.CosmoParams` , optional
        Defines the cosmological parameters used to compute initial conditions.
    astro_params : :class:`~inputs.AstroParams` , optional
        The astrophysical parameters defining the course of reionization.
    flag_options : :class:`~inputs.FlagOptions` , optional
        Some options passed to the reionization routine.
    initial_conditions : :class:`~InitialConditions`, optional
        If given, the user and cosmo params will be set from this object, and it will not
        be re-calculated.
    perturbed_field : list of :class:`~PerturbedField`, optional
        If given, must be compatible with initial_conditions. It will merely negate the necessity
        of re-calculating the perturb fields.
    cleanup : bool, optional
        A flag to specify whether the C routine cleans up its memory before returning.
        Typically, if `spin_temperature` is called directly, you will want this to be
        true, as if the next box to be calculate has different shape, errors will occur
        if memory is not cleaned. Note that internally, this is set to False until the
        last iteration.
    \*\*global_kwargs :
        Any attributes for :class:`~py21cmfast.inputs.GlobalParams`. This will
        *temporarily* set global attributes for the duration of the function. Note that
        arguments will be treated as case-insensitive.

    Returns
    -------
    coevals : :class:`~py21cmfast.outputs.Coeval`
        The full data for the Coeval class, with init boxes, perturbed fields, ionized boxes,
        brightness temperature, and potential data from the conservation of photons. If a
        single redshift was specified, it will return such a class. If multiple redshifts
        were passed, it will return a list of such classes.

    Other Parameters
    ----------------
    regenerate, write, direc, random_seed :
        See docs of :func:`initial_conditions` for more information.
    """
    if out_redshifts is None and perturbed_field is None:
        raise ValueError("Either out_redshifts or perturb must be given")

    direc, regenerate, hooks = _get_config_options(direc, regenerate, write, hooks)

    singleton = False
    # Ensure perturb is a list of boxes, not just one.
    if perturbed_field is None:
        perturbed_field = ()
    elif not hasattr(perturbed_field, "__len__"):
        perturbed_field = (perturbed_field,)
        singleton = True

    random_seed = (
        initial_conditions.random_seed
        if initial_conditions is not None
        else random_seed
    )

    if isinstance(inputs, str):
        inputs = InputParameters.from_template(inputs, seed=random_seed)
    elif inputs is None:
        inputs = InputParameters.from_defaults(seed=random_seed)

    inputs = InputParameters.from_output_structs(
        (initial_conditions,) + perturbed_field,
        cosmo_params=inputs.cosmo_params,
        user_params=inputs.user_params,
        astro_params=inputs.astro_params,
        flag_options=inputs.flag_options,
    )

    random_seed = inputs.random_seed

    iokw = {"regenerate": regenerate, "hooks": hooks, "direc": direc}

    if initial_conditions is None:
        initial_conditions = sf.compute_initial_conditions(
            user_params=inputs.user_params,
            cosmo_params=inputs.cosmo_params,
            random_seed=random_seed,
            **iokw,
        )

    # We can go ahead and purge some of the stuff in the initial_conditions, but only if
    # it is cached -- otherwise we could be losing information.
    with contextlib.suppress(OSError):
        initial_conditions.prepare_for_perturb(
            flag_options=inputs.flag_options, force=always_purge
        )
    if perturbed_field:
        if out_redshifts is not None and any(
            p.redshift != z for p, z in zip(perturbed_field, out_redshifts)
        ):
            raise ValueError("Input redshifts do not match perturb field redshifts")
        else:
            out_redshifts = [p.redshift for p in perturbed_field]

    kw = {
        **{
            "astro_params": inputs.astro_params,
            "flag_options": inputs.flag_options,
            "initial_conditions": initial_conditions,
        },
        **iokw,
    }
    photon_nonconservation_data = None
    if inputs.flag_options.PHOTON_CONS_TYPE != "no-photoncons":
        photon_nonconservation_data = setup_photon_cons(**kw)

    if not hasattr(out_redshifts, "__len__"):
        singleton = True
        out_redshifts = [out_redshifts]

    if isinstance(out_redshifts, np.ndarray):
        out_redshifts = out_redshifts.tolist()

    # Get the list of redshift we need to scroll through.
    node_redshifts = _get_required_redshifts_coeval(inputs.flag_options, out_redshifts)

    # Get all the perturb boxes early. We need to get the perturb at every
    # redshift.
    pz = [p.redshift for p in perturbed_field]
    perturb_ = []
    for z in node_redshifts:
        p = (
            sf.perturb_field(redshift=z, initial_conditions=initial_conditions, **iokw)
            if z not in pz
            else perturbed_field[pz.index(z)]
        )

        if inputs.user_params.MINIMIZE_MEMORY:
            with contextlib.suppress(OSError):
                p.purge(force=always_purge)
        perturb_.append(p)

    perturbed_field = perturb_

    # get the halos (reverse redshift order)
    pt_halos = []
    if inputs.flag_options.USE_HALO_FIELD and not inputs.flag_options.FIXED_HALO_GRIDS:
        halos_desc = None
        for i, z in enumerate(node_redshifts[::-1]):
            halos = sf.determine_halo_list(
                redshift=z, descendant_halos=halos_desc, **kw
            )
            pt_halos += [sf.perturb_halo_list(halo_field=halos, **kw)]

            # we never want to store every halofield
            with contextlib.suppress(OSError):
                pt_halos[i].purge(force=always_purge)
            halos_desc = halos

        # reverse to get the right redshift order
        pt_halos = pt_halos[::-1]

    # Now we can purge initial_conditions further.
    with contextlib.suppress(OSError):
        initial_conditions.prepare_for_spin_temp(
            flag_options=inputs.flag_options, force=always_purge
        )
    if (
        inputs.flag_options.PHOTON_CONS_TYPE == "z-photoncons"
        and np.amin(node_redshifts) < global_params.PhotonConsEndCalibz
    ):
        raise ValueError(
            f"You have passed a redshift (z = {np.amin(node_redshifts)}) that is lower than"
            "the endpoint of the photon non-conservation correction"
            f"(global_params.PhotonConsEndCalibz = {global_params.PhotonConsEndCalibz})."
            "If this behaviour is desired then set global_params.PhotonConsEndCalibz"
            f"to a value lower than z = {np.amin(node_redshifts)}."
        )

    ib_tracker = [0] * len(out_redshifts)
    bt = [0] * len(out_redshifts)
    # At first we don't have any "previous" st or ib.
    st, ib, pf, hb = None, None, None, None
    # optional fields which remain None if their flags are off
    hb2, ph2, st2, xrs = None, None, None, None

    hb_tracker = [None] * len(out_redshifts)
    st_tracker = [None] * len(out_redshifts)

    spin_temp_files = []
    hbox_files = []
    perturb_files = []
    ionize_files = []
    brightness_files = []
    pth_files = []

    # Iterate through redshift from top to bottom
    hbox_arr = []
    for iz, z in enumerate(node_redshifts):
        logger.info(
            f"Computing Redshift {z} ({iz + 1}/{len(node_redshifts)}) iterations."
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
                **kw,
            )

        if inputs.flag_options.USE_TS_FLUCT:
            # append the halo redshift array so we have all halo boxes [z,zmax]
            hbox_arr += [hb2]
            if inputs.flag_options.USE_HALO_FIELD:
                xrs = sf.compute_xray_source_field(
                    hboxes=hbox_arr,
                    **kw,
                )

            st2 = sf.spin_temperature(
                previous_spin_temp=st,
                perturbed_field=pf2,
                xray_source_box=xrs,
                **kw,
                cleanup=(cleanup and z == node_redshifts[-1]),
            )

        ib2 = sf.compute_ionization_field(
            previous_ionized_box=ib,
            perturbed_field=pf2,
            # perturb field *not* interpolated here.
            previous_perturbed_field=pf,
            halobox=hb2,
            spin_temp=st2,
            z_heat_max=global_params.Z_HEAT_MAX,
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
        if z in out_redshifts:
            logger.debug(f"PID={os.getpid()} doing brightness temp for z={z}")
            ib_tracker[out_redshifts.index(z)] = ib2
            st_tracker[out_redshifts.index(z)] = st2
            hb_tracker[out_redshifts.index(z)] = hb2

            _bt = sf.brightness_temperature(
                ionized_box=ib2,
                perturbed_field=pf2,
                spin_temp=st2,
                **iokw,
            )

            bt[out_redshifts.index(z)] = _bt
        else:
            ib = ib2
            pf = pf2
            _bt = None
            hb = hb2
            st = st2

        perturb_files.append((z, os.path.join(direc, pf2.filename)))
        if inputs.flag_options.USE_HALO_FIELD:
            hbox_files.append((z, os.path.join(direc, hb2.filename)))
            pth_files.append((z, os.path.join(direc, ph2.filename)))
        if inputs.flag_options.USE_TS_FLUCT:
            spin_temp_files.append((z, os.path.join(direc, st2.filename)))
        ionize_files.append((z, os.path.join(direc, ib2.filename)))

        if _bt is not None:
            brightness_files.append((z, os.path.join(direc, _bt.filename)))

    if inputs.flag_options.PHOTON_CONS_TYPE == "z-photoncons":
        photon_nonconservation_data = _get_photon_nonconservation_data()

    if lib.photon_cons_allocated:
        lib.FreePhotonConsMemory()

    coevals = [
        Coeval(
            redshift=z,
            initial_conditions=initial_conditions,
            perturbed_field=perturbed_field[node_redshifts.index(z)],
            ionized_box=ib,
            brightness_temp=_bt,
            ts_box=st,
            halobox=hb,
            photon_nonconservation_data=photon_nonconservation_data,
            cache_files={
                "init": [(0, os.path.join(direc, initial_conditions.filename))],
                "perturb_field": perturb_files,
                "halobox": hbox_files,
                "ionized_box": ionize_files,
                "brightness_temp": brightness_files,
                "spin_temp": spin_temp_files,
                "pt_halos": pth_files,
            },
        )
        for z, ib, _bt, st, hb in zip(
            out_redshifts, ib_tracker, bt, st_tracker, hb_tracker
        )
    ]

    # If a single redshift was passed, then pass back singletons.
    if singleton:
        coevals = coevals[0]

    logger.debug("Returning from Coeval")

    return coevals


def _get_required_redshifts_coeval(flag_options, redshift) -> list[float]:
    if min(redshift) < global_params.Z_HEAT_MAX and (
        flag_options.INHOMO_RECO or flag_options.USE_TS_FLUCT
    ):
        redshifts = get_logspaced_redshifts(
            min(redshift),
            global_params.ZPRIME_STEP_FACTOR,
            global_params.Z_HEAT_MAX,
        )
        # Set the highest redshift to exactly Z_HEAT_MAX. This makes the coeval run
        # at exactly the same redshift as the spin temperature box. There's literally
        # no point going higher for a coeval, since the user is only interested in
        # the final "redshift" (if they've specified a z in redshift that is higher
        # that Z_HEAT_MAX, we add it back in below, and so they'll still get it).
        redshifts = np.clip(redshifts, None, global_params.Z_HEAT_MAX)

    else:
        redshifts = [min(redshift)]
    # Add in the redshift defined by the user, and sort in order
    # Turn into a set so that exact matching user-set redshift
    # don't double-up with scrolling ones.
    redshifts = np.concatenate((redshifts, redshift))
    redshifts = np.sort(np.unique(redshifts))[::-1]
    return redshifts.tolist()


def _get_coeval_callbacks(
    scrollz: list[float], coeval_callback, coeval_callback_redshifts
) -> list[bool]:
    compute_coeval_callback = [False] * len(scrollz)

    if coeval_callback is not None:
        if isinstance(coeval_callback_redshifts, (list, np.ndarray)):
            for coeval_z in coeval_callback_redshifts:
                assert isinstance(coeval_z, (int, float, np.number))
                compute_coeval_callback[
                    np.argmin(np.abs(np.array(scrollz) - coeval_z))
                ] = True
            if sum(compute_coeval_callback) != len(coeval_callback_redshifts):
                logger.warning(
                    "some of the coeval_callback_redshifts refer to the same node_redshift"
                )
        elif (
            isinstance(coeval_callback_redshifts, int) and coeval_callback_redshifts > 0
        ):
            compute_coeval_callback = [
                not i % coeval_callback_redshifts for i in range(len(scrollz))
            ]
        else:
            raise ValueError("coeval_callback_redshifts has to be list or integer > 0.")

    return compute_coeval_callback
