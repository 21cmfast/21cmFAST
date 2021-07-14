"""
Output class objects.

The classes provided by this module exist to simplify access to large datasets created within C.
Fundamentally, ownership of the data belongs to these classes, and the C functions merely accesses
this and fills it. The various boxes and lightcones associated with each output are available as
instance attributes. Along with the output data, each output object contains the various input
parameter objects necessary to define it.

.. warning:: These should not be instantiated or filled by the user, but always handled
             as output objects from the various functions contained here. Only the data
             within the objects should be accessed.
"""
import h5py
import numpy as np
import os
import warnings
from astropy import units
from astropy.cosmology import z_at_value
from cached_property import cached_property
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from . import __version__
from . import _utils as _ut
from ._cfg import config
from ._utils import OutputStruct as _BaseOutputStruct
from ._utils import _check_compatible_inputs
from .c_21cmfast import ffi, lib
from .inputs import AstroParams, CosmoParams, FlagOptions, UserParams, global_params


class _OutputStruct(_BaseOutputStruct):
    _global_params = global_params

    def __init__(self, *, user_params=None, cosmo_params=None, **kwargs):

        self.cosmo_params = cosmo_params or CosmoParams()
        self.user_params = user_params or UserParams()

        super().__init__(**kwargs)

    _ffi = ffi


class _OutputStructZ(_OutputStruct):
    _inputs = _OutputStruct._inputs + ("redshift",)


class InitialConditions(_OutputStruct):
    """A class containing all initial conditions boxes."""

    _c_compute_function = lib.ComputeInitialConditions

    # The filter params indicates parameters to overlook when deciding if a cached box
    # matches current parameters.
    # It is useful for ignoring certain global parameters which may not apply to this
    # step or its dependents.
    _meta = False
    _filter_params = _OutputStruct._filter_params + [
        "ALPHA_UVB",  # ionization
        "EVOLVE_DENSITY_LINEARLY",  # perturb
        "SMOOTH_EVOLVED_DENSITY_FIELD",  # perturb
        "R_smooth_density",  # perturb
        "HII_ROUND_ERR",  # ionization
        "FIND_BUBBLE_ALGORITHM",  # ib
        "N_POISSON",  # ib
        "T_USE_VELOCITIES",  # bt
        "MAX_DVDR",  # bt
        "DELTA_R_HII_FACTOR",  # ib
        "HII_FILTER",  # ib
        "INITIAL_REDSHIFT",  # pf
        "HEAT_FILTER",  # st
        "CLUMPING_FACTOR",  # st
        "Z_HEAT_MAX",  # st
        "R_XLy_MAX",  # st
        "NUM_FILTER_STEPS_FOR_Ts",  # ts
        "ZPRIME_STEP_FACTOR",  # ts
        "TK_at_Z_HEAT_MAX",  # ts
        "XION_at_Z_HEAT_MAX",  # ts
        "Pop",  # ib
        "Pop2_ion",  # ib
        "Pop3_ion",  # ib
        "NU_X_BAND_MAX",  # st
        "NU_X_MAX",  # ib
    ]

    def prepare_for_perturb(self, flag_options: FlagOptions, force: bool = False):
        """Ensure the ICs have all the boxes loaded for perturb, but no extra."""
        keep = ["hires_density"]

        if not self.user_params.PERTURB_ON_HIGH_RES:
            keep.append("lowres_density")
            keep.append("lowres_vx")
            keep.append("lowres_vy")
            keep.append("lowres_vz")

            if self.user_params.USE_2LPT:
                keep.append("lowres_vx_2LPT")
                keep.append("lowres_vy_2LPT")
                keep.append("lowres_vz_2LPT")

            if flag_options.USE_HALO_FIELD:
                keep.append("hires_density")
        else:
            keep.append("hires_vx")
            keep.append("hires_vy")
            keep.append("hires_vz")

            if self.user_params.USE_2LPT:
                keep.append("hires_vx_2LPT")
                keep.append("hires_vy_2LPT")
                keep.append("hires_vz_2LPT")

        if self.user_params.USE_RELATIVE_VELOCITIES:
            keep.append("lowres_vcb")

        self.prepare(keep=keep, force=force)

    def prepare_for_spin_temp(self, flag_options: FlagOptions, force: bool = False):
        """Ensure ICs have all boxes required for spin_temp, and no more."""
        keep = []
        if self.user_params.USE_RELATIVE_VELOCITIES:
            keep.append("lowres_vcb")
        self.prepare(keep=keep, force=force)

    def _get_box_structures(self) -> Dict[str, Union[Dict, Tuple[int]]]:
        shape = (self.user_params.HII_DIM,) * 3
        hires_shape = (self.user_params.DIM,) * 3

        out = {
            "lowres_density": shape,
            "lowres_vx": shape,
            "lowres_vy": shape,
            "lowres_vz": shape,
            "hires_density": hires_shape,
            "hires_vx": hires_shape,
            "hires_vy": hires_shape,
            "hires_vz": hires_shape,
        }

        if self.user_params.USE_2LPT:
            out.update(
                {
                    "lowres_vx_2LPT": shape,
                    "lowres_vy_2LPT": shape,
                    "lowres_vz_2LPT": shape,
                    "hires_vx_2LPT": hires_shape,
                    "hires_vy_2LPT": hires_shape,
                    "hires_vz_2LPT": hires_shape,
                }
            )

        if self.user_params.USE_RELATIVE_VELOCITIES:
            out.update({"lowres_vcb": shape})

        return out

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> List[str]:
        """Return all input arrays required to compute this object."""
        return []

    def compute(self, hooks: dict):
        """Compute the function."""
        return self._compute(
            self.random_seed,
            self.user_params,
            self.cosmo_params,
            hooks=hooks,
        )


class PerturbedField(_OutputStructZ):
    """A class containing all perturbed field boxes."""

    _c_compute_function = lib.ComputePerturbField

    _meta = False
    _filter_params = _OutputStruct._filter_params + [
        "ALPHA_UVB",  # ionization
        "HII_ROUND_ERR",  # ionization
        "FIND_BUBBLE_ALGORITHM",  # ib
        "N_POISSON",  # ib
        "T_USE_VELOCITIES",  # bt
        "MAX_DVDR",  # bt
        "DELTA_R_HII_FACTOR",  # ib
        "HII_FILTER",  # ib
        "HEAT_FILTER",  # st
        "CLUMPING_FACTOR",  # st
        "Z_HEAT_MAX",  # st
        "R_XLy_MAX",  # st
        "NUM_FILTER_STEPS_FOR_Ts",  # ts
        "ZPRIME_STEP_FACTOR",  # ts
        "TK_at_Z_HEAT_MAX",  # ts
        "XION_at_Z_HEAT_MAX",  # ts
        "Pop",  # ib
        "Pop2_ion",  # ib
        "Pop3_ion",  # ib
        "NU_X_BAND_MAX",  # st
        "NU_X_MAX",  # ib
    ]

    def _get_box_structures(self) -> Dict[str, Union[Dict, Tuple[int]]]:
        return {
            "density": (self.user_params.HII_DIM,) * 3,
            "velocity": (self.user_params.HII_DIM,) * 3,
        }

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> List[str]:
        """Return all input arrays required to compute this object."""
        required = []

        if not isinstance(input_box, InitialConditions):
            raise ValueError(
                f"{type(input_box)} is not an input required for PerturbedField!"
            )

        # Always require hires_density
        required += ["hires_density"]

        if self.user_params.PERTURB_ON_HIGH_RES:
            required += ["hires_vx", "hires_vy", "hires_vz"]

            if self.user_params.USE_2LPT:
                required += ["hires_vx_2LPT", "hires_vy_2LPT", "hires_vz_2LPT"]

        else:
            required += ["lowres_density", "lowres_vx", "lowres_vy", "lowres_vz"]

            if self.user_params.USE_2LPT:
                required += [
                    "lowres_vx_2LPT",
                    "lowres_vy_2LPT",
                    "lowres_vz_2LPT",
                ]

        if self.user_params.USE_RELATIVE_VELOCITIES:
            required.append("lowres_vcb")

        return required

    def compute(self, *, ics: InitialConditions, hooks: dict):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.user_params,
            self.cosmo_params,
            ics,
            hooks=hooks,
        )


class _AllParamsBox(_OutputStructZ):
    _meta = True
    _inputs = _OutputStructZ._inputs + ("flag_options", "astro_params")

    _filter_params = _OutputStruct._filter_params + [
        "T_USE_VELOCITIES",  # bt
        "MAX_DVDR",  # bt
    ]

    def __init__(
        self,
        *,
        astro_params: Optional[AstroParams] = None,
        flag_options: Optional[FlagOptions] = None,
        first_box: bool = False,
        **kwargs,
    ):
        self.flag_options = flag_options or FlagOptions()
        self.astro_params = astro_params or AstroParams(
            INHOMO_RECO=self.flag_options.INHOMO_RECO
        )

        self.log10_Mturnover_ave = 0.0
        self.log10_Mturnover_MINI_ave = 0.0

        self.first_box = first_box
        if first_box:
            self.mean_f_coll = 0.0
            self.mean_f_coll_MINI = 0.0

        super().__init__(**kwargs)


class HaloField(_AllParamsBox):
    """A class containing all fields related to halos."""

    _c_based_pointers = (
        "halo_masses",
        "halo_coords",
        "mass_bins",
        "fgtrm",
        "sqrt_dfgtrm",
        "dndlm",
        "sqrtdn_dlm",
    )
    _c_compute_function = lib.ComputeHaloField

    def _get_box_structures(self) -> Dict[str, Union[Dict, Tuple[int]]]:
        return {}

    def _c_shape(self, cstruct):
        return {
            "halo_masses": (cstruct.n_halos,),
            "halo_coords": (cstruct.n_halos, 3),
            "mass_bins": (cstruct.n_mass_bins,),
            "fgtrm": (cstruct.n_mass_bins,),
            "sqrt_dfgtrm": (cstruct.n_mass_bins,),
            "dndlm": (cstruct.n_mass_bins,),
            "sqrtdn_dlm": (cstruct.n_mass_bins,),
        }

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> List[str]:
        """Return all input arrays required to compute this object."""
        if isinstance(input_box, InitialConditions):
            return ["hires_density"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for HaloField!"
            )

    def compute(self, *, ics: InitialConditions, hooks: dict):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            ics,
            hooks=hooks,
        )


class PerturbHaloField(_AllParamsBox):
    """A class containing all fields related to halos."""

    _c_compute_function = lib.ComputePerturbHaloField
    _c_based_pointers = ("halo_masses", "halo_coords")

    def _get_box_structures(self) -> Dict[str, Union[Dict, Tuple[int]]]:
        return {}

    def _c_shape(self, cstruct):
        return {
            "halo_masses": (cstruct.n_halos,),
            "halo_coords": (cstruct.n_halos, 3),
        }

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> List[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if self.user_params.PERTURB_ON_HIGH_RES:
                required += ["hires_vx", "hires_vy", "hires_vz"]
            else:
                required += ["lowres_vx", "lowres_vy", "lowres_vz"]

            if self.user_params.USE_2LPT:
                required += [k + "_2LPT" for k in required]
        elif isinstance(input_box, HaloField):
            required += ["halo_coords", "halo_masses"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for PerturbHaloField!"
            )

        return required

    def compute(self, *, ics: InitialConditions, halo_field: HaloField, hooks: dict):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            ics,
            halo_field,
            hooks=hooks,
        )


class TsBox(_AllParamsBox):
    """A class containing all spin temperature boxes."""

    _c_compute_function = lib.ComputeTsBox
    _meta = False
    _inputs = _AllParamsBox._inputs + ("prev_spin_redshift", "perturbed_field_redshift")

    def __init__(
        self,
        *,
        prev_spin_redshift: Optional[float] = None,
        perturbed_field_redshift: Optional[float] = None,
        **kwargs,
    ):
        self.prev_spin_redshift = prev_spin_redshift
        self.perturbed_field_redshift = perturbed_field_redshift
        super().__init__(**kwargs)

    def _get_box_structures(self) -> Dict[str, Union[Dict, Tuple[int]]]:
        shape = (self.user_params.HII_DIM,) * 3
        return {
            "Ts_box": shape,
            "x_e_box": shape,
            "Tk_box": shape,
            "J_21_LW_box": shape,
        }

    @cached_property
    def global_Ts(self):
        """Global (mean) spin temperature."""
        if "Ts_box" not in self._computed_arrays:
            raise AttributeError(
                "global_Ts is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.Ts_box)

    @cached_property
    def global_Tk(self):
        """Global (mean) Tk."""
        if "Tk_box" not in self._computed_arrays:
            raise AttributeError(
                "global_Tk is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.Tk_box)

    @cached_property
    def global_x_e(self):
        """Global (mean) x_e."""
        if "x_e_box" not in self._computed_arrays:
            raise AttributeError(
                "global_x_e is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.x_e_box)

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> List[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if (
                self.user_params.USE_RELATIVE_VELOCITIES
                and self.flag_options.USE_MINI_HALOS
            ):
                required += ["lowres_vcb"]
        elif isinstance(input_box, PerturbedField):
            required += ["density"]
        elif isinstance(input_box, TsBox):
            required += [
                "Tk_box",
                "x_e_box",
            ]
            if self.flag_options.USE_MINI_HALOS:
                required += ["J_21_LW_box"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for PerturbHaloField!"
            )

        return required

    def compute(
        self,
        *,
        cleanup: bool,
        perturbed_field: PerturbedField,
        prev_spin_temp,
        ics: InitialConditions,
        hooks: dict,
    ):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.prev_spin_redshift,
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            self.perturbed_field_redshift,
            cleanup,
            perturbed_field,
            prev_spin_temp,
            ics,
            hooks=hooks,
        )


class IonizedBox(_AllParamsBox):
    """A class containing all ionized boxes."""

    _meta = False
    _c_compute_function = lib.ComputeIonizedBox
    _inputs = _AllParamsBox._inputs + ("prev_ionize_redshift",)

    def __init__(self, *, prev_ionize_redshift: Optional[float] = None, **kwargs):
        self.prev_ionize_redshift = prev_ionize_redshift
        super().__init__(**kwargs)

    def _get_box_structures(self) -> Dict[str, Union[Dict, Tuple[int]]]:
        if self.flag_options.USE_MINI_HALOS:
            n_filtering = (
                int(
                    np.log(
                        min(
                            self.astro_params.R_BUBBLE_MAX,
                            0.620350491 * self.user_params.BOX_LEN,
                        )
                        / max(
                            global_params.R_BUBBLE_MIN,
                            0.620350491
                            * self.user_params.BOX_LEN
                            / self.user_params.HII_DIM,
                        )
                    )
                    / np.log(global_params.DELTA_R_HII_FACTOR)
                )
                + 1
            )
        else:
            n_filtering = 1

        shape = (self.user_params.HII_DIM,) * 3
        filter_shape = (n_filtering,) + shape

        out = {
            "xH_box": {"init": np.ones, "shape": shape},
            "Gamma12_box": shape,
            "MFP_box": shape,
            "z_re_box": shape,
            "dNrec_box": shape,
            "temp_kinetic_all_gas": shape,
            "Fcoll": filter_shape,
        }

        if self.flag_options.USE_MINI_HALOS:
            out["Fcoll_MINI"] = filter_shape

        return out

    @cached_property
    def global_xH(self):
        """Global (mean) neutral fraction."""
        if not self.filled:
            raise AttributeError(
                "global_xH is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.xH_box)

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> List[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if (
                self.user_params.USE_RELATIVE_VELOCITIES
                and self.flag_options.USE_MASS_DEPENDENT_ZETA
            ):
                required += ["lowres_vcb"]
        elif isinstance(input_box, PerturbedField):
            required += ["density"]
        elif isinstance(input_box, TsBox):
            required += ["J_21_LW_box", "x_e_box", "Tk_box"]
        elif isinstance(input_box, IonizedBox):
            required += ["z_re_box", "Gamma12_box"]
            if self.flag_options.INHOMO_RECO:
                required += [
                    "dNrec_box",
                ]
            if (
                self.flag_options.USE_MASS_DEPENDENT_ZETA
                and self.flag_options.USE_MINI_HALOS
            ):
                required += ["Fcoll", "Fcoll_MINI"]
        elif isinstance(input_box, PerturbHaloField):
            required += ["halo_coords", "halo_masses"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for IonizedBox!"
            )

        return required

    def compute(
        self,
        *,
        perturbed_field: PerturbedField,
        prev_perturbed_field: PerturbedField,
        prev_ionize_box,
        spin_temp: TsBox,
        pt_halos: PerturbHaloField,
        ics: InitialConditions,
        hooks: dict,
    ):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.prev_ionize_redshift,
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            perturbed_field,
            prev_perturbed_field,
            prev_ionize_box,
            spin_temp,
            pt_halos,
            ics,
            hooks=hooks,
        )


class BrightnessTemp(_AllParamsBox):
    """A class containing the brightness temperature box."""

    _c_compute_function = lib.ComputeBrightnessTemp

    _meta = False
    _filter_params = _OutputStructZ._filter_params

    def _get_box_structures(self) -> Dict[str, Union[Dict, Tuple[int]]]:
        return {"brightness_temp": (self.user_params.HII_DIM,) * 3}

    @cached_property
    def global_Tb(self):
        """Global (mean) brightness temperature."""
        if not self.is_computed:
            raise AttributeError(
                "global_Tb is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.brightness_temp)

    def get_required_input_arrays(self, input_box: _BaseOutputStruct) -> List[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, PerturbedField):
            required += ["velocity"]
        elif isinstance(input_box, TsBox):
            required += ["Ts_box"]
        elif isinstance(input_box, IonizedBox):
            required += ["xH_box"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for BrightnessTemp!"
            )

        return required

    def compute(
        self,
        *,
        spin_temp: TsBox,
        ionized_box: IonizedBox,
        perturbed_field: PerturbedField,
        hooks: dict,
    ):
        """Compute the function."""
        return self._compute(
            self.redshift,
            self.user_params,
            self.cosmo_params,
            self.astro_params,
            self.flag_options,
            spin_temp,
            ionized_box,
            perturbed_field,
            hooks=hooks,
        )


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
            "ionized_box": IonizedBox,
            "spin_temp": TsBox,
            "brightness_temp": BrightnessTemp,
        }
        cls = kinds[kind]

        return cls.from_file(fname, load_data=load_data)

    def gather(
        self,
        fname: Union[str, None, Path] = None,
        kinds: Union[Sequence, None] = None,
        clean: Union[bool, dict] = False,
        direc: Union[str, Path, None] = None,
    ) -> Path:
        """Gather the cached data associated with this object into its file."""
        kinds = kinds or [
            "init",
            "perturb_field",
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
        return "{name}_z{redshift:.4}_{{hash}}_r{seed}.h5".format(
            name=self.__class__.__name__,
            redshift=float(self.redshift),
            seed=self.random_seed,
        )

    def _input_rep(self):
        rep = ""
        for inp in [
            "user_params",
            "cosmo_params",
            "astro_params",
            "flag_options",
            "global_params",
        ]:
            rep += repr(getattr(self, inp))
        return rep

    def get_unique_filename(self):
        """Generate a unique hash filename for this instance."""
        return self._get_prefix().format(
            hash=md5((self._input_rep() + self._particular_rep()).encode()).hexdigest()
        )

    def _write(self, direc=None, fname=None, clobber=False):
        """
        Write the lightcone to file in standard HDF5 format.

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
                "The file {} already exists. If you want to overwrite, set clobber=True.".format(
                    fname
                )
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
                    dct = q.self
                except AttributeError:
                    dct = q

                for kk, v in dct.items():
                    if v is None:
                        continue
                    try:
                        grp.attrs[kk] = v
                    except TypeError:
                        # external_table_path is a cdata object and can't be written.
                        pass

            if self.photon_nonconservation_data is not None:
                photon_data = f.create_group("photon_nonconservation_data")
                for k, val in self.photon_nonconservation_data.items():
                    photon_data[k] = val

            f.attrs["redshift"] = self.redshift
            f.attrs["random_seed"] = self.random_seed
            f.attrs["version"] = __version__

        self._write_particulars(fname)

        return fname

    def _write_particulars(self, fname):
        pass

    def save(self, fname=None, direc="."):
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
        return self._write(direc=direc, fname=fname)

    @classmethod
    def _read_inputs(cls, fname):
        kwargs = {}
        with h5py.File(fname, "r") as fl:
            glbls = dict(fl["_globals"].attrs)
            kwargs["redshift"] = fl.attrs["redshift"]

            if "photon_nonconservation_data" in fl.keys():
                d = fl["photon_nonconservation_data"]
                kwargs["photon_nonconservation_data"] = {k: d[k][...] for k in d.keys()}

        return kwargs, glbls

    @classmethod
    def read(cls, fname, direc="."):
        """Read a lightcone file from disk, creating a LightCone object.

        Parameters
        ----------
        fname : str
            The filename path. Can be absolute or relative.
        direc : str
            If fname, is relative, the directory in which to find the file. By default,
            both the current directory and default cache and the  will be searched, in
            that order.

        Returns
        -------
        LightCone :
            A :class:`LightCone` instance created from the file's data.
        """
        if not os.path.isabs(fname):
            fname = os.path.abspath(os.path.join(direc, fname))

        if not os.path.exists(fname):
            raise FileExistsError(f"The file {fname} does not exist!")

        park, glbls = cls._read_inputs(fname)
        boxk = cls._read_particular(fname)

        with global_params.use(**glbls):
            out = cls(**park, **boxk)

        return out


class Coeval(_HighLevelOutput):
    """A full coeval box with all associated data."""

    def __init__(
        self,
        redshift: float,
        initial_conditions: InitialConditions,
        perturbed_field: PerturbedField,
        ionized_box: IonizedBox,
        brightness_temp: BrightnessTemp,
        ts_box: Union[TsBox, None] = None,
        cache_files: Union[dict, None] = None,
        photon_nonconservation_data=None,
        _globals=None,
    ):
        _check_compatible_inputs(
            initial_conditions,
            perturbed_field,
            ionized_box,
            brightness_temp,
            ts_box,
            ignore=[],
        )

        self.redshift = redshift
        self.init_struct = initial_conditions
        self.perturb_struct = perturbed_field
        self.ionization_struct = ionized_box
        self.brightness_temp_struct = brightness_temp
        self.spin_temp_struct = ts_box

        self.cache_files = cache_files

        self.photon_nonconservation_data = photon_nonconservation_data
        # A *copy* of the current global parameters.
        self.global_params = _globals or dict(global_params.items())

        # Expose all the fields of the structs to the surface of the Coeval object
        for box in [
            initial_conditions,
            perturbed_field,
            ionized_box,
            brightness_temp,
            ts_box,
        ]:
            if box is None:
                continue
            for field in box._get_box_structures():
                setattr(self, field, getattr(box, field))

    @classmethod
    def get_fields(cls, spin_temp: bool = True) -> List[str]:
        """Obtain a list of name of simulation boxes saved in the Coeval object."""
        pointer_fields = []
        for cls in [InitialConditions, PerturbedField, IonizedBox, BrightnessTemp]:
            pointer_fields += cls.get_pointer_fields()

        if spin_temp:
            pointer_fields += TsBox.get_pointer_fields()

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

    def _particular_rep(self):
        return ""

    def _write_particulars(self, fname):
        for name in ["init", "perturb", "ionization", "brightness_temp", "spin_temp"]:
            struct = getattr(self, name + "_struct")
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
    def _read_particular(cls, fname):
        kwargs = {}

        with h5py.File(fname, "r") as fl:
            for output_class in _ut.OutputStruct._implementations():
                if output_class.__name__ in fl:
                    kwargs[
                        _ut.camel_to_snake(output_class.__name__)
                    ] = output_class.from_file(fname)

        return kwargs

    def __eq__(self, other):
        """Determine if this is equal to another object."""
        return (
            isinstance(other, self.__class__)
            and other.redshift == self.redshift
            and self.user_params == other.user_params
            and self.cosmo_params == other.cosmo_params
            and self.flag_options == other.flag_options
            and self.astro_params == other.astro_params
        )


class LightCone(_HighLevelOutput):
    """A full Lightcone with all associated evolved data."""

    def __init__(
        self,
        redshift,
        user_params,
        cosmo_params,
        astro_params,
        flag_options,
        random_seed,
        lightcones,
        node_redshifts=None,
        global_quantities=None,
        photon_nonconservation_data=None,
        cache_files: Union[dict, None] = None,
        _globals=None,
    ):
        self.redshift = redshift
        self.random_seed = random_seed
        self.user_params = user_params
        self.cosmo_params = cosmo_params
        self.astro_params = astro_params
        self.flag_options = flag_options
        self.node_redshifts = node_redshifts
        self.cache_files = cache_files

        # A *copy* of the current global parameters.
        self.global_params = _globals or dict(global_params.items())

        if global_quantities:
            for name, data in global_quantities.items():
                if name.endswith("_box"):
                    # Remove the _box because it looks dumb.
                    setattr(self, "global_" + name[:-4], data)
                else:
                    setattr(self, "global_" + name, data)

        self.photon_nonconservation_data = photon_nonconservation_data

        for name, data in lightcones.items():
            setattr(self, name, data)

        # Hold a reference to the global/lightcones in a dict form for easy reference.
        self.global_quantities = global_quantities
        self.lightcones = lightcones

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
        return self.brightness_temp.shape

    @property
    def n_slices(self):
        """Number of redshift slices in the lightcone."""
        return self.shape[-1]

    @property
    def lightcone_coords(self):
        """Co-ordinates [Mpc] of each cell along the redshift axis."""
        return np.linspace(0, self.lightcone_dimensions[-1], self.n_slices)

    @property
    def lightcone_distances(self):
        """Comoving distance to each cell along the redshift axis, from z=0."""
        return (
            self.cosmo_params.cosmo.comoving_distance(self.redshift).value
            + self.lightcone_coords
        )

    @property
    def lightcone_redshifts(self):
        """Redshift of each cell along the redshift axis."""
        return np.array(
            [
                z_at_value(self.cosmo_params.cosmo.comoving_distance, d * units.Mpc)
                for d in self.lightcone_distances
            ]
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

    @classmethod
    def _read_inputs(cls, fname):
        kwargs = {}
        with h5py.File(fname, "r") as fl:
            for (k, kls) in [
                ("user_params", UserParams),
                ("cosmo_params", CosmoParams),
                ("flag_options", FlagOptions),
                ("astro_params", AstroParams),
            ]:
                grp = fl[k]
                kwargs[k] = kls(dict(grp.attrs))
            kwargs["random_seed"] = fl.attrs["random_seed"]

        # Get the standard inputs.
        kw, glbls = _HighLevelOutput._read_inputs(fname)
        return {**kw, **kwargs}, glbls

    @classmethod
    def _read_particular(cls, fname):
        kwargs = {}
        with h5py.File(fname, "r") as fl:
            boxes = fl["lightcones"]
            kwargs["lightcones"] = {k: boxes[k][...] for k in boxes.keys()}

            glb = fl["global_quantities"]
            kwargs["global_quantities"] = {k: glb[k][...] for k in glb.keys()}

            kwargs["node_redshifts"] = fl["node_redshifts"][...]

        return kwargs

    def __eq__(self, other):
        """Determine if this is equal to another object."""
        return (
            isinstance(other, self.__class__)
            and other.redshift == self.redshift
            and np.all(np.isclose(other.node_redshifts, self.node_redshifts, atol=1e-3))
            and self.user_params == other.user_params
            and self.cosmo_params == other.cosmo_params
            and self.flag_options == other.flag_options
            and self.astro_params == other.astro_params
            and self.global_quantities.keys() == other.global_quantities.keys()
            and self.lightcones.keys() == other.lightcones.keys()
        )
