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

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from functools import cached_property
from typing import Any, Self

import attrs
import numpy as np
from astropy import units as un
from astropy.cosmology import z_at_value
from bidict import bidict

from .._cfg import config
from ..c_21cmfast import lib
from .arrays import Array
from .exceptions import _process_exitcode
from .inputs import (
    AstroOptions,
    AstroParams,
    CosmoParams,
    InputParameters,
    InputStruct,
    MatterOptions,
    SimulationOptions,
)
from .structs import StructWrapper

logger = logging.getLogger(__name__)

_ALL_OUTPUT_STRUCTS = {}


def _arrayfield(optional: bool = False, **kw):
    if optional:
        return attrs.field(
            default=None,
            validator=attrs.validators.optional(attrs.validators.instance_of(Array)),
            eq=False,
            type=Array,
        )
    else:
        return attrs.field(
            validator=attrs.validators.instance_of(Array),
            eq=False,
            type=Array,
        )


class _HashType(Enum):
    user_cosmo = 0
    zgrid = 1
    full = 2


@attrs.define(slots=False, kw_only=True)
class OutputStruct(ABC):
    """Base class for any class that wraps a C struct meant to be output from a C function."""

    _meta = False
    _c_compute_function = None
    _compat_hash = _HashType.full

    _TYPEMAP = bidict({"float32": "float *", "float64": "double *", "int32": "int *"})

    inputs: InputParameters = attrs.field(
        validator=attrs.validators.instance_of(InputParameters)
    )
    dummy: bool = attrs.field(default=False, converter=bool)
    initial: bool = attrs.field(default=False, converter=bool)

    @property
    def _name(self):
        """The name of the struct."""
        return self.__class__.__name__

    def __init_subclass__(cls):
        """Store subclasses for easy access."""
        if not cls._meta:
            _ALL_OUTPUT_STRUCTS[cls.__name__] = cls

        return super().__init_subclass__()

    @property
    def simulation_options(self) -> SimulationOptions:
        """The SimulationOptions object for this output struct."""
        return self.inputs.simulation_options

    @property
    def matter_options(self) -> MatterOptions:
        """The SimulationOptions object for this output struct."""
        return self.inputs.matter_options

    @property
    def cosmo_params(self) -> CosmoParams:
        """The CosmoParams object for this output struct."""
        return self.inputs.cosmo_params

    @property
    def astro_params(self) -> AstroParams:
        """The AstroParams object for this output struct."""
        return self.inputs.astro_params

    @property
    def astro_options(self) -> AstroOptions:
        """The AstroOptions object for this output struct."""
        return self.inputs.astro_options

    def _inputs_compatible_with(self, other: OutputStruct | InputParameters) -> bool:
        """Check whether this objects' inputs are compatible with another object's.

        This check is sensitive to the fact that the other object may be at a different
        level of the simulation heirarchy, and therefore may be compatible even if the
        params are different. As long as they are the same at the level higher than the
        minimum level of the simulation, they are considered compatible.
        """
        if not isinstance(other, OutputStruct | InputParameters):
            return False

        if isinstance(other, InputParameters):
            # Compare at the level required by this object only
            return getattr(self.inputs, f"_{self._compat_hash.name}_hash") == getattr(
                other, f"_{self._compat_hash.name}_hash"
            )

        min_req = min(self._compat_hash.value, other._compat_hash.value)
        min_req = _HashType(min_req)

        return getattr(self.inputs, f"_{min_req.name}_hash") == getattr(
            other.inputs, f"_{min_req.name}_hash"
        )

    @property
    def arrays(self) -> dict[str, Array]:
        """A dictionary of Array objects whose memory is shared between this object and the C backend."""
        me = attrs.asdict(self, recurse=False)
        return {k: x for k, x in me.items() if isinstance(x, Array)}

    @cached_property
    def _struct(self) -> StructWrapper:
        """The python-wrapped struct associated with this input object."""
        return StructWrapper(self._name)

    @cached_property
    def _cstruct(self) -> StructWrapper:
        """The object pointing to the memory accessed by C-code for this struct."""
        return self._struct.cstruct

    def _init_arrays(self):
        if self.dummy:
            return

        for k, array in self.arrays.items():
            # Don't initialize C-based pointers or already-inited stuff, or stuff
            # that's computed on disk (if it's on disk, accessing the array should
            # just give the computed version, which is what we would want, not a
            # zero-inited array).
            if array.state.c_memory or array.state.initialized or array.state.on_disk:
                continue

            setattr(self, k, array.initialize())

    @property
    def random_seed(self) -> int:
        """The random seed for this particular instance."""
        return self.inputs.random_seed

    def push_to_backend(self):
        """Push the current state of the object with the underlying C-struct.

        This will link any memory initialized by numpy in this object with the underlying
        C-struct, and also update the C struct with any values in the python object.
        """
        # Initialize all uninitialized arrays.
        self._init_arrays()
        for name, array in self.arrays.items():
            # We do *not* set COMPUTED_ON_DISK items to the C-struct here, because we have no
            # way of knowing (in this function) what is required to load in, and we don't want
            # to unnecessarily load things in. We leave it to the user to ensure that all
            # required arrays are loaded into memory before calling this function.
            if array.state.initialized:
                self._struct.expose_to_c(array, name)

        for k in self._struct.primitive_fields:
            if getattr(self, k) is not None:
                setattr(self._cstruct, k, getattr(self, k))

    def pull_from_backend(self):
        """Sync the current state of the object with the underlying C-struct.

        This will pull any primitives calculated in the backend to the python object.
        Arrays are passed in as pointers, and do not need to be copied back.
        """
        for k in self._struct.primitive_fields:
            setattr(self, k, getattr(self._cstruct, k))

    def get(self, ary: str | Array):
        """If possible, load an array from disk, storing it and returning the underlying array."""
        if isinstance(ary, str):
            name = ary
            try:
                ary = self.arrays[ary]
            except KeyError as e:
                try:
                    return getattr(self, ary)  # could be a different attribute...
                except AttributeError:
                    raise AttributeError(f"The array {ary} does not exist") from e
        elif names := [name for name, x in self.arrays.items() if x is ary]:
            name = names[0]
        else:
            raise ValueError("The given array is not a part of this instance.")

        if not ary.state.on_disk and not ary.state.initialized:
            raise ValueError(f"Array '{name}' is not on disk and not initialized.")

        if ary.state.on_disk and not ary.state.computed_in_mem:
            ary = ary.loaded_from_disk()
            setattr(self, name, ary)

        return ary.value

    def set(self, name: str, value: Any):
        """Set the value of an array."""
        if name not in self.arrays:
            try:
                setattr(self, name, value)
            except AttributeError:
                raise AttributeError(f"The attribute '{name}' does not exist") from None
        else:
            setattr(self, name, self.arrays[name].with_value(value))

    def prepare(
        self,
        flush: Sequence[str] | None = None,
        keep: Sequence[str] | None = None,
        force: bool = False,
    ):
        """Prepare the instance for being passed to another function.

        This will flush all arrays in "flush" from memory, and ensure all arrays
        in "keep" are in memory. At least one of these must be provided. By default,
        the complement of the given parameter is all flushed/kept.

        Parameters
        ----------
        flush
            Arrays to flush out of memory. Note that if no file is associated with this
            instance, these arrays will be lost forever.
        keep
            Arrays to keep or load into memory. Note that if these do not already
            exist, they will be loaded from file (if the file exists). Only one of
            ``flush`` and ``keep`` should be specified.
        force
            Whether to force flushing arrays even if no disk storage exists.
        """
        if flush is None and keep is None:
            raise ValueError("Must provide either flush or keep")

        if flush is not None and keep is None:
            keep = [k for k in self.arrays if k not in flush]
        elif flush is None:
            flush = [
                k
                for k, array in self.arrays.items()
                if k not in keep and array.state.initialized
            ]

        flush = flush or []
        keep = keep or []

        for k in flush:
            self._remove_array(k, force=force)

        # For everything we want to keep, we check if it is computed in memory,
        # and if not, load it from disk.
        for k in keep:
            self.get(k)

    def _remove_array(self, k: str, *, force=False):
        array = self.arrays[k]
        state = array.state

        if not state.initialized:
            warnings.warn(
                f"Trying to remove array that isn't yet created: {k}", stacklevel=2
            )
            return

        if state.computed_in_mem and not state.on_disk and not force:
            # if we don't have the array on disk, don't purge unless we really want to
            warnings.warn(
                f"Trying to purge array '{k}' from memory that hasn't been stored! Use force=True if you meant to do this.",
                stacklevel=2,
            )
            return

        if state.c_has_active_memory:
            lib.free(getattr(self._cstruct, k))

        setattr(self, k, array.without_value())

    def purge(self, force=False):
        """Flush all the boxes out of memory.

        Parameters
        ----------
        force
            Whether to force the purge even if no disk storage exists.
        """
        self.prepare(keep=[], force=force)

    def load_all(self):
        """Load all possible arrays into memory."""
        for x in self.arrays:
            self.get(x)

    @property
    def is_computed(self) -> bool:
        """Whether this instance has been computed at all.

        This is true either if the current instance has called :meth:`compute`,
        or if it has a current existing :attr:`path` pointing to stored data,
        or if such a path exists.

        Just because the instance has been computed does *not* mean that all
        relevant quantities are available -- some may have been purged from
        memory without writing. Use :meth:`has` to check whether certain arrays
        are available.
        """
        return any(v.state.is_computed for v in self.arrays.values())

    def ensure_arrays_computed(self, *arrays, load=False) -> bool:
        """Check if the given arrays are computed (not just initialized)."""
        if not self.is_computed:
            return False

        computed = all(self.arrays[k].state.is_computed for k in arrays)

        if computed and load:
            self.prepare(keep=arrays, flush=[])

        return computed

    def ensure_arrays_inited(self, *arrays, init=False) -> bool:
        """Check if the given arrays are initialized (or computed)."""
        inited = all(self.arrays[k].state.initialized for k in arrays)

        if init and not inited:
            self._init_arrays()
        return True

    @abstractmethod
    def get_required_input_arrays(self, input_box: Self) -> list[str]:
        """Return all input arrays required to compute this object."""

    def ensure_input_computed(self, input_box: Self, load: bool = False) -> bool:
        """Ensure all the inputs have been computed."""
        if input_box.dummy:
            return True

        arrays = self.get_required_input_arrays(input_box)
        if input_box.initial:
            return input_box.ensure_arrays_inited(*arrays, init=load)

        return input_box.ensure_arrays_computed(*arrays, load=load)

    def summarize(self, indent: int = 0) -> str:
        """Generate a string summary of the struct."""
        indent = indent * "    "

        # print array type and column headings
        out = (
            f"\n{indent}{self.__class__.__name__:>25}    "
            + "   1st:         End:         Min:         Max:         Mean:         \n"
        )

        # print array extrema and means
        for fieldname, array in self.arrays.items():
            state = array.state
            if not state.initialized:
                out += f"{indent}    {fieldname:>25}:  uninitialized\n"
            elif not state.is_computed:
                out += f"{indent}    {fieldname:>25}:  initialized\n"
            elif not state.computed_in_mem:
                out += f"{indent}    {fieldname:>25}:  computed on disk\n"
            else:
                x = self.get(fieldname).flatten()
                if len(x) > 0:
                    out += f"{indent}    {fieldname:>25}: {x[0]:11.4e}, {x[-1]:11.4e}, {x.min():11.4e}, {x.max():11.4e}, {np.mean(x):11.4e}\n"
                else:
                    out += f"{indent}    {fieldname:>25}: size zero\n"

        # print primitive fields
        out += "".join(
            f"{indent}    {fieldname:>25}: {getattr(self, fieldname, 'non-existent')}\n"
            for fieldname in self._struct.primitive_fields
        )

        return out

    @classmethod
    def _log_call_arguments(cls, *args):
        logger.debug(f"Calling {cls._c_compute_function.__name__} with following args:")

        for arg in args:
            if isinstance(arg, OutputStruct):
                for line in arg.summarize(indent=1).split("\n"):
                    logger.debug(line)
            elif isinstance(arg, InputStruct):
                for line in str(arg).split("\n"):
                    logger.debug(f"    {line}")
            else:
                logger.debug(f"    {arg}")

    def _ensure_arguments_exist(self, *args):
        for arg in args:
            if (
                isinstance(arg, OutputStruct)
                and not arg.dummy
                and not self.ensure_input_computed(arg, load=True)
            ):
                raise ValueError(
                    f"Trying to use {arg.__class__.__name__} to compute "
                    f"{self.__class__.__name__}, but some required arrays "
                    f"are not computed!\nArrays required: "
                    f"{self.get_required_input_arrays(arg)}\n"
                    f"Current State: {[(k, str(v.state)) for k, v in self.arrays.items()]}"
                )

    def _compute(self, allow_already_computed: bool = False, *args):
        """Compute the actual function that fills this struct."""
        # Check that all required inputs are really computed, and load them into memory
        # if they're not already.
        self._ensure_arguments_exist(*args)

        # Write a detailed message about call arguments if debug turned on.
        if logger.getEffectiveLevel() <= logging.DEBUG:
            self._log_call_arguments(*args)

        # Construct the args. All StructWrapper objects need to actually pass their
        # underlying cstruct, rather than themselves.
        inputs = [
            arg._cstruct if isinstance(arg, OutputStruct | InputStruct) else arg
            for arg in args
        ]
        # Sync the python/C memory
        self.push_to_backend()
        for arg in args:
            if isinstance(arg, OutputStruct):
                arg.push_to_backend()

        # Ensure we haven't already tried to compute this instance.
        if self.is_computed and not allow_already_computed:
            raise ValueError(
                f"You are trying to compute {self.__class__.__name__}, but it has already been computed."
            )

        # Perform the C computation
        try:
            exitcode = self._c_compute_function(*inputs, self._cstruct)
        except TypeError as e:
            logger.error(f"Arguments to {self._c_compute_function.__name__}: {inputs}")
            raise e

        _process_exitcode(exitcode, self._c_compute_function, args)

        for name, array in self.arrays.items():
            setattr(self, name, array.computed())

        self.pull_from_backend()
        return self

    @classmethod
    @abstractmethod
    def new(cls, inputs: InputParameters, **kwargs) -> Self:
        """Instantiate the class from InputParameters."""

    def get_full_size(self) -> int:
        """Return the size of the object in bytes.

        This represents the size of the object if it is fully initialized/computed
        and all in memory. Equivalently, it is close to the file size on disk.
        """
        size = 0
        for ary in self.arrays.values():
            size += np.prod(ary.shape) * np.dtype(ary.dtype).itemsize
        return size


@attrs.define(slots=False, kw_only=True)
class InitialConditions(OutputStruct):
    """A class representing an InitialConditions C-struct."""

    _c_compute_function = lib.ComputeInitialConditions
    _meta = False
    _compat_hash = _HashType.user_cosmo

    lowres_density = _arrayfield()
    lowres_vx = _arrayfield(optional=True)
    lowres_vy = _arrayfield(optional=True)
    lowres_vz = _arrayfield(optional=True)
    hires_density = _arrayfield()
    hires_vx = _arrayfield(optional=True)
    hires_vy = _arrayfield(optional=True)
    hires_vz = _arrayfield(optional=True)

    lowres_vx_2LPT = _arrayfield(optional=True)
    lowres_vy_2LPT = _arrayfield(optional=True)
    lowres_vz_2LPT = _arrayfield(optional=True)
    hires_vx_2LPT = _arrayfield(optional=True)
    hires_vy_2LPT = _arrayfield(optional=True)
    hires_vz_2LPT = _arrayfield(optional=True)

    lowres_vcb = _arrayfield(optional=True)

    @classmethod
    def new(cls, inputs: InputParameters, **kw) -> Self:
        """Create a new instance, given a set of input parameters."""
        shape = (inputs.simulation_options.HII_DIM,) * 2 + (
            int(
                inputs.simulation_options.NON_CUBIC_FACTOR
                * inputs.simulation_options.HII_DIM
            ),
        )
        hires_shape = (inputs.simulation_options.DIM,) * 2 + (
            int(
                inputs.simulation_options.NON_CUBIC_FACTOR
                * inputs.simulation_options.DIM
            ),
        )

        out = {
            "lowres_density": Array(shape, dtype=np.float32),
            "hires_density": Array(hires_shape, dtype=np.float32),
        }
        if inputs.matter_options.PERTURB_ON_HIGH_RES:
            out |= {
                "hires_vx": Array(hires_shape, dtype=np.float32),
                "hires_vy": Array(hires_shape, dtype=np.float32),
                "hires_vz": Array(hires_shape, dtype=np.float32),
            }
        else:
            out |= {
                "lowres_vx": Array(shape, dtype=np.float32),
                "lowres_vy": Array(shape, dtype=np.float32),
                "lowres_vz": Array(shape, dtype=np.float32),
            }

        if inputs.matter_options.PERTURB_ALGORITHM == "2LPT":
            out |= {
                "hires_vx_2LPT": Array(hires_shape, dtype=np.float32),
                "hires_vy_2LPT": Array(hires_shape, dtype=np.float32),
                "hires_vz_2LPT": Array(hires_shape, dtype=np.float32),
            }
            if not inputs.matter_options.PERTURB_ON_HIGH_RES:
                out |= {
                    "lowres_vx_2LPT": Array(shape, dtype=np.float32),
                    "lowres_vy_2LPT": Array(shape, dtype=np.float32),
                    "lowres_vz_2LPT": Array(shape, dtype=np.float32),
                }

        if inputs.matter_options.V_CB_MODEL == "FLUCTS":
            out["lowres_vcb"] = Array(shape, dtype=np.float32)

        return cls(inputs=inputs, **out, **kw)

    def prepare_for_perturb(self, force: bool = False):
        """Ensure the ICs have all the boxes loaded for perturb, but no extra."""
        keep = ["hires_density"]

        if not self.matter_options.PERTURB_ON_HIGH_RES:
            keep.append("lowres_density")
            keep.append("lowres_vx")
            keep.append("lowres_vy")
            keep.append("lowres_vz")

            if self.matter_options.PERTURB_ALGORITHM == "2LPT":
                keep.append("lowres_vx_2LPT")
                keep.append("lowres_vy_2LPT")
                keep.append("lowres_vz_2LPT")

        else:
            keep.append("hires_vx")
            keep.append("hires_vy")
            keep.append("hires_vz")

            if self.matter_options.PERTURB_ALGORITHM == "2LPT":
                keep.append("hires_vx_2LPT")
                keep.append("hires_vy_2LPT")
                keep.append("hires_vz_2LPT")

        if self.matter_options.V_CB_MODEL == "FLUCTS":
            keep.append("lowres_vcb")

        self.prepare(keep=keep, force=force)

    def prepare_for_spin_temp(self, force: bool = False):
        """Ensure ICs have all boxes required for spin_temp, and no more."""
        keep = []
        if self.matter_options.V_CB_MODEL == "FLUCTS":
            keep.append("lowres_vcb")

        if self.matter_options.lagrangian_source_grid:
            if not self.matter_options.PERTURB_ON_HIGH_RES:
                keep.append("lowres_density")
                keep.append("lowres_vx")
                keep.append("lowres_vy")
                keep.append("lowres_vz")

                if self.matter_options.PERTURB_ALGORITHM == "2LPT":
                    keep.append("lowres_vx_2LPT")
                    keep.append("lowres_vy_2LPT")
                    keep.append("lowres_vz_2LPT")

            else:
                keep.append("hires_density")
                keep.append("hires_vx")
                keep.append("hires_vy")
                keep.append("hires_vz")

                if self.matter_options.PERTURB_ALGORITHM == "2LPT":
                    keep.append("hires_vx_2LPT")
                    keep.append("hires_vy_2LPT")
                    keep.append("hires_vz_2LPT")

        self.prepare(keep=keep, force=force)

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        return []

    def compute(self, allow_already_computed: bool = False):
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.random_seed,
        )


@attrs.define(slots=False, kw_only=True)
class OutputStructZ(OutputStruct):
    """The same as an OutputStruct, but containing a redshift."""

    _meta = True
    redshift: float = attrs.field(converter=float)

    @classmethod
    def dummy(cls):
        """Create a dummy instance with the given inputs."""
        return cls.new(inputs=InputParameters(random_seed=1), redshift=-1.0, dummy=True)

    @classmethod
    def initial(cls, inputs):
        """Create a dummy instance with the given inputs."""
        return cls.new(inputs=inputs, redshift=-1.0, initial=True)


@attrs.define(slots=False, kw_only=True)
class PerturbedField(OutputStructZ):
    """A class containing all perturbed field boxes."""

    _c_compute_function = lib.ComputePerturbedField
    _meta = False
    _compat_hash = _HashType.zgrid

    density = _arrayfield()
    velocity_z = _arrayfield(optional=True)
    velocity_x = _arrayfield(optional=True)
    velocity_y = _arrayfield(optional=True)

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw) -> Self:
        """Create a new PerturbedField instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.

        Other Parameters
        ----------------
        All other parameters are passed through to the :class:`PerturbedField`
        constructor.
        """
        dim = inputs.simulation_options.HII_DIM

        shape = (dim, dim, int(inputs.simulation_options.NON_CUBIC_FACTOR * dim))

        out = {
            "density": Array(shape, dtype=np.float32),
        }

        # No need for velocities in a homogeneous universe!
        if dim > 1:
            out["velocity_z"] = Array(shape, dtype=np.float32)
            if inputs.matter_options.KEEP_3D_VELOCITIES:
                out["velocity_x"] = Array(shape, dtype=np.float32)
                out["velocity_y"] = Array(shape, dtype=np.float32)

        return cls(inputs=inputs, redshift=redshift, **out, **kw)

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []

        if not isinstance(input_box, InitialConditions):
            raise ValueError(
                f"{type(input_box)} is not an input required for PerturbedField!"
            )

        # Always require hires_density
        required += ["hires_density"]

        if self.matter_options.PERTURB_ON_HIGH_RES:
            required += ["hires_vx", "hires_vy", "hires_vz"]

            if self.matter_options.PERTURB_ALGORITHM == "2LPT":
                required += ["hires_vx_2LPT", "hires_vy_2LPT", "hires_vz_2LPT"]

        else:
            required += ["lowres_density", "lowres_vx", "lowres_vy", "lowres_vz"]

            if self.matter_options.PERTURB_ALGORITHM == "2LPT":
                required += [
                    "lowres_vx_2LPT",
                    "lowres_vy_2LPT",
                    "lowres_vz_2LPT",
                ]

        if self.matter_options.V_CB_MODEL == "FLUCTS":
            required.append("lowres_vcb")

        return required

    def compute(self, *, allow_already_computed: bool = False, ics: InitialConditions):
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.redshift,
            ics,
        )

    @property
    def velocity(self):
        """The velocity of the box in the 3rd dimension (for backwards compat)."""
        return self.velocity_z  # for backwards compatibility


@attrs.define(frozen=True, slots=True)
class Halo:
    """Represents a single halo from a HaloCatalog.

    This class wraps individual halo data extracted from a HaloCatalog,
    providing easy access to its properties (mass, coordinates, RNG seeds, etc.).
    """

    mass: float
    coords: np.ndarray
    redshift: float
    star_rng: float | None = None
    sfr_rng: float | None = None
    xray_rng: float | None = None


@attrs.define(slots=False, kw_only=True)
class HaloCatalog(OutputStructZ):
    """A class containing all fields related to halos."""

    _c_compute_function = lib.ComputeHaloCatalog
    _meta = False
    desc_redshift: float | None = attrs.field(default=None)
    _compat_hash = _HashType.zgrid

    halo_masses = _arrayfield()
    star_rng = _arrayfield()
    sfr_rng = _arrayfield()
    xray_rng = _arrayfield()
    halo_coords = _arrayfield()
    n_halos: int = attrs.field(default=None)
    buffer_size: int = attrs.field(default=None)

    @classmethod
    def new(
        cls,
        inputs: InputParameters,
        redshift: float,
        buffer_size: float | None = None,
        **kw,
    ) -> Self:
        """Create a new PerturbedHaloCatalog instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.

        Other Parameters
        ----------------
        All other parameters are passed through to the :class:`PerturbedHaloCatalog`
        constructor.
        """
        from .cfuncs import get_halo_catalog_buffer_size

        if kw.get("dummy", False):
            buffer_size = 0
        elif buffer_size is None:
            buffer_size = get_halo_catalog_buffer_size(
                redshift=redshift,
                inputs=inputs,
            )

        return cls(
            inputs=inputs,
            halo_masses=Array((buffer_size,), dtype=np.float32),
            star_rng=Array((buffer_size,), dtype=np.float32),
            sfr_rng=Array((buffer_size,), dtype=np.float32),
            xray_rng=Array((buffer_size,), dtype=np.float32),
            halo_coords=Array((buffer_size, 3), dtype=np.float32),
            redshift=redshift,
            buffer_size=buffer_size,
            **kw,
        )

    def trim_to_n_halos(self) -> Self:
        """Trim the halo catalog to have its size the same as the actual number of halos."""
        self.buffer_size = self.n_halos
        for name, array in self.arrays.items():
            new_shape = (self.n_halos, *array.shape[1:])
            new_array = array.trimmed(new_shape)
            setattr(self, name, new_array)

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if self.matter_options.SOURCE_MODEL == "CHMF-SAMPLER":
                # when the sampler is on, the grids are only needed for the first sample
                if self.desc_redshift <= 0:
                    required += ["hires_density"]
                    required += ["lowres_density"]
            # without the sampler, dexm needs the hires density at each redshift
            else:
                required += ["hires_density"]
        elif isinstance(input_box, HaloCatalog):
            if self.matter_options.SOURCE_MODEL == "CHMF-SAMPLER":
                required += [
                    "halo_masses",
                    "halo_coords",
                    "star_rng",
                    "sfr_rng",
                    "xray_rng",
                ]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for HaloCatalog!"
            )
        return required

    def compute(
        self,
        *,
        descendant_halos: HaloCatalog,
        ics: InitialConditions,
        allow_already_computed: bool = False,
    ):
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.desc_redshift,
            self.redshift,
            ics,
            ics.random_seed,
            descendant_halos,
        )

    def __getitem__(self, index: int) -> Halo:
        """Get a specific halo by index."""
        if not isinstance(index, int) or index < 0 or index >= self.n_halos:
            raise IndexError(f"Halo index {index} out of range [0, {self.n_halos})")
        return Halo(
            mass=float(self.halo_masses.value[index]),
            coords=self.halo_coords.value[index].copy(),
            redshift=self.redshift,
            star_rng=float(self.star_rng.value[index]),
            sfr_rng=float(self.sfr_rng.value[index]),
            xray_rng=float(self.xray_rng.value[index]),
        )

    def __iter__(self):
        """Iterate over halos in the catalog, yielding Halo objects."""
        for i in range(self.n_halos):
            yield self[i]

    def __len__(self):
        """Return the number of halos in the catalog."""
        return self.n_halos


@attrs.define(slots=False, kw_only=True)
class PerturbedHaloCatalog(OutputStructZ):
    """A class to hold a HaloCatalog whose coordinates are in real (Eulerian) space."""

    _c_compute_function = lib.ComputePerturbedHaloCatalog
    _meta = False
    desc_redshift: float | None = attrs.field(default=None)
    _compat_hash = _HashType.zgrid

    halo_masses = _arrayfield()
    halo_coords = _arrayfield()

    sfr = _arrayfield()
    stellar_masses = _arrayfield()
    ion_emissivity = _arrayfield()
    xray_emissivity = _arrayfield(optional=True)
    fesc_sfr = _arrayfield(optional=True)

    stellar_mini = _arrayfield(optional=True)
    sfr_mini = _arrayfield(optional=True)

    n_halos: int = attrs.field(default=None)
    buffer_size: int = attrs.field(default=None)

    @classmethod
    def new(
        cls,
        inputs: InputParameters,
        redshift: float,
        buffer_size: float | None = None,
        **kw,
    ) -> Self:
        """Create a new PerturbedHaloCatalog instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.

        Other Parameters
        ----------------
        All other parameters are passed through to the :class:`PerturbedHaloCatalog`
        constructor.
        """
        from .cfuncs import get_halo_catalog_buffer_size

        if kw.get("dummy", False):
            buffer_size = 0
        elif buffer_size is None:
            buffer_size = get_halo_catalog_buffer_size(
                redshift=redshift,
                inputs=inputs,
            )

        out = {
            "halo_coords": Array((buffer_size, 3), dtype=np.float32),
            "halo_masses": Array((buffer_size,), dtype=np.float32),
            "stellar_masses": Array((buffer_size,), dtype=np.float32),
            "sfr": Array((buffer_size,), dtype=np.float32),
            "ion_emissivity": Array((buffer_size,), dtype=np.float32),
        }
        if inputs.astro_options.USE_TS_FLUCT:
            out["xray_emissivity"] = Array((buffer_size,), dtype=np.float32)
        if inputs.astro_options.RECOMB_MODEL != "none":
            out["fesc_sfr"] = Array((buffer_size,), dtype=np.float32)
        if inputs.astro_options.USE_MINI_HALOS:
            out["stellar_mini"] = Array((buffer_size,), dtype=np.float32)
            out["sfr_mini"] = Array((buffer_size,), dtype=np.float32)

        return cls(
            inputs=inputs,
            redshift=redshift,
            buffer_size=buffer_size,
            **out,
            **kw,
        )

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if self.matter_options.PERTURB_ON_HIGH_RES:
                required += ["hires_vx", "hires_vy", "hires_vz"]
            else:
                required += ["lowres_vx", "lowres_vy", "lowres_vz"]

            if self.matter_options.PERTURB_ALGORITHM == "2LPT":
                required += [f"{k}_2LPT" for k in required]

            if self.matter_options.V_CB_MODEL == "FLUCTS":
                required += ["lowres_vcb"]

        elif isinstance(input_box, TsBox):
            if self.astro_options.USE_MINI_HALOS:
                required += ["J_21_LW"]
        elif isinstance(input_box, IonizedBox):
            required += ["ionisation_rate_G12", "z_reion"]

        elif isinstance(input_box, HaloCatalog):
            required += [
                "halo_coords",
                "halo_masses",
                "star_rng",
                "sfr_rng",
                "xray_rng",
            ]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for PerturbedHaloCatalog!"
            )

        return required

    def compute(
        self,
        *,
        ics: InitialConditions,
        previous_spin_temp: TsBox,
        previous_ionize_box: IonizedBox,
        halo_catalog: HaloCatalog,
        allow_already_computed: bool = False,
    ):
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.redshift,
            ics,
            previous_spin_temp,
            previous_ionize_box,
            halo_catalog,
        )

    def __getitem__(self, index: int) -> Halo:
        """Get a specific halo by index."""
        if not isinstance(index, int) or index < 0 or index >= self.n_halos:
            raise IndexError(f"Halo index {index} out of range [0, {self.n_halos})")
        return Halo(
            mass=float(self.halo_masses.value[index]),
            coords=self.halo_coords.value[index].copy(),
            redshift=self.redshift,
        )

    def __iter__(self):
        """Iterate over halos in the catalog, yielding Halo objects."""
        for i in range(self.n_halos):
            yield self[i]

    def __len__(self):
        """Return the number of halos in the catalog."""
        return self.n_halos


@attrs.define(slots=False, kw_only=True)
class HaloBox(OutputStructZ):
    """A class containing all gridded halo properties."""

    _meta = False
    _c_compute_function = lib.ComputeHaloBox

    count = _arrayfield(optional=True)
    halo_mass = _arrayfield(optional=True)
    halo_stars = _arrayfield(optional=True)
    halo_stars_mini = _arrayfield(optional=True)
    halo_sfr = _arrayfield(optional=True)
    halo_sfr_mini = _arrayfield(optional=True)
    halo_xray = _arrayfield(optional=True)
    n_ion = _arrayfield()
    whalo_sfr = _arrayfield(optional=True)

    log10_Mcrit_ACG_ave: float = attrs.field(default=None)
    log10_Mcrit_MCG_ave: float = attrs.field(default=None)

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw) -> Self:
        """Create a new HaloBox instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.

        Other Parameters
        ----------------
        All other parameters are passed through to the :class:`HaloBox`
        constructor.
        """
        dim = inputs.simulation_options.HII_DIM
        shape = (dim, dim, int(inputs.simulation_options.NON_CUBIC_FACTOR * dim))

        # TODO: right now it's a waste of memory and time to compute n_ion for Eulerian source models,
        # this might change in the future
        out = {"n_ion": Array(shape, dtype=np.float32)}

        # TODO: similarly, as above, whalo_sfr is only needed for Lagrangian source models at the moment, but this might change
        if (
            inputs.astro_options.RECOMB_MODEL != "none"
            and inputs.matter_options.lagrangian_source_grid
        ):
            out["whalo_sfr"] = Array(shape, dtype=np.float32)

        if inputs.astro_options.USE_TS_FLUCT:
            out["halo_xray"] = Array(shape, dtype=np.float32)
            out["halo_sfr"] = Array(shape, dtype=np.float32)
            if inputs.astro_options.USE_MINI_HALOS:
                out["halo_sfr_mini"] = Array(shape, dtype=np.float32)

        if config["EXTRA_HALOBOX_FIELDS"]:
            out["count"] = Array(shape, dtype=np.float32)
            out["halo_mass"] = Array(shape, dtype=np.float32)
            out["halo_stars"] = Array(shape, dtype=np.float32)
            if inputs.astro_options.USE_MINI_HALOS:
                out["halo_stars_mini"] = Array(shape, dtype=np.float32)

        return cls(
            inputs=inputs,
            redshift=redshift,
            **out,
            **kw,
        )

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, HaloCatalog):
            if self.matter_options.has_discrete_halos:
                required += [
                    "halo_coords",
                    "halo_masses",
                    "star_rng",
                    "sfr_rng",
                    "xray_rng",
                ]
        elif isinstance(input_box, TsBox):
            if self.astro_options.USE_MINI_HALOS:
                required += ["J_21_LW"]
        elif isinstance(input_box, IonizedBox):
            required += ["ionisation_rate_G12", "z_reion"]
        elif isinstance(input_box, PerturbedField):
            if not self.matter_options.lagrangian_source_grid:
                required += ["density"]
        elif isinstance(input_box, InitialConditions):
            if self.matter_options.lagrangian_source_grid:
                if self.matter_options.PERTURB_ON_HIGH_RES:
                    required += ["hires_density", "hires_vx", "hires_vy", "hires_vz"]
                else:
                    required += [
                        "lowres_density",
                        "lowres_vx",
                        "lowres_vy",
                        "lowres_vz",
                    ]

                if self.matter_options.PERTURB_ALGORITHM == "2LPT":
                    required += [f"{k}_2LPT" for k in required if "_v" in k]

            if (
                self.matter_options.V_CB_MODEL == "FLUCTS"
                and self.astro_options.USE_MINI_HALOS
            ):
                required += ["lowres_vcb"]
        else:
            raise ValueError(f"{type(input_box)} is not an input required for HaloBox!")

        return required

    def compute(
        self,
        *,
        initial_conditions: InitialConditions,
        perturbed_field: PerturbedField,
        halo_catalog: HaloCatalog,
        previous_spin_temp: TsBox,
        previous_ionize_box: IonizedBox,
        allow_already_computed: bool = False,
    ):
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.redshift,
            initial_conditions,
            perturbed_field,
            halo_catalog,
            previous_spin_temp,
            previous_ionize_box,
        )

    def prepare_for_next_snapshot(self, next_z, force: bool = False):
        """Prepare the HaloBox for the next snapshot."""
        keep = []
        # We need to keep fields for interpolation only if we calculate the spin temperature
        if self.astro_options.USE_TS_FLUCT:
            # find maximum z
            d_max_needed = (
                self.cosmo_params.cosmo.comoving_distance(next_z)
                + self.astro_params.R_MAX_TS * un.Mpc
            )
            max_z_needed = z_at_value(
                self.cosmo_params.cosmo.comoving_distance, d_max_needed
            )

            z_arr = np.array(self.inputs.node_redshifts)
            # we need one redshift above the max z for interpolation, so find that value
            last_z_above = (
                z_arr[z_arr > max_z_needed].min()
                if z_arr.max() > max_z_needed
                else z_arr.max() + 1
            )

            # If we need the box, only keep the interpolated fields
            if self.redshift <= last_z_above:
                keep += ["halo_sfr", "halo_xray"]
                if self.astro_options.USE_MINI_HALOS:
                    keep += ["halo_sfr_mini"]
        self.prepare(keep=keep, force=force)


@attrs.define(slots=False, kw_only=True)
class RadiationFieldsSetup(OutputStructZ):
    """A class containing the fields that are neccesary for setting up the radiation fields."""

    _meta = False
    _c_compute_function = lib.SetupRadiationFields

    # R-dependent arrays which are set once
    zpp_avg = _arrayfield()
    R_values = _arrayfield()
    zpp_edges = _arrayfield()
    # Arrays for the filtered emissivity fields
    filtered_sfr = _arrayfield()
    filtered_sfr_mini = _arrayfield(optional=True)
    filtered_xray = _arrayfield()
    filtered_sfr_lw = _arrayfield(optional=True)
    filtered_sfr_mini_lw = _arrayfield(optional=True)
    # Frequency integral tables
    freq_int_heat_tbl = _arrayfield()
    freq_int_ion_tbl = _arrayfield()
    freq_int_lya_tbl = _arrayfield()
    freq_int_heat_tbl_diff = _arrayfield()
    freq_int_ion_tbl_diff = _arrayfield()
    freq_int_lya_tbl_diff = _arrayfield()
    # helpers for the interpolation
    inverse_diff = _arrayfield()
    m_xHII_low_box = _arrayfield()
    inverse_val_box = _arrayfield()
    # arrays for R-dependent prefactors
    lya_flux_continuum_prefactor = _arrayfield(optional=True)
    lya_flux_injected_prefactor = _arrayfield(optional=True)
    lya_flux_continuum_injected_prefactor = _arrayfield(optional=True)
    lyw_flux_prefactor = _arrayfield(optional=True)
    lyw_flux_prefactor_MINI = _arrayfield(optional=True)
    lya_flux_continuum_prefactor_MINI = _arrayfield(optional=True)
    lya_flux_injected_prefactor_MINI = _arrayfield(optional=True)
    lya_flux_continuum_injected_prefactor_MINI = _arrayfield(optional=True)
    # array and floats required for the X-ray optical depth calculation
    ave_log10_MturnLW = _arrayfield(optional=True)
    Q_HI_zp: float = attrs.field(default=1.0)
    x_e_ave_zp: float = attrs.field(default=0.0)
    # boolean to indicate whether there's enough light
    NO_LIGHT: bool = attrs.field(default=True)
    # maximum source redshift for the radiation fields calculation
    source_z_max: float = attrs.field(default=0.0)
    # redshifts of the input hboxes
    hbox_redshifts: list[float] = attrs.field(factory=list)

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw) -> Self:
        """Create a new RadiationFieldsSetup instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.

        Other Parameters
        ----------------
        All other parameters are passed through to the :class:`RadiationFieldsSetup`
        constructor.
        """
        x_int_NXHII = 14  # defined in elec_interp.h
        shape = (inputs.simulation_options.HII_DIM,) * 2 + (
            int(
                inputs.simulation_options.NON_CUBIC_FACTOR
                * inputs.simulation_options.HII_DIM
            ),
        )

        out = {
            "R_values": Array((inputs.astro_params.N_STEP_TS,), dtype=np.float64),
            "zpp_avg": Array((inputs.astro_params.N_STEP_TS,), dtype=np.float64),
            "zpp_edges": Array((inputs.astro_params.N_STEP_TS,), dtype=np.float64),
            "freq_int_heat_tbl": Array(
                (x_int_NXHII, inputs.astro_params.N_STEP_TS), dtype=np.float64
            ),
            "freq_int_ion_tbl": Array(
                (x_int_NXHII, inputs.astro_params.N_STEP_TS), dtype=np.float64
            ),
            "freq_int_lya_tbl": Array(
                (x_int_NXHII, inputs.astro_params.N_STEP_TS), dtype=np.float64
            ),
            "freq_int_heat_tbl_diff": Array(
                (x_int_NXHII, inputs.astro_params.N_STEP_TS), dtype=np.float64
            ),
            "freq_int_ion_tbl_diff": Array(
                (x_int_NXHII, inputs.astro_params.N_STEP_TS), dtype=np.float64
            ),
            "freq_int_lya_tbl_diff": Array(
                (x_int_NXHII, inputs.astro_params.N_STEP_TS), dtype=np.float64
            ),
            "inverse_diff": Array((x_int_NXHII,), dtype=np.float32),
            "m_xHII_low_box": Array(shape, dtype=np.int32),
            "inverse_val_box": Array(shape, dtype=np.float32),
            "filtered_sfr": Array(shape, dtype=np.float32),
            "filtered_xray": Array(shape, dtype=np.float32),
        }

        if inputs.astro_options.USE_MINI_HALOS:
            out["filtered_sfr_mini"] = Array(shape, dtype=np.float32)
            if inputs.astro_options.LYA_MULTIPLE_SCATTERING:
                out["filtered_sfr_lw"] = Array(shape, dtype=np.float32)
                out["filtered_sfr_mini_lw"] = Array(shape, dtype=np.float32)

        if inputs.astro_options.USE_LYA_HEATING:
            out["lya_flux_continuum_prefactor"] = Array(
                (inputs.astro_params.N_STEP_TS,), dtype=np.float64
            )
            out["lya_flux_injected_prefactor"] = Array(
                (inputs.astro_params.N_STEP_TS,), dtype=np.float64
            )
        else:
            out["lya_flux_continuum_injected_prefactor"] = Array(
                (inputs.astro_params.N_STEP_TS,), dtype=np.float64
            )

        if inputs.astro_options.USE_MINI_HALOS:
            out["lyw_flux_prefactor"] = Array(
                (inputs.astro_params.N_STEP_TS,), dtype=np.float64
            )
            out["lyw_flux_prefactor_MINI"] = Array(
                (inputs.astro_params.N_STEP_TS,), dtype=np.float64
            )
            if inputs.astro_options.USE_LYA_HEATING:
                out["lya_flux_continuum_prefactor_MINI"] = Array(
                    (inputs.astro_params.N_STEP_TS,), dtype=np.float64
                )
                out["lya_flux_injected_prefactor_MINI"] = Array(
                    (inputs.astro_params.N_STEP_TS,), dtype=np.float64
                )
            else:
                out["lya_flux_continuum_injected_prefactor_MINI"] = Array(
                    (inputs.astro_params.N_STEP_TS,), dtype=np.float64
                )

        if inputs.astro_options.USE_MINI_HALOS:
            out["ave_log10_MturnLW"] = Array(
                (inputs.astro_params.N_STEP_TS,), dtype=np.float64
            )

        return cls(
            inputs=inputs,
            redshift=redshift,
            **out,
            **kw,
        )

    def setup_shells(
        self,
        inputs: InputParameters,
        redshift: float,
    ) -> Self:
        """
        Set up shells information for a given redshift.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters specifying the run.
        redshift : float
            The redshift at which to compute the radiation fields.
        """
        # set minimum R at cell size
        l_factor = (4 * np.pi / 3.0) ** (-1 / 3)
        if inputs.simulation_options.HII_DIM == 1:
            # If HII_DIM=1 (happens when we run_global_evolution), we take a typical cell size of 1.5Mpc,
            # just to for setting the z'' array (note that filtering won't be done on a box with a single cell)
            R_min = 1.5 * l_factor
        else:
            R_min = (
                inputs.simulation_options.BOX_LEN
                / inputs.simulation_options.HII_DIM
                * l_factor
            )
        # now we need to find the closest halo box to the redshift of the shell
        cosmo_ap = inputs.cosmo_params.cosmo
        cmd_zp = cosmo_ap.comoving_distance(redshift)
        R_steps = np.arange(0, inputs.astro_params.N_STEP_TS)
        R_factor = (inputs.astro_params.R_MAX_TS / R_min) ** (
            R_steps / inputs.astro_params.N_STEP_TS
        )
        self.set("R_values", R_min * R_factor)
        cmd_edges = cmd_zp + self.R_values.value * un.Mpc  # comoving distance edges
        # Get the edges of the shells
        zmin = z_at_value(cosmo_ap.comoving_distance, cmd_edges.min()).value
        zmax = z_at_value(cosmo_ap.comoving_distance, cmd_edges.max()).value
        zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 100)
        dgrid = cosmo_ap.comoving_distance(zgrid)
        self.set("zpp_edges", np.interp(cmd_edges.value, dgrid.value, zgrid))
        # the `average` redshift of the shell is the average of the
        # inner and outer redshifts (following the C code)
        self.set(
            "zpp_avg",
            self.zpp_edges.value
            - np.diff(np.insert(self.zpp_edges.value, 0, redshift)) / 2,
        )
        return self

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, TsBox):
            required += ["xray_ionised_fraction"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for RadiationFieldsSetup!"
            )

        return required

    def compute(
        self,
        *,
        redshift,
        previous_spin_temp: TsBox,
        allow_already_computed: bool = False,
    ):
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            redshift,
            previous_spin_temp,
        )


@attrs.define(slots=False, kw_only=True)
class RadiationFields(OutputStructZ):
    """A class containing the radiation fields."""

    _meta = False
    _c_compute_function = lib.UpdateRadiationFields

    xray_heating_rate = _arrayfield(optional=True)
    xray_ionization_rate = _arrayfield()
    xray_lya_flux = _arrayfield()
    lya_flux_continuum_injected = _arrayfield(optional=True)
    lyw_flux = _arrayfield(optional=True)
    lya_flux_continuum = _arrayfield(optional=True)
    lya_flux_injected = _arrayfield(optional=True)
    Q_HI: float = attrs.field(default=1.0)

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw) -> Self:
        """Create a new RadiationFields instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.

        Other Parameters
        ----------------
        All other parameters are passed through to the :class:`RadiationFields`
        constructor.
        """
        shape = (inputs.simulation_options.HII_DIM,) * 2 + (
            int(
                inputs.simulation_options.NON_CUBIC_FACTOR
                * inputs.simulation_options.HII_DIM
            ),
        )

        # TODO: the arrays below are defined as np.float64, but should be np.float32 - see https://github.com/21cmfast/21cmFAST/issues/744
        out = {
            "xray_ionization_rate": Array(shape, dtype=np.float64),
            "xray_lya_flux": Array(shape, dtype=np.float64),
        }
        if inputs.astro_options.USE_MINI_HALOS:
            out["lyw_flux"] = Array(shape, dtype=np.float64)

        if inputs.astro_options.USE_X_RAY_HEATING:
            out["xray_heating_rate"] = Array(shape, dtype=np.float64)

        if inputs.astro_options.USE_LYA_HEATING:
            out["lya_flux_continuum"] = Array(shape, dtype=np.float64)
            out["lya_flux_injected"] = Array(shape, dtype=np.float64)
        else:
            out["lya_flux_continuum_injected"] = Array(shape, dtype=np.float64)

        return cls(
            inputs=inputs,
            redshift=redshift,
            **out,
            **kw,
        )

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, PerturbedField):
            required += ["density"]
        elif isinstance(input_box, TsBox):
            required += ["xray_ionised_fraction"]
        elif isinstance(input_box, HaloBox):
            required += ["halo_sfr", "halo_xray"]
            if self.astro_options.USE_MINI_HALOS:
                required += ["halo_sfr_mini"]
        elif isinstance(input_box, RadiationFieldsSetup):
            required += [
                "R_values",
                "zpp_avg",
                "zpp_edges",
                "freq_int_heat_tbl",
                "freq_int_ion_tbl",
                "freq_int_lya_tbl",
                "freq_int_heat_tbl_diff",
                "freq_int_ion_tbl_diff",
                "freq_int_lya_tbl_diff",
                "inverse_diff",
                "m_xHII_low_box",
                "inverse_val_box",
                "filtered_sfr",
                "filtered_xray",
            ]
            if self.astro_options.USE_MINI_HALOS:
                required += ["filtered_sfr_mini"]
                if self.astro_options.LYA_MULTIPLE_SCATTERING:
                    required += ["filtered_sfr_lw", "filtered_sfr_mini_lw"]

            if self.astro_options.USE_LYA_HEATING:
                required += [
                    "lya_flux_continuum_prefactor",
                    "lya_flux_injected_prefactor",
                ]
            else:
                required += ["lya_flux_continuum_injected_prefactor"]

            if self.astro_options.USE_MINI_HALOS:
                required += ["lyw_flux_prefactor", "lyw_flux_prefactor_MINI"]
                if self.astro_options.USE_LYA_HEATING:
                    required += [
                        "lya_flux_continuum_prefactor_MINI",
                        "lya_flux_injected_prefactor_MINI",
                    ]
                else:
                    required += ["lya_flux_continuum_injected_prefactor_MINI"]

            if self.astro_options.USE_MINI_HALOS:
                required += ["ave_log10_MturnLW"]
        else:
            raise ValueError(
                f"{type(input_box)} is not an input required for RadiationFields!"
            )

        return required

    def compute(
        self,
        *,
        redshift,
        halobox: HaloBox,
        R_ct,
        R_star,
        perturbed_field: PerturbedField,
        previous_spin_temp: TsBox,
        rad_setup: RadiationFieldsSetup,
        allow_already_computed: bool = False,
    ):
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            redshift,
            halobox,
            R_ct,
            R_star,
            perturbed_field,
            previous_spin_temp,
            rad_setup,
        )


@attrs.define(slots=False, kw_only=True)
class TsBox(OutputStructZ):
    """A class containing all spin temperature boxes."""

    _c_compute_function = lib.ComputeTsBox
    _meta = False

    spin_temperature = _arrayfield()
    xray_ionised_fraction = _arrayfield()
    kinetic_temp_neutral = _arrayfield()
    J_21_LW = _arrayfield(optional=True)
    Q_HI: float = attrs.field(default=1.0)

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw) -> Self:
        """Create a new TsBox instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.

        Other Parameters
        ----------------
        All other parameters are passed through to the :class:`TsBox`
        constructor.
        """
        shape = (inputs.simulation_options.HII_DIM,) * 2 + (
            int(
                inputs.simulation_options.NON_CUBIC_FACTOR
                * inputs.simulation_options.HII_DIM
            ),
        )
        out = {
            "spin_temperature": Array(shape, dtype=np.float32),
            "xray_ionised_fraction": Array(shape, dtype=np.float32),
            "kinetic_temp_neutral": Array(shape, dtype=np.float32),
        }
        if inputs.astro_options.USE_MINI_HALOS:
            out["J_21_LW"] = Array(shape, dtype=np.float32)

        return cls(inputs=inputs, redshift=redshift, **out, **kw)

    @cached_property
    def global_Ts(self):
        """Global (mean) spin temperature."""
        if not self.is_computed:
            raise AttributeError(
                "global_Ts is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.get("spin_temperature"))

    @cached_property
    def global_Tk(self):
        """Global (mean) Tk."""
        if not self.is_computed:
            raise AttributeError(
                "global_Tk is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.get("kinetic_temp_neutral"))

    @cached_property
    def global_x_e(self):
        """Global (mean) x_e."""
        if not self.is_computed:
            raise AttributeError(
                "global_x_e is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.get("xray_ionised_fraction"))

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if (
                self.matter_options.V_CB_MODEL == "FLUCTS"
                and self.astro_options.USE_MINI_HALOS
            ):
                required += ["lowres_vcb"]
        elif isinstance(input_box, PerturbedField):
            required += ["density"]
        elif isinstance(input_box, TsBox):
            required += [
                "kinetic_temp_neutral",
                "xray_ionised_fraction",
                "spin_temperature",
            ]
            if self.astro_options.USE_MINI_HALOS:
                required += ["J_21_LW"]
        elif isinstance(input_box, RadiationFields):
            required += [
                "xray_ionization_rate",
                "xray_lya_flux",
            ]
            if self.astro_options.USE_X_RAY_HEATING:
                required += ["xray_heating_rate"]
            if self.astro_options.USE_MINI_HALOS:
                required += ["lyw_flux"]
            if self.astro_options.USE_LYA_HEATING:
                required += ["lya_flux_continuum", "lya_flux_injected"]
            else:
                required += ["lya_flux_continuum_injected"]
        else:
            raise ValueError(f"{type(input_box)} is not an input required for TsBox!")

        return required

    def compute(
        self,
        *,
        cleanup: bool,
        perturbed_field: PerturbedField,
        radiation_fields: RadiationFields,
        prev_spin_temp: TsBox,
        ics: InitialConditions,
        allow_already_computed: bool = False,
    ):
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.redshift,
            prev_spin_temp.redshift,
            perturbed_field.redshift,
            cleanup,
            perturbed_field,
            radiation_fields,
            prev_spin_temp,
            ics,
        )


@attrs.define(slots=False, kw_only=True)
class IonizedBox(OutputStructZ):
    """A class containing all ionized boxes."""

    _meta = False
    _c_compute_function = lib.ComputeIonizedBox

    neutral_fraction = _arrayfield()
    ionisation_rate_G12 = _arrayfield()
    z_reion = _arrayfield()
    mean_free_path = _arrayfield(optional=True)
    cumulative_recombinations = _arrayfield(optional=True)
    kinetic_temperature = _arrayfield(optional=True)
    unnormalised_nion = _arrayfield(optional=True)
    unnormalised_nion_mini = _arrayfield(optional=True)
    log10_Mturnover_ave: float = attrs.field(default=None)
    log10_Mturnover_MINI_ave: float = attrs.field(default=None)
    mean_f_coll: float = attrs.field(default=None)
    mean_f_coll_MINI: float = attrs.field(default=None)

    @classmethod
    def new(cls, inputs, redshift: float, **kw) -> Self:
        """Create a new IonizedBox instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.

        Other Parameters
        ----------------
        All other parameters are passed through to the :class:`IonizedBox`
        constructor.
        """
        if (
            inputs.astro_options.USE_MINI_HALOS
            and not inputs.matter_options.lagrangian_source_grid
            and inputs.simulation_options.HII_DIM > 1
        ):
            n_filtering = (
                int(
                    np.log(
                        min(
                            inputs.astro_params.R_BUBBLE_MAX,
                            0.620350491 * inputs.simulation_options.BOX_LEN,
                        )
                        / max(
                            inputs.astro_params.R_BUBBLE_MIN,
                            0.620350491
                            * inputs.simulation_options.BOX_LEN
                            / inputs.simulation_options.HII_DIM,
                        )
                    )
                    / np.log(inputs.astro_params.DELTA_R_HII_FACTOR)
                )
                + 1
            )
        else:
            n_filtering = 1

        shape = (inputs.simulation_options.HII_DIM,) * 2 + (
            int(
                inputs.simulation_options.NON_CUBIC_FACTOR
                * inputs.simulation_options.HII_DIM
            ),
        )
        filter_shape = (n_filtering, *shape)

        out = {
            "neutral_fraction": Array(shape, initfunc=np.ones, dtype=np.float32),
            "ionisation_rate_G12": Array(shape, dtype=np.float32),
            "z_reion": Array(shape, dtype=np.float32),
        }

        if not inputs.matter_options.MINIMIZE_MEMORY:
            out["kinetic_temperature"] = Array(shape, dtype=np.float32)
            if inputs.astro_options.RECOMB_MODEL != "none":
                out["mean_free_path"] = Array(shape, dtype=np.float32)

        if inputs.astro_options.RECOMB_MODEL == "inhomogeneous":
            out["cumulative_recombinations"] = Array(shape, dtype=np.float32)
        elif inputs.astro_options.RECOMB_MODEL == "homogeneous":
            out["cumulative_recombinations"] = Array((1, 1, 1), dtype=np.float32)

        if not inputs.matter_options.lagrangian_source_grid:
            out["unnormalised_nion"] = Array(filter_shape, dtype=np.float32)

            if inputs.astro_options.USE_MINI_HALOS:
                out["unnormalised_nion_mini"] = Array(filter_shape, dtype=np.float32)

        return cls(inputs=inputs, redshift=redshift, **out, **kw)

    @cached_property
    def global_xH(self):
        """Global (mean) neutral fraction."""
        if not self.is_computed:
            raise AttributeError(
                "global_xH is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.get("neutral_fraction"))

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if (
                self.matter_options.V_CB_MODEL == "FLUCTS"
                and self.astro_options.USE_MINI_HALOS
            ):
                required += ["lowres_vcb"]
        elif isinstance(input_box, PerturbedField):
            required += ["density"]
        elif isinstance(input_box, TsBox):
            required += ["kinetic_temp_neutral", "xray_ionised_fraction"]
            if self.astro_options.USE_MINI_HALOS:
                required += ["J_21_LW"]
        elif isinstance(input_box, IonizedBox):
            required += ["z_reion", "ionisation_rate_G12"]
            if self.astro_options.RECOMB_MODEL != "none":
                required += [
                    "cumulative_recombinations",
                ]
            if (
                self.astro_options.USE_MINI_HALOS
                and not self.matter_options.lagrangian_source_grid
            ):
                required += [
                    "unnormalised_nion",
                    "unnormalised_nion_mini",
                ]
        elif isinstance(input_box, HaloBox):
            if self.matter_options.lagrangian_source_grid:
                required += ["n_ion"]
                if self.astro_options.RECOMB_MODEL != "none":
                    required += ["whalo_sfr"]
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
        halobox: HaloBox,
        ics: InitialConditions,
        allow_already_computed: bool = False,
    ):
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.redshift,
            prev_perturbed_field.redshift,
            perturbed_field,
            prev_perturbed_field,
            prev_ionize_box,
            spin_temp,
            halobox,
            ics,
        )


@attrs.define(slots=False, kw_only=True)
class BrightnessTemp(OutputStructZ):
    """A class containing the brightness temperature box."""

    _c_compute_function = lib.ComputeBrightnessTemp

    _meta = False
    brightness_temp = _arrayfield()
    tau_21 = _arrayfield(optional=True)

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw) -> Self:
        """Create a new BrightnessTemp instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.

        Other Parameters
        ----------------
        All other parameters are passed through to the :class:`BrightnessTemp`
        constructor.
        """
        shape = (inputs.simulation_options.HII_DIM,) * 2 + (
            int(
                inputs.simulation_options.NON_CUBIC_FACTOR
                * inputs.simulation_options.HII_DIM
            ),
        )

        out = {"brightness_temp": Array(shape, dtype=np.float32)}
        if inputs.astro_options.USE_TS_FLUCT:
            out["tau_21"] = Array(shape, dtype=np.float32)

        return cls(
            inputs=inputs,
            redshift=redshift,
            **out,
            **kw,
        )

    @cached_property
    def global_Tb(self):
        """Global (mean) brightness temperature."""
        if not self.is_computed:
            raise AttributeError(
                "global_Tb is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.get("brightness_temp"))

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, PerturbedField):
            required += ["density"]
        elif isinstance(input_box, TsBox):
            required += ["spin_temperature"]
        elif isinstance(input_box, IonizedBox):
            required += ["neutral_fraction"]
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
        allow_already_computed: bool = False,
    ):
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.redshift,
            spin_temp,
            ionized_box,
            perturbed_field,
        )
