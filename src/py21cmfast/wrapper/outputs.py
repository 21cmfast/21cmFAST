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
from collections.abc import Callable, Iterator, Sequence
from enum import Enum
from functools import cache, cached_property, wraps
from typing import (
    Any,
    ClassVar,
    Concatenate,
    Self,
    dataclass_transform,
    get_type_hints,
    overload,
)

import attrs
import deprecation
import numpy as np
from astropy import units as un
from astropy.cosmology import z_at_value
from bidict import bidict

from .._cfg import config
from ..c_21cmfast import lib
from .arrays import Array, ArrayProxy, expose_arrays
from .exceptions import CFunction, _process_exitcode
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

_ALL_OUTPUT_STRUCTS: dict[str, type[OutputStruct]] = {}


@overload
def _arrayfield(*, alias: str) -> Array: ...


@overload
def _arrayfield(*, alias: str, default: None) -> Array | None: ...


def _arrayfield(
    *, alias: str, default: attrs.NothingType | None = attrs.NOTHING
) -> Array | None:
    """Define an attrs field holding an :class:`Array`.

    The field must be private (``_<name>``), with its public name given as ``alias``
    (the name of its ``__init__`` argument), and be paired with an
    :class:`~.arrays.ArrayProxy` declared under the public name. Pass ``default=None``
    for an array that only exists for some sets of inputs.

    This is a field specifier of :func:`_output_struct` (see
    :func:`typing.dataclass_transform`), so static type checkers understand the
    ``alias`` and ``default`` arguments. Like :func:`attrs.field`, it is typed as
    returning the field's *value* type, so that it can be used as the default of an
    annotated attribute.
    """
    metadata = {"array": True}
    if default is None:
        return attrs.field(
            default=None,
            validator=attrs.validators.optional(attrs.validators.instance_of(Array)),
            eq=False,
            alias=alias,
            metadata=metadata,
        )
    return attrs.field(
        validator=attrs.validators.instance_of(Array),
        eq=False,
        alias=alias,
        metadata=metadata,
    )


class _HashType(Enum):
    user_cosmo = 0
    zgrid = 1
    full = 2


@cache
def _struct_params(compute: Callable[..., object]) -> dict[str, type[OutputStruct]]:
    """Return the parameters of a ``compute`` method that are annotated as structs."""
    return {
        name: tp
        for name, tp in get_type_hints(compute).items()
        if isinstance(tp, type) and issubclass(tp, OutputStruct)
    }


def _require_struct_inputs[S: OutputStruct, **P, R](
    compute: Callable[Concatenate[S, P], R],
) -> Callable[Concatenate[S, P], R]:
    """Wrap a ``compute`` method so that None for any OutputStruct input raises.

    Single-field functions accept None for boxes that are not needed when the output is
    read from the cache, or not needed for the given redshift and parameters (in which
    case they substitute a dummy). By the time ``compute`` is called, every struct input
    must be a real (or dummy) box: passing None to the C code is never valid.
    """

    @wraps(compute)
    def wrapper(self: S, *args: P.args, **kwargs: P.kwargs) -> R:
        params = _struct_params(compute)
        if missing := [
            f"{k} ({params[k].__name__})"
            for k, v in kwargs.items()
            if v is None and k in params
        ]:
            raise ValueError(
                f"Cannot compute {self._name} at z={getattr(self, 'redshift', None)} "
                f"(Z_HEAT_MAX={self.inputs.simulation_options.Z_HEAT_MAX}): required "
                f"input(s) {', '.join(missing)} were None. Such boxes may only be "
                "omitted when the output is read from the cache, or when they are not "
                "needed for this redshift and set of parameters."
            )
        return compute(self, *args, **kwargs)

    return wrapper


@dataclass_transform(kw_only_default=True, field_specifiers=(attrs.field, _arrayfield))
def _output_struct[S: OutputStruct](cls: type[S]) -> type[S]:
    """Class decorator defining a concrete :class:`OutputStruct` (sub)class.

    This is ``attrs.define(slots=False, kw_only=True)``, followed by
    :func:`~.arrays.expose_arrays` to register the public names of its array fields.
    It is a single decorator (rather than stacking the two) so that it can declare
    :func:`_arrayfield` as a field specifier, which tells static type checkers the
    ``__init__`` arguments (and their defaults) of the array fields.
    """
    return expose_arrays(attrs.define(slots=False, kw_only=True)(cls))


@attrs.define(slots=False, kw_only=True)
class OutputStruct(ABC):
    """Base class for any class that wraps a C struct meant to be output from a C function."""

    #: Whether this is an intermediate (abstract) class, not registered as an output.
    _meta: ClassVar[bool] = False
    #: The C function filling this struct (``None`` for intermediate classes).
    _c_compute_function: ClassVar[CFunction | None] = None
    _compat_hash: ClassVar[_HashType] = _HashType.full

    _TYPEMAP: ClassVar[bidict[str, str]] = bidict(
        {"float32": "float *", "float64": "double *", "int32": "int *"}
    )

    inputs: InputParameters = attrs.field(
        validator=attrs.validators.instance_of(InputParameters)
    )
    dummy: bool = attrs.field(default=False, converter=bool)
    initial: bool = attrs.field(default=False, converter=bool)

    @property
    def _name(self) -> str:
        """The name of the struct."""
        return self.__class__.__name__

    def __init_subclass__(cls) -> None:
        """Store subclasses for easy access, and guard their ``compute`` methods."""
        if not cls._meta:
            _ALL_OUTPUT_STRUCTS[cls.__name__] = cls

        if "compute" in cls.__dict__:
            cls.compute = _require_struct_inputs(cls.compute)

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

    #: Public names of this class's Array fields, filled in by `expose_arrays`.
    #: ClassVar, so that attrs treats it as a plain class attribute rather than a field.
    _array_field_names: ClassVar[tuple[str, ...]] = ()

    @property
    def arrays(self) -> dict[str, Array]:
        """A dictionary of Array objects whose memory is shared between this object and the C backend.

        This is the handle for anything to do with an array's *memory* - its state,
        its cache backend, and the functional transitions between them. Reading the
        attribute of the same name off this object gives you the array's *data*, as a
        plain numpy array.
        """
        out = {}
        for name in self._array_field_names:
            array = getattr(self, f"_{name}")
            if array is not None:
                out[name] = array
        return out

    @cached_property
    def _struct(self) -> StructWrapper:
        """The python-wrapped struct associated with this input object."""
        return StructWrapper(self._name)

    @cached_property
    def _cstruct(self) -> Any:
        """The cffi pointer to the memory accessed by C-code for this struct."""
        return self._struct.cstruct

    def _init_arrays(self) -> None:
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

    def push_to_backend(self) -> None:
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

    def pull_from_backend(self) -> None:
        """Sync the current state of the object with the underlying C-struct.

        This will pull any primitives calculated in the backend to the python object.
        Arrays are passed in as pointers, and do not need to be copied back.
        """
        for k in self._struct.primitive_fields:
            setattr(self, k, getattr(self._cstruct, k))

    @overload
    def get(self, ary: Array) -> np.ndarray: ...

    @overload
    def get(self, ary: str) -> Any: ...

    def get(self, ary: str | Array) -> Any:
        """If possible, load an array from disk, storing it and returning the underlying array.

        ``ary`` may be an `Array` of this instance, or the name of one. For backwards
        compatibility, it may also be the name of any other attribute, whose value is
        then returned (hence the return type ``Any`` when given a name).
        """
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

        return ary._resolve_value()

    def set(self, name: str, value: Any) -> None:
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
    ) -> None:
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
        if flush is None:
            if keep is None:
                raise ValueError("Must provide either flush or keep")
            flush = [
                k
                for k, array in self.arrays.items()
                if k not in keep and array.state.initialized
            ]
        elif keep is None:
            keep = [k for k in self.arrays if k not in flush]

        for k in flush:
            self._remove_array(k, force=force)

        # For everything we want to keep, we check if it is computed in memory,
        # and if not, load it from disk.
        for k in keep:
            self.get(k)

    def _remove_array(self, k: str, *, force: bool = False) -> None:
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

    def purge(self, force: bool = False) -> None:
        """Flush all the boxes out of memory.

        Parameters
        ----------
        force
            Whether to force the purge even if no disk storage exists.
        """
        self.prepare(keep=[], force=force)

    def has(self, name: str) -> bool:
        """Whether the named array currently has data available, in memory or on disk.

        Reading the attribute of the same name is only meaningful when this is true;
        otherwise it raises, because there is no data anywhere to give you. Use this
        rather than testing the attribute against None: an array that exists but has
        never been computed is not the same thing as an optional array that this set
        of inputs doesn't produce (for which the attribute *is* None).
        """
        array = self.arrays.get(name)
        return array is not None and (
            array.state.computed_in_mem or array.state.on_disk
        )

    def load_all(self) -> None:
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

    def ensure_arrays_computed(self, *arrays: str, load: bool = False) -> bool:
        """Check if the given arrays are computed (not just initialized)."""
        if not self.is_computed:
            return False

        computed = all(self.arrays[k].state.is_computed for k in arrays)

        if computed and load:
            self.prepare(keep=arrays, flush=[])

        return computed

    def ensure_arrays_inited(self, *arrays: str, init: bool = False) -> bool:
        """Check if the given arrays are initialized (or computed)."""
        inited = all(self.arrays[k].state.initialized for k in arrays)

        if init and not inited:
            self._init_arrays()
        return True

    @abstractmethod
    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""

    def ensure_input_computed(
        self, input_box: OutputStruct, load: bool = False
    ) -> bool:
        """Ensure all the inputs have been computed."""
        if input_box.dummy:
            return True

        arrays = self.get_required_input_arrays(input_box)
        if input_box.initial:
            return input_box.ensure_arrays_inited(*arrays, init=load)

        return input_box.ensure_arrays_computed(*arrays, load=load)

    def summarize(self, indent: int = 0) -> str:
        """Generate a string summary of the struct."""
        pad = indent * "    "

        # print array type and column headings
        out = (
            f"\n{pad}{self.__class__.__name__:>25}    "
            + "   1st:         End:         Min:         Max:         Mean:         \n"
        )

        # print array extrema and means
        for fieldname, array in self.arrays.items():
            state = array.state
            if not state.initialized:
                out += f"{pad}    {fieldname:>25}:  uninitialized\n"
            elif not state.is_computed:
                out += f"{pad}    {fieldname:>25}:  initialized\n"
            elif not state.computed_in_mem:
                out += f"{pad}    {fieldname:>25}:  computed on disk\n"
            else:
                x = self.get(fieldname).flatten()
                if len(x) > 0:
                    out += f"{pad}    {fieldname:>25}: {x[0]:11.4e}, {x[-1]:11.4e}, {x.min():11.4e}, {x.max():11.4e}, {np.mean(x):11.4e}\n"
                else:
                    out += f"{pad}    {fieldname:>25}: size zero\n"

        # print primitive fields
        out += "".join(
            f"{pad}    {fieldname:>25}: {getattr(self, fieldname, 'non-existent')}\n"
            for fieldname in self._struct.primitive_fields
        )

        return out

    @classmethod
    def _get_c_compute_function(cls) -> CFunction:
        """Return the C function that fills this struct."""
        fnc = cls._c_compute_function
        if fnc is None:
            raise TypeError(f"{cls.__name__} is not computed by a C function.")
        return fnc

    @classmethod
    def _log_call_arguments(cls, *args: Any) -> None:
        logger.debug(
            f"Calling {cls._get_c_compute_function().__name__} with following args:"
        )

        for arg in args:
            if isinstance(arg, OutputStruct):
                for line in arg.summarize(indent=1).split("\n"):
                    logger.debug(line)
            elif isinstance(arg, InputStruct):
                for line in str(arg).split("\n"):
                    logger.debug(f"    {line}")
            else:
                logger.debug(f"    {arg}")

    def _ensure_arguments_exist(self, *args: Any) -> None:
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

    def _compute(
        self,
        allow_already_computed: bool = False,
        *args: OutputStruct | InputStruct | float,
    ) -> Self:
        """Compute the actual function that fills this struct.

        ``args`` are the arguments of the C function (excluding this struct, which is
        always passed last), in order.
        """
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
        c_compute_function = self._get_c_compute_function()
        try:
            exitcode = c_compute_function(*inputs, self._cstruct)
        except TypeError:
            logger.error(f"Arguments to {c_compute_function.__name__}: {inputs}")
            raise

        _process_exitcode(exitcode, c_compute_function, args)

        for name, array in self.arrays.items():
            setattr(self, name, array.computed())

        self.pull_from_backend()
        return self

    @classmethod
    @abstractmethod
    def new(cls, inputs: InputParameters, *args: Any, **kwargs: Any) -> Self:
        """Instantiate the class from InputParameters.

        Subclasses define which further arguments they require (e.g. ``redshift``).
        Any other keyword arguments are passed through to the constructor.
        """

    def get_full_size(self) -> int:
        """Return the size of the object in bytes.

        This represents the size of the object if it is fully initialized/computed
        and all in memory. Equivalently, it is close to the file size on disk.
        """
        size = 0
        for ary in self.arrays.values():
            size += int(np.prod(ary.shape)) * np.dtype(ary.dtype).itemsize
        return size


@_output_struct
class InitialConditions(OutputStruct):
    """A class representing an InitialConditions C-struct."""

    _c_compute_function = lib.ComputeInitialConditions
    _meta = False
    _compat_hash = _HashType.user_cosmo

    _lowres_density: Array = _arrayfield(alias="lowres_density")
    lowres_density = ArrayProxy[np.ndarray]()
    _lowres_vx: Array | None = _arrayfield(alias="lowres_vx", default=None)
    lowres_vx = ArrayProxy[np.ndarray | None]()
    _lowres_vy: Array | None = _arrayfield(alias="lowres_vy", default=None)
    lowres_vy = ArrayProxy[np.ndarray | None]()
    _lowres_vz: Array | None = _arrayfield(alias="lowres_vz", default=None)
    lowres_vz = ArrayProxy[np.ndarray | None]()
    _hires_density: Array = _arrayfield(alias="hires_density")
    hires_density = ArrayProxy[np.ndarray]()
    _hires_vx: Array | None = _arrayfield(alias="hires_vx", default=None)
    hires_vx = ArrayProxy[np.ndarray | None]()
    _hires_vy: Array | None = _arrayfield(alias="hires_vy", default=None)
    hires_vy = ArrayProxy[np.ndarray | None]()
    _hires_vz: Array | None = _arrayfield(alias="hires_vz", default=None)
    hires_vz = ArrayProxy[np.ndarray | None]()

    _lowres_vx_2LPT: Array | None = _arrayfield(alias="lowres_vx_2LPT", default=None)
    lowres_vx_2LPT = ArrayProxy[np.ndarray | None]()
    _lowres_vy_2LPT: Array | None = _arrayfield(alias="lowres_vy_2LPT", default=None)
    lowres_vy_2LPT = ArrayProxy[np.ndarray | None]()
    _lowres_vz_2LPT: Array | None = _arrayfield(alias="lowres_vz_2LPT", default=None)
    lowres_vz_2LPT = ArrayProxy[np.ndarray | None]()
    _hires_vx_2LPT: Array | None = _arrayfield(alias="hires_vx_2LPT", default=None)
    hires_vx_2LPT = ArrayProxy[np.ndarray | None]()
    _hires_vy_2LPT: Array | None = _arrayfield(alias="hires_vy_2LPT", default=None)
    hires_vy_2LPT = ArrayProxy[np.ndarray | None]()
    _hires_vz_2LPT: Array | None = _arrayfield(alias="hires_vz_2LPT", default=None)
    hires_vz_2LPT = ArrayProxy[np.ndarray | None]()

    _lowres_vcb: Array | None = _arrayfield(alias="lowres_vcb", default=None)
    lowres_vcb = ArrayProxy[np.ndarray | None]()

    @classmethod
    def new(cls, inputs: InputParameters, **kw: Any) -> Self:
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

    # The instance attribute `dummy` (a bool field of OutputStruct) and this classmethod
    # share a name by design: the field is set on every instance, so it shadows the
    # classmethod there, while `InitialConditions.dummy()` reaches the classmethod.
    # Type checkers can't model one name having different types on the class and its
    # instances, and both are public API.
    @classmethod
    def dummy(cls) -> Self:  # ty: ignore[invalid-attribute-override]
        """Create a dummy instance, for passing to C when the ICs are not needed."""
        return cls.new(inputs=InputParameters(random_seed=1), dummy=True)

    def prepare_for_perturb(self, force: bool = False) -> None:
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

    def prepare_for_spin_temp(self, force: bool = False) -> None:
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

    def compute(self, allow_already_computed: bool = False) -> Self:
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.random_seed,
        )


@_output_struct
class OutputStructZ(OutputStruct):
    """The same as an OutputStruct, but containing a redshift."""

    _meta = True
    redshift: float = attrs.field(converter=float)

    @classmethod
    @abstractmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw: Any) -> Self:
        """Instantiate the class from InputParameters, at a given redshift."""

    # See InitialConditions.dummy for why these classmethods shadow the bool fields
    # `dummy` and `initial` of OutputStruct.
    @classmethod
    def dummy(cls) -> Self:  # ty: ignore[invalid-attribute-override]
        """Create a dummy instance with the given inputs."""
        return cls.new(inputs=InputParameters(random_seed=1), redshift=-1.0, dummy=True)

    @classmethod
    def initial(  # ty: ignore[invalid-attribute-override]
        cls, inputs: InputParameters
    ) -> Self:
        """Create an instance representing the initial state, with the given inputs."""
        return cls.new(inputs=inputs, redshift=-1.0, initial=True)


@_output_struct
class PerturbedField(OutputStructZ):
    """A class containing all perturbed field boxes."""

    _c_compute_function = lib.ComputePerturbedField
    _meta = False
    _compat_hash = _HashType.zgrid

    _density: Array = _arrayfield(alias="density")
    density = ArrayProxy[np.ndarray]()
    _velocity_z: Array | None = _arrayfield(alias="velocity_z", default=None)
    velocity_z = ArrayProxy[np.ndarray | None]()
    _velocity_x: Array | None = _arrayfield(alias="velocity_x", default=None)
    velocity_x = ArrayProxy[np.ndarray | None]()
    _velocity_y: Array | None = _arrayfield(alias="velocity_y", default=None)
    velocity_y = ArrayProxy[np.ndarray | None]()

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw: Any) -> Self:
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
            # Kept as ValueError (not TypeError): part of the public API contract,
            # asserted verbatim by
            # tests/test_output_structs.py::test_bad_required_array.
            raise ValueError(  # noqa: TRY004
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

    def compute(
        self, *, allow_already_computed: bool = False, ics: InitialConditions
    ) -> Self:
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.redshift,
            ics,
        )

    @property
    def velocity(self) -> np.ndarray | None:
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


@_output_struct
class HaloCatalog(OutputStructZ):
    """A class containing all fields related to halos."""

    _c_compute_function = lib.ComputeHaloCatalog
    _meta = False
    desc_redshift: float | None = attrs.field(default=None)
    _compat_hash = _HashType.zgrid

    _halo_masses: Array = _arrayfield(alias="halo_masses")
    halo_masses = ArrayProxy[np.ndarray]()
    _star_rng: Array = _arrayfield(alias="star_rng")
    star_rng = ArrayProxy[np.ndarray]()
    _sfr_rng: Array = _arrayfield(alias="sfr_rng")
    sfr_rng = ArrayProxy[np.ndarray]()
    _xray_rng: Array = _arrayfield(alias="xray_rng")
    xray_rng = ArrayProxy[np.ndarray]()
    _halo_coords: Array = _arrayfield(alias="halo_coords")
    halo_coords = ArrayProxy[np.ndarray]()
    n_halos: int = attrs.field(default=None)
    buffer_size: int = attrs.field(default=None)

    @classmethod
    def new(
        cls,
        inputs: InputParameters,
        redshift: float,
        buffer_size: int | None = None,
        **kw: Any,
    ) -> Self:
        """Create a new HaloCatalog instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.
        buffer_size : int, optional
            The number of halos the catalog can hold. By default, this is estimated
            from the inputs (or zero for a dummy instance).

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
        return self

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, InitialConditions):
            if self.matter_options.SOURCE_MODEL == "CHMF-SAMPLER":
                # when the sampler is on, the grids are only needed for the first
                # sample (which has no descendant catalog).
                if self.desc_redshift is None or self.desc_redshift <= 0:
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
            # Kept as ValueError (not TypeError): part of the public API contract,
            # asserted verbatim by
            # tests/test_output_structs.py::test_bad_required_array.
            raise ValueError(  # noqa: TRY004
                f"{type(input_box)} is not an input required for HaloCatalog!"
            )
        return required

    def compute(
        self,
        *,
        descendant_halos: HaloCatalog,
        ics: InitialConditions,
        allow_already_computed: bool = False,
    ) -> Self:
        """Compute the function."""
        if self.desc_redshift is None:
            raise ValueError(
                "desc_redshift must be set to compute a HaloCatalog: it is the "
                "redshift of descendant_halos (or -1 if there is no descendant)."
            )
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
            mass=float(self.halo_masses[index]),
            coords=self.halo_coords[index].copy(),
            redshift=self.redshift,
            star_rng=float(self.star_rng[index]),
            sfr_rng=float(self.sfr_rng[index]),
            xray_rng=float(self.xray_rng[index]),
        )

    def __iter__(self) -> Iterator[Halo]:
        """Iterate over halos in the catalog, yielding Halo objects."""
        for i in range(self.n_halos):
            yield self[i]

    def __len__(self) -> int:
        """Return the number of halos in the catalog."""
        return self.n_halos


@_output_struct
class PerturbedHaloCatalog(OutputStructZ):
    """A class to hold a HaloCatalog whose coordinates are in real (Eulerian) space."""

    _c_compute_function = lib.ComputePerturbedHaloCatalog
    _meta = False
    desc_redshift: float | None = attrs.field(default=None)
    _compat_hash = _HashType.zgrid

    _halo_masses: Array = _arrayfield(alias="halo_masses")
    halo_masses = ArrayProxy[np.ndarray]()
    _halo_coords: Array = _arrayfield(alias="halo_coords")
    halo_coords = ArrayProxy[np.ndarray]()

    _sfr_acg: Array = _arrayfield(alias="sfr_acg")
    sfr_acg = ArrayProxy[np.ndarray]()
    _stellar_masses_acg: Array = _arrayfield(alias="stellar_masses_acg")
    stellar_masses_acg = ArrayProxy[np.ndarray]()
    _n_ion: Array = _arrayfield(alias="n_ion")
    n_ion = ArrayProxy[np.ndarray]()
    _xray_luminosity: Array | None = _arrayfield(alias="xray_luminosity", default=None)
    xray_luminosity = ArrayProxy[np.ndarray | None]()
    _fesc_weighted_sfr: Array | None = _arrayfield(
        alias="fesc_weighted_sfr", default=None
    )
    fesc_weighted_sfr = ArrayProxy[np.ndarray | None]()

    _stellar_masses_mcg: Array | None = _arrayfield(
        alias="stellar_masses_mcg", default=None
    )
    stellar_masses_mcg = ArrayProxy[np.ndarray | None]()
    _sfr_mcg: Array | None = _arrayfield(alias="sfr_mcg", default=None)
    sfr_mcg = ArrayProxy[np.ndarray | None]()

    n_halos: int = attrs.field(default=None)
    buffer_size: int = attrs.field(default=None)

    @classmethod
    def new(
        cls,
        inputs: InputParameters,
        redshift: float,
        buffer_size: int | None = None,
        **kw: Any,
    ) -> Self:
        """Create a new PerturbedHaloCatalog instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.
        buffer_size : int, optional
            The number of halos the catalog can hold. By default, this is estimated
            from the inputs (or zero for a dummy instance).

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

        shape = (buffer_size,)
        astro_options = inputs.astro_options

        return cls(
            inputs=inputs,
            redshift=redshift,
            buffer_size=buffer_size,
            halo_coords=Array((buffer_size, 3), dtype=np.float32),
            halo_masses=Array(shape, dtype=np.float32),
            stellar_masses_acg=Array(shape, dtype=np.float32),
            sfr_acg=Array(shape, dtype=np.float32),
            n_ion=Array(shape, dtype=np.float32),
            xray_luminosity=(
                Array(shape, dtype=np.float32) if astro_options.USE_TS_FLUCT else None
            ),
            fesc_weighted_sfr=(
                Array(shape, dtype=np.float32)
                if astro_options.RECOMB_MODEL != "none"
                else None
            ),
            stellar_masses_mcg=(
                Array(shape, dtype=np.float32) if astro_options.USE_MCGS else None
            ),
            sfr_mcg=Array(shape, dtype=np.float32) if astro_options.USE_MCGS else None,
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
            if self.astro_options.USE_MCGS:
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
            # Kept as ValueError (not TypeError): part of the public API contract,
            # asserted verbatim by
            # tests/test_output_structs.py::test_bad_required_array.
            raise ValueError(  # noqa: TRY004
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
    ) -> Self:
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
            mass=float(self.halo_masses[index]),
            coords=self.halo_coords[index].copy(),
            redshift=self.redshift,
        )

    def __iter__(self) -> Iterator[Halo]:
        """Iterate over halos in the catalog, yielding Halo objects."""
        for i in range(self.n_halos):
            yield self[i]

    def __len__(self) -> int:
        """Return the number of halos in the catalog."""
        return self.n_halos


@_output_struct
class EmissivityFields(OutputStructZ):
    """A class containing all gridded halo properties."""

    _meta = False
    _c_compute_function = lib.ComputeEmissivityFields

    _halo_number: Array | None = _arrayfield(alias="halo_number", default=None)
    halo_number = ArrayProxy[np.ndarray | None]()
    _halo_mass_density: Array | None = _arrayfield(
        alias="halo_mass_density", default=None
    )
    halo_mass_density = ArrayProxy[np.ndarray | None]()
    _stellar_mass_density_acg: Array | None = _arrayfield(
        alias="stellar_mass_density_acg", default=None
    )
    stellar_mass_density_acg = ArrayProxy[np.ndarray | None]()
    _stellar_mass_density_mcg: Array | None = _arrayfield(
        alias="stellar_mass_density_mcg", default=None
    )
    stellar_mass_density_mcg = ArrayProxy[np.ndarray | None]()
    _sfrd_acg: Array | None = _arrayfield(alias="sfrd_acg", default=None)
    sfrd_acg = ArrayProxy[np.ndarray | None]()
    _sfrd_mcg: Array | None = _arrayfield(alias="sfrd_mcg", default=None)
    sfrd_mcg = ArrayProxy[np.ndarray | None]()
    _xray_emissivity: Array | None = _arrayfield(alias="xray_emissivity", default=None)
    xray_emissivity = ArrayProxy[np.ndarray | None]()
    _n_ion: Array = _arrayfield(alias="n_ion")
    n_ion = ArrayProxy[np.ndarray]()
    _fesc_weighted_sfrd: Array | None = _arrayfield(
        alias="fesc_weighted_sfrd", default=None
    )
    fesc_weighted_sfrd = ArrayProxy[np.ndarray | None]()

    log10_mturn_acg_ave: float = attrs.field(default=None)
    log10_mturn_mcg_ave: float = attrs.field(default=None)

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw: Any) -> Self:
        """Create a new EmissivityFields instance with the given inputs.

        Parameters
        ----------
        inputs : InputParameters
            The input parameters defining the output struct.
        redshift : float
            The redshift at which to compute fields.

        Other Parameters
        ----------------
        All other parameters are passed through to the :class:`EmissivityFields`
        constructor.
        """
        dim = inputs.simulation_options.HII_DIM
        shape = (dim, dim, int(inputs.simulation_options.NON_CUBIC_FACTOR * dim))

        astro_options = inputs.astro_options
        use_ts_fluct = astro_options.USE_TS_FLUCT
        use_mcgs = astro_options.USE_MCGS
        extra = bool(config["EXTRA_EMISSIVITY_FIELDS"])

        def grid(needed: bool) -> Array | None:
            return Array(shape, dtype=np.float32) if needed else None

        return cls(
            inputs=inputs,
            redshift=redshift,
            # TODO: right now it's a waste of memory and time to compute n_ion for Eulerian source models,
            # this might change in the future
            n_ion=Array(shape, dtype=np.float32),
            # TODO: similarly, as above, fesc_weighted_sfrd is only needed for Lagrangian source models at the moment, but this might change
            fesc_weighted_sfrd=grid(
                astro_options.RECOMB_MODEL != "none"
                and inputs.matter_options.lagrangian_source_grid
            ),
            xray_emissivity=grid(use_ts_fluct),
            sfrd_acg=grid(use_ts_fluct),
            sfrd_mcg=grid(use_ts_fluct and use_mcgs),
            halo_number=grid(extra),
            halo_mass_density=grid(extra),
            stellar_mass_density_acg=grid(extra),
            stellar_mass_density_mcg=grid(extra and use_mcgs),
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
            if self.astro_options.USE_MCGS:
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
                and self.astro_options.USE_MCGS
            ):
                required += ["lowres_vcb"]
        else:
            raise TypeError(
                f"{type(input_box)} is not an input required for EmissivityFields!"
            )

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
    ) -> Self:
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

    def prepare_for_next_snapshot(self, next_z: float, force: bool = False) -> None:
        """Prepare the EmissivityFields for the next snapshot."""
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
                keep += ["sfrd_acg", "xray_emissivity"]
                if self.astro_options.USE_MCGS:
                    keep += ["sfrd_mcg"]
        self.prepare(keep=keep, force=force)

    @property
    def count(self) -> np.ndarray | None:
        """The halo number (per cell).

        This property is deprecated and will be removed in a future version.
        Please use `halo_number` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "count",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="count is deprecated and will be removed in a future version. "
                "Please use halo_number directly instead.",
            ),
            stacklevel=2,
        )
        return self.halo_number

    @property
    def halo_mass(self) -> np.ndarray | None:
        """The halo mass density field.

        This property is deprecated and will be removed in a future version.
        Please use `halo_mass_density` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "halo_mass",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="halo_mass is deprecated and will be removed in a future version. "
                "Please use halo_mass_density directly instead.",
            ),
            stacklevel=2,
        )
        return self.halo_mass_density

    @property
    def halo_stars(self) -> np.ndarray | None:
        """The stellar mass density field in ACGs.

        This property is deprecated and will be removed in a future version.
        Please use `stellar_mass_density_acg` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "halo_stars",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="halo_stars is deprecated and will be removed in a future version. "
                "Please use stellar_mass_density_acg directly instead.",
            ),
            stacklevel=2,
        )
        return self.stellar_mass_density_acg

    @property
    def halo_stars_mini(self) -> np.ndarray | None:
        """The stellar mass density field in MCGs.

        This property is deprecated and will be removed in a future version.
        Please use `stellar_mass_density_mcg` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "halo_stars_mini",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="halo_stars_mini is deprecated and will be removed in a future version. "
                "Please use stellar_mass_density_mcg directly instead.",
            ),
            stacklevel=2,
        )
        return self.stellar_mass_density_mcg

    @property
    def halo_sfr(self) -> np.ndarray | None:
        """The star formation rate density field in ACGs.

        This property is deprecated and will be removed in a future version.
        Please use `sfrd_acg` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "halo_sfr",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="halo_sfr is deprecated and will be removed in a future version. "
                "Please use sfrd_acg directly instead.",
            ),
            stacklevel=2,
        )
        return self.sfrd_acg

    @property
    def halo_sfr_mini(self) -> np.ndarray | None:
        """The star formation rate density field in MCGs.

        This property is deprecated and will be removed in a future version.
        Please use `sfrd_mcg` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "halo_sfr_mini",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="halo_sfr_mini is deprecated and will be removed in a future version. "
                "Please use sfrd_mcg directly instead.",
            ),
            stacklevel=2,
        )
        return self.sfrd_mcg

    @property
    def halo_xray(self) -> np.ndarray | None:
        """The X-ray emissivity field.

        This property is deprecated and will be removed in a future version.
        Please use `xray_emissivity` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "halo_xray",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="halo_xray is deprecated and will be removed in a future version. "
                "Please use xray_emissivity directly instead.",
            ),
            stacklevel=2,
        )
        return self.xray_emissivity

    @property
    def whalo_sfr(self) -> np.ndarray | None:
        """The star formation rate density field, weighted by escape fraction, in both ACGs and MCGs.

        This property is deprecated and will be removed in a future version.
        Please use `fesc_weighted_sfrd` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "whalo_sfr",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="whalo_sfr is deprecated and will be removed in a future version. "
                "Please use fesc_weighted_sfrd directly instead.",
            ),
            stacklevel=2,
        )
        return self.fesc_weighted_sfrd

    @property
    def log10_Mcrit_ACG_ave(self) -> float:
        """The average log10 of the turnover mass for ACGs.

        This property is deprecated and will be removed in a future version.
        Please use `log10_mturn_acg_ave` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "log10_Mcrit_ACG_ave",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="log10_Mcrit_ACG_ave is deprecated and will be removed in a future version. "
                "Please use log10_mturn_acg_ave directly instead.",
            ),
            stacklevel=2,
        )
        return self.log10_mturn_acg_ave

    @property
    def log10_Mcrit_MCG_ave(self) -> float:
        """The average log10 of the turnover mass for MCGs.

        This property is deprecated and will be removed in a future version.
        Please use `log10_mturn_mcg_ave` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "log10_Mcrit_MCG_ave",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="log10_Mcrit_MCG_ave is deprecated and will be removed in a future version. "
                "Please use log10_mturn_mcg_ave directly instead.",
            ),
            stacklevel=2,
        )
        return self.log10_mturn_mcg_ave


@_output_struct
class HaloBox(EmissivityFields):
    """Deprecated alias for :class:`EmissivityFields`.

    .. deprecated:: 4.3.0
        ``HaloBox`` was renamed to :class:`EmissivityFields`. This alias will be
        removed in v5.0.0.
    """

    _meta = True  # don't register as a separate output-struct kind

    def __attrs_post_init__(self) -> None:
        """Post-init.

        Warns about deprecation of HaloBox.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "HaloBox",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details=(
                    "HaloBox has been renamed to EmissivityFields. "
                    "Please use EmissivityFields instead."
                ),
            ),
            stacklevel=2,
        )


@_output_struct
class RadiationFieldsSetup(OutputStructZ):
    """A class containing the fields that are neccesary for setting up the radiation fields."""

    _meta = False
    _c_compute_function = lib.SetupRadiationFields

    # R-dependent arrays which are set once
    _zpp_avg: Array = _arrayfield(alias="zpp_avg")
    zpp_avg = ArrayProxy[np.ndarray]()
    _R_values: Array = _arrayfield(alias="R_values")
    R_values = ArrayProxy[np.ndarray]()
    _zpp_edges: Array = _arrayfield(alias="zpp_edges")
    zpp_edges = ArrayProxy[np.ndarray]()
    # Arrays for the filtered emissivity fields
    _filtered_sfrd_acg_for_lya: Array = _arrayfield(alias="filtered_sfrd_acg_for_lya")
    filtered_sfrd_acg_for_lya = ArrayProxy[np.ndarray]()
    _filtered_sfrd_mcg_for_lya: Array | None = _arrayfield(
        alias="filtered_sfrd_mcg_for_lya", default=None
    )
    filtered_sfrd_mcg_for_lya = ArrayProxy[np.ndarray | None]()
    _filtered_xray_emissivity: Array = _arrayfield(alias="filtered_xray_emissivity")
    filtered_xray_emissivity = ArrayProxy[np.ndarray]()
    _filtered_sfrd_acg_for_lw: Array | None = _arrayfield(
        alias="filtered_sfrd_acg_for_lw", default=None
    )
    filtered_sfrd_acg_for_lw = ArrayProxy[np.ndarray | None]()
    _filtered_sfrd_mcg_for_lw: Array | None = _arrayfield(
        alias="filtered_sfrd_mcg_for_lw", default=None
    )
    filtered_sfrd_mcg_for_lw = ArrayProxy[np.ndarray | None]()
    # Frequency integral tables
    _freq_int_heat_tbl: Array = _arrayfield(alias="freq_int_heat_tbl")
    freq_int_heat_tbl = ArrayProxy[np.ndarray]()
    _freq_int_ion_tbl: Array = _arrayfield(alias="freq_int_ion_tbl")
    freq_int_ion_tbl = ArrayProxy[np.ndarray]()
    _freq_int_lya_tbl: Array = _arrayfield(alias="freq_int_lya_tbl")
    freq_int_lya_tbl = ArrayProxy[np.ndarray]()
    _freq_int_heat_tbl_diff: Array = _arrayfield(alias="freq_int_heat_tbl_diff")
    freq_int_heat_tbl_diff = ArrayProxy[np.ndarray]()
    _freq_int_ion_tbl_diff: Array = _arrayfield(alias="freq_int_ion_tbl_diff")
    freq_int_ion_tbl_diff = ArrayProxy[np.ndarray]()
    _freq_int_lya_tbl_diff: Array = _arrayfield(alias="freq_int_lya_tbl_diff")
    freq_int_lya_tbl_diff = ArrayProxy[np.ndarray]()
    # helpers for the interpolation
    _inverse_diff: Array = _arrayfield(alias="inverse_diff")
    inverse_diff = ArrayProxy[np.ndarray]()
    _m_xHII_low_box: Array = _arrayfield(alias="m_xHII_low_box")
    m_xHII_low_box = ArrayProxy[np.ndarray]()
    _inverse_val_box: Array = _arrayfield(alias="inverse_val_box")
    inverse_val_box = ArrayProxy[np.ndarray]()
    # arrays for R-dependent prefactors
    _lya_flux_continuum_prefactor_acg: Array | None = _arrayfield(
        alias="lya_flux_continuum_prefactor_acg", default=None
    )
    lya_flux_continuum_prefactor_acg = ArrayProxy[np.ndarray | None]()
    _lya_flux_injected_prefactor_acg: Array | None = _arrayfield(
        alias="lya_flux_injected_prefactor_acg", default=None
    )
    lya_flux_injected_prefactor_acg = ArrayProxy[np.ndarray | None]()
    _lya_flux_continuum_injected_prefactor_acg: Array | None = _arrayfield(
        alias="lya_flux_continuum_injected_prefactor_acg", default=None
    )
    lya_flux_continuum_injected_prefactor_acg = ArrayProxy[np.ndarray | None]()
    _lyw_flux_prefactor_acg: Array | None = _arrayfield(
        alias="lyw_flux_prefactor_acg", default=None
    )
    lyw_flux_prefactor_acg = ArrayProxy[np.ndarray | None]()
    _lyw_flux_prefactor_mcg: Array | None = _arrayfield(
        alias="lyw_flux_prefactor_mcg", default=None
    )
    lyw_flux_prefactor_mcg = ArrayProxy[np.ndarray | None]()
    _lya_flux_continuum_prefactor_mcg: Array | None = _arrayfield(
        alias="lya_flux_continuum_prefactor_mcg", default=None
    )
    lya_flux_continuum_prefactor_mcg = ArrayProxy[np.ndarray | None]()
    _lya_flux_injected_prefactor_mcg: Array | None = _arrayfield(
        alias="lya_flux_injected_prefactor_mcg", default=None
    )
    lya_flux_injected_prefactor_mcg = ArrayProxy[np.ndarray | None]()
    _lya_flux_continuum_injected_prefactor_mcg: Array | None = _arrayfield(
        alias="lya_flux_continuum_injected_prefactor_mcg", default=None
    )
    lya_flux_continuum_injected_prefactor_mcg = ArrayProxy[np.ndarray | None]()
    # array and floats required for the X-ray optical depth calculation
    _ave_log10_MturnLW: Array | None = _arrayfield(
        alias="ave_log10_MturnLW", default=None
    )
    ave_log10_MturnLW = ArrayProxy[np.ndarray | None]()
    Q_HI_zp: float = attrs.field(default=1.0)
    x_e_ave_zp: float = attrs.field(default=0.0)
    # boolean to indicate whether there's enough light
    NO_LIGHT: bool = attrs.field(default=True)
    # maximum source redshift for the radiation fields calculation
    source_z_max: float = attrs.field(default=0.0)
    # redshifts of the input emissivity_fields
    emissivity_fields_redshifts: list[float] = attrs.field(factory=list)

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw: Any) -> Self:
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

        n_r = inputs.astro_params.N_STEP_TS
        tbl_shape = (x_int_NXHII, n_r)
        use_mcgs = inputs.astro_options.USE_MCGS
        use_lya_heating = inputs.astro_options.USE_LYA_HEATING
        lya_multiple_scattering = inputs.astro_options.LYA_MULTIPLE_SCATTERING

        def box(needed: bool) -> Array | None:
            return Array(shape, dtype=np.float32) if needed else None

        def r_array(needed: bool) -> Array | None:
            return Array((n_r,), dtype=np.float64) if needed else None

        return cls(
            inputs=inputs,
            redshift=redshift,
            R_values=Array((n_r,), dtype=np.float64),
            zpp_avg=Array((n_r,), dtype=np.float64),
            zpp_edges=Array((n_r,), dtype=np.float64),
            freq_int_heat_tbl=Array(tbl_shape, dtype=np.float64),
            freq_int_ion_tbl=Array(tbl_shape, dtype=np.float64),
            freq_int_lya_tbl=Array(tbl_shape, dtype=np.float64),
            freq_int_heat_tbl_diff=Array(tbl_shape, dtype=np.float64),
            freq_int_ion_tbl_diff=Array(tbl_shape, dtype=np.float64),
            freq_int_lya_tbl_diff=Array(tbl_shape, dtype=np.float64),
            inverse_diff=Array((x_int_NXHII,), dtype=np.float32),
            m_xHII_low_box=Array(shape, dtype=np.int32),
            inverse_val_box=Array(shape, dtype=np.float32),
            filtered_sfrd_acg_for_lya=Array(shape, dtype=np.float32),
            filtered_xray_emissivity=Array(shape, dtype=np.float32),
            filtered_sfrd_mcg_for_lya=box(use_mcgs),
            filtered_sfrd_acg_for_lw=box(use_mcgs and lya_multiple_scattering),
            filtered_sfrd_mcg_for_lw=box(use_mcgs and lya_multiple_scattering),
            lya_flux_continuum_prefactor_acg=r_array(use_lya_heating),
            lya_flux_injected_prefactor_acg=r_array(use_lya_heating),
            lya_flux_continuum_injected_prefactor_acg=r_array(not use_lya_heating),
            lyw_flux_prefactor_acg=r_array(use_mcgs),
            lyw_flux_prefactor_mcg=r_array(use_mcgs),
            lya_flux_continuum_prefactor_mcg=r_array(use_mcgs and use_lya_heating),
            lya_flux_injected_prefactor_mcg=r_array(use_mcgs and use_lya_heating),
            lya_flux_continuum_injected_prefactor_mcg=r_array(
                use_mcgs and not use_lya_heating
            ),
            ave_log10_MturnLW=r_array(use_mcgs),
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
        # now we need to find the edges of the shells (comoving distance)
        cosmo_ap = inputs.cosmo_params.cosmo
        cmd_zp = cosmo_ap.comoving_distance(redshift)
        R_steps = np.arange(0, inputs.astro_params.N_STEP_TS)
        R_factor = (inputs.astro_params.R_MAX_TS / R_min) ** (
            R_steps / inputs.astro_params.N_STEP_TS
        )
        self.set("R_values", R_min * R_factor)
        cmd_edges = cmd_zp + self.R_values * un.Mpc  # comoving distance edges
        # Get the edges of the shells (redshift)
        zmin = z_at_value(cosmo_ap.comoving_distance, cmd_edges.min()).value
        zmax = z_at_value(cosmo_ap.comoving_distance, cmd_edges.max()).value
        zgrid = np.logspace(np.log10(zmin), np.log10(zmax), 100)
        dgrid = cosmo_ap.comoving_distance(zgrid)
        self.set("zpp_edges", np.interp(cmd_edges.value, dgrid.value, zgrid))
        # the `average` redshift of the shell is the average of the
        # inner and outer redshifts (following the C code)
        self.set(
            "zpp_avg",
            self.zpp_edges - np.diff(np.insert(self.zpp_edges, 0, redshift)) / 2,
        )
        return self

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, TsBox):
            required += ["xray_ionised_fraction"]
        else:
            raise TypeError(
                f"{type(input_box)} is not an input required for RadiationFieldsSetup!"
            )

        return required

    def compute(
        self,
        *,
        redshift: float,
        previous_spin_temp: TsBox,
        allow_already_computed: bool = False,
    ) -> Self:
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            redshift,
            previous_spin_temp,
        )


@_output_struct
class RadiationFields(OutputStructZ):
    """A class containing the radiation fields."""

    _meta = False
    _c_compute_function = lib.UpdateRadiationFields

    _xray_heating_rate: Array | None = _arrayfield(
        alias="xray_heating_rate", default=None
    )
    xray_heating_rate = ArrayProxy[np.ndarray | None]()
    _xray_ionization_rate: Array = _arrayfield(alias="xray_ionization_rate")
    xray_ionization_rate = ArrayProxy[np.ndarray]()
    _xray_lya_flux: Array = _arrayfield(alias="xray_lya_flux")
    xray_lya_flux = ArrayProxy[np.ndarray]()
    _lya_flux_continuum_injected: Array | None = _arrayfield(
        alias="lya_flux_continuum_injected", default=None
    )
    lya_flux_continuum_injected = ArrayProxy[np.ndarray | None]()
    _lyw_flux: Array | None = _arrayfield(alias="lyw_flux", default=None)
    lyw_flux = ArrayProxy[np.ndarray | None]()
    _lya_flux_continuum: Array | None = _arrayfield(
        alias="lya_flux_continuum", default=None
    )
    lya_flux_continuum = ArrayProxy[np.ndarray | None]()
    _lya_flux_injected: Array | None = _arrayfield(
        alias="lya_flux_injected", default=None
    )
    lya_flux_injected = ArrayProxy[np.ndarray | None]()
    Q_HI: float = attrs.field(default=1.0)

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw: Any) -> Self:
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

        astro_options = inputs.astro_options

        # TODO: the arrays below are defined as np.float64, but should be np.float32 - see https://github.com/21cmfast/21cmFAST/issues/744
        def box(needed: bool) -> Array | None:
            return Array(shape, dtype=np.float64) if needed else None

        return cls(
            inputs=inputs,
            redshift=redshift,
            xray_ionization_rate=Array(shape, dtype=np.float64),
            xray_lya_flux=Array(shape, dtype=np.float64),
            lyw_flux=box(astro_options.USE_MCGS),
            xray_heating_rate=box(astro_options.USE_X_RAY_HEATING),
            lya_flux_continuum=box(astro_options.USE_LYA_HEATING),
            lya_flux_injected=box(astro_options.USE_LYA_HEATING),
            lya_flux_continuum_injected=box(not astro_options.USE_LYA_HEATING),
            **kw,
        )

    def get_required_input_arrays(self, input_box: OutputStruct) -> list[str]:
        """Return all input arrays required to compute this object."""
        required = []
        if isinstance(input_box, PerturbedField):
            required += ["density"]
        elif isinstance(input_box, TsBox):
            required += ["xray_ionised_fraction"]
        elif isinstance(input_box, EmissivityFields):
            required += ["sfrd_acg", "xray_emissivity"]
            if self.astro_options.USE_MCGS:
                required += ["sfrd_mcg"]
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
                "filtered_sfrd_acg_for_lya",
                "filtered_xray_emissivity",
            ]
            if self.astro_options.USE_MCGS:
                required += ["filtered_sfrd_mcg_for_lya"]
                if self.astro_options.LYA_MULTIPLE_SCATTERING:
                    required += ["filtered_sfrd_acg_for_lw", "filtered_sfrd_mcg_for_lw"]

            if self.astro_options.USE_LYA_HEATING:
                required += [
                    "lya_flux_continuum_prefactor_acg",
                    "lya_flux_injected_prefactor_acg",
                ]
            else:
                required += ["lya_flux_continuum_injected_prefactor_acg"]

            if self.astro_options.USE_MCGS:
                required += ["lyw_flux_prefactor_acg", "lyw_flux_prefactor_mcg"]
                if self.astro_options.USE_LYA_HEATING:
                    required += [
                        "lya_flux_continuum_prefactor_mcg",
                        "lya_flux_injected_prefactor_mcg",
                    ]
                else:
                    required += ["lya_flux_continuum_injected_prefactor_mcg"]

            if self.astro_options.USE_MCGS:
                required += ["ave_log10_MturnLW"]
        else:
            raise TypeError(
                f"{type(input_box)} is not an input required for RadiationFields!"
            )

        return required

    def compute(
        self,
        *,
        redshift: float,
        emissivity_fields: EmissivityFields,
        R_ct: int,
        R_star: float,
        perturbed_field: PerturbedField,
        previous_spin_temp: TsBox,
        rad_setup: RadiationFieldsSetup,
        allow_already_computed: bool = False,
    ) -> Self:
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            redshift,
            emissivity_fields,
            R_ct,
            R_star,
            perturbed_field,
            previous_spin_temp,
            rad_setup,
        )


@_output_struct
class TsBox(OutputStructZ):
    """A class containing all spin temperature boxes."""

    _c_compute_function = lib.ComputeTsBox
    _meta = False

    _spin_temperature: Array = _arrayfield(alias="spin_temperature")
    spin_temperature = ArrayProxy[np.ndarray]()
    _xray_ionised_fraction: Array = _arrayfield(alias="xray_ionised_fraction")
    xray_ionised_fraction = ArrayProxy[np.ndarray]()
    _kinetic_temp_neutral: Array = _arrayfield(alias="kinetic_temp_neutral")
    kinetic_temp_neutral = ArrayProxy[np.ndarray]()
    _J_21_LW: Array | None = _arrayfield(alias="J_21_LW", default=None)
    J_21_LW = ArrayProxy[np.ndarray | None]()
    Q_HI: float = attrs.field(default=1.0)

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw: Any) -> Self:
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
        return cls(
            inputs=inputs,
            redshift=redshift,
            spin_temperature=Array(shape, dtype=np.float32),
            xray_ionised_fraction=Array(shape, dtype=np.float32),
            kinetic_temp_neutral=Array(shape, dtype=np.float32),
            J_21_LW=(
                Array(shape, dtype=np.float32)
                if inputs.astro_options.USE_MCGS
                else None
            ),
            **kw,
        )

    @cached_property
    def global_Ts(self) -> np.floating:
        """Global (mean) spin temperature."""
        if not self.is_computed:
            raise AttributeError(
                "global_Ts is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.get("spin_temperature"))

    @cached_property
    def global_Tk(self) -> np.floating:
        """Global (mean) Tk."""
        if not self.is_computed:
            raise AttributeError(
                "global_Tk is not defined until the ionization calculation has been performed"
            )
        else:
            return np.mean(self.get("kinetic_temp_neutral"))

    @cached_property
    def global_x_e(self) -> np.floating:
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
                and self.astro_options.USE_MCGS
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
            if self.astro_options.USE_MCGS:
                required += ["J_21_LW"]
        elif isinstance(input_box, RadiationFields):
            required += [
                "xray_ionization_rate",
                "xray_lya_flux",
            ]
            if self.astro_options.USE_X_RAY_HEATING:
                required += ["xray_heating_rate"]
            if self.astro_options.USE_MCGS:
                required += ["lyw_flux"]
            if self.astro_options.USE_LYA_HEATING:
                required += ["lya_flux_continuum", "lya_flux_injected"]
            else:
                required += ["lya_flux_continuum_injected"]
        else:
            raise TypeError(f"{type(input_box)} is not an input required for TsBox!")

        return required

    def compute(
        self,
        *,
        perturbed_field: PerturbedField,
        radiation_fields: RadiationFields,
        prev_spin_temp: TsBox,
        ics: InitialConditions,
        allow_already_computed: bool = False,
    ) -> Self:
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.redshift,
            prev_spin_temp.redshift,
            perturbed_field,
            radiation_fields,
            prev_spin_temp,
            ics,
        )


@_output_struct
class IonizedBox(OutputStructZ):
    """A class containing all ionized boxes."""

    _meta = False
    _c_compute_function = lib.ComputeIonizedBox

    _neutral_fraction: Array = _arrayfield(alias="neutral_fraction")
    neutral_fraction = ArrayProxy[np.ndarray]()
    _ionisation_rate_G12: Array = _arrayfield(alias="ionisation_rate_G12")
    ionisation_rate_G12 = ArrayProxy[np.ndarray]()
    _z_reion: Array = _arrayfield(alias="z_reion")
    z_reion = ArrayProxy[np.ndarray]()
    _mean_free_path: Array | None = _arrayfield(alias="mean_free_path", default=None)
    mean_free_path = ArrayProxy[np.ndarray | None]()
    _cumulative_recombinations: Array | None = _arrayfield(
        alias="cumulative_recombinations", default=None
    )
    cumulative_recombinations = ArrayProxy[np.ndarray | None]()
    _kinetic_temperature: Array | None = _arrayfield(
        alias="kinetic_temperature", default=None
    )
    kinetic_temperature = ArrayProxy[np.ndarray | None]()
    _nion_conditional_filtered_acg: Array | None = _arrayfield(
        alias="nion_conditional_filtered_acg", default=None
    )
    nion_conditional_filtered_acg = ArrayProxy[np.ndarray | None]()
    _nion_conditional_filtered_mcg: Array | None = _arrayfield(
        alias="nion_conditional_filtered_mcg", default=None
    )
    nion_conditional_filtered_mcg = ArrayProxy[np.ndarray | None]()
    log10_mturn_ave_acg: float = attrs.field(default=None)
    log10_mturn_ave_mcg: float = attrs.field(default=None)
    nion_unconditional_acg: float = attrs.field(default=None)
    nion_unconditional_mcg: float = attrs.field(default=None)

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw: Any) -> Self:
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
            inputs.astro_options.USE_MCGS
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

        recomb_model = inputs.astro_options.RECOMB_MODEL
        keep_extra = not inputs.matter_options.MINIMIZE_MEMORY
        filtered_nion = not inputs.matter_options.lagrangian_source_grid

        def box(needed: bool, box_shape: tuple[int, ...] = shape) -> Array | None:
            return Array(box_shape, dtype=np.float32) if needed else None

        cumulative_recombinations = None
        if recomb_model == "inhomogeneous":
            cumulative_recombinations = Array(shape, dtype=np.float32)
        elif recomb_model == "homogeneous":
            cumulative_recombinations = Array((1, 1, 1), dtype=np.float32)

        return cls(
            inputs=inputs,
            redshift=redshift,
            neutral_fraction=Array(shape, initfunc=np.ones, dtype=np.float32),
            ionisation_rate_G12=Array(shape, dtype=np.float32),
            z_reion=Array(shape, dtype=np.float32),
            kinetic_temperature=box(keep_extra),
            mean_free_path=box(keep_extra and recomb_model != "none"),
            cumulative_recombinations=cumulative_recombinations,
            nion_conditional_filtered_acg=box(filtered_nion, filter_shape),
            nion_conditional_filtered_mcg=box(
                filtered_nion and inputs.astro_options.USE_MCGS, filter_shape
            ),
            **kw,
        )

    @cached_property
    def global_xH(self) -> np.floating:
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
                and self.astro_options.USE_MCGS
            ):
                required += ["lowres_vcb"]
        elif isinstance(input_box, PerturbedField):
            required += ["density"]
        elif isinstance(input_box, TsBox):
            required += ["kinetic_temp_neutral", "xray_ionised_fraction"]
            if self.astro_options.USE_MCGS:
                required += ["J_21_LW"]
        elif isinstance(input_box, IonizedBox):
            required += ["z_reion", "ionisation_rate_G12"]
            if self.astro_options.RECOMB_MODEL != "none":
                required += [
                    "cumulative_recombinations",
                ]
            if (
                self.astro_options.USE_MCGS
                and not self.matter_options.lagrangian_source_grid
            ):
                required += [
                    "nion_conditional_filtered_acg",
                    "nion_conditional_filtered_mcg",
                ]
        elif isinstance(input_box, EmissivityFields):
            if self.matter_options.lagrangian_source_grid:
                required += ["n_ion"]
                if self.astro_options.RECOMB_MODEL != "none":
                    required += ["fesc_weighted_sfrd"]
        else:
            # Kept as ValueError (not TypeError): part of the public API contract,
            # asserted verbatim by
            # tests/test_output_structs.py::test_bad_required_array.
            raise ValueError(  # noqa: TRY004
                f"{type(input_box)} is not an input required for IonizedBox!"
            )

        return required

    def compute(
        self,
        *,
        perturbed_field: PerturbedField,
        prev_perturbed_field: PerturbedField,
        prev_ionize_box: IonizedBox,
        spin_temp: TsBox,
        emissivity_fields: EmissivityFields,
        ics: InitialConditions,
        allow_already_computed: bool = False,
    ) -> Self:
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.redshift,
            prev_perturbed_field.redshift,
            perturbed_field,
            prev_perturbed_field,
            prev_ionize_box,
            spin_temp,
            emissivity_fields,
            ics,
        )

    @property
    def log10_Mturnover_ave(self) -> float:
        """The average log10 of the turnover mass for ACGs.

        This property is deprecated and will be removed in a future version.
        Please use `log10_mturn_ave_acg` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "log10_Mturnover_ave",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="log10_Mturnover_ave is deprecated and will be removed in a future version. "
                "Please use log10_mturn_ave_acg directly instead.",
            ),
            stacklevel=2,
        )
        return self.log10_mturn_ave_acg

    @property
    def log10_Mturnover_MINI_ave(self) -> float:
        """The average log10 of the turnover mass for MCGs.

        This property is deprecated and will be removed in a future version.
        Please use `log10_mturn_ave_mcg` directly instead.
        """
        warnings.warn(
            deprecation.DeprecatedWarning(
                "log10_Mturnover_MINI_ave",
                deprecated_in="4.3.0",
                removed_in="5.0.0",
                details="log10_Mturnover_MINI_ave is deprecated and will be removed in a future version. "
                "Please use log10_mturn_ave_mcg directly instead.",
            ),
            stacklevel=2,
        )
        return self.log10_mturn_ave_mcg


@_output_struct
class BrightnessTemp(OutputStructZ):
    """A class containing the brightness temperature box."""

    _c_compute_function = lib.ComputeBrightnessTemp

    _meta = False
    _brightness_temp: Array = _arrayfield(alias="brightness_temp")
    brightness_temp = ArrayProxy[np.ndarray]()
    _tau_21: Array | None = _arrayfield(alias="tau_21", default=None)
    tau_21 = ArrayProxy[np.ndarray | None]()

    @classmethod
    def new(cls, inputs: InputParameters, redshift: float, **kw: Any) -> Self:
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
    def global_Tb(self) -> np.floating:
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
            # Kept as ValueError (not TypeError): part of the public API contract,
            # asserted verbatim by
            # tests/test_output_structs.py::test_bad_required_array.
            raise ValueError(  # noqa: TRY004
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
    ) -> Self:
        """Compute the function."""
        return self._compute(
            allow_already_computed,
            self.redshift,
            spin_temp,
            ionized_box,
            perturbed_field,
        )
