"""Data structure wrappers for the C code."""

from __future__ import annotations

import attrs
import contextlib
import h5py
import logging
import numpy as np
import warnings
from abc import ABCMeta, abstractmethod
from bidict import bidict
from functools import cached_property
from hashlib import md5
from pathlib import Path
from typing import Any, Self, Sequence

from .. import __version__
from .._cfg import config
from ..c_21cmfast import ffi, lib
from ._utils import (
    asarray,
    float_to_string_precision,
    get_all_subclasses,
    snake_to_camel,
)
from .arrays import Array
from .arraystate import ArrayState
from .exceptions import _process_exitcode

logger = logging.getLogger(__name__)


@attrs.define(slots=False)
class StructWrapper:
    """
    A base-class python wrapper for C structures (not instances of them).

    Provides simple methods for creating new instances and accessing field names and values.

    To implement wrappers of specific structures, make a subclass with the same name as the
    appropriate C struct (which must be defined in the C code that has been compiled to the ``ffi``
    object), *or* use an arbitrary name, but set the ``_name`` attribute to the C struct name.
    """

    _name: str = attrs.field(converter=str)
    cstruct = attrs.field(default=None)
    _ffi = attrs.field(default=ffi)

    _TYPEMAP = bidict({"float32": "float *", "float64": "double *", "int32": "int *"})

    @_name.default
    def _name_default(self):
        return self.__class__.__name__

    def __init__(self, *args):
        """Custom initializion actions.

        This instantiates the memory associated with the C struct, attached to this inst.
        """
        self.__attrs_init__(*args)
        self.cstruct = self._new()

    def _new(self):
        """Return a new empty C structure corresponding to this class."""
        return self._ffi.new(f"struct {self._name}*")

    @property
    def fields(self) -> dict[str, Any]:
        """A list of fields of the underlying C struct (a list of tuples of "name, type")."""
        return dict(self._ffi.typeof(self.cstruct[0]).fields)

    @property
    def fieldnames(self) -> list[str]:
        """A list of names of fields of the underlying C struct."""
        return [f for f, t in self.fields.items()]

    @property
    def pointer_fields(self) -> list[str]:
        """A list of names of fields which have pointer type in the C struct."""
        return [f for f, t in self.fields.items() if t.type.kind == "pointer"]

    @property
    def primitive_fields(self) -> list[str]:
        """The list of names of fields which have primitive type in the C struct."""
        return [f for f, t in self.fields.items() if t.type.kind == "primitive"]

    def __getstate__(self):
        """Return the current state of the class without pointers."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["_strings", "cstruct", "_ffi"]
        }


@attrs.define(frozen=True, kw_only=True)
class InputStruct:
    """
    A convenient interface to create a C structure with defaults specified.

    It is provided for the purpose of *creating* C structures in Python to be passed to
    C functions, where sensible defaults are available. Structures which are created
    within C and passed back do not need to be wrapped.

    This provides a *fully initialised* structure, and will fail if not all fields are
    specified with defaults.

    .. note:: The actual C structure is gotten by calling an instance. This is
              auto-generated when called, based on the parameters in the class.

    .. warning:: This class will *not* deal well with parameters of the struct which are
                 pointers. All parameters should be primitive types, except for strings,
                 which are dealt with specially.

    Parameters
    ----------
    ffi : cffi object
        The ffi object from any cffi-wrapped library.
    """

    _write_exclude_fields = ()

    @classmethod
    def new(cls, x: dict | InputStruct | None = None, **kwargs):
        """
        Create a new instance of the struct.

        Parameters
        ----------
        x : dict | InputStruct | None
            Initial values for the struct. If `x` is a dictionary, it should map field
            names to their corresponding values. If `x` is an instance of this class,
            its attributes will be used as initial values. If `x` is None, the
            struct will be initialised with default values.
        """
        if isinstance(x, dict):
            return cls(**x, **kwargs)
        elif isinstance(x, InputStruct):
            return x.clone(**kwargs)
        elif x is None:
            return cls(**kwargs)
        else:
            raise ValueError(
                f"Cannot instantiate {cls.__name__} with type {x.__class__}"
            )

    @cached_property
    def struct(self) -> StructWrapper:
        """The python-wrapped struct associated with this input object."""
        return StructWrapper(self.__class__.__name__)

    @cached_property
    def cstruct(self) -> StructWrapper:
        """The object pointing to the memory accessed by C-code for this struct."""
        cdict = self.cdict
        for k in self.struct.fieldnames:
            val = cdict[k]

            if isinstance(val, str):
                # If it is a string, need to convert it to C string ourselves.
                val = self.ffi.new("char[]", val.encode())

            setattr(self.struct.cstruct, k, val)

        return self.struct.cstruct

    def clone(self, **kwargs):
        """Make a fresh copy of the instance with arbitrary parameters updated."""
        return attrs.evolve(self, **kwargs)

    def asdict(self) -> dict:
        """Return a dict representation of the instance.

        Examples
        --------
        This dict should be such that doing the following should work, i.e. it can be
        used exactly to construct a new instance of the same object::

        >>> inp = InputStruct(**params)
        >>> newinp =InputStruct(**inp.asdict())
        >>> inp == newinp
        """
        return attrs.asdict(self)

    @cached_property
    def cdict(self) -> dict:
        """A python dictionary containing the properties of the wrapped C-struct.

        The memory pointed to by this dictionary is *not* owned by the wrapped C-struct,
        but is rather just a python dict. However, in contrast to :meth:`asdict`, this
        method transforms the properties to what they should be in C (e.g. linear space
        vs. log-space) before putting them into the dict.

        This dict also contains *only* the properties of the wrapped C-struct, rather
        than all properties of the :class:`InputStruct` instance (some attributes of the
        python instance are there only to guide setting of defaults, and don't appear
        in the C-struct at all).
        """
        fields = attrs.fields(self.__class__)
        transformers = {
            field.name: field.metadata.get("transformer", None) for field in fields
        }

        out = {}
        for k in self.struct.fieldnames:
            val = getattr(self, k)
            # we assume properties (as opposed to attributes) are already converted
            trns = transformers[k] if k in transformers.keys() else None
            out[k] = val if trns is None else trns(val)
        return out

    def __str__(self):
        """Human-readable string representation of the object."""
        d = self.asdict()
        biggest_k = max(len(k) for k in d)
        params = "\n    ".join(sorted(f"{k:<{biggest_k}}: {v}" for k, v in d.items()))
        return f"""{self.__class__.__name__}:{params} """

    @classmethod
    def from_subdict(cls, dct, safe=True):
        """Construct an instance of a parameter structure from a dictionary."""
        fieldnames = [
            field.name
            for field in attrs.fields(cls)
            if field.eq and field.default is not None
        ]
        if set(fieldnames) != set(dct.keys()):
            missing_items = [
                (field.name, field.default)
                for field in attrs.fields(cls)
                if field.name not in dct.keys() and field.name in fieldnames
            ]
            extra_items = [(k, v) for k, v in dct.items() if k not in fieldnames]
            message = (
                f"There are extra or missing {cls.__name__} in the file to be read.\n"
                f"EXTRAS: {extra_items}\n"
                f"MISSING: {missing_items}\n"
            )
            if safe:
                raise ValueError(
                    message
                    + "set `safe=False` to load structures from previous versions"
                )
            else:
                warnings.warn(
                    message
                    + "\nExtras are ignored and missing are set to default (shown) values."
                    + "\nUsing these parameter structures in further computation will give inconsistent results."
                )
            dct = {k: v for k, v in dct.items() if k in fieldnames}

        return cls.new(dct)


@attrs.define(slots=False)
class OutputStruct(ABCMeta):
    """Base class for any class that wraps a C struct meant to be output from a C function."""

    _meta = True
    _fields_ = []
    _global_params = None
    _c_based_pointers = ()
    _c_compute_function = None

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

    @property
    def arrays(self) -> dict[str, Array]:
        me = attrs.asdict(self)
        return {k: x for k, x in me.items() if isinstance(x, Array)}

    # @cached_property
    # def _array_structure(self) -> dict[str, tuple[int] | dict[str, Any]]:
    #     """A dictionary of names and shapes of arrays in the struct."""
    #     return self._get_box_structures()

    # def __init__(self, *, dummy=False, initial=False, **kwargs):
    #     """
    #     Base type for output structures from C functions.

    #     Parameters
    #     ----------
    #     random_seed
    #         Seed associated with the output.
    #     dummy
    #         Specify this as a dummy struct, in which no arrays are to be
    #         initialized or computed.
    #     initial
    #         Specify this as an initial struct, where arrays are to be
    #         initialized, but do not need to be computed to pass into another
    #         struct's compute().
    #     """
    #     self._name = self.__class__.__name__

    #     self._array_structure = self._get_box_structures()
    #     self._array_state = {k: ArrayState() for k in self._array_structure} | {
    #         k: ArrayState() for k in self._c_based_pointers
    #     }
    #     for k in self._array_structure:
    #         if k not in self.struct.pointer_fields:
    #             raise TypeError(f"Key {k} in {self} not a defined pointer field in C.")

    @cached_property
    def struct(self) -> StructWrapper:
        """The python-wrapped struct associated with this input object."""
        return StructWrapper(self._name)

    @cached_property
    def cstruct(self) -> StructWrapper:
        """The object pointing to the memory accessed by C-code for this struct."""
        self._init_cstruct()
        return self.struct.cstruct

    # @property
    # def path(self) -> tuple[None, Path]:
    #     """The path to an on-disk version of this object."""
    #     if not self._paths:
    #         return None

    #     for pth in self._paths:
    #         if pth.exists():
    #             return pth

    #     logger.info(f"All paths that defined {self} have been deleted on disk.")
    #     return None

    # @abstractmethod
    # def _get_box_structures(self) -> dict[str, dict | tuple[int]]:
    #     """Return a dictionary of names mapping to shapes for each array in the struct.

    #     The reason this is a function, not a simple attribute, is that we may need to
    #     decide on what arrays need to be initialized based on the inputs (eg. if USE_2LPT
    #     is True or False).

    #     Each actual OutputStruct subclass needs to implement this. Note that the arrays
    #     are not actually initialized here -- that's done automatically by :func:`_init_arrays`
    #     using this information. This function means that the names of the actually required
    #     arrays can be accessed without doing any actual initialization.

    #     Note also that this only contains arrays allocated *by Python* not C. Arrays
    #     allocated by C are specified in :func:`_c_shape`.
    #     """
    #     pass

    def _c_shape(self, cstruct) -> dict[str, tuple[int]]:
        """Return a dictionary of field: shape for arrays allocated within C."""
        return {}

    # @classmethod
    # def _implementations(cls):
    #     all_classes = get_all_subclasses(cls)
    #     return [c for c in all_classes if not c._meta]

    def _init_arrays(self):
        for k, array in self.arrays.items():
            # Don't initialize C-based pointers or already-inited stuff, or stuff
            # that's computed on disk (if it's on disk, accessing the array should
            # just give the computed version, which is what we would want, not a
            # zero-inited array).
            if array.state.c_memory or array.state.initialized or array.state.on_disk:
                continue

            # TODO: maybe change to a simple InputParameters object
            setattr(self, k, array.initialize(self.user_params, self.flag_options))

    @property
    def random_seed(self):
        """The random seed for this particular instance."""
        if self._random_seed is None:
            self._random_seed = int(np.random.randint(1, int(1e12)))

        return self._random_seed

    def sync(self):
        """Sync the current state of the object with the underlying C-struct.

        This will link any memory initialized by numpy in this object with the underlying
        C-struct, and also update this object with any values computed from within C.
        """
        # Initialize all uninitialized arrays.
        self._init_arrays()

        for name, array in self.arrays.items():

            # We do *not* set COMPUTED_ON_DISK items to the C-struct here, because we have no
            # way of knowing (in this function) what is required to load in, and we don't want
            # to unnecessarily load things in. We leave it to the user to ensure that all
            # required arrays are loaded into memory before calling this function.
            if array.state.initialized:
                array.expose_to_c(self.struct, name)

        for k in self.struct.primitive_fields:
            if hasattr(self, k):
                setattr(self.struct.cstruct, k, getattr(self, k))
            else:
                setattr(self, k, getattr(self.cstruct, k))

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
            self._remove_array(k, force)

        # Accessing the array loads it into memory.
        for k in keep:
            setattr(self, k, getattr(self, k).loaded_from_disk())

    def _remove_array(self, k: str, force=False):
        array = self.arrays[k]
        state = array.state

        if (
            not state.initialized
        ):  # TODO: how to handle the case where some arrays aren't required at all?
            warnings.warn(f"Trying to remove array that isn't yet created: {k}")
            return

        if state.computed_in_mem and not state.on_disk and not force:
            raise OSError(
                f"Trying to purge array '{k}' from memory that hasn't been stored! Use force=True if you meant to do this."
            )

        if state.c_has_active_memory:  # TODO: do we need C-managed memory any more?
            lib.free(getattr(self.cstruct, k))

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
        self.prepare(flush=[])

    # def __repr__(self):
    #     """Return a fully unique representation of the instance."""
    #     # This is the class name and all parameters which belong to C-based input structs,
    #     # eg. InitialConditions(HII_DIM:100,SIGMA_8:0.8,...)
    #     # eg. InitialConditions(HII_DIM:100,SIGMA_8:0.8,...)
    #     return f"{self._seedless_repr()}_random_seed={self._random_seed}"

    # def _seedless_repr(self):
    #     # The same as __repr__ except without the seed.
    #     return (
    #         (
    #             self._name
    #             + "("
    #             + "; ".join(
    #                 (
    #                     repr(v)
    #                     if isinstance(v, InputStruct)
    #                     else (
    #                         v.filtered_repr(self._filter_params)
    #                         if isinstance(v, StructInstanceWrapper)
    #                         else k.lstrip("_")
    #                         + ":"
    #                         + (
    #                             float_to_string_precision(
    #                                 v, config["cache_param_sigfigs"]
    #                             )
    #                             if isinstance(v, (float, np.float32))
    #                             else repr(v)
    #                         )
    #                     )
    #                 )
    #                 for k, v in [
    #                     (k, getattr(self, k))
    #                     for k in self._all_inputs
    #                     if k != "_random_seed"
    #                 ]
    #             )
    #         )
    #         + f"; v{self.version}"
    #         + ")"
    #     )

    # def __str__(self):
    #     """Return a human-readable representation of the instance."""
    #     # this is *not* a unique representation, and doesn't include global params.
    #     return (
    #         self._name
    #         + "("
    #         + ";\n\t".join(
    #             (
    #                 repr(v)
    #                 if isinstance(v, InputStruct)
    #                 else k.lstrip("_") + ":" + repr(v)
    #             )
    #             for k, v in [(k, getattr(self, k)) for k in self._inputs]
    #         )
    #     ) + ")"

    # def __hash__(self):
    #     """Return a unique hsh for this instance, even global params and random seed."""
    #     return hash(repr(self))

    # @property
    # def _md5(self):
    #     """Return a unique hsh of the object, *not* taking into account the random seed."""
    #     return md5(self._seedless_repr().encode()).hexdigest()

    # def __eq__(self, other):
    #     """Check equality with another object via its __repr__."""
    #     return repr(self) == repr(other)

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
        return any(v.state.computed for v in self.arrays.values())

    def ensure_arrays_computed(self, *arrays, load=False) -> bool:
        """Check if the given arrays are computed (not just initialized)."""
        if not self.is_computed:
            return False

        computed = all(self.arrays[k].state.computed for k in arrays)

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
        pass

    def ensure_input_computed(self, input_box, load=False) -> bool:
        """Ensure all the inputs have been computed."""
        if input_box.dummy:
            return True

        arrays = self.get_required_input_arrays(input_box)

        if input_box.initial:
            return input_box.ensure_arrays_inited(*arrays, init=load)

        return input_box.ensure_arrays_computed(*arrays, load=load)

    def summarize(self, indent=0) -> str:
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
            elif not state.computed:
                out += f"{indent}    {fieldname:>25}:  initialized\n"
            elif not state.computed_in_mem:
                out += f"{indent}    {fieldname:>25}:  computed on disk\n"
            else:
                x = getattr(self, fieldname).flatten()
                if len(x) > 0:
                    out += f"{indent}    {fieldname:>25}: {x[0]:11.4e}, {x[-1]:11.4e}, {x.min():11.4e}, {x.max():11.4e}, {np.mean(x):11.4e}\n"
                else:
                    out += f"{indent}    {fieldname:>25}: size zero\n"

        # print primitive fields
        out += "".join(
            f"{indent}    {fieldname:>25}: {getattr(self, fieldname, 'non-existent')}\n"
            for fieldname in self.struct.primitive_fields
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
                    f"Current State: {[(k, str(v)) for k, v in self._array_state.items()]}"
                )

    def _compute(
        self, *args, hooks: dict[str | callable, dict[str, Any]] | None = None
    ):
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
            arg.struct.cstruct if isinstance(arg, (OutputStruct, InputStruct)) else arg
            for arg in args
        ]

        # Ensure we haven't already tried to compute this instance.
        if self.is_computed:
            raise ValueError(
                f"You are trying to compute {self.__class__.__name__}, but it has already been computed."
            )

        # Perform the C computation
        try:
            exitcode = self._c_compute_function(*inputs, self.struct.cstruct)
        except TypeError as e:
            logger.error(
                f"Arguments to {self._c_compute_function.__name__}: " f"{inputs}"
            )
            raise e

        _process_exitcode(exitcode, self._c_compute_function, args)

        # Ensure memory created in C gets mapped to numpy arrays in this struct.
        for k, state in self._array_state.items():
            if state.initialized:
                state.computed_in_mem = True

        self.sync()

        # Optionally do stuff with the result (like writing it)
        self._call_hooks(hooks)

    def _call_hooks(self, hooks):
        if hooks is None:
            hooks = {"write": {"direc": config["direc"]}}

        for hook, params in hooks.items():
            if callable(hook):
                hook(self, **params)
            else:
                getattr(self, hook)(**params)


class StructInstanceWrapper:
    """A wrapper for *instances* of C structs.

    This is as opposed to :class:`StructWrapper`, which is for the un-instantiated structs.

    Parameters
    ----------
    wrapped :
        The reference to the C object to wrap (contained in the ``cffi.lib`` object).
    ffi :
        The ``cffi.ffi`` object.
    """

    def __init__(self, wrapped, ffi):
        self._cobj = wrapped
        self._ffi = ffi

        for nm, tp in self._ffi.typeof(self._cobj).fields:
            setattr(self, nm, getattr(self._cobj, nm))

        # Get the name of the structure
        self._ctype = self._ffi.typeof(self._cobj).cname.split()[-1]

    def __setattr__(self, name, value):
        """Set an attribute of the instance, attempting to change it in the C struct as well."""
        with contextlib.suppress(AttributeError):
            setattr(self._cobj, name, value)
        object.__setattr__(self, name, value)

    def items(self):
        """Yield (name, value) pairs for each element of the struct."""
        for nm, tp in self._ffi.typeof(self._cobj).fields:
            yield nm, getattr(self, nm)

    def keys(self):
        """Return a list of names of elements in the struct."""
        return [nm for nm, tp in self.items()]

    def __repr__(self):
        """Return a unique representation of the instance."""
        return (
            self._ctype
            + "("
            + ";".join(f"{k}={str(v)}" for k, v in sorted(self.items()))
        ) + ")"

    def filtered_repr(self, filter_params):
        """Get a fully unique representation of the instance that filters out some parameters.

        Parameters
        ----------
        filter_params : list of str
            The parameter names which should not appear in the representation.
        """
        return (
            self._ctype
            + "("
            + ";".join(
                f"{k}={str(v)}"
                for k, v in sorted(self.items())
                if k not in filter_params
            )
        ) + ")"
