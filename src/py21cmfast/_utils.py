"""Utilities that help with wrapping various C structures."""

import glob
import h5py
import logging
import numpy as np
import warnings
from abc import ABCMeta, abstractmethod
from bidict import bidict
from cffi import FFI
from enum import IntEnum
from hashlib import md5
from os import makedirs, path
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from . import __version__
from ._cfg import config
from .c_21cmfast import lib

_ffi = FFI()

logger = logging.getLogger(__name__)


class ArrayStateError(ValueError):
    """Errors arising from incorrectly modifying array state."""

    pass


class ArrayState:
    """Define the memory state of a struct array."""

    def __init__(
        self, initialized=False, c_memory=False, computed_in_mem=False, on_disk=False
    ):
        self._initialized = initialized
        self._c_memory = c_memory
        self._computed_in_mem = computed_in_mem
        self._on_disk = on_disk

    @property
    def initialized(self):
        """Whether the array is initialized (i.e. allocated memory)."""
        return self._initialized

    @initialized.setter
    def initialized(self, val):
        if not val:
            # if its not initialized, can't be computed in memory
            self.computed_in_mem = False
        self._initialized = bool(val)

    @property
    def c_memory(self):
        """Whether the array's memory (if any) is controlled by C."""
        return self._c_memory

    @c_memory.setter
    def c_memory(self, val):
        self._c_memory = bool(val)

    @property
    def computed_in_mem(self):
        """Whether the array is computed and stored in memory."""
        return self._computed_in_mem

    @computed_in_mem.setter
    def computed_in_mem(self, val):
        if val:
            # any time we pull something into memory, it must be initialized.
            self.initialized = True
        self._computed_in_mem = bool(val)

    @property
    def on_disk(self):
        """Whether the array is computed and store on disk."""
        return self._on_disk

    @on_disk.setter
    def on_disk(self, val):
        self._on_disk = bool(val)

    @property
    def computed(self):
        """Whether the array is computed anywhere."""
        return self.computed_in_mem or self.on_disk

    @property
    def c_has_active_memory(self):
        """Whether C currently has initialized memory for this array."""
        return self.c_memory and self.initialized

    def __str__(self):
        """Returns a string representation of the ArrayState."""
        if self.computed_in_mem:
            return "computed (in mem)"
        elif self.on_disk:
            return "computed (on disk)"
        elif self.initialized:
            return "memory initialized (not computed)"
        else:
            return "uncomputed and uninitialized"


class ParameterError(RuntimeError):
    """An exception representing a bad choice of parameters."""

    default_message = "21cmFAST does not support this combination of parameters."

    def __init__(self, msg=None):
        super().__init__(msg or self.default_message)


class FatalCError(Exception):
    """An exception representing something going wrong in C."""

    default_message = "21cmFAST is exiting."

    def __init__(self, msg=None):
        super().__init__(msg or self.default_message)


class FileIOError(FatalCError):
    """An exception when an error occurs with file I/O."""

    default_message = "Expected file could not be found! (check the LOG for more info)"


class GSLError(ParameterError):
    """An exception when a GSL routine encounters an error."""

    default_message = "A GSL routine has errored! (check the LOG for more info)"


class ArgumentValueError(FatalCError):
    """An exception when a function takes an unexpected input."""

    default_message = "An incorrect argument has been defined or passed! (check the LOG for more info)"


class PhotonConsError(ParameterError):
    """An exception when the photon non-conservation correction routine errors."""

    default_message = "An error has occured with the Photon non-conservation correction! (check the LOG for more info)"


class TableGenerationError(ParameterError):
    """An exception when an issue arises populating one of the interpolation tables."""

    default_message = """An error has occured when generating an interpolation table!
                This has likely occured due to the choice of input AstroParams (check the LOG for more info)"""


class TableEvaluationError(ParameterError):
    """An exception when an issue arises populating one of the interpolation tables."""

    default_message = """An error has occured when evaluating an interpolation table!
                This can sometimes occur due to small boxes (either small DIM/HII_DIM or BOX_LEN) (check the LOG for more info)"""


class InfinityorNaNError(ParameterError):
    """An exception when an infinity or NaN is encountered in a calculated quantity."""

    default_message = """Something has returned an infinity or a NaN! This could be due to an issue with an
                input parameter choice (check the LOG for more info)"""


class MassDepZetaError(ParameterError):
    """An exception when determining the bisection for stellar mass/escape fraction."""

    default_message = """There is an issue with the choice of parameters under MASS_DEPENDENT_ZETA. Could be an issue with
                any of the chosen F_STAR10, ALPHA_STAR, F_ESC10 or ALPHA_ESC."""


class MemoryAllocError(FatalCError):
    """An exception when unable to allocated memory."""

    default_message = """An error has occured while attempting to allocate memory! (check the LOG for more info)"""


SUCCESS = 0
IOERROR = 1
GSLERROR = 2
VALUEERROR = 3
PHOTONCONSERROR = 4
TABLEGENERATIONERROR = 5
TABLEEVALUATIONERROR = 6
INFINITYORNANERROR = 7
MASSDEPZETAERROR = 8
MEMORYALLOCERROR = 9


def _process_exitcode(exitcode, fnc, args):
    """Determine what happens for different values of the (integer) exit code from a C function."""
    if exitcode != SUCCESS:
        logger.error(f"In function: {fnc.__name__}.  Arguments: {args}")

        if exitcode:
            try:
                raise {
                    IOERROR: FileIOError,
                    GSLERROR: GSLError,
                    VALUEERROR: ArgumentValueError,
                    PHOTONCONSERROR: PhotonConsError,
                    TABLEGENERATIONERROR: TableGenerationError,
                    TABLEEVALUATIONERROR: TableEvaluationError,
                    INFINITYORNANERROR: InfinityorNaNError,
                    MASSDEPZETAERROR: MassDepZetaError,
                    MEMORYALLOCERROR: MemoryAllocError,
                }[exitcode]
            except KeyError:  # pragma: no cover
                raise FatalCError(
                    "Unknown error in C. Please report this error!"
                )  # Unknown C code


ctype2dtype = {}

# Integer types
for prefix in ("int", "uint"):
    for log_bytes in range(4):
        ctype = "%s%d_t" % (prefix, 8 * (2**log_bytes))
        dtype = "%s%d" % (prefix[0], 2**log_bytes)
        ctype2dtype[ctype] = np.dtype(dtype)

# Floating point types
ctype2dtype["float"] = np.dtype("f4")
ctype2dtype["double"] = np.dtype("f8")
ctype2dtype["int"] = np.dtype("i4")


def asarray(ptr, shape):
    """Get the canonical C type of the elements of ptr as a string."""
    ctype = _ffi.getctype(_ffi.typeof(ptr).item).split("*")[0].strip()

    if ctype not in ctype2dtype:
        raise RuntimeError(
            f"Cannot create an array for element type: {ctype}. Can do {list(ctype2dtype.values())}."
        )

    array = np.frombuffer(
        _ffi.buffer(ptr, _ffi.sizeof(ctype) * np.prod(shape)), ctype2dtype[ctype]
    )
    array.shape = shape
    return array


class StructWrapper:
    """
    A base-class python wrapper for C structures (not instances of them).

    Provides simple methods for creating new instances and accessing field names and values.

    To implement wrappers of specific structures, make a subclass with the same name as the
    appropriate C struct (which must be defined in the C code that has been compiled to the ``ffi``
    object), *or* use an arbitrary name, but set the ``_name`` attribute to the C struct name.
    """

    _name = None
    _ffi = None

    def __init__(self):
        # Set the name of this struct in the C code
        self._name = self._get_name()

    @classmethod
    def _get_name(cls):
        return cls._name or cls.__name__

    @property
    def _cstruct(self):
        """
        The actual structure which needs to be passed around to C functions.

        .. note:: This is best accessed by calling the instance (see __call__).

        The reason it is defined as this (manual) cached property is so that it can be created
        dynamically, but not lost. It must not be lost, or else C functions which use it will lose
        access to its memory. But it also must be created dynamically so that it can be recreated
        after pickling (pickle can't handle CData).
        """
        try:
            return self.__cstruct
        except AttributeError:
            self.__cstruct = self._new()
            return self.__cstruct

    def _new(self):
        """Return a new empty C structure corresponding to this class."""
        return self._ffi.new("struct " + self._name + "*")

    @classmethod
    def get_fields(cls, cstruct=None) -> Dict[str, Any]:
        """Obtain the C-side fields of this struct."""
        if cstruct is None:
            cstruct = cls._ffi.new("struct " + cls._get_name() + "*")
        return dict(cls._ffi.typeof(cstruct[0]).fields)

    @classmethod
    def get_fieldnames(cls, cstruct=None) -> List[str]:
        """Obtain the C-side field names of this struct."""
        fields = cls.get_fields(cstruct)
        return [f for f, t in fields]

    @classmethod
    def get_pointer_fields(cls, cstruct=None) -> List[str]:
        """Obtain all pointer fields of the struct (typically simulation boxes)."""
        return [f for f, t in cls.get_fields(cstruct) if t.type.kind == "pointer"]

    @property
    def fields(self) -> Dict[str, Any]:
        """List of fields of the underlying C struct (a list of tuples of "name, type")."""
        return self.get_fields(self._cstruct)

    @property
    def fieldnames(self) -> List[str]:
        """List names of fields of the underlying C struct."""
        return [f for f, t in self.fields.items()]

    @property
    def pointer_fields(self) -> List[str]:
        """List of names of fields which have pointer type in the C struct."""
        return [f for f, t in self.fields.items() if t.type.kind == "pointer"]

    @property
    def primitive_fields(self) -> List[str]:
        """List of names of fields which have primitive type in the C struct."""
        return [f for f, t in self.fields.items() if t.type.kind == "primitive"]

    def __getstate__(self):
        """Return the current state of the class without pointers."""
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["_strings", "_StructWrapper__cstruct"]
        }

    def refresh_cstruct(self):
        """Delete the underlying C object, forcing it to be rebuilt."""
        try:
            del self.__cstruct
        except AttributeError:
            pass

    def __call__(self):
        """Return an instance of the C struct."""
        pass


class StructWithDefaults(StructWrapper):
    """
    A convenient interface to create a C structure with defaults specified.

    It is provided for the purpose of *creating* C structures in Python to be passed to C functions,
    where sensible defaults are available. Structures which are created within C and passed back do
    not need to be wrapped.

    This provides a *fully initialised* structure, and will fail if not all fields are specified
    with defaults.

    .. note:: The actual C structure is gotten by calling an instance. This is auto-generated when
              called, based on the parameters in the class.

    .. warning:: This class will *not* deal well with parameters of the struct which are pointers.
                 All parameters should be primitive types, except for strings, which are dealt with
                 specially.

    Parameters
    ----------
    ffi : cffi object
        The ffi object from any cffi-wrapped library.
    """

    _defaults_ = {}

    def __init__(self, *args, **kwargs):
        super().__init__()
        if args:
            if len(args) > 1:
                raise TypeError(
                    "%s takes up to one position argument, %s were given"
                    % (self.__class__.__name__, len(args))
                )
            elif args[0] is None:
                pass
            elif isinstance(args[0], self.__class__):
                kwargs.update(args[0].self)
            elif isinstance(args[0], dict):
                kwargs.update(args[0])
            else:
                raise TypeError(
                    f"optional positional argument for {self.__class__.__name__} must be"
                    f" None, dict, or an instance of itself. Got {type(args[0])}"
                )

        for k, v in self._defaults_.items():
            # Prefer arguments given to the constructor.
            _v = kwargs.pop(k, None)

            if _v is not None:
                v = _v

            try:
                setattr(self, k, v)
            except AttributeError:
                # The attribute has been defined as a property, save it as a hidden variable
                setattr(self, "_" + k, v)

        if kwargs:
            warnings.warn(
                "The following parameters to {thisclass} are not supported: {lst}".format(
                    thisclass=self.__class__.__name__, lst=list(kwargs.keys())
                )
            )

    def convert(self, key, val):
        """Make any conversions of values before saving to the instance."""
        return val

    def update(self, **kwargs):
        """
        Update the parameters of an existing class structure.

        This should always be used instead of attempting to *assign* values to instance attributes.
        It consistently re-generates the underlying C memory space and sets some book-keeping
        variables.

        Parameters
        ----------
        kwargs:
            Any argument that may be passed to the class constructor.
        """
        # Start a fresh cstruct.
        if kwargs:
            self.refresh_cstruct()

        for k in self._defaults_:
            # Prefer arguments given to the constructor.
            if k in kwargs:
                v = kwargs.pop(k)

                try:
                    setattr(self, k, v)
                except AttributeError:
                    # The attribute has been defined as a property, save it as a hidden variable
                    setattr(self, "_" + k, v)

        # Also ensure that parameters that are part of the class, but not the defaults, are set
        # this will fail if these parameters cannot be set for some reason, hence doing it
        # last.
        for k in list(kwargs.keys()):
            if hasattr(self, k):
                setattr(self, k, kwargs.pop(k))

        if kwargs:
            warnings.warn(
                "The following arguments to be updated are not compatible with this class: %s"
                % kwargs
            )

    def clone(self, **kwargs):
        """Make a fresh copy of the instance with arbitrary parameters updated."""
        new = self.__class__(self.self)
        new.update(**kwargs)
        return new

    def __call__(self):
        """Return a filled C Structure corresponding to this instance."""
        for key, val in self.pystruct.items():
            # Find the value of this key in the current class
            if isinstance(val, str):
                # If it is a string, need to convert it to C string ourselves.
                val = self.ffi.new("char[]", getattr(self, key).encode())

            try:
                setattr(self._cstruct, key, val)
            except TypeError:
                logger.info(f"For key {key}, value {val}:")
                raise

        return self._cstruct

    @property
    def pystruct(self):
        """A pure-python dictionary representation of the corresponding C structure."""
        return {fld: self.convert(fld, getattr(self, fld)) for fld in self.fieldnames}

    @property
    def defining_dict(self):
        """
        Pure python dictionary representation of this class, as it would appear in C.

        .. note:: This is not the same as :attr:`pystruct`, as it omits all variables that don't
                  need to be passed to the constructor, but appear in the C struct (some can be
                  calculated dynamically based on the inputs). It is also not the same as
                  :attr:`self`, as it includes the 'converted' values for each variable, which are
                  those actually passed to the C code.
        """
        return {k: self.convert(k, getattr(self, k)) for k in self._defaults_}

    @property
    def self(self):
        """
        Dictionary which if passed to its own constructor will yield an identical copy.

        .. note:: This differs from :attr:`pystruct` and :attr:`defining_dict` in that it uses the
                  hidden variable value, if it exists, instead of the exposed one. This prevents
                  from, for example, passing a value which is 10**10**val (and recurring!).
        """
        # Try to first use the hidden variable before using the non-hidden variety.
        dct = {}
        for k in self._defaults_:
            if hasattr(self, "_" + k):
                dct[k] = getattr(self, "_" + k)
            else:
                dct[k] = getattr(self, k)

        return dct

    def __repr__(self):
        """Full unique representation of the instance."""
        return (
            self.__class__.__name__
            + "("
            + ", ".join(
                sorted(
                    k
                    + ":"
                    + (
                        float_to_string_precision(v, config["cache_redshift_sigfigs"])
                        if isinstance(v, (float, np.float32))
                        else str(v)
                    )
                    for k, v in self.defining_dict.items()
                )
            )
            + ")"
        )

    def __eq__(self, other):
        """Check whether this instance is equal to another object (by checking the __repr__)."""
        return self.__repr__() == repr(other)

    def __hash__(self):
        """Generate a unique hsh for the instance."""
        return hash(self.__repr__())

    def __str__(self):
        """Human-readable string representation of the object."""
        biggest_k = max(len(k) for k in self.defining_dict)
        params = "\n    ".join(
            sorted(f"{k:<{biggest_k}}: {v}" for k, v in self.defining_dict.items())
        )
        return f"""{self.__class__.__name__}:
    {params}
    """


def snake_to_camel(word: str, publicize: bool = True):
    """Convert snake case to camel case."""
    if publicize:
        word = word.lstrip("_")
    return "".join(x.capitalize() or "_" for x in word.split("_"))


def camel_to_snake(word: str, depublicize: bool = False):
    """Convert came case to snake case."""
    word = "".join("_" + i.lower() if i.isupper() else i for i in word)

    if not depublicize:
        word = word.lstrip("_")

    return word


def float_to_string_precision(x, n):
    """Prints out a standard float number at a given number of significant digits.

    Code here: https://stackoverflow.com/a/48812729
    """
    return f'{float(f"{x:.{int(n)}g}"):g}'


def get_all_subclasses(cls):
    """Get a list of all subclasses of a given class, recursively."""
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


class OutputStruct(StructWrapper, metaclass=ABCMeta):
    """Base class for any class that wraps a C struct meant to be output from a C function."""

    _meta = True
    _fields_ = []
    _global_params = None
    _inputs = ("user_params", "cosmo_params", "_random_seed")
    _filter_params = ["external_table_path", "wisdoms_path"]
    _c_based_pointers = ()
    _c_compute_function = None

    _TYPEMAP = bidict({"float32": "float *", "float64": "double *", "int32": "int *"})

    def __init__(self, *, random_seed=None, dummy=False, initial=False, **kwargs):
        """
        Base type for output structures from C functions.

        Parameters
        ----------
        random_seed
            Seed associated with the output.
        dummy
            Specify this as a dummy struct, in which no arrays are to be
            initialized or computed.
        initial
            Specify this as an initial struct, where arrays are to be
            initialized, but do not need to be computed to pass into another
            struct's compute().
        """
        super().__init__()

        self.version = ".".join(__version__.split(".")[:2])
        self.patch_version = ".".join(__version__.split(".")[2:])
        self._paths = []

        self._random_seed = random_seed

        for k in self._inputs:
            if k not in self.__dict__:
                try:
                    setattr(self, k, kwargs.pop(k))
                except KeyError:
                    raise KeyError(
                        f"{self.__class__.__name__} requires the keyword argument {k}"
                    )

        if kwargs:
            warnings.warn(
                f"{self.__class__.__name__} received the following unexpected "
                f"arguments: {list(kwargs.keys())}"
            )

        self.dummy = dummy
        self.initial = initial

        self._array_structure = self._get_box_structures()
        self._array_state = {k: ArrayState() for k in self._array_structure}
        self._array_state.update({k: ArrayState() for k in self._c_based_pointers})

        for k in self._array_structure:
            if k not in self.pointer_fields:
                raise TypeError(f"Key {k} in {self} not a defined pointer field in C.")

    @property
    def path(self) -> Tuple[None, Path]:
        """The path to an on-disk version of this object."""
        if not self._paths:
            return None

        for pth in self._paths:
            if pth.exists():
                return pth

        logger.info(f"All paths that defined {self} have been deleted on disk.")
        return None

    @abstractmethod
    def _get_box_structures(self) -> Dict[str, Union[Dict, Tuple[int]]]:
        """Return a dictionary of names mapping to shapes for each array in the struct.

        The reason this is a function, not a simple attribute, is that we may need to
        decide on what arrays need to be initialized based on the inputs (eg. if USE_2LPT
        is True or False).

        Each actual OutputStruct subclass needs to implement this. Note that the arrays
        are not actually initialized here -- that's done automatically by :func:`_init_arrays`
        using this information. This function means that the names of the actually required
        arrays can be accessed without doing any actual initialization.

        Note also that this only contains arrays allocated *by Python* not C. Arrays
        allocated by C are specified in :func:`_c_shape`.
        """
        pass

    def _c_shape(self, cstruct) -> Dict[str, Tuple[int]]:
        """Return a dictionary of field: shape for arrays allocated within C."""
        return {}

    @classmethod
    def _implementations(cls):
        all_classes = get_all_subclasses(cls)
        return [c for c in all_classes if not c._meta]

    def _init_arrays(self):
        for k, state in self._array_state.items():
            # Don't initialize C-based pointers or already-inited stuff, or stuff
            # that's computed on disk (if it's on disk, accessing the array should
            # just give the computed version, which is what we would want, not a
            # zero-inited array).
            if k in self._c_based_pointers or state.initialized or state.on_disk:
                continue

            params = self._array_structure[k]
            tp = self._TYPEMAP.inverse[self.fields[k].type.cname]

            if isinstance(params, tuple):
                shape = params
                fnc = np.zeros
            elif isinstance(params, dict):
                fnc = params.get("init", np.zeros)
                shape = params.get("shape")
            else:
                raise ValueError("params is not a tuple or dict")

            setattr(self, k, fnc(shape, dtype=tp))

            # Add it to initialized arrays.
            state.initialized = True

    @property
    def random_seed(self):
        """The random seed for this particular instance."""
        if self._random_seed is None:
            self._random_seed = int(np.random.randint(1, int(1e12)))

        return self._random_seed

    def _init_cstruct(self):
        # Initialize all uninitialized arrays.
        self._init_arrays()

        for k, state in self._array_state.items():
            # We do *not* set COMPUTED_ON_DISK items to the C-struct here, because we have no
            # way of knowing (in this function) what is required to load in, and we don't want
            # to unnecessarily load things in. We leave it to the user to ensure that all
            # required arrays are loaded into memory before calling this function.
            if state.initialized:
                setattr(self._cstruct, k, self._ary2buf(getattr(self, k)))

        for k in self.primitive_fields:
            try:
                setattr(self._cstruct, k, getattr(self, k))
            except AttributeError:
                pass

    def _ary2buf(self, ary):
        if not isinstance(ary, np.ndarray):
            raise ValueError("ary must be a numpy array")
        return self._ffi.cast(
            OutputStruct._TYPEMAP[ary.dtype.name], self._ffi.from_buffer(ary)
        )

    def __call__(self):
        """Initialize/allocate a fresh C struct in memory and return it."""
        if not self.dummy:
            self._init_cstruct()

        return self._cstruct

    def __expose(self):
        """Expose the non-array primitives of the ctype to the top-level object."""
        for k in self.primitive_fields:
            setattr(self, k, getattr(self._cstruct, k))

    @property
    def _fname_skeleton(self):
        """The filename without specifying the random seed."""
        return self._name + "_" + self._md5 + "_r{seed}.h5"

    def prepare(
        self,
        flush: Optional[Sequence[str]] = None,
        keep: Optional[Sequence[str]] = None,
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
            keep = [k for k in self._array_state if k not in flush]
        elif flush is None:
            flush = [
                k
                for k in self._array_state
                if k not in keep and self._array_state[k].initialized
            ]

        flush = flush or []
        keep = keep or []

        for k in flush:
            self._remove_array(k, force)

        # Accessing the array loads it into memory.
        for k in keep:
            getattr(self, k)

    def _remove_array(self, k, force=False):
        state = self._array_state[k]

        if not state.initialized and k in self._array_structure:
            warnings.warn(f"Trying to remove array that isn't yet created: {k}")
            return

        if state.computed_in_mem and not state.on_disk and not force:
            raise OSError(
                f"Trying to purge array '{k}' from memory that hasn't been stored! Use force=True if you meant to do this."
            )

        if state.c_has_active_memory:
            lib.free(getattr(self._cstruct, k))

        delattr(self, k)
        state.initialized = False

    def __getattr__(self, item):
        """Gets arrays that aren't already in memory."""
        # Have to use __dict__ here to test membership, otherwise we get recursion error.
        if "_array_state" not in self.__dict__ or item not in self._array_state:
            raise self.__getattribute__(item)

        if not self._array_state[item].on_disk:
            raise OSError(
                f"Cannot get {item} as it is not in memory, and this object is not cached to disk."
            )

        self.read(fname=self.path, keys=[item])
        return getattr(self, item)

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

    @property
    def filename(self):
        """The base filename of this object."""
        if self._random_seed is None:
            raise AttributeError("filename not defined until random_seed has been set")

        return self._fname_skeleton.format(seed=self.random_seed)

    def _get_fname(self, direc=None):
        direc = path.abspath(path.expanduser(direc or config["direc"]))
        return path.join(direc, self.filename)

    def _find_file_without_seed(self, direc):
        allfiles = glob.glob(path.join(direc, self._fname_skeleton.format(seed="*")))

        if allfiles:
            return allfiles[0]
        else:
            return None

    def find_existing(self, direc=None):
        """
        Try to find existing boxes which match the parameters of this instance.

        Parameters
        ----------
        direc : str, optional
            The directory in which to search for the boxes. By default, this is the
            centrally-managed directory, given by the ``config.yml`` in ``~/.21cmfast/``.

        Returns
        -------
        str
            The filename of an existing set of boxes, or None.
        """
        # First, if appropriate, find a file without specifying seed.
        # Need to do this first, otherwise the seed will be chosen randomly upon
        # choosing a filename!
        direc = path.expanduser(direc or config["direc"])

        if not self._random_seed:
            f = self._find_file_without_seed(direc)
            if f and self._check_parameters(f):
                return f
        else:
            f = self._get_fname(direc)
            if path.exists(f) and self._check_parameters(f):
                return f
        return None

    def _check_parameters(self, fname):
        with h5py.File(fname, "r") as f:
            for k in self._inputs + ("_global_params",):
                q = getattr(self, k)

                # The key name as it should appear in file.
                kfile = k.lstrip("_")

                # If this particular variable is set to None, this is interpreted
                # as meaning that we don't care about matching it to file.
                if q is None:
                    continue

                if (
                    not isinstance(q, StructWithDefaults)
                    and not isinstance(q, StructInstanceWrapper)
                    and f.attrs[kfile] != q
                ):
                    return False
                elif isinstance(q, (StructWithDefaults, StructInstanceWrapper)):
                    grp = f[kfile]

                    dct = q.self if isinstance(q, StructWithDefaults) else q
                    for kk, v in dct.items():
                        if kk not in self._filter_params:
                            file_v = grp.attrs[kk]
                            if file_v == "none":
                                file_v = None
                            if file_v != v:
                                logger.debug("For file %s:" % fname)
                                logger.debug(
                                    f"\tThough md5 and seed matched, the parameter {kk} did not match,"
                                    f" with values {file_v} and {v} in file and user respectively"
                                )
                                return False
        return True

    def exists(self, direc=None):
        """
        Return a bool indicating whether a box matching the parameters of this instance is in cache.

        Parameters
        ----------
        direc : str, optional
            The directory in which to search for the boxes. By default, this is the
            centrally-managed directory, given by the ``config.yml`` in ``~/.21cmfast/``.
        """
        return self.find_existing(direc) is not None

    def write(
        self,
        direc=None,
        fname: Union[str, Path, None, h5py.File, h5py.Group] = None,
        write_inputs=True,
        mode="w",
    ):
        """
        Write the struct in standard HDF5 format.

        Parameters
        ----------
        direc : str, optional
            The directory in which to write the boxes. By default, this is the
            centrally-managed directory, given by the ``config.yml`` in ``~/.21cmfast/``.
        fname : str, optional
            The filename to write to. By default creates a unique filename from the hash.
        write_inputs : bool, optional
            Whether to write the inputs to the file. Can be useful to set to False if
            the input file already exists and has parts already written.
        """
        if not all(v.computed for v in self._array_state.values()):
            raise OSError(
                "Not all boxes have been computed (or maybe some have been purged). Cannot write."
                f"Non-computed boxes: {[k for k, v in self._array_state.items() if not v.computed]}"
            )

        if not self._random_seed:
            raise ValueError(
                "Attempting to write when no random seed has been set. "
                "Struct has been 'computed' inconsistently."
            )

        if not write_inputs:
            mode = "a"

        try:
            if not isinstance(fname, (h5py.File, h5py.Group)):
                direc = path.expanduser(direc or config["direc"])

                if not path.exists(direc):
                    makedirs(direc)

                fname = fname or self._get_fname(direc)
                if not path.isabs(fname):
                    fname = path.abspath(path.join(direc, fname))

                fl = h5py.File(fname, mode)
            else:
                fl = fname

            try:
                # Save input parameters to the file
                if write_inputs:
                    for k in self._inputs + ("_global_params",):
                        q = getattr(self, k)

                        kfile = k.lstrip("_")

                        if isinstance(q, (StructWithDefaults, StructInstanceWrapper)):
                            grp = fl.create_group(kfile)
                            dct = q.self if isinstance(q, StructWithDefaults) else q
                            for kk, v in dct.items():
                                if kk not in self._filter_params:
                                    try:
                                        grp.attrs[kk] = "none" if v is None else v
                                    except TypeError:
                                        raise TypeError(
                                            f"key {kk} with value {v} is not able to be written to HDF5 attrs!"
                                        )
                        else:
                            fl.attrs[kfile] = q

                    # Write 21cmFAST version to the file
                    fl.attrs["version"] = __version__

                # Save the boxes to the file
                boxes = fl.create_group(self._name)

                self.write_data_to_hdf5_group(boxes)

            finally:
                if not isinstance(fname, (h5py.File, h5py.Group)):
                    fl.close()
                    self._paths.insert(0, Path(fname))

        except OSError as e:
            logger.warning(
                f"When attempting to write {self.__class__.__name__} to file, write failed with the following error. Continuing without caching."
            )

            logger.warning(e)

    def write_data_to_hdf5_group(self, group: h5py.Group):
        """
        Write out this object to a particular HDF5 subgroup.

        Parameters
        ----------
        group
            The HDF5 group into which to write the object.
        """
        # Go through all fields in this struct, and save
        for k, state in self._array_state.items():
            group.create_dataset(k, data=getattr(self, k))
            state.on_disk = True

        for k in self.primitive_fields:
            group.attrs[k] = getattr(self, k)

    def save(self, fname=None, direc=".", h5_group=None):
        """Save the box to disk.

        In detail, this just calls write, but changes the default directory to the
        local directory. This is more user-friendly, while :meth:`write` is for
        automatic use under-the-hood.

        Parameters
        ----------
        fname : str, optional
            The filename to write. Can be an absolute or relative path. If relative,
            by default it is relative to the current directory (otherwise relative
            to ``direc``). By default, the filename is auto-generated as unique to
            the set of parameters that go into producing the data.
        direc : str, optional
            The directory into which to write the data. By default the current directory.
            Ignored if ``fname`` is an absolute path.
        """
        # If fname is absolute path, then get direc from it, otherwise assume current dir.
        if path.isabs(fname):
            direc = path.dirname(fname)
            fname = path.basename(fname)

        if h5_group is not None:
            if not path.isabs(fname):
                fname = path.abspath(path.join(direc, fname))

            fl = h5py.File(fname, "a")

            try:
                grp = fl.create_group(h5_group)
                self.write(direc, grp)
            finally:
                fl.close()
        else:
            self.write(direc, fname)

    def _get_path(
        self, direc: Union[str, Path, None] = None, fname: Union[str, Path, None] = None
    ) -> Path:
        if direc is None and fname is None and self.path:
            return self.path

        if fname is None:
            pth = self.find_existing(direc)

            if pth is None:
                raise OSError("No boxes exist for these parameters.")
        else:
            direc = Path(direc or config["direc"]).expanduser()
            fname = Path(fname)
            pth = fname if fname.exists() else direc / fname
        return pth

    def read(
        self,
        direc: Union[str, Path, None] = None,
        fname: Union[str, Path, None, h5py.File, h5py.Group] = None,
        keys: Optional[Sequence[str]] = None,
    ):
        """
        Try find and read existing boxes from cache, which match the parameters of this instance.

        Parameters
        ----------
        direc
            The directory in which to search for the boxes. By default, this is the
            centrally-managed directory, given by the ``config.yml`` in ``~/.21cmfast/``.
        fname
            The filename to read. By default, use the filename associated with this
            object. Can be an open h5py File or Group, which will be directly written to.
        keys
            The names of boxes to read in (can be a subset). By default, read everything.
        """
        if not isinstance(fname, (h5py.File, h5py.Group)):
            pth = self._get_path(direc, fname)
            fl = h5py.File(pth, "r")
        else:
            fl = fname

        keys = keys or []
        try:
            try:
                boxes = fl[self._name]
            except KeyError:
                raise OSError(
                    f"While trying to read in {self._name}, the file exists, but does not have the "
                    "correct structure."
                )

            # Set our arrays.
            for k in boxes.keys():
                self._array_state[k].on_disk = True
                if k in keys:
                    setattr(self, k, boxes[k][...])
                    self._array_state[k].computed_in_mem = True
                    setattr(self._cstruct, k, self._ary2buf(getattr(self, k)))

            for k in boxes.attrs.keys():
                if k == "version":
                    version = ".".join(boxes.attrs[k].split(".")[:2])
                    patch = ".".join(boxes.attrs[k].split(".")[2:])

                    if version != ".".join(__version__.split(".")[:2]):
                        # Ensure that the major and minor versions are the same.
                        warnings.warn(
                            f"The file {pth} is out of date (version = {version}.{patch}). "
                            f"Consider using another box and removing it!"
                        )

                    self.version = version
                    self.patch_version = patch

                setattr(self, k, boxes.attrs[k])
                try:
                    setattr(self._cstruct, k, getattr(self, k))
                except AttributeError:
                    pass

            # Need to make sure that the seed is set to the one that's read in.
            seed = fl.attrs["random_seed"]
            self._random_seed = int(seed)
        finally:
            self.__expose()
            if isinstance(fl, h5py.File):
                self._paths.insert(0, Path(fl.filename))
            else:
                self._paths.insert(0, Path(fl.file.filename))

            if not isinstance(fname, (h5py.File, h5py.Group)):
                fl.close()

    @classmethod
    def from_file(
        cls,
        fname,
        direc=None,
        load_data=True,
        h5_group: Union[str, None] = None,
        arrays_to_load=None,
    ):
        """Create an instance from a file on disk.

        Parameters
        ----------
        fname : str, optional
            Path to the file on disk. May be relative or absolute.
        direc : str, optional
            The directory from which fname is relative to (if it is relative). By
            default, will be the cache directory in config.
        load_data : bool, optional
            Whether to read in the data when creating the instance. If False, a bare
            instance is created with input parameters -- the instance can read data
            with the :func:`read` method.
        h5_group
            The path to the group within the file in which the object is stored.
        """
        direc = path.expanduser(direc or config["direc"])

        if not path.exists(fname):
            fname = path.join(direc, fname)

        with h5py.File(fname, "r") as fl:
            if h5_group is not None:
                self = cls(**cls._read_inputs(fl[h5_group]))
            else:
                self = cls(**cls._read_inputs(fl))

        if not load_data:
            arrays_to_load = []

        if h5_group is not None:
            with h5py.File(fname, "r") as fl:
                self.read(fname=fl[h5_group], keys=arrays_to_load)
        else:
            self.read(fname=fname, keys=arrays_to_load)

        return self

    @classmethod
    def _read_inputs(cls, grp: Union[h5py.File, h5py.Group]):
        input_classes = [c.__name__ for c in StructWithDefaults.__subclasses__()]

        # Read the input parameter dictionaries from file.
        kwargs = {}
        for k in cls._inputs:
            kfile = k.lstrip("_")
            input_class_name = snake_to_camel(kfile)

            if input_class_name in input_classes:
                input_class = StructWithDefaults.__subclasses__()[
                    input_classes.index(input_class_name)
                ]
                subgrp = grp[kfile]
                kwargs[k] = input_class(
                    {k: v for k, v in dict(subgrp.attrs).items() if v != "none"}
                )
            else:
                kwargs[kfile] = grp.attrs[kfile]
        return kwargs

    def __repr__(self):
        """Return a fully unique representation of the instance."""
        # This is the class name and all parameters which belong to C-based input structs,
        # eg. InitialConditions(HII_DIM:100,SIGMA_8:0.8,...)
        # eg. InitialConditions(HII_DIM:100,SIGMA_8:0.8,...)
        return f"{self._seedless_repr()}_random_seed={self._random_seed}"

    def _seedless_repr(self):
        # The same as __repr__ except without the seed.
        return (
            (
                self._name
                + "("
                + "; ".join(
                    (
                        repr(v)
                        if isinstance(v, StructWithDefaults)
                        else (
                            v.filtered_repr(self._filter_params)
                            if isinstance(v, StructInstanceWrapper)
                            else k.lstrip("_")
                            + ":"
                            + (
                                float_to_string_precision(
                                    v, config["cache_param_sigfigs"]
                                )
                                if isinstance(v, (float, np.float32))
                                else repr(v)
                            )
                        )
                    )
                    for k, v in [
                        (k, getattr(self, k))
                        for k in self._inputs + ("_global_params",)
                        if k != "_random_seed"
                    ]
                )
            )
            + f"; v{self.version}"
            + ")"
        )

    def __str__(self):
        """Return a human-readable representation of the instance."""
        # this is *not* a unique representation, and doesn't include global params.
        return (
            self._name
            + "("
            + ";\n\t".join(
                (
                    repr(v)
                    if isinstance(v, StructWithDefaults)
                    else k.lstrip("_") + ":" + repr(v)
                )
                for k, v in [(k, getattr(self, k)) for k in self._inputs]
            )
        ) + ")"

    def __hash__(self):
        """Return a unique hsh for this instance, even global params and random seed."""
        return hash(repr(self))

    @property
    def _md5(self):
        """Return a unique hsh of the object, *not* taking into account the random seed."""
        return md5(self._seedless_repr().encode()).hexdigest()

    def __eq__(self, other):
        """Check equality with another object via its __repr__."""
        return repr(self) == repr(other)

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
        return any(v.computed for v in self._array_state.values())

    def ensure_arrays_computed(self, *arrays, load=False) -> bool:
        """Check if the given arrays are computed (not just initialized)."""
        if not self.is_computed:
            return False

        computed = all(self._array_state[k].computed for k in arrays)

        if computed and load:
            self.prepare(keep=arrays, flush=[])

        return computed

    def ensure_arrays_inited(self, *arrays, init=False) -> bool:
        """Check if the given arrays are initialized (or computed)."""
        inited = all(self._array_state[k].initialized for k in arrays)

        if init and not inited:
            self._init_arrays()

        return True

    @abstractmethod
    def get_required_input_arrays(self, input_box) -> List[str]:
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
        out = f"\n{indent}{self.__class__.__name__}\n"

        out += "".join(
            f"{indent}    {fieldname:>15}: {getattr(self, fieldname, 'non-existent')}\n"
            for fieldname in self.primitive_fields
        )

        for fieldname, state in self._array_state.items():
            if not state.initialized:
                out += f"{indent}    {fieldname:>15}: uninitialized\n"
            elif not state.computed:
                out += f"{indent}    {fieldname:>15}: initialized\n"
            elif not state.computed_in_mem:
                out += f"{indent}    {fieldname:>15}: computed on disk\n"
            else:
                x = getattr(self, fieldname).flatten()
                if len(x) > 0:
                    out += f"{indent}    {fieldname:>15}: {x[0]:1.4e}, {x[-1]:1.4e}, {x.min():1.4e}, {x.max():1.4e}, {np.mean(x):1.4e}\n"
                else:
                    out += f"{indent}    {fieldname:>15}: size zero\n"

        return out

    @classmethod
    def _log_call_arguments(cls, *args):
        logger.debug(f"Calling {cls._c_compute_function.__name__} with following args:")

        for arg in args:
            if isinstance(arg, OutputStruct):
                for line in arg.summarize(indent=1).split("\n"):
                    logger.debug(line)
            elif isinstance(arg, StructWithDefaults):
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
        self, *args, hooks: Optional[Dict[Union[str, Callable], Dict[str, Any]]] = None
    ):
        """Compute the actual function that fills this struct."""
        # Check that all required inputs are really computed, and load them into memory
        # if they're not already.
        self._ensure_arguments_exist(*args)

        # Write a detailed message about call arguments if debug turned on.
        if logger.getEffectiveLevel() <= logging.DEBUG:
            self._log_call_arguments(*args)

        # Construct the args. All StructWrapper objects need to actually pass their
        # underlying cstruct, rather than themselves. OutputStructs also pass the
        # class in that's calling this.
        inputs = [arg() if isinstance(arg, StructWrapper) else arg for arg in args]

        # Ensure we haven't already tried to compute this instance.
        if self.is_computed:
            raise ValueError(
                f"You are trying to compute {self.__class__.__name__}, but it has already been computed."
            )

        # Perform the C computation
        try:
            exitcode = self._c_compute_function(*inputs, self())
        except TypeError as e:
            logger.error(
                f"Arguments to {self._c_compute_function.__name__}: "
                f"{[arg() if isinstance(arg, StructWrapper) else arg for arg in args]}"
            )
            raise e

        _process_exitcode(exitcode, self._c_compute_function, args)

        # Ensure memory created in C gets mapped to numpy arrays in this struct.
        for k, state in self._array_state.items():
            if state.initialized:
                state.computed_in_mem = True

        self.__memory_map()
        self.__expose()

        # Optionally do stuff with the result (like writing it)
        self._call_hooks(hooks)

        return self

    def _call_hooks(self, hooks):
        if hooks is None:
            hooks = {"write": {"direc": config["direc"]}}

        for hook, params in hooks.items():
            if callable(hook):
                hook(self, **params)
            else:
                getattr(self, hook)(**params)

    def __memory_map(self):
        shapes = self._c_shape(self._cstruct)
        for item in self._c_based_pointers:
            setattr(self, item, asarray(getattr(self._cstruct, item), shapes[item]))
            self._array_state[item].c_memory = True
            self._array_state[item].computed_in_mem = True

    def __del__(self):
        """Safely delete the object and its C-allocated memory."""
        for k in self._c_based_pointers:
            if self._array_state[k].c_has_active_memory:
                lib.free(getattr(self._cstruct, k))


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
        try:
            setattr(self._cobj, name, value)
        except AttributeError:
            pass
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
            + ";".join(k + "=" + str(v) for k, v in sorted(self.items()))
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
                k + "=" + str(v)
                for k, v in sorted(self.items())
                if k not in filter_params
            )
        ) + ")"


def _check_compatible_inputs(*datasets, ignore=["redshift"]):
    """Ensure that all defined input parameters for the provided datasets are equal.

    Parameters
    ----------
    datasets : list of :class:`~_utils.OutputStruct`
        A number of output datasets to cross-check.
    ignore : list of str
        Attributes to ignore when ensuring that parameter inputs are the same.

    Raises
    ------
    ValueError :
        If datasets are not compatible.
    """
    done = []  # keeps track of inputs we've checked so we don't double check.

    for i, d in enumerate(datasets):
        # If a dataset is None, just ignore and move on.
        if d is None:
            continue

        # noinspection PyProtectedMember
        for inp in d._inputs:
            # Skip inputs that we want to ignore
            if inp in ignore:
                continue

            if inp not in done:
                for j, d2 in enumerate(datasets[(i + 1) :]):
                    if d2 is None:
                        continue

                    # noinspection PyProtectedMember
                    if inp in d2._inputs and getattr(d, inp) != getattr(d2, inp):
                        raise ValueError(
                            f"""
                            {d.__class__.__name__} and {d2.__class__.__name__} are incompatible.
                            {inp}: {getattr(d, inp)}
                            vs.
                            {inp}: {getattr(d2, inp)}
                            """
                        )
                done += [inp]
