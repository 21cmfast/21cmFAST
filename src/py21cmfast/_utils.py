"""Utilities that help with wrapping various C structures."""
import copy
import glob
import h5py
import logging
import numpy as np
import warnings
from cffi import FFI
from hashlib import md5
from os import makedirs, path
from pathlib import Path

from . import __version__
from ._cfg import config

_ffi = FFI()

logger = logging.getLogger("21cmFAST")


class ParameterError(RuntimeError):
    """An exception representing a bad choice of parameters."""

    def __init__(self):
        default_message = "21cmFAST does not support this combination of parameters."
        super().__init__(default_message)


class FatalCError(Exception):
    """An exception representing something going wrong in C."""

    def __init__(self, msg=None):
        default_message = "21cmFAST is exiting."
        super().__init__(msg or default_message)


SUCCESS = 0
IOERROR = 1
GSLERROR = 2
VALUEERROR = 3
PARAMETERERROR = 4
MEMORYALLOCERROR = 5
FILEERROR = 6


def _process_exitcode(exitcode, fnc, args):
    """Determine what happens for different values of the (integer) exit code from a C function."""
    if exitcode != SUCCESS:
        logger.error(f"In function: {fnc.__name__}.  Arguments: {args}")

        if exitcode in (GSLERROR, PARAMETERERROR):
            raise ParameterError
        elif exitcode in (IOERROR, VALUEERROR, MEMORYALLOCERROR, FILEERROR):
            raise FatalCError
        else:  # Unknown C code
            raise FatalCError("Unknown error in C. Please report this error!")


ctype2dtype = {}

# Integer types
for prefix in ("int", "uint"):
    for log_bytes in range(4):
        ctype = "%s%d_t" % (prefix, 8 * (2 ** log_bytes))
        dtype = "%s%d" % (prefix[0], 2 ** log_bytes)
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
    def get_fields(cls, cstruct=None):
        """Obtain the C-side fields of this struct."""
        if cstruct is None:
            cstruct = cls._ffi.new("struct " + cls._get_name() + "*")
        return cls._ffi.typeof(cstruct[0]).fields

    @classmethod
    def get_fieldnames(cls, cstruct=None):
        """Obtain the C-side field names of this struct."""
        fields = cls.get_fields(cstruct)
        return [f for f, t in fields]

    @classmethod
    def get_pointer_fields(cls, cstruct=None):
        """Obtain all pointer fields of the struct (typically simulation boxes)."""
        return [f for f, t in cls.get_fields(cstruct) if t.type.kind == "pointer"]

    @property
    def fields(self):
        """List of fields of the underlying C struct (a list of tuples of "name, type")."""
        return self.get_fields(self._cstruct)

    @property
    def fieldnames(self):
        """List names of fields of the underlying C struct."""
        return [f for f, t in self.fields]

    @property
    def pointer_fields(self):
        """List of names of fields which have pointer type in the C struct."""
        return [f for f, t in self.fields if t.type.kind == "pointer"]

    @property
    def primitive_fields(self):
        """List of names of fields which have primitive type in the C struct."""
        return [f for f, t in self.fields if t.type.kind == "primitive"]

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
                    "optional positional argument for %s must be None, dict, or an instance of itself"
                    % self.__class__.__name__
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
            logger.warning(
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
                print("For key %s, value %s:" % (key, val))
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
            + ", ".join(sorted(k + ":" + str(v) for k, v in self.defining_dict.items()))
            + ")"
        )

    def __eq__(self, other):
        """Check whether this instance is equal to another object (by checking the __repr__)."""
        return self.__repr__() == repr(other)

    def __hash__(self):
        """Generate a unique hsh for the instance."""
        return hash(self.__repr__())


def snake_to_camel(word: str, publicize: bool = True):
    """Convert snake case to camel case."""
    if publicize:
        word = word.lstrip("_")
    return "".join(x.capitalize() or "_" for x in word.split("_"))


def camel_to_snake(word: str, depublicize: bool = False):
    """Convert came case to snake case."""
    word = "".join(["_" + i.lower() if i.isupper() else i for i in word])

    if not depublicize:
        word = word.lstrip("_")

    return word


def get_all_subclasses(cls):
    """Get a list of all subclasses of a given class, recursively."""
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


class OutputStruct(StructWrapper):
    """Base class for any class that wraps a C struct meant to be output from a C function."""

    _meta = True
    _fields_ = []
    _global_params = None
    _inputs = ["user_params", "cosmo_params", "_random_seed"]
    _filter_params = ["external_table_path"]
    _c_based_pointers = ()
    _c_compute_function = None
    _c_free_function = None

    _TYPEMAP = {"float32": "float *", "float64": "double *", "int32": "int *"}

    def __init__(self, *, random_seed=None, init=False, dummy=False, **kwargs):
        super().__init__()

        self.filled = False
        self.version = ".".join(__version__.split(".")[:2])
        self.patch_version = ".".join(__version__.split(".")[2:])

        self._random_seed = random_seed

        for k in self._inputs:

            if not hasattr(self, k):
                try:
                    setattr(self, k, kwargs.pop(k))
                except KeyError:
                    raise KeyError(
                        "%s requires the keyword argument %s"
                        % (self.__class__.__name__, k)
                    )

        if kwargs:
            warnings.warn(
                "%s received the following unexpected arguments: %s"
                % (self.__class__.__name__, list(kwargs.keys()))
            )

        self.dummy = dummy

        if init:
            self._init_cstruct()

    def _c_shape(self, cstruct):
        """Return a dictionary of field: shape for arrays allocated within C."""
        return {}

    @classmethod
    def _implementations(cls):
        all_classes = get_all_subclasses(cls)
        return [c for c in all_classes if not c._meta]

    def _init_arrays(self):  # pragma: nocover
        """Abstract base method for initializing any arrays that the structure has."""
        pass

    @property
    def random_seed(self):
        """The random seed for this particular instance."""
        if self._random_seed is None:
            self._random_seed = int(np.random.randint(1, int(1e12)))

        return self._random_seed

    @property
    def arrays_initialized(self):
        """Whether all necessary arrays are initialized.

        .. note:: This must be true before passing to a C function.
        """
        # This assumes that all pointer fields will be arrays...
        for k in self.pointer_fields:
            if k in self._c_based_pointers:
                continue
            if not hasattr(self, k):
                return False
            elif getattr(self._cstruct, k) == self._ffi.NULL:
                return False
        return True

    def _init_cstruct(self):
        if not self.filled:
            self._init_arrays()

        for k in self.pointer_fields:
            if k not in self._c_based_pointers:
                setattr(self._cstruct, k, self._ary2buf(getattr(self, k)))
        for k in self.primitive_fields:
            try:
                setattr(self._cstruct, k, getattr(self, k))
            except AttributeError:
                pass

        if not self.arrays_initialized:
            raise AttributeError(
                f"{self.__class__.__name__} is ill-defined. It has not initialized all necessary arrays."
            )

    def _ary2buf(self, ary):
        if not isinstance(ary, np.ndarray):
            raise ValueError("ary must be a numpy array")
        return self._ffi.cast(
            OutputStruct._TYPEMAP[ary.dtype.name], self._ffi.from_buffer(ary)
        )

    def __call__(self):
        """Initialize/allocate a fresh C struct in memory and return it."""
        if not (self.arrays_initialized or self.dummy):
            self._init_cstruct()

        return self._cstruct

    def _expose(self):
        """Expose the non-array primitives of the ctype to the top-level object."""
        if not self.filled:
            raise Exception(
                "You need to have actually called the C code before the primitives can be exposed."
            )
        for k in self.primitive_fields:
            setattr(self, k, getattr(self._cstruct, k))

    @property
    def _fname_skeleton(self):
        """The filename without specifying the random seed."""
        return self._name + "_" + self._md5 + "_r{seed}.h5"

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
            for k in self._inputs + ["_global_params"]:
                q = getattr(self, k)

                # The key name as it should appear in file.
                kfile = k.lstrip("_")

                # If this particular variable is set to None, this is interpreted
                # as meaning that we don't care about matching it to file.
                if q is None:
                    continue

                if isinstance(q, StructWithDefaults) or isinstance(
                    q, StructInstanceWrapper
                ):
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
                else:
                    if f.attrs[kfile] != q:
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

    def write(self, direc=None, fname=None, write_inputs=True, mode="w"):
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
        if not self.filled:
            raise IOError("The boxes have not yet been computed.")

        if not self._random_seed:
            raise ValueError(
                "Attempting to write when no random seed has been set. "
                "Struct has been 'filled' inconsistently."
            )

        if not write_inputs:
            mode = "a"

        try:
            direc = path.expanduser(direc or config["direc"])

            if not path.exists(direc):
                makedirs(direc)

            fname = fname or self._get_fname(direc)
            if not path.isabs(fname):
                fname = path.abspath(path.join(direc, fname))

            with h5py.File(fname, mode) as f:
                # Save input parameters to the file
                if write_inputs:
                    for k in self._inputs + ["_global_params"]:
                        q = getattr(self, k)

                        kfile = k.lstrip("_")

                        if isinstance(q, StructWithDefaults) or isinstance(
                            q, StructInstanceWrapper
                        ):
                            grp = f.create_group(kfile)
                            if isinstance(q, StructWithDefaults):
                                # using self allows to rebuild the object from HDF5 file.
                                dct = q.self
                            else:
                                dct = q

                            for kk, v in dct.items():
                                if kk not in self._filter_params:
                                    grp.attrs[kk] = "none" if v is None else v
                        else:
                            f.attrs[kfile] = q

                    # Write 21cmFAST version to the file
                    f.attrs["version"] = __version__

                # Save the boxes to the file
                boxes = f.create_group(self._name)

                self.write_data_to_hdf5_group(boxes)

        except OSError as e:
            logger.warning(
                "When attempting to write {} to file, write failed with the "
                "following error. Continuing without caching.".format(
                    self.__class__.__name__
                )
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
        for k in self.pointer_fields:
            group.create_dataset(k, data=getattr(self, k))

        for k in self.primitive_fields:
            group.attrs[k] = getattr(self, k)

    def save(self, fname=None, direc="."):
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

        self.write(direc, fname)

    def read(self, direc: [str, Path, None] = None, fname: [str, Path, None] = None):
        """
        Try find and read existing boxes from cache, which match the parameters of this instance.

        Parameters
        ----------
        direc
            The directory in which to search for the boxes. By default, this is the
            centrally-managed directory, given by the ``config.yml`` in ``~/.21cmfast/``.
        fname
            The filename to read. By default, use the filename associated with this
            object.
        """
        if self.filled:
            raise IOError("This data is already filled, no need to read in.")

        if fname is None:
            pth = self.find_existing(direc)

            if pth is None:
                raise IOError("No boxes exist for these parameters.")
        else:
            direc = Path(direc or config["direc"]).expanduser()
            fname = Path(fname)
            pth = fname if fname.exists() else direc / fname

        # Need to make sure arrays are initialized before reading in data to them.
        if not self.arrays_initialized:
            self._init_cstruct()

        with h5py.File(pth, "r") as f:
            try:
                boxes = f[self._name]
            except KeyError:
                raise IOError(
                    f"While trying to read in {self._name}, the file exists, but does not have the "
                    "correct structure."
                )

            # Fill our arrays.
            for k in boxes.keys():
                if k in self._c_based_pointers:
                    # C-based pointers can just be read straight in.
                    setattr(self, k, boxes[k][...])
                else:
                    # Other pointers should fill the already-instantiated arrays.
                    getattr(self, k)[...] = boxes[k][...]

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

            # Need to make sure that the seed is set to the one that's read in.
            seed = f.attrs["random_seed"]
            self._random_seed = seed

        self.filled = True
        self._expose()

    @classmethod
    def from_file(cls, fname, direc=None, load_data=True):
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
        """
        direc = path.expanduser(direc or config["direc"])

        if not path.exists(fname):
            fname = path.join(direc, fname)

        self = cls(**cls._read_inputs(fname))

        if load_data:
            self.read(fname=fname)
        return self

    @classmethod
    def _read_inputs(cls, fname):
        input_classes = [c.__name__ for c in StructWithDefaults.__subclasses__()]

        # Read the input parameter dictionaries from file.
        kwargs = {}
        with h5py.File(fname, "r") as fl:
            for k in cls._inputs:
                kfile = k.lstrip("_")
                input_class_name = snake_to_camel(kfile)

                if input_class_name in input_classes:
                    input_class = StructWithDefaults.__subclasses__()[
                        input_classes.index(input_class_name)
                    ]
                    grp = fl[kfile]
                    kwargs[k] = input_class(
                        {k: v for k, v in dict(grp.attrs).items() if v != "none"}
                    )
                else:
                    kwargs[kfile] = fl.attrs[kfile]
        return kwargs

    def __repr__(self):
        """Return a fully unique representation of the instance."""
        # This is the class name and all parameters which belong to C-based input structs,
        # eg. InitialConditions(HII_DIM:100,SIGMA_8:0.8,...)
        return self._seedless_repr() + "_random_seed={}".format(self._random_seed)

    def _seedless_repr(self):
        # The same as __repr__ except without the seed.
        return (
            self._name
            + "("
            + "; ".join(
                [
                    repr(v)
                    if isinstance(v, StructWithDefaults)
                    else (
                        v.filtered_repr(self._filter_params)
                        if isinstance(v, StructInstanceWrapper)
                        else k.lstrip("_") + ":" + repr(v)
                    )
                    for k, v in [
                        (k, getattr(self, k))
                        for k in self._inputs + ["_global_params"]
                        if k != "_random_seed"
                    ]
                ]
            )
            + "; v{}".format(self.version)
            + ")"
        )

    def __str__(self):
        """Return a human-readable representation of the instance."""
        # this is *not* a unique representation, and doesn't include global params.
        return (
            self._name
            + "("
            + ";\n\t".join(
                [
                    repr(v)
                    if isinstance(v, StructWithDefaults)
                    else k.lstrip("_") + ":" + repr(v)
                    for k, v in [(k, getattr(self, k)) for k in self._inputs]
                ]
            )
            + ")"
        )

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

    def compute(self, direc, *args, write=True):
        """Compute the actual function that fills this struct."""
        logger.debug(f"Calling {self._c_compute_function.__name__} with args: {args}")
        try:
            exitcode = self._c_compute_function(
                *[arg() if isinstance(arg, StructWrapper) else arg for arg in args],
                self(),
            )
        except TypeError as e:
            logger.error(
                f"Arguments to {self._c_compute_function.__name__}: "
                f"{[arg() if isinstance(arg, StructWrapper) else arg for arg in args]}"
            )
            raise e

        _process_exitcode(exitcode, self._c_compute_function, args)

        # Ensure memory created in C gets mapped to numpy arrays in this struct.
        self.filled = True
        self._memory_map()
        self._expose()

        # Optionally do stuff with the result (like writing it)
        if write:
            self.write(direc)

        return self

    def _memory_map(self):
        if not self.filled:
            warnings.warn("Do not call _memory_map yourself!")

        shapes = self._c_shape(self._cstruct)
        for item in self._c_based_pointers:
            setattr(self, item, asarray(getattr(self._cstruct, item), shapes[item]))

    def __del__(self):
        """Safely delete the object and its C-allocated memory."""
        if self._c_free_function is not None:
            self._c_free_function(self._cstruct)


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
            + ";".join([k + "=" + str(v) for k, v in sorted(self.items())])
            + ")"
        )

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
                [
                    k + "=" + str(v)
                    for k, v in sorted(self.items())
                    if k not in filter_params
                ]
            )
            + ")"
        )


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
                            "%s and %s are incompatible"
                            % (d.__class__.__name__, d2.__class__.__name__)
                        )
                done += [inp]
