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
from typing import Any, Sequence

from .. import __version__
from .._cfg import config
from ..c_21cmfast import lib
from ._utils import (
    asarray,
    float_to_string_precision,
    get_all_subclasses,
    snake_to_camel,
)
from .arraystate import ArrayState
from .exceptions import _process_exitcode

logger = logging.getLogger(__name__)


@attrs.define
class StructWrapper:
    """
    A base-class python wrapper for C structures (not instances of them).

    Provides simple methods for creating new instances and accessing field names and values.

    To implement wrappers of specific structures, make a subclass with the same name as the
    appropriate C struct (which must be defined in the C code that has been compiled to the ``ffi``
    object), *or* use an arbitrary name, but set the ``_name`` attribute to the C struct name.
    """

    _name: str = attrs.field(converter=str)

    _ffi = None

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
        return self.get_fields(self.cstruct)

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
            k: v for k, v in self.__dict__.items() if k not in ["_strings", "cstruct"]
        }


@attrs.define(frozen=True, kw_only=True)
class InputStruct(StructWrapper):
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

    @classmethod
    def new(cls, x: dict | InputStruct | None):
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
            return cls(**x)
        elif isinstance(x, InputStruct):
            return x
        elif isinstance(x, None):
            return cls()
        else:
            raise ValueError(
                f"Cannot instantiate {cls.__name__} with type {x.__class__}"
            )

    @cached_property
    def struct(self) -> StructWrapper:
        """The python-wrapped struct associated with this input object."""
        return StructWrapper(name=self.__class__.__name__)

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

    @property
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
            trns = transformers[k]
            out[k] = val if trns is None else trns(val)
        return out

    def __str__(self):
        """Human-readable string representation of the object."""
        d = self.asdict()
        biggest_k = max(len(k) for k in d)
        params = "\n    ".join(sorted(f"{k:<{biggest_k}}: {v}" for k, v in d.items()))
        return f"""{self.__class__.__name__}:
    {params}
    """


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
                except KeyError as e:
                    raise KeyError(
                        f"{self.__class__.__name__} requires the keyword argument {k}"
                    ) from e

        if kwargs:
            warnings.warn(
                f"{self.__class__.__name__} received the following unexpected "
                f"arguments: {list(kwargs.keys())}"
            )

        self.dummy = dummy
        self.initial = initial

        self._array_structure = self._get_box_structures()
        self._array_state = {k: ArrayState() for k in self._array_structure} | {
            k: ArrayState() for k in self._c_based_pointers
        }
        for k in self._array_structure:
            if k not in self.pointer_fields:
                raise TypeError(f"Key {k} in {self} not a defined pointer field in C.")

    @property
    def path(self) -> tuple[None, Path]:
        """The path to an on-disk version of this object."""
        if not self._paths:
            return None

        for pth in self._paths:
            if pth.exists():
                return pth

        logger.info(f"All paths that defined {self} have been deleted on disk.")
        return None

    @abstractmethod
    def _get_box_structures(self) -> dict[str, dict | tuple[int]]:
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

    def _c_shape(self, cstruct) -> dict[str, tuple[int]]:
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
                setattr(self.cstruct, k, self._ary2buf(getattr(self, k)))

        for k in self.primitive_fields:
            with contextlib.suppress(AttributeError):
                setattr(self.cstruct, k, getattr(self, k))

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

        return self.cstruct

    def __expose(self):
        """Expose the non-array primitives of the ctype to the top-level object."""
        for k in self.primitive_fields:
            setattr(self, k, getattr(self.cstruct, k))

    @property
    def _fname_skeleton(self):
        """The filename without specifying the random seed."""
        return f"{self._name}_{self._md5}" + "_r{seed}.h5"

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
            lib.free(getattr(self.cstruct, k))

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
        direc = Path(direc or config["direc"]).expanduser().absolute()
        return direc / self.filename

    def _find_file_without_seed(self, direc):
        if allfiles := list(Path(direc).glob(self._fname_skeleton.format(seed="*"))):
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
        direc = Path(direc or config["direc"]).expanduser()

        if not self._random_seed:
            f = self._find_file_without_seed(direc)
            if f and self._check_parameters(f):
                return f
        else:
            f = self._get_fname(direc)
            if f.exists() and self._check_parameters(f):
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
                    not isinstance(q, InputStruct)
                    and not isinstance(q, StructInstanceWrapper)
                    and f.attrs[kfile] != q
                ):
                    return False
                elif isinstance(q, (InputStruct, StructInstanceWrapper)):
                    grp = f[kfile]

                    dct = q.self if isinstance(q, InputStruct) else q
                    for kk, v in dct.items():
                        if kk not in self._filter_params:
                            file_v = grp.attrs[kk]
                            if file_v == "none":
                                file_v = None
                            if file_v != v:
                                logger.debug(f"For file {fname}:")
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
        fname: str | Path | None | h5py.File | h5py.Group = None,
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
                direc = Path(direc or config["direc"]).expanduser()

                if not direc.exists():
                    direc.mkdir()

                fname = Path(fname or self._get_fname(direc))
                if not fname.is_absolute():
                    fname = direc / fname

                fl = h5py.File(fname, mode)
            else:
                fl = fname

            try:
                # Save input parameters to the file
                if write_inputs:
                    for k in self._inputs + ("_global_params",):
                        q = getattr(self, k)

                        kfile = k.lstrip("_")

                        if isinstance(q, (InputStruct, StructInstanceWrapper)):
                            grp = fl.create_group(kfile)
                            dct = q.self if isinstance(q, InputStruct) else q
                            for kk, v in dct.items():
                                if kk not in self._filter_params:
                                    try:
                                        grp.attrs[kk] = "none" if v is None else v
                                    except TypeError as e:
                                        raise TypeError(
                                            f"key {kk} with value {v} is not able to be written to HDF5 attrs!"
                                        ) from e
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
        fname = Path(fname)
        if fname.is_absolute():
            direc = fname.parent
            fname = fname.name

        if h5_group is not None:
            if not fname.is_absolute():
                fname = direc / fname

            fl = h5py.File(fname, "a")

            try:
                grp = fl.create_group(h5_group)
                self.write(direc, grp)
            finally:
                fl.close()
        else:
            self.write(direc, fname)

    def _get_path(
        self, direc: str | Path | None = None, fname: str | Path | None = None
    ) -> Path:
        if direc is None and fname is None and self.path:
            return self.path

        if fname is None:
            pth = self.find_existing(direc)

            if pth is None:
                raise OSError(f"No boxes exist for these parameters. {pth} {direc}")
        else:
            direc = Path(direc or config["direc"]).expanduser()
            fname = Path(fname)
            pth = fname if fname.exists() else direc / fname
        return pth

    def read(
        self,
        direc: str | Path | None = None,
        fname: str | Path | None | h5py.File | h5py.Group = None,
        keys: Sequence[str] | None = None,
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
            The names of boxes to read in (can be a subset). By default, read nothing.
            If `None` is explicitly passed, read everything
        """
        if not isinstance(fname, (h5py.File, h5py.Group)):
            pth = self._get_path(direc, fname)
            fl = h5py.File(pth, "r")
        else:
            fl = fname

        if keys is None:
            keys = self._array_structure
        try:
            try:
                boxes = fl[self._name]
            except KeyError as e:
                raise OSError(
                    f"While trying to read in {self._name}, the file exists, but does not have the "
                    "correct structure."
                ) from e

            # Set our arrays.
            for k in boxes.keys():
                self._array_state[k].on_disk = True
                if k in keys:
                    setattr(self, k, boxes[k][...])
                    self._array_state[k].computed_in_mem = True
                    setattr(self.cstruct, k, self._ary2buf(getattr(self, k)))

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
                with contextlib.suppress(AttributeError):
                    setattr(self.cstruct, k, getattr(self, k))

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
        h5_group: str | None = None,
        arrays_to_load=(),
    ):
        """Create an instance from a file on disk.

        Parameters
        ----------
        fname : str, optional
            Path to the file on disk. May be relative or absolute.
        direc : str, optional
            The directory from which fname is relative to (if it is relative). By
            default, will be the cache directory in config.
        h5_group
            The path to the group within the file in which the object is stored.
        arrays_to_load : list of str, optional
            A list of array names to load into memory
            If the list is empty (default), a bare instance is created with input parameters
            -- the instance can read data with the :func:`read` method.
            If `None` is explicitly passed, all arrays are loaded into memory
        """
        direc = Path(direc or config["direc"]).expanduser()
        fname = Path(fname)

        if not fname.exists():
            fname = direc / fname

        with h5py.File(fname, "r") as fl:
            if h5_group is not None:
                self = cls(**cls._read_inputs(fl[h5_group]))
            else:
                self = cls(**cls._read_inputs(fl))

        if h5_group is not None:
            with h5py.File(fname, "r") as fl:
                self.read(fname=fl[h5_group], keys=arrays_to_load)
        else:
            self.read(fname=fname, keys=arrays_to_load)

        return self

    @classmethod
    def _read_inputs(cls, grp: h5py.File | h5py.Group):
        input_classes = [c.__name__ for c in InputStruct.__subclasses__()]

        # Read the input parameter dictionaries from file.
        kwargs = {}
        for k in cls._inputs:
            kfile = k.lstrip("_")
            input_class_name = snake_to_camel(kfile)

            if input_class_name in input_classes:
                input_class = InputStruct.__subclasses__()[
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
                        if isinstance(v, InputStruct)
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
                    if isinstance(v, InputStruct)
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
    def get_required_input_arrays(self, input_box) -> list[str]:
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
        shapes = self._c_shape(self.cstruct)
        for item in self._c_based_pointers:
            setattr(self, item, asarray(getattr(self.cstruct, item), shapes[item]))
            self._array_state[item].c_memory = True
            self._array_state[item].computed_in_mem = True

    def __del__(self):
        """Safely delete the object and its C-allocated memory."""
        # TODO: figure out why this breaks the C memory if purged, _remove_array should set .initialised to false,
        #       which should make .c_has_active_memory false
        for k in self._c_based_pointers:
            if self._array_state[k].c_has_active_memory:
                lib.free(getattr(self.cstruct, k))


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
