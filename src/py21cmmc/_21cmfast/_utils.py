"""
Utilities that help with wrapping various C structures.
"""
import glob
import warnings
from hashlib import md5
from os import path

import h5py
import numpy as np
import yaml

# Global Options
with open(path.expanduser(path.join("~", '.21CMMC', "config.yml"))) as f:
    config = yaml.load(f)

# The following is just an *empty* ffi object, which can perform certain operations which are not specific
# to a certain library.
from cffi import FFI

_ffi = FFI()

import logging

logger = logging.getLogger("21CMMC")


class StructWrapper:
    """
    A base-class python wrapper for C structures (not instances of them) providing simple methods for creating new
    instances and accessing field names and values.

    To implement wrappers of specific structures, make a subclass with the same name as the appropriate C struct (which
    must be defined in the C code that has been compiled to the ffi object), *or* use an arbitrary name, but set the
    _name attribute to the C struct name.
    """
    _name = None
    _ffi = None

    def __init__(self):

        # Set the name of this struct in the C code
        if self._name is None:
            self._name = self.__class__.__name__

    @property
    def _cstruct(self):
        """
        This is the actual structure which needs to be passed around to C functions.
        It is best accessed by calling the instance (see __call__)

        Note that the reason it is defined as this cached property is so that it can be created dynamically, but not
        lost. It must not be lost, or else C functions which use it will lose access to its memory. But it also must
        be created dynamically so that it can be recreated after pickling (pickle can't handle CData).
        """

        try:
            return self.__cstruct
        except AttributeError:
            self.__cstruct = self._new()
            return self.__cstruct

    def _new(self):
        """
        Return a new empty C structure corresponding to this class.
        """
        return self._ffi.new("struct " + self._name + "*")

    @property
    def fields(self):
        """
        List of fields of the underlying C struct (a list of tuples of "name, type")
        """
        return self._ffi.typeof(self._cstruct[0]).fields

    @property
    def fieldnames(self):
        """
        List names of fields of the underlying C struct.
        """
        return [f for f, t in self.fields]

    @property
    def pointer_fields(self):
        """List of names of fields which have pointer type in the C struct"""
        return [f for f, t in self.fields if t.type.kind == "pointer"]

    @property
    def primitive_fields(self):
        """List of names of fields which have primitive type in the C struct"""
        return [f for f, t in self.fields if t.type.kind == "primitive"]

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in ["_strings", "_StructWrapper__cstruct"]}

    def refresh_cstruct(self):
        """Delete the underlying C object, forcing it to be rebuilt."""
        try:
            del self.__cstruct
        except AttributeError:
            pass


class StructWithDefaults(StructWrapper):
    """
    A class which provides a convenient interface to create a C structure with defaults specified.

    It is provided for the purpose of *creating* C structures in Python to be passed to C functions, where sensible
    defaults are available. Structures which are created within C and passed back do not need to be wrapped.

    This provides a *fully initialised* structure, and will fail if not all fields are specified with defaults.

    .. note:: The actual C structure is gotten by calling an instance. This is auto-generated when called, based on the
              parameters in the class.

    .. warning:: This class will *not* deal well with parameters of the struct which are pointers. All parameters
                 should be primitive types, except for strings, which are dealt with specially.

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
                    "%s takes up to one position argument, %s were given" % (self.__class__.__name__, len(args)))
            elif args[0] is None:
                pass
            elif isinstance(args[0], self.__class__):
                kwargs.update(args[0].self)
            elif isinstance(args[0], dict):
                kwargs.update(args[0])
            else:
                raise TypeError(
                    "optional positional argument for %s must be None, dict, or an instance of itself" % self.__class__.__name__)

        for k, v in self._defaults_.items():

            # Prefer arguments given to the constructor.
            if k in kwargs:
                v = kwargs.pop(k)

            try:
                setattr(self, k, v)
            except AttributeError:
                # The attribute has been defined as a property, save it as a hidden variable
                setattr(self, "_" + k, v)

        if kwargs:
            logger.warning("The following parameters to {thisclass} are not supported: {lst}".format(
                thisclass=self.__class__.__name__, lst=list(kwargs.keys())
            ))

    def convert(self, key, val):
        return val

    def update(self, **kwargs):
        """
        Update the parameters of an existing class structure.

        This should always be used instead of attempting to *assign* values to instance attributes.
        It consistently re-generates the underlying C memory space and sets some book-keeping variables.

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
            warnings.warn("The following arguments to be updated are not compatible with this class: %s" % kwargs)

    def __call__(self):
        """
        Return a filled C Structure corresponding to this instance.
        """

        for key, val in self.pystruct.items():

            # Find the value of this key in the current class
            if isinstance(val, str):
                # If it is a string, need to convert it to C string ourselves.
                val = self.ffi.new('char[]', getattr(self, key).encode())

            try:
                setattr(self._cstruct, key, val)
            except TypeError:
                print("For key %s, value %s:" % (key, val))
                raise

        return self._cstruct

    @property
    def pystruct(self):
        """A pure-python dictionary representation of the corresponding C structure"""
        return {fld: self.convert(fld, getattr(self, fld)) for fld in self.fieldnames}

    @property
    def defining_dict(self):
        """
        Pure python dictionary representation of this class, as it would appear in C

        Note: This is not the same as :attr:`pystruct`, as it omits all variables that don't need to be passed to the
              constructor, but appear in the C struct (some can be calculated dynamically based on the inputs). It is
              also not the same as :attr:`self`, as it includes the 'converted' values for each variable, which are
              those actually passed to the C code.
        """
        return {k: self.convert(k, getattr(self, k)) for k in self._defaults_}

    @property
    def self(self):
        """
        Dictionary which if passed to its own constructor will yield an identical copy

        Note: this differs from :attr:`pystruct` and :attr:`defining_dict` in that it uses the hidden variable
              value, if it exists, instead of the exposed one. This prevents from, for example, passing a value which
              is 10**10**val (and recurring!).
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
        return self.__class__.__name__ + "(" + ", ".join(
            sorted([k + ":" + str(v) for k, v in self.defining_dict.items()])) + ")"

    def __eq__(self, other):
        return self.__repr__() == repr(other)

    def __hash__(self):
        return hash(self.__repr__())


class OutputStruct(StructWrapper):
    _fields_ = []
    _global_params = None
    _inputs = ["user_params", "cosmo_params", "_random_seed"]
    _filter_params = ['external_table_path']

    _TYPEMAP = {
        'float32': 'float *',
        'float64': 'double *',
        'int32': 'int *'
    }

    def __init__(self, *, random_seed=None, init=False, **kwargs):
        super().__init__()

        self.filled = False
        self._random_seed = random_seed

        for k in self._inputs:

            if not hasattr(self, k):
                try:
                    setattr(self, k, kwargs.pop(k))
                except KeyError:
                    raise KeyError("%s requires the keyword argument %s" % (self.__class__.__name__, k))

        if kwargs:
            warnings.warn(
                "%s received the following unexpected arguments: %s" % (self.__class__.__name__, list(kwargs.keys())))

        if init:
            self._init_cstruct()

    def _init_arrays(self):  # pragma: nocover
        """Abstract base method for initializing any arrays that the structure has."""
        pass

    @property
    def random_seed(self):
        if self._random_seed is None:
            self._random_seed = int(np.random.randint(1, int(1e12)))

        return self._random_seed

    @property
    def arrays_initialized(self):
        """Whether all necessary arrays are initialized (this must be true before passing to a C function)."""
        # This assumes that all pointer fields will be arrays...
        for k in self.pointer_fields:
            if not hasattr(self, k):
                return False
            elif getattr(self._cstruct, k) == self._ffi.NULL:
                return False
        return True

    def _init_cstruct(self):

        if not self.filled:
            self._init_arrays()

        for k in self.pointer_fields:
            setattr(self._cstruct, k, self._ary2buf(getattr(self, k)))
        for k in self.primitive_fields:
            try:
                setattr(self._cstruct, k, getattr(self, k))
            except AttributeError:
                pass

        if not self.arrays_initialized:
            raise AttributeError(
                "%s is ill-defined. It has not initialized all necessary arrays." % self.__class__.__name__)

    def _ary2buf(self, ary):
        if not isinstance(ary, np.ndarray):
            raise ValueError("ary must be a numpy array")
        return self._ffi.cast(OutputStruct._TYPEMAP[ary.dtype.name], self._ffi.from_buffer(ary))

    def __call__(self):
        if not self.arrays_initialized:
            self._init_cstruct()

        return self._cstruct

    def _expose(self):
        """This method exposes the non-array primitives of the ctype to the top-level object."""
        if not self.filled:
            raise Exception("You need to have actually called the C code before the primitives can be exposed.")
        for k in self.primitive_fields:
            setattr(self, k, getattr(self._cstruct, k))

    @property
    def _fname_skeleton(self):
        """The filename without specifying the random seed"""
        return self._name + "_" + self._md5 + "_r{seed}.h5"

    @property
    def filename(self):
        """The base filename of this object"""
        if self._random_seed is None:
            raise AttributeError("filename not defined until random_seed has been set")

        return self._fname_skeleton.format(seed=self.random_seed)

    def _get_fname(self, direc=None):
        direc = direc or path.expanduser(config['boxdir'])
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
            The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC``.

        Returns
        -------
        str
            The filename of an existing set of boxes, or None.
        """
        # First, if appropriate, find a file without specifying seed.
        # Need to do this first, otherwise the seed will be chosen randomly upon choosing a filename!
        direc = direc or path.expanduser(config['boxdir'])

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
        with h5py.File(fname, 'r') as f:
            for k in self._inputs + ["_global_params"]:
                q = getattr(self, k)

                # The key name as it should appear in file.
                kfile = k.lstrip("_")

                # If this particular variable is set to None, this is interpreted
                # as meaning that we don't care about matching it to file.
                if q is None:
                    continue

                if isinstance(q, StructWithDefaults) or isinstance(q, StructInstanceWrapper):
                    grp = f[kfile]

                    if isinstance(q, StructWithDefaults):
                        dct = q.self
                    else:
                        dct = q

                    for kk, v in dct.items():
                        if kk not in self._filter_params:
                            file_v = grp.attrs[kk]
                            if file_v == u'none': file_v = None
                            if file_v != v:
                                logger.debug("For file %s:" % fname)
                                logger.debug(
                                    "\tThough md5 and seed matched, the parameter %s did not match, with values %s and %s in file and user respectively" % (
                                        kk, file_v, v))
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
            The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC``.
        """
        return self.find_existing(direc) is not None

    def write(self, direc=None):
        """
        Write the initial conditions boxes in standard HDF5 format.

        Parameters
        ----------
        direc : str, optional
            The directory in which to write the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC``.
        """
        if not self.filled:
            raise IOError("The boxes have not yet been computed.")

        if not self._random_seed:
            raise ValueError(
                "Attempting to write when no random seed has been set. Struct has been 'filled' inconsistently.")

        try:
            with h5py.File(self._get_fname(direc), 'w') as f:
                # Save input parameters to the file
                for k in self._inputs + ["_global_params"]:
                    q = getattr(self, k)

                    kfile = k.lstrip("_")

                    if isinstance(q, StructWithDefaults) or isinstance(q, StructInstanceWrapper):
                        grp = f.create_group(kfile)
                        if isinstance(q, StructWithDefaults):
                            dct = q.self  # using self allows to rebuild the object from HDF5 file.
                        else:
                            dct = q

                        for kk, v in dct.items():
                            if kk not in self._filter_params:
                                grp.attrs[kk] = u"none" if v is None else v
                    else:
                        f.attrs[kfile] = q

                # Save the boxes to the file
                boxes = f.create_group(self._name)

                # Go through all fields in this struct, and save
                for k in self.pointer_fields:
                    boxes.create_dataset(k, data=getattr(self, k))

                for k in self.primitive_fields:
                    boxes.attrs[k] = getattr(self, k)
        except OSError as e:
            logger.warning("When attempting to write {} to file, write failed with the following error. Continuing without caching.")
            logger.warning(e)

    def read(self, direc=None):
        """
        Try to find and read in existing boxes from cache, which match the parameters of this instance.

        Parameters
        ----------
        direc : str, optional
            The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC``.
        """
        if self.filled:
            raise IOError("This data is already filled, no need to read in.")

        pth = self.find_existing(direc)

        if pth is None:
            raise IOError("No boxes exist for these parameters.")

        # Need to make sure arrays are initialized before reading in data to them.
        if not self.arrays_initialized:
            self._init_cstruct()

        with h5py.File(pth, 'r') as f:
            try:
                boxes = f[self._name]
            except KeyError:
                raise IOError(
                    "While trying to read in %s, the file exists, but does not have the correct structure." % self._name)

            # Fill our arrays.
            for k in boxes.keys():
                getattr(self, k)[...] = boxes[k][...]

            for k in boxes.attrs.keys():
                setattr(self, k, boxes.attrs[k])

            # Need to make sure that the seed is set to the one that's read in.
            seed = f.attrs['random_seed']
            self._random_seed = seed

        self.filled = True
        self._expose()

    def __repr__(self):
        # This is the class name and all parameters which belong to C-based input structs,
        # eg. InitialConditions(HII_DIM:100,SIGMA_8:0.8,...)
        return self._seedless_repr() + "_random_seed={}".format(self._random_seed)

    def _seedless_repr(self):
        # The same as __repr__ except without the seed.
        return self._name + "(" + "; ".join(
            [repr(v) if isinstance(v, StructWithDefaults) else (
                v.filtered_repr(self._filter_params) if isinstance(v, StructInstanceWrapper)
                else k.lstrip("_") + ":" + repr(v)) for k, v in
             [(k, getattr(self, k)) for k in self._inputs + ['_global_params'] if k != "_random_seed"]]) + ")"

    def __str__(self):
        # this is *not* a unique representation, and doesn't include global params.
        return self._name + "(" + ";\n\t".join(
            [repr(v) if isinstance(v, StructWithDefaults) else k.lstrip("_") + ":" + repr(v) for k, v in
             [(k, getattr(self, k)) for k in self._inputs]]) + ")"

    def __hash__(self):
        """this should be unique for this combination of parameters, even global params and random seed."""
        return hash(repr(self))

    @property
    def _md5(self):
        """A hash of the object, which does *not* take into account the random seed."""
        return md5(self._seedless_repr().encode()).hexdigest()

    def __eq__(self, other):
        return repr(self) == repr(other)


class StructInstanceWrapper:
    """
    A wrapper for *instances* of C structs.
    """

    def __init__(self, wrapped, ffi):
        self._cobj = wrapped
        self._ffi = ffi

        for nm, tp in self._ffi.typeof(self._cobj).fields:
            setattr(self, nm, getattr(self._cobj, nm))

        # Get the name of the structure
        self._ctype = self._ffi.typeof(self._cobj).cname.split()[-1]

    def __setattr__(self, name, value):
        try:
            setattr(self._cobj, name, value)
        except AttributeError:
            pass
        object.__setattr__(self, name, value)

    def items(self):
        for nm, tp in self._ffi.typeof(self._cobj).fields:
            yield nm, getattr(self, nm)

    def keys(self):
        return [nm for nm, tp in self.items()]

    def __repr__(self):
        return self._ctype + "(" + ";".join([k + "=" + str(v) for k, v in sorted(self.items())]) + ")"

    def filtered_repr(self, filter_params):
        return self._ctype + "(" + ";".join([k + "=" + str(v) for k, v in sorted(self.items()) if k not in filter_params]) + ")"
