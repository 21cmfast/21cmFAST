"""
Utilities that help with wrapping various C structures.
"""
from hashlib import md5
from os import path
import yaml
import re, glob
import numpy as np
import h5py
import warnings

# Global Options
with open(path.expanduser(path.join("~", '.21CMMC', "config.yml"))) as f:
    config = yaml.load(f)

# The following is just an *empty* ffi object, which can perform certain operations which are not specific
# to a certain library.
from cffi import FFI
_ffi = FFI()

import logging
logger = logging.getLogger("21CMMC")

class StructWithDefaults:
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
    _name = None
    _defaults_ = {}
    _ffi = None

    def __init__(self, *args, **kwargs):

        if args:
            if len(args)>1:
                raise TypeError("%s takes up to one position argument, %s were given"%(self.__class__.__name__, len(args)))
            elif args[0] is None:
                pass
            elif isinstance(args[0], self.__class__):
                kwargs.update(args[0].self)
            elif isinstance(args[0], dict):
                kwargs.update(args[0])
            else:
                raise TypeError("optional positional argument for %s must be None, dict, or an instance of itself"%self.__class__.__name__)

        for k, v in self._defaults_.items():


            # Prefer arguments given to the constructor.
            if k in kwargs:
                v = kwargs[k]

            try:
                setattr(self, k, v)
            except AttributeError:
                # The attribute has been defined as a property, save it as a hidden variable

                setattr(self, "_" + k, v)

        # Set the name of this struct in the C code
        if self._name is None:
            self._name = self.__class__.__name__

        # A little list to hold references to strings so they don't de-reference
        self._strings = []
        self._cstruct_inited = False

    def convert(self, key, val):
        return val

    @property
    def _cstruct(self):
        """
        This is the actual structure which needs to be passed around to C functions.
        It is best accessed by calling the instance (see __call__)

        Note that the reason it is defined as this cached property is so that it can be created dynamically, but not
        lost. It must not be lost, or else C functions which use it will lose access to its memory. But it also must
        be created dynamically so that it can be recreated after pickling (pickle can't handle CData).
        """
        if hasattr(self, "_StructWithDefaults__cstruct") and self._cstruct_inited:
            return self.__cstruct
        else:
            self.__cstruct = self._new()
            self._cstruct_inited = True
            return self.__cstruct

    def _new(self):
        """
        Return a new empty C structure corresponding to this class.
        """
        return self._ffi.new("struct " + self._name + "*")

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
            self._cstruct_inited = False
            self._strings = []

        for k in self._defaults_:
            # Prefer arguments given to the constructor.
            if k in kwargs:
                v = kwargs.pop(k)

                try:
                    setattr(self, k, v)
                except AttributeError:
                    # The attribute has been defined as a property, save it as a hidden variable
                    setattr(self, "_" + k, v)

        if kwargs:
            warnings.warn("The following arguments to be updated are not compatible with this class: %s"%kwargs)

    def __call__(self):
        """
        Return a filled C Structure corresponding to this instance.
        """

        for fld in self._ffi.typeof(self._cstruct[0]).fields:
            key = fld[0]
            val = self.convert(key, getattr(self, key))

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
        "A Python dictionary containing every field which needs to be initialized in the C struct."
        return {fld[0]:getattr(self, fld[0]) for fld in self._ffi.typeof(self._cstruct[0]).fields}

    @property
    def defining_dict(self):
        """
        Everything that defines the structure, but without anything that constitutes a random seed, which should be
        defined as RANDOM_SEED*
        """
        return {k:getattr(self, k) for k in self._defaults_ if not k.startswith("RANDOM_SEED")}

    @property
    def self(self):
        "Dictionary which if passed to its own constructor will yield an identical copy"
        # Try to first use the hidden variable (eg. _RANDOM_SEED) before using the non-hidden variety.
        dct = {}
        for k in self._defaults_:
            if hasattr(self, "_"+k):
                dct[k] = getattr(self, "_"+k)
            else:
                dct[k] = getattr(self, k)

        return dct

    def __repr__(self):
        return self.__class__.__name__ +"(" + ", ".join(sorted([k +":" + str(v) for k,v in self.defining_dict.items()])) + ")"

    def __eq__(self, other):
        return self.__repr__() == repr(other)

    def __hash__(self):
        return hash(self.__repr__)

    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if k not in ["_strings", "_StructWithDefaults__cstruct"]}


class OutputStruct:


    _fields_ = []
    _name = None   # This must match the name of the C struct
    _id = None
    _ffi = None
    _global_params = None
    _inputs = ["user_params", "cosmo_params"]
    _filter_params = ['external_table_path']

    _TYPEMAP = {
        'float32': 'float *',
        'float64': 'double *',
        'int32': 'int *'
    }

    def __init__(self, init=False, **kwargs):
        for k in self._inputs:
            try:
                setattr(self, k, kwargs.pop(k))
            except KeyError:
                raise("%s requires the keyword argument %s"%(self.__class__.__name__, k))

        if kwargs:
            warnings.warn("%s received the following unexpected arguments: %s"%(self.__class__.__name__, list(kwargs.keys())))


        if init:
            self._init_cstruct()

        # Set the name of this struct in the C code
        if self._name is None:
            self._name = self.__class__.__name__

        # Set the name of this struct in the C code
        if self._id is None:
            self._id = self.__class__.__name__

        self.filled = False

    @property
    def _cstruct(self):
        """
        This is the actual structure which needs to be passed around to C functions.
        It is best accessed by calling the instance (see __call__)

        Note that the reason it is defined as this cached property is so that it can be created dynamically, but not
        lost. It must not be lost, or else C functions which use it will lose access to its memory. But it also must
        be created dynamically so that it can be recreated after pickling (pickle can't handle CData).
        """
        if hasattr(self, "_OutputStruct__cstruct"):
            return self.__cstruct
        else:
            self.__cstruct = self._new()
            return self.__cstruct

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
    def _pointer_fields(self):
        "List of names of fields which have pointer type in the C struct"
        return [f for f, t in self.fields if t.type.kind == "pointer"]

    @property
    def _primitive_fields(self):
        "List of names of fields which have primitive type in the C struct"
        return [f for f, t in self.fields if t.type.kind == "primitive"]

    @property
    def arrays_initialized(self):
        "Whether all necessary arrays are initialized (this must be true before passing to a C function)."
        # This assumes that all pointer fields will be arrays...
        for k in self._pointer_fields:
            if not hasattr(self, k):
                return False
            elif getattr(self._cstruct, k) == self._ffi.NULL:
                return False
        return True

    def _init_cstruct(self):

        if not self.filled:
            self._init_arrays()

        for k in self._pointer_fields:
            setattr(self._cstruct, k, self._ary2buf(getattr(self, k)))
        for k in self._primitive_fields:
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
        "This method exposes the non-array primitives of the ctype to the top-level object."
        if not self.filled:
            raise Exception("You need to have actually called the C code before the primitives can be exposed.")
        for k in self._primitive_fields:
            setattr(self, k, getattr(self._cstruct, k))

    def _new(self):
        """
        Return a new empty C structure corresponding to this class.
        """
        obj = self._ffi.new("struct " + self._id + "*")
        return obj

    def _get_fname(self, direc=None):
        direc = direc or path.expanduser(config['boxdir'])
        return path.join(direc, self.filename)

    def _find_file_without_seed(self, direc):
        allfiles = glob.glob(path.join(direc, self._fname_skeleton.format(seed="*")))
        if allfiles:
            return allfiles[0]
        else:
            return None

    @property
    def _current_seed(self):
        """
        Convenience property indicate the current random seed. Useful to check if it is (still) None.

        Will return None if cosmo_params is not a part of the struct.
        """
        return getattr(getattr(self, "cosmo_params", None), "_RANDOM_SEED", None)

    def find_existing(self, direc=None):
        """
        Try to find existing boxes which match the parameters of this instance.

        Parameters
        ----------
        direc : str, optional
            The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC``. This central directory will be searched in addition to whatever is
            passed to `direc`.

        match_seed : bool, optional
            Whether to force the random seed to also match in order to be considered a match.

        Returns
        -------
        str
            The filename of an existing set of boxes, or None.
        """
        # First, if appropriate, find a file without specifying seed.
        # Need to do this first, otherwise the seed will be chosen randomly upon choosing a filename!
        direc = direc or path.expanduser(config['boxdir'])

        if not self._current_seed:
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
            for k in self._inputs +["_global_params"]:
                q = getattr(self, k)

                if isinstance(q, StructWithDefaults) or isinstance(q, _StructWrapper):
                    grp = f[k]
                    if isinstance(q, StructWithDefaults):
                        # The following *assigns* a random seed if one does not exist.
                        # If the file has been read in, this will be fixed.
                        dct = q.pystruct
                    else:
                        dct = q

                    for kk, v in dct.items():
                        if kk not in self._filter_params + ['RANDOM_SEED']:
                            # Never check random seed. If we need to check it, it's definitely matching (because
                            # it's been matched in the filename). If not, then we don't check it anyway.

                            if grp.attrs[kk] != v:
                                logger.debug("For file %s:"%fname)
                                logger.debug("\tThough md5 and seed matched, the parameter %s did not match, with values %s and %s in file and user respectively"%(kk, grp.attrs[kk], v))
                                return False
                else:
                    if f.attrs[k] != q:
                        return False
        return True

    def exists(self, direc=None):
        """
        Return a bool indicating whether a box matching the parameters of this instance is in cache.

        Parameters
        ----------
        direc : str, optional
            The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC``. This central directory will be searched in addition to whatever is
            passed to `direc`.

        match_seed : bool, optional
            Whether to force the random seed to also match in order to be considered a match.
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

        with h5py.File(self._get_fname(direc), 'w') as f:
            # Save input parameters to the file
            for k in self._inputs +["_global_params"]:
                q = getattr(self, k)

                if isinstance(q, StructWithDefaults) or isinstance(q, _StructWrapper):
                    grp = f.create_group(k)
                    if isinstance(q, StructWithDefaults):
                        dct = q.pystruct
                    else:
                        dct = q

                    for kk, v in dct.items():
                        if kk not in self._filter_params:
                            grp.attrs[kk] = v
                else:
                    f.attrs[k] = q

            # Save the boxes to the file
            boxes = f.create_group(self._name)

            # Go through all fields in this struct, and save
            for k in self._pointer_fields:
                boxes.create_dataset(k, data = getattr(self, k))

            for k in self._primitive_fields:
                boxes.attrs[k] = getattr(self, k)

    def read(self, direc=None):
        """
        Try to find and read in existing boxes from cache, which match the parameters of this instance.

        Parameters
        ----------
        direc : str, optional
            The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC``. This central directory will be searched in addition to whatever is
            passed to `direc`.
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
            except:
                raise IOError("There is no group %s in the file"%self._name)

            # Fill our arrays.
            for k in boxes.keys():
                getattr(self, k)[...] = boxes[k][...]

            for k in boxes.attrs.keys():
                setattr(self, k, boxes.attrs[k])

            # Need to make sure that the seed is set to the one that's read in.
            seed = f['cosmo_params'].attrs['RANDOM_SEED']
            self.cosmo_params.update(RANDOM_SEED=seed)

        self.filled = True

    def __repr__(self):
        # This is the class name and all parameters which belong to C-based input structs,
        # eg. InitialConditions(HII_DIM:100,SIGMA_8:0.8,...)
        return self._name + "("+ "; ".join([repr(v) if isinstance(v,StructWithDefaults) else k+":"+str(v) for k,v in [(k,getattr(self, k)) for k in self._inputs]]) +")"

    def __str__(self):
        return self.__repr__().replace(";",";\n\t")

    def __hash__(self):
        # the global params are here tacked on the end to ensure the hash is reconciled to them.
        return hash(repr(self) + [k+":"+str(v) for k,v in sorted(self._global_params.items())])

    @property
    def _md5(self):
        return md5(repr(self).encode()).hexdigest()

    @property
    def filename(self):
        return self._fname_skeleton.format(seed=self.cosmo_params.RANDOM_SEED)

    @property
    def _fname_skeleton(self):
        "The filename without specifying the random seed"
        return self._name + "_" + self._md5 + "_r{seed}.h5"

    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if not isinstance(v, self._ffi.CData)}


class _StructWrapper:

    def __init__(self, wrapped, ffi):
        self._cobj = wrapped
        self._ffi = ffi

        for nm,tp in self._ffi.typeof(self._cobj).fields:
            setattr(self, nm, getattr(self._cobj, nm))

    def __setattr__(self, name, value):
        try:
            setattr(self._cobj, name, value)
        except AttributeError:
            pass
        object.__setattr__(self, name, value)

    def items(self):
        for nm, tp in self._ffi.typeof(self._cobj).fields:
            yield nm, getattr(self, nm)