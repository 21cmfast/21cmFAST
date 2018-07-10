"""
Utilities that help with wrapping various C structures.
"""
from hashlib import md5
from os import path
import yaml
import re, glob
import numpy as np
import h5py

# Global Options
with open(path.expanduser(path.join("~", '.21CMMC', "config.yml"))) as f:
    config = yaml.load(f)


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
    ffi = None

    def __init__(self, **kwargs):

        for k, v in self._defaults_.items():

            # Prefer arguments given to the constructor.
            if k in kwargs:
                v = kwargs[k]

            try:
                setattr(self, k, v)
            except AttributeError:
                # The attribute has been defined as a property, save it as a hidden variable

                setattr(self, "_" + k, v)

        self._logic()

        # Set the name of this struct in the C code
        if self._name is None:
            self._name = self.__class__.__name__

        # A little list to hold references to strings so they don't de-reference
        self._strings = []

    def _logic(self):
        pass

    def new(self):
        """
        Return a new empty C structure corresponding to this class.
        """
        obj = self.ffi.new("struct " + self._name + "*")
        return obj

    def __call__(self):
        """
        Return a filled C Structure corresponding to this instance.
        """

        obj = self.new()

        self._logic() # call this here to make sure any changes by the user to the arguments are re-processed.

        for fld in self.ffi.typeof(obj[0]).fields:
            key = fld[0]
            val = getattr(self, key)

            # Find the value of this key in the current class
            if isinstance(val, str):
                # If it is a string, need to convert it to C string ourselves.
                val = self.ffi.new('char[]', getattr(self, key).encode())

            try:
                setattr(obj, key, val)
            except TypeError:
                print("For key %s, value %s:" % (key, val))
                raise

        self._cstruct = obj
        return obj

    @property
    def pystruct(self):
        "A Python dictionary containing every field which needs to be initialized in the C struct."
        obj = self.new()
        return {fld[0]:getattr(self, fld[0]) for fld in self.ffi.typeof(obj[0]).fields}

    @property
    def __defining_dict(self):
        # The defining dictionary is everything that defines the structure,
        # but without anything that constitutes a random seed, which should be defined as RANDOM_SEED*
        return {k:getattr(self, k) for k in self._defaults_ if not k.startswith("RANDOM_SEED")}

    def __repr__(self):
        return ", ".join(sorted([k+":"+str(v) for k,v in self.__defining_dict.items()]))

    def __eq__(self, other):
        return isinstance(self, other) and self.__repr__() == repr(other)

    def __hash__(self):
        return hash(self.__repr__)

    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if k not in ["_strings", "_cstruct"]}


class OutputStruct:
    filled = False
    _fields_ = []
    _name = None   # This must match the name of the C struct

    ffi = None

    _TYPEMAP = {
        'float32': 'float *',
        'float64': 'double *',
        'int32': 'int *'
    }

    def __init__(self, user_params, cosmo_params, **kwargs):
        # These two parameter dicts will exist for every output struct.
        # Additional ones can be supplied with kwargs.
        self.user_params = user_params
        self.cosmo_params = cosmo_params

        for k,v in kwargs.items():
            setattr(self, k, v)

        self._fields_ = self._init_boxes()

        # Set the name of this struct in the C code
        if self._name is None:
            self._name = self.__class__.__name__

    def _ary2buf(self, ary):
        if not isinstance(ary, np.ndarray):
            raise ValueError("ary must be a numpy array")

        return self.ffi.cast(OutputStruct._TYPEMAP[ary.dtype.name], self.ffi.from_buffer(ary))

    def __call__(self):
        self._cstruct = self._new()
        for k in self._fields_:
            setattr(self._cstruct, k, self._ary2buf(getattr(self, k)))

        return self._cstruct

    def _new(self):
        """
        Return a new empty C structure corresponding to this class.
        """
        obj = self.ffi.new("struct " + self._name + "*")
        return obj

    def _get_fname(self, direc=None, fname=None):
        if direc:
            fname = fname or self._hashname
        else:
            fname = self._hashname

        direc = direc or path.expanduser(config['boxdir'])

        return path.join(direc, fname)

    @staticmethod
    def _find_file_without_seed(f):
        f = re.sub("r\d+\.", "r*.", f)
        allfiles = glob.glob(f)
        if allfiles:
            return allfiles[0]
        else:
            return None

    def find_existing(self, direc=None, fname=None, match_seed=False):
        """
        Try to find existing boxes which match the parameters of this instance.

        Parameters
        ----------
        direc : str, optional
            The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC`. This central directory will be searched in addition to whatever is
            passed to `direc`.

        fname : str, optional
            The filename to search for. This is used in addition to the filename automatically assigned by the hash
            of this instance.

        match_seed : bool, optional
            Whether to force the random seed to also match in order to be considered a match.

        Returns
        -------
        str
            The filename of an existing set of boxes, or None.
        """
        if direc is not None:
            if fname is not None:
                if path.exists(self._get_fname(direc, fname)):
                    return self._get_fname(direc, fname)

            f = self._get_fname(direc, None)
            if path.exists(f):
                return f
            elif not match_seed:
                f = self._find_file_without_seed(f)
                if f: return f

        f = self._get_fname(None, None)
        if path.exists(f):
            return f
        else:
            f = self._find_file_without_seed(f)
            if f: return f

        return None

    def exists(self, direc=None, fname=None, match_seed=False):
        return self.find_existing(direc, fname, match_seed) is not None

    def write(self, direc=None, fname=None):
        """
        Write the initial conditions boxes in standard HDF5 format.

        Parameters
        ----------
        direc : str, optional
            The directory in which to write the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC`.

        fname : str, optional
            The filename to write to. This is only used if `direc` is not None. By default, the filename is a hash
            which accounts for the various parameters that define the boxes, to ensure uniqueness.
        """
        if not self.filled:
            raise IOError("The boxes have not yet been computed.")

        with h5py.File(self._get_fname(direc,fname), 'w') as f:
            # Save the cosmo and user params to the file
            cosmo = f.create_group("cosmo")
            for k,v in self.cosmo_params.pystruct.items():
                cosmo.attrs[k] = v

            user = f.create_group("user_params")
            for k, v in self.user_params.pystruct.items():
                user.attrs[k] = v

            # Save the boxes to the file
            boxes = f.create_group(self._name)

            # Go through all fields in this struct, and save
            for k in self._fields_:
                boxes.create_dataset(k, data = getattr(self, k))

    def read(self, direc=None, fname=None, match_seed=False):
        """
        Try to find and read in existing boxes which match the parameters of this instance.

        Parameters
        ----------
        direc : str, optional
            The directory in which to search for the boxes. By default, this is the centrally-managed directory, given
            by the ``config.yml`` in ``.21CMMC`. This central directory will be searched in addition to whatever is
            passed to `direc`.

        fname : str, optional
            The filename to search for. This is used in addition to the filename automatically assigned by the hash
            of this instance.

        match_seed : bool, optional
            Whether to force the random seed to also match in order to be considered a match.
        """
        pth = self.find_existing(direc, fname, match_seed)

        if pth is None:
            raise IOError("No boxes exist for these cosmo and user parameters.")

        with h5py.File(pth,'r') as f:
            try:
                boxes = f[self._name]
            except:
                raise IOError("There is no group %s in the file"%self._name)

            # Fill our arrays.
            for k in boxes.keys():
                getattr(self, k)[...] = boxes[k][...]

            # Need to make sure that the seed is set to the one that's read in.
            seed = f['cosmo'].attrs['RANDOM_SEED']
            self.cosmo_params._RANDOM_SEED = seed

        self.filled = True

    def __repr__(self):
        # This is the class name and all parameters which belong to C-based input structs,
        # eg. InitialConditions(HII_DIM:100,SIGMA_8:0.8,...)
        return self._name + "("+ "".join([repr(v) for k,v in self.__dict__.items() if isinstance(v, StructWithDefaults)]) +")"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(repr(self))

    @property
    def _md5(self):
        # this hash takes only the stuff after the class's name, so that different classes can have the same hash.
        return md5(repr(self).split("(")[-1].encode()).hexdigest()

    @property
    def _hashname(self):
        return self._name + "_" + self._md5 + "_r%s" % self.cosmo_params.RANDOM_SEED + ".h5"

    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if not isinstance(k, self.ffi.CData)}