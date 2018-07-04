"""
Utilities that help with wrapping various C structures.
"""


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

    def __getstate__(self):
        return {k:v for k,v in self.__dict__.items() if k not in ["_strings", "_cstruct"]}