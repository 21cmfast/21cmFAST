"""Utilities that help with wrapping various C structures."""

import logging

import numpy as np

import py21cmfast.c_21cmfast as lib

from .exceptions import _process_exitcode

logger = logging.getLogger(__name__)


ctype2dtype = {}

# Integer types
for prefix in ("int", "uint"):
    for log_bytes in range(4):
        ctype = f"{prefix}{8 * 2**log_bytes:d}_t"
        dtype = f"{prefix[0]}{2**log_bytes:d}"
        ctype2dtype[ctype] = np.dtype(dtype)

# Floating point types
ctype2dtype["float"] = np.dtype("f4")
ctype2dtype["double"] = np.dtype("f8")
ctype2dtype["int"] = np.dtype("i4")


def asarray(ptr, shape):
    """Get the canonical C type of the elements of ptr as a string."""
    ctype = type(ptr).__name__  # TODO: check

    if ctype not in ctype2dtype:
        raise RuntimeError(
            f"Cannot create an array for element type: {ctype}. Can do {list(ctype2dtype.values())}."
        )

    array = np.frombuffer(ptr, ctype2dtype[ctype])  # TODO: check
    array.shape = shape
    return array


def _nb_initialise_return_value(arg_string, out_shape=(1,)):
    """Return a zero-initialised object of the correct type given a nanobind signature.

    Currently only works with wrapped structures or numpy arrays.
    """
    # If it's a wrapped class, return the class
    if "py21cmfast.c_21cmfast" in arg_string:
        return getattr(lib, arg_string.split("py21cmfast.c_21cmfast")[-1])()

    if "*" in arg_string or "ndarray" in arg_string:
        base_type = arg_string.split("dtype=")[1].split("]")[0]
        return np.zeros(out_shape, dtype=getattr(np, base_type))

    raise ValueError(
        f"Cannot create a zero-initialised object of type {arg_string}."
        "As it is not a pointer, array or class. Please check the function signature."
    )


def _call_c_simple(fnc, *args):
    """Call a simple C function that just returns an object.

    Assumes that the last argument is a pointer to an object that will be filled in by the C function.
    This argument is initialised here and returned.
    """
    # Parse the function to get the type of the last argument
    cdata = fnc.__nb_signature__[0][0]
    # Nanobind signature is 'def fnc.__name__(arg0: type0, arg1: type1, ..., argN: typeN, /) -> returntype'
    # We wish to extract the type of the last argument only.
    signature_string = (
        cdata.split("(")[-1].split(")")[0].split(",")[-2].replace("arg: ", "").strip()
    )
    # NOTE: This uses the default return size == 1 for arrays
    result = _nb_initialise_return_value(signature_string)
    status = fnc(*args, result)
    _process_exitcode(status, fnc, args)
    return result


def camel_to_snake(word: str, depublicize: bool = False):
    """Convert came case to snake case."""
    word = "".join(f"_{i.lower()}" if i.isupper() else i for i in word)

    if not depublicize:
        word = word.lstrip("_")

    return word


def snake_to_camel(word: str, publicize: bool = True):
    """Convert snake case to camel case."""
    if publicize:
        word = word.lstrip("_")
    return "".join(x.capitalize() or "_" for x in word.split("_"))


def get_all_subclasses(cls):
    """Get a list of all subclasses of a given class, recursively."""
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def float_to_string_precision(x, n):
    """Print out a standard float number at a given number of significant digits.

    Code here: https://stackoverflow.com/a/48812729
    """
    return f"{float(f'{x:.{int(n)}g}'):g}"
