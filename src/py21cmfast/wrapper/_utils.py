"""Utilities that help with wrapping various C structures."""

import logging
import numpy as np
from cffi import FFI

from .. import __version__
from ..c_21cmfast import ffi, lib
from .exceptions import _process_exitcode

_ffi = FFI()

logger = logging.getLogger(__name__)


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


def _call_c_simple(fnc, *args):
    """Call a simple C function that just returns an object.

    Any such function should be defined such that the last argument is an int pointer generating
    the status.
    """
    # Parse the function to get the type of the last argument
    cdata = str(ffi.addressof(lib, fnc.__name__))
    kind = cdata.split("(")[-1].split(")")[0].split(",")[-1]
    result = ffi.new(kind)
    status = fnc(*args, result)
    _process_exitcode(status, fnc, args)
    return result[0]
