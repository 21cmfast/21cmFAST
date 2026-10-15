"""Utilities that help with wrapping various C structures."""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
from cffi import FFI

from ..c_21cmfast import ffi, lib
from .exceptions import CFunction, _process_exitcode

_ffi = FFI()

logger = logging.getLogger(__name__)


ctype2dtype: dict[str, np.dtype] = {}

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


def asarray(ptr: Any, shape: Sequence[int]) -> np.ndarray:
    """Return a numpy array that views the memory of a C array, without copying.

    ``ptr`` is a cffi pointer (``cdata``) to the first element of the C array, whose
    element type determines the dtype of the returned array.
    """
    ctype = _ffi.getctype(_ffi.typeof(ptr).item).split("*")[0].strip()

    if ctype not in ctype2dtype:
        raise RuntimeError(
            f"Cannot create an array for element type: {ctype}. Can do {list(ctype2dtype.values())}."
        )

    array = np.frombuffer(
        _ffi.buffer(ptr, _ffi.sizeof(ctype) * np.prod(shape)), ctype2dtype[ctype]
    )
    return array.reshape(shape)


def _call_c_simple(fnc: CFunction, *args: Any) -> Any:
    """Call a simple C function that just returns an object.

    Any such function should be defined such that the last argument is an int pointer generating
    the status.

    The return value is whatever C type that output pointer points to (converted to
    the equivalent python type by cffi), so it cannot be typed more precisely here.
    """
    # Parse the function to get the type of the last argument
    cdata = str(ffi.addressof(lib, fnc.__name__))
    kind = cdata.split("(")[-1].split(")")[0].split(",")[-1]
    result = ffi.new(kind)
    status = fnc(*args, result)
    _process_exitcode(status, fnc, args)
    return result[0]


def camel_to_snake(word: str, depublicize: bool = False) -> str:
    """Convert came case to snake case."""
    word = "".join(f"_{i.lower()}" if i.isupper() else i for i in word)

    if not depublicize:
        word = word.lstrip("_")

    return word


def snake_to_camel(word: str, publicize: bool = True) -> str:
    """Convert snake case to camel case."""
    if "_" not in word:
        return word

    if publicize:
        word = word.lstrip("_")
    return "".join(x.capitalize() or "_" for x in word.split("_"))


def get_all_subclasses[T](cls: type[T]) -> list[type[T]]:
    """Get a list of all subclasses of a given class, recursively."""
    all_subclasses: list[type[T]] = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def float_to_string_precision(x: float, n: int) -> str:
    """Print out a standard float number at a given number of significant digits.

    Code here: https://stackoverflow.com/a/48812729
    """
    return f"{float(f'{x:.{int(n)}g}'):g}"
