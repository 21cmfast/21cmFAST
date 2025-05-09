"""Data structure wrappers for the C code."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import attrs
from bidict import bidict

import py21cmfast.c_21cmfast as lib

from .. import __version__
from .._cfg import config
from ._utils import (
    asarray,
    float_to_string_precision,
    get_all_subclasses,
    snake_to_camel,
)
from .arrays import Array
from .arraystate import ArrayState
from .exceptions import _process_exitcode

logger = logging.getLogger(__name__)


@attrs.define(slots=False)
class StructWrapper:
    """
    A base-class python wrapper for C structures (not instances of them).

    Provides simple methods for creating new instances and accessing field names and values.

    To implement wrappers of specific structures, make a subclass with the same name as the
    appropriate C struct (which must be defined in the C code that has been compiled to the ``ffi``
    object), *or* use an arbitrary name, but set the ``_name`` attribute to the C struct name.
    """

    _name: str = attrs.field(converter=str)
    cstruct = attrs.field(default=None)

    _TYPEMAP = bidict({"float32": "float *", "float64": "double *", "int32": "int *"})

    @_name.default
    def _name_default(self):
        return self.__class__.__name__

    def __init__(self, *args):
        """Perform custom initializion actions.

        This instantiates the memory associated with the C struct, attached to this inst.
        """
        self.__attrs_init__(*args)
        if args[0] == "InitialConditions":
            self._cobj = lib.InitialConditions
        else:
            raise NotImplementedError(
                f"Wrapped class {args[0]} not listed as an option in StructWrapper."
            )
        self.cstruct = self._new()

    def _new(self):
        """Return a new empty C structure corresponding to this class."""
        return self._cobj()

    @property
    def fields(self) -> dict[str, Any]:
        """A list of fields of the underlying C struct (a list of tuples of "name, type")."""
        result = {}
        for attr in dir(self._cobj):
            if not attr.startswith("__") and not callable(getattr(self._cobj, attr)):
                result[attr] = type(getattr(self._cobj, attr))
        return result

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

    def expose_to_c(self, array: Array, name: str):
        """Expose the memory of a particular Array to the backend C code."""
        if not array.state.initialized:
            raise ValueError("Array must be initialized before exposing to C")

        # TODO: check if we need to cast or anything like that
        try:
            setattr(self.cstruct, name, array.value)
        except TypeError as e:
            raise TypeError(f"Error setting {name}") from e


class StructInstanceWrapper:
    """A wrapper for *instances* of C structs.

    This is as opposed to :class:`StructWrapper`, which is for the un-instantiated structs.

    Parameters
    ----------
    wrapped :
        The reference to the C object to wrap.
    """

    def __init__(self, wrapped):
        self._cobj = wrapped

        # nanobind does not supply a list of fileds like CFFI does, so we do
        #   this instead to return a list of members
        for attr in dir(self._cobj):
            if not attr.startswith("__") and not callable(getattr(self._cobj, attr)):
                setattr(self, attr, getattr(self._cobj, attr))

        # Get the name of the structure
        # WIP: CFFI Refactor
        self._ctype = type(self._cobj).__name__

    def __setattr__(self, name, value):
        """Set an attribute of the instance, attempting to change it in the C struct as well."""
        with contextlib.suppress(AttributeError):
            setattr(self._cobj, name, value)
        object.__setattr__(self, name, value)

    def items(self):
        """Yield (name, value) pairs for each element of the struct."""
        # nanobind does not supply a list of fileds like CFFI does, so we do
        #   this instead to return a list of members
        for attr in dir(self._cobj):
            if not attr.startswith("__") and not callable(getattr(self._cobj, attr)):
                yield attr, getattr(self, attr)

    def keys(self):
        """Return a list of names of elements in the struct."""
        return [nm for nm, tp in self.items()]

    def __iter__(self):
        """Iterate over the object like a dict."""
        yield from self.keys()

    def __repr__(self):
        """Return a unique representation of the instance."""
        return (
            self._ctype + "(" + ";".join(f"{k}={v!s}" for k, v in sorted(self.items()))
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
                f"{k}={v!s}" for k, v in sorted(self.items()) if k not in filter_params
            )
        ) + ")"
