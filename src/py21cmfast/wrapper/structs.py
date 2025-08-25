"""Data structure wrappers for the C code."""

from __future__ import annotations

import logging
from typing import Any

import attrs

import py21cmfast.c_21cmfast as lib

from .arrays import Array

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

    primitive_types = (bool, str, int, float)

    @_name.default
    def _name_default(self):
        return self.__class__.__name__

    def __init__(self, *args):
        """Perform custom initializion actions.

        This instantiates the memory associated with the C struct, attached to this inst.
        """
        self.__attrs_init__(*args)
        self._cobj = getattr(lib, self._name)  # The wrapped class
        self.cstruct = self._new()  # The instance of the wrapped class

    def _new(self):
        """Return a new empty C structure corresponding to this class."""
        return self._cobj()

    @property
    def fields(self) -> dict[str, Any]:
        """A list of fields of the underlying C struct (a list of tuples of "name, type")."""
        result = {}
        for attr in dir(self.cstruct):
            if not attr.startswith("__"):
                result[attr] = type(getattr(self.cstruct, attr))
        return result

    @property
    def fieldnames(self) -> list[str]:
        """A list of names of fields of the underlying C struct."""
        return [f for f, t in self.fields.items()]

    @property
    def pointer_fields(self) -> list[str]:
        """A list of names of fields which have pointer type in the C struct."""
        return [f.split("set_")[1] for f in self.fields if f.startswith("set_")]

    @property
    def primitive_fields(self) -> list[str]:
        """The list of names of fields which have primitive type in the C struct."""
        return [f for f, t in self.fields.items() if t in self.primitive_types]

    def __getstate__(self):
        """Return the current state of the class without pointers."""
        return {
            k: v for k, v in self.__dict__.items() if k not in ["_strings", "cstruct"]
        }

    def expose_to_c(self, array: Array, name: str):
        """Expose the memory of a particular Array to the backend C code."""
        if not array.state.initialized:
            raise ValueError("Array must be initialized before exposing to C")

        try:
            setter = getattr(self.cstruct, "set_" + name)
            setter(array.value)
        except AttributeError as e:
            raise TypeError(
                f"Error setting {name} on {self.__class__.__name__}, no setter found"
            ) from e


class StructInstanceWrapper:
    """A wrapper for *instances* of C structs.

    This is as opposed to :class:`StructWrapper`, which is for the un-instantiated structs.

    Parameters
    ----------
    wrapped :
        The reference to the C object to wrap.
    """

    # NOTE: currently assumes that the C object is not internally changed
    #   We get all the values from C on initialization, and pass changes back to C
    #   The StructInstanceWrapper holds the attributes as they appear in python,
    #   whereas ._cobj holds primitives and getters/setters for pointers.
    # TODO: we should ditch the object attributes and just use the C object
    #   with a custom __getattr__
    def __init__(self, wrapped):
        self._cobj = wrapped
        # nanobind does not supply a list of fields like CFFI does, so we do
        #   this instead to return a list of members
        for attr in dir(self._cobj):
            # ignore dunders
            if not attr.startswith("__"):
                if attr.startswith("get_"):
                    # If the attribute is a getter, we need to set the value in python
                    #   to the value of the C++ attribute without the "get_" prefix
                    setattr(self, attr[4:], getattr(self._cobj, attr)())
                elif not callable(getattr(self._cobj, attr)):
                    # Otherwise, we just set the attribute to the value
                    setattr(self, attr, getattr(self._cobj, attr))

        # Get the name of the structure
        self._ctype = type(self._cobj).__name__

    def __setattr__(self, name, value):
        """Set an attribute of the instance, attempting to change it in the C struct as well."""
        # use the non-overridden __setattr__ to set the attribute in Python
        object.__setattr__(self, name, value)

        # Set the attribute in the C struct
        if not name.startswith("_"):
            if "set_" + name in dir(self._cobj):
                getattr(self._cobj, "set_" + name)(value)
            elif name in dir(self._cobj):
                setattr(self._cobj, name, value)
            else:
                raise AttributeError(
                    f"Attribute {name} not found in {self.__class__.__name__}"
                )

    def items(self):
        """Yield (name, value) pairs for each element of the struct."""
        # nanobind does not supply a list of fileds like CFFI does, so we do
        #   this instead to return a list of members
        for attr in dir(self._cobj):
            if not attr.startswith("__"):
                if attr.startswith("get_"):
                    yield attr[4:], getattr(self._cobj, attr)()
                elif not attr.startswith("set_"):
                    yield attr, getattr(self._cobj, attr)

    def keys(self):
        """Return a list of names of elements in the struct."""
        return [nm for nm, _ in self.items()]

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
