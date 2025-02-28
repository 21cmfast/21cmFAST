"""Data structure wrappers for the C code."""

from __future__ import annotations

import attrs
import contextlib
import logging
from bidict import bidict
from typing import Any

from .. import __version__
from ..c_21cmfast import ffi
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
    _ffi = attrs.field(default=ffi)

    _TYPEMAP = bidict({"float32": "float *", "float64": "double *", "int32": "int *"})

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
        return dict(self._ffi.typeof(self.cstruct[0]).fields)

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
            k: v
            for k, v in self.__dict__.items()
            if k not in ["_strings", "cstruct", "_ffi"]
        }

    def expose_to_c(self, array: Array, name: str):
        """Expose the memory of a particular Array to the backend C code."""
        if not array.state.initialized:
            raise ValueError("Array must be initialized before exposing to C")

        def _ary2buf(ary):
            return self._ffi.cast(
                self._TYPEMAP[ary.dtype.name], self._ffi.from_buffer(ary)
            )

        try:
            setattr(self.cstruct, name, _ary2buf(array.value))
        except TypeError as e:
            raise TypeError(f"Error setting {name}") from e


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
