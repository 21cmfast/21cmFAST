"""Type stub for the cffi-compiled C extension module (built by ``build_cffi.py``)."""

from typing import Any

import cffi

ffi: cffi.FFI

# The cffi ``lib`` object exposes every C function and constant declared in the
# ``cdef`` dynamically, so it cannot be described more precisely without generating
# a stub from the C headers.
lib: Any
