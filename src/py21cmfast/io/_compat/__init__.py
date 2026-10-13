"""The changes to the file format made in each release of 21cmFAST since v4.0.

Each module of this package describes the changes made in a single release, as a
:class:`~py21cmfast.io._compat_framework.FormatRelease` called ``RELEASE``, along
with any helper functions required for those changes. When a release changes the file
format, add a new module for it here (e.g. ``_v44.py`` for v4.4.0), and add its
``RELEASE`` to :data:`FORMAT_HISTORY` below.
"""

from .._compat_framework import FormatRelease
from . import _v41, _v42, _v43

#: The history of changes to the format of 21cmFAST's output files, since v4.0, in
#: the order in which they were made.
FORMAT_HISTORY: tuple[FormatRelease, ...] = (
    _v41.RELEASE,
    _v42.RELEASE,
    _v43.RELEASE,
)
