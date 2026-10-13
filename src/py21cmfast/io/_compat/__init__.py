"""Compatibility with files written by older versions of 21cmFAST.

Every change to the on-disk format of 21cmFAST outputs since v4.0 is recorded in
:data:`FORMAT_HISTORY`. Each step of the history is a
:class:`~py21cmfast.io._compat_framework.FormatRelease`, which converts files from the
format of one release to that of the next release that changed the format, with a
list of :class:`~py21cmfast.io._compat_framework.FormatChange` objects describing what
changed: renamed, added or removed input parameters, renamed output structs, renamed
or removed fields of those output structs, and so on. Each step is defined in its own
module of this package, named after the versions it converts between (e.g.
``_v4_2_v4_3.py``), using the building blocks in
:mod:`py21cmfast.io._compat_framework`. The history serves two purposes:

1. When a file written by an older version of 21cmFAST is read, the changes made
   since that version are applied, in order, to the raw contents of the file, so that
   they can be turned into objects of the current version of the code. This is done
   automatically by the readers in :mod:`py21cmfast.io.h5` (and therefore by
   ``Coeval.from_file``, ``LightCone.from_file`` etc.).
2. It is a running changelog of the I/O format, which can be printed with
   :func:`describe_format_changes` (or ``21cmfast migrate --explain``).

Conversions are not always lossless. Wherever an older file lacks information that
the current version of the code expects (e.g. a field of an output struct that did
not exist, or whose meaning changed and can't be converted), a
:class:`~py21cmfast.io._compat_framework.CompatibilityWarning` is emitted.

Files written by the previous major version of 21cmFAST are deprecated, and files
written by any older version are not supported (see
:func:`~py21cmfast.io._compat_framework.support_status`). When a release changes the
file format, add a module for it here, add its ``RELEASE`` to :data:`FORMAT_HISTORY`
below, and add tests for its changes to ``tests/io/test_compat.py``. When a major
version drops support for files of an old major version, remove the modules that
convert from that major version.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from packaging.version import Version

from .. import _compat_framework
from .._compat_framework import (
    ConvertOutputStruct,
    FormatRelease,
    RawInputs,
    RawOutputStruct,
    RemoveField,
    RenameField,
    RenameOutputStruct,
    UnconvertibleOutputStruct,
    parse_version,
)
from . import _v4_0_v4_1, _v4_1_v4_2, _v4_2_v4_3

#: The history of changes to the format of 21cmFAST's output files, since v4.0, in
#: the order in which they were made.
FORMAT_HISTORY: tuple[FormatRelease, ...] = (
    _v4_0_v4_1.RELEASE,
    _v4_1_v4_2.RELEASE,
    _v4_2_v4_3.RELEASE,
)


def releases_since(version: str | Version) -> list[FormatRelease]:
    """Return the format releases whose changes apply to files of a given version."""
    version = parse_version(version)
    return [rel for rel in FORMAT_HISTORY if rel.applies_to(version)]


def needs_upgrade(version: str | Version) -> bool:
    """Whether a file written by a given version needs to be upgraded."""
    return bool(releases_since(version))


def is_legacy(version: str | Version) -> bool:
    """Whether files of a given version must be converted to the current format.

    This is the case for files written by older versions than the current one, that
    are affected by any changes in :data:`FORMAT_HISTORY`.
    """
    version = parse_version(version)
    return version < _compat_framework.CURRENT_VERSION and needs_upgrade(version)


def upgrade_inputs(inputs: RawInputs) -> RawInputs:
    """Upgrade raw input parameters from an older version to the current format.

    The input is modified in-place, and also returned.
    """
    for release in releases_since(inputs.version):
        for change in release.changes:
            change.upgrade_inputs(inputs)
    return inputs


def upgrade_output_struct(struct: RawOutputStruct) -> RawOutputStruct:
    """Upgrade a raw output struct from an older version to the current format.

    Its inputs are upgraded as well. The input is modified in-place, and also
    returned.

    Raises
    ------
    UnconvertibleError
        If the output struct has no equivalent in the current version.
    """
    upgrade_inputs(struct.inputs)
    for release in releases_since(struct.version):
        for change in release.changes:
            change.upgrade_output_struct(struct)
    return struct


def historical_struct_names(name: str) -> set[str]:
    """Return all names an output struct has had in the history of the file format.

    Parameters
    ----------
    name
        The current name of the output struct.
    """
    names = {name}
    for release in reversed(FORMAT_HISTORY):
        for change in release.changes:
            if isinstance(change, RenameOutputStruct) and change.new in names:
                names.add(change.old)
            elif (
                isinstance(change, UnconvertibleOutputStruct)
                and change.successor in names
            ):
                names.add(change.name)
    return names


def current_struct_name(name: str) -> str:
    """Return the current name of an output struct, given a name it had in the past."""
    for release in FORMAT_HISTORY:
        for change in release.changes:
            if isinstance(change, RenameOutputStruct) and change.old == name:
                name = change.new
            elif (
                isinstance(change, UnconvertibleOutputStruct)
                and change.name == name
                and change.successor is not None
            ):
                name = change.successor
    return name


#: Output structs whose fields are never boxes (so can't be in lightcones etc.)
_NON_BOX_STRUCTS = ("HaloCatalog", "PerturbedHaloCatalog")


def upgrade_box_quantities(
    quantities: dict[str, np.ndarray], inputs: RawInputs
) -> dict[str, np.ndarray]:
    """Upgrade quantities named after fields of output structs, from an older version.

    Lightcones, and the global quantities of lightcones and global-evolution
    simulations, store quantities named after the arrays of the output structs that
    they come from (e.g. ``brightness_temp`` or ``halo_sfr``). This function renames
    (and converts) them in the same way as the arrays of the output structs.

    Parameters
    ----------
    quantities
        A dictionary of the quantities, keyed by their (old) names. It is modified
        in-place, and also returned.
    inputs
        The raw input parameters of the simulation, *already upgraded* to the current
        version with :func:`upgrade_inputs`. Its ``version`` should be the version
        that wrote the quantities.
    """
    field_changes = (RenameField, RemoveField, ConvertOutputStruct)
    for release in releases_since(inputs.version):
        for change in release.changes:
            if (
                isinstance(change, field_changes)
                and change.struct not in _NON_BOX_STRUCTS
            ):
                # The quantities are the arrays of a (fake) struct of the right kind.
                struct = RawOutputStruct(
                    kind=change.struct,
                    inputs=inputs,
                    arrays=quantities,
                    version=inputs.version,
                )
                change.upgrade_output_struct(struct)
    return quantities


def historical_field_names(struct: str, name: str) -> set[str]:
    """Return all names a field of an output struct has had in the format history.

    Parameters
    ----------
    struct
        The current name of the output struct.
    name
        The current name of the field.
    """
    structs = historical_struct_names(struct)
    names = {name}
    for release in reversed(FORMAT_HISTORY):
        for change in release.changes:
            if (
                isinstance(change, RenameField)
                and change.struct in structs
                and change.new in names
            ):
                names.add(change.old)
    return names


def current_field_name(old_name: str, structs: Iterable[str] | None = None) -> str:
    """Return the current name of a field, given an old name.

    Only renames of fields of the given structs are considered (or all structs, if
    None). If the old name was renamed differently in different structs, it is
    returned unchanged.
    """
    name = old_name
    for release in FORMAT_HISTORY:
        candidates = {
            change.new
            for change in release.changes
            if isinstance(change, RenameField)
            and change.old == name
            and (structs is None or change.struct in structs)
        }
        if len(candidates) == 1:
            name = candidates.pop()
    return name


def describe_format_changes(since: str | Version | None = None) -> str:
    """Return a human-readable changelog of the file format.

    Parameters
    ----------
    since
        Only describe the changes affecting files written by this version. By default,
        all changes since v4.0 are described.
    """
    releases = FORMAT_HISTORY if since is None else releases_since(since)
    lines = []
    for release in releases:
        lines.append(f"v{release.from_version} -> v{release.to_version}:")
        lines.extend(f"  - {change.description}" for change in release.changes)
    return "\n".join(lines)
