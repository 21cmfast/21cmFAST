"""The building blocks for describing changes to the format of 21cmFAST's files.

This module defines the simple, format-agnostic representations of the contents of a
file (:class:`RawInputs` and :class:`RawOutputStruct`) that the changes operate on, so
that they don't need to know anything about HDF5, and the kinds of change that can be
made to the format (:class:`FormatChange` and its subclasses), which are grouped by
the release that made them in a :class:`FormatRelease`.

The changes made in each release are defined in the modules of
:mod:`py21cmfast.io._compat` (one per release), and are applied to old files by
the functions in :mod:`py21cmfast.io._compat`. The parts of this that users may need
(e.g. the warnings and errors) are exposed publicly in :mod:`py21cmfast.io.migrate`.

Changes are written to be *tolerant*: each one only does something if it finds the
old format in the data it is given (e.g. a renamed parameter is only renamed if the
old name is there). This means they can also safely be applied to files written by
development versions that may have only some of the changes of a given release.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal

import attrs
import numpy as np
from packaging.version import InvalidVersion, Version

from .. import __version__
from ..wrapper.arrays import H5Backend

#: The first version whose file format is supported at all (including its
#: pre-releases): files written before it don't record the version that wrote them.
#: The oldest version that is *currently* supported also depends on the current
#: version (see :func:`oldest_supported_version`).
FIRST_FORMAT_VERSION = Version("4.0.0.dev0")


def _short(version: Version) -> str:
    """Return a short form of a version for messages, e.g. 4.0 for 4.0.0.dev0."""
    return f"{version.major}.{version.minor}"


class CompatibilityWarning(UserWarning):
    """Warning for when a file from an older version can't be fully converted."""


class FormatDeprecationWarning(FutureWarning):
    """Warning for when a file was written by a deprecated version of 21cmFAST.

    Files written by the previous major version of 21cmFAST can still be read and
    migrated, but support for them will be removed in the next major version.
    """


class UnsupportedVersionError(ValueError):
    """Error raised when a file was written by an unsupported version of 21cmFAST."""


class UnconvertibleError(ValueError):
    """Error raised when an object in an old file has no equivalent in this version."""


def parse_version(version: str | bytes | Version | None) -> Version:
    """Parse a 21cmFAST version string, as written into its output files.

    Parameters
    ----------
    version
        The version string (as a str, or bytes as read from some HDF5 files).
        Development versions (e.g. ``4.3.dev12+g1234abc``) are supported.

    Raises
    ------
    UnsupportedVersionError
        If the version is None (as for files written before the first supported
        format), or can't be parsed.
        Whether files of the version are supported is checked by
        :func:`check_support`.
    """
    if version is None:
        raise UnsupportedVersionError(
            "The file does not record the version of 21cmFAST that wrote it, so it "
            "was most likely written by a version older than "
            f"v{_short(FIRST_FORMAT_VERSION)}, which is not supported. The arrays in "
            "such files can still be read directly using h5py."
        )
    if isinstance(version, Version):
        return version
    if isinstance(version, bytes):
        version = version.decode()

    try:
        return Version(str(version))
    except InvalidVersion as e:
        raise UnsupportedVersionError(
            f"Could not understand the 21cmFAST version '{version}'."
        ) from e


CURRENT_VERSION = Version(__version__)

SupportStatus = Literal["supported", "deprecated", "unsupported"]


def oldest_supported_version(current: str | Version | None = None) -> Version:
    """Return the oldest version of 21cmFAST whose files can be read.

    This is the first release of the previous major version (whose files are
    deprecated), but never older than :data:`FIRST_FORMAT_VERSION`. Pre-releases are
    included.

    Parameters
    ----------
    current
        The version of 21cmFAST reading the files. By default, this version.
    """
    current = CURRENT_VERSION if current is None else parse_version(current)
    # For major version 0 (e.g. an install from a git checkout without tags), there
    # is no previous major version.
    previous_major = max(current.major - 1, 0)
    return max(FIRST_FORMAT_VERSION, Version(f"{previous_major}.0.0.dev0"))


def support_status(
    version: str | bytes | Version, current: str | Version | None = None
) -> SupportStatus:
    """Return whether files written by a version of 21cmFAST are supported.

    The policy is that files written by the current major version of 21cmFAST are
    supported, files written by the previous major version are deprecated (they can
    still be read and migrated, with a warning), and files written by any older
    version are not supported. For example, v5 supports files from v5, and
    deprecates files from v4, while v6 no longer supports files from v4 at all.
    Files written before :data:`FIRST_FORMAT_VERSION` are never supported (see
    :func:`oldest_supported_version`). Patch releases are treated like any other
    release of their major version.

    Parameters
    ----------
    version
        The version of 21cmFAST that wrote a file.
    current
        The version of 21cmFAST reading the file. By default, this version.
    """
    version = parse_version(version)
    current = CURRENT_VERSION if current is None else parse_version(current)

    if version < oldest_supported_version(current):
        return "unsupported"
    if version.major < current.major:
        return "deprecated"
    return "supported"


def check_support(version: str | bytes | Version) -> Version:
    """Check that files written by a version of 21cmFAST can be read.

    Raises
    ------
    UnsupportedVersionError
        If files of the version are not supported.

    Warns
    -----
    FormatDeprecationWarning
        If support for files of the version is deprecated.
    """
    version = parse_version(version)
    status = support_status(version)
    major = CURRENT_VERSION.major

    if status == "unsupported":
        msg = (
            f"Files written by 21cmFAST v{version} are not supported (the oldest "
            f"supported version is v{_short(oldest_supported_version())})."
        )
        if version >= FIRST_FORMAT_VERSION:
            msg += (
                f" Support for files written by v{version.major} was removed in "
                f"v{version.major + 2}.0: migrate them with 21cmFAST v{major - 1} "
                "first (using `21cmfast migrate`)."
            )
        raise UnsupportedVersionError(msg)

    if status == "deprecated":
        warnings.warn(
            f"Files written by 21cmFAST v{version.major} (this one was written by "
            f"v{version}) are deprecated, and will not be readable by 21cmFAST "
            f"v{major + 1}. Migrate them to the current format with "
            "`21cmfast migrate`.",
            FormatDeprecationWarning,
            stacklevel=3,
        )
    return version


# ======================================================================================
# Raw representations of the contents of files.
# ======================================================================================
@attrs.define
class RawInputs:
    """The contents of a set of input parameters, as read from a file.

    Parameters
    ----------
    structs
        A dictionary mapping the snake-case name of each input struct (e.g.
        ``"astro_params"``) to a dictionary of the parameters in the file.
    random_seed
        The random seed.
    node_redshifts
        The node redshifts (or None).
    cosmo_tables
        The (raw) cosmological tables stored in the file, if any.
    version
        The version of 21cmFAST that wrote the file.
    """

    structs: dict[str, dict[str, Any]]
    random_seed: int | None = None
    node_redshifts: Sequence[float] | np.ndarray | None = None
    cosmo_tables: dict[str, Any] | None = None
    version: Version = attrs.field(default=CURRENT_VERSION, converter=parse_version)

    def get(self, struct: str, param: str, default: Any = None) -> Any:
        """Get a parameter of a given struct, or a default if it is not there."""
        return self.structs.get(struct, {}).get(param, default)


@attrs.define(frozen=True)
class ArrayOnDisk:
    """A pointer to an array in a file (that hasn't been read into memory)."""

    backend: H5Backend
    shape: tuple[int, ...] = attrs.field(converter=tuple)
    dtype: np.dtype = attrs.field(converter=np.dtype)


#: The source of an array in a raw output struct: either a pointer to an array in a
#: file, or the array itself.
ArraySource = ArrayOnDisk | np.ndarray


@attrs.define
class RawOutputStruct:
    """The contents of an output struct, as read from a file.

    Parameters
    ----------
    kind
        The name of the kind of output struct (e.g. ``"InitialConditions"``).
    inputs
        The raw input parameters of the struct.
    arrays
        A dictionary mapping the name of each array in the file to its source.
    primitives
        A dictionary of the non-array fields of the struct.
    redshift
        The redshift of the struct, if it has one.
    version
        The version of 21cmFAST that wrote the file.
    """

    kind: str
    inputs: RawInputs
    arrays: dict[str, ArraySource] = attrs.field(factory=dict)
    primitives: dict[str, Any] = attrs.field(factory=dict)
    redshift: float | None = None
    version: Version = attrs.field(default=CURRENT_VERSION, converter=parse_version)


def warn(msg: str) -> None:
    """Emit a :class:`CompatibilityWarning`."""
    warnings.warn(msg, CompatibilityWarning, stacklevel=4)


# ======================================================================================
# Kinds of change to the format.
# ======================================================================================
class FormatChange:
    """A change to the format of 21cmFAST outputs.

    Subclasses implement :meth:`upgrade_inputs` and/or :meth:`upgrade_output_struct`,
    which modify raw data in-place, from the format before the change to the format
    after the change. Both should do nothing if the data is already in the new format.
    """

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        raise NotImplementedError

    def upgrade_inputs(self, inputs: RawInputs) -> None:
        """Upgrade raw input parameters (in-place)."""

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Upgrade a raw output struct (in-place)."""


@attrs.define(frozen=True)
class RenameParameter(FormatChange):
    """An input parameter was renamed (and possibly moved to another struct)."""

    struct: str
    old: str
    new: str
    new_struct: str | None = None

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        new_struct = self.new_struct or self.struct
        return f"Input parameter {self.struct}.{self.old} -> {new_struct}.{self.new}"

    def upgrade_inputs(self, inputs: RawInputs) -> None:
        """Rename the parameter, if its old name is present."""
        params = inputs.structs.get(self.struct, {})
        if self.old not in params:
            return
        val = params.pop(self.old)
        inputs.structs.setdefault(self.new_struct or self.struct, {}).setdefault(
            self.new, val
        )


@attrs.define(frozen=True)
class RemoveParameter(FormatChange):
    """An input parameter was removed."""

    struct: str
    name: str
    reason: str

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        return f"Input parameter {self.struct}.{self.name} removed: {self.reason}"

    def upgrade_inputs(self, inputs: RawInputs) -> None:
        """Remove the parameter, if present."""
        inputs.structs.get(self.struct, {}).pop(self.name, None)


@attrs.define(frozen=True)
class AddParameter(FormatChange):
    """A new input parameter was added.

    When the parameter is not present in an older file, it is set to the value that
    reproduces the behaviour of the older version, as returned by ``old_behaviour``
    (a function of the raw inputs). If ``old_behaviour`` is None, or returns None,
    the parameter is left out, so that it takes its default value.
    """

    struct: str
    name: str
    old_behaviour: Callable[[RawInputs], Any] | None = None
    explanation: str = ""

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        out = f"New input parameter {self.struct}.{self.name}"
        if self.explanation:
            out += f": {self.explanation}"
        return out

    def upgrade_inputs(self, inputs: RawInputs) -> None:
        """Set the value of the parameter to reproduce the older version's behaviour."""
        params = inputs.structs.setdefault(self.struct, {})
        if self.name in params or self.old_behaviour is None:
            return
        if (val := self.old_behaviour(inputs)) is not None:
            params[self.name] = val


@attrs.define(frozen=True)
class ConvertParameters(FormatChange):
    """A more complex change to input parameters, done by a function."""

    func: Callable[[RawInputs], None]
    explanation: str

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        return self.explanation

    def upgrade_inputs(self, inputs: RawInputs) -> None:
        """Apply the conversion function."""
        self.func(inputs)


@attrs.define(frozen=True)
class RenameOutputStruct(FormatChange):
    """An output struct was renamed."""

    old: str
    new: str

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        return f"Output struct {self.old} -> {self.new}"

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Rename the kind of the struct."""
        if struct.kind == self.old:
            struct.kind = self.new


@attrs.define(frozen=True)
class UnconvertibleOutputStruct(FormatChange):
    """An output struct was removed, or changed so much that it can't be converted.

    Parameters
    ----------
    name
        The (old) name of the output struct.
    reason
        Why it can't be converted.
    successor
        The name of the output struct that replaced it, if any.
    """

    name: str
    reason: str
    successor: str | None = None

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        if self.successor is None:
            return f"Output struct {self.name} removed: {self.reason}"
        return (
            f"Output struct {self.name} -> {self.successor}, but can't be "
            f"converted: {self.reason}"
        )

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Raise an error, since the struct can't be converted."""
        if struct.kind == self.name:
            raise UnconvertibleError(
                f"{self.name} (written by 21cmFAST v{struct.version}) can't be "
                f"converted to the current version of 21cmFAST: {self.reason}"
            )


def load_array(source: ArraySource) -> np.ndarray:
    """Return the data of an array in a raw output struct."""
    return source.backend.read() if isinstance(source, ArrayOnDisk) else source


@attrs.define(frozen=True)
class RenameField(FormatChange):
    """A field (array or other attribute) of an output struct was renamed.

    The ``struct`` is the name of the struct *after* any renaming of the struct in the
    same release (renamings of structs are applied first).

    If the meaning of the field also changed, ``transform`` converts the old value
    (for arrays, the data of the array) to the new one, given the old value and the
    raw struct. It may return None, in which case the field is dropped (the
    transform itself should warn if information is lost).
    """

    struct: str
    old: str
    new: str
    transform: Callable[[Any, RawOutputStruct], Any] | None = None
    explanation: str = ""

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        out = f"{self.struct}.{self.old} -> {self.struct}.{self.new}"
        if self.explanation:
            out += f": {self.explanation}"
        return out

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Rename the field, if present."""
        if struct.kind != self.struct:
            return
        for dct in (struct.arrays, struct.primitives):
            if self.old not in dct or self.new in dct:
                continue

            val = dct.pop(self.old)
            if self.transform is not None:
                if dct is struct.arrays:
                    val = load_array(val)
                val = self.transform(val, struct)
            if val is not None:
                dct[self.new] = val


@attrs.define(frozen=True)
class ConvertOutputStruct(FormatChange):
    """A more complex change to an output struct, done in-place by a function."""

    struct: str
    func: Callable[[RawOutputStruct], None]
    explanation: str

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        return f"{self.struct}: {self.explanation}"

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Apply the conversion function."""
        if struct.kind == self.struct:
            self.func(struct)


@attrs.define(frozen=True)
class RemoveField(FormatChange):
    """A field of an output struct was removed or can't be converted.

    If ``warn`` is True, a :class:`CompatibilityWarning` is emitted when the field is
    found in a file, since the information in it is lost.
    """

    struct: str
    name: str
    reason: str
    warn: bool = True

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        return f"{self.struct}.{self.name} removed: {self.reason}"

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Remove the field, if present."""
        if struct.kind != self.struct:
            return
        for dct in (struct.arrays, struct.primitives):
            if self.name in dct:
                del dct[self.name]
                if self.warn:
                    warn(
                        f"The field '{self.name}' of {self.struct} in this file "
                        f"(written by 21cmFAST v{struct.version}) can't be used by "
                        f"this version of 21cmFAST, and is ignored: {self.reason}"
                    )


@attrs.define(frozen=True)
class AddField(FormatChange):
    """A new non-array field was added to an output struct.

    When the field is missing in an older file, it is set to the value given by
    ``value`` (a function of the raw struct). If that is None (or returns None), the
    field is left unset, so that it takes its default value.
    """

    struct: str
    name: str
    value: Callable[[RawOutputStruct], Any] | None = None
    explanation: str = ""

    @property
    def description(self) -> str:
        """A human-readable description of the change."""
        out = f"New field {self.struct}.{self.name}"
        if self.explanation:
            out += f": {self.explanation}"
        return out

    def upgrade_output_struct(self, struct: RawOutputStruct) -> None:
        """Set the value of the field, if it is not present."""
        if struct.kind != self.struct or self.name in struct.primitives:
            return
        if self.value is not None and (val := self.value(struct)) is not None:
            struct.primitives[self.name] = val


def _to_version_after_from_version(
    inst: FormatRelease, att: attrs.Attribute[Version], val: Version
) -> None:
    if val <= inst.from_version:
        raise ValueError(
            f"to_version ({val}) must be after from_version ({inst.from_version})"
        )


@attrs.define(frozen=True)
class FormatRelease:
    """The changes to the file format made in a release of 21cmFAST.

    Parameters
    ----------
    from_version
        The previous release that changed the file format (or the oldest supported
        version, for the first step of the history). The changes convert files from
        this format, and from any later version before ``to_version`` (including
        patch releases that didn't change the format).
    to_version
        The release in which the changes were made.
    changes
        The changes, in the order in which they should be applied.
    """

    from_version: Version = attrs.field(converter=Version)
    to_version: Version = attrs.field(
        converter=Version, validator=_to_version_after_from_version
    )
    changes: tuple[FormatChange, ...] = attrs.field(converter=tuple)

    @property
    def module_name(self) -> str:
        """The name of the module of :mod:`py21cmfast.io._compat` defining this."""

        def tag(version: Version) -> str:
            parts = list(version.release)
            while len(parts) > 2 and parts[-1] == 0:
                parts.pop()
            return "_".join(str(p) for p in parts)

        return f"_v{tag(self.from_version)}_v{tag(self.to_version)}"

    def applies_to(self, version: Version) -> bool:
        """Whether the changes of this release need to be applied to a file version.

        Files written by any version before ``to_version`` need the changes (older
        versions get the changes of earlier releases first). Files written by
        development versions leading up to this release (e.g. ``4.3.dev12``) are
        considered to need the changes, since they may have been written before some
        of them were made.
        """
        return version < self.to_version
