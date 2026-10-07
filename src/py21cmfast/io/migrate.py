"""Migrate files written by older versions of 21cmFAST to the current version.

The readers in :mod:`py21cmfast.io.h5` can read files written by older versions of
21cmFAST directly (see :mod:`py21cmfast.io.compat`). However, the cache of a
simulation (see :class:`~py21cmfast.io.caching.OutputCache`) locates files by hashes
of their input parameters, which change whenever the input parameters change between
versions, so a cache written by an older version can't be *found* by the current
version. The functions in this module migrate whole caches (re-writing each file in
the current format, at the location the current version expects), as well as
high-level outputs (coeval boxes, lightcones and global evolutions).

Migrated files are always written as new files, and the original files are never
removed.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import attrs
import h5py

from .._cfg import config
from ..wrapper import outputs as ostruct
from . import compat, h5
from .caching import OutputCache

logger = logging.getLogger(__name__)

MigrationStatus = Literal[
    "migrated", "up-to-date", "exists", "skipped", "unconvertible", "failed"
]


@attrs.define(frozen=True)
class MigrationResult:
    """The result of migrating a single file.

    Parameters
    ----------
    source
        The file that was migrated.
    status
        What happened to the file. One of:

        * ``migrated``: the file was converted, and written to ``destination``.
        * ``up-to-date``: the file is already in the current format and location.
        * ``exists``: the destination already exists (and overwriting was not asked
          for), so nothing was written.
        * ``skipped``: the file is not something that can be migrated (e.g. it is not
          a cache file, or not of the kind requested).
        * ``unconvertible``: the file can't be converted to the current version.
        * ``failed``: an error occurred while migrating the file.
    destination
        Where the migrated file is (or would be) written.
    kind
        The (current) name of the kind of object in the file.
    version
        The version of 21cmFAST that wrote the file.
    message
        More information (e.g. why the file was skipped).
    warnings
        Any compatibility warnings raised while converting the file, which indicate
        that information in the file was lost or could not be converted.
    """

    source: Path
    status: MigrationStatus
    destination: Path | None = None
    kind: str | None = None
    version: str | None = None
    message: str = ""
    warnings: tuple[str, ...] = attrs.field(default=(), converter=tuple)


def _compat_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    msgs = [
        str(w.message)
        for w in caught
        if issubclass(w.category, compat.CompatibilityWarning)
    ]
    return list(dict.fromkeys(msgs))


def _cache_path(cache: OutputCache, obj: ostruct.OutputStruct) -> Path:
    # Extra emissivity fields are stored in a differently-named file.
    with config.use(
        EXTRA_EMISSIVITY_FIELDS=isinstance(obj, ostruct.EmissivityFields)
        and "halo_number" in obj.arrays
    ):
        return cache.get_path(obj)


def _write_atomically(obj: ostruct.OutputStruct, path: Path) -> None:
    """Write an output struct, without leaving a partial file if it fails."""
    tmp = path.with_name(path.name + ".migrating")
    try:
        h5.write_output_to_hdf5(obj, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def migrate_cache_file(
    path: str | Path,
    cache: OutputCache,
    *,
    kinds: Iterable[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    safe: bool = True,
) -> MigrationResult:
    """Migrate a single cache file into a cache, in the current format.

    Parameters
    ----------
    path
        The cache file to migrate.
    cache
        The cache into which to write the migrated file. The file is written to the
        location that the current version of 21cmFAST expects for it.
    kinds
        If given, only migrate files containing these kinds of output struct (by
        their current names, e.g. "EmissivityFields"). Others are skipped.
    overwrite
        Whether to overwrite an existing file at the destination.
    dry_run
        If True, don't write anything, but report what would be done.
    safe
        Whether to fail if there are unknown input parameters in the file.

    Returns
    -------
    MigrationResult
        What happened to the file.
    """
    path = Path(path)
    try:
        raw = h5.read_raw_output_struct(path)
    except (OSError, KeyError, h5.HDF5FileStructureError, NotImplementedError) as e:
        return MigrationResult(
            path, "skipped", message=f"not a single-struct cache file ({e})"
        )
    except compat.UnsupportedVersionError as e:
        return MigrationResult(path, "unconvertible", message=str(e))

    version = str(raw.version)
    kind = compat.current_struct_name(raw.kind)
    if kinds is not None and kind not in kinds:
        return MigrationResult(
            path, "skipped", kind=kind, version=version, message="kind not requested"
        )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            obj = h5.output_struct_from_raw(raw, safe=safe)
        except compat.UnconvertibleError as e:
            return MigrationResult(
                path, "unconvertible", kind=kind, version=version, message=str(e)
            )
        except Exception as e:
            logger.debug("Failed to read %s", path, exc_info=True)
            return MigrationResult(
                path,
                "failed",
                kind=kind,
                version=version,
                message=f"{type(e).__name__}: {e}",
            )
    msgs = _compat_warnings(caught)

    destination = _cache_path(cache, obj)
    common = {"destination": destination, "kind": kind, "version": version}

    if destination.exists() and destination.samefile(path):
        if not compat.is_legacy(raw.version):
            return MigrationResult(path, "up-to-date", **common)
        if not overwrite:
            return MigrationResult(
                path,
                "exists",
                message="the migrated file would replace the original file",
                warnings=msgs,
                **common,
            )
    elif destination.exists() and not overwrite:
        return MigrationResult(path, "exists", warnings=msgs, **common)

    if missing := [k for k, v in obj.arrays.items() if not v.state.is_computed]:
        return MigrationResult(
            path,
            "unconvertible",
            message=f"the arrays {missing} required by the current version are not "
            "available in the file",
            warnings=msgs,
            **common,
        )

    if not dry_run:
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_atomically(obj, destination)
        except Exception as e:
            logger.debug("Failed to write %s", destination, exc_info=True)
            return MigrationResult(
                path,
                "failed",
                message=f"{type(e).__name__}: {e}",
                warnings=msgs,
                **common,
            )

    return MigrationResult(path, "migrated", warnings=msgs, **common)


def migrate_cache(
    source: str | Path,
    destination: str | Path | None = None,
    *,
    kinds: Iterable[str] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    safe: bool = True,
) -> list[MigrationResult]:
    """Migrate a cache of boxes written by an older version of 21cmFAST.

    Every cache file found (recursively) in ``source`` is read, converted to the
    current format, and written into the ``destination`` cache, at the location that
    the current version of 21cmFAST expects for it (so that it can be found by
    :class:`~py21cmfast.io.caching.OutputCache` and used by the simulation drivers).
    Original files are never removed.

    .. note:: Converting the format of a box does not change its contents: boxes
       that depend on parts of the physics that have changed since the version that
       wrote them will not be what the current version would compute. Continuing a
       simulation from such boxes mixes the physics of the two versions.

    Parameters
    ----------
    source
        The directory of the cache to migrate.
    destination
        The directory of the cache into which to write the migrated files. By
        default, the same as ``source``.
    kinds
        If given, only migrate files containing these kinds of output struct (by
        their current names, e.g. "InitialConditions").
    overwrite
        Whether to overwrite existing files in the destination.
    dry_run
        If True, don't write anything, but report what would be done.
    safe
        Whether to fail if there are unknown input parameters in a file.

    Returns
    -------
    list of MigrationResult
        What happened to each file found in the source directory.
    """
    source = Path(source)
    if not source.is_dir():
        raise NotADirectoryError(f"{source} is not a directory.")

    cache = OutputCache(Path(destination) if destination is not None else source)
    kinds = None if kinds is None else {compat.current_struct_name(k) for k in kinds}

    return [
        migrate_cache_file(
            path, cache, kinds=kinds, overwrite=overwrite, dry_run=dry_run, safe=safe
        )
        for path in sorted(source.rglob("*.h5"))
    ]


def migrate_high_level_file(
    path: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
    safe: bool = True,
) -> MigrationResult:
    """Migrate a saved high-level output (coeval, lightcone or global evolution).

    Parameters
    ----------
    path
        The file to migrate.
    destination
        Where to write the migrated file.
    overwrite
        Whether to overwrite an existing file at the destination.
    safe
        Whether to fail if there are unknown input parameters in the file.
    """
    path = Path(path)
    destination = Path(destination)

    if destination.exists() and destination.samefile(path):
        raise ValueError("The destination can't be the same as the original file.")

    with h5py.File(path, "r") as fl:
        version = fl.attrs.get("__version__", fl.attrs.get("21cmFAST-version"))
    version = None if version is None else str(version)

    if destination.exists() and not overwrite:
        return MigrationResult(path, "exists", destination, version=version)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            obj = h5.load_high_level_simulation(path, safe=safe)
        except compat.UnconvertibleError as e:
            return MigrationResult(
                path, "unconvertible", destination, version=version, message=str(e)
            )

    tmp = destination.with_name(destination.name + ".migrating")
    try:
        obj.save(tmp, clobber=True)
        tmp.replace(destination)
    except Exception as e:
        logger.debug("Failed to write %s", destination, exc_info=True)
        return MigrationResult(
            path,
            "failed",
            destination,
            kind=type(obj).__name__,
            version=version,
            message=f"{type(e).__name__}: {e}",
            warnings=_compat_warnings(caught),
        )
    finally:
        tmp.unlink(missing_ok=True)

    return MigrationResult(
        path,
        "migrated",
        destination,
        kind=type(obj).__name__,
        version=version,
        warnings=_compat_warnings(caught),
    )
