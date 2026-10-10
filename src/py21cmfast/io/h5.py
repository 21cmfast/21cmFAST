"""Module defining HDF5 backends for reading/writing output structures.

These functions are those used by default in the caching system of 21cmFAST.
In the future, it is possible that other backends might be implemented.

As of version 4, all cache files from 21cmFAST will have the following heirarchical
structure::

    /attrs/
      |-- 21cmFAST-version
      |-- [redshift]
    /<OutputStructName>/
      /InputParameters/
        /attrs/
          |-- 21cmFAST-version
          |-- random_seed
        /simulation_options/
        /matter_options/
        /cosmo_params/
        /astro_options/
        /astro_params/
        /node_redshifts/
      /OutputFields/
        /attrs/
          |-- [primitive_field_1]
          |-- [primitive_field_2]
          |-- [...]
        /[field_1]/
        /[field_2]/
        /.../

"""

import warnings
from pathlib import Path
from typing import Any

import attrs
import deprecation
import h5py
import numpy as np
from packaging.version import Version

from .. import __version__
from .._cfg import config
from ..input_serialization import deserialize_inputs, prepare_inputs_for_serialization
from ..wrapper import outputs as ostruct
from ..wrapper.arrays import Array, H5Backend
from ..wrapper.arraystate import ArrayState
from ..wrapper.inputs import InputParameters
from . import compat


def hdf5_to_dict(grp: h5py.Group) -> dict[str, Any]:
    """Load all data from an HDF5 Group into a dict.

    Essentially the same as toml.load() but for HDF5.
    """
    out = dict(grp.attrs)

    for k, v in grp.items():
        if isinstance(v, h5py.Group):
            out[k] = hdf5_to_dict(v)
        else:
            out[k] = v[()]

    return out


class HDF5FileStructureError(ValueError):
    """An error in the structure of an HDF5 file for 21cmFAST."""


def write_output_to_hdf5(
    output: ostruct.OutputStruct,
    path: Path,
    group: str | None = None,
    mode: str = "w",
    keep_in_memory: bool = True,
):
    """
    Write an output struct in standard HDF5 format.

    Parameters
    ----------
    output
        The OutputStruct to write.
    path : Path
        The path to write the output struct to.
    group : str, optional
        The HDF5 group into which to write the object. By default, this is the root.
    mode : str
        The mode in which to open the file.
    keep_in_memory
        Whether to keep all arrays in memory after writing. If False, arrays are
        loaded and written one at a time, and purged from memory once written, so
        that writing never needs more memory than the largest array.
    """
    if not all(v.state.is_computed for v in output.arrays.values()):
        raise ValueError(
            "Not all boxes have been computed (or maybe some have been purged). Cannot write."
            f"Non-computed boxes: {[k for k, v in output.arrays.items() if not v.state.is_computed]}. "
            f"Computed boxes: {[k for k, v in output.arrays.items() if v.state.is_computed]}"
        )

    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(exist_ok=True, parents=True)

    with h5py.File(path, mode) as fl:
        if group is not None:
            group = fl[group] if group in fl else fl.create_group(group)
        else:
            group = fl

        group.attrs["21cmFAST-version"] = __version__
        group = group.create_group(output._name)

        if hasattr(output, "redshift"):
            group.attrs["redshift"] = output.redshift

        write_outputs_to_group(output, group, keep_in_memory=keep_in_memory)
        _write_inputs_to_group(output.inputs, group)


def _write_inputs_to_group(
    inputs: InputParameters, group: h5py.Group | h5py.File | str | Path
) -> None:
    """Write an InputParameters object into a cache file.

    Here we are careful to close the file only if a raw Path is given, and keep it open
    if a h5py.File/Group is given (since then this is likely being called from another
    function that is also writing other objects to the same file).

    Parameters
    ----------
    inputs
        The input parameters object to write.
    group : h5py.Group | h5py.File | str | Path
        The group or file into which to write the inputs. Note that a new group called
        "InputParameters" will be created inside this group/file.
    """
    if not isinstance(group, h5py.Group):
        with h5py.File(group, "a") as fl:
            _write_inputs_to_group(inputs, fl)
        return

    grp = group.create_group("InputParameters")

    # Write 21cmFAST version to the file
    grp.attrs["21cmFAST-version"] = __version__

    inputsdct = prepare_inputs_for_serialization(
        inputs,
        mode="full",
        only_structs=True,
        camel=False,
        # Cache files should persist cosmo tables only when already materialized,
        # so serialization never forces a CLASS run on write.
        include_cosmo_tables="if_cached",
    )

    # Write the input structs. Note that all the "work" for converting attributes
    # to appropriate values is done in the serialization method above, not here.
    for name, dct in inputsdct.items():
        _grp = grp.create_group(name)
        for key, val in dct.items():
            try:
                _grp.attrs[key] = val
            except TypeError as e:
                if isinstance(val, dict):
                    # A "second layer" of recursion is needed since CosmoTables has an attribute that is itself a non-primitive class (Table1D)
                    _grp_dict = _grp.create_group(key)
                    for key_dict, val_dict in val.items():
                        _grp_dict.attrs[key_dict] = val_dict
                else:
                    raise TypeError(
                        f"key {key} with value {val} is not able to be written to HDF5 attrs!"
                    ) from e

    grp.attrs["random_seed"] = inputs.random_seed
    grp["node_redshifts"] = (
        h5py.Empty(None)
        if inputs.node_redshifts is None
        else np.array(inputs.node_redshifts)
    )


def write_outputs_to_group(
    output: ostruct.OutputStruct,
    group: h5py.Group | h5py.File | str | Path,
    keep_in_memory: bool = True,
):
    """
    Write the compute fields of an OutputStruct to a particular HDF5 subgroup.

    Here we are careful to close the file only if a raw Path is given, and keep it open
    if a h5py.File/Group is given (since then this is likely being called from another
    function that is also writing other objects to the same file).

    Parameters
    ----------
    output
        The OutputStruct to write.
    group
        The HDF5 group into which to write the object. A new group "OutputFields" will
        be created inside this group/file.
    keep_in_memory
        Whether to keep all arrays in memory after writing. If False, arrays are
        loaded and written one at a time, and purged from memory once written.
    """
    need_to_close = False
    if isinstance(group, str | Path):
        file = h5py.File(group, "r")
        group = file
        need_to_close = True

    # Go through all fields in this struct, and save
    group = group.create_group("OutputFields")

    if keep_in_memory:
        # First make sure we have everything in memory
        output.load_all()

    for k, array in output.arrays.items():
        backend = H5Backend(group.file.filename, f"{group.name}/{k}")
        if keep_in_memory:
            new = array.written_to_disk(backend)
        else:
            new = array.loaded_from_disk().purged_to_disk(backend)
        setattr(output, k, new)

    for k in output._struct.primitive_fields:
        try:
            group.attrs[k] = getattr(output, k)
        except TypeError as e:
            raise TypeError(f"Error writing attribute {k} to HDF5") from e

    group.attrs["21cmFAST-version"] = __version__

    if need_to_close:
        file.close()


def read_output_struct(
    path: Path, group: str = "/", struct: str | None = None, safe: bool = True
) -> ostruct.OutputStruct:
    """
    Read an output struct from an HDF5 file.

    Files written by older versions of 21cmFAST (since v4.0) are converted to the
    current format on the fly (see :mod:`py21cmfast.io.compat`).

    Parameters
    ----------
    path : Path
        The path to the HDF5 file.
    group : str, optional
        A path within the HDF5 heirarchy to the top-level of the OutputStruct. This is
        usually the root of the file.
    struct
        A string specifying the kind of OutputStruct to read (e.g. InitialConditions).
        Generally, this does not need to be provided, as cache files contain just a
        single output struct. Either the current name of the kind of struct, or the
        name it had in the version of 21cmFAST that wrote the file, can be given.
    safe
        Whether to read the file in "safe" mode. If True, keys found in the file that
        are not valid attributes of the struct will raise an exception. If False, only
        a warning will be raised.

    Returns
    -------
    OutputStruct
        An OutputStruct that is contained in the cache file.
    """
    raw = read_raw_output_struct(path, group=group, struct=struct)
    return output_struct_from_raw(raw, safe=safe)


def _find_struct_group(group: h5py.Group, struct: str | None) -> h5py.Group:
    if struct is None:
        if len(group.keys()) > 1:
            raise HDF5FileStructureError(
                f"Multiple structs found in {group.file.filename}:{group.name}"
            )
        return group[next(iter(group.keys()))]

    for name in compat.historical_struct_names(struct):
        if name in group:
            return group[name]

    raise KeyError(f"struct {struct} not found in the H5DF group {group}")


def _check_file_version(version: str | None, filename: str) -> Version:
    """Parse the version of 21cmFAST that wrote a file, warning if it is too new."""
    if version is None:
        raise NotImplementedError(
            f"The file {filename} is not a valid 21cmFAST v4 file."
        )

    version = compat.parse_version(version)
    if version > compat.CURRENT_VERSION:
        warnings.warn(
            f"File created with a newer version {version} of 21cmFAST than this "
            f"{__version__}. Reading may break. Consider updating 21cmFAST.",
            stacklevel=3,
        )
    return version


def read_raw_output_struct(
    path: Path, group: str = "/", struct: str | None = None
) -> compat.RawOutputStruct:
    """Read the raw contents of an output struct from an HDF5 file.

    The contents are read exactly as they are in the file, i.e. without converting
    files from older versions to the current format. Arrays are not read into
    memory.

    Parameters
    ----------
    path : Path
        The path to the HDF5 file.
    group : str, optional
        A path within the HDF5 heirarchy to the top-level of the OutputStruct.
    struct
        The kind of OutputStruct to read (if there is more than one in the group).
    """
    with h5py.File(path, "r") as fl:
        group = _find_struct_group(fl[group], struct)
        if "InputParameters" not in group or "OutputFields" not in group:
            raise HDF5FileStructureError(
                f"The group {group.name} in {path} is not a valid output struct."
            )

        fields = group["OutputFields"]
        version = _check_file_version(
            fields.attrs.get("21cmFAST-version", None), fl.filename
        )
        primitives = dict(fields.attrs)
        del primitives["21cmFAST-version"]

        return compat.RawOutputStruct(
            kind=group.name.split("/")[-1],
            inputs=_read_raw_inputs(group["InputParameters"]),
            arrays={
                name: compat.ArrayOnDisk(
                    backend=H5Backend(path=fl.filename, dataset=dataset.name),
                    shape=dataset.shape,
                    dtype=dataset.dtype,
                )
                for name, dataset in fields.items()
                if isinstance(dataset, h5py.Dataset)
            },
            primitives=primitives,
            redshift=group.attrs.get("redshift"),
            version=version,
        )


def output_struct_from_raw(
    raw: compat.RawOutputStruct, safe: bool = True
) -> ostruct.OutputStruct:
    """Create an OutputStruct from its raw contents (as read from a file).

    If the raw contents are from a file written by an older version of 21cmFAST,
    they are first converted to the current format (see :mod:`py21cmfast.io.compat`).
    Any arrays of the output struct that do not exist in such a file are left
    uninitialized (with a warning), while for files written by the current version
    an error is raised.

    Parameters
    ----------
    raw
        The raw contents of the output struct.
    safe
        Whether to raise an error if there are unknown input parameters.
    """
    legacy = compat.is_legacy(raw.version)
    if legacy:
        compat.upgrade_output_struct(raw)

    kls = getattr(ostruct, raw.kind, None)
    if kls is None or not issubclass(kls, ostruct.OutputStruct):
        raise HDF5FileStructureError(f"Unknown kind of output struct: '{raw.kind}'")

    inputs = _inputs_from_raw(raw.inputs, safe=safe, legacy=legacy)

    kwargs = dict(raw.primitives)
    if raw.redshift is not None:
        kwargs["redshift"] = raw.redshift

    if legacy:
        known = {f.alias for f in attrs.fields(kls)}
        if unknown := set(kwargs) - known:
            warnings.warn(
                f"The {raw.kind} (written by 21cmFAST v{raw.version}) contains "
                f"unknown fields {sorted(unknown)}, which are ignored.",
                compat.CompatibilityWarning,
                stacklevel=2,
            )
            kwargs = {k: v for k, v in kwargs.items() if k in known}

    # Create the object with those attributes. Extra emissivity fields are included
    # if they are in the file, regardless of the current configuration.
    extra = {}
    if issubclass(kls, ostruct.EmissivityFields):
        extra["EXTRA_EMISSIVITY_FIELDS"] = "halo_number" in raw.arrays
    with config.use(**extra):
        obj = kls.new(inputs, **kwargs)

    # Now go and make sure all the arrays exist in the file, and have the correct shape.
    # We don't actually read these right now, we just make pointers to the file.
    for name, array in obj.arrays.items():
        if name not in raw.arrays:
            msg = f"Required Array {name} not found in the {raw.kind} in the file."
            if not legacy:
                raise HDF5FileStructureError(msg + " This file is not valid.")
            warnings.warn(
                f"{msg} The file was written by 21cmFAST v{raw.version}, which did "
                "not save this array for these parameters, so it is unavailable.",
                compat.CompatibilityWarning,
                stacklevel=2,
            )
            continue

        setattr(obj, name, _array_from_source(raw.arrays.pop(name), array, name))

    # Arrays known to the class, but not used with these parameters by the current
    # version (e.g. because it saves less), are dropped. Others are unknown.
    unknown = sorted(set(raw.arrays) - set(kls._array_field_names))
    if legacy and unknown:
        warnings.warn(
            f"The {raw.kind} (written by 21cmFAST v{raw.version}) contains "
            f"unknown arrays {unknown}, which are ignored.",
            compat.CompatibilityWarning,
            stacklevel=2,
        )

    return obj


def _array_from_source(source: compat.ArraySource, array: Array, name: str) -> Array:
    if source.shape != array.shape:
        filename = (
            f" in the file {source.backend.path}"
            if isinstance(source, compat.ArrayOnDisk)
            else ""
        )
        raise HDF5FileStructureError(
            f"Array {name} has shape {source.shape}{filename}, but requires shape "
            f"{array.shape}"
        )

    # We don't check dtype because it can be usually safely cast.
    if isinstance(source, compat.ArrayOnDisk):
        return attrs.evolve(
            array, state=ArrayState(on_disk=True), cache_backend=source.backend
        )
    return array.with_value(source.astype(array.dtype))


def read_inputs(
    group: h5py.Group | Path | h5py.File, safe: bool = True
) -> InputParameters:
    """Read the InputParameters from a cache file.

    Files written by older versions of 21cmFAST (since v4.0) are converted to the
    current format on the fly (see :mod:`py21cmfast.io.compat`).

    Parameters
    ----------
    group : h5py.Group | Path | h5py.File
        A file, or HDF5 Group within a file, to read the input parameters from.
    safe : bool, optional
        If in safe mode, errors will be raised if keys exist in the file that are not
        valid attributes of the InputParameters. Otherwise, only warnings will be raised.

    Returns
    -------
    inputs : InputParameters
        The input parameters contained in the file.
    """
    if not isinstance(group, h5py.Group):
        with h5py.File(group, "r") as file:
            if "InputParameters" in file:
                return read_inputs(file["InputParameters"], safe=safe)
            if len(file.keys()) > 1:
                raise HDF5FileStructureError(
                    f"Multiple sub-groups found in {group}, none of them 'InputParameters'"
                )
            groupname = next(iter(file.keys()))
            return read_inputs(file[groupname]["InputParameters"], safe=safe)

    if isinstance(group, h5py.File):
        group = group["InputParameters"]

    raw = _read_raw_inputs(group)
    legacy = compat.is_legacy(raw.version)
    if legacy:
        compat.upgrade_inputs(raw)
    return _inputs_from_raw(raw, safe=safe, legacy=legacy)


def read_box_quantities(
    group: h5py.Group, inputs_group: h5py.Group, index: Any = ...
) -> dict[str, np.ndarray]:
    """Read a group of quantities named after arrays of OutputStructs.

    This is used for the lightcones and global quantities of high-level outputs.
    Quantities in files written by older versions of 21cmFAST are converted to the
    current format (see :func:`py21cmfast.io.compat.upgrade_box_quantities`).

    Parameters
    ----------
    group
        The group containing a dataset for each quantity.
    inputs_group
        The group containing the input parameters of the simulation.
    index
        An index to apply to each dataset when reading it (e.g. a slice).
    """
    quantities = {k: v[index] for k, v in group.items()}

    raw = _read_raw_inputs(inputs_group)
    if compat.is_legacy(raw.version):
        compat.upgrade_inputs(raw)
        compat.upgrade_box_quantities(quantities, raw)
    return quantities


def _read_raw_inputs(group: h5py.Group) -> compat.RawInputs:
    """Read the raw input parameters from an InputParameters group."""
    kwargs = hdf5_to_dict(group)
    version = _check_file_version(
        kwargs.pop("21cmFAST-version", None), group.file.filename
    )

    return compat.RawInputs(
        # The node_redshifts and random_seed are treated differently.
        node_redshifts=kwargs.pop("node_redshifts", None),
        random_seed=kwargs.pop("random_seed", None),
        cosmo_tables=kwargs.pop("cosmo_tables", None),
        structs=kwargs,
        version=version,
    )


def _inputs_from_raw(
    raw: compat.RawInputs, safe: bool = True, legacy: bool = False
) -> InputParameters:
    structs = dict(raw.structs)
    if raw.cosmo_tables is not None:
        structs["cosmo_tables"] = raw.cosmo_tables

    if legacy:
        # Deprecation warnings for old parameter names are irrelevant to the user,
        # since the names come from the file.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", deprecation.DeprecatedWarning)
            kwargs = deserialize_inputs(structs, safe=safe, include_cosmo_tables=True)
    else:
        kwargs = deserialize_inputs(structs, safe=safe, include_cosmo_tables=True)

    cosmo_tables = kwargs.pop("cosmo_tables", None)
    out = InputParameters(
        node_redshifts=raw.node_redshifts, random_seed=raw.random_seed, **kwargs
    )
    if cosmo_tables is not None:
        # Populate the cached_property slot directly so tables loaded from file are
        # reused without recomputing or triggering the cached_property getter.
        # InputParameters is frozen, and cached_property has no public setter, so
        # direct slot assignment is required to pre-populate the cached value.
        object.__setattr__(out, "cosmo_tables", cosmo_tables)
    return out


def load_high_level_simulation(path: str | Path, safe: bool = True):
    """Read a saved high-level simulation output, determining its type automatically.

    This is a convenience wrapper around the ``from_file`` methods of the high-level
    simulation outputs, for when you don't know (or don't care) which kind of output
    a file holds. To read the lower-level cache files instead, use
    :func:`read_output_struct`.

    Parameters
    ----------
    path
        The path to a saved coeval, lightcone or global-evolution file.
    safe
        Whether to raise an error if the input parameters in the file are not
        readable by this version of 21cmFAST.

    Returns
    -------
    :class:`~py21cmfast.drivers.coeval.Coeval`, :class:`~py21cmfast.drivers.lightcone.LightCone` or :class:`~py21cmfast.drivers.global_evolution.GlobalEvolution`
        The object stored in the file.

    Raises
    ------
    ValueError
        If the file is not a recognized 21cmFAST high-level output.
    """
    # Imported here to avoid a circular import: the drivers use this module to do
    # their own reading and writing.
    from ..drivers.coeval import Coeval
    from ..drivers.global_evolution import GlobalEvolution
    from ..drivers.lightcone import LightCone

    path = Path(path)

    with h5py.File(path, "r") as fl:
        file_attrs = dict(fl.attrs)

    for marker, cls in (
        ("lightcone", LightCone),
        ("coeval", Coeval),
        ("global_evolution", GlobalEvolution),
    ):
        if file_attrs.get(marker, False):
            return cls.from_file(path, safe=safe)

    raise ValueError(
        f"The file {path} is not a recognized 21cmFAST output file "
        "(expected a coeval, lightcone or global-evolution file)."
    )
