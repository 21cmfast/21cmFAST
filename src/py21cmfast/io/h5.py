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

import h5py
import numpy as np

from .. import __version__
from ..input_serialization import deserialize_inputs, prepare_inputs_for_serialization
from ..wrapper import outputs as ostruct
from ..wrapper.arrays import Array, H5Backend
from ..wrapper.arraystate import ArrayState
from ..wrapper.inputs import InputParameters


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


def write_output_to_hdf5(
    output: ostruct.OutputStruct,
    path: Path,
    group: str | None = None,
    mode: str = "w",
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
    """
    if not all(v.state.is_computed for v in output.arrays.values()):
        raise OSError(
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

        write_outputs_to_group(output, group)
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
        print("I should be closed now!", fl)
        return

    grp = group.create_group("InputParameters")

    # Write 21cmFAST version to the file
    grp.attrs["21cmFAST-version"] = __version__

    inputsdct = prepare_inputs_for_serialization(
        inputs, mode="full", only_structs=True, camel=False
    )

    # Write the input structs. Note that all the "work" for converting attributes
    # to appropriate values is done in the serialization method above, not here.
    # Here, we just write primitives to group attrs.
    for name, dct in inputsdct.items():
        _grp = grp.create_group(name)
        for key, val in dct.items():
            try:
                _grp.attrs[key] = val
            except TypeError as e:
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
    output: ostruct.OutputStruct, group: h5py.Group | h5py.File | str | Path
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
    """
    need_to_close = False
    if isinstance(group, str | Path):
        file = h5py.File(group, "r")
        group = file
        need_to_close = True

    # Go through all fields in this struct, and save
    group = group.create_group("OutputFields")

    # First make sure we have everything in memory
    output.load_all()

    for k, array in output.arrays.items():
        new = array.written_to_disk(H5Backend(group.file.filename, f"{group.name}/{k}"))
        setattr(output, k, new)

    for k in output.struct.primitive_fields:
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
        single output struct.
    safe
        Whether to read the file in "safe" mode. If True, keys found in the file that
        are not valid attributes of the struct will raise an exception. If False, only
        a warning will be raised.

    Returns
    -------
    OutputStruct
        An OutputStruct that is contained in the cache file.
    """
    with h5py.File(path, "r") as fl:
        group = fl[group]

        if struct is None:
            if len(group.keys()) > 1:
                raise ValueError(f"Multiple structs found in {path}:{group}")
            else:
                struct = next(iter(group.keys()))
            group = group[struct]

        elif struct in group:
            group = group[struct]
        else:
            raise KeyError(f"struct {struct} not found in the H5DF group {group}")
        assert "InputParameters" in group
        assert "OutputFields" in group

        redshift = group.attrs.get("redshift")
        inputs = read_inputs(group["InputParameters"], safe=safe)
        outputs = _read_outputs(group["OutputFields"])

    if redshift is not None:
        outputs["redshift"] = redshift
    kls = getattr(ostruct, struct)
    out = kls(inputs=inputs, **outputs)
    return out


def read_inputs(
    group: h5py.Group | Path | h5py.File, safe: bool = True
) -> InputParameters:
    """Read the InputParameters from a cache file.

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
    close_after = False
    if isinstance(group, Path):
        file = h5py.File(group, "r")
        if "InputParameters" in file:
            group = file["InputParameters"]
        elif len(file.keys()) > 1:
            raise ValueError(
                f"Multiple sub-groups found in {group}, none of them 'InputParameters'"
            )
        else:
            groupname = next(iter(file.keys()))
            group = file[groupname]["InputParameters"]
        close_after = True
    elif isinstance(group, h5py.File):
        group = group["InputParameters"]

    file_version = group.attrs.get("21cmFAST-version", None)
    if file_version > __version__:
        warnings.warn(
            f"File created with a newer version {file_version} of 21cmFAST than this {__version__}. "
            f"Reading may break. Consider updating 21cmFAST.",
            stacklevel=2,
        )

    out = _read_inputs_v4(group, safe=safe)

    if close_after:
        file.close()

    return out


def _read_inputs_v4(group: h5py.Group, safe: bool = True):
    # Read the input parameter dictionaries from file.
    kwargs = hdf5_to_dict(group)

    # The node_redshifts and random_seed are treated differently.
    node_redshifts = kwargs.pop("node_redshifts")
    random_seed = kwargs.pop("random_seed")

    kwargs = deserialize_inputs(kwargs)
    return InputParameters(
        node_redshifts=node_redshifts, random_seed=random_seed, **kwargs
    )


def _read_outputs(group: h5py.Group):
    file_version = group.attrs.get("21cmFAST-version", None)

    if file_version > __version__:
        warnings.warn(
            f"File created with a newer version of 21cmFAST than this. Reading may break. Consider updating 21cmFAST to at least {file_version}",
            stacklevel=2,
        )
    else:
        return _read_outputs_v4(group)


def _read_outputs_v4(group: h5py.Group):
    arrays = {
        name: Array(
            dtype=box.dtype,
            shape=box.shape,
            state=ArrayState(on_disk=True),
            cache_backend=H5Backend(path=group.file.filename, dataset=box.name),
        )
        for name, box in group.items()
    }
    for k, val in group.attrs.items():
        if k == "21cmFAST-version":
            continue

        arrays[k] = val

    return arrays
