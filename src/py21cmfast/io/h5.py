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
        /user_params/
        /cosmo_params/
        /flag_options/
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

import attrs
import h5py
import numpy as np
import warnings
from pathlib import Path

from .. import __version__
from ..wrapper import inputs as istruct
from ..wrapper import outputs as ostruct
from ..wrapper._utils import snake_to_camel
from ..wrapper.arrays import Array, H5Backend
from ..wrapper.arraystate import ArrayState
from ..wrapper.inputs import InputParameters


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
            if group in fl:
                group = fl[group]
            else:
                group = fl.create_group(group)
        else:
            group = fl

        group.attrs["21cmFAST-version"] = __version__
        group = group.create_group(output._name)

        if hasattr(output, "redshift"):
            group.attrs["redshift"] = output.redshift

        write_outputs_to_group(output, group)
        _write_inputs_to_group(output.inputs, group)


def write_input_struct(struct, fl: h5py.File | h5py.Group) -> None:
    """Write a particular input struct (e.g. UserParams) to an HDF5 file."""
    dct = struct.asdict()

    for kk, v in dct.items():
        try:
            fl.attrs[kk] = "none" if v is None else v
        except TypeError as e:
            raise TypeError(
                f"key {kk} with value {v} is not able to be written to HDF5 attrs!"
            ) from e


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
    must_close = False
    if isinstance(group, str | Path):
        file = h5py.File(group, "a")
        group = file
        must_close = True

    grp = group.create_group("InputParameters")

    # Write 21cmFAST version to the file
    grp.attrs["21cmFAST-version"] = __version__

    write_input_struct(inputs.user_params, grp.create_group("user_params"))
    write_input_struct(inputs.cosmo_params, grp.create_group("cosmo_params"))
    write_input_struct(inputs.astro_params, grp.create_group("astro_params"))
    write_input_struct(inputs.flag_options, grp.create_group("flag_options"))

    grp.attrs["random_seed"] = inputs.random_seed
    grp["node_redshifts"] = (
        h5py.Empty(None)
        if inputs.node_redshifts is None
        else np.array(inputs.node_redshifts)
    )

    if must_close:
        file.close()


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
        group.attrs[k] = getattr(output, k)

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

        if struct is not None and struct in group:
            group = group[struct]
        elif len(group.keys()) > 1:
            raise ValueError(f"Multiple structs found in {path}:{group}")
        else:
            struct = list(group.keys())[0]
            group = group[struct]

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
        group = file["InputParameters"]
        close_after = True
    elif isinstance(group, h5py.File):
        group = group["InputParameters"]

    file_version = group.attrs.get("21cmFAST-version", None)
    if file_version > __version__:
        warnings.warn(
            f"File created with a newer version {file_version} of 21cmFAST than this {__version__}. "
            f"Reading may break. Consider updating 21cmFAST."
        )

    if file_version is None:
        # pre-v4 file
        out = _read_inputs_pre_v4(group, safe=safe)
    else:
        out = _read_inputs_v4(group, safe=safe)

    if close_after:
        file.close()

    return out


def _read_inputs_pre_v4(group: h5py.Group, safe: bool = True):

    input_classes = [
        istruct.UserParams,
        istruct.CosmoParams,
        istruct.AstroParams,
        istruct.FlagOptions,
    ]
    input_class_names = [cls.__name__ for cls in input_classes]

    # Read the input parameter dictionaries from file.
    kwargs = {}
    for k in attrs.fields_dict(InputParameters):
        kfile = k.lstrip("_")
        input_class_name = snake_to_camel(kfile)

        if input_class_name in input_class_names:
            kls = input_classes[input_class_names.index(input_class_name)]

            subgrp = group[kfile]
            dct = dict(subgrp.attrs)
            kwargs[k] = kls.from_subdict(dct, safe=safe)
        else:
            kwargs[k] = group.attrs[kfile]
    return InputParameters(**kwargs)


def _read_inputs_v4(group: h5py.Group, safe: bool = True):
    # Read the input parameter dictionaries from file.
    kwargs = {}
    for k, fld in attrs.fields_dict(InputParameters).items():
        if fld.type in istruct.InputStruct._subclasses:
            kls = istruct.InputStruct._subclasses[fld.type]

            subgrp = group[k]
            dct = dict(subgrp.attrs)
            kwargs[k] = kls.from_subdict(dct, safe=safe)
        elif k in group.attrs:
            kwargs[k] = group.attrs[k]
        else:
            d = group[k][()]
            if d is h5py.Empty(None):
                kwargs[k] = None
            else:
                kwargs[k] = d

    return InputParameters(**kwargs)


def _read_outputs(group: h5py.Group):
    file_version = group.attrs.get("21cmFAST-version", None)

    if file_version > __version__:
        warnings.warn(
            f"File created with a newer version of 21cmFAST than this. Reading may break. Consider updating 21cmFAST to at least {file_version}"
        )
    if file_version is None:
        # pre-v4 file
        return _read_outputs_pre_v4(group)
    else:
        return _read_outputs_v4(group)


def _read_outputs_pre_v4(group: h5py.Group):
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


def _read_outputs_v4(group: h5py.Group):
    # I actually think the reader is the same in v4.
    return _read_outputs_pre_v4(group)
