"""Module defining HDF5 backends for reading/writing output structures."""

import attrs
import h5py
import warnings
from pathlib import Path

from .. import __version__
from ..drivers.param_config import InputParameters
from ..wrapper import inputs as istruct
from ..wrapper import outputs as ostruct
from ..wrapper._utils import snake_to_camel
from ..wrapper.arrays import Array, H5Backend
from ..wrapper.arraystate import ArrayState


def write_output_to_hdf5(
    output,
    path: Path,
    group: str | None = None,
    write_inputs: bool = True,
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
    write_inputs : bool, optional
        Whether to write the inputs to the file. Can be useful to set to False if
        the input file already exists and has parts already written.
    """
    if not all(v.state.computed for v in output.arrays.values()):
        raise OSError(
            "Not all boxes have been computed (or maybe some have been purged). Cannot write."
            f"Non-computed boxes: {[k for k, v in output.arrays.items() if not v.state.computed]}"
        )

    if not write_inputs:
        mode = "a"

    with h5py.File(path, mode) as fl:
        if group is not None:
            if group in fl:
                group = fl[group]
            else:
                group = fl.create_group(group)

        group.attrs["21cmFAST-version"] = __version__

        # Save input parameters to the file
        if write_inputs:
            write_inputs_to_group(output.inputs, group)

        write_outputs_to_group(output, group)


def write_input_struct(struct, fl: h5py.File | h5py.Group):
    """Write a particular input struct (e.g. UserParams) to an HDF5 file."""
    try:
        dct = struct.asdict()
    except AttributeError:
        # Fires when struct is a StructInstanceWrapper like GlobalParams.
        dct = struct

    for kk, v in dct.items():
        try:
            fl.attrs[kk] = "none" if v is None else v
        except TypeError as e:
            raise TypeError(
                f"key {kk} with value {v} is not able to be written to HDF5 attrs!"
            ) from e


def write_inputs_to_group(inputs, group: h5py.Group):
    grp = group.create_group("InputParams")

    # Write 21cmFAST version to the file
    grp.attrs["21cmFAST-version"] = __version__

    # TODO: need to get global params in here somehow
    write_input_struct(inputs.user_params, grp.create_group("UserParams"))
    write_input_struct(inputs.cosmo_params, grp.create_group("CosmoParams"))
    if inputs.astro_params is not None:
        write_input_struct(inputs.astro_params, grp.create_group("AstroParams"))
    if inputs.flag_options is not None:
        write_input_struct(inputs.flag_options, grp.create_group("FlagOptions"))

    grp.attrs["redshift"] = inputs.redshift
    grp.attrs["random_seed"] = inputs.random_seed


def write_outputs_to_group(output, group: h5py.Group):
    """
    Write out this object to a particular HDF5 subgroup.

    Parameters
    ----------
    group
        The HDF5 group into which to write the object.
    """
    # Go through all fields in this struct, and save
    group = group.create_group(output._name)
    group.attrs["21cmFAST-version"] = __version__

    for k, array in output.arrays.items():
        new = array.written_to_disk(H5Backend(group.file.filename, group.name))
        setattr(output, k, new)

    for k in output.struct.primitive_fields:
        group.attrs[k] = getattr(output, k)


def read(path: Path, group: str | None = None):
    """
    Read an output struct from an HDF5 file.

    Parameters
    ----------
    path : Path
        The path to the HDF5 file.
    """
    with h5py.File(path, "r") as fl:
        group = fl[group]

        # There must be two groups within this group:
        subgroups = list(group.keys())
        if len(subgroups) != 2:
            raise ValueError(
                f"Only one subgroup is allowed in {path}, found {len(subgroups)}"
            )

        clsname = [name for name in subgroups if name != "InputParameters"][0]

        kls = getattr(ostruct, clsname)

        inputs = read_inputs(group["InputParameters"])
        outputs = read_outputs(group[clsname])

    out = kls(inputs, **outputs)
    out.sync()  # maybe we shouldn't do this
    return out


def read_inputs(group: h5py.Group):
    file_version = group.attrs.get("21cmFAST-version", None)

    if file_version >= __version__:
        warnings.warn(
            f"File created with a newer version of 21cmFAST than this. Reading may break. Consider updating 21cmFAST to at least {file_version}"
        )
    if file_version is None:
        # pre-v4 file
        return _read_inputs_pre_v4(group)
    else:
        return _read_inputs_v4(group)


def _read_inputs_pre_v4(group: h5py.Group):
    # TODO: I think I'm still missing global params

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
            kwargs[k] = kls.from_subdict(dct, safe=False)
        else:
            kwargs[k] = group.attrs[kfile]
    return kwargs


def _read_inputs_v4(group: h5py.Group):
    # TODO: I think I'm still missing global params

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
        if k in input_class_names:
            kls = input_classes[input_class_names.index(k)]

            subgrp = group[k]
            dct = dict(subgrp.attrs)
            kwargs[k] = kls.from_subdict(dct, safe=False)
        else:
            kwargs[k] = group.attrs[k]
    return kwargs


def read_outputs(group: h5py.Group):
    file_version = group.attrs.get("21cmFAST-version", None)

    if file_version >= __version__:
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
        if k == "version":
            continue

        arrays[k] = val

    return arrays


def _read_outputs_v4(group: h5py.Group):
    # I actually think the reader is the same in v4.
    return _read_inputs_pre_v4(group)
