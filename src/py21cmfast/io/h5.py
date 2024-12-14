"""Module defining HDF5 backends for reading/writing output structures."""

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
    write_inputs : bool, optional
        Whether to write the inputs to the file. Can be useful to set to False if
        the input file already exists and has parts already written.
    """
    if not all(v.state.computed for v in output.arrays.values()):
        raise OSError(
            "Not all boxes have been computed (or maybe some have been purged). Cannot write."
            f"Non-computed boxes: {[k for k, v in output.arrays.items() if not v.state.computed]}. "
            f"Computed boxes: {[k for k, v in output.arrays.items() if v.state.computed]}"
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
        write_inputs_to_group(output.inputs, group)


def write_input_struct(struct, fl: h5py.File | h5py.Group):
    """Write a particular input struct (e.g. UserParams) to an HDF5 file."""
    dct = struct.asdict()

    for kk, v in dct.items():
        try:
            fl.attrs[kk] = "none" if v is None else v
        except TypeError as e:
            raise TypeError(
                f"key {kk} with value {v} is not able to be written to HDF5 attrs!"
            ) from e


def write_inputs_to_group(inputs, group: h5py.Group | h5py.File | str | Path):
    must_close = False
    if isinstance(group, str | Path):
        file = h5py.File(group, "a")
        group = file
        must_close = True

    grp = group.create_group("InputParameters")

    # Write 21cmFAST version to the file
    grp.attrs["21cmFAST-version"] = __version__

    # TODO: need to get global params in here somehow
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
    Write out this object to a particular HDF5 subgroup.

    Parameters
    ----------
    group
        The HDF5 group into which to write the object.
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
):
    """
    Read an output struct from an HDF5 file.

    Parameters
    ----------
    path : Path
        The path to the HDF5 file.
    group : str, optional
        A path within the HDF5 heirarchy to the top-level of the OutputStruct. This is
        usually the root of the file.
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
        outputs = read_outputs(group["OutputFields"])

    if redshift is not None:
        outputs["redshift"] = redshift
    kls = getattr(ostruct, struct)
    out = kls(inputs=inputs, **outputs)
    out.sync()  # maybe we shouldn't do this
    return out


def read_inputs(group: h5py.Group | Path | h5py.File, safe: bool = True):
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


def read_outputs(group: h5py.Group):
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
