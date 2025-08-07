"""I/O for the 21cmFAST package."""

__all__ = [
    "read_inputs",
    "read_output_struct",
    "write_output_to_hdf5",
]

from .h5 import read_inputs, read_output_struct, write_output_to_hdf5
