"""Tests of the HDF5 read/write functionality."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from py21cmfast import InputParameters
from py21cmfast.io import h5
from py21cmfast.run_templates import list_templates


class TestHDF5ToDict:
    """Tests of the hdf5_to_dict function."""

    def test_empty(self, tmp_path: Path):
        """Test reading/writing an empty dict."""
        pth = tmp_path / "tmp.h5"
        with h5py.File(pth, "w") as fl:
            fl.create_group("group")

        with h5py.File(pth, "r") as fl:
            out = h5.hdf5_to_dict(fl["group"])

        assert out == {}

    def test_only_attrs(self, tmp_path: Path):
        """Test reading only from  the attrs."""
        pth = tmp_path / "tmp.h5"
        dct = {
            "int": 1,
            "str": "hi",
            "float": 3.14,
        }

        with h5py.File(pth, "w") as fl:
            for k, v in dct.items():
                fl.attrs[k] = v

        with h5py.File(pth, "r") as fl:
            out = h5.hdf5_to_dict(fl)

        assert out == dct

    def test_with_dataset(self, tmp_path):
        """Test a case with a dataset, not only attrs."""
        pth = tmp_path / "tmp.h5"

        with h5py.File(pth, "w") as fl:
            fl["dataset"] = np.zeros(10)

        with h5py.File(pth, "r") as fl:
            out = h5.hdf5_to_dict(fl)

        assert np.allclose(out["dataset"], np.zeros(10))

    def test_recursive(self, tmp_path: Path):
        """Test a recursive set of groups and datasets."""
        pth = tmp_path / "tmp.h5"
        with h5py.File(pth, "w") as fl:
            fl["dataset1"] = np.zeros(10)
            grp = fl.create_group("group")
            grp["dataset2"] = np.ones(10)

        with h5py.File(pth, "r") as fl:
            out = h5.hdf5_to_dict(fl)

        assert np.allclose(out["dataset1"], np.zeros(10))
        assert np.allclose(out["group"]["dataset2"], np.ones(10))


_ALL_TEMPLATE_NAMES = [t["name"] for t in list_templates()]


class TestInputsIO:
    """Tests of reading and writing InputParameters to HDF5."""

    def setup_class(self):
        """Set up the class."""
        self.defaults = InputParameters(random_seed=0)

    def test_appending(self, tmp_path: Path):
        """Test that writing to an existing file doesn't wipe out existing info."""
        pth = tmp_path / "tmp.h5"

        with h5py.File(pth, "w") as fl:
            fl["sentinel"] = 1
            h5._write_inputs_to_group(self.defaults, pth)
            assert "sentinel" in fl

    def test_not_closing_group(self, tmp_path):
        """Test that reading from an open Hdpy.File doesn't close it."""
        pth = tmp_path / "tmp.h5"

        with h5py.File(pth, "w") as fl:
            # Make sure this is not overwritten
            fl["sentinel"] = 2

            h5._write_inputs_to_group(self.defaults, fl)
            fl["new"] = 1  # can write because it's not closed.

        with h5py.File(pth, "r") as fl:
            assert "new" in fl
            assert "InputParameters" in fl
            assert "sentinel" in fl

    @pytest.mark.parametrize(
        "inputs",
        [
            (t, s)
            for t in _ALL_TEMPLATE_NAMES
            if not t.startswith("size-")
            for s in _ALL_TEMPLATE_NAMES
            if s.startswith("size-")
        ],
    )
    def test_roundtrip(
        self, tmp_path: Path, inputs: InputParameters | str | tuple[str]
    ):
        """Test that writing inputs to file then reading them again gets the same answer."""
        if not isinstance(inputs, InputParameters):
            inputs = InputParameters.from_template(inputs, random_seed=0)

        pth = tmp_path / "tmp.h5"
        h5._write_inputs_to_group(inputs, pth)
        new = h5.read_inputs(pth)
        assert new == inputs
