"""Tests of the HDF5 read/write functionality."""

from pathlib import Path

import h5py
import numpy as np
import pytest

import py21cmfast as p21c
from py21cmfast import InputParameters
from py21cmfast._templates import list_templates
from py21cmfast.io import h5
from py21cmfast.wrapper.outputs import IonizedBox, TsBox


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


# Marks for the advisories that some templates trigger (to be removed with #778).
def _h5_roundtrip_marks(model, size):
    marks = []
    if model == "Qin20":
        marks.append(
            pytest.mark.filterwarnings(
                "ignore:^USE_MCGS needs a non-trivial V_CB_MODEL:UserWarning"
            )
        )
    if model in ("const-zeta", "Munoz21"):
        marks.append(
            pytest.mark.filterwarnings(
                "ignore:^Your model .*uses the EPS conditional mass function:UserWarning"
            )
        )
    if model in (
        "latest",
        "minihalos",
        "latest-discrete",
        "minihalos-discrete",
        "Park19",
        "fixed-halos",
        "Qin20",
        "Munoz21",
    ) and size in ("size-tiny", "size-small"):
        marks.append(
            pytest.mark.filterwarnings(
                "ignore:^You are setting R_BUBBLE_MAX != 50 when RECOMB_MODEL:UserWarning"
            )
        )
    return marks


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
            pytest.param((t, s), marks=_h5_roundtrip_marks(t, s))
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

    def test_roundtrip_without_precomputed_cosmo_tables(self, tmp_path, monkeypatch):
        """Cosmo tables are omitted in cache files when not precomputed."""
        inputs = InputParameters(random_seed=0).evolve_input_structs(
            POWER_SPECTRUM="CLASS"
        )
        pth = tmp_path / "tmp.h5"
        h5._write_inputs_to_group(inputs, pth)

        with h5py.File(pth, "r") as fl:
            assert "cosmo_tables" not in fl["InputParameters"]

        new = h5.read_inputs(pth)
        with pytest.raises(AttributeError, match="has no attribute 'cosmo_tables'"):
            object.__getattribute__(new, "cosmo_tables")

        def _raise_recompute(*args, **kwargs):
            raise RuntimeError("recompute")

        monkeypatch.setattr("py21cmfast.wrapper.inputs.run_classy", _raise_recompute)
        # Access after deserialization without cached tables should invoke computation.
        with pytest.raises(RuntimeError, match="recompute"):
            _ = new.cosmo_tables

    def test_roundtrip_with_precomputed_cosmo_tables(self, tmp_path, monkeypatch):
        """Cached cosmo tables in file are reused on read."""
        inputs = InputParameters(random_seed=0).evolve_input_structs(
            POWER_SPECTRUM="CLASS", K_MAX_FOR_CLASS=1.0
        )
        _ = inputs.cosmo_tables

        pth = tmp_path / "tmp.h5"
        h5._write_inputs_to_group(inputs, pth)

        with h5py.File(pth, "r") as fl:
            assert "cosmo_tables" in fl["InputParameters"]

        new = h5.read_inputs(pth)
        assert object.__getattribute__(new, "cosmo_tables") is not None

        def _raise_recompute(*args, **kwargs):
            raise RuntimeError("recompute")

        monkeypatch.setattr("py21cmfast.wrapper.inputs.run_classy", _raise_recompute)
        assert new.cosmo_tables == inputs.cosmo_tables


@pytest.mark.parametrize("struct", [TsBox, IonizedBox])
@pytest.mark.parametrize("cache_on_access", [False, True])
def test_output_units_restored_from_cache_without_unit_attributes(
    default_input_struct_lc, tmp_path, struct, cache_on_access
):
    inputs = default_input_struct_lc.evolve_input_structs(
        USE_TS_FLUCT=True,
        USE_MCGS=True,
        V_CB_MODEL="AVG-DEBUG",
        M_TURN_STELLAR_FEEDBACK=5.0,
        RECOMB_MODEL="inhomogeneous",
        R_BUBBLE_MAX=50.0,
    )
    box = struct.new(inputs=inputs, redshift=10.0)
    for name, array in box.arrays.items():
        setattr(box, name, array.with_value(np.full(array.shape, 2, dtype=np.float32)))
    for name in box._struct.primitive_fields:
        if getattr(box, name) is None:
            setattr(box, name, 0.0)

    path = tmp_path / "output.h5"
    h5.write_output_to_hdf5(box, path)
    # The existing cache format has no dataset unit/description attributes.
    with h5py.File(path, "r") as file:
        for dataset in file[box._name]["OutputFields"].values():
            assert "unit" not in dataset.attrs
            assert "description" not in dataset.attrs

    restored = h5.read_output_struct(path)
    for name, array in restored.arrays.items():
        assert array.unit == box.arrays[name].unit
        assert array.description == box.arrays[name].description
        assert array._value is None
        assert array.state.on_disk
        with p21c.config.use(CACHE_ARRAYS_ON_ACCESS=cache_on_access):
            quantity = array.get_quantity()
            assert quantity.dtype == np.float32
            np.testing.assert_array_equal(quantity.value, 2)
            quantity.value.flat[0] = 3
            assert array.get_quantity().value.flat[0] == (3 if cache_on_access else 2)
        assert array.state.computed_in_mem == cache_on_access
        with h5py.File(path, "r") as file:
            assert file[box._name]["OutputFields"][name][...].flat[0] == 2


class TestLoadHighLevelSimulation:
    """Tests of the `load_high_level_simulation` convenience reader."""

    def test_lightcone(self, lc, tmp_path: Path):
        """A lightcone file should come back as a LightCone."""
        pth = tmp_path / "lightcone.h5"
        lc.save(pth)

        assert isinstance(h5.load_high_level_simulation(pth), p21c.LightCone)

    def test_global_evolution(self, default_input_struct_lc, tmp_path: Path):
        """A global-evolution file should come back as a GlobalEvolution."""
        pth = tmp_path / "global.h5"
        p21c.GlobalEvolution(
            inputs=default_input_struct_lc,
            quantities={
                "brightness_temp": np.zeros(len(default_input_struct_lc.node_redshifts))
            },
        ).save(pth)

        assert isinstance(h5.load_high_level_simulation(pth), p21c.GlobalEvolution)

    def test_unrecognized_file(self, tmp_path: Path):
        """Anything else should raise a helpful error."""
        pth = tmp_path / "not-a-simulation.h5"
        with h5py.File(pth, "w") as fl:
            fl.attrs["something_else"] = True

        with pytest.raises(ValueError, match="not a recognized 21cmFAST output"):
            h5.load_high_level_simulation(pth)
