"""Tests of the caching module."""

from pathlib import Path

import attrs
import numpy as np
import pytest

from py21cmfast import Coeval, InputParameters
from py21cmfast.io import caching, h5
from py21cmfast.wrapper import outputs


def create_full_run_cache(cachedir: Path) -> caching.RunCache:
    inputs = InputParameters.from_template(
        "latest",
        random_seed=12345,
        node_redshifts=np.arange(12, 38, 3.0)[::-1],
    ).evolve_input_structs(HII_DIM=10, DIM=20, BOX_LEN=75.0, ZPRIME_STEP_FACTOR=1.3)
    cache = caching.RunCache.from_inputs(inputs, caching.OutputCache(cachedir))

    for fldname, fld in attrs.asdict(cache, recurse=False).items():
        if isinstance(fld, dict):
            for z, fname in fld.items():
                o = getattr(outputs, fldname).new(redshift=z, inputs=inputs)
                o._init_arrays()

                # Go through each array and set it to be "computed" so we can trick
                # the writer into writing it out to file.
                for k, v in o.arrays.items():
                    setattr(o, k, v.with_value(v.value))

                # Mock the primitive fields as well...
                for fld in o.struct.primitive_fields:
                    setattr(o, fld, 0.0)

                h5.write_output_to_hdf5(o, fname)
        elif fldname == "InitialConditions":
            o = outputs.InitialConditions.new(inputs=inputs)
            o._init_arrays()
            for k, v in o.arrays.items():
                setattr(o, k, v.with_value(v.value))
            h5.write_output_to_hdf5(o, fld)
    return cache


@pytest.fixture(scope="module")
def full_run_cache(tmp_path_factory):
    return create_full_run_cache(tmp_path_factory.mktemp("full_run_cache"))


@pytest.fixture(scope="module")
def partial_run_cache(tmp_path_factory):
    cache = create_full_run_cache(tmp_path_factory.mktemp("partial_run_cache"))
    cache.PerturbedField[cache.inputs.node_redshifts[-1]].unlink()
    return cache


class TestRunCache:
    """Tests of the RunCache class."""

    def test_optional_boxes(self, tmp_path: Path):
        """Test that the RunCache can be created with optional boxes."""
        inputs = InputParameters.from_template("latest-dhalos", random_seed=12345)
        cache = caching.RunCache.from_inputs(inputs, caching.OutputCache(tmp_path))

        assert isinstance(cache.HaloBox, dict)
        assert isinstance(cache.PerturbHaloField, dict)
        assert isinstance(cache.InitialConditions, Path)
        assert isinstance(cache.PerturbedField, dict)
        assert isinstance(cache.IonizedBox, dict)
        assert isinstance(cache.BrightnessTemp, dict)
        assert isinstance(cache.TsBox, dict)
        assert isinstance(cache.XraySourceBox, dict)

        assert len(cache.HaloBox) == len(inputs.node_redshifts)

        inputs = InputParameters.from_template("simple", random_seed=12345)
        cache = caching.RunCache.from_inputs(inputs, caching.OutputCache(tmp_path))

        assert cache.HaloBox is None

    def test_from_example_file(self, full_run_cache: caching.RunCache):
        """Test the from_example_file classmethod."""
        for fld in attrs.asdict(full_run_cache, recurse=False).values():
            if isinstance(fld, dict):
                for fname in fld.values():
                    print(fname)
                    cache2 = caching.RunCache.from_example_file(fname)
                    assert full_run_cache == cache2

    def test_is_complete_at(self, full_run_cache, partial_run_cache):
        """Test that is_complete_at works as expected."""
        assert full_run_cache.is_complete_at(z=full_run_cache.inputs.node_redshifts[-1])
        assert full_run_cache.is_complete_at(
            index=len(full_run_cache.inputs.node_redshifts) - 1
        )

        assert not partial_run_cache.is_complete_at(
            z=partial_run_cache.inputs.node_redshifts[-1]
        )
        assert partial_run_cache.is_complete_at(index=0)

    def test_get_output_struct_at_z(self, full_run_cache):
        """Test that get_output_struct_at_z works as expected."""
        cache = full_run_cache
        for name, filedict in attrs.asdict(cache, recurse=False).items():
            if not isinstance(filedict, dict):
                continue

            for idx, z in enumerate(cache.inputs.node_redshifts):
                output = cache.get_output_struct_at_z(kind=name, z=z)
                assert isinstance(output, getattr(outputs, name))
                assert output.redshift == z

                output = cache.get_output_struct_at_z(kind=name, index=idx)
                assert isinstance(output, getattr(outputs, name))
                assert output.redshift == z

                output = cache.get_output_struct_at_z(
                    kind=getattr(outputs, name), index=idx
                )
                assert isinstance(output, getattr(outputs, name))
                assert output.redshift == z

    def test_get_output_struct_at_z_raises(self, full_run_cache):
        """Test that get_output_struct_at_z raises as expected."""
        cache = full_run_cache

        with pytest.raises(ValueError, match="Cannot specify both z and index"):
            cache.get_output_struct_at_z(kind="PerturbedField", z=3.0, index=4)

        with pytest.raises(ValueError, match="Unknown output kind"):
            cache.get_output_struct_at_z(kind="UnknownKind", z=3.0)

        with pytest.raises(ValueError, match="No output struct found"):
            cache.get_output_struct_at_z(kind="TsBox", z=0.0)

    def test_get_all_boxes_at_z(self, full_run_cache):
        """Test get_all_boxes_at_z functionality."""
        cache = full_run_cache

        for z in cache.inputs.node_redshifts:
            boxes = cache.get_all_boxes_at_z(z)
            assert len(boxes) == 4  # number of structs with redshifts (PF, Ts, IB, BT)
            for b in boxes.values():
                assert b.redshift == z

    def test_get_coeval_at_z(self, full_run_cache):
        """Test get_coeval_at_z functionality."""
        cache = full_run_cache

        for z in cache.inputs.node_redshifts:
            coeval = cache.get_coeval_at_z(z)
            assert isinstance(coeval, Coeval)
            assert coeval.redshift == z

    def test_is_complete(self, full_run_cache, partial_run_cache):
        """Test that complete boxes are complete and incomplete boxes are not."""
        assert full_run_cache.is_complete()

        assert not partial_run_cache.is_complete()
