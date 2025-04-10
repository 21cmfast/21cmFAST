"""Tests of the caching module."""

from pathlib import Path

import attrs
import numpy as np
import pytest

from py21cmfast import (
    Coeval,
    InitialConditions,
    InputParameters,
    IonizedBox,
    PerturbedField,
)
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
                    cache2 = caching.RunCache.from_example_file(fname)
                    assert full_run_cache == cache2

    def test_bad_example_file(self, perturbed_field: PerturbedField, tmpdirec):
        """Test that the RunCache fails to construct when given a non-cache file."""
        badpath = tmpdirec / Path("testpf.h5")
        h5.write_output_to_hdf5(perturbed_field, badpath)

        with pytest.raises(
            ValueError, match="does not seem to be within a cache structure."
        ):
            caching.RunCache.from_example_file(badpath)

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


class TestOutputCache:
    """Tests of the OutputCache class."""

    def test_readability(
        self,
        ic: InitialConditions,
        cache: caching.OutputCache,
        default_input_struct: InputParameters,
        ionize_box: IonizedBox,
        perturbed_field: PerturbedField,
    ):
        """Test that each type of field can be read from cache."""
        for field in (ic, perturbed_field, ionize_box):
            kwargs = {"inputs": default_input_struct}
            if field.__class__ != InitialConditions:
                kwargs["redshift"] = field.redshift
            field2 = field.__class__.new(**kwargs)
            existing = cache.find_existing(field2)

            assert existing is not None
            assert existing.exists()

            field2 = cache.load(field2)

            assert field is not field2
            assert field == field2

    def test_read_from_filename(
        self, default_input_struct, perturbed_field, ionize_box, ic, cache
    ):
        """Test that each type of field can be read using a filename."""
        for field in (ic, perturbed_field, ionize_box):
            kwargs = {
                "inputs": default_input_struct,
                "kind": field.__class__.__name__,
            }
            if field.__class__ != InitialConditions:
                kwargs["redshift"] = field.redshift
            dlist = cache.list_datasets(**kwargs)

            assert len(dlist) == 1
            field2 = h5.read_output_struct(dlist[0])
            assert field == field2
            assert field is not field2

    def test_read_wildcards(self, perturbed_field, ionize_box, ic, cache):
        """Test that we can read files without specific inputs."""
        dlist = cache.list_datasets()

        assert len(dlist) >= 3
        for field in (ic, perturbed_field, ionize_box):
            filename = cache.get_path(field)
            assert filename in dlist

    def test_match_seed(
        self, cache: caching.OutputCache, default_input_struct: InputParameters
    ):
        """Test that changing the random seed results in a different cache location."""
        ic2 = InitialConditions.new(inputs=default_input_struct.clone(random_seed=3))

        # This fails because we've set the seed and it's different to the existing one.
        with pytest.raises(IOError, match="No cache exists for"):
            cache.load(ic2)

    def test_astro_param_change(
        self, default_input_struct, ic, perturbed_field, ionize_box, cache
    ):
        """Test that changing parameters affects IonizedBox but not ICs or PerturbedField."""
        input_change = default_input_struct.evolve_input_structs(
            F_ESC10=-1.5,
        )

        ic2 = InitialConditions.new(inputs=input_change)
        pf2 = PerturbedField.new(inputs=input_change, redshift=perturbed_field.redshift)
        ib2 = IonizedBox.new(inputs=input_change, redshift=ionize_box.redshift)

        for f2, f in zip((ic2, pf2), (ic, perturbed_field), strict=False):
            existing = cache.find_existing(f2)

            assert existing is not None
            assert existing.exists()

            f2 = cache.load(f2)

            assert f == f2
            assert f is not f2

        assert not cache.find_existing(ib2)

    def test_read_pre_v4(self):
        """Test old-style parameter loading."""
        with pytest.warns(
            UserWarning,
            match="You are loading a file from a previous iteration of 21cmFAST",
        ):
            v3_field = h5.read_output_struct(
                Path(__file__).parent / "test_data" / "test_v3_box.h5"
            )

        # make sure it read in an object
        assert isinstance(v3_field, IonizedBox)
        # make sure it read data into the arrays
        assert not np.all(v3_field.get("xH_box") == 0)
        # make sure that it got *some* inputs correct
        assert v3_field.inputs.simulation_options.BOX_LEN == 64
