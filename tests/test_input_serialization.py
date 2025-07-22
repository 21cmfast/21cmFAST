"""Tests of the input_serialization module."""

from typing import Literal

import pytest

from py21cmfast import InputParameters
from py21cmfast import input_serialization as srlz


class TestConvertInputsToDict:
    def test_default_minimal(self):
        """Test that default inputs have no difference to default inputs."""
        inputs = InputParameters(random_seed=0)
        out = srlz.convert_inputs_to_dict(inputs, mode="minimal")
        assert len(out) == 0

    @pytest.mark.parametrize("mode", ["full", "minimal"])
    def test_default_with_nonstructs(self, mode):
        inputs = InputParameters(random_seed=42)
        out = srlz.convert_inputs_to_dict(inputs, mode=mode, only_structs=False)
        assert out["random_seed"] == 42

    def test_default_full_snake(self):
        inputs = InputParameters(random_seed=42)
        out = srlz.convert_inputs_to_dict(inputs, mode="full", camel=False)
        assert "cosmo_params" in out

    def test_not_use_aliases(self):
        inputs = InputParameters(random_seed=42)
        out = srlz.convert_inputs_to_dict(inputs, mode="full", use_aliases=False)
        print(out["SimulationOptions"].keys())
        assert "_DIM" in out["SimulationOptions"]
        assert "DIM" not in out["SimulationOptions"]


class TestPrepareInputsForSerialization:
    """Tests of the prepare_inputs_for_serialization function."""

    def test_default_minimal(self):
        inputs = InputParameters(random_seed=1)
        out = srlz.prepare_inputs_for_serialization(inputs, mode="minimal")
        assert out == {}

    @pytest.mark.parametrize(
        "inputs",
        [
            InputParameters(random_seed=0),
            InputParameters.from_template("Park19", random_seed=0),
            InputParameters.from_template(
                "default", HII_DIM=50, DIM=100, BOX_LEN=50, random_seed=0
            ),
        ],
        ids=[
            "default",
            "park19",
            "explicit-dim-boxlen",
        ],
    )
    @pytest.mark.parametrize("mode", ["full", "minimal"])
    @pytest.mark.parametrize("camel", [True, False])
    def test_roundtrip(
        self, inputs: InputParameters, mode: Literal["full", "minimal"], camel: bool
    ):
        dct = srlz.prepare_inputs_for_serialization(
            inputs, mode=mode, only_structs=True, camel=camel
        )
        # Here we would generally write the output to some format like TOML, then
        # read that back in as a dict, then we need to deserialize from dict:
        dct = srlz.deserialize_inputs(dct)
        new = InputParameters(random_seed=inputs.random_seed, **dct)
        assert new == inputs


class TestDeserializeInputs:
    """Tests of the deserialize_inputs function."""

    def test_extra_toplevel_param_warns(self):
        inputs = srlz.prepare_inputs_for_serialization(
            InputParameters(random_seed=0), mode="full"
        )
        inputs["extra_key"] = {}

        with pytest.warns(UserWarning, match="The following keys were not recognized"):
            srlz.deserialize_inputs(inputs)

    def test_extra_loose_param_warns(self):
        inputs = srlz.prepare_inputs_for_serialization(
            InputParameters(random_seed=0), mode="full"
        )

        with pytest.warns(UserWarning, match="Excess arguments"):
            srlz.deserialize_inputs(inputs, MY_EXTRA_PARAM=True)
