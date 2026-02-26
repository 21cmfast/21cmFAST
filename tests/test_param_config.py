"""Test the _param_config.py module."""

import pytest

from py21cmfast import InputParameters
from py21cmfast.drivers._param_config import _get_inputs_from_call


def test_get_inputs_from_call(ionize_box):
    """Test that we can get inputs, given positional or keyword arguments."""
    assert isinstance(_get_inputs_from_call(ionize_box.inputs), InputParameters)
    assert isinstance(_get_inputs_from_call(inputs=ionize_box.inputs), InputParameters)
    assert isinstance(_get_inputs_from_call(ionize_box), InputParameters)
    assert isinstance(_get_inputs_from_call(kwarg=ionize_box), InputParameters)
    assert isinstance(
        _get_inputs_from_call(
            [
                ionize_box,
            ]
        ),
        InputParameters,
    )
    assert isinstance(
        _get_inputs_from_call(
            kwarg=[
                ionize_box,
            ]
        ),
        InputParameters,
    )

    with pytest.raises(ValueError, match="Could not determine InputParameters"):
        _get_inputs_from_call()
