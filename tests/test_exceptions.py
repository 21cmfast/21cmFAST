"""Test exceptions raised in the C backend."""

import numpy as np
import pytest

import py21cmfast.c_21cmfast as lib
from py21cmfast.wrapper.exceptions import (
    PHOTONCONSERROR,
    ParameterError,
    _process_exitcode,
)


@pytest.mark.parametrize("subfunc", [True, False])
def test_basic(subfunc):
    status = lib.SomethingThatCatches(subfunc)
    assert status == PHOTONCONSERROR


@pytest.mark.parametrize("subfunc", [True, False])
def test_simple(subfunc):
    answer = np.array([0], dtype="f8")

    status = lib.FunctionThatCatches(subfunc, False)
    with pytest.raises(ParameterError):
        _process_exitcode(
            status,
            lib.FunctionThatCatches,
            (False, answer),
        )


def test_pass():
    answer = np.array([0], dtype="f8")
    lib.FunctionThatCatches(True, True, answer)
    assert answer == 5.0
