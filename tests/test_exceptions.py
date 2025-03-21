import pytest

import numpy as np

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
    with pytest.raises(ParameterError):
        status = lib.FunctionThatCatches(
            # WIP: CFFI Refactor
            # subfunc, False, ffi.cast("double *", ffi.from_buffer(answer))
            subfunc,
            False,
            answer,
        )
        _process_exitcode(
            status,
            lib.FunctionThatCatches,
            # WIP: CFFI Refactor
            # (False, ffi.cast("double *", ffi.from_buffer(answer))),
            (False, answer),
        )


def test_pass():
    answer = np.array([0], dtype="f8")
    # WIP: CFFI Refactor
    # lib.FunctionThatCatches(True, True, ffi.cast( "double *", ffi.from_buffer(answer)))
    lib.FunctionThatCatches(True, True, answer)
    assert answer == 5.0
