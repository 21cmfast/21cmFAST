import pytest

import numpy as np

from py21cmfast.c_21cmfast import ffi, lib
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
            subfunc, False, ffi.cast("double *", ffi.from_buffer(answer))
        )
        _process_exitcode(
            status,
            lib.FunctionThatCatches,
            (False, ffi.cast("double *", ffi.from_buffer(answer))),
        )


def test_pass():
    answer = np.array([0], dtype="f8")
    lib.FunctionThatCatches(True, True, ffi.cast("double *", ffi.from_buffer(answer)))
    assert answer == 5.0
