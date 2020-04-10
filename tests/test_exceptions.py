import pytest

from py21cmfast.c_21cmfast import ffi
from py21cmfast.c_21cmfast import lib
from py21cmfast.wrapper import PARAMETERERROR
from py21cmfast.wrapper import ParameterError
from py21cmfast.wrapper import _call_c_simple


@pytest.mark.parametrize("subfunc", [True, False])
def test_basic(subfunc):
    status = lib.SomethingThatCatches(subfunc)
    assert status == PARAMETERERROR


@pytest.mark.parametrize("subfunc", [True, False])
def test_simple(subfunc):
    with pytest.raises(ParameterError):
        _call_c_simple(lib.FunctionThatCatches, subfunc, False)


def test_pass():
    answer = _call_c_simple(lib.FunctionThatCatches, True, True)
    assert answer == 5.0
