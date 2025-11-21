"""Test initializing tables in C."""

from py21cmfast.c_21cmfast import lib

from py21cmfast.wrapper.cfuncs import broadcast_input_struct


def test_init_heat(default_input_struct):
    # NOTE: this is not ideal, but the tables depend on POPX_NION which is no longer global
    broadcast_input_struct(inputs=default_input_struct)
    out = lib.init_heat()
    assert out == 0
