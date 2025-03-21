import py21cmfast.c_21cmfast as lib


def test_init_heat():
    out = lib.init_heat()
    assert out == 0
