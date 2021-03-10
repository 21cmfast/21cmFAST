from py21cmfast.c_21cmfast import lib


def test_init_heat():
    out = lib.init_heat()
    assert out == 0
