from py21cmfast import AstroParams, CosmoParams, FlagOptions, UserParams
from py21cmfast.c_21cmfast import lib


def test_init_heat():
    # NOTE: this is not ideal, but the tables depend on POPX_NION which is no longer global
    lib.Broadcast_struct_global_all(
        UserParams().cstruct,
        CosmoParams().cstruct,
        AstroParams().cstruct,
        FlagOptions().cstruct,
    )
    out = lib.init_heat()
    assert out == 0
