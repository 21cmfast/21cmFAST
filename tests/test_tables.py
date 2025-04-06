"""Test initializing tables in C."""

from py21cmfast import AstroFlags, AstroParams, CosmoParams, MatterParams
from py21cmfast.c_21cmfast import lib


def test_init_heat():
    # NOTE: this is not ideal, but the tables depend on POPX_NION which is no longer global
    lib.Broadcast_struct_global_all(
        MatterParams().cstruct,
        CosmoParams().cstruct,
        AstroParams().cstruct,
        AstroFlags().cstruct,
    )
    out = lib.init_heat()
    assert out == 0
