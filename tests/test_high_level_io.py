import pytest

import attrs
import h5py
import numpy as np
from pathlib import Path

from py21cmfast import (
    BrightnessTemp,
    Coeval,
    InitialConditions,
    LightCone,
    OutputCache,
    TsBox,
    UserParams,
    exhaust_lightcone,
    global_params,
    run_coeval,
)
from py21cmfast.drivers import exhaust
from py21cmfast.drivers.lightcone import AngularLightcone
from py21cmfast.io import h5
from py21cmfast.lightcones import AngularLightconer, RectilinearLightconer


@pytest.fixture(scope="module")
def coeval(ic, default_input_struct_ts, cache):
    return exhaust(
        run_coeval(
            out_redshifts=25.0,
            initial_conditions=ic,
            write=True,
            inputs=default_input_struct_ts,
            cache=cache,
        )
    )


@pytest.fixture(scope="module")
def ang_lightcone(ic, lc, default_input_struct_lc, default_flag_options, cache):
    lcn = AngularLightconer.like_rectilinear(
        match_at_z=lc.lightcone_redshifts.min(),
        max_redshift=lc.lightcone_redshifts.max(),
        user_params=ic.user_params,
        get_los_velocity=True,
    )

    iz, z, coev, anglc = exhaust_lightcone(
        lightconer=lcn,
        initial_conditions=ic,
        write=True,
        inputs=default_input_struct_lc.clone(
            flag_options=default_flag_options.clone(
                APPLY_RSDS=False,
            )
        ),
        cache=cache,
    )
    return anglc


def test_read_bad_file_lc(test_direc: Path, lc: LightCone):
    # create a bad hdf5 file with some good fields,
    #  some bad fields, and some missing fields
    #  in both input parameters and box structures
    fname = test_direc / "_lc.h5"
    lc.save(path=fname)

    with h5py.File(fname, "r+") as f:
        # make gluts, these should be ignored on reading
        f["InputParameters"]["user_params"].attrs["NotARealParameter"] = "fake_param"
        # f["_globals"].attrs["NotARealGlobal"] = "fake_param"

        # make gaps
        del f["InputParameters"]["user_params"].attrs["BOX_LEN"]

    # load without compatibility mode, make sure we throw the right error
    with pytest.raises(ValueError, match="There are extra or missing"):
        LightCone.from_file(fname, safe=True)

    # load in compatibility mode, check that we warn correctly
    with pytest.warns(UserWarning, match="There are extra or missing"):
        lc2 = LightCone.from_file(fname, safe=False)

    # check that the fake fields didn't show up in the struct
    assert not hasattr(lc2.user_params, "NotARealParameter")

    # check that missing fields are set to default
    assert lc2.user_params.BOX_LEN == UserParams().BOX_LEN

    # check that the fields which are good are read in the struct
    assert all(
        getattr(lc2.user_params, field.name) == getattr(lc.user_params, field.name)
        for field in attrs.fields(UserParams)
        if field.name != "BOX_LEN"
    )


def test_read_bad_file_coev(test_direc: Path, coeval: Coeval):
    # create a bad hdf5 file with some good fields,
    #  some bad fields, and some missing fields
    #  in both input parameters and box structures
    fname = test_direc / "_a_bad_file.h5"

    coeval.save(path=fname)
    with h5py.File(fname, "r+") as f:
        # make gluts, these should be ignored on reading
        f["BrightnessTemp"]["InputParameters"]["user_params"].attrs[
            "NotARealParameter"
        ] = "fake_param"

        # make gaps
        del f["BrightnessTemp"]["InputParameters"]["user_params"].attrs["BOX_LEN"]

    # load in the coeval check that we warn correctly
    with pytest.raises(ValueError, match="There are extra or missing"):
        Coeval.from_file(fname, safe=True)

    with pytest.warns(UserWarning, match="There are extra or missing"):
        cv2 = Coeval.from_file(fname, safe=False)

    # check that the fake params didn't show up in the struct
    assert not hasattr(cv2.user_params, "NotARealParameter")

    # check that missing fields are set to default
    assert cv2.user_params.BOX_LEN == UserParams().BOX_LEN

    # check that the fields which are good are read in the struct
    assert all(
        getattr(cv2.user_params, k) == getattr(coeval.user_params, k)
        for k in coeval.user_params.asdict().keys()
        if k != "BOX_LEN"
    )


def test_lightcone_roundtrip(test_direc, lc):
    fname = test_direc / "lc.h5"
    lc.save(path=fname)
    lc2 = LightCone.from_file(fname)

    assert lc == lc2
    assert np.allclose(lc.lightcone_redshifts, lc2.lightcone_redshifts)
    assert np.all(
        np.isclose(lc.lightcones["brightness_temp"], lc2.lightcones["brightness_temp"])
    )


def test_coeval_roundtrip(test_direc, coeval):
    fname = test_direc / "a_coeval.h5"
    coeval.save(fname)
    coeval2 = Coeval.from_file(fname)

    assert coeval == coeval2
    assert np.all(np.isclose(coeval.brightness_temp, coeval2.brightness_temp))


def test_ang_lightcone(lc, ang_lightcone: AngularLightcone):
    # we test that the fields are "highly correlated",
    # and moreso in the one corner where the lightcones
    # should be almost exactly the same, and less so in the other
    # corners, and also less so at the highest redshifts.
    rbt = lc.lightcones["brightness_temp"]
    abt = ang_lightcone.lightcones["brightness_temp"].reshape(rbt.shape)

    fullcorr0 = np.corrcoef(rbt[:, :, 0].flatten(), abt[:, :, 0].flatten())
    fullcorrz = np.corrcoef(rbt[:, :, -1].flatten(), abt[:, :, -1].flatten())

    print("correlation at low z: ", fullcorr0)
    print("correlation at highz: ", fullcorrz)
    assert fullcorr0[0, 1] > fullcorrz[0, 1]  # 0,0 and 1,1 are autocorrs.
    assert fullcorr0[0, 1] > 0.5

    # check corners
    n = rbt.shape[0]
    topcorner = np.corrcoef(
        rbt[: n // 2, : n // 2, 0].flatten(), abt[: n // 2, : n // 2, 0].flatten()
    )
    bottomcorner = np.corrcoef(
        rbt[n // 2 :, n // 2 :, 0].flatten(), abt[n // 2 :, n // 2 :, 0].flatten()
    )
    assert topcorner[0, 1] > bottomcorner[0, 1]


def test_write_to_group(ic: InitialConditions, cache: OutputCache):
    h5.write_output_to_hdf5(ic, path=cache.direc / "a_new_file.h5", group="new_group")

    with h5py.File(cache.direc / "a_new_file.h5", "r") as fl:
        assert "new_group" in fl

    ic2 = h5.read_output_struct(cache.direc / "a_new_file.h5", group="new_group")

    assert ic2 == ic
