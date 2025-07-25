"""Test high-level I/O."""

from pathlib import Path

import attrs
import h5py
import numpy as np
import pytest

from py21cmfast import (
    Coeval,
    InitialConditions,
    InputParameters,
    LightCone,
    OutputCache,
    SimulationOptions,
    run_coeval,
    run_lightcone,
)
from py21cmfast.drivers.lightcone import AngularLightcone
from py21cmfast.io import h5
from py21cmfast.lightconers import AngularLightconer


@pytest.fixture(scope="module")
def coeval(ic, default_input_struct_ts, cache) -> Coeval:
    return run_coeval(
        out_redshifts=25.0,
        initial_conditions=ic,
        write=True,
        inputs=default_input_struct_ts,
        cache=cache,
    )[0]


@pytest.fixture(scope="module")
def ang_lightcone(ic, lc, default_input_struct_lc, default_astro_options, cache):
    lcn = AngularLightconer.like_rectilinear(
        match_at_z=lc.lightcone_redshifts.min(),
        max_redshift=lc.lightcone_redshifts.max(),
        simulation_options=ic.simulation_options,
    )

    return run_lightcone(
        lightconer=lcn,
        initial_conditions=ic,
        write=True,
        inputs=default_input_struct_lc,
        include_dvdr_in_tau21=False,
        cache=cache,
    )


def test_read_bad_file_lc(test_direc: Path, lc: LightCone):
    # create a bad hdf5 file with some good fields,
    #  some bad fields, and some missing fields
    #  in both input parameters and box structures
    fname = test_direc / "_lc.h5"
    lc.save(path=fname)

    with h5py.File(fname, "r+") as f:
        # make gluts, these should be ignored on reading
        f["InputParameters"]["simulation_options"].attrs["NotARealParameter"] = (
            "fake_param"
        )

        # make gaps
        del f["InputParameters"]["simulation_options"].attrs["BOX_LEN"]

    # load without compatibility mode, make sure we throw the right error
    with pytest.raises(ValueError, match="There are extra or missing"):
        LightCone.from_file(fname, safe=True)

    # load in compatibility mode, check that we warn correctly
    with pytest.warns(UserWarning, match="There are extra or missing"):
        lc2 = LightCone.from_file(fname, safe=False)

    # check that the fake fields didn't show up in the struct
    assert not hasattr(lc2.simulation_options, "NotARealParameter")

    # check that missing fields are set to default
    assert lc2.simulation_options.BOX_LEN == SimulationOptions().BOX_LEN

    # check that the fields which are good are read in the struct
    assert all(
        getattr(lc2.simulation_options, field.name)
        == getattr(lc.simulation_options, field.name)
        for field in attrs.fields(SimulationOptions)
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
        f["BrightnessTemp"]["InputParameters"]["simulation_options"].attrs[
            "NotARealParameter"
        ] = "fake_param"

        # make gaps
        del f["BrightnessTemp"]["InputParameters"]["simulation_options"].attrs[
            "BOX_LEN"
        ]

    # load in the coeval check that we warn correctly
    with pytest.raises(ValueError, match="There are extra or missing"):
        Coeval.from_file(fname, safe=True)

    with pytest.warns(UserWarning, match="There are extra or missing"):
        cv2 = Coeval.from_file(fname, safe=False)

    # check that the fake params didn't show up in the struct
    assert not hasattr(cv2.simulation_options, "NotARealParameter")

    # check that missing fields are set to default
    assert cv2.simulation_options.BOX_LEN == SimulationOptions().BOX_LEN

    # check that the fields which are good are read in the struct
    assert all(
        getattr(cv2.simulation_options, k) == getattr(coeval.simulation_options, k)
        for k in coeval.simulation_options.asdict()
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


def test_ang_lightcone(lc, ang_lightcone: AngularLightcone, plt):
    # we test that the fields are "highly correlated",
    # and moreso in the one corner where the lightcones
    # should be almost exactly the same, and less so in the other
    # corners, and also less so at the highest redshifts.
    rbt = lc.lightcones["brightness_temp"]
    abt = ang_lightcone.lightcones["brightness_temp"].reshape(rbt.shape)

    fullcorr0 = np.corrcoef(rbt[:, :, 0].flatten(), abt[:, :, 0].flatten())
    fullcorrz = np.corrcoef(rbt[:, :, -1].flatten(), abt[:, :, -1].flatten())

    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    ax[0, 0].imshow(rbt[:, :, 0])
    ax[0, 0].set_title("lowest rect")
    ax[0, 1].imshow(rbt[:, :, -1])
    ax[0, 1].set_title("highest rect")
    ax[1, 0].imshow(abt[:, :, 0])
    ax[1, 0].set_title("lowest ang")
    ax[1, 1].imshow(abt[:, :, -1])
    ax[1, 1].set_title("highest ang")

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


def test_write_output_to_hdf5_uncomputed(tmp_path):
    """Test that writing an uncomputed box to file raises an error."""
    inputs = InputParameters(random_seed=0)
    ics = InitialConditions.new(inputs)
    with pytest.raises(ValueError, match="Not all boxes have been computed"):
        h5.write_output_to_hdf5(ics, path=tmp_path / "ics.h5")


def read_nonexistent_output_struct(ic: InitialConditions, tmp_path: Path):
    """Test that attempting to read a non-existent struct from a file fails."""
    pth = tmp_path / "ics.h5"
    h5.write_output_to_hdf5(ic, pth)

    with pytest.raises(
        KeyError, match="struct non-existent not found in the HDF5 group"
    ):
        h5.read_output_struct(pth, struct="non-existent")


def test_read_inputs_from_filepath(tmp_path: Path):
    pth = tmp_path / "tmp.h5"
    inputs = InputParameters(random_seed=0)

    with h5py.File(pth, "w") as fl:
        h5._write_inputs_to_group(inputs, fl)

    new = h5.read_inputs(pth)  # directly read from path.
    assert new == inputs


def test_read_inputs_badfile(ic: InitialConditions, perturbed_field, tmp_path: Path):
    pth = tmp_path / "ics.h5"
    h5.write_output_to_hdf5(ic, pth)
    h5.write_output_to_hdf5(perturbed_field, pth, mode="a")

    with pytest.raises(h5.HDF5FileStructureError, match="Multiple sub-groups found"):
        h5.read_inputs(pth)


def test_read_inputs_version_warning(tmp_path: Path):
    pth = tmp_path / "tmp.h5"
    inputs = InputParameters(random_seed=0)

    with h5py.File(pth, "w") as fl:
        h5._write_inputs_to_group(inputs, fl)
        fl["InputParameters"].attrs["21cmFAST-version"] = "5.0.0"  # larger than reality

    with pytest.warns(UserWarning, match="File created with a newer version"):
        h5.read_inputs(pth)


def test_read_outputs_version_warning(ic: InitialConditions, tmp_path: Path):
    pth = tmp_path / "ics.h5"
    h5.write_output_to_hdf5(ic, pth)

    # Mock the version as newer than it really is.
    with h5py.File(pth, "a") as fl:
        fl["InitialConditions"]["OutputFields"].attrs["21cmFAST-version"] = "5.0.0"

    with pytest.warns(UserWarning, match="File created with a newer version"):
        h5.read_output_struct(pth)


def test_read_output_struct_missing_arrays(ic: InitialConditions, tmp_path: Path):
    pth = tmp_path / "ics.h5"
    h5.write_output_to_hdf5(ic, pth)

    # Artificially remove an array
    with h5py.File(pth, "a") as fl:
        del fl["InitialConditions"]["OutputFields"]["lowres_density"]

    with pytest.raises(
        h5.HDF5FileStructureError, match="Required Array lowres_density not found"
    ):
        h5.read_output_struct(pth)


def test_read_output_struct_badshape_arrays(ic: InitialConditions, tmp_path: Path):
    pth = tmp_path / "ics.h5"
    h5.write_output_to_hdf5(ic, pth)

    # Artificially remove an array
    with h5py.File(pth, "a") as fl:
        dens = fl["InitialConditions"]["OutputFields"]["lowres_density"][:10, :11, :12]
        del fl["InitialConditions"]["OutputFields"]["lowres_density"]
        fl["InitialConditions"]["OutputFields"]["lowres_density"] = dens

    with pytest.raises(
        h5.HDF5FileStructureError, match="Array lowres_density has shape"
    ):
        h5.read_output_struct(pth)


def test_read_inputs_non_existent_version(tmp_path: Path):
    pth = tmp_path / "tmp.h5"
    inputs = InputParameters(random_seed=0)

    with h5py.File(pth, "w") as fl:
        h5._write_inputs_to_group(inputs, fl)
        del fl["InputParameters"].attrs["21cmFAST-version"]

    with pytest.raises(NotImplementedError, match=f"The file {pth} is not a valid"):
        h5.read_inputs(pth)


def test_read_output_struct_no_version(ic: InitialConditions, tmp_path: Path):
    pth = tmp_path / "ics.h5"
    h5.write_output_to_hdf5(ic, pth)

    # Artificially remove an array
    with h5py.File(pth, "a") as fl:
        del fl["InitialConditions"]["OutputFields"].attrs["21cmFAST-version"]

    with pytest.raises(NotImplementedError, match=f"The file {pth} is not a valid"):
        h5.read_output_struct(pth)
