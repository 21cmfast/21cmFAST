"""Various tests of the compute_initial_conditions() function and InitialConditions class."""

from multiprocessing import cpu_count

import numpy as np
import pytest

import py21cmfast as p21c


@pytest.fixture(scope="module")
def ic_hires(default_input_struct) -> p21c.InitialConditions:
    return p21c.compute_initial_conditions(
        inputs=default_input_struct.evolve_input_structs(PERTURB_ON_HIGH_RES=True),
    )


@pytest.fixture(scope="module")
def ic_from_ic(default_input_struct, ic) -> p21c.InitialConditions:
    """Make initial conditions, given the hires density of an initial condition instance."""
    return p21c.compute_initial_conditions(
        inputs=default_input_struct, initial_density=ic.hires_density.value
    )


@pytest.fixture(scope="module")
def single_pxl_array_mean_zero(default_input_struct) -> np.ndarray:
    """Make a single pixel array, with mean zero."""
    dim = default_input_struct.simulation_options.DIM
    array = -np.ones((dim,) * 3)
    array[dim // 2, dim // 2, dim // 2] = array[0, 0, 0] - array.sum()
    return array


@pytest.fixture(scope="module")
def ic_from_array(
    default_input_struct, single_pxl_array_mean_zero
) -> p21c.InitialConditions:
    """Make initial conditions, given the hires density array of a single pixel."""
    return p21c.compute_initial_conditions(
        inputs=default_input_struct, initial_density=single_pxl_array_mean_zero
    )


def test_box_shape(ic_hires: p21c.InitialConditions, ic: p21c.InitialConditions):
    """Test basic properties of the InitialConditions struct."""
    shape = (35, 35, 35)
    hires_shape = tuple(2 * s for s in shape)
    # test common fields
    for box in (ic, ic_hires):
        assert box.lowres_density.shape == shape
        assert box.hires_density.shape == hires_shape
        assert box.hires_vx_2LPT.shape == hires_shape
        assert box.hires_vy_2LPT.shape == hires_shape
        assert box.hires_vz_2LPT.shape == hires_shape
        assert box.lowres_vcb is None
        assert box.cosmo_params == p21c.CosmoParams()

    # test hires only fields

    assert ic_hires.hires_vx.shape == hires_shape
    assert ic_hires.hires_vy.shape == hires_shape
    assert ic_hires.hires_vz.shape == hires_shape

    assert ic_hires.lowres_vx is None
    assert ic_hires.lowres_vy is None
    assert ic_hires.lowres_vz is None

    assert ic_hires.lowres_vx_2LPT is None
    assert ic_hires.lowres_vy_2LPT is None
    assert ic_hires.lowres_vz_2LPT is None

    # test lowres only fields

    assert ic.lowres_vx.shape == shape
    assert ic.lowres_vy.shape == shape
    assert ic.lowres_vz.shape == shape

    assert ic.lowres_vx_2LPT.shape == shape
    assert ic.lowres_vy_2LPT.shape == shape
    assert ic.lowres_vz_2LPT.shape == shape

    assert ic.hires_vx is None
    assert ic.hires_vy is None
    assert ic.hires_vz is None


def test_modified_cosmo(
    ic: p21c.InitialConditions, default_input_struct: p21c.InputParameters, cache
):
    """Test using a modified cosmology."""
    inputs = default_input_struct.evolve_input_structs(SIGMA_8=0.9)
    ic2 = p21c.compute_initial_conditions(inputs=inputs, cache=cache)

    assert ic2.cosmo_params != ic.cosmo_params
    assert ic2.cosmo_params == inputs.cosmo_params
    assert ic2.cosmo_params.SIGMA_8 == inputs.cosmo_params.SIGMA_8


def test_transfer_function(
    ic: p21c.InitialConditions, default_input_struct: p21c.InputParameters, cache
):
    """Test using a modified transfer function."""
    inputs = default_input_struct.evolve_input_structs(POWER_SPECTRUM="CLASS")
    ic2 = p21c.compute_initial_conditions(inputs=inputs, cache=cache)
    hrd2 = ic2.hires_density.value
    hrd = ic.hires_density.value

    rmsnew = np.sqrt(np.mean(hrd2**2))
    rmsdelta = np.sqrt(np.mean((hrd2 - hrd) ** 2))
    assert rmsdelta < rmsnew
    assert rmsnew > 0.0
    assert not np.allclose(hrd2, hrd)


def test_relvels():
    """Test for relative velocity initial conditions."""
    inputs = p21c.InputParameters(random_seed=1).evolve_input_structs(
        HII_DIM=100,
        DIM=300,
        BOX_LEN=300,
        POWER_SPECTRUM="CLASS",
        USE_RELATIVE_VELOCITIES=True,
        N_THREADS=cpu_count(),  # To make this one a bit faster.
    )
    ic = p21c.compute_initial_conditions(inputs=inputs)

    vcbrms_lowres = np.sqrt(np.mean(ic.lowres_vcb.value**2))
    vcbavg_lowres = np.mean(ic.lowres_vcb.value)

    # we test the lowres box
    # rms should be about 30 km/s for LCDM, so we check it is finite and not far off
    # the average should be 0.92*vrms, since it follows a maxwell boltzmann
    assert vcbrms_lowres > 20.0
    assert vcbrms_lowres < 40.0
    assert vcbavg_lowres < 0.97 * vcbrms_lowres
    assert vcbavg_lowres > 0.88 * vcbrms_lowres


@pytest.mark.parametrize(
    "name",
    [
        "lowres_density",
        "hires_density",
        "lowres_vx",
        "lowres_vy",
        "lowres_vz",
        "lowres_vx_2LPT",
        "lowres_vy_2LPT",
        "lowres_vz_2LPT",
    ],
)
def test_initial_density_array(
    ic: p21c.InitialConditions,
    ic_from_ic: p21c.InitialConditions,
    ic_from_array: p21c.InitialConditions,
    single_pxl_array_mean_zero: np.ndarray,
    name: str,
):
    """Test the functionality with the initial_density argument."""
    # Test that the hires_density arrays are exactly the same (by definition)
    assert np.all(ic_from_ic.hires_density.value == ic.hires_density.value)

    # Test that the other arrays are close (numerical differences exist due to FFT-IFFT)
    np.testing.assert_allclose(
        getattr(ic, name).value, getattr(ic_from_ic, name).value, atol=1e-5, rtol=0.0
    )

    # Test the array we use actually has mean zero
    assert single_pxl_array_mean_zero.mean() == 0.0

    # Test that the hires_density array is exactly our single pixel array input
    assert np.all(ic_from_array.hires_density.value == single_pxl_array_mean_zero)

    # Test that the arrays are different between the original ic and the ic we got from array
    assert not np.allclose(
        getattr(ic, name).value, getattr(ic_from_array, name).value, atol=1e-5, rtol=0.0
    )


def test_bad_initial_density_array(
    default_input_struct: p21c.InitialConditions, single_pxl_array_mean_zero: np.ndarray
):
    """Test bad/weird initial_density array."""
    # Run initial conditions with hires density box that has non-zero mean, just to throw the relevant warning
    ic_non_zero = p21c.compute_initial_conditions(
        inputs=default_input_struct,
        initial_density=np.ones_like(single_pxl_array_mean_zero),
    )
    assert isinstance(ic_non_zero, p21c.InitialConditions)

    with pytest.raises(
        ValueError,
        match="The shape of your high resolution initial_density is not consistent with inputs!",
    ):
        p21c.compute_initial_conditions(
            inputs=default_input_struct,
            initial_density=np.zeros((single_pxl_array_mean_zero.shape[0] * 2,) * 3),
        )
