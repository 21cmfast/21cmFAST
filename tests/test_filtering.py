import pytest

import matplotlib as mpl
import numpy as np

from py21cmfast import (
    AstroParams,
    CosmoParams,
    FlagOptions,
    PerturbHaloField,
    UserParams,
    global_params,
)
from py21cmfast.c_21cmfast import ffi, lib

from . import produce_integration_test_data as prd

RELATIVE_TOLERANCE = 1e-4

options_filter = [0, 1, 2, 3, 4]  # cell densities to draw samples from
R_PARAM_LIST = [1.5, 5, 10, 30, 60]


@pytest.mark.parametrize("R", R_PARAM_LIST)
@pytest.mark.parametrize("filter_flag", options_filter)
def test_filters(filter_flag, R, plt):
    opts = prd.get_all_options(redshift=10.0)

    up = UserParams(opts["user_params"])
    cp = CosmoParams(opts["cosmo_params"])
    ap = AstroParams(opts["astro_params"])
    fo = FlagOptions(opts["flag_options"])

    # testing a single pixel source
    input_box_centre = np.zeros((up.HII_DIM,) * 3, dtype="f4")
    input_box_centre[up.HII_DIM // 2, up.HII_DIM // 2, up.HII_DIM // 2] = 1.0

    # testing a uniform grid
    input_box_uniform = np.full((up.HII_DIM,) * 3, 1.0, dtype="f4")

    output_box_centre = np.zeros((up.HII_DIM,) * 3, dtype="f8")
    output_box_uniform = np.zeros((up.HII_DIM,) * 3, dtype="f8")

    # use MFP=20 for the exp filter, use a 3 cell shell for the annular filter
    R_param_list = [0.0, 0.0, 0.0, 20, max(R - 3 * (up.BOX_LEN / up.HII_DIM), 0)]

    lib.test_filter(
        up(),
        cp(),
        ap(),
        fo(),
        ffi.cast("float *", input_box_centre.ctypes.data),
        R,
        R_param_list[filter_flag],
        filter_flag,
        ffi.cast("double *", output_box_centre.ctypes.data),
    )

    lib.test_filter(
        up(),
        cp(),
        ap(),
        fo(),
        ffi.cast("float *", input_box_uniform.ctypes.data),
        R,
        R_param_list[filter_flag],
        filter_flag,
        ffi.cast("double *", output_box_uniform.ctypes.data),
    )

    r_from_centre = np.linalg.norm(
        np.mgrid[0 : up.HII_DIM, 0 : up.HII_DIM, 0 : up.HII_DIM]
        - np.array(
            [
                up.HII_DIM // 2,
                up.HII_DIM // 2,
                up.HII_DIM // 2,
            ]
        )[:, None, None, None],
        axis=0,
    )
    # single pixel boxes have a specific shape based on the filter
    R_ratio = r_from_centre / R
    if filter_flag == 0.0:
        # output is uniform sphere around the centre point
        exp_vol = 4 / 3 * np.pi * R**3
        expected_output = 1 / exp_vol * (R_ratio < 1)
    elif filter_flag == 1.0:
        # output is sinc function
        expected_output = np.sin(R_ratio) / R_ratio
    elif filter_flag == 2.0:
        # output is Gaussian
        expected_output = 1 / np.sqrt(2 * np.pi * R) * np.exp(-(R_ratio**2) / 2)
    elif filter_flag == 3.0:
        exp_vol = 4 / 3 * np.pi * R**3
        expected_output = 1 / exp_vol * (R_ratio < 1) * np.exp(-R_ratio)
    elif filter_flag == 4.0:
        R_i = R_param_list[4]
        exp_vol = 4 / 3 * np.pi * (R**3 - R_i**3)
        expected_output = 1 / exp_vol * (R_ratio < 1) * (R_i / R > 1)

    # uniform boxes should come out uniform aside from normalisation
    norm_factor = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    if plt == mpl.pyplot:
        filter_plot_symmetric(
            [input_box_centre, input_box_uniform],
            [output_box_centre, output_box_uniform],
            [expected_output, input_box_uniform / norm_factor[filter_flag]],
            up.HII_DIM,
            plt,
        )

    np.testing.assert_allclose(
        input_box_centre, expected_output, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        input_box_uniform,
        norm_factor[filter_flag] * output_box_uniform,
        rtol=RELATIVE_TOLERANCE,
    )


# since the filters are symmetric I'm doing an R vs value plot instead of imshowing slices
def filter_plot_symmetric(inputs, outputs, truths, dimension, plt):
    if not (len(inputs) == len(truths) == len(outputs)):
        raise ValueError(
            f"inputs {len(inputs)}, outputs {len(outputs)} and truths {len(truths)}"
            "must have the same length."
        )

    n_plots = len(inputs)
    fig, axs = plt.subplots(
        n_plots, 3, figsize=(16.0, 9.0 * n_plots / 3.0), layout="constrained"
    )
    # fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.0,
    #                         wspace=0.0)

    for idx, (i, o, t) in enumerate(zip(inputs, outputs, truths)):
        r = np.linalg.norm(np.mgrid[0:dimension, 0:dimension, 0:dimension], axis=0)
        axs[idx, 0].scatter(r, i, s=1)
        axs[idx, 1].scatter(r, o, s=1)
        axs[idx, 2].scatter(r, t, s=1)
