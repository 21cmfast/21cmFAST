"""Test filtering of the density field."""

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib.colors import Normalize
from py21cmfast.c_21cmfast import ffi, lib
from scipy.stats import binned_statistic as binstat

from py21cmfast.wrapper.cfuncs import broadcast_input_struct

from . import produce_integration_test_data as prd
from .test_c_interpolation_tables import print_failure_stats

options_filter = [0, 1, 2, 3, 4]  # cell densities to draw samples from
R_PARAM_LIST = [
    1.5,
    5,
    10,
    20,
]  # default test HII_DIM = 50, we want max R < BOX_LEN*HII_DIM/3


# NOTE: These don't directly test against the expected FFT of these filters applied
#   to a central cell, but the continuous FT filters applied to a delta function.
#   this makes it a test of both the filter construction, as well as the aliasing.
# NOTE: These do not include the periodic boundary conditions, so face issues with R ~> HII_DIM/2.
def get_expected_output_centre(r_in, R_filter, R_param, filter_flag):
    # single pixel boxes have a specific shape based on the filter
    R_ratio = r_in / R_filter
    if filter_flag == 0:
        # output is uniform sphere around the centre point
        exp_vol = 4 / 3 * np.pi * R_filter**3
        return (R_ratio < 1) / exp_vol
    elif filter_flag == 1:
        # output is the tophat FFT
        R_ratio *= 1 / 0.413566994  # == this is the 2*pi*k factor for equating volumes
        result = (np.sin(R_ratio) - R_ratio * np.cos(R_ratio)) / (
            2 * np.pi**2 * r_in**3
        )
        result[r_in == 0] = (
            1 / 6 / np.pi**2 * (0.413566994 * R_filter) ** 3
        )  # r->0 limit
        return result
    elif filter_flag == 2:
        # output is Gaussian
        const = (0.643 * R_filter) ** 2
        exp_vol = (2 * np.pi * const) ** (3.0 / 2.0)
        return np.exp(-((r_in) ** 2 / const / 2)) / exp_vol
    elif filter_flag == 3:
        # output is sphere with exponential damping
        exp_vol = 4 / 3 * np.pi * R_filter**3
        return (R_ratio < 1) * np.exp(-r_in / R_param) / exp_vol
    elif filter_flag == 4:
        # output is spherical shell
        exp_vol = 4 / 3 * np.pi * (R_filter**3 - R_param**3)
        return (R_ratio < 1) * (R_param <= r_in) / exp_vol


# return binned quantities
def get_binned_stats(x_arr, y_arr, bins, stats):
    x_in = x_arr.flatten()
    y_in = y_arr.flatten()
    result = {}

    statistic_dict = {
        "pc1u": lambda x: np.percentile(x, 84),
        "pc1l": lambda x: np.percentile(x, 16),
        "pc2u": lambda x: np.percentile(x, 97.5),
        "pc2l": lambda x: np.percentile(x, 2.5),
        # used to mark percentiles in errorbar plots, since these cannot be negative
        "err1u": lambda x: np.maximum(np.percentile(x, 84) - np.mean(x), 0),
        "err2u": lambda x: np.maximum(np.percentile(x, 97.5) - np.mean(x), 0),
        "err1l": lambda x: np.maximum(np.mean(x) - np.percentile(x, 16), 0),
        "err2l": lambda x: np.maximum(np.mean(x) - np.percentile(x, 2.5), 0),
        "errmin": lambda x: np.mean(x) - np.amin(x),
        "errmax": lambda x: np.amax(x) - np.mean(x),
    }

    for stat in stats:
        spstatkey = statistic_dict.get(stat, stat)
        result[stat], _, _ = binstat(x_in, y_in, bins=bins, statistic=spstatkey)

    return result


@pytest.mark.parametrize("R", R_PARAM_LIST)
@pytest.mark.parametrize("filter_flag", options_filter)
def test_filters(filter_flag, R, plt):
    opts = prd.get_all_options_struct(redshift=10.0)

    inputs = opts["inputs"]
    up = inputs.simulation_options

    # testing a single pixel source
    input_box_centre = np.zeros((up.HII_DIM,) * 3, dtype="f4")
    input_box_centre[up.HII_DIM // 2, up.HII_DIM // 2, up.HII_DIM // 2] = 1.0
    output_box_centre = np.zeros((up.HII_DIM,) * 3, dtype="f8")
    # use MFP=20 for the exp filter, use a 4 cell shell for the annular filter
    if filter_flag == 3:
        R_param = 20
    elif filter_flag == 4:
        R_param = max(R - 4 * (up.BOX_LEN / up.HII_DIM), 0)
    else:
        R_param = 0

    broadcast_input_struct(inputs)
    lib.test_filter(
        ffi.cast("float *", input_box_centre.ctypes.data),
        R,
        R_param,
        filter_flag,
        ffi.cast("double *", output_box_centre.ctypes.data),
    )

    # expected outputs given in cell units
    R_cells = R / up.BOX_LEN * up.HII_DIM
    Rp_cells = R_param / up.BOX_LEN * up.HII_DIM
    r_from_centre = np.linalg.norm(
        np.mgrid[0 : up.HII_DIM, 0 : up.HII_DIM, 0 : up.HII_DIM]
        - np.array([up.HII_DIM // 2, up.HII_DIM // 2, up.HII_DIM // 2])[
            :, None, None, None
        ],
        axis=0,
    )
    # prevent divide by zero in the central cell
    r_from_centre[up.HII_DIM // 2, up.HII_DIM // 2, up.HII_DIM // 2] = 1e-6

    exp_output_centre = get_expected_output_centre(
        r_from_centre, R_cells, Rp_cells, filter_flag
    )

    # these are very wide tolerances, just to make sure there aren't
    # cells 100x the expected values
    abs_tol_pixel = exp_output_centre.max() * 0.8
    rel_tol_pixel = 0
    # we take bins of 2 pixels to smooth over sharp edged filters
    r_bins = np.arange(0, int(up.HII_DIM / 2 * np.sqrt(3)), 2)
    r_cen = (r_bins[1:] + r_bins[:-1]) / 2

    binned_truth_centre = get_binned_stats(
        r_from_centre,
        exp_output_centre,
        r_bins,
        stats=["mean", "errmin", "errmax", "err1l", "err1u"],
    )
    binned_truth_centre = binned_truth_centre["mean"]

    stats_o = get_binned_stats(
        r_from_centre,
        output_box_centre,
        r_bins,
        stats=["mean", "errmin", "errmax", "err1l", "err1u"],
    )

    if plt == mpl.pyplot:
        filter_plot(
            inputs=[input_box_centre],
            outputs=[output_box_centre],
            binned_truths=[binned_truth_centre],
            binned_stats=[stats_o],
            truths=[exp_output_centre],
            r_bins=r_bins,
            r_grid=r_from_centre,
            slice_index=up.HII_DIM // 2,
            slice_axis=0,
            abs_tol=abs_tol_pixel,
            rel_tol=rel_tol_pixel,
            plt=plt,
        )

    # All filters should be normalised aside from the exp filter
    if filter_flag == 3:
        # ratio of exponential and sphere volume integrals
        R_q = R_param / R
        norm_factor = 6 * R_q**3 - np.exp(-1 / R_q) * (
            6 * R_q**3 + 6 * R_q**2 + 3 * R_q
        )
    else:
        norm_factor = 1
    # firstly, make sure the filters are normalised
    np.testing.assert_allclose(
        input_box_centre.sum() * norm_factor, output_box_centre.sum(), atol=1e-4
    )

    # secondly, make sure binned results are reasonable
    rel_tol_bins = 1e-1
    abs_tol_bins = exp_output_centre.max() * 1e-1
    print_failure_stats(
        stats_o["mean"],
        binned_truth_centre,
        [r_cen],
        abs_tol=abs_tol_bins,
        rel_tol=rel_tol_bins,
        name="bins",
    )
    np.testing.assert_allclose(
        binned_truth_centre, stats_o["mean"], atol=abs_tol_bins, rtol=rel_tol_bins
    )

    # lastly, make sure no pixels are way out of line.
    # this has a wide tolerance due to significant aliasing
    print_failure_stats(
        output_box_centre,
        exp_output_centre,
        [r_from_centre],
        abs_tol=abs_tol_pixel,
        rel_tol=rel_tol_pixel,
        name="pixel",
    )
    np.testing.assert_allclose(
        output_box_centre, exp_output_centre, rtol=rel_tol_pixel, atol=abs_tol_pixel
    )


# since the filters are symmetric I'm doing an R vs value plot instead of imshowing slices
def filter_plot(
    inputs,
    outputs,
    binned_truths,
    binned_stats,
    truths,
    r_bins,
    r_grid,
    slice_index,
    slice_axis,
    abs_tol,
    rel_tol,
    plt,
):
    if not (len(inputs) == len(binned_truths) == len(outputs)):
        raise ValueError(
            f"inputs {len(inputs)}, outputs {len(outputs)} and truths {len(binned_truths)}"
            "must have the same length."
        )

    n_plots = len(inputs)
    fig, axs = plt.subplots(
        n_plots,
        4,
        figsize=(16.0, 12.0 * n_plots / 4.0),
        layout="constrained",
        squeeze=False,
    )
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0.0, wspace=0.0)

    r_cen = (r_bins[1:] + r_bins[:-1]) / 2

    axs[0, 0].set_title("Output")
    axs[0, 1].set_title("Expected")
    axs[0, 2].set_title("Radii")
    axs[0, 3].set_title("Pixels")
    for idx, (_i, o, bo, t, tt) in enumerate(
        zip(inputs, outputs, binned_stats, binned_truths, truths, strict=True)
    ):
        axs[idx, 0].pcolormesh(
            o.take(indices=slice_index, axis=slice_axis),
            norm=Normalize(vmin=0, vmax=o.max()),
        )
        axs[idx, 1].pcolormesh(
            tt.take(indices=slice_index, axis=slice_axis),
            norm=Normalize(vmin=0, vmax=o.max()),
        )

        lns = []
        lns.append(
            axs[idx, 2].errorbar(
                r_cen,
                bo["mean"],
                markerfacecolor="b",
                elinewidth=1,
                capsize=3,
                markersize=5,
                marker="o",
                color="b",
                yerr=[bo["errmin"], bo["errmax"]],
                label="filtered grid",
                zorder=2,
            )
        )

        lns.append(
            axs[idx, 2].plot(r_cen, t, "m:", linewidth=2, label="Expected", zorder=3)[0]
        )
        axs[idx, 2].grid()
        axs[idx, 2].set_xlabel("dist from centre")
        axs[idx, 2].set_ylabel("cell value")
        axs[idx, 2].legend()

        err_base = np.linspace(tt.min(), tt.max(), num=100)
        err_max = err_base + (abs_tol + np.fabs(err_base) * rel_tol)
        err_min = err_base - (abs_tol + np.fabs(err_base) * rel_tol)
        axs[idx, 3].plot(err_base, err_max, "k:")
        axs[idx, 3].plot(err_base, err_min, "k:")
        axs[idx, 3].plot(err_base, err_base, "k--")

        axs[idx, 3].scatter(tt, o, s=1, alpha=0.5, rasterized=True)
        axs[idx, 3].grid()
        axs[idx, 3].set_xlabel("expected cell value")
        axs[idx, 3].set_ylabel("filtered cell value")
