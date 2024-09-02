import pytest

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from scipy.stats import binned_statistic as binstat

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

# tolerance for aliasing errors
RELATIVE_TOLERANCE = 1e-1

options_filter = [0, 1, 2, 3, 4]  # cell densities to draw samples from
R_PARAM_LIST = [1.5, 5, 10, 30, 60]


def get_expected_output_centre(r_in, R_filter, R_param, filter_flag):
    # single pixel boxes have a specific shape based on the filter
    R_ratio = r_in / R_filter
    if filter_flag == 0:
        # output is uniform sphere around the centre point
        exp_vol = 4 / 3 * np.pi * R_filter**3
        return (R_ratio < 1) / exp_vol
    elif filter_flag == 1:
        # output is sinc function
        R_ratio /= 0.413566994
        R_filter /= 0.413566994
        exp_vol = R_filter**3 / 3.0
        return (np.sin(R_ratio) / R_ratio) / exp_vol
    elif filter_flag == 2:
        # output is Gaussian
        R_ratio /= 0.643
        R_filter /= 0.643
        exp_vol = (2 * np.pi) ** (3 / 2) * R_filter**3
        return np.exp(-(R_ratio**2) / 2) / exp_vol
    elif filter_flag == 3:
        # output is sphere with exponential damping
        exp_vol = 4 / 3 * np.pi * R_filter**3
        return (R_ratio < 1) * np.exp(-r_in / R_param) / exp_vol
    elif filter_flag == 4:
        # output is spherical shell
        exp_vol = 4 / 3 * np.pi * (R_filter**3 - R_param**3)
        return (R_ratio < 1) * (R_param <= r_in) / exp_vol


def get_expected_output_uniform(in_box, R_filter, R_param, filter_flag):
    # uniform boxes should come out uniform, the exp filter will be
    if filter_flag == 3:
        norm_factor = (
            R_param - R_param * np.exp(-R_filter / R_param)
        ) / R_filter  # TODO: this is wrong, change it
    else:
        norm_factor = 1
    return in_box * norm_factor


# return binned mean & 1-2 sigma quantiles
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
    }

    for stat in stats:
        spstatkey = statistic_dict[stat] if stat in statistic_dict.keys() else stat
        result[stat], _, _ = binstat(x_in, y_in, bins=bins, statistic=spstatkey)

    return result


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
    if filter_flag == 3:
        R_param = 20
    elif filter_flag == 4:
        R_param = max(R - 3 * (up.BOX_LEN / up.HII_DIM), 0)
    else:
        R_param = 0

    lib.test_filter(
        up(),
        cp(),
        ap(),
        fo(),
        ffi.cast("float *", input_box_centre.ctypes.data),
        R,
        R_param,
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
        R_param,
        filter_flag,
        ffi.cast("double *", output_box_uniform.ctypes.data),
    )

    R_cells = R / up.BOX_LEN * up.HII_DIM
    Rp_cells = R_param / up.BOX_LEN * up.HII_DIM
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
    exp_output_centre = get_expected_output_centre(
        r_from_centre, R_cells, Rp_cells, filter_flag
    )
    exp_output_uniform = get_expected_output_uniform(
        input_box_uniform, R_cells, Rp_cells, filter_flag
    )

    if plt == mpl.pyplot:
        r_bins = np.linspace(0, (up.HII_DIM / 2 * np.sqrt(3)), num=32)
        uniform_bin = np.ones_like(r_bins)
        binned_truth_centre = get_expected_output_centre(
            r_bins, R_cells, Rp_cells, filter_flag
        )
        binned_truth_uniform = get_expected_output_uniform(
            uniform_bin, R_cells, Rp_cells, filter_flag
        )
        filter_plot(
            inputs=[input_box_centre, input_box_uniform],
            outputs=[output_box_centre, output_box_uniform],
            binned_truths=[binned_truth_centre, binned_truth_uniform],
            truths=[exp_output_centre, exp_output_uniform],
            r_bins=r_bins,
            r_grid=r_from_centre,
            slice_index=up.HII_DIM // 2,
            slice_axis=0,
            plt=plt,
        )

    np.testing.assert_allclose(
        input_box_centre, exp_output_centre, rtol=RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        input_box_uniform,
        exp_output_uniform,
        rtol=RELATIVE_TOLERANCE,
    )


# since the filters are symmetric I'm doing an R vs value plot instead of imshowing slices
def filter_plot(
    inputs, outputs, binned_truths, truths, r_bins, r_grid, slice_index, slice_axis, plt
):
    if not (len(inputs) == len(binned_truths) == len(outputs)):
        raise ValueError(
            f"inputs {len(inputs)}, outputs {len(outputs)} and truths {len(binned_truths)}"
            "must have the same length."
        )

    n_plots = len(inputs)
    fig, axs = plt.subplots(
        n_plots, 4, figsize=(16.0, 12.0 * n_plots / 4.0), layout="constrained"
    )
    fig.get_layout_engine().set(w_pad=2 / 72, h_pad=2 / 72, hspace=0.0, wspace=0.0)

    r_cen = (r_bins[1:] + r_bins[:-1]) / 2

    axs[0, 0].set_title("Input")
    axs[0, 1].set_title("output")
    axs[0, 2].set_title("Expected")
    axs[0, 3].set_title("Radii")
    for idx, (i, o, t, tt) in enumerate(zip(inputs, outputs, binned_truths, truths)):
        axs[idx, 0].pcolormesh(
            i.take(indices=slice_index, axis=slice_axis),
            norm=Normalize(vmin=0, vmax=o.max()),
        )
        axs[idx, 1].pcolormesh(
            o.take(indices=slice_index, axis=slice_axis),
            norm=Normalize(vmin=0, vmax=o.max()),
        )
        axs[idx, 2].pcolormesh(
            tt.take(indices=slice_index, axis=slice_axis),
            norm=Normalize(vmin=0, vmax=o.max()),
        )

        stats_o = get_binned_stats(r_grid, o, r_bins, stats=["mean", "err1u", "err1l"])
        axs[idx, 3].errorbar(
            r_cen,
            stats_o["mean"],
            markerfacecolor="b",
            elinewidth=1,
            capsize=3,
            markersize=5,
            marker="o",
            color="b",
            yerr=[stats_o["err1l"], stats_o["err1u"]],
            label="filtered grid",
        )
        axs[idx, 3].plot(r_bins, t, "m:", label="Expected")
        axs[idx, 3].grid()
        axs[idx, 3].set_xlabel("dist from centre")
        axs[idx, 3].set_ylabel("cell value")
