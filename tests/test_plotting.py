"""Testing plots is kind of hard, but we just check that it runs through without crashing."""

import pytest

import py21cmfast as p21c
from py21cmfast import plotting


def test_coeval_sliceplot(ic: p21c.InitialConditions):
    fig, ax = plotting.coeval_sliceplot(ic)

    assert ax.xaxis.get_label().get_text() == "x-axis [Mpc]"
    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"

    with pytest.raises(
        ValueError, match="slice_axis should be between -1 and 2"
    ):  # bad slice axis
        plotting.coeval_sliceplot(ic, slice_axis=-2)

    with pytest.raises(
        IndexError, match="slice_index is too large"
    ):  # tring to plot slice that doesn't exist
        plotting.coeval_sliceplot(ic, slice_index=50)

    fig2, ax2 = plotting.coeval_sliceplot(ic, fig=fig, ax=ax)

    assert fig2 is fig
    assert ax2 is ax

    fig2, ax2 = plotting.coeval_sliceplot(ic, fig=fig)

    assert fig2 is fig
    assert ax2 is ax

    fig2, ax2 = plotting.coeval_sliceplot(ic, ax=ax)

    assert fig2 is fig
    assert ax2 is ax

    fig, ax = plotting.coeval_sliceplot(
        ic, kind="hires_density", slice_index=50, slice_axis=1
    )
    assert ax.xaxis.get_label().get_text() == "x-axis [Mpc]"
    assert ax.yaxis.get_label().get_text() == "z-axis [Mpc]"


def test_lightcone_sliceplot_default(lc: p21c.LightCone):
    fig, ax = plotting.lightcone_sliceplot(lc)

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "Redshift"


def test_lightcone_sliceplot_vertical(lc: p21c.LightCone):
    fig, ax = plotting.lightcone_sliceplot(lc, vertical=True)

    assert ax.yaxis.get_label().get_text() == "Redshift"
    assert ax.xaxis.get_label().get_text() == "y-axis [Mpc]"


def test_lc_sliceplot_freq(lc: p21c.LightCone):
    fig, ax = plotting.lightcone_sliceplot(lc, zticks="frequency")

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "Frequency [MHz]"


def test_lc_sliceplot_cdist(lc: p21c.LightCone):
    fig, ax = plotting.lightcone_sliceplot(lc, zticks="comoving_distance")

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "Comoving Distance [Mpc]"

    xlim = ax.get_xticks()
    assert xlim.min() >= 0
    assert xlim.max() <= lc.lightcone_dimensions[-1]


def test_lc_sliceplot_sliceax(lc: p21c.LightCone):
    fig, ax = plotting.lightcone_sliceplot(lc, slice_axis=2)

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "x-axis [Mpc]"


def test_global_plot(lc: p21c.LightCone):
    fig, ax = plotting.plot_global_history(lc)

    assert ax.xaxis.get_label().get_text() == "Redshift"
