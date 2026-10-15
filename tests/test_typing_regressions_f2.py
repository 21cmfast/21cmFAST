"""Regression tests for bugs found while adding type annotations to cli and plotting."""

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pytest

import py21cmfast as p21c
from py21cmfast import cli, plotting


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _non_current_axes():
    """Create a figure/axes pair that is *not* the current figure."""
    fig, ax = plt.subplots()
    plt.figure()  # make a different figure current
    assert plt.gcf() is not fig
    return fig, ax


def test_coeval_sliceplot_on_non_current_axes(ic: p21c.InitialConditions):
    """Passing only ``ax`` should use ``ax``'s figure, not the current figure."""
    fig, ax = _non_current_axes()
    fig2, ax2 = plotting.coeval_sliceplot(ic, ax=ax, cbar_label="label")

    assert ax2 is ax
    assert fig2 is fig
    assert ax.images[-1].colorbar is not None
    assert ax.images[-1].colorbar.ax.get_ylabel() == "label"


def test_coeval_sliceplot_on_non_current_figure(ic: p21c.InitialConditions):
    """Passing only ``fig`` should plot on that figure, not the current figure."""
    fig, ax = _non_current_axes()
    fig2, ax2 = plotting.coeval_sliceplot(ic, fig=fig)

    assert fig2 is fig
    assert ax2 is ax
    assert len(ax.images) == 1


def test_lightcone_sliceplot_with_only_ax(lc: p21c.LightCone):
    """Passing a fresh ``ax`` (without ``fig``) used to raise an AttributeError."""
    fig, ax = plt.subplots()
    fig2, ax2 = plotting.lightcone_sliceplot(lc, ax=ax)

    assert fig2 is fig
    assert ax2 is ax
    assert len(ax.images) == 1


def test_lightcone_sliceplot_with_only_fig(lc: p21c.LightCone):
    """Passing ``fig`` (without ``ax``) used to pass a *list* of axes along."""
    fig, ax = plt.subplots()
    fig2, ax2 = plotting.lightcone_sliceplot(lc, fig=fig)

    assert fig2 is fig
    assert ax2 is ax
    assert len(ax.images) == 1


def test_lightcone_difference_cmap(lc: p21c.LightCone):
    """A difference plot uses "bwr" by default, but a given cmap used to be ignored."""
    _, ax = plotting.lightcone_sliceplot(lc, lightcone2=lc)
    assert ax.images[-1].get_cmap().name == "bwr"

    _, ax = plotting.lightcone_sliceplot(lc, lightcone2=lc, cmap="viridis")
    assert ax.images[-1].get_cmap().name == "viridis"


def test_lightcone_difference_with_z_range(lc: p21c.LightCone):
    """The z-range selection used to be applied twice for difference plots."""
    zs = lc.lightcone_redshifts
    z_range = (zs[len(zs) // 4], zs[3 * len(zs) // 4])

    _, ax = plotting.lightcone_sliceplot(lc, z_range=z_range)
    _, ax_diff = plotting.lightcone_sliceplot(lc, lightcone2=lc, z_range=z_range)

    single = ax.images[-1].get_array()
    diff = ax_diff.images[-1].get_array()
    assert diff.shape == single.shape
    np.testing.assert_allclose(diff, 0)


def test_lightcone_sliceplot_dimensionless_zticks(lc: p21c.LightCone):
    """Using a cosmology function returning a plain array for zticks used to fail."""
    # Planck18.efunc returns a plain array (not a Quantity)
    _, ax = plotting.lightcone_sliceplot(lc, zticks="efunc")
    assert ax.xaxis.get_label().get_text() == "Efunc"
    assert len(ax.get_xticks()) > 0


def test_cli_multiple_lightcone_quantities():
    """``--lq`` used to accept exactly one quantity (it was typed ``tuple[str]``)."""
    _, bound, _ = cli.app.parse_args(
        ["run", "lightcone", "--lq", "brightness_temp", "--lq", "density"]
    )
    assert bound.arguments["lightcone_quantities"] == ("brightness_temp", "density")


@pytest.mark.parametrize(
    ("override_cls", "struct_cls"),
    [
        (cli._AstroOptions, p21c.AstroOptions),
        (cli._AstroParams, p21c.AstroParams),
        (cli._CosmoParams, p21c.CosmoParams),
        (cli._MatterOptions, p21c.MatterOptions),
        (cli._SimulationOptions, p21c.SimulationOptions),
    ],
)
def test_cli_override_classes_mirror_input_structs(override_cls, struct_cls):
    """The CLI override classes have one None-defaulting field per struct field."""
    fields = attrs.fields(override_cls)
    assert [f.name for f in fields] == [f.alias for f in attrs.fields(struct_cls)]
    assert all(f.default is None for f in fields)
    assert attrs.asdict(override_cls()) == dict.fromkeys((f.name for f in fields), None)
    assert override_cls.__doc__ == struct_cls.__doc__.replace(":class:", "")
