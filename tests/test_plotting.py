"""Testing plots is kind of hard, but we just check that it runs through without crashing."""

import attrs
import matplotlib.pyplot as plt
import numpy as np
import pytest

import py21cmfast as p21c
from py21cmfast import plotting


@pytest.fixture(autouse=True)
def _close_figures():
    """Don't leak figures between tests (matplotlib warns after 20 are open)."""
    yield
    plt.close("all")


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
    _fig, ax = plotting.lightcone_sliceplot(lc)

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "Redshift"


def test_lightcone_sliceplot_vertical(lc: p21c.LightCone):
    _fig, ax = plotting.lightcone_sliceplot(lc, vertical=True)

    assert ax.yaxis.get_label().get_text() == "Redshift"
    assert ax.xaxis.get_label().get_text() == "y-axis [Mpc]"


def test_lc_sliceplot_freq(lc: p21c.LightCone):
    _fig, ax = plotting.lightcone_sliceplot(lc, zticks="frequency")

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "Frequency [MHz]"


def test_lc_sliceplot_cdist(lc: p21c.LightCone):
    _fig, ax = plotting.lightcone_sliceplot(lc, zticks="comoving_distance")

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "Comoving Distance [Mpc]"

    xlim = ax.get_xticks()
    assert xlim.min() >= 0
    assert xlim.max() <= lc.lightcone_dimensions[-1]


def test_lc_sliceplot_sliceax(lc: p21c.LightCone):
    _fig, ax = plotting.lightcone_sliceplot(lc, slice_axis=2)

    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.xaxis.get_label().get_text() == "x-axis [Mpc]"


def test_global_plot(lc: p21c.LightCone):
    _fig, ax = plotting.plot_global_history(lc)

    assert ax.xaxis.get_label().get_text() == "Redshift"


def test_global_plot_kwargs(lc: p21c.LightCone):
    _fig, ax = plotting.plot_global_history(lc, kind="neutral_fraction", ylog=True)

    assert ax.get_yscale() == "log"


def test_get_global_quantities(lc: p21c.LightCone):
    z, q = plotting.get_global_quantities(lc)

    assert len(z) == len(lc.inputs.node_redshifts)
    assert "brightness_temp" in q

    with pytest.raises(TypeError, match="has no global quantities"):
        plotting.get_global_quantities("not-a-simulation")


def test_plot_global_signal(lc: p21c.LightCone):
    _fig, ax = plotting.plot_global_signal(lc)

    assert ax.xaxis.get_label().get_text() == "Redshift"
    assert len(ax.lines) > 0


def test_plot_global_signal_missing(lc: p21c.LightCone):
    bad = attrs.evolve(
        lc,
        global_quantities={
            "neutral_fraction": lc.global_quantities["neutral_fraction"]
        },
    )

    with pytest.raises(ValueError, match="does not contain a global brightness"):
        plotting.plot_global_signal(bad)


def test_plot_global_evolution(lc: p21c.LightCone):
    _fig, axes = plotting.plot_global_evolution(lc)

    # The "signal" and "ionization" panels are always available.
    assert len(axes) >= 2
    assert axes[-1].xaxis.get_label().get_text() == "Redshift"


def test_plot_global_evolution_single_panel(lc: p21c.LightCone):
    _fig, axes = plotting.plot_global_evolution(lc, panels=["signal"])

    assert len(axes) == 1


def test_plot_global_evolution_no_data(lc: p21c.LightCone):
    bad = attrs.evolve(lc, global_quantities={})

    with pytest.raises(ValueError, match="None of the requested panels"):
        plotting.plot_global_evolution(bad)


def test_lightcone_summary_plot(lc: p21c.LightCone):
    _fig, axes = plotting.lightcone_summary_plot(lc)

    assert len(axes) >= 2
    assert axes[0].images  # the lightcone slice itself
    assert axes[-1].xaxis.get_label().get_text() == "Redshift"


def test_lightcone_summary_plot_bad_kind(lc: p21c.LightCone):
    with pytest.raises(ValueError, match="does not contain the field"):
        plotting.lightcone_summary_plot(lc, kind="a_field_that_does_not_exist")


def test_summary_plot_dispatch(lc: p21c.LightCone):
    _fig, axes = plotting.summary_plot(lc)
    assert axes[0].images

    with pytest.raises(TypeError, match="Cannot make a summary plot"):
        plotting.summary_plot("not-a-simulation")


@pytest.fixture(scope="module")
def coeval(ic, default_input_struct_ts, cache) -> p21c.Coeval:
    """A coeval box including spin temperature, for testing summary plots."""
    return p21c.run_coeval(
        initial_conditions=ic,
        inputs=default_input_struct_ts,
        cache=cache,
    )[-1]


def test_coeval_summary_plot(coeval: p21c.Coeval):
    _fig, axes = plotting.coeval_summary_plot(coeval)

    # brightness_temp, neutral_fraction, density and spin_temperature.
    assert len(axes) == 4
    assert all(ax.images for ax in axes)

    # Only the first panel keeps its y-label.
    assert axes[0].yaxis.get_label().get_text() == "y-axis [Mpc]"
    assert axes[1].yaxis.get_label().get_text() == ""


def test_coeval_summary_plot_log(coeval: p21c.Coeval):
    """Wide-dynamic-range fields should get a clipped, log color scale."""
    # The default slice is through the middle of the box.
    slc = coeval.spin_temperature[..., coeval.simulation_options.HII_DIM // 2]

    _fig, axes = plotting.coeval_summary_plot(coeval, kinds=["spin_temperature"])

    assert axes[0].images[-1].norm.__class__.__name__ == "LogNorm"

    vmin, vmax = axes[0].images[-1].get_clim()
    assert vmin > slc.min()
    assert vmax < slc.max()

    # Explicitly turning off the log scale should also turn off the clipping.
    _fig, axes = plotting.coeval_summary_plot(
        coeval, kinds=["spin_temperature"], log=False
    )
    vmin, vmax = axes[0].images[-1].get_clim()
    assert vmin == pytest.approx(slc.min())
    assert vmax == pytest.approx(slc.max())


def test_coeval_summary_plot_no_fields(coeval: p21c.Coeval):
    with pytest.raises(ValueError, match="No fields available to plot"):
        plotting.coeval_summary_plot(coeval, kinds=[])


def test_summary_plot_dispatch_coeval(coeval: p21c.Coeval):
    _fig, axes = plotting.summary_plot(coeval)
    assert axes[0].images


@pytest.fixture(scope="module")
def global_evolution() -> p21c.GlobalEvolution:
    """A synthetic global evolution containing every quantity we plot by default.

    The actual values don't matter (we only check that plotting runs and produces
    the right axes), but the redshift range is wide on purpose, so that the
    frequency axis has to thin out its ticks.
    """
    inputs = p21c.InputParameters.from_template(
        "simple", random_seed=1
    ).with_logspaced_redshifts(zmin=5.5, zmax=35)
    z = np.array(inputs.node_redshifts)

    return p21c.GlobalEvolution(
        inputs=inputs,
        quantities={
            "brightness_temp": -50 * np.exp(-(((z - 15) / 5) ** 2)),
            "neutral_fraction": 1 / (1 + np.exp(-(z - 8))),
            "xray_ionised_fraction": 0.01 * np.ones_like(z),
            "spin_temperature": 10 * (1 + z),
            "kinetic_temp_neutral": 5 * (1 + z),
        },
    )


def test_plot_global_evolution_all_panels(global_evolution: p21c.GlobalEvolution):
    _fig, axes = plotting.plot_global_evolution(global_evolution)

    assert len(axes) == 3

    # The temperature panel is logarithmic, and includes a T_CMB line on top of
    # the two temperatures actually stored in the object.
    assert axes[2].get_yscale() == "log"
    assert len(axes[2].lines) == 3

    # The frequency axis should have dropped ticks that are too close together
    # over this (wide) redshift range.
    secax = next(c for c in axes[0].child_axes if c.get_xlabel())
    assert secax.get_xlabel() == "Frequency [MHz]"
    assert 3 <= len(secax.get_xticks()) <= 8


def test_plot_global_evolution_no_frequency_axis(
    global_evolution: p21c.GlobalEvolution,
):
    _fig, axes = plotting.plot_global_evolution(global_evolution, frequency_axis=False)

    assert not axes[0].child_axes


def test_plot_global_evolution_existing_axes(global_evolution: p21c.GlobalEvolution):
    """Both `fig` and `axes` should be inferrable from the other."""
    fig, axes = plt.subplots(3, 1)

    fig2, axes2 = plotting.plot_global_evolution(global_evolution, fig=fig)
    assert fig2 is fig
    assert axes2[0] is axes[0]

    fig3, axes3 = plotting.plot_global_evolution(global_evolution, axes=list(axes))
    assert fig3 is fig
    assert axes3[0] is axes[0]

    fig4, axes4 = plotting.plot_global_evolution(
        global_evolution, fig=fig, axes=list(axes)
    )
    assert fig4 is fig
    assert axes4[0] is axes[0]


def test_plot_global_signal_on_existing_ax(global_evolution: p21c.GlobalEvolution):
    fig, ax = plt.subplots(1, 1)

    fig2, ax2 = plotting.plot_global_signal(
        global_evolution, ax=ax, frequency_axis=False
    )

    assert fig2 is fig
    assert ax2 is ax
    assert not ax.child_axes


def test_summary_plot_dispatch_global_evolution(
    global_evolution: p21c.GlobalEvolution,
):
    _fig, axes = plotting.summary_plot(global_evolution)
    assert len(axes) == 3


def test_coeval_summary_plot_explicit_slice(coeval: p21c.Coeval):
    _fig, axes = plotting.coeval_summary_plot(
        coeval, kinds=["neutral_fraction"], slice_index=0
    )

    expected = coeval.neutral_fraction[..., 0]
    assert np.allclose(axes[0].images[-1].get_array(), expected.T)


def test_coeval_summary_plot_no_cbar(coeval: p21c.Coeval):
    """Without a colorbar there is nothing to put log ticks on."""
    _fig, axes = plotting.coeval_summary_plot(
        coeval, kinds=["spin_temperature"], cbar=False
    )

    assert axes[0].images[-1].colorbar is None


def test_coeval_summary_plot_log_with_negatives(coeval: p21c.Coeval):
    """Forcing a log scale on a field with negative values is an error."""
    with pytest.raises(ValueError, match="non-positive values"):
        plotting.coeval_summary_plot(coeval, kinds=["density"], log=True)


def test_lightcone_summary_plot_explicit_panels(lc: p21c.LightCone):
    _fig, axes = plotting.lightcone_summary_plot(lc, panels=["signal"])

    # One lightcone slice plus the single requested panel.
    assert len(axes) == 2
    assert axes[0].images


def test_global_plot_zmax(lc: p21c.LightCone):
    zmax = np.mean(lc.inputs.node_redshifts)
    _fig, ax = plotting.plot_global_history(lc, zmax=zmax)

    assert ax.lines[0].get_xdata().max() < zmax


def test_global_plot_bad_kind(lc: p21c.LightCone):
    with pytest.raises(ValueError, match="is not a global quantity"):
        plotting.plot_global_history(lc, kind="a_quantity_that_does_not_exist")


def test_global_plot_of_global_evolution(global_evolution: p21c.GlobalEvolution):
    """`plot_global_history` should work on a GlobalEvolution too."""
    _fig, ax = plotting.plot_global_history(global_evolution, kind="neutral_fraction")

    assert ax.yaxis.get_label().get_text() == "neutral_fraction"
    assert len(ax.lines[0].get_xdata()) == len(global_evolution.node_redshifts)


def test_frequency_axis_with_no_nice_ticks():
    """A very narrow redshift range contains no round frequency to tick."""
    # 5.2 < z < 5.4 maps to 221 < nu < 229 MHz, which contains no round number.
    inputs = p21c.InputParameters.from_template(
        "simple", random_seed=1
    ).with_logspaced_redshifts(zmin=5.2, zmax=5.4)
    z = np.array(inputs.node_redshifts)

    ge = p21c.GlobalEvolution(
        inputs=inputs, quantities={"brightness_temp": np.zeros(len(z))}
    )

    _fig, ax = plotting.plot_global_signal(ge)

    secax = next(c for c in ax.child_axes if c.get_xlabel())
    assert secax.get_xlabel() == "Frequency [MHz]"

    # We leave matplotlib's own ticks alone rather than blanking the axis out.
    assert len(secax.get_xticks()) > 0


def test_global_plot_on_existing_ax(lc: p21c.LightCone):
    fig, ax = plt.subplots(1, 1)
    fig2, ax2 = plotting.plot_global_history(lc, ax=ax)

    assert fig2 is fig
    assert ax2 is ax


@pytest.mark.parametrize("ylabel", ["Custom Label", ""])
def test_global_plot_ylabel(lc: p21c.LightCone, ylabel: str):
    _fig, ax = plotting.plot_global_history(lc, ylabel=ylabel)

    assert ax.yaxis.get_label().get_text() == ylabel
