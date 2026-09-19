"""Simple plotting functions for 21cmFAST objects."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as un
from astropy.cosmology import z_at_value
from matplotlib import colormaps, colors
from matplotlib.ticker import AutoLocator

from .drivers.coeval import Coeval
from .drivers.lightcone import LightCone
from .wrapper import outputs

eor_colour = colors.LinearSegmentedColormap.from_list(
    "EoR",
    [
        (0, "white"),
        (0.21, "yellow"),
        (0.42, "orange"),
        (0.63, "red"),
        (0.86, "black"),
        (0.9, "blue"),
        (1, "cyan"),
    ],
)
logger = logging.getLogger(__name__)
colormaps.register(cmap=eor_colour)


def _imshow_slice(
    cube,
    slice_axis=-1,
    slice_index=0,
    fig=None,
    ax=None,
    fig_kw=None,
    cbar=True,
    cbar_horizontal=False,
    rotate=False,
    cmap="EoR",
    log: bool = False,
    **imshow_kw,
):
    """
    Plot a slice of some kind of cube.

    Parameters
    ----------
    cube : nd-array
        A 3D array of some quantity.
    slice_axis : int, optional
        The axis over which to take a slice, in order to plot.
    slice_index :
        The index of the slice.
    fig : Figure object
        An optional matplotlib figure object on which to plot
    ax : Axis object
        The matplotlib axis object on which to plot (created by default).
    fig_kw :
        Optional arguments passed to the figure construction.
    cbar : bool
        Whether to plot the colorbar
    cbar_horizontal : bool
        Whether the colorbar should be horizontal underneath the plot.
    rotate : bool
        Whether to rotate the plot vertically.
    imshow_kw :
        Optional keywords to pass to :func:`maplotlib.imshow`.

    Returns
    -------
    fig, ax :
        The figure and axis objects from matplotlib.
    """
    # If no axis is passed, create a new one
    # This allows the user to add this plot into an existing grid, or alter it afterwards.
    if fig_kw is None:
        fig_kw = {}
    if ax is None and fig is None:
        fig, ax = plt.subplots(1, 1, **fig_kw)
    elif ax is None:
        ax = plt.gca()
    elif fig is None:
        fig = plt.gcf()

    plt.sca(ax)

    if slice_index >= cube.shape[slice_axis]:
        raise IndexError(
            f"slice_index is too large for that axis (slice_index={slice_index} >= {cube.shape[slice_axis]}"
        )

    slc = np.take(cube, slice_index, axis=slice_axis)
    if not rotate:
        slc = slc.T

    if cmap == "EoR" and "norm" not in imshow_kw:
        imshow_kw["norm"] = colors.Normalize(vmin=-150, vmax=30)

    norm_kw = {k: imshow_kw.pop(k) for k in ["vmin", "vmax"] if k in imshow_kw}
    norm = imshow_kw.pop(
        "norm", colors.LogNorm(**norm_kw) if log else colors.Normalize(**norm_kw)
    )
    plt.imshow(slc, origin="lower", cmap=cmap, norm=norm, **imshow_kw)

    if cbar:
        ax_long = np.amax(slc.shape)
        ax_short = np.amin(slc.shape)
        asp = 40 if cbar_horizontal == rotate else 10
        if cbar_horizontal == rotate:
            # long edge colorbar
            frac = ax_long / (asp * ax_short + ax_long)
            pad = 0.10
        else:
            # short edge colorbar
            frac = ax_short / (asp * ax_long + ax_short)
            pad = 0.02

        cb = plt.colorbar(
            orientation="horizontal" if cbar_horizontal else "vertical",
            aspect=asp,
            ax=ax,
            fraction=frac,
            pad=pad,
        )
        cb.outline.set_edgecolor(None)

    return fig, ax


def coeval_sliceplot(
    struct: outputs._OutputStruct | Coeval,
    kind: str | None = None,
    cbar_label: str | None = None,
    **kwargs,
):
    """
    Show a slice of a given coeval box.

    Parameters
    ----------
    struct : :class:`~outputs._OutputStruct` or :class:`~wrapper.Coeval` instance
        The output of a function such as `ionize_box` (a class containing several quantities), or
        `run_coeval`.
    kind : str or nd-array
        If str: The quantity within the structure to be shown. A full list of available options
                can be obtained by running ``Coeval.get_fields()``.
        If nd-array: A 3D box to be shown. This could be useful for plotting the coeval brightness temperature box
                     after redshift space distortions have been applied.
    cbar_label : str, optional
        A label for the colorbar. Some values of `kind` will have automatically chosen
        labels, but these can be turned off by setting ``cbar_label=''``.

    Returns
    -------
    fig, ax :
        figure and axis objects from matplotlib

    Other Parameters
    ----------------
    All other parameters are passed directly to :func:`_imshow_slice`. These include `slice_axis`
    and `slice_index`,
    which choose the actual slice to plot, optional `fig` and `ax` keywords which enable
    over-plotting previous figures,
    and the `imshow_kw` argument, which allows arbitrary styling of the plot.
    """
    if kind is None:
        if isinstance(struct, outputs.OutputStruct):
            kind = struct._struct.fieldnames[0]
        elif isinstance(struct, Coeval):
            kind = "brightness_temp"

    if isinstance(kind, str):
        if isinstance(struct, outputs.OutputStruct):
            cube = struct.get(kind)
        elif isinstance(struct, Coeval):
            cube = getattr(struct, kind)
    else:
        cube = kind

    if isinstance(kind, str) and kind != "brightness_temp" and "cmap" not in kwargs:
        kwargs["cmap"] = "viridis"

    fig, ax = _imshow_slice(
        cube, extent=(0, struct.simulation_options.BOX_LEN) * 2, **kwargs
    )

    slice_axis = kwargs.get("slice_axis", -1)

    # Determine which axes are being plotted.
    if slice_axis in (2, -1):
        xax = "x"
        yax = "y"
    elif slice_axis == 1:
        xax = "x"
        yax = "z"
    elif slice_axis == 0:
        xax = "y"
        yax = "z"
    else:
        raise ValueError("slice_axis should be between -1 and 2")

    # Now put on the decorations.
    ax.set_xlabel(f"{xax}-axis [Mpc]")
    ax.set_ylabel(f"{yax}-axis [Mpc]")

    cbar = fig._gci().colorbar

    if cbar is not None:
        if cbar_label is None:
            if isinstance(kind, str) and kind == "brightness_temp":
                cbar_label = r"Brightness Temperature, $\delta T_B$ [mK]"
            elif isinstance(kind, str) and kind == "neutral_fraction":
                cbar_label = r"Neutral fraction"

        if kwargs.get("cbar_horizontal", False):
            cbar.ax.set_xlabel(cbar_label)
        else:
            cbar.ax.set_ylabel(cbar_label)

    return fig, ax


def lightcone_sliceplot(
    lightcone: LightCone,
    kind: str = "brightness_temp",
    lightcone2: LightCone = None,
    vertical: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cbar_label: str | None = None,
    zticks: str = "redshift",
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
    z_range=None,
    **kwargs,
):
    """Create a 2D plot of a slice through a lightcone.

    Parameters
    ----------
    lightcone : :class:`~py21cmfast.wrapper.Lightcone`
        The lightcone object to plot
    kind : str, optional
        The attribute of the lightcone to plot. Must be an array.
    lightcone2 : str, optional
        If provided, plot the _difference_ of the selected attribute between the two
        lightcones.
    vertical : bool, optional
        Whether to plot the redshift in the vertical direction.
    cbar_label : str, optional
        A label for the colorbar. Some quantities have automatically chosen labels, but
        these can be removed by setting `cbar_label=''`.
    zticks : str, optional
        Defines the co-ordinates of the ticks along the redshift axis.
        Can be "redshift" (default), "frequency", "distance" (which starts at zero
        for the lowest redshift) or the name of any function in an astropy cosmology
        that is purely a function of redshift.
    kwargs :
        Passed through to ``imshow()``.

    Returns
    -------
    fig :
        The matplotlib Figure object
    ax :
        The matplotlib Axis object onto which the plot was drawn.
    """
    slice_axis = kwargs.pop("slice_axis", 0)
    if slice_axis <= -2 or slice_axis >= 3:
        raise ValueError(f"slice_axis should be between -1 and 2 (got {slice_axis})")

    z_axis = ("y" if vertical else "x") if slice_axis in (0, 1) else None

    # Dictionary mapping axis to dimension in lightcone
    axis_dct = {
        "x": 2 if z_axis == "x" else [1, 0, 0][slice_axis],
        "y": 2 if z_axis == "y" else [1, 0, 1][slice_axis],
    }

    plot_shape = [lightcone.shape[axis_dct["x"]], lightcone.shape[axis_dct["y"]]]
    plot_crd = [
        [0, lightcone.lightcone_dimensions[axis_dct["x"]]],
        [0, lightcone.lightcone_dimensions[axis_dct["y"]]],
    ]

    plot_sel = Ellipsis
    if z_range is not None and slice_axis in (0, 1):
        zmax_idx = np.argmin(np.fabs(z_range[1] - lightcone.lightcone_redshifts))
        zmin_idx = np.argmin(np.fabs(z_range[0] - lightcone.lightcone_redshifts))
        ax_idx = 1 if vertical else 0
        plot_shape[ax_idx] = zmax_idx - zmin_idx

        plot_crd[ax_idx][1] = lightcone.lightcone_coords[zmax_idx].to("Mpc").value
        plot_crd[ax_idx][0] = lightcone.lightcone_coords[zmin_idx].to("Mpc").value
        plot_sel = slice(zmin_idx, zmax_idx, 1)

    if fig is None and ax is None:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(
                plot_shape[0] * 0.015 + 0.5,
                plot_shape[1] * 0.015 + (2.5 if kwargs.get("cbar", True) else 0.05),
            ),
        )
    elif fig is None:
        fig = ax._gci().figure
    elif ax is None:
        ax = fig.get_axes()

    # Get x,y labels if they're not the redshift axis.
    if xlabel is None:
        xlabel = (
            None if axis_dct["x"] == 2 else "{}-axis [Mpc]".format("xy"[axis_dct["x"]])
        )
    if ylabel is None:
        ylabel = (
            None if axis_dct["y"] == 2 else "{}-axis [Mpc]".format("xy"[axis_dct["y"]])
        )

    extent = (
        plot_crd[0][0],
        plot_crd[0][1],
        plot_crd[1][0],
        plot_crd[1][1],
    )

    cmap = kwargs.pop(
        "cmap", "EoR" if kind.startswith("brightness_temp") else "viridis"
    )
    cbar_horizontal = kwargs.pop("cbar_horizontal", not vertical)

    if lightcone2 is None:
        fig, ax = _imshow_slice(
            lightcone.lightcones[kind][:, :, plot_sel],
            extent=extent,
            slice_axis=slice_axis,
            rotate=not vertical,
            cbar_horizontal=cbar_horizontal,
            cmap=cmap,
            fig=fig,
            ax=ax,
            **kwargs,
        )
    else:
        d = (lightcone.lightcones[kind][:, :, plot_sel] - lightcone2.lightcones[kind])[
            :, :, plot_sel
        ]
        fig, ax = _imshow_slice(
            d,
            extent=extent,
            slice_axis=slice_axis,
            rotate=not vertical,
            cbar_horizontal=cbar_horizontal,
            cmap=kwargs.pop("cmap", "bwr"),
            fig=fig,
            ax=ax,
            **kwargs,
        )

    if z_axis:
        zlabel = _set_zaxis_ticks(ax, lightcone, zticks, z_axis, z_range)

    if ylabel != "":
        ax.set_ylabel(ylabel or zlabel)
    if xlabel != "":
        ax.set_xlabel(xlabel or zlabel)

    cbar = fig._gci().colorbar

    if cbar_label is None:
        if kind == "brightness_temp":
            cbar_label = r"Brightness Temperature, $\delta T_B$ [mK]"
        elif kind == "neutral_fraction":
            cbar_label = r"Neutral fraction"
        else:
            cbar_label = kind

    if cbar is not None:
        if cbar_horizontal:
            cbar.ax.set_xlabel(cbar_label)
        else:
            cbar.ax.set_ylabel(cbar_label)

    return fig, ax


def _set_zaxis_ticks(ax, lightcone, zticks, z_axis, z_range):
    if zticks == "none":
        getattr(ax, f"set_{z_axis}ticks")([])
        getattr(ax, f"set_{z_axis}ticklabels")([])
        return ""

    if zticks != "distance":
        if z_range is None:
            z_max = lightcone.lightcone_redshifts.max()
            z_min = lightcone.lightcone_redshifts.min()
        else:
            z_min, z_max = z_range

        loc = AutoLocator()
        # Get redshift ticks.

        lc_z = lightcone.lightcone_redshifts[
            (lightcone.lightcone_redshifts < z_max)
            & (lightcone.lightcone_redshifts > z_min)
        ]

        if zticks == "redshift":
            coords = lc_z
        elif zticks == "frequency":
            coords = 1420 / (1 + lc_z) * un.MHz
        else:
            try:
                coords = getattr(lightcone.cosmo_params.cosmo, zticks)(lc_z)
            except AttributeError as e:
                raise AttributeError(
                    f"zticks '{zticks}' is not a cosmology function."
                ) from e

        zlabel = " ".join(z.capitalize() for z in zticks.split("_"))
        units = getattr(coords, "unit", None)
        if units:
            zlabel += f" [{coords.unit!s}]"
            coords = coords.value

        ticks = loc.tick_values(coords.min(), coords.max())

        if ticks.min() < coords.min() / 1.00001:
            ticks = ticks[1:]
        if ticks.max() > coords.max() * 1.00001:
            ticks = ticks[:-1]

        if coords[1] < coords[0]:
            ticks = ticks[::-1]

        if zticks == "redshift":
            z_ticks = ticks
        elif zticks == "frequency":
            z_ticks = 1420 / ticks - 1
        else:
            z_ticks = [
                z_at_value(getattr(lightcone.cosmo_params.cosmo, zticks), z * units)
                for z in ticks
            ]

        d_ticks = (
            lightcone.cosmo_params.cosmo.comoving_distance(z_ticks)
            - lightcone.lightcone_distances[0]
        )
        getattr(ax, f"set_{z_axis}ticks")(d_ticks.to_value("Mpc"))
        getattr(ax, f"set_{z_axis}ticklabels")(ticks)

    else:
        zlabel = "Line-of-Sight Distance [Mpc]"
    return zlabel


def plot_global_history(
    lightcone: LightCone,
    kind: str | None = None,
    ylabel: str | None = None,
    ylog: bool = False,
    ax: plt.Axes | None = None,
    zmax: float | None = None,
    **kwargs,
):
    """
    Plot the global history of a given quantity from a lightcone.

    Parameters
    ----------
    lightcone : :class:`~LightCone` or :class:`~GlobalEvolution` instance
        The object containing the quantity to plot.
    kind : str, optional
        The quantity to plot. Must be one of the global quantities of the given
        object. By default, will choose the first one.
    ylabel : str, optional
        A y-label for the plot. If None, will use ``kind``.
    ylog : bool, optional
        Whether to use a logarithmic y-axis.
    ax : Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    zmax : float, optional
        If given, only plot redshifts below this value.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    else:
        fig = ax.get_figure()

    redshifts, quantities = get_global_quantities(lightcone)

    if kind is None:
        kind = next(iter(quantities.keys()))

    if kind not in quantities:
        raise ValueError(
            f"'{kind}' is not a global quantity of the given object. "
            f"Available quantities: {sorted(quantities.keys())}"
        )

    sel = redshifts < zmax if zmax is not None else Ellipsis

    ax.plot(redshifts[sel], quantities[kind][sel], **kwargs)
    ax.set_xlabel("Redshift")
    if ylabel is None:
        ylabel = kind
    if ylabel:
        ax.set_ylabel(ylabel)

    if ylog:
        ax.set_yscale("log")

    return fig, ax


# Rest-frame frequency of the 21-cm line, in MHz.
_NU_21CM = 1420.405751768

_FIELD_LABELS = {
    "brightness_temp": r"$\delta T_b$ [mK]",
    "brightness_temp_with_rsds": r"$\delta T_b$ (RSD) [mK]",
    "neutral_fraction": r"$x_{\rm HI}$",
    "xray_ionised_fraction": r"$x_e$",
    "density": r"$\delta$",
    "spin_temperature": r"$T_S$ [K]",
    "kinetic_temp_neutral": r"$T_K$ [K]",
    "J_21_LW": r"$J_{21}^{\rm LW}$",
    "ionisation_rate_G12": r"$\Gamma_{12}$",
    "log10_mturn_acg": r"$\log_{10}(M_{\rm turn}^{\rm ACG}/M_\odot)$",
    "log10_mturn_mcg": r"$\log_{10}(M_{\rm turn}^{\rm MCG}/M_\odot)$",
}

# Panels used by the "default" global-evolution summary plot. Each panel is only
# drawn if at least one of its quantities is present in the object being plotted.
_GLOBAL_PANELS = {
    "signal": {
        "quantities": ("brightness_temp",),
        "ylabel": r"$\overline{\delta T_b}$ [mK]",
        "ylog": False,
        "legend": False,
    },
    "ionization": {
        "quantities": ("neutral_fraction", "xray_ionised_fraction"),
        "ylabel": "Fraction",
        "ylog": False,
        "legend": True,
    },
    "temperature": {
        "quantities": ("spin_temperature", "kinetic_temp_neutral"),
        "ylabel": "Temperature [K]",
        "ylog": True,
        "legend": True,
    },
}

# The fields shown, in order, by :func:`coeval_summary_plot`, if they exist.
_DEFAULT_COEVAL_FIELDS = (
    "brightness_temp",
    "neutral_fraction",
    "density",
    "spin_temperature",
)

# Fields whose dynamic range is large enough that they are best shown on a log scale.
_LOG_FIELDS = ("spin_temperature", "kinetic_temp_neutral", "J_21_LW")


def _field_label(kind: str) -> str:
    """Return a nice math-mode label for a given field name."""
    return _FIELD_LABELS.get(kind, kind)


def get_global_quantities(obj) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Get the redshifts and global quantities of a lightcone or global evolution.

    Parameters
    ----------
    obj
        Either a :class:`~py21cmfast.drivers.lightcone.LightCone` or a
        :class:`~py21cmfast.drivers.global_evolution.GlobalEvolution`.

    Returns
    -------
    redshifts
        The node redshifts of the simulation.
    quantities
        A dictionary of global (i.e. mean) quantities at each node redshift.
    """
    quantities = getattr(obj, "global_quantities", None)
    if quantities is None:
        quantities = getattr(obj, "quantities", None)

    if quantities is None:
        raise TypeError(
            f"Object of type {type(obj).__name__} has no global quantities to plot."
        )

    return np.array(obj.inputs.node_redshifts), quantities


def _plot_global_panel(
    ax: plt.Axes,
    panel: dict,
    redshifts: np.ndarray,
    quantities: dict[str, np.ndarray],
    x: np.ndarray | None = None,
    cosmo=None,
    **kwargs,
):
    """Draw a single panel of the default global-evolution plot.

    Callers are responsible for only passing panels that have data available (see
    :func:`_available_panels`).
    """
    xx = redshifts if x is None else x

    for i, q in enumerate(panel["quantities"]):
        if q not in quantities:
            continue
        ax.plot(xx, quantities[q], color=f"C{i}", label=_field_label(q), **kwargs)

    if panel is _GLOBAL_PANELS["temperature"] and cosmo is not None:
        ax.plot(
            xx,
            cosmo.Tcmb0.to_value("K") * (1 + redshifts),
            color="k",
            ls=":",
            label=r"$T_{\rm CMB}$",
        )
    elif panel is _GLOBAL_PANELS["signal"]:
        ax.axhline(0, color="k", ls=":", lw=0.5)

    ax.set_ylabel(panel["ylabel"])
    if panel["ylog"]:
        ax.set_yscale("log")
    if panel["legend"]:
        ax.legend(frameon=False, ncols=3, fontsize=9)


def _available_panels(
    panels: Sequence[str] | None, quantities: dict[str, np.ndarray]
) -> list[str]:
    """Filter ``panels`` down to those with at least one quantity available."""
    if panels is None:
        panels = list(_GLOBAL_PANELS.keys())

    return [
        p
        for p in panels
        if any(q in quantities for q in _GLOBAL_PANELS[p]["quantities"])
    ]


def plot_global_evolution(
    obj,
    panels: Sequence[str] | None = None,
    fig: plt.Figure | None = None,
    axes: Sequence[plt.Axes] | None = None,
    frequency_axis: bool = True,
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot a default set of global (mean) quantities as a function of redshift.

    This is a simple summary plot, giving a quick overview of a simulation: the
    global 21-cm signal, the ionization history and the temperature history
    (whichever of these are available in the given object).

    Parameters
    ----------
    obj
        A :class:`~py21cmfast.drivers.lightcone.LightCone` or
        :class:`~py21cmfast.drivers.global_evolution.GlobalEvolution` object.
    panels
        Which panels to plot. By default, all of "signal", "ionization" and
        "temperature" that have data available.
    fig, axes
        Optional pre-existing figure/axes on which to plot. If given, there must be
        at least as many axes as panels.
    frequency_axis
        Whether to add a secondary axis at the top of the figure showing the
        observed frequency of the 21-cm line.
    kwargs
        Passed through to ``ax.plot()``.

    Returns
    -------
    fig
        The matplotlib Figure object.
    axes
        The list of matplotlib Axes objects that were drawn onto.
    """
    redshifts, quantities = get_global_quantities(obj)

    # Only keep panels for which we actually have data.
    panels = _available_panels(panels, quantities)
    if not panels:
        raise ValueError("None of the requested panels have any data to plot!")

    if fig is None and axes is None:
        fig, axes = plt.subplots(
            len(panels),
            1,
            sharex=True,
            figsize=(7, 2.5 * len(panels)),
            gridspec_kw={"hspace": 0.05},
            squeeze=False,
        )
        axes = list(axes.flatten())
    elif axes is None:
        axes = list(fig.get_axes())
    elif fig is None:
        fig = axes[0].get_figure()

    for ax, panel in zip(axes, panels, strict=False):
        _plot_global_panel(
            ax,
            _GLOBAL_PANELS[panel],
            redshifts,
            quantities,
            cosmo=obj.inputs.cosmo_params.cosmo,
            **kwargs,
        )

    axes[len(panels) - 1].set_xlabel("Redshift")

    if frequency_axis:
        _add_frequency_axis(axes[0])

    return fig, axes[: len(panels)]


def _add_frequency_axis(ax: plt.Axes) -> plt.Axes:
    """Add a secondary x-axis to ``ax`` showing observed 21-cm frequency."""

    def z_to_nu(z):
        with np.errstate(divide="ignore", invalid="ignore"):
            return _NU_21CM / (1 + np.asarray(z, dtype=float))

    def nu_to_z(nu):
        with np.errstate(divide="ignore", invalid="ignore"):
            return _NU_21CM / np.asarray(nu, dtype=float) - 1

    secax = ax.secondary_xaxis("top", functions=(z_to_nu, nu_to_z))
    secax.set_xlabel("Frequency [MHz]")

    # The redshift <-> frequency mapping is strongly non-linear, so the automatic
    # ticks tend to bunch up at the low-redshift end. Instead, pick "nice" round
    # frequencies that are also well-separated on the redshift axis.
    zmin, zmax = sorted(ax.get_xlim())
    numin, numax = z_to_nu(zmax), z_to_nu(zmin)

    nice = np.outer(10.0 ** np.arange(0, 4), [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8]).flatten()
    # Descending in frequency == ascending in redshift, i.e. left-to-right.
    nice = np.sort(nice[(nice >= numin) & (nice <= numax)])[::-1]

    ticks = []
    last = None
    for nu in nice:
        z = nu_to_z(nu)
        if last is None or abs(z - last) > 0.08 * (zmax - zmin):
            ticks.append(nu)
            last = z

    if ticks:
        secax.set_xticks(ticks)

    return secax


def plot_global_signal(
    obj,
    ax: plt.Axes | None = None,
    frequency_axis: bool = True,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the global 21-cm signal as a function of redshift.

    Parameters
    ----------
    obj
        A :class:`~py21cmfast.drivers.lightcone.LightCone` or
        :class:`~py21cmfast.drivers.global_evolution.GlobalEvolution` object.
    ax
        The matplotlib Axes on which to plot. Created by default.
    frequency_axis
        Whether to add a secondary axis at the top showing the observed frequency
        of the 21-cm line.
    kwargs
        Passed through to ``ax.plot()``.

    Returns
    -------
    fig, ax
        The matplotlib Figure and Axes objects.
    """
    redshifts, quantities = get_global_quantities(obj)

    if "brightness_temp" not in quantities:
        raise ValueError(
            "The given object does not contain a global brightness temperature. "
            f"Available quantities: {sorted(quantities.keys())}"
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4), constrained_layout=True)
    else:
        fig = ax.get_figure()

    ax.plot(redshifts, quantities["brightness_temp"], **kwargs)
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"$\overline{\delta T_b}$ [mK]")

    if frequency_axis:
        _add_frequency_axis(ax)

    return fig, ax


def _set_log_cbar_ticks(cbar, nticks: int = 4):
    """Put a few readable, geometrically-spaced ticks on a log-scaled colorbar."""
    if cbar is None:
        return

    ticks = np.geomspace(*cbar.mappable.get_clim(), nticks)
    cbar.set_ticks(ticks, labels=[f"{t:.3g}" for t in ticks])
    cbar.minorticks_off()


def _available_coeval_fields(coeval: Coeval, kinds: Sequence[str]) -> list[str]:
    """Return the subset of ``kinds`` that are actually available on ``coeval``."""
    out = []
    for kind in kinds:
        try:
            getattr(coeval, kind)
        except (AttributeError, ValueError):
            continue
        out.append(kind)
    return out


def coeval_summary_plot(
    coeval: Coeval,
    kinds: Sequence[str] | None = None,
    slice_index: int | None = None,
    log: bool | None = None,
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Make a simple summary plot of a coeval cube.

    This plots a slice through the box for each of a default set of fields, as a
    quick visual check that the simulation ran as expected.

    Parameters
    ----------
    coeval
        The :class:`~py21cmfast.drivers.coeval.Coeval` object to plot.
    kinds
        The fields to plot. By default, whichever of ``brightness_temp``,
        ``neutral_fraction``, ``density`` and ``spin_temperature`` are available.
    slice_index
        The index of the slice to plot. By default, the middle of the box.
    log
        Whether to use a logarithmic color scale. By default, chosen per-field:
        fields spanning several orders of magnitude (e.g. the temperatures) are
        plotted logarithmically, provided they are everywhere positive. These
        fields also have their color scale clipped to their 1st-99th percentile
        range, unless you pass explicit ``vmin``/``vmax``.
    kwargs
        Passed through to :func:`coeval_sliceplot`.

    Returns
    -------
    fig, axes
        The matplotlib Figure and list of Axes objects.
    """
    if kinds is None:
        kinds = _available_coeval_fields(coeval, _DEFAULT_COEVAL_FIELDS)

    if not kinds:
        raise ValueError("No fields available to plot in the given coeval box!")

    if slice_index is None:
        slice_index = coeval.simulation_options.HII_DIM // 2

    slice_axis = kwargs.get("slice_axis", -1)

    fig, axes = plt.subplots(
        1,
        len(kinds),
        figsize=(3.6 * len(kinds), 4.2),
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(axes.flatten())

    for ax, kind in zip(axes, kinds, strict=True):
        # Some fields span several orders of magnitude, and are unreadable on a
        # linear scale (but a log scale is only possible if they're all positive).
        positive = np.min(getattr(coeval, kind)) > 0
        this_log = kind in _LOG_FIELDS and positive if log is None else log

        if this_log and not positive:
            raise ValueError(
                f"Can't use a log color scale for '{kind}', since it has "
                "non-positive values."
            )

        # These fields also have long tails (a handful of very bright cells around
        # sources), so we clip the color scale to show the bulk of the structure.
        limits = {}
        if this_log and not {"vmin", "vmax", "norm"} & set(kwargs):
            slc = np.take(getattr(coeval, kind), slice_index, axis=slice_axis)
            limits = dict(
                zip(("vmin", "vmax"), np.percentile(slc, [1, 99]), strict=True)
            )

        coeval_sliceplot(
            coeval,
            kind=kind,
            fig=fig,
            ax=ax,
            slice_index=slice_index,
            cbar_horizontal=True,
            cbar_label=_field_label(kind),
            log=this_log,
            **limits,
            **kwargs,
        )

        if this_log:
            # Log colorbars spanning less than a decade get unreadably crowded
            # tick labels by default, so we place a few of our own.
            _set_log_cbar_ticks(ax.images[-1].colorbar)

        if ax is not axes[0]:
            ax.set_ylabel("")

    fig.suptitle(f"Coeval box at $z = {coeval.redshift:.2f}$")

    return fig, axes


def lightcone_summary_plot(
    lightcone: LightCone,
    kind: str = "brightness_temp",
    panels: Sequence[str] | None = None,
    width: float = 9.0,
    panel_height: float = 1.8,
    **kwargs,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Make a simple summary plot of a lightcone.

    The top panel is a slice through the lightcone, and the panels below show the
    global evolution of a default set of quantities. All panels share the same
    line-of-sight axis, so that features can be matched up by eye.

    Parameters
    ----------
    lightcone
        The :class:`~py21cmfast.drivers.lightcone.LightCone` to plot.
    kind
        The lightcone field to show in the top panel.
    panels
        Which global-evolution panels to show below the lightcone. See
        :func:`plot_global_evolution`.
    width
        The width of the figure, in inches.
    panel_height
        The height of each global-evolution panel, in inches.
    kwargs
        Passed through to :func:`lightcone_sliceplot`.

    Returns
    -------
    fig, axes
        The matplotlib Figure and list of Axes objects. The first is the lightcone
        slice, the rest are the global-evolution panels.
    """
    if kind not in lightcone.lightcones:
        raise ValueError(
            f"The lightcone does not contain the field '{kind}'. "
            f"Available fields: {sorted(lightcone.lightcones.keys())}"
        )

    redshifts, quantities = get_global_quantities(lightcone)

    panels = _available_panels(panels, quantities)

    # Plot the global quantities against line-of-sight distance (measured from the
    # front of the lightcone), so that they line up with the lightcone slice.
    dist = (
        lightcone.cosmo_params.cosmo.comoving_distance(redshifts)
        - lightcone.lightcone_distances[0]
    ).to_value("Mpc")

    # Only show the part of the global history actually covered by the lightcone.
    xmax = lightcone.lightcone_dimensions[2]
    sel = (dist >= 0) & (dist <= xmax)

    # Margins (in inches) around the axes, and the gap between them.
    left, right, bottom, top, gap = 0.95, 0.7, 0.55, 0.1, 0.15

    # The lightcone slice is drawn with an equal aspect ratio, so we size its panel
    # to match the shape of the lightcone itself, to avoid any dead space.
    # We enforce a minimum height, since lightcones are typically long and thin,
    # and the axis labels need somewhere to live.
    cbar_frac = 0.02
    ax_width = (width - left - right) / (1 + cbar_frac)
    lc_height = max(1.1, ax_width * lightcone.lightcone_dimensions[1] / xmax)

    heights = [lc_height] + [panel_height] * len(panels)
    height = sum(heights) + len(panels) * gap + bottom + top

    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(
        len(heights),
        2,
        height_ratios=heights,
        width_ratios=[1, cbar_frac],
        hspace=gap / np.mean(heights),
        wspace=0.01,
        left=left / width,
        right=1 - right / width,
        bottom=bottom / height,
        top=1 - top / height,
    )

    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[i + 1, 0], sharex=axes[0]) for i in range(len(panels))]
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    lightcone_sliceplot(
        lightcone,
        kind=kind,
        fig=fig,
        ax=axes[0],
        cbar=False,
        zticks="none",
        **kwargs,
    )

    # The default label is too long to fit next to a short, wide lightcone panel.
    axes[0].set_ylabel("y [Mpc]")

    cbar = fig.colorbar(axes[0].images[-1], cax=fig.add_subplot(gs[0, 1]))
    cbar.set_label(_field_label(kind))
    cbar.outline.set_edgecolor(None)

    for ax, panel in zip(axes[1:], panels, strict=True):
        _plot_global_panel(
            ax,
            _GLOBAL_PANELS[panel],
            redshifts[sel],
            {k: v[sel] for k, v in quantities.items()},
            x=dist[sel],
            cosmo=lightcone.cosmo_params.cosmo,
        )

    axes[0].set_xlim(0, xmax)
    zlabel = _set_zaxis_ticks(axes[-1], lightcone, "redshift", "x", None)
    axes[-1].set_xlabel(zlabel)

    return fig, axes


def summary_plot(obj, **kwargs) -> tuple[plt.Figure, list[plt.Axes]]:
    """Make a simple default summary plot of a 21cmFAST simulation output.

    The plot produced depends on the type of ``obj``: see
    :func:`coeval_summary_plot`, :func:`lightcone_summary_plot` and
    :func:`plot_global_evolution`.

    Parameters
    ----------
    obj
        A ``Coeval``, ``LightCone`` or ``GlobalEvolution`` object.
    kwargs
        Passed through to the relevant plotting function.

    Returns
    -------
    fig, axes
        The matplotlib Figure and list of Axes objects.
    """
    if isinstance(obj, Coeval):
        return coeval_summary_plot(obj, **kwargs)
    if isinstance(obj, LightCone):
        return lightcone_summary_plot(obj, **kwargs)
    if hasattr(obj, "quantities"):
        return plot_global_evolution(obj, **kwargs)

    raise TypeError(f"Cannot make a summary plot for an object of type {type(obj)}.")
