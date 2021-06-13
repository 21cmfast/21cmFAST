"""Simple plotting functions for 21cmFAST objects."""
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as un
from astropy.cosmology import z_at_value
from matplotlib import colors
from matplotlib.ticker import AutoLocator
from typing import Optional

from . import outputs
from .outputs import Coeval, LightCone

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
plt.register_cmap(cmap=eor_colour)


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
    log: [bool] = False,
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
            "slice_index is too large for that axis (slice_index=%s >= %s"
            % (slice_index, cube.shape[slice_axis])
        )

    slc = np.take(cube, slice_index, axis=slice_axis)
    if not rotate:
        slc = slc.T

    if cmap == "EoR":
        imshow_kw["vmin"] = -150
        imshow_kw["vmax"] = 30

    norm = imshow_kw.get("norm", colors.LogNorm() if log else colors.Normalize())
    plt.imshow(slc, origin="lower", cmap=cmap, norm=norm, **imshow_kw)

    if cbar:
        cb = plt.colorbar(
            orientation="horizontal" if cbar_horizontal else "vertical", aspect=40
        )
        cb.outline.set_edgecolor(None)

    return fig, ax


def coeval_sliceplot(
    struct: [outputs._OutputStruct, Coeval],
    kind: [str, None] = None,
    cbar_label: [str, None] = None,
    **kwargs,
):
    """
    Show a slice of a given coeval box.

    Parameters
    ----------
    struct : :class:`~outputs._OutputStruct` or :class:`~wrapper.Coeval` instance
        The output of a function such as `ionize_box` (a class containing several quantities), or
        `run_coeval`.
    kind : str
        The quantity within the structure to be shown. A full list of available options
        can be obtained by running ``Coeval.get_fields()``.
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
        if isinstance(struct, outputs._OutputStruct):
            kind = struct.fieldnames[0]
        elif isinstance(struct, Coeval):
            kind = "brightness_temp"

    try:
        cube = getattr(struct, kind)
    except AttributeError:
        raise AttributeError(
            f"The given OutputStruct does not have the quantity {kind}"
        )

    if kind != "brightness_temp" and "cmap" not in kwargs:
        kwargs["cmap"] = "viridis"

    fig, ax = _imshow_slice(cube, extent=(0, struct.user_params.BOX_LEN) * 2, **kwargs)

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
            if kind == "brightness_temp":
                cbar_label = r"Brightness Temperature, $\delta T_B$ [mK]"
            elif kind == "xH_box":
                cbar_label = r"Neutral fraction"

        cbar.ax.set_ylabel(cbar_label)

    return fig, ax


def lightcone_sliceplot(
    lightcone: LightCone,
    kind: str = "brightness_temp",
    lightcone2: LightCone = None,
    vertical: bool = False,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cbar_label: Optional[str] = None,
    zticks: str = "redshift",
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
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

    if fig is None and ax is None:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(
                lightcone.shape[axis_dct["x"]] * 0.015 + 0.5,
                lightcone.shape[axis_dct["y"]] * 0.015
                + (2.5 if kwargs.get("cbar", True) else 0.05),
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
        0,
        lightcone.lightcone_dimensions[axis_dct["x"]],
        0,
        lightcone.lightcone_dimensions[axis_dct["y"]],
    )

    if lightcone2 is None:
        fig, ax = _imshow_slice(
            getattr(lightcone, kind),
            extent=extent,
            slice_axis=slice_axis,
            rotate=not vertical,
            cbar_horizontal=not vertical,
            cmap=kwargs.get("cmap", "EoR" if kind == "brightness_temp" else "viridis"),
            fig=fig,
            ax=ax,
            **kwargs,
        )
    else:
        d = getattr(lightcone, kind) - getattr(lightcone2, kind)
        fig, ax = _imshow_slice(
            d,
            extent=extent,
            slice_axis=slice_axis,
            rotate=not vertical,
            cbar_horizontal=not vertical,
            cmap=kwargs.pop("cmap", "bwr"),
            vmin=-np.abs(d.max()),
            vmax=np.abs(d.max()),
            fig=fig,
            ax=ax,
            **kwargs,
        )

    if z_axis:
        zlabel = _set_zaxis_ticks(ax, lightcone, zticks, z_axis)

    if ylabel != "":
        ax.set_ylabel(ylabel or zlabel)
    if xlabel != "":
        ax.set_xlabel(xlabel or zlabel)

    cbar = fig._gci().colorbar

    if cbar_label is None:
        if kind == "brightness_temp":
            cbar_label = r"Brightness Temperature, $\delta T_B$ [mK]"
        elif kind == "xH":
            cbar_label = r"Neutral fraction"

    if vertical:
        cbar.ax.set_ylabel(cbar_label)
    else:
        cbar.ax.set_xlabel(cbar_label)

    return fig, ax


def _set_zaxis_ticks(ax, lightcone, zticks, z_axis):
    if zticks != "distance":
        loc = AutoLocator()
        # Get redshift ticks.
        lc_z = lightcone.lightcone_redshifts

        if zticks == "redshift":
            coords = lc_z
        elif zticks == "frequency":
            coords = 1420 / (1 + lc_z) * un.MHz
        else:
            try:
                coords = getattr(lightcone.cosmo_params.cosmo, zticks)(lc_z)
            except AttributeError:
                raise AttributeError(f"zticks '{zticks}' is not a cosmology function.")

        zlabel = " ".join(z.capitalize() for z in zticks.split("_"))
        units = getattr(coords, "unit", None)
        if units:
            zlabel += f" [{str(coords.unit)}]"
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
            lightcone.cosmo_params.cosmo.comoving_distance(z_ticks).value
            - lightcone.lightcone_distances[0]
        )
        getattr(ax, f"set_{z_axis}ticks")(d_ticks)
        getattr(ax, f"set_{z_axis}ticklabels")(ticks)

    else:
        zlabel = "Line-of-Sight Distance [Mpc]"
    return zlabel


def plot_global_history(
    lightcone: [LightCone],
    kind: [str, None] = None,
    ylabel: [str, None] = None,
    ylog: [bool] = False,
    ax: [plt.Axes, None] = None,
):
    """
    Plot the global history of a given quantity from a lightcone.

    Parameters
    ----------
    lightcone : :class:`~LightCone` instance
        The lightcone containing the quantity to plot.
    kind : str, optional
        The quantity to plot. Must be in the `global_quantities` dict in the lightcone.
        By default, will choose the first entry in the dict.
    ylabel : str, optional
        A y-label for the plot. If None, will use ``kind``.
    ax : Axes, optional
        The matplotlib Axes object on which to plot. Otherwise, created.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    else:
        fig = ax._gci().figure

    if kind is None:
        kind = list(lightcone.global_quantities.keys())[0]

    assert (
        kind in lightcone.global_quantities
        or hasattr(lightcone, "global_" + kind)
        or (kind.startswith("global_") and hasattr(lightcone, kind))
    )

    if kind in lightcone.global_quantities:
        value = lightcone.global_quantities[kind]
    elif kind.startswith("global)"):
        value = getattr(lightcone, kind)
    else:
        value = getattr(lightcone, "global_" + kind)

    ax.plot(lightcone.node_redshifts, value)
    ax.set_xlabel("Redshift")
    if ylabel is None:
        ylabel = kind
    if ylabel:
        ax.set_ylabel(ylabel)

    if ylog:
        ax.set_yscale("log")

    return fig, ax
