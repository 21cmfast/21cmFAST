"""
Simple plotting functions for 21cmFAST objects.
"""

import matplotlib.pyplot as plt
import numpy as np


def _imshow_slice(cube, slice_axis=-1, slice_index=0, fig=None, ax=None, fig_kw=None, cbar=True,
                  **imshow_kw):
    """
    Helper function to plot a slice of some kind of cube.

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
    ax
    fig_kw
    cbar
    imshow_kw

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
        raise IndexError("slice_index is too large for that axis (slice_index=%s >= %s"%(slice_index, cube.shape[slice_axis]))

    slc = np.take(cube, slice_index, axis=slice_axis)
    plt.imshow(slc.T, origin='lower', **imshow_kw)

    if cbar: plt.colorbar()

    return fig, ax


def coeval_sliceplot(struct, kind=None, **kwargs):
    """
    Show a slice of a given coeval box.

    Parameters
    ----------
    struct : :class:`py21cmmc._OutputStruct` instance
        The output of a function such as `ionize_box` (a class containing several quantities).
    kind : str
        The quantity within the structure to be shown.


    Returns
    -------
    fig, ax :
        figure and axis objects from matplotlib

    Other Parameters
    ----------------
    All other parameters are passed directly to :func:`_imshow_slice`. These include `slice_axis` and `slice_index`,
    which choose the actual slice to plot, optional `fig` and `ax` keywords which enable over-plotting previous figures,
    and the `imshow_kw` argument, which allows arbitrary styling of the plot.
    """
    if kind is None:
        kind = struct.fieldnames[0]

    try:
        cube = getattr(struct, kind)
    except AttributeError:
        raise AttributeError("The given OutputStruct does not have the quantity {kind}".format(kind=kind))

    fig, ax = _imshow_slice(cube, extent=(0, struct.user_params.BOX_LEN) * 2, **kwargs)

    slice_axis = kwargs.get("slice_axis", -1)

    # Determine which axes are being plotted.
    if slice_axis in (2, -1):
        xax = "x"
        yax = 'y'
    elif slice_axis == 1:
        xax = 'x'
        yax = 'z'
    elif slice_axis == 0:
        xax = 'y'
        yax = 'z'
    else:
        raise ValueError("slice_axis should be between -1 and 2")

    # Now put on the decorations.
    ax.set_xlabel("{xax}-axis [Mpc]".format(xax=xax))
    ax.set_ylabel("{yax}-axis [Mpc]".format(yax=yax))

    return fig, ax


def lightcone_sliceplot(lightcone, **kwargs):
    slice_axis = kwargs.pop("slice_axis", 0)

    if slice_axis == 0:
        extent = (0, lightcone.user_params.BOX_LEN, 0, lightcone.lightcone_coords[-1])
        ylabel = "Redshift Axis [Mpc]"
        xlabel = "y-axis [Mpc]"

    else:
        extent = (0, lightcone.user_params.BOX_LEN)*2

        if slice_axis == 1:
            xlabel = "x-axis [Mpc]"
            ylabel = "Redshift Axis [Mpc]"

        elif slice_axis in (2,-1):
            xlabel = "x-axis [Mpc]"
            ylabel = 'y-axis [Mpc]'
        else:
            raise ValueError("slice_axis must be between -1 and 2")

    fig, ax = _imshow_slice(lightcone.brightness_temp,
                            extent=extent,
                            slice_axis=slice_axis)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # TODO: use twinx to put a redshift axis on it.
    return fig, ax
