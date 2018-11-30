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
        raise AttributeError(f"The given OutputStruct does not have the quantity {kind}")

    fig, ax = _imshow_slice(cube, extent=(0, struct.user_params.BOX_LEN) * 2, **kwargs)

    # Determine which axes are being plotted.
    if kwargs.get("slice_axis", -1) in (2, -1):
        xax = "x"
        yax = 'y'
    elif kwargs.get("slice_axis", -1) == 1:
        xax = 'x'
        yax = 'z'
    else:
        xax = 'y'
        yax = 'z'

    # Now put on the decorations.
    ax.set_xlabel(f"{xax}-axis [Mpc]")
    ax.set_ylabel(f"{yax}-axis [Mpc]")

    return fig, ax


def lightcone_sliceplot(lightcone, **kwargs):
    slice_axis = kwargs.pop("slice_axis", 0)

    fig, ax = _imshow_slice(lightcone.brightness_temp,
                            extent=(0, lightcone.user_params.BOX_LEN, 0, lightcone.lightcone_coords[-1]),
                            slice_axis=slice_axis)

    ax.set_ylabel("Redshift Axis [Mpc]")
    ax.set_xlabel("Y-Axis [Mpc]")

    # TODO: use twinx to put a redshift axis on it.
    return fig, ax
