"""
Testing plots is kind of hard, but we just check that it runs through without crashing.
"""

from py21cmmc import plotting
import pytest
from py21cmmc import initial_conditions, run_lightcone


def test_coeval_sliceplot():
    ic = initial_conditions(user_params={"HII_DIM": 35, "DIM": 70})

    fig, ax = plotting.coeval_sliceplot(ic)

    assert ax.xaxis.get_label().get_text() == "x-axis [Mpc]"
    assert ax.yaxis.get_label().get_text() == "y-axis [Mpc]"

    with pytest.raises(ValueError): # bad slice axis
        plotting.coeval_sliceplot(ic, slice_axis=-2)

    with pytest.raises(IndexError): # tring to plot slice that doesn't exist
        plotting.coeval_sliceplot(ic, slice_index = 50)

    fig2, ax2 = plotting.coeval_sliceplot(ic, fig=fig, ax=ax)

    assert fig2 is fig
    assert ax2 is ax

    fig2, ax2 = plotting.coeval_sliceplot(ic, fig=fig)

    assert fig2 is fig
    assert ax2 is ax

    fig2, ax2 = plotting.coeval_sliceplot(ic, ax=ax)

    assert fig2 is fig
    assert ax2 is ax

    fig, ax = plotting.coeval_sliceplot(ic, kind="hires_density", slice_index=50, slice_axis=1)
    assert ax.xaxis.get_label().get_text() == "x-axis [Mpc]"
    assert ax.yaxis.get_label().get_text() == "z-axis [Mpc]"


def test_lightcone_sliceplot():
    lc = run_lightcone(redshift=25, max_redshift=30, user_params={"HII_DIM":35, "DIM":70})

    fig, ax = plotting.lightcone_sliceplot(lc)

    assert ax.xaxis.get_label().get_text() == "y-axis [Mpc]"
    assert ax.yaxis.get_label().get_text() == "Redshift Axis [Mpc]"

    