"""
Tests for the tools in the wrapper.
"""

import pytest

import numpy as np

from py21cmfast import cache_tools


def test_query(ic):
    things = list(cache_tools.query_cache())

    print(things)

    classes = [t[1] for t in things]
    assert ic in classes


def test_bad_fname(tmpdirec):
    with pytest.raises(ValueError):
        cache_tools.readbox(direc=str(tmpdirec), fname="a_really_fake_file.h5")


def test_readbox_data(tmpdirec, ic):
    box = cache_tools.readbox(direc=str(tmpdirec), fname=ic.filename)

    assert np.all(box.hires_density == ic.hires_density)


def test_readbox_filter(ic, tmpdirec):
    ic2 = cache_tools.readbox(
        kind="InitialConditions", hsh=ic._md5, direc=str(tmpdirec)
    )
    assert np.all(ic2.hires_density == ic.hires_density)


def test_readbox_seed(ic, tmpdirec):
    ic2 = cache_tools.readbox(
        kind="InitialConditions",
        hsh=ic._md5,
        seed=ic.random_seed,
        direc=str(tmpdirec),
    )
    assert np.all(ic2.hires_density == ic.hires_density)


def test_readbox_nohash(ic, tmpdirec):
    with pytest.raises(ValueError):
        cache_tools.readbox(
            kind="InitialConditions", seed=ic.random_seed, direc=str(tmpdirec)
        )


def test_get_boxes_at_redshift(redshift, tmpdirec, perturbed_field):
    boxes = cache_tools.get_boxes_at_redshift(redshift, direc=tmpdirec)
    assert len(boxes["PerturbedField"]) == 1
    assert boxes["PerturbedField"][0].redshift == redshift
    assert boxes["PerturbedField"][0] == perturbed_field


def test_get_boxes_at_redshift_range(redshift, tmpdirec, perturbed_field):
    boxes = cache_tools.get_boxes_at_redshift(
        (redshift - 3, redshift + 3), direc=tmpdirec
    )
    assert len(boxes["PerturbedField"]) == 1
    assert boxes["PerturbedField"][0].redshift == redshift
    assert boxes["PerturbedField"][0] == perturbed_field

    # But at a different range...
    boxes = cache_tools.get_boxes_at_redshift(
        (redshift - 3, redshift - 1), direc=tmpdirec
    )
    assert len(boxes["PerturbedField"]) == 0


def test_get_boxes_at_redshift_badfile(redshift, tmpdirec, perturbed_field):
    # Add a file that should not be read
    badpath = tmpdirec / "I_am_a_bad_file.h5"
    badpath.touch()

    # And it still works fine.
    boxes = cache_tools.get_boxes_at_redshift(redshift, direc=tmpdirec)
    assert len(boxes["PerturbedField"]) == 1
    assert boxes["PerturbedField"][0].redshift == redshift
    assert boxes["PerturbedField"][0] == perturbed_field


def test_get_boxes_at_redshift_seed(redshift, tmpdirec, perturbed_field):
    boxes = cache_tools.get_boxes_at_redshift(redshift, seed=12, direc=tmpdirec)
    assert len(boxes["PerturbedField"]) == 1
    assert boxes["PerturbedField"][0].redshift == redshift
    assert boxes["PerturbedField"][0] == perturbed_field

    # Use a different seed...
    boxes = cache_tools.get_boxes_at_redshift(redshift, seed=13, direc=tmpdirec)
    assert len(boxes["PerturbedField"]) == 0


def test_get_boxes_at_redshift_with_params(
    redshift, tmpdirec, default_user_params, perturbed_field
):
    boxes = cache_tools.get_boxes_at_redshift(
        redshift, direc=tmpdirec, user_params=default_user_params
    )
    assert len(boxes["PerturbedField"]) == 1
    assert boxes["PerturbedField"][0].redshift == redshift
    assert boxes["PerturbedField"][0] == perturbed_field

    # Use a different set of params
    new = default_user_params.clone(DIM=2 * default_user_params.DIM)
    boxes = cache_tools.get_boxes_at_redshift(redshift, direc=tmpdirec, user_params=new)
    assert len(boxes["PerturbedField"]) == 0
