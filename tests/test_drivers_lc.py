"""
Unit-tests of the lightcone drivers.

They do not test for correctness of simulations, but whether different parameter
options work/don't work as intended.
"""

import numpy as np
import pytest

import py21cmfast as p21c


def test_lightcone(
    lc, default_simulation_options, lightcone_min_redshift, max_redshift
):
    assert lc.lightcone_redshifts[-1] >= max_redshift
    assert np.isclose(lc.lightcone_redshifts[0], lightcone_min_redshift, atol=1e-4)
    assert (
        lc.cell_size
        == default_simulation_options.BOX_LEN / default_simulation_options.HII_DIM
    )


def test_lightcone_quantities(
    ic, default_input_struct_lc, lightcone_min_redshift, max_redshift, cache
):
    lcn = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=lightcone_min_redshift,
        max_redshift=max_redshift,
        resolution=ic.simulation_options.cell_size,
        cosmo=ic.cosmo_params.cosmo,
        quantities=(
            "density",
            "brightness_temp",
            "ionisation_rate_G12",
        ),
    )

    lc = p21c.run_lightcone(
        lightconer=lcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        cache=cache,
    )

    assert "density" in lc.lightcones
    assert "brightness_temp" in lc.lightcones
    assert "ionisation_rate_G12" in lc.lightcones
    assert "ionisation_rate_G12" in lc.global_quantities
    assert "density" in lc.global_quantities
    assert "log10_mturn_acg" in lc.global_quantities
    assert "log10_mturn_mcg" in lc.global_quantities

    # density should be filled with not zeros.
    assert lc.lightcones["density"].min() != lc.lightcones["density"].max() != 0

    # Simply ensure that different quantities are not getting crossed/referred to each other.
    assert (
        lc.lightcones["density"].min()
        != lc.lightcones["brightness_temp"].min()
        != lc.lightcones["brightness_temp"].max()
    )

    lcn_ts = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=lightcone_min_redshift,
        max_redshift=max_redshift,
        resolution=ic.simulation_options.cell_size,
        cosmo=ic.cosmo_params.cosmo,
        quantities=("spin_temperature", "density"),
    )

    # Raise an error since we're not doing spin temp.
    with pytest.raises(AttributeError):
        p21c.run_lightcone(
            lightconer=lcn_ts,
            initial_conditions=ic,
            inputs=default_input_struct_lc,
            cache=cache,
        )


def test_lightcone_coords(lc):
    assert lc.lightcone_coords.shape == (lc.lightcone_distances.shape[0],)
    assert lc.lightcone_coords[0] == 0.0
    np.testing.assert_allclose(
        np.diff(lc.lightcone_coords.to_value("Mpc")),
        lc.simulation_options.BOX_LEN / lc.simulation_options.HII_DIM,
    )


def test_run_lc_bad_inputs(
    rectlcn,
    default_input_struct_lc: p21c.InputParameters,
    cache,
):
    with pytest.raises(
        ValueError,
        match="You are attempting to run a lightcone with no node_redshifts",
    ):
        p21c.run_lightcone(
            lightconer=rectlcn,
            inputs=default_input_struct_lc.clone(node_redshifts=[]),
        )

    cosmo = rectlcn.cosmo
    with pytest.raises(
        ValueError,
        match="The cosmology used to construct the lightcone is not the same as the cosmology in the input parameters",
    ):
        p21c.run_lightcone(
            lightconer=rectlcn,
            inputs=default_input_struct_lc.evolve_input_structs(hlittle=0.9 * cosmo.h),
        )
    with pytest.raises(
        ValueError,
        match="The cosmology used to construct the lightcone is not the same as the cosmology in the input parameters",
    ):
        p21c.run_lightcone(
            lightconer=rectlcn,
            inputs=default_input_struct_lc.evolve_input_structs(OMm=0.9 * cosmo.Om0),
        )
    with pytest.raises(
        ValueError,
        match="The cosmology used to construct the lightcone is not the same as the cosmology in the input parameters",
    ):
        p21c.run_lightcone(
            lightconer=rectlcn,
            inputs=default_input_struct_lc.evolve_input_structs(OMb=0.9 * cosmo.Ob0),
        )


# Loading an already-complete lightcone still reaches the cache check, which warns.
@pytest.mark.filterwarnings(
    r"ignore:^The cache at OutputCache.* only contains complete coeval boxes for:UserWarning"
)
def test_lc_with_lightcone_filename(
    ic, rectlcn, default_input_struct_lc, tmpdirec, cache
):
    fname = tmpdirec / "lightcone.h5"
    lc = p21c.run_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
        cache=cache,
    )
    assert fname.exists()

    lc_loaded = p21c.LightCone.from_file(fname)
    assert lc_loaded == lc
    del lc_loaded

    # This one should NOT run anything.
    lc2 = p21c.run_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
        cache=cache,
    )

    assert lc2 == lc
    del lc2

    fname.unlink()


def test_lc_partial_eval(rectlcn, default_input_struct_lc, tmpdirec, lc, cache):
    fname = tmpdirec / "lightcone_partial.h5"

    z = rectlcn.lc_redshifts.max()
    lc_gen = p21c.generate_lightcone(
        lightconer=rectlcn,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
        write=True,
        cache=cache,
        include_dvdr_in_tau21=False,
    )
    while z > 20.0:
        iz_1, z, _, partial = next(lc_gen)

    assert 0 < partial._last_completed_node < len(rectlcn.lc_redshifts) - 1

    np.testing.assert_allclose(
        partial.global_quantities["brightness_temp"][
            : partial._last_completed_node + 1
        ],
        lc.global_quantities["brightness_temp"][: partial._last_completed_node + 1],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        partial.lightcones["brightness_temp"][partial._last_completed_lcidx :],
        lc.lightcones["brightness_temp"][partial._last_completed_lcidx :],
        rtol=1e-4,
    )

    lc_gen_cont = p21c.generate_lightcone(
        lightconer=rectlcn,
        inputs=default_input_struct_lc,
        lightcone_filename=fname,
        cache=cache,
        write=False,
        regenerate=False,
        include_dvdr_in_tau21=False,
    )
    for iz_2, z, _, finished in lc_gen_cont:  # noqa: B007
        assert z <= 20.0
        assert iz_2 > iz_1

    # this only checks the lightcone object, not the array fields
    assert finished == lc
    # make sure we wrote to the correct indices
    np.testing.assert_allclose(
        finished.global_quantities["brightness_temp"],
        lc.global_quantities["brightness_temp"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        finished.lightcones["brightness_temp"],
        lc.lightcones["brightness_temp"],
        rtol=1e-4,
    )


def test_lc_lowerz_than_photon_cons(
    ic, default_input_struct_lc, default_astro_options, max_redshift, cache
):
    inputs = default_input_struct_lc.clone(
        node_redshifts=p21c.get_logspaced_redshifts(
            min_redshift=1.9,
            max_redshift=max(default_input_struct_lc.node_redshifts),
            z_step_factor=default_input_struct_lc.simulation_options.ZPRIME_STEP_FACTOR,
        ),
        astro_options=default_astro_options.clone(PHOTON_CONS_TYPE="z-photoncons"),
    )
    lcn = p21c.RectilinearLightconer.between_redshifts(
        min_redshift=2.0,
        max_redshift=max_redshift,
        resolution=ic.simulation_options.cell_size,
        cosmo=ic.cosmo_params.cosmo,
    )

    with pytest.raises(ValueError, match="You have passed a redshift"):
        p21c.run_lightcone(
            lightconer=lcn, initial_conditions=ic, inputs=inputs, cache=cache
        )


def _count_array_reads(monkeypatch) -> list[str]:
    """Record the dataset of every array read from the cache from now on."""
    from py21cmfast.wrapper.arrays import Array

    reads = []
    loaded_from_disk = Array.loaded_from_disk

    def counting(self, *args, **kwargs):
        reads.append(self.cache_backend.dataset)
        return loaded_from_disk(self, *args, **kwargs)

    monkeypatch.setattr(Array, "loaded_from_disk", counting)
    return reads


def test_lc_global_quantities(
    lc, rectlcn, ic, default_input_struct_lc, cache, monkeypatch
):
    """Only the requested global quantities are saved, and only they are read."""
    reads = _count_array_reads(monkeypatch)
    lc_some = p21c.run_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        cache=cache,
        regenerate=False,
        include_dvdr_in_tau21=False,
        global_quantities=("brightness_temp", "neutral_fraction"),
    )

    assert set(lc_some.global_quantities) == {
        "brightness_temp",
        "neutral_fraction",
        "log10_mturn_acg",
        "log10_mturn_mcg",
    }
    for name, values in lc_some.global_quantities.items():
        np.testing.assert_allclose(
            values, lc.global_quantities[name], rtol=1e-6, err_msg=name
        )
    np.testing.assert_allclose(
        lc_some.lightcones["brightness_temp"],
        lc.lightcones["brightness_temp"],
        rtol=1e-5,
    )

    # Every coeval came from the cache. Of the fields that the lightcone needs only
    # for global quantities, only the requested ones were read, once per node
    # redshift. (The redshift loop also loads each PerturbedField, whatever the
    # global quantities.)
    n_nodes = len(default_input_struct_lc.node_redshifts)
    read_fields = [dataset.split("/")[-1] for dataset in reads]
    assert read_fields.count("brightness_temp") == n_nodes
    assert read_fields.count("neutral_fraction") == n_nodes
    not_requested = {
        name
        for kind in (p21c.EmissivityFields, p21c.IonizedBox, p21c.BrightnessTemp)
        for name in kind.new(default_input_struct_lc, redshift=0).arrays
    } - {"brightness_temp", "neutral_fraction"}
    assert not_requested
    assert not not_requested & set(read_fields)


def test_lc_global_quantities_default(
    lc, rectlcn, ic, default_input_struct_lc, cache, monkeypatch
):
    """By default, the mean of every field is saved, which reads every field."""
    reads = _count_array_reads(monkeypatch)
    lc_all = p21c.run_lightcone(
        lightconer=rectlcn,
        initial_conditions=ic,
        inputs=default_input_struct_lc,
        cache=cache,
        regenerate=False,
        include_dvdr_in_tau21=False,
    )

    fields = p21c.LightCone.get_fields(default_input_struct_lc)
    assert set(lc_all.global_quantities) == set(fields)
    assert set(lc.global_quantities) == set(fields)
    assert {dataset.split("/")[-1] for dataset in reads} == set(fields) - {
        "log10_mturn_acg",
        "log10_mturn_mcg",
    }


def test_lc_global_quantities_bad(rectlcn, default_input_struct_lc):
    with pytest.raises(ValueError, match="global quantities \\['spin_temperature'\\]"):
        p21c.run_lightcone(
            lightconer=rectlcn,
            inputs=default_input_struct_lc,
            global_quantities=("brightness_temp", "spin_temperature"),
        )


def test_lc_global_quantities_resume(
    ic, rectlcn, default_input_struct_lc, tmpdirec, cache
):
    """A lightcone file can only be resumed with its own global quantities."""
    fname = tmpdirec / "lightcone_global_quantities.h5"
    kwargs = {
        "lightconer": rectlcn,
        "initial_conditions": ic,
        "inputs": default_input_struct_lc,
        "lightcone_filename": fname,
        "cache": cache,
        "regenerate": False,
    }
    lc = p21c.run_lightcone(global_quantities=("neutral_fraction",), **kwargs)
    assert set(lc.global_quantities) == {
        "neutral_fraction",
        "log10_mturn_acg",
        "log10_mturn_mcg",
    }

    with pytest.raises(ValueError, match="has the global quantities"):
        p21c.run_lightcone(global_quantities=("brightness_temp",), **kwargs)

    # Without global_quantities, those of the file are used.
    lc2 = p21c.run_lightcone(**kwargs)
    assert set(lc2.global_quantities) == set(lc.global_quantities)

    fname.unlink()
