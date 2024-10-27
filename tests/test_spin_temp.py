import py21cmfast as p21c


def test_first_box(default_user_params, default_cosmo_params):
    """Tests whether the first_box idea works for spin_temp.
    This test was breaking before we set the z_heat_max box to actually get
    the correct dimensions (before it was treated as a dummy).
    """
    initial_conditions = p21c.compute_initial_conditions(
        user_params=default_user_params.clone(HII_DIM=default_user_params.HII_DIM + 1),
        cosmo_params=default_cosmo_params,
        random_seed=1,
    )

    perturbed_field = p21c.perturb_field(
        redshift=29.0, initial_conditions=initial_conditions
    )

    spin_temp = p21c.spin_temperature(
        initial_conditions=initial_conditions, perturbed_field=perturbed_field
    )

    assert spin_temp.redshift == 29.0
