import py21cmfast as p21c


def test_first_box():
    """Tests whether the first_box idea works for spin_temp.
    This test was breaking before we set the z_heat_max box to actually get
    the correct dimensions (before it was treated as a dummy).
    """
    initial_conditions = p21c.initial_conditions(
        user_params=p21c.UserParams(HII_DIM=100, BOX_LEN=100, DIM=200),
        random_seed=1,
        write=False,
    )

    perturbed_field = p21c.perturb_field(
        redshift=29.0, init_boxes=initial_conditions, write=False
    )

    spin_temp = p21c.spin_temperature(
        perturbed_field=perturbed_field, z_heat_max=30.0, write=False,
    )

    assert spin_temp.redshift == 29.0
