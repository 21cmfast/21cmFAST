"""Contains the tests for the Perturbation algorithm (Linear, Zel'dovich, 2LPT).

Including perturbation of galaxy properties
"""

import numpy as np
import pytest

from py21cmfast import (
    InitialConditions,
    compute_halo_grid,
    perturb_field,
)
from py21cmfast.wrapper import cfuncs as cf


class TestPerturb:
    """Tests regarding the perturbation algorithms."""

    @pytest.fixture(scope="class")
    def test_pt_z(self):
        """Set redshift at which to test the 2LPT."""
        return 8.0

    @pytest.fixture(scope="class")
    def inputs_low(self, default_input_struct_ts):
        """Parameters for 2LPT tests."""
        # using 3-1 ratio for testing
        return default_input_struct_ts.evolve_input_structs(
            DIM=12,
            HII_DIM=4,
            BOX_LEN=8,
            USE_HALO_FIELD=True,
            FIXED_HALO_GRIDS=True,
            PERTURB_ON_HIGH_RES=False,
            R_BUBBLE_MAX=1.0,
        )

    @pytest.fixture(scope="class")
    def inputs_zel(self, inputs_low):
        """Parameters for Zel'dovich test."""
        return inputs_low.evolve_input_structs(
            PERTURB_ALGORITHM="ZELDOVICH",
        )

    @pytest.fixture(scope="class")
    def inputs_linear(self, inputs_low):
        """Parameters for Linear test."""
        return inputs_low.evolve_input_structs(
            PERTURB_ALGORITHM="LINEAR",
        )

    def get_fake_ics(self, inputs, test_pt_z):
        """Make an IC instance for the testing.

        These are inconsistent and strange values for real ICS but
        very trackable.
        """
        ics = InitialConditions.new(inputs=inputs)
        d_z = cf.get_growth_factor(inputs=inputs, redshift=test_pt_z)
        d_z_i = cf.get_growth_factor(
            inputs=inputs, redshift=inputs.simulation_options.INITIAL_REDSHIFT
        )

        res_fac = int(inputs.simulation_options.HIRES_TO_LOWRES_FACTOR)
        lo_dim = inputs.simulation_options.HII_DIM
        hi_dim = inputs.simulation_options.DIM
        fac_1lpt = inputs.simulation_options.cell_size / (d_z - d_z_i)
        fac_2lpt = inputs.simulation_options.cell_size / (
            (-3.0 / 7.0) * (d_z**2 - d_z_i**2)
        )
        for name, array in ics.arrays.items():
            setattr(ics, name, array.initialize().computed())

        # setup the velocities
        # NOTE: IC velocities are in Mpc
        if not inputs.matter_options.PERTURB_ON_HIGH_RES:
            fake_v = np.ones_like(ics.get("lowres_vx"))
            ics.set("lowres_vx", 0 * fake_v)
            ics.set("lowres_vy", fac_1lpt * fake_v)
            ics.set("lowres_vz", 0 * fake_v)
            if inputs.matter_options.PERTURB_ALGORITHM == "2LPT":
                ics.set("lowres_vx_2LPT", 0 * fake_v)
                ics.set("lowres_vy_2LPT", 0 * fake_v)
                ics.set("lowres_vz_2LPT", fac_2lpt * fake_v)
        else:
            fake_v = np.ones_like(ics.get("hires_vx"))
            ics.set("hires_vx", 0 * fake_v)
            ics.set("hires_vy", -fac_1lpt * fake_v)
            ics.set("hires_vz", 0 * fake_v)
            if inputs.matter_options.PERTURB_ALGORITHM == "2LPT":
                ics.set("hires_vx_2LPT", 0 * fake_v)
                ics.set("hires_vy_2LPT", 0 * fake_v)
                ics.set("hires_vz_2LPT", -fac_2lpt * fake_v)

        # set some densities that can be easily tracked
        d_lo = np.zeros_like(ics.get("lowres_density"))
        d_lo[0, 0, 0] = 1
        d_lo[lo_dim // 2, lo_dim // 2, lo_dim // 2] = -1
        ics.set("lowres_density", d_lo)
        # make similar hires densities
        d_hi = np.zeros_like(ics.get("hires_density"))
        d_hi[0, 0, 0] = res_fac**3
        d_hi[hi_dim // 2, hi_dim // 2, hi_dim // 2] = -(res_fac**3)
        ics.set("hires_density", d_hi)

        return ics

    @pytest.mark.parametrize("inputs", ["inputs_low", "inputs_zel"])
    def test_lowres_perturb(self, inputs, test_pt_z, request):
        """Tests low-resolution perturbation."""
        inputs = request.getfixturevalue(inputs)
        ics = self.get_fake_ics(inputs, test_pt_z)
        z_d = (
            test_pt_z
            if inputs.matter_options.PERTURB_ALGORITHM == "LINEAR"
            else inputs.simulation_options.INITIAL_REDSHIFT
        )
        roll_var = {
            "LINEAR": (0, 0, 0),
            "ZELDOVICH": (0, 1, 0),
            "2LPT": (0, 1, -1),
        }[inputs.matter_options.PERTURB_ALGORITHM]
        d_z = cf.get_growth_factor(inputs=inputs, redshift=z_d)

        # since we set the IC y,z velocities to move exactly one HII_DIM cell,
        # The expected density is be the IC density rolled by one cell on two axes
        expected_dens = np.roll(ics.get("lowres_density"), roll_var, (0, 1, 2))
        expected_dens *= d_z
        pt = perturb_field(
            initial_conditions=ics,
            redshift=test_pt_z,
            regenerate=True,
            write=False,
        )
        np.testing.assert_allclose(pt.get("density"), expected_dens, atol=1e-3)

    @pytest.mark.skip(
        reason="aliasing in downsampling makes hires 2lpt unit tests difficult"
    )
    def test_hires_perturb(self, inputs_hi, test_pt_z):
        """Tests the high resolution perturbation."""
        ics = self.get_fake_ics(inputs_hi, test_pt_z)
        expected_dens = np.roll(ics.get("lowres_density"), (0, -1, 1), (0, 1, 2))
        d_z_i = cf.get_growth_factor(inputs=inputs_hi, redshift=test_pt_z)
        expected_dens *= d_z_i
        pt = perturb_field(
            initial_conditions=ics,
            redshift=test_pt_z,
            regenerate=True,
            write=False,
        )
        np.testing.assert_allclose(pt.get("density"), expected_dens, atol=1e-3)

    # TODO: include minihalo properties
    # TODO: include linear (for some reason)
    @pytest.mark.parametrize("inputs", ["inputs_low", "inputs_zel"])
    def test_hb_perturb(self, inputs, test_pt_z, request):
        """Tests the halo property perturbation."""
        inputs = request.getfixturevalue(inputs)
        ics = self.get_fake_ics(inputs, test_pt_z)
        hbox = compute_halo_grid(
            redshift=test_pt_z,
            initial_conditions=ics,
            inputs=inputs,
        )
        cell_radius = 0.620350491 * (
            inputs.simulation_options.BOX_LEN / inputs.simulation_options.HII_DIM
        )
        d_z = cf.get_growth_factor(
            inputs=inputs,
            redshift=test_pt_z,
        )
        roll_var = {
            "LINEAR": (0, 0, 0),
            "ZELDOVICH": (0, 1, 0),
            "2LPT": (0, 1, -1),
        }[inputs.matter_options.PERTURB_ALGORITHM]
        dens = np.roll(ics.get("lowres_density"), roll_var, (0, 1, 2)) * d_z
        mt_grid = np.full_like(dens, inputs.astro_params.M_TURN)

        prefac_sfr = (
            inputs.cosmo_params.cosmo.critical_density(0).to("Msun Mpc-3").value
            * inputs.astro_params.cdict["F_STAR10"]
            * inputs.cosmo_params.OMb
            * inputs.cosmo_params.cosmo.H(test_pt_z).to("s-1").value
            / inputs.astro_params.t_STAR
        )
        prefac_nion = (
            inputs.cosmo_params.cosmo.critical_density(0).to("Msun Mpc-3").value
            * inputs.astro_params.cdict["F_STAR10"]
            * inputs.cosmo_params.OMb
            * inputs.astro_params.cdict["F_ESC10"]
            * inputs.astro_params.cdict["POP2_ION"]
        )
        prefac_xray = (
            inputs.cosmo_params.cosmo.critical_density(0).to("Msun Mpc-3").value
            * inputs.cosmo_params.OMm
        )
        integral_sfrd, _ = cf.evaluate_SFRD_cond(
            inputs=inputs,
            redshift=test_pt_z,
            radius=cell_radius,
            densities=dens,
            log10mturns=mt_grid,
        )
        integral_sfrd *= prefac_sfr

        integral_nion, _ = cf.evaluate_Nion_cond(
            inputs=inputs,
            redshift=test_pt_z,
            radius=cell_radius,
            densities=dens,
            l10mturns_acg=mt_grid,
            l10mturns_mcg=mt_grid,
        )
        integral_nion *= prefac_nion

        integral_xray = cf.evaluate_Xray_cond(
            inputs=inputs,
            redshift=test_pt_z,
            radius=cell_radius,
            densities=dens,
            log10mturns=mt_grid,
        )
        integral_xray *= prefac_xray

        rtol = 1e-2
        np.testing.assert_allclose(hbox.get("halo_sfr"), integral_sfrd, rtol=rtol)
        np.testing.assert_allclose(hbox.get("n_ion"), integral_nion, rtol=rtol)
        np.testing.assert_allclose(hbox.get("halo_xray"), integral_xray, rtol=rtol)
