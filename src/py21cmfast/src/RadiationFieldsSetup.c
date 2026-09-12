/*
Module for setting up the radiation fields in 21cmFAST.
*/
#include "RadiationFieldsSetup.h"

#include "Constants.h"
#include "cexcept.h"
#include "cosmology.h"
#include "debugging.h"
#include "elec_interp.h"
#include "exceptions.h"
#include "heating_helper_progs.h"
#include "indexing.h"
#include "interp_tables.h"
#include "logger.h"

// This function should construct all the tables which depend on R
// TODO: This function is a dead code because we now do the initialization of the R-dependent arrays
// in setup_shells in python. I still leave it here because maybe we will change our mind in the
// future and we would perfer instead to make the initializations in C
void setup_z_edges(double zp, RadiationFieldsSetup *rad_setup) {
    double R, R_factor;
    double prev_zpp, prev_R;
    int R_ct;

    if (simulation_options_global->HII_DIM == 1) {
        // If HII_DIM=1 (happens when we run_global_evolution), we take a typical cell size
        // of 1.5Mpc, just to for setting the z'' array (note that filtering won't be done on a box
        // with a single cell)
        R = physconst.l_factor * 1.5;
    } else {
        R = physconst.l_factor * simulation_options_global->BOX_LEN /
            (float)simulation_options_global->HII_DIM;
    }
    R_factor = pow(astro_params_global->R_MAX_TS / R, 1 / ((float)astro_params_global->N_STEP_TS));

    for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
        rad_setup->R_values[R_ct] = R;
        if (R_ct == 0) {
            prev_zpp = zp;
            prev_R = 0;
        } else {
            prev_zpp = rad_setup->zpp_edges[R_ct - 1];
            prev_R = rad_setup->R_values[R_ct - 1];
        }
        // cell size
        rad_setup->zpp_edges[R_ct] =
            prev_zpp - (rad_setup->R_values[R_ct] - prev_R) * physconst.cm_per_Mpc / drdz(prev_zpp);
        // average redshift value of shell: z'' + 0.5 * dz''
        rad_setup->zpp_avg[R_ct] = (rad_setup->zpp_edges[R_ct] + prev_zpp) * 0.5;

        R *= R_factor;
    }
    LOG_DEBUG("%d steps R range [%.2e,%.2e] z range [%.2f,%.2f]", R_ct, rad_setup->R_values[0],
              rad_setup->R_values[R_ct - 1], zp, rad_setup->zpp_edges[R_ct - 1]);
}

void calculate_spectral_factors(double zp, RadiationFieldsSetup *rad_setup) {
    double nuprime;
    bool first_radii = true, first_zero = true;
    double trial_zpp;
    int counter, ii;
    int n_pts_radii = 1000;
    double weight = 0.;
    int R_ct, n_ct;
    double zpp, zpp_integrand;

    double sum_lyn_val, sum_lyn_val_MINI;
    double sum_lyLW_val, sum_lyLW_val_MINI;
    double sum_lynto2_val, sum_lynto2_val_MINI;
    double sum_ly2_val, sum_ly2_val_MINI;
    // technically don't need to initialise since it is only used in R_ct > 1 conditional
    double sum_lyn_prev = 0., sum_lyn_prev_MINI = 0.;
    double sum_ly2_prev = 0., sum_ly2_prev_MINI = 0.;
    double sum_lynto2_prev = 0., sum_lynto2_prev_MINI = 0.;
    double prev_zpp = 0;
    for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
        zpp = rad_setup->zpp_avg[R_ct];
        // We need to set up prefactors for how much of Lyman-N radiation is recycled to Lyman-alpha
        sum_lyLW_val = 0.;
        sum_lyLW_val_MINI = 0.;
        sum_lynto2_val = 0.;
        sum_lynto2_val_MINI = 0.;
        sum_ly2_val = 0.;
        sum_ly2_val_MINI = 0.;

        // in case we use LYA_HEATING, we separate the ==2 and >2 cases
        nuprime = nu_n(2) * (1. + zpp) / (1. + zp);
        if (zpp < zmax(zp, 2)) {
            sum_ly2_val = frecycle(2) * spectral_emissivity(nuprime, 0, 2);
            if (astro_options_global->USE_MINI_HALOS) {
                sum_ly2_val_MINI = frecycle(2) * spectral_emissivity(nuprime, 0, 3);

                if (nuprime < physconst.nu_LW_thresh / physconst.nu_ion_HI)
                    nuprime = physconst.nu_LW_thresh / physconst.nu_ion_HI;
                // NOTE: are we comparing nuprime at z' and z'' correctly here?
                //   currently: emitted frequency >= received frequency of next n
                if (nuprime >= nu_n(2 + 1)) continue;

                sum_lyLW_val +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 2);
                sum_lyLW_val_MINI +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 3);
            }
        }

        for (n_ct = NSPEC_MAX; n_ct >= 3; n_ct--) {
            if (zpp > zmax(zp, n_ct)) continue;

            nuprime = nu_n(n_ct) * (1 + zpp) / (1.0 + zp);
            sum_lynto2_val += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 2);
            if (astro_options_global->USE_MINI_HALOS) {
                sum_lynto2_val_MINI += frecycle(n_ct) * spectral_emissivity(nuprime, 0, 3);

                if (nuprime < physconst.nu_LW_thresh / physconst.nu_ion_HI)
                    nuprime = physconst.nu_LW_thresh / physconst.nu_ion_HI;
                if (nuprime >= nu_n(n_ct + 1)) continue;
                sum_lyLW_val +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 2);
                sum_lyLW_val_MINI +=
                    (1. - astro_params_global->F_H2_SHIELD) * spectral_emissivity(nuprime, 2, 3);
            }
        }
        sum_lyn_val = sum_ly2_val + sum_lynto2_val;
        sum_lyn_val_MINI = sum_ly2_val_MINI + sum_lynto2_val_MINI;

        // At the edge of the redshift limit, part of the shell will still contain a contribution
        //   This loop approximates the volume which contains the contribution
        //   and multiplies this by the previous shell's value.
        // This should probably be done separately for each line, since they all face the same issue
        //   It could also be avoided by approximating the integral within each bin better
        //   than taking the value at the bin centre
        if (R_ct > 1 && sum_lyn_val == 0.0 && sum_lyn_prev > 0. && first_radii) {
            for (ii = 0; ii < n_pts_radii; ii++) {
                trial_zpp = prev_zpp + (zpp - prev_zpp) * (float)ii / ((float)n_pts_radii - 1.);
                counter = 0;
                for (n_ct = NSPEC_MAX; n_ct >= 2; n_ct--) {
                    if (trial_zpp > zmax(zp, n_ct)) continue;
                    counter += 1;
                }
                // This is the first sub-radius which has no contribution
                // Use this distance to weigh contribution at previous R
                if (counter == 0 && first_zero) {
                    first_zero = false;
                    weight = (float)ii / (float)n_pts_radii;
                }
            }
            sum_lyn_val = weight * sum_lyn_prev;
            sum_ly2_val = weight * sum_ly2_prev;
            sum_lynto2_val = weight * sum_lynto2_prev;
            if (astro_options_global->USE_MINI_HALOS) {
                sum_lyn_val_MINI = weight * sum_lyn_prev_MINI;
                sum_ly2_val_MINI = weight * sum_ly2_prev_MINI;
                sum_lynto2_val_MINI = weight * sum_lynto2_prev_MINI;
            }
            first_radii = false;
        }
        zpp_integrand = (pow(1 + zp, 2) * (1 + zpp));

        if (astro_options_global->USE_LYA_HEATING) {
            rad_setup->lya_flux_continuum_prefactor[R_ct] = zpp_integrand * sum_ly2_val;
            rad_setup->lya_flux_injected_prefactor[R_ct] = zpp_integrand * sum_lynto2_val;
            LOG_ULTRA_DEBUG("cont %.2e inj %.2e", rad_setup->lya_flux_continuum_prefactor[R_ct],
                            rad_setup->lya_flux_injected_prefactor[R_ct]);
        } else {
            rad_setup->lya_flux_continuum_injected_prefactor[R_ct] = zpp_integrand * sum_lyn_val;
            LOG_ULTRA_DEBUG("z: %.2e R: %.2e int %.2e starlya: %.4e", zpp,
                            rad_setup->R_values[R_ct], zpp_integrand,
                            rad_setup->lya_flux_continuum_injected_prefactor[R_ct]);
        }
        if (astro_options_global->USE_MINI_HALOS) {
            rad_setup->lyw_flux_prefactor[R_ct] = zpp_integrand * sum_lyLW_val;
            rad_setup->lyw_flux_prefactor_MINI[R_ct] = zpp_integrand * sum_lyLW_val_MINI;
            LOG_ULTRA_DEBUG("LW: %.2e LWmini: %.2e", rad_setup->lyw_flux_prefactor[R_ct],
                            rad_setup->lyw_flux_prefactor_MINI[R_ct]);
            if (astro_options_global->USE_LYA_HEATING) {
                rad_setup->lya_flux_continuum_prefactor_MINI[R_ct] =
                    zpp_integrand * sum_ly2_val_MINI;
                rad_setup->lya_flux_injected_prefactor_MINI[R_ct] =
                    zpp_integrand * sum_lynto2_val_MINI;
                LOG_ULTRA_DEBUG("cont mini %.2e inj mini %.2e",
                                rad_setup->lya_flux_continuum_prefactor_MINI[R_ct],
                                rad_setup->lya_flux_injected_prefactor_MINI[R_ct]);
            } else {
                rad_setup->lya_flux_continuum_injected_prefactor_MINI[R_ct] =
                    zpp_integrand * sum_lyn_val_MINI;
                LOG_ULTRA_DEBUG("starmini: %.2e",
                                rad_setup->lya_flux_continuum_injected_prefactor_MINI[R_ct]);
            }
        }

        sum_lyn_prev = sum_lyn_val;
        sum_lyn_prev_MINI = sum_lyn_val_MINI;
        sum_ly2_prev = sum_ly2_val;
        sum_ly2_prev_MINI = sum_ly2_val_MINI;
        sum_lynto2_prev = sum_lynto2_val;
        sum_lynto2_prev_MINI = sum_lynto2_val_MINI;
        prev_zpp = zpp;
    }
}

// construct the [x_e][R_ct] tables
// NOTE: Frequency integrals are based on PREVIOUS x_e_ave
//   The x_e tables are not regular, hence the precomputation of indices/interp points
void fill_freqint_tables(double zp, RadiationFieldsSetup *rad_setup, ScalingConstants *sc) {
    double lower_int_limit;
    int x_e_ct, R_ct;
    int num_R = astro_params_global->N_STEP_TS;

#pragma omp parallel private(R_ct, x_e_ct, lower_int_limit) \
    num_threads(simulation_options_global -> N_THREADS)
    {
#pragma omp for
        // In TauX we integrate Nion from zpp to zp using the LW turnover mass at zp (predending its
        // at zpp)
        //   Calculated from the average smoothed zp grid (from previous LW field) at radius R
        // NOTE: The one difference currently between the halobox and density field options is the
        // weighting of the average
        //   density -> volume weighted cell average || halo -> halo weighted average
        for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
            // TODO: At the moment, inhomogeneous reionization feedback cannot be accounted in
            // SpinTemperatureBox.c,
            //      see https://github.com/21cmfast/21cmFAST/issues/470. Thus, we use the
            //      homogeneous (feedback-free) ACG turnover mass. It is important to remember to
            //      fix this when issue #470 is fixed!
            if (astro_options_global->USE_MINI_HALOS) {
                lower_int_limit =
                    fmax(nu_tau_one_MINI(zp, rad_setup->zpp_avg[R_ct], rad_setup->x_e_ave_zp,
                                         rad_setup->Q_HI_zp, log10(sc->mturn_acg_homogeneous),
                                         rad_setup->ave_log10_MturnLW[R_ct], sc),
                         (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz);
            } else {
                lower_int_limit =
                    fmax(nu_tau_one(zp, rad_setup->zpp_avg[R_ct], rad_setup->x_e_ave_zp,
                                    rad_setup->Q_HI_zp, log10(sc->mturn_acg_homogeneous), sc),
                         (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz);
            }
            // set up frequency integral table for later interpolation for the cell's x_e value
            for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++) {
                rad_setup->freq_int_heat_tbl[freq_index(x_e_ct, R_ct, num_R)] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                rad_setup->freq_int_ion_tbl[freq_index(x_e_ct, R_ct, num_R)] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
                rad_setup->freq_int_lya_tbl[freq_index(x_e_ct, R_ct, num_R)] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);

                // we store these to avoid calculating them in the box_ct loop
                if (x_e_ct > 0) {
                    rad_setup->freq_int_heat_tbl_diff[freq_index(x_e_ct - 1, R_ct, num_R)] =
                        rad_setup->freq_int_heat_tbl[freq_index(x_e_ct, R_ct, num_R)] -
                        rad_setup->freq_int_heat_tbl[freq_index(x_e_ct - 1, R_ct, num_R)];
                    rad_setup->freq_int_ion_tbl_diff[freq_index(x_e_ct - 1, R_ct, num_R)] =
                        rad_setup->freq_int_ion_tbl[freq_index(x_e_ct, R_ct, num_R)] -
                        rad_setup->freq_int_ion_tbl[freq_index(x_e_ct - 1, R_ct, num_R)];
                    rad_setup->freq_int_lya_tbl_diff[freq_index(x_e_ct - 1, R_ct, num_R)] =
                        rad_setup->freq_int_lya_tbl[freq_index(x_e_ct, R_ct, num_R)] -
                        rad_setup->freq_int_lya_tbl[freq_index(x_e_ct - 1, R_ct, num_R)];
                }
            }
            LOG_ULTRA_DEBUG("Nu Integrals || R_ct %d R %.2e zpp %.2f nu_min %.2e", R_ct,
                            rad_setup->R_values[R_ct], rad_setup->zpp_avg[R_ct], lower_int_limit);
            LOG_ULTRA_DEBUG("heat[x_e=0] %.2e ion[x_e=0] %.2e lya[x_e=0] %.2e",
                            rad_setup->freq_int_heat_tbl[freq_index(0, R_ct, num_R)],
                            rad_setup->freq_int_ion_tbl[freq_index(0, R_ct, num_R)],
                            rad_setup->freq_int_lya_tbl[freq_index(0, R_ct, num_R)]);
        }
// separating the inverse diff loop to prevent a race on different R_ct (shouldn't matter)
#pragma omp for
        for (x_e_ct = 0; x_e_ct < x_int_NXHII - 1; x_e_ct++) {
            rad_setup->inverse_diff[x_e_ct] = 1. / (x_int_XHII[x_e_ct + 1] - x_int_XHII[x_e_ct]);
        }
    }

    for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
        for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++) {
            if (isfinite(rad_setup->freq_int_heat_tbl[freq_index(x_e_ct, R_ct, num_R)]) == 0 ||
                isfinite(rad_setup->freq_int_ion_tbl[freq_index(x_e_ct, R_ct, num_R)]) == 0 ||
                isfinite(rad_setup->freq_int_lya_tbl[freq_index(x_e_ct, R_ct, num_R)]) == 0) {
                LOG_ERROR("One of the frequency interpolation tables has an infinity or a NaN");
                Throw(TableGenerationError);
            }
        }
    }
}

// calculate the global properties used for making the frequency integrals,
//   used for filling factor and NO_LIGHT
int global_reion_properties(double zp, RadiationFieldsSetup *rad_setup) {
    double sum_nion = 0, sum_nion_mini = 0;

    // For a lot of global evolution, this code uses Nion_general. We can replace this with the halo
    // field at the same snapshot, but the nu integrals go from zp to zpp to find the tau = 1
    // barrier so it needs the QHII in a range [zp,zpp]. I want to replace this whole thing with a
    // global history struct but I will need to change the Tau function chain.
    double determine_zpp_max, determine_zpp_min;

    // at z', we need a differenc constant struct
    ScalingConstants sc;
    set_scaling_constants(zp, &sc, false);

    if (uses_hmf_interpolation(matter_options_global->USE_INTERPOLATION_TABLES)) {
        determine_zpp_min = zp * 0.999;
        determine_zpp_max = rad_setup->zpp_avg[astro_params_global->N_STEP_TS - 1] * 1.001;

        // We need the tables for the frequency integrals & mean fixing
        // NOTE: These global tables confuse me, we do ~400 (x50 for mini) integrals to build the
        // table, despite only having
        //   ~100 redshifts. The benefit of interpolating here would only matter if we keep the same
        //   table over subsequent snapshots, which we don't seem to do. The Nion table is used in
        //   nu_tau_one a lot but I think there's a better way to do that
        if (source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL)) {
            /* initialise interpolation of the mean collapse fraction for global reionization.*/
            initialise_Nion_Ts_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                      &sc);
        } else {
            init_FcollTable(determine_zpp_min, determine_zpp_max, true);
        }
    }

    // For consistency between halo and non-halo based, the NO_LIGHT and filling_factor_zp
    //   are based on the expected global Nion. as mentioned above it would be nice to
    //   change this to a saved reionisation/sfrd history from previous snapshots
    // TODO: At the moment, inhomogeneous reionization feedback cannot be accounted in
    // SpinTemperatureBox.c,
    //      see https://github.com/21cmfast/21cmFAST/issues/470. Thus, we use the homogeneous
    //      (feedback-free) ACG turnover mass. It is important to remember to fix this when issue
    //      #470 is fixed!
    sum_nion = EvaluateNionTs(zp, log10(sc.mturn_acg_homogeneous), &sc);
    if (astro_options_global->USE_MINI_HALOS) {
        sum_nion_mini = EvaluateNionTs_MINI(zp, log10(sc.mturn_acg_homogeneous),
                                            rad_setup->ave_log10_MturnLW[0], &sc);
    }

    LOG_DEBUG("nion zp = %.3e (%.3e MINI)", sum_nion, sum_nion_mini);

    double ION_EFF_FACTOR, ION_EFF_FACTOR_MINI = 0.;
    if (source_model_is_mass_dependent(matter_options_global->SOURCE_MODEL)) {
        ION_EFF_FACTOR = astro_params_global->F_STAR10 * astro_params_global->F_ESC10 *
                         astro_params_global->POP2_ION;
        ION_EFF_FACTOR_MINI = astro_params_global->F_STAR7_MINI * astro_params_global->F_ESC7_MINI *
                              astro_params_global->POP3_ION;
    } else {
        // no mini-halos when SOURCE_MODE is mass independent (constant ionization efficiency)
        ION_EFF_FACTOR = astro_params_global->HII_EFF_FACTOR;
    }

    // NOTE: only used without MASS_DEPENDENT_ZETA
    rad_setup->Q_HI_zp = 1 - (ION_EFF_FACTOR * sum_nion + ION_EFF_FACTOR_MINI * sum_nion_mini) /
                                 (1.0 - rad_setup->x_e_ave_zp);

    // Initialise freq tables & prefactors (x_e by R tables)
    fill_freqint_tables(zp, rad_setup, &sc);

    // free the global tables if we used them
    free_global_tables();

    return sum_nion + sum_nion_mini > 1e-15 ? 0 : 1;  // NO_LIGHT returned
}

/*
    This function calculates calculates R-indpendent quantities that are useful for the computation
   of the radiation fields (x-ray heating rate, photoionization rate, lyman alpha flux, etc.). This
   is done by setting the fields in the input rad_setup.
*/
int SetupRadiationFields(float redshift, TsBox *previous_spin_temp,
                         RadiationFieldsSetup *rad_setup) {
    int status;
    Try {  // This Try{} wraps the whole function.
        index_huge box_ct;

        calculate_spectral_factors(redshift, rad_setup);

        double x_e_ave_p = 0.0;
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
        {
#pragma omp for reduction(+ : x_e_ave_p)
            for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
                x_e_ave_p += previous_spin_temp->xray_ionised_fraction[box_ct];
            }
        }
        rad_setup->x_e_ave_zp = x_e_ave_p / (float)HII_TOT_NUM_PIXELS;
        LOG_DEBUG("Prev Box: x_e_ave %.3e", rad_setup->x_e_ave_zp);

        // this should initialise and use the global tables (given box average turnovers)
        //   and use them to give: Filling factor at zp (only used for !MASS_DEPENDENT_ZETA to get
        //   ion_eff) global SFRD at each filter radius (numerator of ST_over_PS factor)

        rad_setup->NO_LIGHT = global_reion_properties(redshift, rad_setup);

#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
        {
            float xHII_call;
#pragma omp for
            for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
                xHII_call = previous_spin_temp->xray_ionised_fraction[box_ct];
                // Check if ionized fraction is within boundaries; if not, adjust to be within
                if (xHII_call > x_int_XHII[x_int_NXHII - 1] * 0.999) {
                    xHII_call = x_int_XHII[x_int_NXHII - 1] * 0.999;
                } else if (xHII_call < x_int_XHII[0]) {
                    xHII_call = 1.001 * x_int_XHII[0];
                }
                // these are the index and interpolation term, moved outside the R loop and stored
                // to not calculate them R times
                rad_setup->m_xHII_low_box[box_ct] = locate_xHII_index(xHII_call);
                rad_setup->inverse_val_box[box_ct] =
                    (xHII_call - x_int_XHII[rad_setup->m_xHII_low_box[box_ct]]) *
                    rad_setup->inverse_diff[rad_setup->m_xHII_low_box[box_ct]];
            }
        }
    }  // End of try
    Catch(status) { return (status); }
    return (0);
}
