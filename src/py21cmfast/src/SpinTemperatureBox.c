// Re-write of find_HII_bubbles.c for being accessible within the MCMC
#include "SpinTemperatureBox.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "debugging.h"
#include "dft.h"
#include "elec_interp.h"
#include "exceptions.h"
#include "filtering.h"
#include "heating_helper_progs.h"
#include "hmf.h"
#include "indexing.h"
#include "interp_tables.h"
#include "logger.h"
#include "thermochem.h"

/* Maximum allowed value for the kinetic temperature. Useful to set to avoid some spurious behaviour
 when the code is run with redshift poor resolution and very high X-ray heating efficiency */
#define MAX_TK (float)5e4

void ts_main(float redshift, float prev_redshift, float perturbed_field_redshift, short cleanup,
             PerturbedField *perturbed_field, XraySourceBox *source_box, TsBox *previous_spin_temp,
             InitialConditions *ini_boxes, TsBox *this_spin_temp);

// Global arrays which have yet to be moved to structs
// R x box arrays
float **delNL0, **log10_Mcrit_LW;

// arrays for R-dependent prefactors
double *dstarlya_dt_prefactor, *dstarlya_dt_prefactor_MINI;
double *dstarlyLW_dt_prefactor, *dstarlyLW_dt_prefactor_MINI;
double *dstarlya_cont_dt_prefactor, *dstarlya_inj_dt_prefactor;
double *dstarlya_cont_dt_prefactor_MINI, *dstarlya_inj_dt_prefactor_MINI;

// boxes to hold stellar fraction integrals (Fcoll or SFRD)
float *del_fcoll_Rct, *del_fcoll_Rct_MINI;

// radiative term boxes which are summed over R
double *dxheat_dt_box, *dxion_source_dt_box, *dxlya_dt_box, *dstarlya_dt_box;
double *dstarlyLW_dt_box, *dstarlyLW_dt_box_MINI;
double *dstarlya_cont_dt_box, *dstarlya_inj_dt_box;
double *dstarlya_cont_dt_box_MINI, *dstarlya_inj_dt_box_MINI;

// x_e interpolation boxes / arrays (not a RGI)
float *inverse_val_box;
int *m_xHII_low_box;
float *inverse_diff;

// interpolation tables for the heating/ionisation integrals
double **freq_int_heat_tbl, **freq_int_ion_tbl, **freq_int_lya_tbl, **freq_int_heat_tbl_diff;
double **freq_int_ion_tbl_diff, **freq_int_lya_tbl_diff;

// R-dependent arrays which are set once
double *R_values, *dzpp_list, *dtdz_list, *zpp_growth, *zpp_for_evolve_list, *zpp_edge;
double *sigma_min, *sigma_max, *M_max_R, *M_min_R;
double *min_densities, *max_densities;

// Arrays which specify the Radii, distances, redshifts of each shell
//   They will have a global instance since they are reused a lot
//   However it is worth considering passing them into functions instead
//  struct radii_spec{
//      double *R_values; //Radii edge of each shell
//      double *zpp_edge; //redshift of the shell edge
//      double *zpp_cen; //middle redshift of cell (z_inner + z_outer)/2
//      double *dzpp_list; //redshift difference between inner and outer edge
//      double *dtdz_list; //dtdz at zpp_cen
//      double *zpp_growth; //linear growth factor D(z) at zpp_cen
//  }
//  struct radii_spec r_s;

bool TsInterpArraysInitialised = false;

// a debug flag for printing results from a single cell without passing cell number to the functions
static int debug_printed;

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int ComputeTsBox(float redshift, float prev_redshift, float perturbed_field_redshift, short cleanup,
                 PerturbedField *perturbed_field, XraySourceBox *source_box,
                 TsBox *previous_spin_temp, InitialConditions *ini_boxes, TsBox *this_spin_temp) {
    int status;
    Try {  // This Try{} wraps the whole function.
        LOG_DEBUG("Spintemp input values:");
        LOG_DEBUG("redshift=%f, prev_redshift=%f perturbed_field_redshift=%f", redshift,
                  prev_redshift, perturbed_field_redshift);

#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
        writeSimulationOptions(simulation_options_global);
        writeCosmoParams(cosmo_params_global);
        writeAstroParams(astro_params_global);
        writeAstroOptions(astro_options_global);
#endif

        // Makes the parameter structs visible to a variety of functions/macros
        // Do each time to avoid Python garbage collection issues

        omp_set_num_threads(simulation_options_global->N_THREADS);
        debug_printed = 0;

        ts_main(redshift, prev_redshift, perturbed_field_redshift, cleanup, perturbed_field,
                source_box, previous_spin_temp, ini_boxes, this_spin_temp);

    }  // End of try
    Catch(status) { return (status); }
    return (0);
}

void alloc_global_arrays() {
    int i;
    // z-edges
    zpp_for_evolve_list = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    zpp_growth = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    zpp_edge = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    dzpp_list = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    dtdz_list = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    R_values = calloc(astro_params_global->N_STEP_TS, sizeof(double));

    sigma_min = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    sigma_max = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    M_min_R = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    M_max_R = calloc(astro_params_global->N_STEP_TS, sizeof(double));

    // frequency integral tables
    freq_int_heat_tbl = (double **)calloc(x_int_NXHII, sizeof(double *));
    freq_int_ion_tbl = (double **)calloc(x_int_NXHII, sizeof(double *));
    freq_int_lya_tbl = (double **)calloc(x_int_NXHII, sizeof(double *));
    freq_int_heat_tbl_diff = (double **)calloc(x_int_NXHII, sizeof(double *));
    freq_int_ion_tbl_diff = (double **)calloc(x_int_NXHII, sizeof(double *));
    freq_int_lya_tbl_diff = (double **)calloc(x_int_NXHII, sizeof(double *));
    for (i = 0; i < x_int_NXHII; i++) {
        freq_int_heat_tbl[i] = (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
        freq_int_ion_tbl[i] = (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
        freq_int_lya_tbl[i] = (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
        freq_int_heat_tbl_diff[i] =
            (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
        freq_int_ion_tbl_diff[i] = (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
        freq_int_lya_tbl_diff[i] = (double *)calloc(astro_params_global->N_STEP_TS, sizeof(double));
    }
    inverse_diff = (float *)calloc(x_int_NXHII, sizeof(float));
    // actual heating term boxes
    if (astro_options_global->USE_X_RAY_HEATING) {
        dxheat_dt_box = (double *)calloc(HII_TOT_NUM_PIXELS, sizeof(double));
    }
    dxion_source_dt_box = (double *)calloc(HII_TOT_NUM_PIXELS, sizeof(double));
    dxlya_dt_box = (double *)calloc(HII_TOT_NUM_PIXELS, sizeof(double));
    dstarlya_dt_box = (double *)calloc(HII_TOT_NUM_PIXELS, sizeof(double));
    if (astro_options_global->USE_LYA_HEATING) {
        dstarlya_cont_dt_box = (double *)calloc(HII_TOT_NUM_PIXELS, sizeof(double));
        dstarlya_inj_dt_box = (double *)calloc(HII_TOT_NUM_PIXELS, sizeof(double));
    }
    if (astro_options_global->USE_MINI_HALOS) {
        dstarlyLW_dt_box = (double *)calloc(HII_TOT_NUM_PIXELS, sizeof(double));
    }

    // spectral stuff
    dstarlya_dt_prefactor = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    if (astro_options_global->USE_LYA_HEATING) {
        dstarlya_cont_dt_prefactor = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        dstarlya_inj_dt_prefactor = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    }

    if (astro_options_global->USE_MINI_HALOS) {
        dstarlya_dt_prefactor_MINI = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        dstarlyLW_dt_prefactor = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        dstarlyLW_dt_prefactor_MINI = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        if (astro_options_global->USE_LYA_HEATING) {
            dstarlya_cont_dt_prefactor_MINI =
                calloc(astro_params_global->N_STEP_TS, sizeof(double));
            dstarlya_inj_dt_prefactor_MINI = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        }
    }

    // Nonhalo stuff
    int num_R_boxes = matter_options_global->MINIMIZE_MEMORY ? 1 : astro_params_global->N_STEP_TS;
    if (matter_options_global->SOURCE_MODEL < 2) {
        delNL0 = (float **)calloc(num_R_boxes, sizeof(float *));
        for (i = 0; i < num_R_boxes; i++) {
            delNL0[i] = (float *)calloc((float)HII_TOT_NUM_PIXELS, sizeof(float));
        }
        if (astro_options_global->USE_MINI_HALOS) {
            log10_Mcrit_LW = (float **)calloc(num_R_boxes, sizeof(float *));
            for (i = 0; i < num_R_boxes; i++) {
                log10_Mcrit_LW[i] = (float *)calloc(HII_TOT_NUM_PIXELS, sizeof(float));
            }
        }

        del_fcoll_Rct = calloc(HII_TOT_NUM_PIXELS, sizeof(float));
        if (astro_options_global->USE_MINI_HALOS) {
            del_fcoll_Rct_MINI = calloc(HII_TOT_NUM_PIXELS, sizeof(float));
        }

        min_densities = calloc(astro_params_global->N_STEP_TS, sizeof(double));
        max_densities = calloc(astro_params_global->N_STEP_TS, sizeof(double));
    }

    // helpers for the interpolation
    // NOTE: The frequency integrals are tables regardless of the flag
    m_xHII_low_box = (int *)calloc(HII_TOT_NUM_PIXELS, sizeof(int));
    inverse_val_box = (float *)calloc(HII_TOT_NUM_PIXELS, sizeof(float));

    TsInterpArraysInitialised = true;
}

void free_ts_global_arrays() {
    // free external interpolation tables that are initialized in heating_healper_progs.c
    destruct_heat();

    int i;
    // frequency integrals
    for (i = 0; i < x_int_NXHII; i++) {
        free(freq_int_heat_tbl[i]);
        free(freq_int_ion_tbl[i]);
        free(freq_int_lya_tbl[i]);
        free(freq_int_heat_tbl_diff[i]);
        free(freq_int_ion_tbl_diff[i]);
        free(freq_int_lya_tbl_diff[i]);
    }
    free(freq_int_heat_tbl);
    free(freq_int_ion_tbl);
    free(freq_int_lya_tbl);
    free(freq_int_heat_tbl_diff);
    free(freq_int_ion_tbl_diff);
    free(freq_int_lya_tbl_diff);
    free(inverse_diff);

    // z- edges
    free(zpp_growth);
    free(zpp_edge);
    free(zpp_for_evolve_list);
    free(dzpp_list);
    free(dtdz_list);
    free(R_values);

    free(sigma_min);
    free(sigma_max);
    free(M_min_R);
    free(M_max_R);

    // spectral
    free(dstarlya_dt_prefactor);
    if (astro_options_global->USE_LYA_HEATING) {
        free(dstarlya_cont_dt_prefactor);
        free(dstarlya_inj_dt_prefactor);
    }
    if (astro_options_global->USE_MINI_HALOS) {
        free(dstarlya_dt_prefactor_MINI);
        free(dstarlyLW_dt_prefactor);
        free(dstarlyLW_dt_prefactor_MINI);
        if (astro_options_global->USE_LYA_HEATING) {
            free(dstarlya_inj_dt_prefactor_MINI);
            free(dstarlya_cont_dt_prefactor_MINI);
        }
    }

    // boxes
    if (astro_options_global->USE_X_RAY_HEATING) {
        free(dxheat_dt_box);
    }
    free(dxion_source_dt_box);
    free(dxlya_dt_box);
    free(dstarlya_dt_box);
    if (astro_options_global->USE_MINI_HALOS) {
        free(dstarlyLW_dt_box);
    }
    if (astro_options_global->USE_LYA_HEATING) {
        free(dstarlya_cont_dt_box);
        free(dstarlya_inj_dt_box);
    }

    // interpolation helpers
    free(m_xHII_low_box);
    free(inverse_val_box);

    // interp tables
    int num_R_boxes = matter_options_global->MINIMIZE_MEMORY ? 1 : astro_params_global->N_STEP_TS;
    if (matter_options_global->SOURCE_MODEL < 2) {
        for (i = 0; i < num_R_boxes; i++) {
            free(delNL0[i]);
        }
        free(delNL0);

        if (astro_options_global->USE_MINI_HALOS) {
            for (i = 0; i < num_R_boxes; i++) {
                free(log10_Mcrit_LW[i]);
            }
            free(log10_Mcrit_LW);
        }

        free(del_fcoll_Rct);
        if (astro_options_global->USE_MINI_HALOS) {
            free(del_fcoll_Rct_MINI);
        }

        free(min_densities);
        free(max_densities);
    }

    TsInterpArraysInitialised = false;
}

// This function should construct all the tables which depend on R
void setup_z_edges(double zp) {
    double R, R_factor;
    double zpp, prev_zpp, prev_R;
    double dzpp_for_evolve;
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
        R_values[R_ct] = R;
        if (R_ct == 0) {
            prev_zpp = zp;
            prev_R = 0;
        } else {
            prev_zpp = zpp_edge[R_ct - 1];
            prev_R = R_values[R_ct - 1];
        }

        // cell size
        zpp_edge[R_ct] =
            prev_zpp - (R_values[R_ct] - prev_R) * physconst.cm_per_Mpc / drdz(prev_zpp);
        // average redshift value of shell: z'' + 0.5 * dz''
        zpp = (zpp_edge[R_ct] + prev_zpp) * 0.5;

        zpp_for_evolve_list[R_ct] = zpp;
        if (R_ct == 0) {
            dzpp_for_evolve = zp - zpp_edge[0];
        } else {
            dzpp_for_evolve = zpp_edge[R_ct - 1] - zpp_edge[R_ct];
        }
        zpp_growth[R_ct] = dicke(zpp);      // growth factors
        dzpp_list[R_ct] = dzpp_for_evolve;  // z bin width
        dtdz_list[R_ct] = dtdz(zpp);        // dt/dz''

        M_min_R[R_ct] = minimum_source_mass(zpp_for_evolve_list[R_ct], true);
        M_max_R[R_ct] = RtoM(R_values[R_ct]);

        R *= R_factor;
    }
    LOG_DEBUG("%d steps R range [%.2e,%.2e] z range [%.2f,%.2f]", R_ct, R_values[0],
              R_values[R_ct - 1], zp, zpp_edge[R_ct - 1]);
}

void calculate_spectral_factors(double zp) {
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
        zpp = zpp_for_evolve_list[R_ct];
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
        dstarlya_dt_prefactor[R_ct] = zpp_integrand * sum_lyn_val;
        LOG_ULTRA_DEBUG("z: %.2e R: %.2e int %.2e starlya: %.4e", zpp, R_values[R_ct],
                        zpp_integrand, dstarlya_dt_prefactor[R_ct]);

        if (astro_options_global->USE_LYA_HEATING) {
            dstarlya_cont_dt_prefactor[R_ct] = zpp_integrand * sum_ly2_val;
            dstarlya_inj_dt_prefactor[R_ct] = zpp_integrand * sum_lynto2_val;
            LOG_ULTRA_DEBUG("cont %.2e inj %.2e", dstarlya_cont_dt_prefactor[R_ct],
                            dstarlya_inj_dt_prefactor[R_ct]);
        }
        if (astro_options_global->USE_MINI_HALOS) {
            dstarlya_dt_prefactor_MINI[R_ct] = zpp_integrand * sum_lyn_val_MINI;
            dstarlyLW_dt_prefactor[R_ct] = zpp_integrand * sum_lyLW_val;
            dstarlyLW_dt_prefactor_MINI[R_ct] = zpp_integrand * sum_lyLW_val_MINI;
            if (astro_options_global->USE_LYA_HEATING) {
                dstarlya_cont_dt_prefactor_MINI[R_ct] = zpp_integrand * sum_ly2_val_MINI;
                dstarlya_inj_dt_prefactor_MINI[R_ct] = zpp_integrand * sum_lynto2_val_MINI;
            }

            LOG_ULTRA_DEBUG("starmini: %.2e LW: %.2e LWmini: %.2e",
                            dstarlya_dt_prefactor_MINI[R_ct], dstarlyLW_dt_prefactor[R_ct],
                            dstarlyLW_dt_prefactor_MINI[R_ct]);
            LOG_ULTRA_DEBUG("cont mini %.2e inj mini %.2e", dstarlya_cont_dt_prefactor_MINI[R_ct],
                            dstarlya_inj_dt_prefactor_MINI[R_ct]);
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

// fill fftwf boxes, do the r2c transform and normalise
void prepare_filter_boxes(double redshift, float *input_dens, float *input_vcb, float *input_j21,
                          fftwf_complex *output_dens, fftwf_complex *output_LW) {
    int i, j, k;
    unsigned long long int ct, index_f;
    double curr_vcb, curr_j21, M_buf;
    int box_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};

// NOTE: Meraxes just applies a pointer cast box = (fftwf_complex *) input. Figure out why this
// works,
//       They pad the input by a factor of 2 to cover the complex part, but from the type I thought
//       it would be stored [(r,c),(r,c)...] Not [(r,r,r,r....),(c,c,c....)] so the alignment should
//       be wrong, right?
#pragma omp parallel for private(i, j, k, ct, index_f) \
    num_threads(simulation_options_global->N_THREADS) collapse(3)
    for (i = 0; i < box_dim[0]; i++) {
        for (j = 0; j < box_dim[1]; j++) {
            for (k = 0; k < box_dim[2]; k++) {
                ct = grid_index_general(i, j, k, box_dim);
                index_f = grid_index_fftw_r(i, j, k, box_dim);
                *((float *)output_dens + index_f) = input_dens[ct];
            }
        }
    }
    // Transform unfiltered box to k-space to prepare for filtering
    dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                 HII_D_PARA, simulation_options_global->N_THREADS, output_dens);
#pragma omp parallel for num_threads(simulation_options_global->N_THREADS)
    for (ct = 0; ct < HII_KSPACE_NUM_PIXELS; ct++) {
        output_dens[ct] /= (float)HII_TOT_NUM_PIXELS;
    }

    if (astro_options_global->USE_MINI_HALOS) {
        curr_vcb = astro_options_global->FIX_VCB_AVG ? astro_params_global->FIXED_VAVG : 0;
#pragma omp parallel for firstprivate(curr_vcb) private(i, j, k, curr_j21, M_buf, ct, index_f) \
    num_threads(simulation_options_global->N_THREADS) collapse(3)
        for (i = 0; i < box_dim[0]; i++) {
            for (j = 0; j < box_dim[1]; j++) {
                for (k = 0; k < box_dim[2]; k++) {
                    ct = grid_index_general(i, j, k, box_dim);
                    index_f = grid_index_fftw_r(i, j, k, box_dim);
                    if (!astro_options_global->FIX_VCB_AVG &&
                        matter_options_global->USE_RELATIVE_VELOCITIES) {
                        curr_vcb = input_vcb[ct];
                    }
                    curr_j21 = input_j21[ct];
                    // NOTE: we don't use reionization_feedback here, I assume it wouldn't do much
                    // but it's inconsistent
                    M_buf = lyman_werner_threshold(redshift, curr_j21, curr_vcb);
                    M_buf = fmax(M_buf, astro_params_global->M_TURN);
                    *((float *)output_LW + index_f) = log10(M_buf);
                }
            }
        }
        // Transform unfiltered box to k-space to prepare for filtering
        dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, output_LW);
#pragma omp parallel for num_threads(simulation_options_global->N_THREADS)
        for (ct = 0; ct < HII_KSPACE_NUM_PIXELS; ct++) {
            output_LW[ct] /= (float)HII_TOT_NUM_PIXELS;
        }
    }
}

// fill a box[R_ct][box_ct] array for use in TS by filtering on different scales and storing results
void fill_Rbox_table(float **result, fftwf_complex *unfiltered_box, double *R_array, int n_R,
                     double min_value, double const_factor, double *min_arr, double *average_arr,
                     double *max_arr) {
    // allocate table/grid memory
    int i, j, k, R_ct;
    double R;
    double ave_buffer, min_out_R, max_out_R;
    int box_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};

    fftwf_complex *box =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
    // Smooth the density field, at the same time store the minimum and maximum densities for their
    // usage in the interpolation tables
    LOG_ULTRA_DEBUG("db0");
    for (R_ct = 0; R_ct < n_R; R_ct++) {
        R = R_array[R_ct];
        ave_buffer = 0;
        min_out_R = 1e20;
        max_out_R = -1e20;
        // copy over unfiltered box
        memcpy(box, unfiltered_box, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
        LOG_ULTRA_DEBUG("db1 %d", R_ct);

        // don't filter on cell size
        if (R > physconst.l_factor *
                    (simulation_options_global->BOX_LEN / simulation_options_global->HII_DIM)) {
            filter_box(box, box_dim, astro_options_global->HEAT_FILTER, R, 0.);
        }

        LOG_ULTRA_DEBUG("db2 %d", R_ct);

        // now fft back to real space
        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, box);

        LOG_ULTRA_DEBUG("db3 %d", R_ct);
        // copy over the values
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
        {
            float curr;
            unsigned long long int index_r, index_f;
#pragma omp for reduction(+ : ave_buffer) reduction(max : max_out_R) reduction(min : min_out_R)
            for (i = 0; i < box_dim[0]; i++) {
                for (j = 0; j < box_dim[1]; j++) {
                    for (k = 0; k < box_dim[2]; k++) {
                        index_r = grid_index_general(i, j, k, box_dim);
                        index_f = grid_index_fftw_r(i, j, k, box_dim);
                        curr = *((float *)box + index_f);

                        // NOTE: Min value is on the grid BEFORE constant factor
                        //  correct for aliasing in the filtering step
                        if (curr < min_value) {
                            curr = min_value;
                        }

                        // constant factors (i.e linear extrapolation to z=0 for dens.)
                        curr = curr * const_factor;

                        ave_buffer += curr;
                        if (curr < min_out_R) min_out_R = curr;
                        if (curr > max_out_R) max_out_R = curr;
                        result[R_ct][index_r] = curr;
                    }
                }
            }
        }
        average_arr[R_ct] = ave_buffer / HII_TOT_NUM_PIXELS;
        min_arr[R_ct] = min_out_R;
        max_arr[R_ct] = max_out_R;
        LOG_ULTRA_DEBUG("db4 %d", R_ct);
    }
    LOG_ULTRA_DEBUG("db5");
    fftwf_free(box);
}

// NOTE: I've moved this to a function to help in simplicity, it is not clear whether it is faster
//   to do all of one radii at once (more clustered FFT and larger thread blocks) or all of one box
//   (better memory locality)
// TODO: filter speed tests
void one_annular_filter(float *input_box, float *output_box, double R_inner, double R_outer,
                        double *u_avg, double *f_avg) {
    int i, j, k;
    unsigned long long int ct;
    double unfiltered_avg = 0;
    double filtered_avg = 0;
    int box_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};

    fftwf_complex *unfiltered_box =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
    fftwf_complex *filtered_box =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS) \
    reduction(+ : unfiltered_avg)
    {
        float curr_val;
        unsigned long long int index_r, index_f;
#pragma omp for
        for (i = 0; i < box_dim[0]; i++) {
            for (j = 0; j < box_dim[1]; j++) {
                for (k = 0; k < box_dim[2]; k++) {
                    index_r = grid_index_general(i, j, k, box_dim);
                    index_f = grid_index_fftw_r(i, j, k, box_dim);
                    curr_val = input_box[index_r];
                    *((float *)unfiltered_box + index_f) = curr_val;
                    unfiltered_avg += curr_val;
                }
            }
        }
    }
    if (simulation_options_global->HII_DIM >
        1) {  // No need to filter the box if we only have one cell!
        // Transform unfiltered box to k-space to prepare for filtering
        // this would normally only be done once but we're using a different redshift for each R now
        dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, unfiltered_box);

// remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
// Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
        {
#pragma omp for
            for (ct = 0; ct < HII_KSPACE_NUM_PIXELS; ct++) {
                unfiltered_box[ct] /= (float)HII_TOT_NUM_PIXELS;
            }
        }

        // Smooth the density field, at the same time store the minimum and maximum densities for
        // their usage in the interpolation tables copy over unfiltered box
        memcpy(filtered_box, unfiltered_box, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

        // Don't filter on the cell scale
        if (R_inner > 0) {
            filter_box(filtered_box, box_dim, 4, R_inner, R_outer);
        }

        // now fft back to real space
        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, filtered_box);
    }
// copy over the values
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS) \
    reduction(+ : filtered_avg)
    {
        float curr_val;
        unsigned long long int index_f, index_r;
#pragma omp for
        for (i = 0; i < box_dim[0]; i++) {
            for (j = 0; j < box_dim[1]; j++) {
                for (k = 0; k < box_dim[2]; k++) {
                    index_r = grid_index_general(i, j, k, box_dim);
                    index_f = grid_index_fftw_r(i, j, k, box_dim);
                    if (simulation_options_global->HII_DIM > 1) {
                        curr_val = *((float *)filtered_box + index_f);
                    } else {  // Just take the unfiltered box/cell if HII_DIM = 1
                        curr_val = *((float *)unfiltered_box + index_f);
                    }
                    // correct for aliasing in the filtering step
                    if (curr_val < 0.) curr_val = 0.;

                    output_box[index_r] = curr_val;
                    filtered_avg += curr_val;
                }
            }
        }
    }

    unfiltered_avg /= HII_TOT_NUM_PIXELS;
    filtered_avg /= HII_TOT_NUM_PIXELS;

    *u_avg = unfiltered_avg;
    *f_avg = filtered_avg;

    fftwf_free(filtered_box);
    fftwf_free(unfiltered_box);
}

// fill a box[R_ct][box_ct] array for use in TS by filtering on different scales and storing results
// Similar to fill_Rbox_table but called using different redshifts for each scale
int UpdateXraySourceBox(HaloBox *halobox, double R_inner, double R_outer, int R_ct,
                        XraySourceBox *source_box) {
    int status;
    Try {
        // the indexing needs these

        // only print once, since this is called for every R
        if (R_ct == 0) LOG_DEBUG("starting XraySourceBox");

        double sfr_avg, fsfr_avg, sfr_avg_mini = 0., fsfr_avg_mini = 0.;
        double xray_avg, fxray_avg;
        one_annular_filter(halobox->halo_sfr,
                           &(source_box->filtered_sfr[R_ct * HII_TOT_NUM_PIXELS]), R_inner, R_outer,
                           &sfr_avg, &fsfr_avg);
        one_annular_filter(halobox->halo_xray,
                           &(source_box->filtered_xray[R_ct * HII_TOT_NUM_PIXELS]), R_inner,
                           R_outer, &xray_avg, &fxray_avg);

        source_box->mean_sfr[R_ct] = fsfr_avg;
        if (astro_options_global->USE_MINI_HALOS) {
            one_annular_filter(halobox->halo_sfr_mini,
                               &(source_box->filtered_sfr_mini[R_ct * HII_TOT_NUM_PIXELS]), R_inner,
                               R_outer, &sfr_avg_mini, &fsfr_avg_mini);

            source_box->mean_sfr_mini[R_ct] = fsfr_avg_mini;
            source_box->mean_log10_Mcrit_LW[R_ct] = halobox->log10_Mcrit_MCG_ave;
        }

        if (R_ct == astro_params_global->N_STEP_TS - 1) LOG_DEBUG("finished XraySourceBox");

        LOG_SUPER_DEBUG("R = [%8.3f - %8.3f] | mean filtered sfr  = %10.3e unfiltered %10.3e",
                        R_inner, R_outer, fsfr_avg, sfr_avg);
        LOG_ULTRA_DEBUG("mean filtered xray = %10.3e unfiltered %10.3e", fxray_avg, xray_avg);
        if (astro_options_global->USE_MINI_HALOS) {
            LOG_SUPER_DEBUG("MINI: filtered sfr %10.3e unfiltered %10.3e log10_Mcrit_LW = %10.3e",
                            fsfr_avg_mini, sfr_avg_mini, source_box->mean_log10_Mcrit_LW[R_ct]);
        }

        fftwf_forget_wisdom();
        fftwf_cleanup_threads();
        fftwf_cleanup();
    }  // End of try
    Catch(status) { return (status); }
    return (0);
}

// construct the [x_e][R_ct] tables
// NOTE: these have always been interpolation tables in x_e, regardless of flags
// NOTE: Frequency integrals are based on PREVIOUS x_e_ave
//   The x_e tables are not regular, hence the precomputation of indices/interp points
void fill_freqint_tables(double zp, double x_e_ave, double filling_factor_of_HI_zp,
                         double *log10_Mcrit_LW_ave, int R_mm, ScalingConstants *sc) {
    double lower_int_limit;
    int x_e_ct, R_ct;
    int R_start, R_end;
    // if we minimize mem these arrays are filled one by one
    if (matter_options_global->MINIMIZE_MEMORY) {
        R_start = R_mm;
        R_end = R_mm + 1;
    } else {
        R_start = 0;
        R_end = astro_params_global->N_STEP_TS;
    }
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
        for (R_ct = R_start; R_ct < R_end; R_ct++) {
            if (astro_options_global->USE_MINI_HALOS) {
                lower_int_limit =
                    fmax(nu_tau_one_MINI(zp, zpp_for_evolve_list[R_ct], x_e_ave,
                                         filling_factor_of_HI_zp, log10_Mcrit_LW_ave[R_ct], sc),
                         (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz);
            } else {
                lower_int_limit = fmax(
                    nu_tau_one(zp, zpp_for_evolve_list[R_ct], x_e_ave, filling_factor_of_HI_zp, sc),
                    (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz);
            }
            // set up frequency integral table for later interpolation for the cell's x_e value
            for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++) {
                freq_int_heat_tbl[x_e_ct][R_ct] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 0);
                freq_int_ion_tbl[x_e_ct][R_ct] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 1);
                freq_int_lya_tbl[x_e_ct][R_ct] =
                    integrate_over_nu(zp, x_int_XHII[x_e_ct], lower_int_limit, 2);

                // we store these to avoid calculating them in the box_ct loop
                if (x_e_ct > 0) {
                    freq_int_heat_tbl_diff[x_e_ct - 1][R_ct] =
                        freq_int_heat_tbl[x_e_ct][R_ct] - freq_int_heat_tbl[x_e_ct - 1][R_ct];
                    freq_int_ion_tbl_diff[x_e_ct - 1][R_ct] =
                        freq_int_ion_tbl[x_e_ct][R_ct] - freq_int_ion_tbl[x_e_ct - 1][R_ct];
                    freq_int_lya_tbl_diff[x_e_ct - 1][R_ct] =
                        freq_int_lya_tbl[x_e_ct][R_ct] - freq_int_lya_tbl[x_e_ct - 1][R_ct];
                }
            }
            LOG_ULTRA_DEBUG("Nu Integrals || R_ct %d R %.2e zpp %.2f nu_min %.2e", R_ct,
                            R_values[R_ct], zpp_for_evolve_list[R_ct], lower_int_limit);
            LOG_ULTRA_DEBUG("heat[x_e=0] %.2e ion[x_e=0] %.2e lya[x_e=0] %.2e",
                            freq_int_heat_tbl[0][R_ct], freq_int_ion_tbl[0][R_ct],
                            freq_int_lya_tbl[0][R_ct]);
        }
// separating the inverse diff loop to prevent a race on different R_ct (shouldn't matter)
#pragma omp for
        for (x_e_ct = 0; x_e_ct < x_int_NXHII - 1; x_e_ct++) {
            inverse_diff[x_e_ct] = 1. / (x_int_XHII[x_e_ct + 1] - x_int_XHII[x_e_ct]);
        }
    }

    for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
        for (x_e_ct = 0; x_e_ct < x_int_NXHII; x_e_ct++) {
            if (isfinite(freq_int_heat_tbl[x_e_ct][R_ct]) == 0 ||
                isfinite(freq_int_ion_tbl[x_e_ct][R_ct]) == 0 ||
                isfinite(freq_int_lya_tbl[x_e_ct][R_ct]) == 0) {
                LOG_ERROR("One of the frequency interpolation tables has an infinity or a NaN");
                //                        Throw(ParameterError);
                Throw(TableGenerationError);
            }
        }
    }
}

// construct a Ts table above Z_HEAT_MAX, this can happen if we are computing the first box or if we
// request a redshift above Z_HEAT_MAX
void init_first_Ts(TsBox *box, float *dens, float z, float zp, double *x_e_ave, double *Tk_ave) {
    unsigned long long int box_ct;
    // zp is the requested redshift, z is the perturbed field redshift
    float growth_factor_zp;
    float inverse_growth_factor_z;
    double xe, TK, cT_ad;

    xe = xion_RECFAST(zp, 0);
    TK = T_RECFAST(zp, 0);
    cT_ad = cT_approx(zp);

    growth_factor_zp = dicke(zp);
    inverse_growth_factor_z = 1 / dicke(z);

    *x_e_ave = xe;
    *Tk_ave = TK;

#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
    {
        double gdens;
        float curr_xalpha;
#pragma omp for
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            gdens = dens[box_ct] * inverse_growth_factor_z * growth_factor_zp;
            box->kinetic_temp_neutral[box_ct] = TK * (1.0 + cT_ad * gdens);
            box->xray_ionised_fraction[box_ct] = xe;
            // compute the spin temperature
            box->spin_temperature[box_ct] = get_Ts(z, gdens, TK, xe, 0, &curr_xalpha);
        }
    }
}

// calculate the global properties used for making the frequency integrals,
//   used for filling factor, ST_OVER_PS, and NO_LIGHT
int global_reion_properties(double zp, double x_e_ave, double *log10_Mcrit_LW_ave,
                            double *mean_sfr_zpp, double *mean_sfr_zpp_mini, double *Q_HI) {
    int R_ct;
    double sum_nion = 0, sum_nion_mini = 0;
    double zpp;

    // For a lot of global evolution, this code uses Nion_general. We can replace this with the halo
    // field at the same snapshot, but the nu integrals go from zp to zpp to find the tau = 1
    // barrier so it needs the QHII in a range [zp,zpp]. I want to replace this whole thing with a
    // global history struct but I will need to change the Tau function chain.
    double determine_zpp_max, determine_zpp_min;

    // at z', we need a differenc constant struct
    ScalingConstants sc;
    set_scaling_constants(zp, &sc, false);

    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        determine_zpp_min = zp * 0.999;
        // NOTE: must be called after setup_z_edges for this line
        determine_zpp_max = zpp_for_evolve_list[astro_params_global->N_STEP_TS - 1] * 1.001;

        // We need the tables for the frequency integrals & mean fixing
        // NOTE: These global tables confuse me, we do ~400 (x50 for mini) integrals to build the
        // table, despite only having
        //   ~100 redshifts. The benefit of interpolating here would only matter if we keep the same
        //   table over subsequent snapshots, which we don't seem to do. The Nion table is used in
        //   nu_tau_one a lot but I think there's a better way to do that
        if (matter_options_global->SOURCE_MODEL > 0) {
            /* initialise interpolation of the mean collapse fraction for global reionization.*/
            initialise_Nion_Ts_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                      &sc);

            initialise_SFRD_spline(zpp_interp_points_SFR, determine_zpp_min, determine_zpp_max,
                                   &sc);
        } else {
            init_FcollTable(determine_zpp_min, determine_zpp_max, true);
        }
    }

    // For consistency between halo and non-halo based, the NO_LIGHT and filling_factor_zp
    //   are based on the expected global Nion. as mentioned above it would be nice to
    //   change this to a saved reionisation/sfrd history from previous snapshots
    sum_nion = EvaluateNionTs(zp, &sc);
    if (astro_options_global->USE_MINI_HALOS) {
        sum_nion_mini = EvaluateNionTs_MINI(zp, log10_Mcrit_LW_ave[0], &sc);
    }

    LOG_DEBUG("nion zp = %.3e (%.3e MINI)", sum_nion, sum_nion_mini);

    double ION_EFF_FACTOR, ION_EFF_FACTOR_MINI;
    ION_EFF_FACTOR = astro_params_global->F_STAR10 * astro_params_global->F_ESC10 *
                     astro_params_global->POP2_ION;
    ION_EFF_FACTOR_MINI = astro_params_global->F_STAR7_MINI * astro_params_global->F_ESC7_MINI *
                          astro_params_global->POP3_ION;

    // NOTE: only used without MASS_DEPENDENT_ZETA
    *Q_HI = 1 - (ION_EFF_FACTOR * sum_nion + ION_EFF_FACTOR_MINI * sum_nion_mini) / (1.0 - x_e_ave);

    // Initialise freq tables & prefactors (x_e by R tables)
    if (!matter_options_global->MINIMIZE_MEMORY) {
        // Now global SFRD at (R_ct) for the mean fixing
        for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
            zpp = zpp_for_evolve_list[R_ct];
            mean_sfr_zpp[R_ct] = EvaluateSFRD(zpp, &sc);
            if (astro_options_global->USE_MINI_HALOS) {
                mean_sfr_zpp_mini[R_ct] = EvaluateSFRD_MINI(zpp, log10_Mcrit_LW_ave[R_ct], &sc);
            }
        }
        fill_freqint_tables(zp, x_e_ave, *Q_HI, log10_Mcrit_LW_ave, 0, &sc);
    }

    return sum_nion + sum_nion_mini > 1e-15 ? 0 : 1;  // NO_LIGHT returned
}

void calculate_sfrd_from_grid(int R_ct, float *dens_R_grid, float *Mcrit_R_grid, float *sfrd_grid,
                              float *sfrd_grid_mini, double *ave_sfrd, double *ave_sfrd_mini,
                              ScalingConstants *sc) {
    double ave_sfrd_buf = 0;
    double ave_sfrd_buf_mini = 0;
    if (astro_options_global->INTEGRATION_METHOD_ATOMIC == 1 ||
        (astro_options_global->USE_MINI_HALOS &&
         astro_options_global->INTEGRATION_METHOD_MINI == 1))
        initialise_GL(log(M_min_R[R_ct]), log(M_max_R[R_ct]));

    if (matter_options_global->USE_INTERPOLATION_TABLES > 1) {
        if (matter_options_global->SOURCE_MODEL == 1) {
            initialise_SFRD_Conditional_table(zpp_for_evolve_list[R_ct],
                                              min_densities[R_ct] * zpp_growth[R_ct],
                                              max_densities[R_ct] * zpp_growth[R_ct] * 1.001,
                                              M_min_R[R_ct], M_max_R[R_ct], M_max_R[R_ct], sc);
        } else if (matter_options_global->SOURCE_MODEL == 0) {
            initialise_FgtrM_delta_table(
                min_densities[R_ct] * zpp_growth[R_ct], max_densities[R_ct] * zpp_growth[R_ct],
                zpp_for_evolve_list[R_ct], zpp_growth[R_ct], sigma_min[R_ct], sigma_max[R_ct]);
        } else {
            LOG_ERROR("Source model %d is trying to calculate SFRD from grid, something went wrong",
                      matter_options_global->SOURCE_MODEL);
            Throw(ValueError);
        }
    }

#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
        unsigned long long int box_ct;
        double curr_dens;
        double curr_mcrit = 0.;
        double fcoll, dfcoll;
        double fcoll_MINI = 0;

#pragma omp for reduction(+ : ave_sfrd_buf, ave_sfrd_buf_mini)
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            curr_dens = dens_R_grid[box_ct] * zpp_growth[R_ct];
            if (astro_options_global->USE_MINI_HALOS) curr_mcrit = Mcrit_R_grid[box_ct];

            if (matter_options_global->SOURCE_MODEL == 1) {
                fcoll = EvaluateSFRD_Conditional(curr_dens, zpp_growth[R_ct], M_min_R[R_ct],
                                                 M_max_R[R_ct], M_max_R[R_ct], sigma_max[R_ct], sc);
                sfrd_grid[box_ct] = (1. + curr_dens) * fcoll;

                if (astro_options_global->USE_MINI_HALOS) {
                    fcoll_MINI = EvaluateSFRD_Conditional_MINI(
                        curr_dens, curr_mcrit, zpp_growth[R_ct], M_min_R[R_ct], M_max_R[R_ct],
                        M_max_R[R_ct], sigma_max[R_ct], sc);
                    sfrd_grid_mini[box_ct] = (1. + curr_dens) * fcoll_MINI;
                }
            } else {
                fcoll = EvaluateFcoll_delta(curr_dens, zpp_growth[R_ct], sigma_min[R_ct],
                                            sigma_max[R_ct]);
                dfcoll = EvaluatedFcolldz(curr_dens, zpp_for_evolve_list[R_ct], sigma_min[R_ct],
                                          sigma_max[R_ct]);
                sfrd_grid[box_ct] = (1. + curr_dens) * dfcoll;
            }
            ave_sfrd_buf += fcoll;
            ave_sfrd_buf_mini += fcoll_MINI;
        }
    }
    *ave_sfrd = ave_sfrd_buf / HII_TOT_NUM_PIXELS;
    *ave_sfrd_mini = ave_sfrd_buf_mini / HII_TOT_NUM_PIXELS;

    // These functions check for allocation
    free_conditional_tables();
}

// These are factors which only need to be calculated once per redshift
struct spintemp_from_sfr_prefactors {
    double xray_prefactor;       // convserion from SFRD to xray emissivity
    double Trad;                 // CMB temperature
    double Trad_inv;             // inverse for acceleration (/ slower than * sometimes)
    double Ts_prefactor;         // some volume factors
    double xa_tilde_prefactor;   // lyman alpha prefactor
    double xc_inverse;           // collisional prefactor
    double dcomp_dzp_prefactor;  // compton prefactor
    double Nb_zp;                // physical critical density (baryons)
    double N_zp;                 // physical critical density
    double lya_star_prefactor;   // converts SFR density -> stellar baryon density + prefactors
    double volunit_inv;          // inverse volume unit for cm^-3 conversion
    double hubble_zp;            // H(z)
    double growth_zp;
    double dgrowth_dzp;
    double dt_dzp;
};

void set_zp_consts(double zp, struct spintemp_from_sfr_prefactors *consts) {
    // constant prefactors for the R_ct==0 part
    double luminosity_converstion_factor;
    double gamma_alpha;
    consts->growth_zp = dicke(zp);
    consts->hubble_zp = hubble(zp);
    consts->dgrowth_dzp = ddicke_dz(zp);
    consts->dt_dzp = dtdz(zp);
    if (fabs(astro_params_global->X_RAY_SPEC_INDEX - 1.0) < 1e-6) {
        luminosity_converstion_factor =
            (astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz *
            log(astro_params_global->NU_X_BAND_MAX / (astro_params_global->NU_X_THRESH));
        luminosity_converstion_factor = 1. / luminosity_converstion_factor;
    } else {
        luminosity_converstion_factor =
            pow((astro_params_global->NU_X_BAND_MAX) * physconst.eV_to_Hz,
                1. - (astro_params_global->X_RAY_SPEC_INDEX)) -
            pow((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz,
                1. - (astro_params_global->X_RAY_SPEC_INDEX));
        luminosity_converstion_factor = 1. / luminosity_converstion_factor;
        luminosity_converstion_factor *=
            pow((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz,
                -(astro_params_global->X_RAY_SPEC_INDEX)) *
            (1 - (astro_params_global->X_RAY_SPEC_INDEX));
    }
    // Finally, convert to the correct units. physconst.eV_to_Hz*physconst.h_p as only want to
    // divide by eV -> erg (owing to the definition of Luminosity)
    luminosity_converstion_factor /= (physconst.h_p);

    // for halos, we just want the SFR -> X-ray part
    // NOTE: compared to Mesinger+11: (1+zpp)^2 (1+zp) -> (1+zp)^3
    //(1+z)^3 is here because we don't want it in the
    // star lya (already in zpp integrand)
    consts->xray_prefactor =
        luminosity_converstion_factor / ((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz) *
        physconst.c_cms * pow(1 + zp, astro_params_global->X_RAY_SPEC_INDEX + 3);

    // Required quantities for calculating the IGM spin temperature
    // Note: These used to be determined in evolveInt (and other functions). But I moved them all
    // here, into a single location.
    consts->Trad = physconst.T_cmb * (1.0 + zp);
    consts->Trad_inv = 1.0 / consts->Trad;
    consts->Ts_prefactor =
        pow(1.0e-7 * (1.342881e-7 / consts->hubble_zp) * No * pow(1 + zp, 3), 1. / 3.);

    // division of C/10. is converstion of electric charge from esu to coulomb
    gamma_alpha = physconst.f_alpha *
                  pow(physconst.nu_Ly_alpha * physconst.e_charge / (physconst.c_cms / 10.), 2.);
    // division by 1000. to convert gram to kg and division by 100. to convert cm to m
    gamma_alpha /=
        6. * (physconst.m_e / 1000.) * pow(physconst.c_cms / 100., 3.) * physconst.vac_perm;

    // 1e-8 converts angstrom to cm.
    consts->xa_tilde_prefactor =
        8. * M_PI * pow(physconst.lambda_Ly_alpha * 1.e-8, 2.) * gamma_alpha * physconst.T_21;
    consts->xa_tilde_prefactor /= 9. * physconst.A10 * consts->Trad;
    // consts->xa_tilde_prefactor = 1.66e11/(1.0+zp);

    consts->xc_inverse = pow(1.0 + zp, 3.0) * physconst.T_21 / (consts->Trad * physconst.A10);

    consts->dcomp_dzp_prefactor = (-1.51e-4) / (consts->hubble_zp / Ho) /
                                  (cosmo_params_global->hlittle) * pow(consts->Trad, 4.0) /
                                  (1.0 + zp);

    // used for lya_X and sinks NOTE: the 2 density factors are from
    // source & absorber since its downscattered x-ray
    consts->Nb_zp = N_b0 * (1 + zp) * (1 + zp) * (1 + zp);
    consts->N_zp = No * (1 + zp) * (1 + zp) * (1 + zp);  // used for CMB
    // converts SFR density -> stellar baryon density + prefactors
    consts->lya_star_prefactor = physconst.c_cms / (4.0 * M_PI) * physconst.Msun / physconst.m_p *
                                 (1 - 0.75 * cosmo_params_global->Y_He);

    // converts the grid emissivity unit to per cm-3
    if (matter_options_global->SOURCE_MODEL > 1) {
        consts->volunit_inv = pow(physconst.cm_per_Mpc, -3);
    } else {
        consts->volunit_inv = cosmo_params_global->OMb * RHOcrit * pow(physconst.cm_per_Mpc, -3);
    }

    LOG_DEBUG("Set zp consts xr %.2e Tr %.2e Ts %.2e xa %.2e xc %.2e cm %.2e",
              consts->xray_prefactor, consts->Trad, consts->Ts_prefactor,
              consts->xa_tilde_prefactor, consts->xc_inverse, consts->dcomp_dzp_prefactor);
    LOG_DEBUG("Nb %.2e la %.2e vi %.2e D %.2e H %.2e dD %.2e dt %.2e", consts->Nb_zp,
              consts->lya_star_prefactor, consts->volunit_inv, consts->growth_zp, consts->hubble_zp,
              consts->dgrowth_dzp, consts->dt_dzp);
}

// All the cell-dependent stuff needed to calculate Ts
struct Box_rad_terms {
    double dxion_dt;
    double dxheat_dt;
    double dxlya_dt;
    double dstarlya_dt;
    double dstarLW_dt;
    double dstarlya_cont_dt;
    double dstarlya_inj_dt;
    double delta;
    double prev_Ts;
    double prev_Tk;
    double prev_xe;
};

// outputs from the Ts calculation, to go into new boxes
struct Ts_cell {
    double Ts;
    double x_e;
    double Tk;
    double J_21_LW;
};

// Function for calculating the Ts box outputs quickly by using pre-calculated constants
//   as much as possible
struct Ts_cell get_Ts_fast(float zp, float dzp, struct spintemp_from_sfr_prefactors *consts,
                           struct Box_rad_terms *rad) {
    // Now we can solve the evolution equations  //
    struct Ts_cell output;
    double tau21, xCMB, dxion_sink_dt, dxe_dzp, dadia_dzp, dspec_dzp, dcomp_dzp, dxheat_dzp;
    double dCMBheat_dzp, eps_CMB, eps_Lya_cont, eps_Lya_inj, E_continuum, E_injected,
        Ndot_alpha_cont, Ndot_alpha_inj;

    tau21 = (3 * physconst.h_p * physconst.A10 * physconst.c_cms * physconst.lambda_21 *
             physconst.lambda_21 / 32. / M_PI / physconst.k_B) *
            ((1 - rad->prev_xe) * consts->N_zp) / rad->prev_Ts / consts->hubble_zp;
    xCMB = (1. - exp(-tau21)) / tau21;

    // Electron density
    // NOTE: Nb_zp includes helium, TODO: make sure this is right
    dxion_sink_dt = alpha_A(rad->prev_Tk) * astro_params_global->CLUMPING_FACTOR * rad->prev_xe *
                    rad->prev_xe * H_FRAC * consts->Nb_zp * (1. + rad->delta);

    dxe_dzp = consts->dt_dzp * (rad->dxion_dt - dxion_sink_dt);

    // Temperature components //
    // Adiabatic heating/cooling from structure formation
    dadia_dzp = 3 / (1.0 + zp);
    if (fabs(rad->delta) > FRACT_FLOAT_ERR)
        dadia_dzp += consts->dgrowth_dzp / (consts->growth_zp * (1.0 / rad->delta + 1.0));

    dadia_dzp *= (2.0 / 3.0) * rad->prev_Tk;

    // Heating due to the changing species
    dspec_dzp = -dxe_dzp * rad->prev_Tk / (1 + rad->prev_xe);

    // Compton heating
    dcomp_dzp = consts->dcomp_dzp_prefactor * (rad->prev_xe / (1.0 + rad->prev_xe + HE_FRAC)) *
                (consts->Trad - rad->prev_Tk);

    // X-ray heating
    dxheat_dzp = 0.;
    if (astro_options_global->USE_X_RAY_HEATING) {
        dxheat_dzp =
            rad->dxheat_dt * consts->dt_dzp * 2.0 / 3.0 / physconst.k_B / (1.0 + rad->prev_xe);
    }
    // CMB heating rate
    dCMBheat_dzp = 0.;
    if (astro_options_global->USE_CMB_HEATING) {
        // Meiksin et al. 2021
        eps_CMB = (3. / 4.) * (consts->Trad / physconst.T_21) * physconst.A10 * H_FRAC *
                  (physconst.h_p * physconst.h_p / physconst.lambda_21 / physconst.lambda_21 /
                   physconst.m_p) *
                  (1. + 2. * rad->prev_Tk / physconst.T_21);
        dCMBheat_dzp = -eps_CMB * (2. / 3. / physconst.k_B / (1. + rad->prev_xe)) /
                       consts->hubble_zp / (1. + zp);
    }

    // Ly-alpha heating rate
    eps_Lya_cont = 0.;
    eps_Lya_inj = 0.;
    if (astro_options_global->USE_LYA_HEATING) {
        E_continuum =
            Energy_Lya_heating(rad->prev_Tk, rad->prev_Ts, taugp(zp, rad->delta, rad->prev_xe), 2);
        E_injected =
            Energy_Lya_heating(rad->prev_Tk, rad->prev_Ts, taugp(zp, rad->delta, rad->prev_xe), 3);
        if (isnan(E_continuum) || isinf(E_continuum)) {
            E_continuum = 0.;
        }
        if (isnan(E_injected) || isinf(E_injected)) {
            E_injected = 0.;
        }
        Ndot_alpha_cont = (4. * M_PI * physconst.nu_Ly_alpha) /
                          (consts->Nb_zp * (1. + rad->delta)) / (1. + zp) / physconst.c_cms *
                          rad->dstarlya_cont_dt;
        Ndot_alpha_inj = (4. * M_PI * physconst.nu_Ly_alpha) / (consts->Nb_zp * (1. + rad->delta)) /
                         (1. + zp) / physconst.c_cms * rad->dstarlya_inj_dt;
        eps_Lya_cont =
            -Ndot_alpha_cont * E_continuum * (2. / 3. / physconst.k_B / (1. + rad->prev_xe));
        eps_Lya_inj =
            -Ndot_alpha_inj * E_injected * (2. / 3. / physconst.k_B / (1. + rad->prev_xe));
    }

    // Update the cell quantities based on the above terms
    double x_e, Tk;
    x_e = rad->prev_xe + (dxe_dzp * dzp);  // remember dzp is negative
    // can do this late in evolution if dzp is too large
    if (x_e > 1)
        x_e = 1 - FRACT_FLOAT_ERR;
    else if (x_e < 0)
        x_e = 0;
    // NOTE: does this stop cooling if we ever go over the limit? I suppose that shouldn't happen
    // but it's strange anyway
    Tk = rad->prev_Tk;
    if (Tk < MAX_TK) {
        if (debug_printed == 0 && omp_get_thread_num() == 0)
            LOG_SUPER_DEBUG(
                "Heating Terms: T %.4e | X %.4e | c %.4e | S %.4e | A %.4e | c %.4e | lc %.4e | li "
                "%.4e | dz %.4e",
                Tk, dxheat_dzp, dcomp_dzp, dspec_dzp, dadia_dzp, dCMBheat_dzp, eps_Lya_cont,
                eps_Lya_inj, dzp);

        Tk += (dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp + dCMBheat_dzp + eps_Lya_cont +
               eps_Lya_inj) *
              dzp;
        if (debug_printed == 0 && omp_get_thread_num() == 0) LOG_SUPER_DEBUG("--> T %.4e", Tk);
    }
    // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
    if (Tk < 0) Tk = consts->Trad;

    output.x_e = x_e;
    output.Tk = Tk;
    output.J_21_LW = astro_options_global->USE_MINI_HALOS ? rad->dstarLW_dt : 0.;

    double J_alpha_tot = rad->dstarlya_dt + rad->dxlya_dt;  // not really d/dz, but the lya flux

    // JD: I'm leaving these as comments in case I'm wrong, but there's NO WAY a compiler doesn't
    // know the fastest way to invert a number
    //  T_inv = expf((-1.)*logf(Tk));
    //  T_inv_sq = expf((-2.)*logf(Tk));
    double T_inv, T_inv_sq;
    double xc_fast, xi_power, xa_tilde_fast_arg, xa_tilde_fast = 0.;
    double TS_fast, TSold_fast;
    T_inv = 1 / Tk;
    T_inv_sq = T_inv * T_inv;

    xc_fast = (1.0 + rad->delta) * consts->xc_inverse *
              ((1.0 - x_e) * No * kappa_10(Tk, 0) + x_e * N_b0 * kappa_10_elec(Tk, 0) +
               x_e * No * kappa_10_pH(Tk, 0));

    xi_power = consts->Ts_prefactor * cbrt((1.0 + rad->delta) * (1.0 - x_e) * T_inv_sq);

    xa_tilde_fast_arg = consts->xa_tilde_prefactor * J_alpha_tot *
                        pow(1.0 + 2.98394 * xi_power + 1.53583 * xi_power * xi_power +
                                3.85289 * xi_power * xi_power * xi_power,
                            -1.);

    if (J_alpha_tot > 1.0e-20) {  // Must use WF effect
        TS_fast = consts->Trad;
        TSold_fast = 0.0;
        while (fabs(TS_fast - TSold_fast) / TS_fast > 1.0e-3) {
            TSold_fast = TS_fast;

            xa_tilde_fast =
                (1.0 - 0.0631789 * T_inv + 0.115995 * T_inv_sq -
                 0.401403 * T_inv * pow(TS_fast, -1.) + 0.336463 * T_inv_sq * pow(TS_fast, -1.)) *
                xa_tilde_fast_arg;

            TS_fast = (xCMB + xa_tilde_fast + xc_fast) *
                      pow(xCMB * consts->Trad_inv +
                              xa_tilde_fast * (T_inv + 0.405535 * T_inv * pow(TS_fast, -1.) -
                                               0.405535 * T_inv_sq) +
                              xc_fast * T_inv,
                          -1.);
        }
    } else {  // Collisions only
        TS_fast = (xCMB + xc_fast) / (xCMB * consts->Trad_inv + xc_fast * T_inv);
    }
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
    if (debug_printed == 0 && omp_get_thread_num() == 0) {
        LOG_SUPER_DEBUG("Spin terms xc %.5e xa %.5e xC %.5e Ti %.5e T2 %.5e --> T %.4e", xc_fast,
                        xa_tilde_fast, xCMB, T_inv, T_inv_sq, TS_fast);
        debug_printed = 1;
    }
#endif
    // It can very rarely result in a negative spin temperature. If negative, it is a very small
    // number. Take the absolute value, the optical depth can deal with very large numbers, so ok to
    // be small
    TS_fast = fabs(TS_fast);
    output.Ts = TS_fast;

    return output;
}

// outer-level function for calculating Ts based on the Halo boxes
void ts_main(float redshift, float prev_redshift, float perturbed_field_redshift, short cleanup,
             PerturbedField *perturbed_field, XraySourceBox *source_box, TsBox *previous_spin_temp,
             InitialConditions *ini_boxes, TsBox *this_spin_temp) {
    int R_ct;
    unsigned long long int box_ct;
    double x_e_ave_p, Tk_ave_p;
    double growth_factor_z, growth_factor_zp;
    double inverse_growth_factor_z;
    double dzp;
    double start, end;

    // Must be always initialised due to the strange way Ionisationbox.c expects some initialisation
    init_ps();

    // allocate the global arrays we always use
    if (!TsInterpArraysInitialised) {
        alloc_global_arrays();
        // Initialize heating interpolation arrays
        init_heat();
    }

    // NOTE: For the code to work, previous_spin_temp MUST be allocated &
    // calculated if redshift < Z_HEAT_MAX
    growth_factor_z = dicke(perturbed_field_redshift);
    inverse_growth_factor_z = 1. / growth_factor_z;

    growth_factor_zp = dicke(redshift);
    dzp = redshift - prev_redshift;

    // setup the R_ct 1D arrays
    setup_z_edges(redshift);

    // with the TtoM limit, we use the largest redshift, to cover the whole range
    double M_MIN_tb = M_min_R[astro_params_global->N_STEP_TS - 1];
    // This M_MIN just sets the sigma table range, the minimum mass for the integrals is set per
    // radius in setup_z_edges
    if (astro_options_global->INTEGRATION_METHOD_ATOMIC == 2 ||
        astro_options_global->INTEGRATION_METHOD_MINI == 2)
        M_MIN_tb = fmin(MMIN_FAST, M_MIN_tb);

    // we need a larger table here due to the large radii
    if (matter_options_global->USE_INTERPOLATION_TABLES > 0)
        initialiseSigmaMInterpTable(M_MIN_tb / 2, 1e20);

    // now that we have the sigma table we can assign the sigma arrays
    for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
        sigma_min[R_ct] = EvaluateSigma(log(M_min_R[R_ct]));
        sigma_max[R_ct] = EvaluateSigma(log(M_max_R[R_ct]));
    }

    if (redshift >= simulation_options_global->Z_HEAT_MAX) {
        LOG_DEBUG("redshift greater than Z_HEAT_MAX");
        init_first_Ts(this_spin_temp, perturbed_field->density, perturbed_field_redshift, redshift,
                      &x_e_ave_p, &Tk_ave_p);
        return;
    }

    calculate_spectral_factors(redshift);

    // Fill the R_ct,box_ct fields
    // Since we use the average Mturn for the global tables this must be done first
    // NOTE: The filtered Mturn for the previous snapshot is used for Fcoll at ALL zpp
    //   regardless of distance from current reshift, this also goes for the averages
    double ave_log10_MturnLW[astro_params_global->N_STEP_TS];
    double min_log10_MturnLW[astro_params_global->N_STEP_TS];
    double max_log10_MturnLW[astro_params_global->N_STEP_TS];
    double ave_dens[astro_params_global->N_STEP_TS];
    fftwf_complex *log10_Mcrit_LW_unfiltered = NULL;
    fftwf_complex *delta_unfiltered = NULL;
    double log10_Mcrit_limit;

    if (matter_options_global->SOURCE_MODEL < 2) {
        // copy over to FFTW, do the forward FFTs and apply constants
        delta_unfiltered =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
        if (astro_options_global->USE_MINI_HALOS)
            log10_Mcrit_LW_unfiltered =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

        LOG_ULTRA_DEBUG("Starting unfiltered boxes.");
        prepare_filter_boxes(redshift, perturbed_field->density, ini_boxes->lowres_vcb,
                             previous_spin_temp->J_21_LW, delta_unfiltered,
                             log10_Mcrit_LW_unfiltered);
        LOG_ULTRA_DEBUG("Prepared unfiltered boxes.");
        // fill the filtered boxes if we are storing them all
        if (!matter_options_global->MINIMIZE_MEMORY) {
            fill_Rbox_table(delNL0, delta_unfiltered, R_values, astro_params_global->N_STEP_TS, -1,
                            inverse_growth_factor_z, min_densities, ave_dens, max_densities);
            LOG_ULTRA_DEBUG("Filled density filtered boxes.");
            if (astro_options_global->USE_MINI_HALOS) {
                // NOTE: we are using previous_zp LW threshold for all zpp, inconsistent with the
                // halo model
                // minimum turnover NOTE: should be zpp_max?
                log10_Mcrit_limit = log10(lyman_werner_threshold(redshift, 0., 0.));
                fill_Rbox_table(log10_Mcrit_LW, log10_Mcrit_LW_unfiltered, R_values,
                                astro_params_global->N_STEP_TS, log10_Mcrit_limit, 1,
                                min_log10_MturnLW, ave_log10_MturnLW, max_log10_MturnLW);
            }
        } else {
            // we still need the average Mturn at R_ct==0 for NO_LIGHT
            // TODO: Remove this and come up with a better way to get NO_LIGHT
            if (astro_options_global->USE_MINI_HALOS) {
                fill_Rbox_table(log10_Mcrit_LW, log10_Mcrit_LW_unfiltered, &(R_values[0]), 1, 0, 1,
                                &min_log10_MturnLW[0], &ave_log10_MturnLW[0],
                                &max_log10_MturnLW[0]);
            }
        }
        LOG_DEBUG("Constructed filtered boxes.");
    } else if (astro_options_global->USE_MINI_HALOS) {
        for (R_ct = 0; R_ct < astro_params_global->N_STEP_TS; R_ct++) {
            ave_log10_MturnLW[R_ct] = source_box->mean_log10_Mcrit_LW[R_ct];
        }
    }
    // set the constants calculated once per snapshot
    struct spintemp_from_sfr_prefactors zp_consts;
    set_zp_consts(redshift, &zp_consts);

    x_e_ave_p = Tk_ave_p = 0.0;
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
    {
#pragma omp for reduction(+ : x_e_ave_p, Tk_ave_p)
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            x_e_ave_p += previous_spin_temp->xray_ionised_fraction[box_ct];
            Tk_ave_p += previous_spin_temp->kinetic_temp_neutral[box_ct];
        }
    }
    x_e_ave_p /= (float)HII_TOT_NUM_PIXELS;
    Tk_ave_p /= (float)HII_TOT_NUM_PIXELS;  // not used?
    LOG_DEBUG("Prev Box: x_e_ave %.3e | TK_ave %.3e", x_e_ave_p, Tk_ave_p);

    int NO_LIGHT;
    double mean_sfr_zpp[astro_params_global->N_STEP_TS];
    double mean_sfr_zpp_mini[astro_params_global->N_STEP_TS];

    // this should initialise and use the global tables (given box average turnovers)
    //   and use them to give: Filling factor at zp (only used for !MASS_DEPENDENT_ZETA to get
    //   ion_eff) global SFRD at each filter radius (numerator of ST_over_PS factor)
    double Q_HI_zp;

    // start = get_time_ms();
    NO_LIGHT = global_reion_properties(redshift, x_e_ave_p, ave_log10_MturnLW, mean_sfr_zpp,
                                       mean_sfr_zpp_mini, &Q_HI_zp);
    // end = get_time_ms();
    // printf("global_reion_properties took %.3f ms\n", end - start);

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
            // these are the index and interpolation term, moved outside the R loop and stored to
            // not calculate them R times
            m_xHII_low_box[box_ct] = locate_xHII_index(xHII_call);
            inverse_val_box[box_ct] = (xHII_call - x_int_XHII[m_xHII_low_box[box_ct]]) *
                                      inverse_diff[m_xHII_low_box[box_ct]];

            // initialise += boxes (memory sometimes re-used)
            if (astro_options_global->USE_X_RAY_HEATING) {
                dxheat_dt_box[box_ct] = 0.;
            }
            dxion_source_dt_box[box_ct] = 0.;
            dxlya_dt_box[box_ct] = 0.;
            dstarlya_dt_box[box_ct] = 0.;
            if (astro_options_global->USE_MINI_HALOS) dstarlyLW_dt_box[box_ct] = 0.;
            if (astro_options_global->USE_LYA_HEATING) {
                dstarlya_cont_dt_box[box_ct] = 0.;
                dstarlya_inj_dt_box[box_ct] = 0.;
            }
        }
    }

    // MAIN LOOP: SFR -> heating terms with freq integrals
    double z_edge_factor, dzpp_for_evolve, zpp, xray_R_factor;
    double J_alpha_ave, xheat_ave, xion_ave, Ts_ave, Tk_ave, x_e_ave;
    J_alpha_ave = xheat_ave = xion_ave = Ts_ave = Tk_ave = x_e_ave = 0;
    double J_LW_ave = 0., eps_lya_cont_ave = 0, eps_lya_inj_ave = 0;
    double lyacont_factor_mini = 0., lyainj_factor_mini = 0., starlya_factor_mini = 0.;
    double ave_fcoll, ave_fcoll_MINI;
    double avg_fix_term = 1.;
    double avg_fix_term_MINI = 1.;
    int R_index;
    float *delta_box_input;
    float *Mcrit_box_input = NULL;  // may be unused
    ScalingConstants sc, sc_sfrd;

    // if we have stars, fill in the heating term boxes
    if (!NO_LIGHT) {
        for (R_ct = astro_params_global->N_STEP_TS; R_ct--;) {
            dzpp_for_evolve = dzpp_list[R_ct];
            zpp = zpp_for_evolve_list[R_ct];
            // dtdz'' dz'' -> dR for the radius sum (c included in constants)
            if (matter_options_global->SOURCE_MODEL == 0)
                z_edge_factor = dzpp_for_evolve;  // uses dfcoll/dz
            else if (matter_options_global->SOURCE_MODEL == 1)
                z_edge_factor = fabs(dzpp_for_evolve * dtdz_list[R_ct]) * hubble(zpp) /
                                astro_params_global->t_STAR;
            else
                z_edge_factor = fabs(dzpp_for_evolve * dtdz_list[R_ct]);

            xray_R_factor = pow(1 + zpp, -(astro_params_global->X_RAY_SPEC_INDEX));

            set_scaling_constants(zpp, &sc, false);

            // index for grids
            R_index = matter_options_global->MINIMIZE_MEMORY ? 0 : R_ct;

            if (matter_options_global->SOURCE_MODEL < 2) {
                if (matter_options_global->MINIMIZE_MEMORY) {
                    // we call the filtering functions once here per R
                    // This unnecessarily allocates and frees a fftwf box every time but surely
                    // that's not a bottleneck
                    fill_Rbox_table(delNL0, delta_unfiltered, &(R_values[R_ct]), 1, -1,
                                    inverse_growth_factor_z, &min_densities[R_ct], &ave_dens[R_ct],
                                    &max_densities[R_ct]);
                    if (astro_options_global->USE_MINI_HALOS) {
                        fill_Rbox_table(log10_Mcrit_LW, log10_Mcrit_LW_unfiltered,
                                        &(R_values[R_ct]), 1, 0, 1, &min_log10_MturnLW[R_ct],
                                        &ave_log10_MturnLW[R_ct], &max_log10_MturnLW[R_ct]);
                    }
                    // get the global things we missed before
                    mean_sfr_zpp[R_ct] = EvaluateSFRD(zpp_for_evolve_list[R_ct], &sc);
                    if (astro_options_global->USE_MINI_HALOS) {
                        mean_sfr_zpp_mini[R_ct] = EvaluateSFRD_MINI(zpp_for_evolve_list[R_ct],
                                                                    ave_log10_MturnLW[R_ct], &sc);
                    }
                    // fill one row of the interp tables
                    fill_freqint_tables(redshift, x_e_ave_p, Q_HI_zp, ave_log10_MturnLW, R_ct, &sc);
                }
                // set input pointers (doing things this way helps with flag flexibility)
                delta_box_input = delNL0[R_index];
                if (astro_options_global->USE_MINI_HALOS) {
                    Mcrit_box_input = log10_Mcrit_LW[R_index];
                }
                calculate_sfrd_from_grid(R_ct, delta_box_input, Mcrit_box_input, del_fcoll_Rct,
                                         del_fcoll_Rct_MINI, &ave_fcoll, &ave_fcoll_MINI, &sc);
                avg_fix_term = mean_sfr_zpp[R_ct] / ave_fcoll;
                if (astro_options_global->USE_MINI_HALOS)
                    avg_fix_term_MINI = mean_sfr_zpp_mini[R_ct] / ave_fcoll_MINI;

#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
                sc_sfrd = evolve_scaling_constants_sfr(&sc);
                LOG_SUPER_DEBUG(
                    "z %6.2f ave sfrd val %.3e global %.3e (int %.3e) Mmin %.3e ratio %.4e z_edge "
                    "%.4e",
                    zpp_for_evolve_list[R_ct], ave_fcoll, mean_sfr_zpp[R_ct],
                    Nion_General(zpp_for_evolve_list[R_ct], log(M_min_R[R_ct]), log(M_MAX_INTEGRAL),
                                 sc_sfrd.mturn_a_nofb, &sc_sfrd),
                    M_min_R[R_ct], avg_fix_term, z_edge_factor);
                if (astro_options_global->USE_MINI_HALOS) {
                    LOG_SUPER_DEBUG(
                        "MINI sfrd val %.3e global %.3e (int %.3e) ratio %.3e log10McritLW %.3e",
                        ave_fcoll_MINI, mean_sfr_zpp_mini[R_ct],
                        Nion_General_MINI(zpp_for_evolve_list[R_ct], log(M_min_R[R_ct]),
                                          log(M_MAX_INTEGRAL), pow(10., ave_log10_MturnLW[R_ct]),
                                          &sc_sfrd),
                        avg_fix_term_MINI, ave_log10_MturnLW[R_ct]);
                }
#endif
            }

            // minihalo factors should be separated since they may not be allocated
            if (astro_options_global->USE_MINI_HALOS) {
                starlya_factor_mini = dstarlya_dt_prefactor_MINI[R_ct];
                if (astro_options_global->USE_LYA_HEATING) {
                    lyacont_factor_mini = dstarlya_cont_dt_prefactor_MINI[R_ct];
                    lyainj_factor_mini = dstarlya_inj_dt_prefactor_MINI[R_ct];
                }
            }

// NOTE: The ionisation box has a final delta dependence of (1+delta_source)/(1+delta_absorber)
//   But here it's just (1+delta_source). This is for photon conservation.
//   If we assume attenuation at mean density as we do in nu_tau_one(), we HAVE to assume mean
//   density absorption otherwise we do not conserve photons
#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
            {
                // private variables
                int xidx;
                double ival, sfr_term, xray_sfr;
                double sfr_term_mini = 0;
#pragma omp for
                for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
                    // sum each R contribution together
                    // The dxdt boxes exist for two reasons. Firstly it allows the MINIMIZE_MEMORY
                    // to work (replaces ~40*NUM_PIXELS with ~4-16*NUM_PIXELS),
                    //   as the FFT is done in the R-loop.
                    // Secondly, it is *likely* faster to fill these boxes, and sum with a outer R
                    // loop than an inner one.

                    if (matter_options_global->SOURCE_MODEL > 1) {
                        sfr_term = source_box->filtered_sfr[R_index * HII_TOT_NUM_PIXELS + box_ct] *
                                   z_edge_factor;
                        // Minihalos and s->yr conversion are already included here
                        xray_sfr =
                            source_box->filtered_xray[R_index * HII_TOT_NUM_PIXELS + box_ct] *
                            z_edge_factor * xray_R_factor * 1e38;
                    } else {
                        // NOTE: for SOURCE_MODEL==0 F_STAR10 is still used for constant
                        // stellar fraction
                        sfr_term = del_fcoll_Rct[box_ct] * z_edge_factor * avg_fix_term *
                                   astro_params_global->F_STAR10;
                        xray_sfr = sfr_term * astro_params_global->L_X * xray_R_factor *
                                   physconst.s_per_yr;
                    }
                    if (astro_options_global->USE_MINI_HALOS) {
                        if (matter_options_global->SOURCE_MODEL > 1) {
                            sfr_term_mini =
                                source_box->filtered_sfr_mini[R_ct * HII_TOT_NUM_PIXELS + box_ct] *
                                z_edge_factor;
                        } else {
                            sfr_term_mini = del_fcoll_Rct_MINI[box_ct] * z_edge_factor *
                                            avg_fix_term_MINI * astro_params_global->F_STAR7_MINI;
                            xray_sfr += sfr_term_mini * astro_params_global->L_X_MINI *
                                        xray_R_factor * physconst.s_per_yr;
                        }
                        dstarlyLW_dt_box[box_ct] +=
                            sfr_term * dstarlyLW_dt_prefactor[R_ct] +
                            sfr_term_mini * dstarlyLW_dt_prefactor_MINI[R_ct];
                    }
                    xidx = m_xHII_low_box[box_ct];
                    ival = inverse_val_box[box_ct];
                    if (astro_options_global->USE_X_RAY_HEATING) {
                        dxheat_dt_box[box_ct] +=
                            xray_sfr * (freq_int_heat_tbl_diff[xidx][R_ct] * ival +
                                        freq_int_heat_tbl[xidx][R_ct]);
                    }
                    dxion_source_dt_box[box_ct] +=
                        xray_sfr *
                        (freq_int_ion_tbl_diff[xidx][R_ct] * ival + freq_int_ion_tbl[xidx][R_ct]);
                    dxlya_dt_box[box_ct] += xray_sfr * (freq_int_lya_tbl_diff[xidx][R_ct] * ival +
                                                        freq_int_lya_tbl[xidx][R_ct]);
                    dstarlya_dt_box[box_ct] += sfr_term * dstarlya_dt_prefactor[R_ct] +
                                               sfr_term_mini * starlya_factor_mini;
                    if (astro_options_global->USE_LYA_HEATING) {
                        dstarlya_cont_dt_box[box_ct] +=
                            sfr_term * dstarlya_cont_dt_prefactor[R_ct] +
                            sfr_term_mini * lyacont_factor_mini;
                        dstarlya_inj_dt_box[box_ct] += sfr_term * dstarlya_inj_dt_prefactor[R_ct] +
                                                       sfr_term_mini * lyainj_factor_mini;
                    }

                    // I cannot check the integral if we are using the halo field since delNL0
                    // (filtered density) is not calculated
#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
                    if (box_ct == 0 && matter_options_global->SOURCE_MODEL < 2) {
                        double integral_db;
                        if (matter_options_global->SOURCE_MODEL == 1) {
                            integral_db =
                                Nion_ConditionalM(zpp_growth[R_ct], log(M_min_R[R_ct]),
                                                  log(M_max_R[R_ct]), M_max_R[R_ct],
                                                  sigma_max[R_ct],
                                                  delNL0[R_index][box_ct] * zpp_growth[R_ct],
                                                  sc_sfrd.mturn_a_nofb, &sc_sfrd,
                                                  astro_options_global->INTEGRATION_METHOD_ATOMIC) *
                                z_edge_factor * (1 + delNL0[R_index][box_ct] * zpp_growth[R_ct]) *
                                avg_fix_term * astro_params_global->F_STAR10;
                        } else {
                            integral_db = dfcoll_dz(zpp_for_evolve_list[R_ct], sigma_min[R_ct],
                                                    delNL0[R_index][box_ct] * zpp_growth[R_ct],
                                                    sigma_max[R_ct]) *
                                          z_edge_factor *
                                          (1 + delNL0[R_index][box_ct] * zpp_growth[R_ct]) *
                                          avg_fix_term * astro_params_global->F_STAR10;
                        }

                        LOG_SUPER_DEBUG(
                            "Cell 0: R=%.1f (%.3f) | SFR %.4e | integral %.4e | delta %.4e",
                            R_values[R_ct], zpp_for_evolve_list[R_ct], sfr_term, integral_db,
                            delNL0[R_index][box_ct]);
                        if (astro_options_global->USE_MINI_HALOS)
                            LOG_SUPER_DEBUG(
                                "MINI SFR %.4e | integral %.4e", sfr_term_mini,
                                Nion_ConditionalM_MINI(
                                    zpp_growth[R_ct], log(M_min_R[R_ct]), log(M_max_R[R_ct]),
                                    log(M_max_R[R_ct]), sigma_max[R_ct],
                                    delNL0[R_index][box_ct] * zpp_growth[R_ct],
                                    pow(10, log10_Mcrit_LW[R_ct][box_ct]), &sc_sfrd,
                                    astro_options_global->INTEGRATION_METHOD_MINI) *
                                    z_edge_factor *
                                    (1 + delNL0[R_index][box_ct] * zpp_growth[R_ct]) *
                                    avg_fix_term_MINI * astro_params_global->F_STAR7_MINI);

                        if (astro_options_global->USE_X_RAY_HEATING) {
                            LOG_SUPER_DEBUG("xh %.2e | xi %.2e | xl %.2e | sl %.2e",
                                            dxheat_dt_box[box_ct] / astro_params_global->L_X,
                                            dxion_source_dt_box[box_ct] / astro_params_global->L_X,
                                            dxlya_dt_box[box_ct] / astro_params_global->L_X,
                                            dstarlya_dt_box[box_ct]);
                        } else {
                            LOG_SUPER_DEBUG("xi %.2e | xl %.2e | sl %.2e",
                                            dxion_source_dt_box[box_ct] / astro_params_global->L_X,
                                            dxlya_dt_box[box_ct] / astro_params_global->L_X,
                                            dstarlya_dt_box[box_ct]);
                        }

                        if (astro_options_global->USE_LYA_HEATING)
                            LOG_SUPER_DEBUG("ct %.2e | ij %.2e", dstarlya_cont_dt_box[box_ct],
                                            dstarlya_inj_dt_box[box_ct]);
                    }
#endif
                }
            }
        }
    }

    // we definitely don't need these tables anymore
    // Having these free's here instead of after global_reion_properties just for MINIMIZE_MEMORY is
    // not ideal,
    //    but the log10Mturn average is needed
    free_global_tables();

    // R==0 part
#pragma omp parallel private(box_ct)
    {
        double curr_delta;
        struct Ts_cell ts_cell;
        struct Box_rad_terms rad;
#pragma omp for reduction(+ : J_alpha_ave, xheat_ave, xion_ave, Ts_ave, Tk_ave, x_e_ave, \
                              eps_lya_cont_ave, eps_lya_inj_ave)
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            curr_delta =
                perturbed_field->density[box_ct] * growth_factor_zp * inverse_growth_factor_z;
            // NOTE: this corrected for aliasing before, but sometimes there are still some
            // delta==-1 cells
            //   which breaks the adiabatic part
            if (curr_delta <= -1) {
                curr_delta = -1 + FRACT_FLOAT_ERR;
            }

            // Add prefactors that don't depend on R
            if (astro_options_global->USE_X_RAY_HEATING) {
                rad.dxheat_dt =
                    dxheat_dt_box[box_ct] * zp_consts.xray_prefactor * zp_consts.volunit_inv;
            }
            rad.dxion_dt =
                dxion_source_dt_box[box_ct] * zp_consts.xray_prefactor * zp_consts.volunit_inv;
            // 2 density terms from downscattering absorbers
            rad.dxlya_dt = dxlya_dt_box[box_ct] * zp_consts.xray_prefactor * zp_consts.volunit_inv *
                           zp_consts.Nb_zp * (1 + curr_delta);
            rad.dstarlya_dt =
                dstarlya_dt_box[box_ct] * zp_consts.lya_star_prefactor * zp_consts.volunit_inv;
            rad.delta = curr_delta;
            if (astro_options_global->USE_MINI_HALOS) {
                rad.dstarLW_dt = dstarlyLW_dt_box[box_ct] * zp_consts.lya_star_prefactor *
                                 zp_consts.volunit_inv * physconst.h_p * 1e21;
            }
            if (astro_options_global->USE_LYA_HEATING) {
                rad.dstarlya_cont_dt = dstarlya_cont_dt_box[box_ct] * zp_consts.lya_star_prefactor *
                                       zp_consts.volunit_inv;
                rad.dstarlya_inj_dt = dstarlya_inj_dt_box[box_ct] * zp_consts.lya_star_prefactor *
                                      zp_consts.volunit_inv;
            }
            rad.prev_Ts = previous_spin_temp->spin_temperature[box_ct];
            rad.prev_Tk = previous_spin_temp->kinetic_temp_neutral[box_ct];
            rad.prev_xe = previous_spin_temp->xray_ionised_fraction[box_ct];

            ts_cell = get_Ts_fast(redshift, dzp, &zp_consts, &rad);
            this_spin_temp->spin_temperature[box_ct] = ts_cell.Ts;
            this_spin_temp->kinetic_temp_neutral[box_ct] = ts_cell.Tk;
            this_spin_temp->xray_ionised_fraction[box_ct] = ts_cell.x_e;
            if (astro_options_global->USE_MINI_HALOS) {
                this_spin_temp->J_21_LW[box_ct] = ts_cell.J_21_LW;
            }

            // Single cell debug
            if (box_ct == 0) {
                LOG_SUPER_DEBUG(
                    "Cell0: delta: %.3e | xheat: %.3e | dxion: %.3e | dxlya: %.3e | dstarlya: %.3e",
                    curr_delta, rad.dxheat_dt, rad.dxion_dt, rad.dxlya_dt, rad.dstarlya_dt);
                if (astro_options_global->USE_LYA_HEATING) {
                    LOG_SUPER_DEBUG("Lya inj %.3e | Lya cont %.3e", rad.dstarlya_inj_dt,
                                    rad.dstarlya_cont_dt);
                }
                if (astro_options_global->USE_MINI_HALOS) {
                    LOG_SUPER_DEBUG("LyW %.3e", rad.dstarLW_dt);
                }
                LOG_SUPER_DEBUG("Ts %.5e Tk %.5e x_e %.5e J_21_LW %.5e", ts_cell.Ts, ts_cell.Tk,
                                ts_cell.x_e, ts_cell.J_21_LW);
            }

#if LOG_LEVEL >= DEBUG_LEVEL
            J_alpha_ave += rad.dxlya_dt + rad.dstarlya_dt;
            xheat_ave += rad.dxheat_dt;
            xion_ave += rad.dxion_dt;
            Ts_ave += ts_cell.Ts;
            Tk_ave += ts_cell.Tk;
            J_LW_ave += ts_cell.J_21_LW;
            eps_lya_inj_ave += rad.dstarlya_inj_dt;
            eps_lya_cont_ave += rad.dstarlya_cont_dt;
            x_e_ave += ts_cell.x_e;
#endif
        }
    }

#if LOG_LEVEL >= DEBUG_LEVEL
    x_e_ave /= (double)HII_TOT_NUM_PIXELS;
    Ts_ave /= (double)HII_TOT_NUM_PIXELS;
    Tk_ave /= (double)HII_TOT_NUM_PIXELS;
    J_alpha_ave /= (double)HII_TOT_NUM_PIXELS;
    xheat_ave /= (double)HII_TOT_NUM_PIXELS;
    xion_ave /= (double)HII_TOT_NUM_PIXELS;

    LOG_DEBUG("AVERAGES zp = %.2e Ts = %.2e x_e = %.2e Tk %.2e", redshift, Ts_ave, x_e_ave, Tk_ave);
    LOG_DEBUG("J_alpha = %.2e xheat = %.2e xion = %.2e", J_alpha_ave, xheat_ave, xion_ave);
    if (astro_options_global->USE_MINI_HALOS) {
        J_LW_ave /= (double)HII_TOT_NUM_PIXELS;
        LOG_DEBUG("J_LW %.2e", J_LW_ave / 1e21);
    }
    if (astro_options_global->USE_LYA_HEATING) {
        eps_lya_cont_ave /= (double)HII_TOT_NUM_PIXELS;
        eps_lya_inj_ave /= (double)HII_TOT_NUM_PIXELS;
        LOG_DEBUG("eps_cont %.2e eps_inj %.2e", eps_lya_cont_ave, eps_lya_inj_ave);
    }
#endif

    for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
        if (isfinite(this_spin_temp->spin_temperature[box_ct]) == 0) {
            if (astro_options_global->USE_X_RAY_HEATING) {
                LOG_ERROR(
                    "Estimated spin temperature is either infinite of NaN!"
                    "idx %llu delta %.3e dxheat %.3e dxion %.3e dxlya %.3e dstarlya %.3e",
                    box_ct, perturbed_field->density[box_ct], dxheat_dt_box[box_ct],
                    dxion_source_dt_box[box_ct], dxlya_dt_box[box_ct], dstarlya_dt_box[box_ct]);
            } else {
                LOG_ERROR(
                    "Estimated spin temperature is either infinite of NaN!"
                    "idx %llu delta %.3e dxion %.3e dxlya %.3e dstarlya %.3e",
                    box_ct, perturbed_field->density[box_ct], dxion_source_dt_box[box_ct],
                    dxlya_dt_box[box_ct], dstarlya_dt_box[box_ct]);
            }
            Throw(InfinityorNaNError);
        }
    }

    if (matter_options_global->SOURCE_MODEL < 2) {
        fftwf_free(delta_unfiltered);
        if (astro_options_global->USE_MINI_HALOS) {
            fftwf_free(log10_Mcrit_LW_unfiltered);
        }
        fftwf_forget_wisdom();
        fftwf_cleanup_threads();
        fftwf_cleanup();
    }

    if (cleanup) {
        free_ts_global_arrays();
    }

    return;
}
