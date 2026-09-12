/*
Module for computing the radiation fields in 21cmFAST.
This includes X-ray heating rate, photoionization rate, and Lyman-alpha flux.
*/
#include "RadiationFields.h"

#include <complex.h>
#include <fftw3.h>

#include "Constants.h"
#include "cexcept.h"
#include "cosmology.h"
#include "debugging.h"
#include "dft.h"
#include "exceptions.h"
#include "filtering.h"
#include "indexing.h"
#include "logger.h"

/*
    This function helps to calculate the radiation fields (x-ray heating rate, photoionization rate,
   lyman alpha flux, etc.). The radiation fields are all given by an integral over the past source
   emissivities. Numerically, this integral is evaluated via the trapezoidal rule, which is done by
   summing over the contributions from each redshift shell. Thus, this function calculates the
   contribution from a single redshift shell (zpp) and adds it to the total radiation fields. This
   is done by filling the arrays in RadiationFields.
*/
void accumulate_radiation_shell(float redshift, RadiationFieldsSetup *rad_setup,
                                RadiationFields *radiation_fields, int R_ct) {
    index_huge box_ct;
    double z_edge_factor, dzpp, zpp, xray_R_factor;
    double lya_flux_continuum_prefactor_mini = 0., lya_flux_injected_prefactor_mini = 0.,
           lya_flux_continuum_injected_prefactor_mini = 0.;

    zpp = rad_setup->zpp_avg[R_ct];
    if (R_ct == 0) {
        dzpp = redshift - rad_setup->zpp_edges[0];
    } else {
        dzpp = rad_setup->zpp_edges[R_ct - 1] - rad_setup->zpp_edges[R_ct];
    }
    // dtdz'' dz'' -> dR for the radius sum (c included in constants)
    z_edge_factor = fabs(dzpp * dtdz(zpp));

    xray_R_factor = pow(1 + zpp, -(astro_params_global->X_RAY_SPEC_INDEX));

    // minihalo factors should be separated since they may not be allocated
    if (astro_options_global->USE_MINI_HALOS) {
        if (astro_options_global->USE_LYA_HEATING) {
            lya_flux_continuum_prefactor_mini = rad_setup->lya_flux_continuum_prefactor_MINI[R_ct];
            lya_flux_injected_prefactor_mini = rad_setup->lya_flux_injected_prefactor_MINI[R_ct];
        } else {
            lya_flux_continuum_injected_prefactor_mini =
                rad_setup->lya_flux_continuum_injected_prefactor_MINI[R_ct];
        }
    }

// NOTE: The ionisation box has a final delta dependence of (1+delta_source)/(1+delta_absorber)
//   But here it's just (1+delta_source). This is for photon conservation.
//   If we assume attenuation at mean density as we do in nu_tau_one(), we HAVE to assume mean
//   density absorption otherwise we do not conserve photons
#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
    {
        // private variables
        int xidx, freq_table_index;
        double ival;
        double freq_int_heat, freq_int_ion, freq_int_lya;
        double sfr_term, xray_sfr;
        double sfr_term_mini = 0;
        double sfr_term_lw, sfr_term_mini_lw;
        int num_R = astro_params_global->N_STEP_TS;
#pragma omp for
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            sfr_term = rad_setup->filtered_sfr[box_ct] * z_edge_factor;
            // Minihalos and s->yr conversion are already included here
            xray_sfr = rad_setup->filtered_xray[box_ct] * z_edge_factor * xray_R_factor * 1e38;
            if (astro_options_global->USE_MINI_HALOS &&
                astro_options_global->LYA_MULTIPLE_SCATTERING) {
                sfr_term_lw = rad_setup->filtered_sfr_lw[box_ct] * z_edge_factor;
            } else {
                sfr_term_lw = sfr_term;
            }
            if (astro_options_global->USE_MINI_HALOS) {
                sfr_term_mini = rad_setup->filtered_sfr_mini[box_ct] * z_edge_factor;
                if (astro_options_global->LYA_MULTIPLE_SCATTERING) {
                    sfr_term_mini_lw = rad_setup->filtered_sfr_mini_lw[box_ct] * z_edge_factor;
                } else {
                    sfr_term_mini_lw = sfr_term_mini;
                }
            }

            // Evaluate the frequency integrals for this shell (R_ct) and cell (box_ct) via
            // linear interpolation
            xidx = rad_setup->m_xHII_low_box[box_ct];
            ival = rad_setup->inverse_val_box[box_ct];
            freq_table_index = freq_index(xidx, R_ct, num_R);
            freq_int_heat = rad_setup->freq_int_heat_tbl_diff[freq_table_index] * ival +
                            rad_setup->freq_int_heat_tbl[freq_table_index];
            freq_int_ion = rad_setup->freq_int_ion_tbl_diff[freq_table_index] * ival +
                           rad_setup->freq_int_ion_tbl[freq_table_index];
            freq_int_lya = rad_setup->freq_int_lya_tbl_diff[freq_table_index] * ival +
                           rad_setup->freq_int_lya_tbl[freq_table_index];

            // Evaluate the radiation fields by adding the contribution from this shell
            // (R_ct) This implements trapezoidal integration over the shells
            if (astro_options_global->USE_X_RAY_HEATING) {
                radiation_fields->xray_heating_rate[box_ct] += xray_sfr * freq_int_heat;
            }
            radiation_fields->xray_ionization_rate[box_ct] += xray_sfr * freq_int_ion;
            radiation_fields->xray_lya_flux[box_ct] += xray_sfr * freq_int_lya;
            if (astro_options_global->USE_MINI_HALOS) {
                radiation_fields->lyw_flux[box_ct] +=
                    sfr_term_lw * rad_setup->lyw_flux_prefactor[R_ct] +
                    sfr_term_mini_lw * rad_setup->lyw_flux_prefactor_MINI[R_ct];
            }
            if (astro_options_global->USE_LYA_HEATING) {
                radiation_fields->lya_flux_continuum[box_ct] +=
                    sfr_term * rad_setup->lya_flux_continuum_prefactor[R_ct] +
                    sfr_term_mini * lya_flux_continuum_prefactor_mini;
                radiation_fields->lya_flux_injected[box_ct] +=
                    sfr_term * rad_setup->lya_flux_injected_prefactor[R_ct] +
                    sfr_term_mini * lya_flux_injected_prefactor_mini;
            } else {
                radiation_fields->lya_flux_continuum_injected[box_ct] +=
                    sfr_term * rad_setup->lya_flux_continuum_injected_prefactor[R_ct] +
                    sfr_term_mini * lya_flux_continuum_injected_prefactor_mini;
            }
        }
    }
}

/*
    This function multiplies the radiation fields by the appropriate constants to convert them to
   physical meaningful quantities. The radiation fields are all given by an integral over the past
   source emissivities. Numerically, this integral is evaluated via the trapezoidal rule, which is
   done by summing over the contributions from each redshift shell. For efficiency, the constants
   are not included in the integral, but rather multiplied at the end (outside the integral). This
   function does that multiplication.
*/
void multiply_radiation_fields_by_constants(float redshift, RadiationFields *radiation_fields,
                                            PerturbedField *perturbed_field,
                                            TsBox *previous_spin_temp) {
    double luminosity_converstion_factor, xray_prefactor, volunit_inv, Nb_zp, lya_star_prefactor;

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
    xray_prefactor = luminosity_converstion_factor /
                     ((astro_params_global->NU_X_THRESH) * physconst.eV_to_Hz) * physconst.c_cms *
                     pow(1 + redshift, astro_params_global->X_RAY_SPEC_INDEX + 3);
    Nb_zp = N_b0 * (1 + redshift) * (1 + redshift) * (1 + redshift);
    // converts SFR density -> stellar baryon density + prefactors
    lya_star_prefactor = physconst.c_cms / (4.0 * M_PI) * physconst.Msun / physconst.m_p *
                         (1 - 0.75 * cosmo_params_global->Y_He);

    // converts the grid emissivity unit to per cm-3
    volunit_inv = pow(physconst.cm_per_Mpc, -3);

    index_huge box_ct;
#pragma omp parallel private(box_ct) num_threads(simulation_options_global -> N_THREADS)
    {
        double curr_delta, prev_xe;
#pragma omp for
        for (box_ct = 0; box_ct < HII_TOT_NUM_PIXELS; box_ct++) {
            curr_delta = perturbed_field->density[box_ct];
            prev_xe = previous_spin_temp->xray_ionised_fraction[box_ct];
            if (astro_options_global->USE_X_RAY_HEATING) {
                radiation_fields->xray_heating_rate[box_ct] *=
                    xray_prefactor * volunit_inv * 2.0 / 3.0 / physconst.k_B / (1.0 + prev_xe);
                ;
            }
            radiation_fields->xray_ionization_rate[box_ct] *= xray_prefactor * volunit_inv;
            radiation_fields->xray_lya_flux[box_ct] *=
                xray_prefactor * volunit_inv * Nb_zp * (1 + curr_delta);
            if (astro_options_global->USE_MINI_HALOS) {
                radiation_fields->lyw_flux[box_ct] *=
                    lya_star_prefactor * volunit_inv * physconst.h_p * 1e21;
            }
            if (astro_options_global->USE_LYA_HEATING) {
                radiation_fields->lya_flux_continuum[box_ct] *= lya_star_prefactor * volunit_inv;
                radiation_fields->lya_flux_injected[box_ct] *= lya_star_prefactor * volunit_inv;
            } else {
                radiation_fields->lya_flux_continuum_injected[box_ct] *=
                    lya_star_prefactor * volunit_inv;
            }
        }
    }
}

void one_annular_filter(float *input_box, float *output_box, double R_inner, double R_outer,
                        double R_star, int filter_type, double *u_avg, double *f_avg) {
    int i, j, k;
    index_huge ct;
    double unfiltered_avg = 0;
    double filtered_avg = 0;
    int box_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};

    // This dummy box is used to store both filtered and unfiltered boxes, since we don't need to
    // keep both at the same time
    fftwf_complex *dummy_box =
        (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS) \
    reduction(+ : unfiltered_avg)
    {
        float curr_val;
        index_huge index_r, index_f;
#pragma omp for
        for (i = 0; i < box_dim[0]; i++) {
            for (j = 0; j < box_dim[1]; j++) {
                for (k = 0; k < box_dim[2]; k++) {
                    index_r = grid_index_general(i, j, k, box_dim);
                    index_f = grid_index_fftw_r(i, j, k, box_dim);
                    curr_val = input_box[index_r];
                    *((float *)dummy_box + index_f) = curr_val;
                    unfiltered_avg += curr_val;
                }
            }
        }
    }
    // No need to filter the box if we only have one cell, or if we are the cell scale (corrsponds
    // to R_inner = 0, based on the logic in UpdateRadiationFields)
    if (simulation_options_global->HII_DIM > 1 && R_inner > 0) {
        // Transform unfiltered box to k-space to prepare for filtering
        // this would normally only be done once but we're using a different redshift for each R now
        dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, dummy_box);

// remember to add the factor of VOLUME/TOT_NUM_PIXELS when converting from real space to k-space
// Note: we will leave off factor of VOLUME, in anticipation of the inverse FFT below
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
        {
#pragma omp for
            for (ct = 0; ct < HII_KSPACE_NUM_PIXELS; ct++) {
                dummy_box[ct] /= (float)HII_TOT_NUM_PIXELS;
            }
        }

        // Don't filter on the cell scale
        filter_box(dummy_box, box_dim, filter_type, R_inner, R_outer, R_star);

        // now fft back to real space
        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, dummy_box);
    }
// copy over the values
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS) \
    reduction(+ : filtered_avg)
    {
        float curr_val;
        index_huge index_f, index_r;
#pragma omp for
        for (i = 0; i < box_dim[0]; i++) {
            for (j = 0; j < box_dim[1]; j++) {
                for (k = 0; k < box_dim[2]; k++) {
                    index_r = grid_index_general(i, j, k, box_dim);
                    index_f = grid_index_fftw_r(i, j, k, box_dim);
                    curr_val = *((float *)dummy_box + index_f);
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

    fftwf_free(dummy_box);
}

int UpdateRadiationFields(float redshift, HaloBox *halobox, int R_ct, double R_star,
                          PerturbedField *perturbed_field, TsBox *previous_spin_temp,
                          RadiationFieldsSetup *rad_setup, RadiationFields *radiation_fields) {
    int status;
    Try {  // This Try{} wraps the whole function.
        // NOTE: we assume that the first iteration corresponds to the largest shell.
        // This is important because we must do some final steps after the last shell is
        // done, see comment below
        if (R_ct == astro_params_global->N_STEP_TS - 1) LOG_DEBUG("starting RadiationFields");

        double sfr_avg, fsfr_avg, sfr_avg_mini = 0., fsfr_avg_mini = 0.;
        double xray_avg, fxray_avg;
        int filter_type = astro_options_global->LYA_MULTIPLE_SCATTERING
                              ? FILTER_SPHERICAL_SHELL_MULTIPLE_SCATTERING
                              : FILTER_SPHERICAL_SHELL_STRAIGHT_LINE;

        double R_inner = R_ct == 0 ? 0 : rad_setup->R_values[R_ct - 1];
        double R_outer = rad_setup->R_values[R_ct];

        one_annular_filter(halobox->halo_sfr, rad_setup->filtered_sfr, R_inner, R_outer, R_star,
                           filter_type, &sfr_avg, &fsfr_avg);
        one_annular_filter(halobox->halo_xray, rad_setup->filtered_xray, R_inner, R_outer, R_star,
                           FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, &xray_avg, &fxray_avg);
        if (astro_options_global->USE_MINI_HALOS) {
            one_annular_filter(halobox->halo_sfr_mini, rad_setup->filtered_sfr_mini, R_inner,
                               R_outer, R_star, filter_type, &sfr_avg_mini, &fsfr_avg_mini);
            // In case of multiple scattering and mini-halos, we need to filter the SFRD
            // fields again for the the LW feedback, as these photons travel in straight
            // lines
            if (astro_options_global->LYA_MULTIPLE_SCATTERING) {
                one_annular_filter(halobox->halo_sfr, rad_setup->filtered_sfr_lw, R_inner, R_outer,
                                   R_star, FILTER_SPHERICAL_SHELL_STRAIGHT_LINE, &sfr_avg,
                                   &fsfr_avg);
                one_annular_filter(halobox->halo_sfr_mini, rad_setup->filtered_sfr_mini_lw, R_inner,
                                   R_outer, R_star, FILTER_SPHERICAL_SHELL_STRAIGHT_LINE,
                                   &sfr_avg_mini, &fsfr_avg_mini);
            }
        }

        LOG_SUPER_DEBUG("R = [%8.3f - %8.3f] | mean filtered sfr  = %10.3e unfiltered %10.3e",
                        R_inner, R_outer, fsfr_avg, sfr_avg);
        LOG_ULTRA_DEBUG("mean filtered xray = %10.3e unfiltered %10.3e", fxray_avg, xray_avg);
        if (astro_options_global->USE_MINI_HALOS) {
            LOG_SUPER_DEBUG("MINI: filtered sfr %10.3e unfiltered %10.3e", fsfr_avg_mini,
                            sfr_avg_mini);
        }

        // Given the filtered emissivities, we accumulate the contribution of this shell to
        // the radiation fields
        accumulate_radiation_shell(redshift, rad_setup, radiation_fields, R_ct);

        // At the final shell, multiply by constants and free the remaining arrays
        // NOTE: we assume that the last iteration corresponds to the smallest shell
        // This is important in order to be consistent with the shell ordering we make
        // in compute_radiation_fields in python
        if (R_ct == 0) {
            multiply_radiation_fields_by_constants(redshift, radiation_fields, perturbed_field,
                                                   previous_spin_temp);
            // free fftwf only if we have a full box (with more than one cell)
            if (simulation_options_global->HII_DIM > 1) {
                fftwf_forget_wisdom();
                fftwf_cleanup_threads();
                fftwf_cleanup();
            }
            LOG_DEBUG("finished RadiationFields");
        }
    }  // End of try
    Catch(status) { return (status); }
    return (0);
}
