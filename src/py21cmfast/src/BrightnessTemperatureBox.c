/*  */
#include "BrightnessTemperatureBox.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "dft.h"
#include "exceptions.h"
#include "indexing.h"
#include "logger.h"
#include "subcell_rsds.h"

float clip(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

void get_velocity_gradient(float *v, float *vel_gradient) {
    memcpy(vel_gradient, v, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

    dft_r2c_cube(matter_flags_global->USE_FFTW_WISDOM, matter_params_global->HII_DIM, HII_D_PARA,
                 matter_params_global->N_THREADS, (fftwf_complex *)vel_gradient);

    float k_z;
    int n_x, n_y, n_z;
#pragma omp parallel shared(vel_gradient) private(n_x, n_y, n_z, k_z) \
    num_threads(matter_params_global -> N_THREADS)
    {
#pragma omp for
        for (n_x = 0; n_x < matter_params_global->HII_DIM; n_x++) {
            for (n_y = 0; n_y < matter_params_global->HII_DIM; n_y++) {
                for (n_z = 0; n_z <= HII_MID_PARA; n_z++) {
                    k_z = (float)(n_z * DELTA_K_PARA);

                    // take partial deriavative along the line of sight
                    *((fftwf_complex *)vel_gradient + HII_C_INDEX(n_x, n_y, n_z)) *=
                        k_z * I / (float)HII_TOT_NUM_PIXELS;
                }
            }
        }
    }

    dft_c2r_cube(matter_flags_global->USE_FFTW_WISDOM, matter_params_global->HII_DIM, HII_D_PARA,
                 matter_params_global->N_THREADS, (fftwf_complex *)vel_gradient);
}

int ComputeBrightnessTemp(float redshift, TsBox *spin_temp, IonizedBox *ionized_box,
                          PerturbedField *perturb_field, BrightnessTemp *box) {
    int status;
    Try {  // Try block around whole function.
        LOG_DEBUG("Starting Brightness Temperature calculation for redshift %f", redshift);
        // Makes the parameter structs visible to a variety of functions/macros
        // Do each time to avoid Python garbage collection issues

        int i, j, k;
        double ave;

        ave = 0.;

        omp_set_num_threads(matter_params_global->N_THREADS);

        float *v = (float *)calloc(HII_TOT_FFT_NUM_PIXELS, sizeof(float));
        float *vel_gradient = (float *)calloc(HII_TOT_FFT_NUM_PIXELS, sizeof(float));

        float *x_pos = calloc(astro_params_global->N_RSD_STEPS, sizeof(float));
        float *x_pos_offset = calloc(astro_params_global->N_RSD_STEPS, sizeof(float));
        float **delta_T_RSD_LOS =
            (float **)calloc(matter_params_global->N_THREADS, sizeof(float *));
        for (i = 0; i < matter_params_global->N_THREADS; i++) {
            delta_T_RSD_LOS[i] = (float *)calloc(HII_D_PARA, sizeof(float));
        }

#pragma omp parallel shared(v, perturb_field) private(i, j, k) \
    num_threads(matter_params_global -> N_THREADS)
        {
#pragma omp for
            for (i = 0; i < matter_params_global->HII_DIM; i++) {
                for (j = 0; j < matter_params_global->HII_DIM; j++) {
                    for (k = 0; k < HII_D_PARA; k++) {
                        *((float *)v + HII_R_FFT_INDEX(i, j, k)) =
                            perturb_field->velocity_z[HII_R_INDEX(i, j, k)];
                    }
                }
            }
        }

        float gradient_component;

        double dvdx, max_v_deriv, H;
        double const_factor, T_rad, pixel_Ts_factor, pixel_x_HI, pixel_deltax;

        init_ps();

        T_rad = T_cmb * (1 + redshift);
        H = hubble(redshift);
        const_factor = 27 *
                       (cosmo_params_global->OMb * cosmo_params_global->hlittle *
                        cosmo_params_global->hlittle / 0.023) *
                       sqrt((0.15 / (cosmo_params_global->OMm) / (cosmo_params_global->hlittle) /
                             (cosmo_params_global->hlittle)) *
                            (1. + redshift) / 10.0);

        ///////////////////////////////  END INITIALIZATION
        ////////////////////////////////////////////////
        LOG_SUPER_DEBUG("Performed Initialization.");

        // ok, lets fill the delta_T box; which will be the same size as the bubble box
#pragma omp parallel shared(const_factor, perturb_field, ionized_box, box, redshift, spin_temp,    \
                                T_rad) private(i, j, k, pixel_deltax, pixel_x_HI, pixel_Ts_factor) \
    num_threads(matter_params_global -> N_THREADS)
        {
#pragma omp for reduction(+ : ave)
            for (i = 0; i < matter_params_global->HII_DIM; i++) {
                for (j = 0; j < matter_params_global->HII_DIM; j++) {
                    for (k = 0; k < HII_D_PARA; k++) {
                        pixel_deltax = perturb_field->density[HII_R_INDEX(i, j, k)];
                        pixel_x_HI = ionized_box->xH_box[HII_R_INDEX(i, j, k)];

                        box->brightness_temp[HII_R_INDEX(i, j, k)] =
                            const_factor * pixel_x_HI * (1 + pixel_deltax);

                        if (astro_flags_global->USE_TS_FLUCT) {
                            if (astro_flags_global->SUBCELL_RSD) {
                                // Converting the prefactors into the optical depth, tau. Factor of
                                // 1000 is the conversion of spin temperature from K to mK
                                box->brightness_temp[HII_R_INDEX(i, j, k)] *=
                                    (1. + redshift) /
                                    (1000. * spin_temp->Ts_box[HII_R_INDEX(i, j, k)]);
                            } else {
                                pixel_Ts_factor =
                                    (1 - T_rad / spin_temp->Ts_box[HII_R_INDEX(i, j, k)]);
                                box->brightness_temp[HII_R_INDEX(i, j, k)] *= pixel_Ts_factor;
                            }
                        }

                        ave += box->brightness_temp[HII_R_INDEX(i, j, k)];
                    }
                }
            }
        }

        LOG_SUPER_DEBUG("Filled delta_T.");

        if (isfinite(ave) == 0) {
            LOG_ERROR("Average brightness temperature is infinite or NaN!");
            Throw(InfinityorNaNError);
        }

        ave /= (float)HII_TOT_NUM_PIXELS;

        // Get RSDs
        if (astro_flags_global->APPLY_RSDS) {
            ave = 0.;
            get_velocity_gradient(v, vel_gradient);

            // now add the velocity correction to the delta_T maps (only used for T_S >> T_CMB
            // case).
            max_v_deriv = fabs(astro_params_global->MAX_DVDR * H);

            if (!(astro_flags_global->USE_TS_FLUCT && astro_flags_global->SUBCELL_RSD)) {
// Do this unless we are doing BOTH Ts fluctuations and subcell RSDs
#pragma omp parallel shared(vel_gradient, T_rad, redshift, spin_temp, box, max_v_deriv) private( \
        i, j, k, gradient_component, dvdx) num_threads(matter_params_global -> N_THREADS)
                {
#pragma omp for
                    for (i = 0; i < matter_params_global->HII_DIM; i++) {
                        for (j = 0; j < matter_params_global->HII_DIM; j++) {
                            for (k = 0; k < HII_D_PARA; k++) {
                                dvdx = clip(vel_gradient[HII_R_FFT_INDEX(i, j, k)], -max_v_deriv,
                                            max_v_deriv);
                                box->brightness_temp[HII_R_INDEX(i, j, k)] /= (dvdx / H + 1.0);
                                ave += box->brightness_temp[HII_R_INDEX(i, j, k)];
                            }
                        }
                    }
                }
                ave /= (float)HII_TOT_NUM_PIXELS;
            } else {
// This is if we're doing both TS_FLUCT and SUBCELL_RSD
#pragma omp parallel shared(vel_gradient, T_rad, redshift, spin_temp, box, max_v_deriv) private( \
        i, j, k, gradient_component, dvdx) num_threads(matter_params_global -> N_THREADS)
                {
#pragma omp for
                    for (i = 0; i < matter_params_global->HII_DIM; i++) {
                        for (j = 0; j < matter_params_global->HII_DIM; j++) {
                            for (k = 0; k < HII_D_PARA; k++) {
                                gradient_component =
                                    fabs(vel_gradient[HII_R_FFT_INDEX(i, j, k)] / H + 1.0);

                                // Calculate the brightness temperature, using the optical depth
                                if (gradient_component < FRACT_FLOAT_ERR) {
                                    // Gradient component goes to zero, optical depth diverges.
                                    // But, since we take exp(-tau), this goes to zero and (1 -
                                    // exp(-tau)) goes to unity. Again, factors of 1000. are
                                    // conversions from K to mK
                                    box->brightness_temp[HII_R_INDEX(i, j, k)] =
                                        1000. * (spin_temp->Ts_box[HII_R_INDEX(i, j, k)] - T_rad) /
                                        (1. + redshift);
                                } else {
                                    box->brightness_temp[HII_R_INDEX(i, j, k)] =
                                        (1. - exp(-box->brightness_temp[HII_R_INDEX(i, j, k)] /
                                                  gradient_component)) *
                                        1000. * (spin_temp->Ts_box[HII_R_INDEX(i, j, k)] - T_rad) /
                                        (1. + redshift);
                                }
                            }
                        }
                    }
                }
            }

            if (astro_flags_global->SUBCELL_RSD) {
                ave = apply_subcell_rsds(ionized_box, box, redshift, spin_temp, T_rad, v, H);
            }
        }

        LOG_SUPER_DEBUG("Included velocities.");

        if (isfinite(ave) == 0) {
            LOG_ERROR(
                "Average brightness temperature (after including velocities) is infinite or NaN!");
            //        Throw(ParameterError);
            Throw(InfinityorNaNError);
        }

        LOG_DEBUG("z = %.2f, ave Tb = %e", redshift, ave);

        free(v);
        free(vel_gradient);

        free(x_pos);
        free(x_pos_offset);
        for (i = 0; i < matter_params_global->N_THREADS; i++) {
            free(delta_T_RSD_LOS[i]);
        }
        free(delta_T_RSD_LOS);
        fftwf_cleanup_threads();
        fftwf_cleanup();
        LOG_SUPER_DEBUG("Cleaned up.");

    }  // End of try
    Catch(status) { return (status); }

    return (0);
}
