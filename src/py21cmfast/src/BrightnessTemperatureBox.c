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

        omp_set_num_threads(simulation_options_global->N_THREADS);

        float const_factor, T_rad, pixel_x_HI, pixel_deltax, H;

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
#pragma omp parallel shared(const_factor, perturb_field, ionized_box, box, redshift, spin_temp, \
                                T_rad) private(i, j, k, pixel_deltax, pixel_x_HI)               \
    num_threads(simulation_options_global -> N_THREADS)
        {
#pragma omp for reduction(+ : ave)
            for (i = 0; i < simulation_options_global->HII_DIM; i++) {
                for (j = 0; j < simulation_options_global->HII_DIM; j++) {
                    for (k = 0; k < HII_D_PARA; k++) {
                        pixel_deltax = perturb_field->density[HII_R_INDEX(i, j, k)];
                        pixel_x_HI = ionized_box->neutral_fraction[HII_R_INDEX(i, j, k)];

                        box->brightness_temp[HII_R_INDEX(i, j, k)] =
                            const_factor * pixel_x_HI * (1 + pixel_deltax);

                        if (astro_options_global->USE_TS_FLUCT) {
                            // Converting the prefactors into the optical depth, tau. Factor of 1000
                            // is the conversion of spin temperature from K to mK
                            box->brightness_temp[HII_R_INDEX(i, j, k)] *=
                                (1. + redshift) /
                                (1000. * spin_temp->spin_temperature[HII_R_INDEX(i, j, k)]);
                            box->tau_21[HII_R_INDEX(i, j, k)] =
                                box->brightness_temp[HII_R_INDEX(i, j, k)];
                            box->brightness_temp[HII_R_INDEX(i, j, k)] =
                                (1. - exp(-box->brightness_temp[HII_R_INDEX(i, j, k)])) * 1000. *
                                (spin_temp->spin_temperature[HII_R_INDEX(i, j, k)] - T_rad) /
                                (1. + redshift);
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

        LOG_DEBUG("z = %.2f, ave Tb = %e", redshift, ave);

    }  // End of try
    Catch(status) { return (status); }

    return (0);
}
