// Re-write of perturb_field.c for being accessible within the MCMC
#include "PerturbField.h"

#include <complex.h>
#include <fftw3.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cexcept.h"
#include "cosmology.h"
#include "debugging.h"
#include "dft.h"
#include "exceptions.h"
#include "filtering.h"
#include "indexing.h"
#include "logger.h"

void compute_perturbed_velocities(unsigned short axis, fftwf_complex *HIRES_density_perturb,
                                  fftwf_complex *HIRES_density_perturb_saved,
                                  fftwf_complex *LOWRES_density_perturb,
                                  fftwf_complex *LOWRES_density_perturb_saved, float dDdt_over_D,
                                  int dimension, int switch_mid, float f_pixel_factor,
                                  float *velocity) {
    float k_x, k_y, k_z, k_sq;
    int n_x, n_y, n_z;
    int i, j, k;

    float kvec[3];

    if (matter_options_global->PERTURB_ON_HIGH_RES) {
        // We are going to generate the velocity field on the high-resolution perturbed
        // density grid
        memcpy(HIRES_density_perturb, HIRES_density_perturb_saved,
               sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
    } else {
        // We are going to generate the velocity field on the low-resolution perturbed density grid
        memcpy(LOWRES_density_perturb, LOWRES_density_perturb_saved,
               sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
        LOG_SUPER_DEBUG("dDdt_over_D=%.6e, dimension=%d, switch_mid=%d, f_pixel_factor=%f",
                        dDdt_over_D, dimension, switch_mid, f_pixel_factor);
    }

#pragma omp parallel shared(LOWRES_density_perturb, HIRES_density_perturb, dDdt_over_D, dimension, \
                                switch_mid) private(n_x, n_y, n_z, k_x, k_y, k_z, k_sq, kvec)      \
    num_threads(simulation_options_global -> N_THREADS)
    {
#pragma omp for
        for (n_x = 0; n_x < dimension; n_x++) {
            if (n_x > switch_mid)
                k_x = (n_x - dimension) * DELTA_K;  // wrap around for FFT convention
            else
                k_x = n_x * DELTA_K;

            for (n_y = 0; n_y < dimension; n_y++) {
                if (n_y > switch_mid)
                    k_y = (n_y - dimension) * DELTA_K;
                else
                    k_y = n_y * DELTA_K;

                for (n_z = 0;
                     n_z <=
                     (unsigned long long)(simulation_options_global->NON_CUBIC_FACTOR * switch_mid);
                     n_z++) {
                    k_z = n_z * DELTA_K_PARA;

                    kvec[0] = k_x;
                    kvec[1] = k_y;
                    kvec[2] = k_z;

                    k_sq = k_x * k_x + k_y * k_y + k_z * k_z;

                    // now set the velocities
                    if ((n_x == 0) && (n_y == 0) && (n_z == 0)) {  // DC mode
                        if (matter_options_global->PERTURB_ON_HIGH_RES) {
                            HIRES_density_perturb[0] = 0;
                        } else {
                            LOWRES_density_perturb[0] = 0;
                        }
                    } else {
                        if (matter_options_global->PERTURB_ON_HIGH_RES) {
                            HIRES_density_perturb[C_INDEX(n_x, n_y, n_z)] *=
                                dDdt_over_D * kvec[axis] * I / k_sq / (TOT_NUM_PIXELS + 0.0);
                        } else {
                            LOWRES_density_perturb[HII_C_INDEX(n_x, n_y, n_z)] *=
                                dDdt_over_D * kvec[axis] * I / k_sq / (HII_TOT_NUM_PIXELS + 0.0);
                        }
                    }
                }
            }
        }
    }

    LOG_SUPER_DEBUG("density_perturb after modification by dDdt: ");
    debugSummarizeBoxComplex((float complex *)LOWRES_density_perturb,
                             simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                             (int)(HII_D_PARA / 2), "  ");

    if (matter_options_global->PERTURB_ON_HIGH_RES) {
        // smooth the high resolution field ready for resampling
        if (simulation_options_global->DIM != simulation_options_global->HII_DIM)
            filter_box(HIRES_density_perturb, 0, 0,
                       L_FACTOR * simulation_options_global->BOX_LEN /
                           (simulation_options_global->HII_DIM + 0.0),
                       0.);

        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM, D_PARA,
                     simulation_options_global->N_THREADS, HIRES_density_perturb);

#pragma omp parallel shared(velocity, HIRES_density_perturb, f_pixel_factor) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
        {
#pragma omp for
            for (i = 0; i < simulation_options_global->HII_DIM; i++) {
                for (j = 0; j < simulation_options_global->HII_DIM; j++) {
                    for (k = 0; k < HII_D_PARA; k++) {
                        *((float *)velocity + HII_R_INDEX(i, j, k)) =
                            *((float *)HIRES_density_perturb +
                              R_FFT_INDEX((unsigned long long)(i * f_pixel_factor + 0.5),
                                          (unsigned long long)(j * f_pixel_factor + 0.5),
                                          (unsigned long long)(k * f_pixel_factor + 0.5)));
                    }
                }
            }
        }
    } else {
        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, LOWRES_density_perturb);

#pragma omp parallel shared(velocity, LOWRES_density_perturb) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
        {
#pragma omp for
            for (i = 0; i < simulation_options_global->HII_DIM; i++) {
                for (j = 0; j < simulation_options_global->HII_DIM; j++) {
                    for (k = 0; k < HII_D_PARA; k++) {
                        *((float *)velocity + HII_R_INDEX(i, j, k)) =
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k));
                    }
                }
            }
        }
    }
    LOG_SUPER_DEBUG("velocity: ");
    debugSummarizeBox(velocity, simulation_options_global->HII_DIM,
                      simulation_options_global->HII_DIM, HII_D_PARA, "  ");
}

int ComputePerturbField(float redshift, InitialConditions *boxes, PerturbedField *perturbed_field) {
    /*
     ComputePerturbField uses the first-order Langragian displacement field to move the
     masses in the cells of the density field. The high-res density field is extrapolated
     to some high-redshift (simulation_options.INITIAL_REDSHIFT), then uses the zeldovich
     approximation to move the grid "particles" onto the lower-res grid we use for the
     maps. Then we recalculate the velocity fields on the perturbed grid.
    */

    int status;
    Try {  // This Try{} wraps the whole function, so we don't indent.

        // Makes the parameter structs visible to a variety of functions/macros
        // Do each time to avoid Python garbage collection issues

#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
        writeSimulationOptions(simulation_options_global);
        writeCosmoParams(cosmo_params_global);
#endif

        omp_set_num_threads(simulation_options_global->N_THREADS);

        fftwf_complex *HIRES_density_perturb = NULL, *HIRES_density_perturb_saved;
        fftwf_complex *LOWRES_density_perturb, *LOWRES_density_perturb_saved;

        float growth_factor, displacement_factor_2LPT, init_growth_factor,
            init_displacement_factor_2LPT;
        float mass_factor, dDdt, f_pixel_factor, velocity_displacement_factor,
            velocity_displacement_factor_2LPT;
        int i, j, k, xi, yi, zi, dimension, dimension_z, switch_mid;

        // Function for deciding the dimensions of loops when we could
        // use either the low or high resolution grids.
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            dimension = simulation_options_global->DIM;
            dimension_z =
                simulation_options_global->DIM * simulation_options_global->NON_CUBIC_FACTOR;
            switch_mid = MIDDLE;
        } else {
            dimension = simulation_options_global->HII_DIM;
            dimension_z =
                simulation_options_global->HII_DIM * simulation_options_global->NON_CUBIC_FACTOR;
            switch_mid = HII_MIDDLE;
        }

        // ***************   BEGIN INITIALIZATION   ************************** //

        LOG_DEBUG("Computing Perturbed Field at z=%.3f", redshift);
        // perform a very rudimentary check to see if we are underresolved and not using the linear
        // approx
        if ((simulation_options_global->BOX_LEN > simulation_options_global->DIM) &&
            matter_options_global->PERTURB_ALGORITHM > 0) {
            LOG_WARNING(
                "Resolution is likely too low for accurate evolved density fields\n \
                It is recommended that you either increase the resolution (DIM/BOX_LEN) or set PERTURB_ALGORITHM to 'LINEAR'\n");
        }

        growth_factor = dicke(redshift);
        displacement_factor_2LPT = -(3.0 / 7.0) * growth_factor * growth_factor;  // 2LPT eq. D8

        dDdt = ddickedt(redshift);  // time derivative of the growth factor (1/s)
        init_growth_factor = dicke(simulation_options_global->INITIAL_REDSHIFT);
        init_displacement_factor_2LPT =
            -(3.0 / 7.0) * init_growth_factor * init_growth_factor;  // 2LPT eq. D8

        // find factor of HII pixel size / deltax pixel size
        f_pixel_factor =
            simulation_options_global->DIM / (float)(simulation_options_global->HII_DIM);
        mass_factor = pow(f_pixel_factor, 3);

        // allocate memory for the updated density, and initialize
        LOWRES_density_perturb =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
        LOWRES_density_perturb_saved =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            HIRES_density_perturb =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
            HIRES_density_perturb_saved =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
        }

        double *resampled_box;

        // TODO: debugSummarizeIC is bugged when not all the fields are in memory
        //  debugSummarizeIC(boxes, simulation_options_global->HII_DIM,
        //  simulation_options_global->DIM, simulation_options_global->NON_CUBIC_FACTOR);
        LOG_SUPER_DEBUG(
            "growth_factor=%f, displacemet_factor_2LPT=%f, dDdt=%f, init_growth_factor=%f, "
            "init_displacement_factor_2LPT=%f, mass_factor=%f",
            growth_factor, displacement_factor_2LPT, dDdt, init_growth_factor,
            init_displacement_factor_2LPT, mass_factor);

        // check if the linear evolution flag was set
        if (matter_options_global->PERTURB_ALGORITHM == 0) {
#pragma omp parallel shared(growth_factor, boxes, LOWRES_density_perturb, HIRES_density_perturb, \
                                dimension) private(i, j, k)                                      \
    num_threads(simulation_options_global -> N_THREADS)
            {
#pragma omp for
                for (i = 0; i < dimension; i++) {
                    for (j = 0; j < dimension; j++) {
                        for (k = 0; k < dimension_z; k++) {
                            if (matter_options_global->PERTURB_ON_HIGH_RES) {
                                *((float *)HIRES_density_perturb + R_FFT_INDEX(i, j, k)) =
                                    growth_factor * boxes->hires_density[R_INDEX(i, j, k)];
                            } else {
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) =
                                    growth_factor * boxes->lowres_density[HII_R_INDEX(i, j, k)];
                            }
                        }
                    }
                }
            }
        } else {
            // Apply Zel'dovich/2LPT correction

#pragma omp parallel shared(LOWRES_density_perturb, HIRES_density_perturb, dimension) private( \
        i, j, k) num_threads(simulation_options_global -> N_THREADS)
            {
#pragma omp for
                for (i = 0; i < dimension; i++) {
                    for (j = 0; j < dimension; j++) {
                        for (k = 0; k < dimension_z; k++) {
                            if (matter_options_global->PERTURB_ON_HIGH_RES) {
                                *((float *)HIRES_density_perturb + R_FFT_INDEX(i, j, k)) = 0.;
                            } else {
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) = 0.;
                            }
                        }
                    }
                }
            }

            velocity_displacement_factor =
                (growth_factor - init_growth_factor) / simulation_options_global->BOX_LEN;

            // now add the missing factor of D
#pragma omp parallel shared(boxes, velocity_displacement_factor, dimension) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
            {
#pragma omp for
                for (i = 0; i < dimension; i++) {
                    for (j = 0; j < dimension; j++) {
                        for (k = 0; k < dimension_z; k++) {
                            if (matter_options_global->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx[R_INDEX(i, j, k)] *=
                                    velocity_displacement_factor;  // this is now comoving
                                                                   // displacement in units of box
                                                                   // size
                                boxes->hires_vy[R_INDEX(i, j, k)] *=
                                    velocity_displacement_factor;  // this is now comoving
                                                                   // displacement in units of box
                                                                   // size
                                boxes->hires_vz[R_INDEX(i, j, k)] *=
                                    (velocity_displacement_factor /
                                     simulation_options_global
                                         ->NON_CUBIC_FACTOR);  // this is now comoving displacement
                                                               // in units of box size
                            } else {
                                boxes->lowres_vx[HII_R_INDEX(i, j, k)] *=
                                    velocity_displacement_factor;  // this is now comoving
                                                                   // displacement in units of box
                                                                   // size
                                boxes->lowres_vy[HII_R_INDEX(i, j, k)] *=
                                    velocity_displacement_factor;  // this is now comoving
                                                                   // displacement in units of box
                                                                   // size
                                boxes->lowres_vz[HII_R_INDEX(i, j, k)] *=
                                    (velocity_displacement_factor /
                                     simulation_options_global
                                         ->NON_CUBIC_FACTOR);  // this is now comoving displacement
                                                               // in units of box size
                            }
                        }
                    }
                }
            }

            // * ************************************************************************* * //
            // *                           BEGIN 2LPT PART                                 * //
            // * ************************************************************************* * //
            // reference: reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
            if (matter_options_global->PERTURB_ALGORITHM == 2) {
                // allocate memory for the velocity boxes and read them in
                velocity_displacement_factor_2LPT =
                    (displacement_factor_2LPT - init_displacement_factor_2LPT) /
                    simulation_options_global->BOX_LEN;

                // now add the missing factor in eq. D9
#pragma omp parallel shared(boxes, velocity_displacement_factor_2LPT, dimension) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
                {
#pragma omp for
                    for (i = 0; i < dimension; i++) {
                        for (j = 0; j < dimension; j++) {
                            for (k = 0; k < dimension_z; k++) {
                                if (matter_options_global->PERTURB_ON_HIGH_RES) {
                                    // this is now comoving displacement in units of box size
                                    boxes->hires_vx_2LPT[R_INDEX(i, j, k)] *=
                                        velocity_displacement_factor_2LPT;
                                    boxes->hires_vy_2LPT[R_INDEX(i, j, k)] *=
                                        velocity_displacement_factor_2LPT;
                                    boxes->hires_vz_2LPT[R_INDEX(i, j, k)] *=
                                        (velocity_displacement_factor_2LPT /
                                         simulation_options_global->NON_CUBIC_FACTOR);
                                } else {
                                    boxes->lowres_vx_2LPT[HII_R_INDEX(i, j, k)] *=
                                        velocity_displacement_factor_2LPT;
                                    boxes->lowres_vy_2LPT[HII_R_INDEX(i, j, k)] *=
                                        velocity_displacement_factor_2LPT;
                                    boxes->lowres_vz_2LPT[HII_R_INDEX(i, j, k)] *=
                                        (velocity_displacement_factor_2LPT /
                                         simulation_options_global->NON_CUBIC_FACTOR);
                                }
                            }
                        }
                    }
                }
            }

            // * ************************************************************************* * //
            // *                            END 2LPT PART                                  * //
            // * ************************************************************************* * //

            // ************  END INITIALIZATION **************************** //

            // Perturbing the density field required adding over multiple cells. Store intermediate
            // result as a double to avoid rounding errors
            if (matter_options_global->PERTURB_ON_HIGH_RES) {
                resampled_box = (double *)calloc(TOT_NUM_PIXELS, sizeof(double));
            } else {
                resampled_box = (double *)calloc(HII_TOT_NUM_PIXELS, sizeof(double));
            }

            // If using GPU, call CUDA function
            LOG_DEBUG("Perturb the density field");
            bool use_cuda = false;  // pass this as a parameter later
            if (use_cuda) {
#if CUDA_FOUND
                resampled_box = MapMass_gpu(boxes, resampled_box, dimension, f_pixel_factor,
                                            init_growth_factor);
#else
                LOG_ERROR("CUDA version of MapMass() called but code was not compiled for CUDA.");
#endif
            } else {
                resampled_box = MapMass_cpu(boxes, resampled_box, dimension, f_pixel_factor,
                                            init_growth_factor);
            }

            LOG_SUPER_DEBUG("resampled_box: ");
            debugSummarizeBoxDouble(resampled_box, dimension, dimension, dimension_z, "  ");

// Resample back to a float for remaining algorithm
#pragma omp parallel shared(LOWRES_density_perturb, HIRES_density_perturb, resampled_box, \
                                dimension) private(i, j, k)                               \
    num_threads(simulation_options_global -> N_THREADS)
            {
#pragma omp for
                for (i = 0; i < dimension; i++) {
                    for (j = 0; j < dimension; j++) {
                        for (k = 0; k < dimension_z; k++) {
                            if (matter_options_global->PERTURB_ON_HIGH_RES) {
                                *((float *)HIRES_density_perturb + R_FFT_INDEX(i, j, k)) =
                                    (float)resampled_box[R_INDEX(i, j, k)];
                            } else {
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) =
                                    (float)resampled_box[HII_R_INDEX(i, j, k)];
                            }
                        }
                    }
                }
            }
            free(resampled_box);

            LOG_SUPER_DEBUG("density_perturb: ");
            if (matter_options_global->PERTURB_ON_HIGH_RES) {
                debugSummarizeBox((float *)HIRES_density_perturb, dimension, dimension,
                                  2 * (dimension_z / 2 + 1), "  ");
            } else {
                debugSummarizeBox((float *)LOWRES_density_perturb, dimension, dimension,
                                  2 * (dimension_z / 2 + 1), "  ");
            }

            // deallocate
#pragma omp parallel shared(boxes, velocity_displacement_factor, dimension) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
            {
#pragma omp for
                for (i = 0; i < dimension; i++) {
                    for (j = 0; j < dimension; j++) {
                        for (k = 0; k < dimension_z; k++) {
                            if (matter_options_global->PERTURB_ON_HIGH_RES) {
                                boxes->hires_vx[R_INDEX(i, j, k)] /=
                                    velocity_displacement_factor;  // convert back to z = 0 quantity
                                boxes->hires_vy[R_INDEX(i, j, k)] /=
                                    velocity_displacement_factor;  // convert back to z = 0 quantity
                                boxes->hires_vz[R_INDEX(i, j, k)] /=
                                    (velocity_displacement_factor /
                                     simulation_options_global
                                         ->NON_CUBIC_FACTOR);  // convert back to z = 0 quantity
                            } else {
                                boxes->lowres_vx[HII_R_INDEX(i, j, k)] /=
                                    velocity_displacement_factor;  // convert back to z = 0 quantity
                                boxes->lowres_vy[HII_R_INDEX(i, j, k)] /=
                                    velocity_displacement_factor;  // convert back to z = 0 quantity
                                boxes->lowres_vz[HII_R_INDEX(i, j, k)] /=
                                    (velocity_displacement_factor /
                                     simulation_options_global
                                         ->NON_CUBIC_FACTOR);  // convert back to z = 0 quantity
                            }
                        }
                    }
                }
            }

            if (matter_options_global->PERTURB_ALGORITHM == 2) {
#pragma omp parallel shared(boxes, velocity_displacement_factor_2LPT, dimension) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
                {
#pragma omp for
                    for (i = 0; i < dimension; i++) {
                        for (j = 0; j < dimension; j++) {
                            for (k = 0; k < dimension_z; k++) {
                                if (matter_options_global->PERTURB_ON_HIGH_RES) {
                                    // convert back to z = 0 quantity
                                    boxes->hires_vx_2LPT[R_INDEX(i, j, k)] /=
                                        velocity_displacement_factor_2LPT;

                                    boxes->hires_vy_2LPT[R_INDEX(i, j, k)] /=
                                        velocity_displacement_factor_2LPT;

                                    boxes->hires_vz_2LPT[R_INDEX(i, j, k)] /=
                                        (velocity_displacement_factor_2LPT /
                                         simulation_options_global->NON_CUBIC_FACTOR);
                                } else {
                                    boxes->lowres_vx_2LPT[HII_R_INDEX(i, j, k)] /=
                                        velocity_displacement_factor_2LPT;

                                    boxes->lowres_vy_2LPT[HII_R_INDEX(i, j, k)] /=
                                        velocity_displacement_factor_2LPT;

                                    boxes->lowres_vz_2LPT[HII_R_INDEX(i, j, k)] /=
                                        (velocity_displacement_factor_2LPT /
                                         simulation_options_global->NON_CUBIC_FACTOR);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Now, if I still have the high resolution density grid (HIRES_density_perturb) I need to
        // downsample it to the low-resolution grid
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            // Transform to Fourier space to sample (filter) the box
            dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                         D_PARA, simulation_options_global->N_THREADS, HIRES_density_perturb);

            // Need to save a copy of the high-resolution unfiltered density field for the
            // velocities
            memcpy(HIRES_density_perturb_saved, HIRES_density_perturb,
                   sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

            // Now filter the box
            if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
                filter_box(HIRES_density_perturb, 0, 0,
                           L_FACTOR * simulation_options_global->BOX_LEN /
                               (simulation_options_global->HII_DIM + 0.0),
                           0.);
            }

            // FFT back to real space
            dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                         D_PARA, simulation_options_global->N_THREADS, HIRES_density_perturb);

            // Renormalise the FFT'd box
#pragma omp parallel shared(HIRES_density_perturb, LOWRES_density_perturb, f_pixel_factor, \
                                mass_factor) private(i, j, k)                              \
    num_threads(simulation_options_global -> N_THREADS)
            {
#pragma omp for
                for (i = 0; i < simulation_options_global->HII_DIM; i++) {
                    for (j = 0; j < simulation_options_global->HII_DIM; j++) {
                        for (k = 0; k < HII_D_PARA; k++) {
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) =
                                *((float *)HIRES_density_perturb +
                                  R_FFT_INDEX((unsigned long long)(i * f_pixel_factor + 0.5),
                                              (unsigned long long)(j * f_pixel_factor + 0.5),
                                              (unsigned long long)(k * f_pixel_factor + 0.5))) /
                                (float)TOT_NUM_PIXELS;

                            if (matter_options_global->PERTURB_ALGORITHM > 0) {
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) -= 1.;
                            }

                            if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) <
                                -1) {
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) =
                                    -1. + FRACT_FLOAT_ERR;
                            }
                        }
                    }
                }
            }
        } else {
            if (matter_options_global->PERTURB_ALGORITHM > 0) {
#pragma omp parallel shared(LOWRES_density_perturb, mass_factor) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
                {
#pragma omp for
                    for (i = 0; i < simulation_options_global->HII_DIM; i++) {
                        for (j = 0; j < simulation_options_global->HII_DIM; j++) {
                            for (k = 0; k < HII_D_PARA; k++) {
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) /=
                                    mass_factor;
                                *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) -= 1.;
                            }
                        }
                    }
                }
            }
        }

        LOG_SUPER_DEBUG("LOWRES_density_perturb: ");
        debugSummarizeBox((float *)LOWRES_density_perturb, simulation_options_global->HII_DIM,
                          simulation_options_global->HII_DIM, 2 * (HII_D_PARA / 2 + 1), "  ");

        // transform to k-space
        dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, LOWRES_density_perturb);

        // smooth the field
        if (matter_options_global->PERTURB_ALGORITHM > 0 &&
            matter_options_global->SMOOTH_EVOLVED_DENSITY_FIELD) {
            filter_box(LOWRES_density_perturb, 1, 2,
                       simulation_options_global->DENSITY_SMOOTH_RADIUS *
                           simulation_options_global->BOX_LEN /
                           (float)simulation_options_global->HII_DIM,
                       0.);
        }

        LOG_SUPER_DEBUG("LOWRES_density_perturb after smoothing: ");
        debugSummarizeBox((float *)LOWRES_density_perturb, simulation_options_global->HII_DIM,
                          simulation_options_global->HII_DIM, 2 * (HII_D_PARA / 2 + 1), "  ");

        // save a copy of the k-space density field
        memcpy(LOWRES_density_perturb_saved, LOWRES_density_perturb,
               sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                     HII_D_PARA, simulation_options_global->N_THREADS, LOWRES_density_perturb);

        LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space: ");
        debugSummarizeBox((float *)LOWRES_density_perturb, simulation_options_global->HII_DIM,
                          simulation_options_global->HII_DIM, 2 * (HII_D_PARA / 2 + 1), "  ");

        // normalize after FFT
        int bad_count = 0;
#pragma omp parallel shared(LOWRES_density_perturb) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS) reduction(+ : bad_count)
        {
#pragma omp for
            for (i = 0; i < simulation_options_global->HII_DIM; i++) {
                for (j = 0; j < simulation_options_global->HII_DIM; j++) {
                    for (k = 0; k < HII_D_PARA; k++) {
                        *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) /=
                            (float)HII_TOT_NUM_PIXELS;

                        if (*((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) <
                            -1.0) {  // shouldn't happen

                            if (bad_count < 5)
                                LOG_WARNING(
                                    "LOWRES_density_perturb is <-1 for index %d %d %d (value=%f)",
                                    i, j, k,
                                    *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)));
                            if (bad_count == 5)
                                LOG_WARNING(
                                    "Skipping further warnings for LOWRES_density_perturb.");
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) =
                                -1 + FRACT_FLOAT_ERR;
                            bad_count++;
                        }
                    }
                }
            }
        }
        if (bad_count >= 5)
            LOG_WARNING("Total number of bad indices for LOW_density_perturb: %d", bad_count);
        LOG_SUPER_DEBUG("LOWRES_density_perturb back in real space (normalized): ");
        debugSummarizeBox((float *)LOWRES_density_perturb, simulation_options_global->HII_DIM,
                          simulation_options_global->HII_DIM, 2 * (HII_D_PARA / 2 + 1), "  ");

#pragma omp parallel shared(perturbed_field, LOWRES_density_perturb) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
        {
#pragma omp for
            for (i = 0; i < simulation_options_global->HII_DIM; i++) {
                for (j = 0; j < simulation_options_global->HII_DIM; j++) {
                    for (k = 0; k < HII_D_PARA; k++) {
                        *((float *)perturbed_field->density + HII_R_INDEX(i, j, k)) =
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k));
                    }
                }
            }
        }

        // ****  Convert to velocities ***** //
        float dDdt_over_D;

        dDdt_over_D = dDdt / growth_factor;

        if (matter_options_global->KEEP_3D_VELOCITIES) {
            compute_perturbed_velocities(0, HIRES_density_perturb, HIRES_density_perturb_saved,
                                         LOWRES_density_perturb, LOWRES_density_perturb_saved,
                                         dDdt_over_D, dimension, switch_mid, f_pixel_factor,
                                         perturbed_field->velocity_x);
            compute_perturbed_velocities(1, HIRES_density_perturb, HIRES_density_perturb_saved,
                                         LOWRES_density_perturb, LOWRES_density_perturb_saved,
                                         dDdt_over_D, dimension, switch_mid, f_pixel_factor,
                                         perturbed_field->velocity_y);
        }

        compute_perturbed_velocities(2, HIRES_density_perturb, HIRES_density_perturb_saved,
                                     LOWRES_density_perturb, LOWRES_density_perturb_saved,
                                     dDdt_over_D, dimension, switch_mid, f_pixel_factor,
                                     perturbed_field->velocity_z);

        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();

        // deallocate
        fftwf_free(LOWRES_density_perturb);
        fftwf_free(LOWRES_density_perturb_saved);
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            fftwf_free(HIRES_density_perturb);
            fftwf_free(HIRES_density_perturb_saved);
        }
        fftwf_cleanup();
        LOG_DEBUG("Done.");

    }  // End of Try{}
    Catch(status) { return (status); }

    return (0);
}
