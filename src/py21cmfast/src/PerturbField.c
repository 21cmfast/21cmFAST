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

static inline unsigned long long grid_index_general(int x, int y, int z, int dim[3]) {
    return (z) + (dim[2] + 0llu) * (y + (dim[1] + 0llu) * x);
}

static inline unsigned long long grid_index_fftw_r(int x, int y, int z, int dim[3]) {
    return (z) + 2llu * (dim[2] / 2 + 1llu) * (y + (dim[1] + 0llu) * x);
}

static inline unsigned long long grid_index_fftw_c(int x, int y, int z, int dim[3]) {
    return (z) + (dim[2] / 2 + 1llu) * (y + (dim[1] + 0llu) * x);
}

static inline void do_cic_interpolation(double *resampled_box, double pos[3], int box_dim[3],
                                        double curr_dens) {
    // get the CIC indices and distances
    int ipos[3], iposp1[3];
    double dist[3];
    for (int axis = 0; axis < 3; axis++) {
        ipos[axis] = (int)floor(pos[axis] + 0.5) - 1;
        iposp1[axis] = ipos[axis] + 1;
        dist[axis] = pos[axis] - 0.5 - ipos[axis];
    }

    wrap_coord(ipos, box_dim);
    wrap_coord(iposp1, box_dim);

    unsigned long long int cic_indices[8] = {
        grid_index_general(ipos[0], ipos[1], ipos[2], box_dim),
        grid_index_general(iposp1[0], ipos[1], ipos[2], box_dim),
        grid_index_general(ipos[0], iposp1[1], ipos[2], box_dim),
        grid_index_general(iposp1[0], iposp1[1], ipos[2], box_dim),
        grid_index_general(ipos[0], ipos[1], iposp1[2], box_dim),
        grid_index_general(iposp1[0], ipos[1], iposp1[2], box_dim),
        grid_index_general(ipos[0], iposp1[1], iposp1[2], box_dim),
        grid_index_general(iposp1[0], iposp1[1], iposp1[2], box_dim)};

    double cic_weights[8] = {(1. - dist[0]) * (1. - dist[1]) * (1. - dist[2]),
                             dist[0] * (1. - dist[1]) * (1. - dist[2]),
                             (1. - dist[0]) * dist[1] * (1. - dist[2]),
                             dist[0] * dist[1] * (1. - dist[2]),
                             (1. - dist[0]) * (1. - dist[1]) * dist[2],
                             dist[0] * (1. - dist[1]) * dist[2],
                             (1. - dist[0]) * dist[1] * dist[2],
                             dist[0] * dist[1] * dist[2]};

    for (int i = 0; i < 8; i++) {
#pragma omp atomic update
        resampled_box[cic_indices[i]] += curr_dens * cic_weights[i];
    }
}

void compute_perturbed_velocities(unsigned short axis, double redshift,
                                  fftwf_complex *density_saved, fftwf_complex *velocity_fft_grid,
                                  float *velocity) {
    float k_x, k_y, k_z, k_sq;
    int n_x, n_y, n_z;
    int i, j, k;

    float kvec[3];
    double dDdt_over_D = ddickedt(redshift) / dicke(redshift);
    long long switch_mid[3];
    unsigned long long int n_k_pixels, n_r_pixels;
    // Function for deciding the dimensions of loops when we could
    // use either the low or high resolution grids.
    int box_dim[3];
    float *vel_pointers[3], *vel_pointers_2LPT[3];
    double dim_ratio =
        matter_options_global->PERTURB_ON_HIGH_RES
            ? simulation_options_global->DIM / (double)(simulation_options_global->HII_DIM)
            : 1.0;

    if (matter_options_global->PERTURB_ON_HIGH_RES) {
        box_dim[0] = simulation_options_global->DIM;
        box_dim[1] = simulation_options_global->DIM;
        box_dim[2] = D_PARA;
        switch_mid[0] = MID;
        switch_mid[1] = MID;
        switch_mid[2] = MID_PARA;
        n_k_pixels = KSPACE_NUM_PIXELS;
        n_r_pixels = TOT_NUM_PIXELS;
    } else {
        box_dim[0] = simulation_options_global->HII_DIM;
        box_dim[1] = simulation_options_global->HII_DIM;
        box_dim[2] = HII_D_PARA;
        switch_mid[0] = HII_MID;
        switch_mid[1] = HII_MID;
        switch_mid[2] = HII_MID_PARA;
        n_k_pixels = HII_KSPACE_NUM_PIXELS;
        n_r_pixels = HII_TOT_NUM_PIXELS;
    }

    memcpy(velocity_fft_grid, density_saved, sizeof(fftwf_complex) * n_k_pixels);

#pragma omp parallel private(n_x, n_y, n_z, k_x, k_y, k_z, k_sq, kvec) \
    num_threads(simulation_options_global -> N_THREADS)
    {
        unsigned long long grid_index;
#pragma omp for
        for (n_x = 0; n_x < box_dim[0]; n_x++) {
            if (n_x > switch_mid[0])
                k_x = (n_x - box_dim[0]) * DELTA_K;  // wrap around for FFT convention
            else
                k_x = n_x * DELTA_K;

            for (n_y = 0; n_y < box_dim[1]; n_y++) {
                if (n_y > switch_mid[1])
                    k_y = (n_y - box_dim[1]) * DELTA_K;
                else
                    k_y = n_y * DELTA_K;

                for (n_z = 0; n_z <= switch_mid[2]; n_z++) {
                    k_z = n_z * DELTA_K_PARA;

                    kvec[0] = k_x;
                    kvec[1] = k_y;
                    kvec[2] = k_z;
                    grid_index = grid_index_fftw_c(n_x, n_y, n_z, box_dim);

                    k_sq = k_x * k_x + k_y * k_y + k_z * k_z;

                    // now set the velocities
                    if ((n_x == 0) && (n_y == 0) && (n_z == 0)) {  // DC mode
                        velocity_fft_grid[grid_index] = 0.0 + 0.0 * I;
                    } else {
                        velocity_fft_grid[grid_index] *=
                            dDdt_over_D * kvec[axis] * I / k_sq / n_r_pixels;
                    }
                }
            }
        }
    }

    LOG_SUPER_DEBUG("density_perturb after modification by dDdt: ");
    debugSummarizeBoxComplex((float complex *)velocity_fft_grid, box_dim[0], box_dim[1], box_dim[2],
                             "  ");

    if (matter_options_global->PERTURB_ON_HIGH_RES &&
        simulation_options_global->DIM != simulation_options_global->HII_DIM) {
        filter_box(velocity_fft_grid, 0, 0,
                   L_FACTOR * simulation_options_global->BOX_LEN /
                       (simulation_options_global->HII_DIM + 0.0),
                   0.);
    }

    dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, box_dim[0], box_dim[2],
                 simulation_options_global->N_THREADS, velocity_fft_grid);

#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
    {
        unsigned long long int grid_index;
#pragma omp for
        for (i = 0; i < simulation_options_global->HII_DIM; i++) {
            for (j = 0; j < simulation_options_global->HII_DIM; j++) {
                for (k = 0; k < HII_D_PARA; k++) {
                    grid_index = matter_options_global->PERTURB_ON_HIGH_RES
                                     ? R_FFT_INDEX((unsigned long long)(i * dim_ratio + 0.5),
                                                   (unsigned long long)(j * dim_ratio + 0.5),
                                                   (unsigned long long)(k * dim_ratio + 0.5))
                                     : HII_R_FFT_INDEX(i, j, k);
                    velocity[HII_R_INDEX(i, j, k)] = *((float *)velocity_fft_grid + grid_index);
                }
            }
        }
    }
    LOG_SUPER_DEBUG("velocity: ");
    debugSummarizeBox(velocity, box_dim[0], box_dim[1], box_dim[2], "  ");
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

        fftwf_complex *HIRES_density_perturb = NULL;
        fftwf_complex *LOWRES_density_perturb, *density_perturb_saved;

        float growth_factor, displacement_factor_2LPT, init_growth_factor,
            init_displacement_factor_2LPT;
        int i, j, k, axis;

        double dim_ratio, dim_ratio_inv, mass_factor;
        // Function for deciding the dimensions of loops when we could
        // use either the low or high resolution grids.
        double boxlen = simulation_options_global->BOX_LEN;
        double boxlen_z = boxlen * simulation_options_global->NON_CUBIC_FACTOR;
        double box_size[3] = {boxlen, boxlen, boxlen_z};
        int box_dim[3];
        float *vel_pointers[3], *vel_pointers_2LPT[3];
        float *dens_pointer;
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            box_dim[0] = simulation_options_global->DIM;
            box_dim[1] = simulation_options_global->DIM;
            box_dim[2] = D_PARA;
            vel_pointers[0] = boxes->hires_vx;
            vel_pointers[1] = boxes->hires_vy;
            vel_pointers[2] = boxes->hires_vz;
            vel_pointers_2LPT[0] = boxes->hires_vx_2LPT;
            vel_pointers_2LPT[1] = boxes->hires_vy_2LPT;
            vel_pointers_2LPT[2] = boxes->hires_vz_2LPT;
            dens_pointer = boxes->hires_density;
        } else {
            box_dim[0] = simulation_options_global->HII_DIM;
            box_dim[1] = simulation_options_global->HII_DIM;
            box_dim[2] = HII_D_PARA;
            vel_pointers[0] = boxes->lowres_vx;
            vel_pointers[1] = boxes->lowres_vy;
            vel_pointers[2] = boxes->lowres_vz;
            vel_pointers_2LPT[0] = boxes->lowres_vx_2LPT;
            vel_pointers_2LPT[1] = boxes->lowres_vy_2LPT;
            vel_pointers_2LPT[2] = boxes->lowres_vz_2LPT;
            dens_pointer = boxes->lowres_density;
        }

        // ***************   BEGIN INITIALIZATION   ************************** //

        LOG_DEBUG("Computing Perturbed Field at z=%.3f", redshift);

        growth_factor = dicke(redshift);
        displacement_factor_2LPT = -(3.0 / 7.0) * growth_factor * growth_factor;  // 2LPT eq. D8

        init_growth_factor = dicke(simulation_options_global->INITIAL_REDSHIFT);
        init_displacement_factor_2LPT =
            -(3.0 / 7.0) * init_growth_factor * init_growth_factor;  // 2LPT eq. D8

        // low --> high res index factor, used when downsampling
        dim_ratio = simulation_options_global->DIM / (double)(simulation_options_global->HII_DIM);
        // high --> low res index factor
        dim_ratio_inv = matter_options_global->PERTURB_ON_HIGH_RES ? 1.0 : 1. / dim_ratio;
        mass_factor = pow(dim_ratio, 3);

        // allocate memory for the updated density, and initialize
        LOWRES_density_perturb =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            HIRES_density_perturb =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
            density_perturb_saved =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
        } else {
            density_perturb_saved =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
        }

        fftwf_complex *fft_density_grid = matter_options_global->PERTURB_ON_HIGH_RES
                                              ? HIRES_density_perturb
                                              : LOWRES_density_perturb;
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
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
            {
                unsigned long long int grid_index, fft_index;
#pragma omp for
                for (i = 0; i < box_dim[0]; i++) {
                    for (j = 0; j < box_dim[1]; j++) {
                        for (k = 0; k < box_dim[2]; k++) {
                            grid_index = grid_index_general(i, j, k, box_dim);
                            fft_index = grid_index_fftw_r(i, j, k, box_dim);
                            *((float *)fft_density_grid + fft_index) =
                                growth_factor * dens_pointer[grid_index];
                        }
                    }
                }
            }
        } else {
            // Apply Zel'dovich/2LPT correction
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
            {
                unsigned long long int fft_index;
#pragma omp for
                for (i = 0; i < box_dim[0]; i++) {
                    for (j = 0; j < box_dim[1]; j++) {
                        for (k = 0; k < box_dim[2]; k++) {
                            fft_index = grid_index_fftw_r(i, j, k, box_dim);
                            *((float *)fft_density_grid + fft_index) = 0.;
                        }
                    }
                }
            }

            double velocity_displacement_factor[3] = {
                (growth_factor - init_growth_factor) / box_size[0],
                (growth_factor - init_growth_factor) / box_size[1],
                (growth_factor - init_growth_factor) / box_size[2]};
            double velocity_displacement_factor_2LPT[3] = {
                (displacement_factor_2LPT - init_displacement_factor_2LPT) / box_size[0],
                (displacement_factor_2LPT - init_displacement_factor_2LPT) / box_size[1],
                (displacement_factor_2LPT - init_displacement_factor_2LPT) / box_size[2]};

            // ************  END INITIALIZATION **************************** //

            // Perturbing the density field required adding over multiple cells. Store intermediate
            // result as a double to avoid rounding errors
            if (matter_options_global->PERTURB_ON_HIGH_RES) {
                resampled_box = (double *)calloc(TOT_NUM_PIXELS, sizeof(double));
            } else {
                resampled_box = (double *)calloc(HII_TOT_NUM_PIXELS, sizeof(double));
            }

// go through the high-res box, mapping the mass onto the low-res (updated) box
#pragma omp parallel private(i, j, k, axis) num_threads(simulation_options_global -> N_THREADS)
            {
                double pos[3], curr_dens;
                int ipos[3];
                unsigned long long vel_index;
#pragma omp for
                for (i = 0; i < simulation_options_global->DIM; i++) {
                    for (j = 0; j < simulation_options_global->DIM; j++) {
                        for (k = 0; k < D_PARA; k++) {
                            // Transform position to units of box size
                            pos[0] = (i + 0.5) / (simulation_options_global->DIM + 0.0);
                            pos[1] = (j + 0.5) / (simulation_options_global->DIM + 0.0);
                            pos[2] = (k + 0.5) / (D_PARA + 0.0);
                            ipos[0] = (int)(i * dim_ratio_inv);
                            ipos[1] = (int)(j * dim_ratio_inv);
                            ipos[2] = (int)(k * dim_ratio_inv);
                            vel_index = grid_index_general(ipos[0], ipos[1], ipos[2], box_dim);
                            for (axis = 0; axis < 3; axis++) {
                                pos[axis] += vel_pointers[axis][vel_index] *
                                             velocity_displacement_factor[axis];
                                // add 2LPT second order corrections
                                if (matter_options_global->PERTURB_ALGORITHM == 2) {
                                    pos[axis] -= vel_pointers_2LPT[axis][vel_index] *
                                                 velocity_displacement_factor_2LPT[axis];
                                }

                                // transform to units of cell size
                                pos[axis] *= box_dim[axis];
                            }

                            // CIC interpolation
                            curr_dens =
                                1. + boxes->hires_density[R_INDEX(i, j, k)] * init_growth_factor;
                            do_cic_interpolation(resampled_box, pos, box_dim, curr_dens);
                        }
                    }
                }
            }

            LOG_SUPER_DEBUG("resampled_box: ");
            debugSummarizeBoxDouble(resampled_box, box_dim[0], box_dim[1], box_dim[2], "  ");

            // Resample back to a fftw float for remaining algorithm
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
            {
                unsigned long long int grid_index, fft_index;
#pragma omp for
                for (i = 0; i < box_dim[0]; i++) {
                    for (j = 0; j < box_dim[1]; j++) {
                        for (k = 0; k < box_dim[2]; k++) {
                            grid_index = grid_index_general(i, j, k, box_dim);
                            fft_index = grid_index_fftw_r(i, j, k, box_dim);
                            *((float *)fft_density_grid + fft_index) =
                                (float)resampled_box[grid_index];
                        }
                    }
                }
            }
            free(resampled_box);

            LOG_SUPER_DEBUG("density_perturb: ");

            debugSummarizeBox((float *)fft_density_grid, box_dim[0], box_dim[1],
                              2 * (box_dim[2] / 2 + 1), "  ");
        }

        // Now `fft_density` grid holds the perturbed density field,
        //  but we want to smooth if necessary and calculate the velocities
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            // Transform to Fourier space to sample (filter) the box
            dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                         D_PARA, simulation_options_global->N_THREADS, HIRES_density_perturb);

            // Need to save a copy of the unfiltered density field for the velocities
            memcpy(density_perturb_saved, HIRES_density_perturb,
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
#pragma omp parallel shared(HIRES_density_perturb, LOWRES_density_perturb, mass_factor) \
    private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
            {
#pragma omp for
                for (i = 0; i < simulation_options_global->HII_DIM; i++) {
                    for (j = 0; j < simulation_options_global->HII_DIM; j++) {
                        for (k = 0; k < HII_D_PARA; k++) {
                            *((float *)LOWRES_density_perturb + HII_R_FFT_INDEX(i, j, k)) =
                                *((float *)HIRES_density_perturb +
                                  R_FFT_INDEX((unsigned long long)(i * dim_ratio + 0.5),
                                              (unsigned long long)(j * dim_ratio + 0.5),
                                              (unsigned long long)(k * dim_ratio + 0.5))) /
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

        // save a copy of the k-space density field for velocity computation
        if (!matter_options_global->PERTURB_ON_HIGH_RES) {
            memcpy(density_perturb_saved, LOWRES_density_perturb,
                   sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
        }

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
        // We re-use fft_density_grid to hold the FFTW velocity field
        if (matter_options_global->KEEP_3D_VELOCITIES) {
            compute_perturbed_velocities(0, redshift, density_perturb_saved, fft_density_grid,
                                         perturbed_field->velocity_x);
            compute_perturbed_velocities(1, redshift, density_perturb_saved, fft_density_grid,
                                         perturbed_field->velocity_y);
        }
        compute_perturbed_velocities(2, redshift, density_perturb_saved, fft_density_grid,
                                     perturbed_field->velocity_z);

        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();

        // deallocate
        fftwf_free(LOWRES_density_perturb);
        fftwf_free(density_perturb_saved);
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            fftwf_free(HIRES_density_perturb);
        }
        fftwf_cleanup();
        LOG_DEBUG("Done.");

    }  // End of Try{}
    Catch(status) { return (status); }

    return (0);
}
