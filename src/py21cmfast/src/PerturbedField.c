// Re-write of perturb_field.c for being accessible within the MCMC
#include "PerturbedField.h"

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
#include "map_mass.h"

void make_density_grid(float redshift, fftwf_complex *fft_density_grid, InitialConditions *boxes) {
    int i, j, k;

    // Function for deciding the dimensions of loops when we could
    // use either the low or high resolution grids.
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

    double growth_factor = dicke(redshift);
    // high --> low res index factor
    double *resampled_box;

    // check if the linear evolution flag was set
    if (matter_options_global->PERTURB_ALGORITHM == 0) {
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global->N_THREADS)
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
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global->N_THREADS)
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

        // ************  END INITIALIZATION **************************** //

        // Perturbing the density field required adding over multiple cells. Store intermediate
        // result as a double to avoid rounding errors
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            resampled_box = (double *)calloc(TOT_NUM_PIXELS, sizeof(double));
        } else {
            resampled_box = (double *)calloc(HII_TOT_NUM_PIXELS, sizeof(double));
        }
        int hi_dim[3] = {simulation_options_global->DIM, simulation_options_global->DIM, D_PARA};
        move_grid_masses(redshift, boxes->hires_density, hi_dim, vel_pointers, vel_pointers_2LPT,
                         box_dim, resampled_box, box_dim);

        LOG_SUPER_DEBUG("resampled_box: ");
        debugSummarizeBoxDouble(resampled_box, box_dim[0], box_dim[1], box_dim[2], "  ");

        // Resample back to a fftw float for remaining algorithm
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global->N_THREADS)
        {
            unsigned long long int grid_index, fft_index;
#pragma omp for
            for (i = 0; i < box_dim[0]; i++) {
                for (j = 0; j < box_dim[1]; j++) {
                    for (k = 0; k < box_dim[2]; k++) {
                        grid_index = grid_index_general(i, j, k, box_dim);
                        fft_index = grid_index_fftw_r(i, j, k, box_dim);
                        *((float *)fft_density_grid + fft_index) = resampled_box[grid_index];
                    }
                }
            }
        }
        free(resampled_box);

        LOG_SUPER_DEBUG("density_perturb: ");
        debugSummarizeBox((float *)fft_density_grid, box_dim[0], box_dim[1],
                          2 * (box_dim[2] / 2 + 1), "  ");
    }
}

void assign_to_lowres_grid(fftwf_complex *hires_grid, fftwf_complex *lowres_grid,
                           fftwf_complex *saved_grid) {
    int i, j, k;
    int lo_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                     HII_D_PARA};
    int hi_dim[3] = {simulation_options_global->DIM, simulation_options_global->DIM, D_PARA};
    double dim_ratio = hi_dim[0] / (double)lo_dim[0];
    // We need to downsample the high-res grid to the low-res grid
    dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, hi_dim[0], hi_dim[2],
                 simulation_options_global->N_THREADS, hires_grid);

    // Need to save a copy of the unfiltered density field for the velocities

    // TODO: The grid saving is awkward, it happens in different functions depending on the
    //  resolution, and the low-res grid is saved *after* the smoothing
    memcpy(saved_grid, hires_grid, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

    // Now filter the box
    filter_box(hires_grid, hi_dim, 0,
               physconst.l_factor * simulation_options_global->BOX_LEN / (lo_dim[0] + 0.0), 0., 0.);

    // FFT back to real space
    dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, hi_dim[0], hi_dim[2],
                 simulation_options_global->N_THREADS, hires_grid);

#pragma omp parallel private(i, j, k) num_threads(simulation_options_global->N_THREADS)
    {
        int hires_pos[3];
#pragma omp for
        for (i = 0; i < lo_dim[0]; i++) {
            for (j = 0; j < lo_dim[1]; j++) {
                for (k = 0; k < lo_dim[2]; k++) {
                    resample_index((int[3]){i, j, k}, dim_ratio, hires_pos);
                    *((float *)lowres_grid + grid_index_fftw_r(i, j, k, lo_dim)) =
                        *((float *)hires_grid +
                          grid_index_fftw_r(hires_pos[0], hires_pos[1], hires_pos[2], hi_dim)) /
                        (float)TOT_NUM_PIXELS;
                }
            }
        }
    }
}

void normalise_delta_grid(fftwf_complex *deltap1_grid) {
    int i, j, k;
    // NOTE: We could put these in a constant struct, but maybe the stack variables are worth the
    // recomputation
    int lo_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                     HII_D_PARA};
    int hi_dim[3] = {simulation_options_global->DIM, simulation_options_global->DIM, D_PARA};
    // Renormalise the lowres box
    double mass_factor =
        matter_options_global->PERTURB_ON_HIGH_RES
            ? 1.0
            : (lo_dim[0] * lo_dim[1] * lo_dim[2]) / (double)(hi_dim[0] * hi_dim[1] * hi_dim[2]);
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global->N_THREADS)
    {
        unsigned long long int grid_index;
        float *cell_ptr;
#pragma omp for
        for (i = 0; i < lo_dim[0]; i++) {
            for (j = 0; j < lo_dim[1]; j++) {
                for (k = 0; k < lo_dim[2]; k++) {
                    grid_index = grid_index_fftw_r(i, j, k, lo_dim);
                    cell_ptr = (float *)deltap1_grid + grid_index;
                    *cell_ptr *= mass_factor;
                    *cell_ptr -= 1;  // 1+delta --> delta
                }
            }
        }
    }
    LOG_SUPER_DEBUG("delta after normalisation: ");
    debugSummarizeBox((float *)deltap1_grid, lo_dim[0], lo_dim[1], 2 * (lo_dim[2] / 2 + 1), "  ");
}

void smooth_and_clip_density(fftwf_complex *lowres_grid, fftwf_complex *density_perturb_saved) {
    // transform to k-space
    int i, j, k;
    int box_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};
    dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                 HII_D_PARA, simulation_options_global->N_THREADS, lowres_grid);

    // smooth the field
    if (matter_options_global->SMOOTH_EVOLVED_DENSITY_FIELD) {
        filter_box(lowres_grid, box_dim, 2,
                   simulation_options_global->DENSITY_SMOOTH_RADIUS *
                       simulation_options_global->BOX_LEN /
                       (float)simulation_options_global->HII_DIM,
                   0., 0.);
    }

    LOG_SUPER_DEBUG("delta_k after smoothing: ");
    debugSummarizeBox((float *)lowres_grid, simulation_options_global->HII_DIM,
                      simulation_options_global->HII_DIM, 2 * (HII_D_PARA / 2 + 1), "  ");

    // save a copy of the k-space density field for velocity computation
    // TODO: The grid saving is awkward, it happens in different functions depending on the
    // resolution, and the low-res grid is saved *after* the smoothing
    if (!matter_options_global->PERTURB_ON_HIGH_RES) {
        memcpy(density_perturb_saved, lowres_grid, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);
    }

    dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM,
                 HII_D_PARA, simulation_options_global->N_THREADS, lowres_grid);

    LOG_SUPER_DEBUG("delta back in real space: ");
    debugSummarizeBox((float *)lowres_grid, simulation_options_global->HII_DIM,
                      simulation_options_global->HII_DIM, 2 * (HII_D_PARA / 2 + 1), "  ");

    // normalize after FFT
    int bad_count = 0;
#pragma omp parallel shared(lowres_grid) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS) reduction(+ : bad_count)
    {
        unsigned long long int grid_index;
#pragma omp for
        for (i = 0; i < simulation_options_global->HII_DIM; i++) {
            for (j = 0; j < simulation_options_global->HII_DIM; j++) {
                for (k = 0; k < HII_D_PARA; k++) {
                    grid_index = grid_index_fftw_r(i, j, k, box_dim);
                    *((float *)lowres_grid + grid_index) /= (float)HII_TOT_NUM_PIXELS;

                    if (*((float *)lowres_grid + grid_index) <
                        -1.0 + FRACT_FLOAT_ERR) {  // shouldn't happen

                        if (bad_count < 5)
                            LOG_WARNING("delta is <-1 for index %d %d %d (value=%f)", i, j, k,
                                        *((float *)lowres_grid + grid_index));
                        if (bad_count == 5)
                            LOG_WARNING("Skipping further warnings for delta <= -1.");
                        *((float *)lowres_grid + grid_index) = -1.0 + FRACT_FLOAT_ERR;
                        bad_count++;
                    }
                }
            }
        }
    }
    if (bad_count >= 5) LOG_WARNING("Total number of bad indices: %d", bad_count);
    LOG_SUPER_DEBUG("delta normalized: ");
    debugSummarizeBox((float *)lowres_grid, simulation_options_global->HII_DIM,
                      simulation_options_global->HII_DIM, 2 * (HII_D_PARA / 2 + 1), "  ");
}

void compute_perturbed_velocities(unsigned short axis, double redshift,
                                  fftwf_complex *density_saved, fftwf_complex *velocity_fft_grid,
                                  float *velocity) {
    float k_x, k_y, k_z, k_sq;
    int n_x, n_y, n_z;
    int i, j, k;

    float kvec[3];
    double dDdt_over_D = ddickedt(redshift) / dicke(redshift);
    unsigned long long int n_k_pixels, n_r_pixels;
    // Function for deciding the dimensions of loops when we could
    // use either the low or high resolution grids.
    int box_dim[3];

    if (matter_options_global->PERTURB_ON_HIGH_RES) {
        box_dim[0] = simulation_options_global->DIM;
        box_dim[1] = simulation_options_global->DIM;
        box_dim[2] = D_PARA;
        n_k_pixels = KSPACE_NUM_PIXELS;
        n_r_pixels = TOT_NUM_PIXELS;
    } else {
        box_dim[0] = simulation_options_global->HII_DIM;
        box_dim[1] = simulation_options_global->HII_DIM;
        box_dim[2] = HII_D_PARA;
        n_k_pixels = HII_KSPACE_NUM_PIXELS;
        n_r_pixels = HII_TOT_NUM_PIXELS;
    }
    int lo_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                     HII_D_PARA};
    double dim_ratio = box_dim[0] / (double)lo_dim[0];
    double box_len[3] = {
        simulation_options_global->BOX_LEN, simulation_options_global->BOX_LEN,
        simulation_options_global->BOX_LEN * simulation_options_global->NON_CUBIC_FACTOR};

    memcpy(velocity_fft_grid, density_saved, sizeof(fftwf_complex) * n_k_pixels);

#pragma omp parallel private(n_x, n_y, n_z, k_x, k_y, k_z, k_sq, kvec) \
    num_threads(simulation_options_global->N_THREADS)
    {
        unsigned long long grid_index;
#pragma omp for
        for (n_x = 0; n_x < box_dim[0]; n_x++) {
            k_x = index_to_k(n_x, box_len[0], box_dim[0]);
            for (n_y = 0; n_y < box_dim[1]; n_y++) {
                k_y = index_to_k(n_y, box_len[1], box_dim[1]);
                // since physical space field is real, only half contains independent modes
                for (n_z = 0; n_z <= box_dim[2] / 2; n_z++) {
                    k_z = index_to_k(n_z, box_len[2], box_dim[2]);  // never goes above box_dim/2

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
    debugSummarizeBoxComplex((float complex *)velocity_fft_grid, box_dim[0], box_dim[1],
                             box_dim[2] / 2 + 1, "  ");

    if (matter_options_global->PERTURB_ON_HIGH_RES &&
        simulation_options_global->DIM != simulation_options_global->HII_DIM) {
        filter_box(velocity_fft_grid, box_dim, 0,
                   physconst.l_factor * simulation_options_global->BOX_LEN /
                       (simulation_options_global->HII_DIM + 0.0),
                   0., 0.);
    }

    dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, box_dim[0], box_dim[2],
                 simulation_options_global->N_THREADS, velocity_fft_grid);

#pragma omp parallel private(i, j, k) num_threads(simulation_options_global->N_THREADS)
    {
        unsigned long long int grid_index_f, grid_index_r;
        int grid_ipos[3];
#pragma omp for
        for (i = 0; i < lo_dim[0]; i++) {
            for (j = 0; j < lo_dim[1]; j++) {
                for (k = 0; k < lo_dim[2]; k++) {
                    grid_index_r = grid_index_general(i, j, k, lo_dim);
                    resample_index((int[3]){i, j, k}, dim_ratio, grid_ipos);
                    grid_index_f =
                        grid_index_fftw_r(grid_ipos[0], grid_ipos[1], grid_ipos[2], box_dim);
                    velocity[grid_index_r] = *((float *)velocity_fft_grid + grid_index_f);
                }
            }
        }
    }
    LOG_SUPER_DEBUG("velocity: ");
    debugSummarizeBox(velocity, simulation_options_global->HII_DIM,
                      simulation_options_global->HII_DIM, HII_D_PARA, "  ");
}

int ComputePerturbedField(float redshift, InitialConditions *boxes,
                          PerturbedField *perturbed_field) {
    /*
     ComputePerturbedField uses the first-order Langragian displacement field to move the
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

        make_density_grid(redshift, fft_density_grid, boxes);

        // Move data from high-res to low-res grid if needed, and convert to delta
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            assign_to_lowres_grid(HIRES_density_perturb, LOWRES_density_perturb,
                                  density_perturb_saved);
        }
        if (matter_options_global->PERTURB_ALGORITHM > 0) {
            normalise_delta_grid(LOWRES_density_perturb);
        }

        // Smooth if required and make sure we have no values <= -1
        smooth_and_clip_density(LOWRES_density_perturb, density_perturb_saved);

        // Assign to the struct
        int lo_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                         HII_D_PARA};
#pragma omp parallel num_threads(simulation_options_global->N_THREADS)
        {
            unsigned long long int index_r, index_f;
#pragma omp for
            for (int i = 0; i < lo_dim[0]; i++) {
                for (int j = 0; j < lo_dim[1]; j++) {
                    for (int k = 0; k < lo_dim[2]; k++) {
                        index_r = grid_index_general(i, j, k, lo_dim);
                        index_f = grid_index_fftw_r(i, j, k, lo_dim);
                        *((float *)perturbed_field->density + index_r) =
                            *((float *)LOWRES_density_perturb + index_f);
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
