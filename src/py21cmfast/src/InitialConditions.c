#include "InitialConditions.h"

#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <limits.h>
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
#include "rng.h"

void adj_complex_conj(fftwf_complex *HIRES_box) {
    /*****  Adjust the complex conjugate relations for a real array  *****/

    int i, j, k;
    int box_dim[3] = {
        simulation_options_global->DIM, simulation_options_global->DIM,
        (int)(simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->DIM)};
    int mid_index[3];
    for (i = 0; i < 3; i++) {
        mid_index[i] = box_dim[i] / 2;
    }

    unsigned long long int corner_indices[8] = {
        grid_index_fftw_c(0, 0, mid_index[2], box_dim),
        grid_index_fftw_c(0, mid_index[1], 0, box_dim),
        grid_index_fftw_c(0, mid_index[1], mid_index[2], box_dim),
        grid_index_fftw_c(mid_index[0], 0, 0, box_dim),
        grid_index_fftw_c(mid_index[0], 0, mid_index[2], box_dim),
        grid_index_fftw_c(mid_index[0], mid_index[1], 0, box_dim),
        grid_index_fftw_c(mid_index[0], mid_index[1], mid_index[2], box_dim)};

    for (i = 0; i < 7; i++) {
        HIRES_box[corner_indices[i]] = crealf(HIRES_box[corner_indices[i]]);
    }
    // set zero mode
    HIRES_box[grid_index_fftw_c(0, 0, 0, box_dim)] = 0.;

    // do entire i except corners
#pragma omp parallel shared(HIRES_box) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
    {
        unsigned long long int index;
        unsigned long long int index_rx, index_ry, index_rxy;
#pragma omp for
        for (i = 1; i < mid_index[0]; i++) {
            // just j corners
            for (j = 0; j <= mid_index[1]; j += mid_index[1]) {
                for (k = 0; k <= mid_index[2]; k += mid_index[2]) {
                    index = grid_index_fftw_c(i, j, k, box_dim);
                    index_rx = grid_index_fftw_c(box_dim[0] - i, j, k, box_dim);
                    HIRES_box[index] = conjf(HIRES_box[index_rx]);
                }
            }

            // all of j
            for (j = 1; j < mid_index[1]; j++) {
                for (k = 0; k <= mid_index[2]; k += mid_index[2]) {
                    index = grid_index_fftw_c(i, j, k, box_dim);
                    index_rx = grid_index_fftw_c(box_dim[0] - i, j, k, box_dim);
                    index_ry = grid_index_fftw_c(i, box_dim[1] - j, k, box_dim);
                    index_rxy = grid_index_fftw_c(box_dim[0] - i, box_dim[1] - j, k, box_dim);
                    HIRES_box[index] = conjf(HIRES_box[index_rxy]);
                    HIRES_box[index_ry] = conjf(HIRES_box[index_rx]);
                }
            }
        }  // end loop over i
    }

    // now the i corners
#pragma omp parallel shared(HIRES_box) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
    {
        unsigned long long int index;
        unsigned long long int index_ry;
#pragma omp for
        for (i = 0; i <= mid_index[0]; i += mid_index[0]) {
            for (j = 1; j < mid_index[1]; j++) {
                for (k = 0; k <= mid_index[2]; k += mid_index[2]) {
                    index = grid_index_fftw_c(i, j, k, box_dim);
                    index_ry = grid_index_fftw_c(i, box_dim[1] - j, k, box_dim);
                    HIRES_box[index] = conjf(HIRES_box[index_ry]);
                }
            }
        }  // end loop over remaining j
    }
}

// Re-write of init.c for original 21cmFAST

int ComputeInitialConditions(unsigned long long random_seed, InitialConditions *boxes) {
    //     Generates the initial conditions: gaussian random density field
    //     (simulation_options_global->DIM^3) as well as the equal or lower resolution velocity
    //     fields, and smoothed density field (simulation_options_global->HII_DIM^3).
    //
    //     Author: Andrei Mesinger
    //     Date: 9/29/06

    int status;

    Try {  // This Try wraps the entire function so we don't indent.

        // Makes the parameter structs visible to a variety of functions/macros
        // Do each time to avoid Python garbage collection issues

#if LOG_LEVEL >= DEBUG_LEVEL
        writeSimulationOptions(simulation_options_global);
        writeMatterOptions(matter_options_global);
        writeCosmoParams(cosmo_params_global);
#endif

        int n_x, n_y, n_z, i, j, k, ii;
        float k_x, k_y, k_z, k_mag, p, a, b, k_sq;
        float p_vcb, vcb_i;

        gsl_rng *r[simulation_options_global->N_THREADS];
        seed_rng_threads(r, random_seed);

        omp_set_num_threads(simulation_options_global->N_THREADS);

        // setup dimensions
        int hi_dim[3] = {
            simulation_options_global->DIM, simulation_options_global->DIM,
            (int)(simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->DIM)};
        int lo_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                         (int)(simulation_options_global->NON_CUBIC_FACTOR *
                               simulation_options_global->HII_DIM)};
        int pt_dim[3];
        float *vel_pointers[3], *vel_pointers_2LPT[3];
        float *hires_v_2LPT[3] = {boxes->hires_vx_2LPT, boxes->hires_vy_2LPT, boxes->hires_vz_2LPT};
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            pt_dim[0] = hi_dim[0];
            pt_dim[1] = hi_dim[1];
            pt_dim[2] = hi_dim[2];
            vel_pointers[0] = boxes->hires_vx;
            vel_pointers[1] = boxes->hires_vy;
            vel_pointers[2] = boxes->hires_vz;
            vel_pointers_2LPT[0] = boxes->hires_vx_2LPT;
            vel_pointers_2LPT[1] = boxes->hires_vy_2LPT;
            vel_pointers_2LPT[2] = boxes->hires_vz_2LPT;
        } else {
            pt_dim[0] = lo_dim[0];
            pt_dim[1] = lo_dim[1];
            pt_dim[2] = lo_dim[2];
            vel_pointers[0] = boxes->lowres_vx;
            vel_pointers[1] = boxes->lowres_vy;
            vel_pointers[2] = boxes->lowres_vz;
            vel_pointers_2LPT[0] = boxes->lowres_vx_2LPT;
            vel_pointers_2LPT[1] = boxes->lowres_vy_2LPT;
            vel_pointers_2LPT[2] = boxes->lowres_vz_2LPT;
        }
        double dim_ratio_hi_pt = hi_dim[0] / (double)pt_dim[0];
        double dim_ratio_hi_lo = hi_dim[0] / (double)lo_dim[0];

        double box_len[3] = {
            simulation_options_global->BOX_LEN, simulation_options_global->BOX_LEN,
            simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->BOX_LEN};

        // ************  INITIALIZATION ********************** //
        // allocate array for the k-space and real-space boxes
        fftwf_complex *HIRES_box =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
        fftwf_complex *HIRES_box_saved =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

        // find factor of HII pixel size / deltax pixel size

        // ************  END INITIALIZATION ****************** //
        LOG_SUPER_DEBUG("Finished initialization.");
        // ************ CREATE K-SPACE GAUSSIAN RANDOM FIELD *********** //

        init_ps();

#pragma omp parallel shared(HIRES_box, r)                        \
    private(n_x, n_y, n_z, k_x, k_y, k_z, k_mag, p, a, b, p_vcb) \
    num_threads(simulation_options_global -> N_THREADS)
        {
            int thread_num = omp_get_thread_num();
#pragma omp for
            for (n_x = 0; n_x < hi_dim[0]; n_x++) {
                k_x = index_to_k(n_x, box_len[0], hi_dim[0]);
                for (n_y = 0; n_y < hi_dim[1]; n_y++) {
                    k_y = index_to_k(n_y, box_len[1], hi_dim[1]);
                    // since physical space field is real, only half contains independent modes
                    for (n_z = 0; n_z <= hi_dim[2] / 2; n_z++) {
                        k_z = index_to_k(n_z, box_len[2], hi_dim[2]);  // never goes above hi_dim/2

                        // now get the power spectrum; remember, only the magnitude of k counts (due
                        // to issotropy) this could be used to speed-up later maybe
                        k_mag = sqrt(k_x * k_x + k_y * k_y + k_z * k_z);
                        p = power_in_k(k_mag);

                        // ok, now we can draw the values of the real and imaginary part
                        // of our k entry from a Gaussian distribution
                        a = gsl_ran_ugaussian(r[thread_num]);
                        b = gsl_ran_ugaussian(r[thread_num]);

                        HIRES_box[grid_index_fftw_c(n_x, n_y, n_z, hi_dim)] =
                            sqrt(VOLUME * p / 2.0) * (a + b * I);
                    }
                }
            }
        }
        LOG_SUPER_DEBUG("Drawn random fields.");

        // *****  Adjust the complex conjugate relations for a real array  ***** //
        adj_complex_conj(HIRES_box);

        memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

        // FFT back to real space
        int stat =
            dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                         D_PARA, simulation_options_global->N_THREADS, HIRES_box);
        if (stat > 0) Throw(stat);
        LOG_SUPER_DEBUG("FFT'd hires boxes.");

#pragma omp parallel shared(boxes, HIRES_box) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
        {
            unsigned long long int index_r, index_f;
#pragma omp for
            for (i = 0; i < hi_dim[0]; i++) {
                for (j = 0; j < hi_dim[1]; j++) {
                    for (k = 0; k < hi_dim[2]; k++) {
                        index_r = grid_index_general(i, j, k, hi_dim);
                        index_f = grid_index_fftw_r(i, j, k, hi_dim);
                        boxes->hires_density[index_r] = *((float *)HIRES_box + index_f) / VOLUME;
                    }
                }
            }
        }

        LOG_SUPER_DEBUG("Saved HIRES_box to struct.");

        // *** If required, let's also create a lower-resolution version of the density field  ***
        // //
        memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

        // Only filter if we are perturbing on the low-resolution grid
        if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
            filter_box(HIRES_box, hi_dim, 0,
                       L_FACTOR * simulation_options_global->BOX_LEN /
                           (simulation_options_global->HII_DIM + 0.0),
                       0.);
        }

        // FFT back to real space
        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM, D_PARA,
                     simulation_options_global->N_THREADS, HIRES_box);

        // Renormalise the FFT'd box (sample the high-res box if we are perturbing on the
        // low-res grid)
#pragma omp parallel shared(boxes, HIRES_box, dim_ratio_hi_lo) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
        {
            unsigned long long int index_r, index_f;
            int resampled_index[3];
#pragma omp for
            for (i = 0; i < lo_dim[0]; i++) {
                for (j = 0; j < lo_dim[1]; j++) {
                    for (k = 0; k < lo_dim[2]; k++) {
                        index_r = grid_index_general(i, j, k, lo_dim);
                        resample_index((int[3]){i, j, k}, dim_ratio_hi_lo, resampled_index);
                        index_f = grid_index_fftw_r(resampled_index[0], resampled_index[1],
                                                    resampled_index[2], hi_dim);
                        boxes->lowres_density[index_r] = *((float *)HIRES_box + index_f) / VOLUME;
                    }
                }
            }
        }

        // ******* Relative Velocity part ******* //
        if (matter_options_global->USE_RELATIVE_VELOCITIES) {
            // JBM: We use the memory allocated to HIRES_box as it's free.
            for (ii = 0; ii < 3; ii++) {
                memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
#pragma omp parallel shared(HIRES_box, ii) private(n_x, n_y, n_z, k_x, k_y, k_z, k_mag, p, p_vcb) \
    num_threads(simulation_options_global -> N_THREADS)
                {
                    unsigned long long int index;
                    double kvec[3];
#pragma omp for
                    for (n_x = 0; n_x < hi_dim[0]; n_x++) {
                        k_x = index_to_k(n_x, box_len[0], hi_dim[0]);
                        for (n_y = 0; n_y < hi_dim[1]; n_y++) {
                            k_y = index_to_k(n_y, box_len[1], hi_dim[1]);
                            // since physical space field is real, only half contains independent
                            // modes
                            for (n_z = 0; n_z <= hi_dim[2] / 2; n_z++) {
                                k_z = index_to_k(n_z, box_len[2],
                                                 hi_dim[2]);  // never goes above hi_dim/2

                                k_mag = sqrt(k_x * k_x + k_y * k_y + k_z * k_z);
                                p = power_in_k(k_mag);
                                p_vcb = power_in_vcb(k_mag);

                                kvec[0] = k_x;
                                kvec[1] = k_y;
                                kvec[2] = k_z;

                                // now set the velocities
                                if ((n_x == 0) && (n_y == 0) && (n_z == 0)) {  // DC mode
                                    HIRES_box[0] = 0;
                                } else {
                                    index = grid_index_fftw_c(n_x, n_y, n_z, hi_dim);
                                    HIRES_box[index] *=
                                        I * kvec[ii] / k_mag * sqrt(p_vcb / p) * physconst.c_kms;
                                }
                            }
                        }
                    }
                }

                // we only care about the lowres vcb box, so we filter it directly.
                if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
                    filter_box(HIRES_box, hi_dim, 0,
                               L_FACTOR * simulation_options_global->BOX_LEN /
                                   (simulation_options_global->HII_DIM + 0.0),
                               0.);
                }

                // fft each velocity component back to real space
                dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                             D_PARA, simulation_options_global->N_THREADS, HIRES_box);

#pragma omp parallel shared(boxes, HIRES_box, dim_ratio_hi_lo, ii) private(i, j, k, vcb_i) \
    num_threads(simulation_options_global -> N_THREADS)
                {
                    unsigned long long int index_r, index_f;
                    int resampled_index[3];
#pragma omp for
                    for (i = 0; i < lo_dim[0]; i++) {
                        for (j = 0; j < lo_dim[1]; j++) {
                            for (k = 0; k < lo_dim[2]; k++) {
                                index_r = grid_index_general(i, j, k, lo_dim);
                                resample_index((int[3]){i, j, k}, dim_ratio_hi_lo, resampled_index);
                                index_f = grid_index_fftw_r(resampled_index[0], resampled_index[1],
                                                            resampled_index[2], hi_dim);
                                vcb_i = *((float *)HIRES_box + index_f);
                                boxes->lowres_vcb[index_r] += vcb_i * vcb_i;
                            }
                        }
                    }
                }
            }
            // now we take the sqrt of that and normalize the FFT
            for (i = 0; i < lo_dim[0]; i++) {
                for (j = 0; j < lo_dim[1]; j++) {
                    for (k = 0; k < lo_dim[2]; k++) {
                        boxes->lowres_vcb[grid_index_general(i, j, k, lo_dim)] =
                            sqrt(boxes->lowres_vcb[grid_index_general(i, j, k, lo_dim)]) / VOLUME;
                    }
                }
            }
        }
        LOG_SUPER_DEBUG("Completed Relative velocities.");
        // ******* End of Relative Velocity part ******* //

        // Now look at the velocities

        for (ii = 0; ii < 3; ii++) {
            memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
            // Now let's set the velocity field/dD/dt (in comoving Mpc)

#pragma omp parallel shared(HIRES_box, ii) private(n_x, n_y, n_z, k_x, k_y, k_z, k_sq) \
    num_threads(simulation_options_global -> N_THREADS)
            {
                unsigned long long int index;
                double kvec[3];
#pragma omp for
                for (n_x = 0; n_x < hi_dim[0]; n_x++) {
                    k_x = index_to_k(n_x, box_len[0], hi_dim[0]);
                    for (n_y = 0; n_y < hi_dim[1]; n_y++) {
                        k_y = index_to_k(n_y, box_len[1], hi_dim[1]);
                        // since physical space field is real, only half contains independent modes
                        for (n_z = 0; n_z <= hi_dim[2] / 2; n_z++) {
                            k_z = index_to_k(n_z, box_len[2],
                                             hi_dim[2]);  // never goes above hi_dim/2
                            k_sq = k_x * k_x + k_y * k_y + k_z * k_z;
                            kvec[0] = k_x;
                            kvec[1] = k_y;
                            kvec[2] = k_z;

                            // now set the velocities
                            if ((n_x == 0) && (n_y == 0) && (n_z == 0)) {  // DC mode
                                HIRES_box[0] = 0;
                            } else {
                                index = grid_index_fftw_c(n_x, n_y, n_z, hi_dim);
                                HIRES_box[index] *= kvec[ii] * I / k_sq / VOLUME;
                            }
                        }
                    }
                }
            }

            // Filter only if we require perturbing on the low-res grid
            if (!matter_options_global->PERTURB_ON_HIGH_RES) {
                if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
                    filter_box(HIRES_box, hi_dim, 0,
                               L_FACTOR * simulation_options_global->BOX_LEN /
                                   (simulation_options_global->HII_DIM + 0.0),
                               0.);
                }
            }

            dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                         D_PARA, simulation_options_global->N_THREADS, HIRES_box);

            // now sample to lower res
            // now sample the filtered box
#pragma omp parallel shared(boxes, HIRES_box, dim_ratio_hi_lo, ii) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
            {
                unsigned long long int index, index_f;
                int resampled_index[3];
#pragma omp for
                for (i = 0; i < pt_dim[0]; i++) {
                    for (j = 0; j < pt_dim[1]; j++) {
                        for (k = 0; k < pt_dim[2]; k++) {
                            index = grid_index_general(i, j, k, pt_dim);
                            resample_index((int[3]){i, j, k}, dim_ratio_hi_pt, resampled_index);
                            index_f = grid_index_fftw_r(resampled_index[0], resampled_index[1],
                                                        resampled_index[2], hi_dim);
                            vel_pointers[ii][index] = *((float *)HIRES_box + index_f);
                        }
                    }
                }
            }
        }

        LOG_SUPER_DEBUG("Done Inverse FT.");

        // * *************************************************** * //
        // *              BEGIN 2LPT PART                        * //
        // * *************************************************** * //

        // Generation of the second order Lagrangian perturbation theory (2LPT) corrections to the
        // ZA reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D

        // Parameter set in ANAL_PARAMS.H
        if (matter_options_global->PERTURB_ALGORITHM == 2) {
            // use six supplementary boxes to store the gradients of phi_1 (eq. D13b)
            // Allocating the boxes
            fftwf_complex *phi_1 =
                (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

            // First generate the ii,jj phi_1 boxes

            int phi_component;

            float component_ii, component_jj, component_ij;

            // Indexing for the various phy components
            int phi_directions[3][2] = {{0, 1}, {0, 2}, {1, 2}};

#pragma omp parallel shared(HIRES_box, phi_1) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
            {
                unsigned long long int index;
#pragma omp for
                for (i = 0; i < hi_dim[0]; i++) {
                    for (j = 0; j < hi_dim[1]; j++) {
                        for (k = 0; k < hi_dim[2]; k++) {
                            index = grid_index_fftw_r(i, j, k, hi_dim);
                            *((float *)HIRES_box + index) = 0.;
                        }
                    }
                }
            }

            // First iterate over the i = j components to phi
            // We'll also save these temporarily to the hires_vi_2LPT boxes which will get
            // overwritten later with the correct 2LPT velocities
            for (phi_component = 0; phi_component < 3; phi_component++) {
                i = j = phi_component;

                // generate the phi_1 boxes in Fourier transform
#pragma omp parallel shared(HIRES_box, phi_1, i, j) private(n_x, n_y, n_z, k_x, k_y, k_z, k_sq, k) \
    num_threads(simulation_options_global -> N_THREADS)
                {
                    unsigned long long int index;
#pragma omp for
                    for (n_x = 0; n_x < hi_dim[0]; n_x++) {
                        k_x = index_to_k(n_x, box_len[0], hi_dim[0]);
                        for (n_y = 0; n_y < hi_dim[1]; n_y++) {
                            k_y = index_to_k(n_y, box_len[1], hi_dim[1]);
                            // since physical space field is real, only half contains independent
                            // modes
                            for (n_z = 0; n_z <= hi_dim[2] / 2; n_z++) {
                                k_z = index_to_k(n_z, box_len[2],
                                                 hi_dim[2]);  // never goes above hi_dim/2

                                k_sq = k_x * k_x + k_y * k_y + k_z * k_z;

                                float k[] = {k_x, k_y, k_z};
                                // now set the velocities
                                if ((n_x == 0) && (n_y == 0) && (n_z == 0)) {  // DC mode
                                    phi_1[0] = 0;
                                } else {
                                    index = grid_index_fftw_c(n_x, n_y, n_z, hi_dim);
                                    phi_1[index] =
                                        -k[i] * k[j] * HIRES_box_saved[index] / k_sq / VOLUME;
                                    // note the last factor of 1/VOLUME accounts for the scaling in
                                    // real-space, following the FFT
                                }
                            }
                        }
                    }
                }

                dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                             D_PARA, simulation_options_global->N_THREADS, phi_1);

                // Temporarily store in the allocated hires_vi_2LPT boxes
#pragma omp parallel shared(boxes, phi_1, phi_component) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
                {
                    unsigned long long int index;
                    unsigned long long int index_f;
#pragma omp for
                    for (i = 0; i < hi_dim[0]; i++) {
                        for (j = 0; j < hi_dim[1]; j++) {
                            for (k = 0; k < hi_dim[2]; k++) {
                                index = grid_index_general(i, j, k, hi_dim);
                                index_f = grid_index_fftw_r(i, j, k, hi_dim);
                                hires_v_2LPT[phi_component][index] = *((float *)phi_1 + index_f);
                            }
                        }
                    }
                }
            }

            int phi_i, phi_j;
            for (phi_component = 0; phi_component < 3; phi_component++) {
                // Now calculate the cross components and start evaluating the 2LPT field
                phi_i = phi_directions[phi_component][0];
                phi_j = phi_directions[phi_component][1];

                // generate the phi_1 boxes in Fourier transform
#pragma omp parallel shared(HIRES_box, phi_1) private(n_x, n_y, n_z, k_x, k_y, k_z, k_sq, k) \
    num_threads(simulation_options_global -> N_THREADS)
                {
                    unsigned long long int index;
#pragma omp for
                    for (n_x = 0; n_x < hi_dim[0]; n_x++) {
                        k_x = index_to_k(n_x, box_len[0], hi_dim[0]);
                        for (n_y = 0; n_y < hi_dim[1]; n_y++) {
                            k_y = index_to_k(n_y, box_len[1], hi_dim[1]);
                            // since physical space field is real, only half contains independent
                            // modes
                            for (n_z = 0; n_z <= hi_dim[2] / 2; n_z++) {
                                k_z = index_to_k(n_z, box_len[2],
                                                 hi_dim[2]);  // never goes above hi_dim/2

                                k_sq = k_x * k_x + k_y * k_y + k_z * k_z;

                                index = grid_index_fftw_c(n_x, n_y, n_z, hi_dim);

                                float k[] = {k_x, k_y, k_z};
                                // now set the velocities
                                if ((n_x == 0) && (n_y == 0) && (n_z == 0)) {  // DC mode
                                    phi_1[0] = 0;
                                } else {
                                    phi_1[index] = -k[phi_i] * k[phi_j] * HIRES_box_saved[index] /
                                                   k_sq / VOLUME;
                                    // note the last factor of 1/VOLUME accounts for the scaling in
                                    // real-space, following the FFT
                                }
                            }
                        }
                    }
                }

                dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                             D_PARA, simulation_options_global->N_THREADS, phi_1);

                // Then we will have the laplacian of phi_2 (eq. D13b)
                // After that we have to return in Fourier space and generate the Fourier transform
                // of phi_2
#pragma omp parallel shared(HIRES_box, phi_1, phi_component)   \
    private(i, j, k, component_ii, component_jj, component_ij) \
    num_threads(simulation_options_global -> N_THREADS)
                {
                    unsigned long long index, index_f;
#pragma omp for
                    for (i = 0; i < hi_dim[0]; i++) {
                        for (j = 0; j < hi_dim[1]; j++) {
                            for (k = 0; k < hi_dim[2]; k++) {
                                index = grid_index_general(i, j, k, hi_dim);
                                index_f = grid_index_fftw_r(i, j, k, hi_dim);
                                // Note, I have temporarily stored the components into other arrays
                                // to minimise memory usage phi - {0, 1, 2} -> {hires_vx_2LPT,
                                // hires_vy_2LPT, hires_vz_2LPT} This may be opaque to the user, but
                                // this shouldn't need modification
                                component_ii = hires_v_2LPT[phi_i][index];
                                component_jj = hires_v_2LPT[phi_j][index];
                                component_ij = *((float *)phi_1 + index_f);

                                // Kept in this form to maintain similar (possible) rounding errors
                                *((float *)HIRES_box + index_f) += (component_ii * component_jj);
                                *((float *)HIRES_box + index_f) -= (component_ij * component_ij);
                            }
                        }
                    }
                }
            }

            // deallocate the supplementary boxes
            fftwf_free(phi_1);

#pragma omp parallel shared(HIRES_box) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
            {
                unsigned long long int index;
#pragma omp for
                for (i = 0; i < hi_dim[0]; i++) {
                    for (j = 0; j < hi_dim[1]; j++) {
                        for (k = 0; k < hi_dim[2]; k++) {
                            index = grid_index_fftw_r(i, j, k, hi_dim);
                            *((float *)HIRES_box + index) /= TOT_NUM_PIXELS;
                        }
                    }
                }
            }

            // Perform FFTs
            dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                         D_PARA, simulation_options_global->N_THREADS, HIRES_box);

            memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

            // Now we can store the content of box in a back-up array
            // Then we can generate the gradients of phi_2 (eq. D13b and D9)

            // ***** Store back-up k-box RHS eq. D13b ***** //

            // For each component, we generate the velocity field (same as the ZA part)

            // Now let's set the velocity field/dD/dt (in comoving Mpc)

            // read in the box

            for (ii = 0; ii < 3; ii++) {
                if (ii > 0) {
                    memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
                }

#pragma omp parallel shared(HIRES_box, ii) private(n_x, n_y, n_z, k_x, k_y, k_z, k_sq) \
    num_threads(simulation_options_global -> N_THREADS)
                {
                    unsigned long long int index;
                    double kvec[3];
#pragma omp for
                    for (n_x = 0; n_x < hi_dim[0]; n_x++) {
                        k_x = index_to_k(n_x, box_len[0], hi_dim[0]);
                        for (n_y = 0; n_y < hi_dim[1]; n_y++) {
                            k_y = index_to_k(n_y, box_len[1], hi_dim[1]);
                            // since physical space field is real, only half contains independent
                            // modes
                            for (n_z = 0; n_z <= hi_dim[2] / 2; n_z++) {
                                k_z = index_to_k(n_z, box_len[2],
                                                 hi_dim[2]);  // never goes above hi_dim/2

                                k_sq = k_x * k_x + k_y * k_y + k_z * k_z;

                                // now set the velocities
                                if ((n_x == 0) && (n_y == 0) && (n_z == 0)) {  // DC mode
                                    HIRES_box[0] = 0;

                                } else {
                                    kvec[0] = k_x;
                                    kvec[1] = k_y;
                                    kvec[2] = k_z;
                                    index = grid_index_fftw_c(n_x, n_y, n_z, hi_dim);
                                    HIRES_box[index] *= kvec[ii] * I / k_sq;
                                }
                            }
                        }
                    }
                }

                // Filter only if we require perturbing on the low-res grid
                if (!matter_options_global->PERTURB_ON_HIGH_RES) {
                    if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
                        filter_box(HIRES_box, hi_dim, 0,
                                   L_FACTOR * simulation_options_global->BOX_LEN /
                                       (simulation_options_global->HII_DIM + 0.0),
                                   0.);
                    }
                }

                dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                             D_PARA, simulation_options_global->N_THREADS, HIRES_box);

                // now sample the filtered box
#pragma omp parallel shared(boxes, HIRES_box, dim_ratio_hi_lo, ii) private(i, j, k) \
    num_threads(simulation_options_global -> N_THREADS)
                {
                    unsigned long long int index, index_f;
                    int resampled_index[3];
#pragma omp for
                    for (i = 0; i < pt_dim[0]; i++) {
                        for (j = 0; j < pt_dim[1]; j++) {
                            for (k = 0; k < pt_dim[2]; k++) {
                                index = grid_index_general(i, j, k, pt_dim);
                                resample_index((int[3]){i, j, k}, dim_ratio_hi_pt, resampled_index);
                                index_f = grid_index_fftw_r(resampled_index[0], resampled_index[1],
                                                            resampled_index[2], hi_dim);
                                vel_pointers_2LPT[ii][index] = *((float *)HIRES_box + index_f);
                            }
                        }
                    }
                }
            }
        }
        LOG_SUPER_DEBUG("Done 2LPT.");

        // * *********************************************** * //
        // *               END 2LPT PART                     * //
        // * *********************************************** * //
        fftwf_cleanup_threads();
        fftwf_cleanup();
        fftwf_forget_wisdom();

        // deallocate
        fftwf_free(HIRES_box);
        fftwf_free(HIRES_box_saved);

        free_ps();

        free_rng_threads(r);
        LOG_SUPER_DEBUG("Cleaned Up.");
    }  // End of Try{}

    Catch(status) { return (status); }
    return (0);
}
