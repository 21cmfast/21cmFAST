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

void sample_ic_modes(fftwf_complex *box, int grid_dim[3], double box_len[3], gsl_rng *r[]) {
    int n_x, n_y, n_z;
#pragma omp parallel private(n_x, n_y, n_z) num_threads(simulation_options_global -> N_THREADS)
    {
        double a, b;
        double k_x, k_y, k_z, k_mag, p;
        int thread_num = omp_get_thread_num();
#pragma omp for
        for (n_x = 0; n_x < grid_dim[0]; n_x++) {
            k_x = index_to_k(n_x, box_len[0], grid_dim[0]);
            for (n_y = 0; n_y < grid_dim[1]; n_y++) {
                k_y = index_to_k(n_y, box_len[1], grid_dim[1]);
                // since physical space field is real, only half contains independent modes
                for (n_z = 0; n_z <= grid_dim[2] / 2; n_z++) {
                    k_z = index_to_k(n_z, box_len[2], grid_dim[2]);  // never goes above hi_dim/2

                    // now get the power spectrum; remember, only the magnitude of k counts (due
                    // to issotropy) this could be used to speed-up later maybe
                    k_mag = sqrt(k_x * k_x + k_y * k_y + k_z * k_z);
                    p = power_in_k(k_mag);

                    // ok, now we can draw the values of the real and imaginary part
                    // of our k entry from a Gaussian distribution
                    a = gsl_ran_ugaussian(r[thread_num]);
                    b = gsl_ran_ugaussian(r[thread_num]);

                    box[grid_index_fftw_c(n_x, n_y, n_z, grid_dim)] =
                        sqrt(VOLUME * p / 2.0) * (a + b * I);
                }
            }
        }
    }
    LOG_SUPER_DEBUG("Drawn random fields.");

    // *****  Adjust the complex conjugate relations for a real array  ***** //
    adj_complex_conj(box);
}

void compute_relative_velocities(fftwf_complex *box, fftwf_complex *box_saved, float *out_vcb_pt) {
    // Computes the relative velocity field between DM and baryons
    // box: workspace box on high resolution grid
    // box_saved: saved copy of the initial density field in Fourier space on high resolution grid.
    //      This box should not be modified in this function
    // out_vcb_pt: output relative velocity field on low resolution grid

    int ii, n_x, n_y, n_z, i, j, k;

    int hi_dim[3] = {simulation_options_global->DIM, simulation_options_global->DIM, D_PARA};
    int lo_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                     HII_D_PARA};
    double dim_ratio_hi_lo = hi_dim[0] / (double)lo_dim[0];
    double box_len[3] = {
        simulation_options_global->BOX_LEN, simulation_options_global->BOX_LEN,
        simulation_options_global->BOX_LEN * simulation_options_global->NON_CUBIC_FACTOR};
    for (ii = 0; ii < 3; ii++) {
#pragma omp parallel private(n_x, n_y, n_z) num_threads(simulation_options_global -> N_THREADS)
        {
            double k_x, k_y, k_z, k_mag;
            double p, p_vcb;
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
                            box[0] = 0;
                        } else {
                            index = grid_index_fftw_c(n_x, n_y, n_z, hi_dim);
                            box[index] = box_saved[index] * I * kvec[ii] / k_mag * sqrt(p_vcb / p) *
                                         physconst.c_kms;
                        }
                    }
                }
            }
        }

        // we only care about the lowres vcb box, so we filter it directly.
        if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
            filter_box(box, hi_dim, 0,
                       physconst.l_factor * simulation_options_global->BOX_LEN /
                           (simulation_options_global->HII_DIM + 0.0),
                       0.);
        }

        // fft each velocity component back to real space
        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM, D_PARA,
                     simulation_options_global->N_THREADS, box);

#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
        {
            double vcb_i;
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
                        vcb_i = *((float *)box + index_f);
                        out_vcb_pt[index_r] += vcb_i * vcb_i;
                    }
                }
            }
        }
    }
    // now we take the sqrt of that and normalize the FFT
    for (i = 0; i < lo_dim[0]; i++) {
        for (j = 0; j < lo_dim[1]; j++) {
            for (k = 0; k < lo_dim[2]; k++) {
                out_vcb_pt[grid_index_general(i, j, k, lo_dim)] =
                    sqrt(out_vcb_pt[grid_index_general(i, j, k, lo_dim)]) / VOLUME;
            }
        }
    }
    LOG_SUPER_DEBUG("Completed Relative velocities.");
}

void compute_f_gradient(fftwf_complex *box_in, fftwf_complex *box_out, int dim[3], double len[3],
                        int axis) {
    int n_x, n_y, n_z;
#pragma omp parallel private(n_x, n_y, n_z) num_threads(simulation_options_global -> N_THREADS)
    {
        unsigned long long int index;
        double k_x, k_y, k_z, k_sq;
#pragma omp for
        for (n_x = 0; n_x < dim[0]; n_x++) {
            k_x = index_to_k(n_x, len[0], dim[0]);
            for (n_y = 0; n_y < dim[1]; n_y++) {
                k_y = index_to_k(n_y, len[1], dim[1]);
                // since physical space field is real, only half contains independent modes
                for (n_z = 0; n_z <= dim[2] / 2; n_z++) {
                    k_z = index_to_k(n_z, len[2], dim[2]);
                    k_sq = k_x * k_x + k_y * k_y + k_z * k_z;

                    double kvec[3] = {k_x, k_y, k_z};
                    // now set the velocities
                    index = grid_index_fftw_c(n_x, n_y, n_z, dim);
                    box_out[index] = box_in[index] * kvec[axis] * I / k_sq;
                }
            }
        }
    }
    // set the DC mode (will currently be nan from the zero division above)
    box_out[grid_index_fftw_c(0, 0, 0, dim)] = 0;
}

void compute_f_laplacian(fftwf_complex *box_in, fftwf_complex *box_out, int dim[3], double len[3],
                         int axes[2]) {
    int n_x, n_y, n_z;
#pragma omp parallel private(n_x, n_y, n_z) num_threads(simulation_options_global -> N_THREADS)
    {
        unsigned long long int index;
        double k_x, k_y, k_z, k_sq;
#pragma omp for
        for (n_x = 0; n_x < dim[0]; n_x++) {
            k_x = index_to_k(n_x, len[0], dim[0]);
            for (n_y = 0; n_y < dim[1]; n_y++) {
                k_y = index_to_k(n_y, len[1], dim[1]);
                // r2c means z-dim has only half the modes
                for (n_z = 0; n_z <= dim[2] / 2; n_z++) {
                    k_z = index_to_k(n_z, len[2], dim[2]);

                    k_sq = k_x * k_x + k_y * k_y + k_z * k_z;

                    double kvec[3] = {k_x, k_y, k_z};
                    index = grid_index_fftw_c(n_x, n_y, n_z, dim);
                    // The Volume factor anticipates an inverse FFT
                    box_out[index] = -kvec[axes[0]] * kvec[axes[1]] * box_in[index] / k_sq;
                }
            }
        }
    }
    // set the DC mode
    box_out[grid_index_fftw_c(0, 0, 0, dim)] = 0;
}

void compute_velocity_fields(fftwf_complex *box, fftwf_complex *box_saved, float *vel_pointers[3]) {
    // Computes the IC velocity fields
    // box: workspace box on high resolution grid
    // box_saved: saved copy of the initial density field in Fourier space on high resolution grid.
    //      This box should not be modified in this function
    // vel_pointers: output velocity field pointers [vx,vy,vz]
    int ii, n_x, n_y, n_z, i, j, k;

    int hi_dim[3] = {simulation_options_global->DIM, simulation_options_global->DIM, D_PARA};
    int pt_dim[3];

    if (matter_options_global->PERTURB_ON_HIGH_RES) {
        pt_dim[0] = simulation_options_global->DIM;
        pt_dim[1] = simulation_options_global->DIM;
        pt_dim[2] = D_PARA;
    } else {
        pt_dim[0] = simulation_options_global->HII_DIM;
        pt_dim[1] = simulation_options_global->HII_DIM;
        pt_dim[2] = HII_D_PARA;
    }

    double dim_ratio_hi_pt = hi_dim[0] / (double)pt_dim[0];
    double box_len[3] = {
        simulation_options_global->BOX_LEN, simulation_options_global->BOX_LEN,
        simulation_options_global->BOX_LEN * simulation_options_global->NON_CUBIC_FACTOR};

    for (ii = 0; ii < 3; ii++) {
        // Now let's set the velocity field/dD/dt (in comoving Mpc)
        compute_f_gradient(box_saved, box, hi_dim, box_len, ii);

        // Filter only if we require perturbing on the low-res grid
        if (!matter_options_global->PERTURB_ON_HIGH_RES) {
            if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
                filter_box(box, hi_dim, 0,
                           physconst.l_factor * simulation_options_global->BOX_LEN /
                               (simulation_options_global->HII_DIM + 0.0),
                           0.);
            }
        }

        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM, D_PARA,
                     simulation_options_global->N_THREADS, box);

        // now sample to lower res
        // now sample the filtered box
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
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
                        vel_pointers[ii][index] = *((float *)box + index_f) / VOLUME;
                    }
                }
            }
        }
    }

    LOG_SUPER_DEBUG("Done Velocity Fields.");
}

void compute_velocity_fields_2LPT(fftwf_complex *box, fftwf_complex *box_saved,
                                  float *diag_storage_pt[3], float *out_v_pointers[3]) {
    // Computes the relative velocity field between DM and baryons
    // box: workspace box on high resolution grid
    // box_saved: saved copy of the initial density field in Fourier space at high resolution.
    //     NOTE: This box will be modified in this function, and not restored afterwards.
    // diag_storage_pt: boxes to store diagonal components, we usually use the hires_vi_2LPT boxes
    //      on high-res grid, used as workspace before writing final 2LPT velocities
    // out_v_pointers: output velocity field pointers [vx,vy,vz]
    // If we are perturbing on high-res, this is the same as vel_pointers_hires
    int ii, i, j, k;

    // get dimensions
    int hi_dim[3] = {simulation_options_global->DIM, simulation_options_global->DIM, D_PARA};
    int pt_dim[3];

    if (matter_options_global->PERTURB_ON_HIGH_RES) {
        pt_dim[0] = simulation_options_global->DIM;
        pt_dim[1] = simulation_options_global->DIM;
        pt_dim[2] = D_PARA;
    } else {
        pt_dim[0] = simulation_options_global->HII_DIM;
        pt_dim[1] = simulation_options_global->HII_DIM;
        pt_dim[2] = HII_D_PARA;
    }

    double dim_ratio_hi_pt = hi_dim[0] / (double)pt_dim[0];
    double box_len[3] = {
        simulation_options_global->BOX_LEN, simulation_options_global->BOX_LEN,
        simulation_options_global->BOX_LEN * simulation_options_global->NON_CUBIC_FACTOR};

    // Allocating the boxes
    fftwf_complex *phi_1 = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

    int phi_component, phi_i, phi_j;

    // Indexing for the various phy components
    int phi_directions[3][2] = {{0, 1}, {0, 2}, {1, 2}};

    // First zero the workspace box for summation
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
    {
        unsigned long long int index;
#pragma omp for
        for (i = 0; i < hi_dim[0]; i++) {
            for (j = 0; j < hi_dim[1]; j++) {
                for (k = 0; k < hi_dim[2]; k++) {
                    index = grid_index_fftw_r(i, j, k, hi_dim);
                    *((float *)box + index) = 0.;
                }
            }
        }
    }

    // Iterate over the diagonal i == j components to phi
    for (phi_component = 0; phi_component < 3; phi_component++) {
        phi_i = phi_j = phi_component;

        // generate the phi_1 boxes in Fourier transform
        compute_f_laplacian(box_saved, phi_1, hi_dim, box_len, (int[2]){phi_i, phi_j});

        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM, D_PARA,
                     simulation_options_global->N_THREADS, phi_1);

        // Temporarily store the diagonal components.
        // This will usually be in the allocated hires_vi_2LPT boxes
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
        {
            unsigned long long int index;
            unsigned long long int index_f;
#pragma omp for
            for (i = 0; i < hi_dim[0]; i++) {
                for (j = 0; j < hi_dim[1]; j++) {
                    for (k = 0; k < hi_dim[2]; k++) {
                        index = grid_index_general(i, j, k, hi_dim);
                        index_f = grid_index_fftw_r(i, j, k, hi_dim);
                        diag_storage_pt[phi_component][index] = *((float *)phi_1 + index_f);
                    }
                }
            }
        }
    }

    // We now sum the non-diagonal components onto the grid on-the-fly, We only need to save the
    // diagonal components since Phi2_ij = sum_i sum_j!=i (phi_ii phi_jj - phi_ij^2)
    for (phi_component = 0; phi_component < 3; phi_component++) {
        phi_i = phi_directions[phi_component][0];
        phi_j = phi_directions[phi_component][1];

        compute_f_laplacian(box_saved, phi_1, hi_dim, box_len, (int[2]){phi_i, phi_j});

        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM, D_PARA,
                     simulation_options_global->N_THREADS, phi_1);

        // Now sum the stored components to get the laplacian of phi_2 (eq. D13b)
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
        {
            unsigned long long index, index_f;
            double component_ii, component_jj, component_ij;
#pragma omp for
            for (i = 0; i < hi_dim[0]; i++) {
                for (j = 0; j < hi_dim[1]; j++) {
                    for (k = 0; k < hi_dim[2]; k++) {
                        index = grid_index_general(i, j, k, hi_dim);
                        index_f = grid_index_fftw_r(i, j, k, hi_dim);
                        component_ii = diag_storage_pt[phi_i][index];
                        component_jj = diag_storage_pt[phi_j][index];
                        component_ij = *((float *)phi_1 + index_f);

                        // Kept in this form to maintain similar (possible) rounding errors
                        *((float *)box + index_f) += (component_ii * component_jj);
                        *((float *)box + index_f) -= (component_ij * component_ij);
                    }
                }
            }
        }
    }
    // deallocate the supplementary boxes
    fftwf_free(phi_1);

#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
    {
        unsigned long long int index;
#pragma omp for
        for (i = 0; i < hi_dim[0]; i++) {
            for (j = 0; j < hi_dim[1]; j++) {
                for (k = 0; k < hi_dim[2]; k++) {
                    index = grid_index_fftw_r(i, j, k, hi_dim);
                    *((float *)box + index) /= VOLUME * VOLUME * TOT_NUM_PIXELS;
                }
            }
        }
    }

    // Perform FFTs
    dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM, D_PARA,
                 simulation_options_global->N_THREADS, box);

    // Then we can generate the gradients of phi_2 (eq. D13b and D9)

    // ***** Store back-up k-box RHS eq. D13b ***** //
    memcpy(box_saved, box, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

    // For each component, we generate the velocity field (same as the ZA part)
    for (ii = 0; ii < 3; ii++) {
        compute_f_gradient(box_saved, box, hi_dim, box_len, ii);
        // Filter only if we require perturbing on the low-res grid
        if (!matter_options_global->PERTURB_ON_HIGH_RES) {
            if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
                filter_box(box, hi_dim, 0,
                           physconst.l_factor * simulation_options_global->BOX_LEN /
                               (simulation_options_global->HII_DIM + 0.0),
                           0.);
            }
        }

        dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM, D_PARA,
                     simulation_options_global->N_THREADS, box);

        // now sample the filtered box
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global -> N_THREADS)
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
                        out_v_pointers[ii][index] = *((float *)box + index_f);
                    }
                }
            }
        }
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

        int i, j, k;

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
        float *vel_pointers[3], *vel_pointers_2LPT[3];
        float *hires_v_2LPT[3] = {boxes->hires_vx_2LPT, boxes->hires_vy_2LPT, boxes->hires_vz_2LPT};
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            vel_pointers[0] = boxes->hires_vx;
            vel_pointers[1] = boxes->hires_vy;
            vel_pointers[2] = boxes->hires_vz;
            vel_pointers_2LPT[0] = boxes->hires_vx_2LPT;
            vel_pointers_2LPT[1] = boxes->hires_vy_2LPT;
            vel_pointers_2LPT[2] = boxes->hires_vz_2LPT;
        } else {
            vel_pointers[0] = boxes->lowres_vx;
            vel_pointers[1] = boxes->lowres_vy;
            vel_pointers[2] = boxes->lowres_vz;
            vel_pointers_2LPT[0] = boxes->lowres_vx_2LPT;
            vel_pointers_2LPT[1] = boxes->lowres_vy_2LPT;
            vel_pointers_2LPT[2] = boxes->lowres_vz_2LPT;
        }
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

        sample_ic_modes(HIRES_box, hi_dim, box_len, r);

        memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

        /* ASSIGN HIRES DENSITY */
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

        /* FILTER AND ASSIGN LOWRES DENSITY */
        memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

        // Only filter if we are perturbing on the low-resolution grid
        if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
            filter_box(HIRES_box, hi_dim, 0,
                       physconst.l_factor * simulation_options_global->BOX_LEN /
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

        if (matter_options_global->USE_RELATIVE_VELOCITIES) {
            compute_relative_velocities(HIRES_box, HIRES_box_saved, boxes->lowres_vcb);
        }

        // Now look at the velocities

        compute_velocity_fields(HIRES_box, HIRES_box_saved, vel_pointers);

        // * *************************************************** * //
        // *              BEGIN 2LPT PART                        * //
        // * *************************************************** * //

        // Generation of the second order Lagrangian perturbation theory (2LPT) corrections to the
        // ZA reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D

        // NOTE: we are using the hires_vi_2LPT boxes as workspace for storing diagonal components
        // of phi_1, to avoid allocating more memory (argument 3)
        if (matter_options_global->PERTURB_ALGORITHM == 2) {
            compute_velocity_fields_2LPT(HIRES_box, HIRES_box_saved, hires_v_2LPT,
                                         vel_pointers_2LPT);
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
