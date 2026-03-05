/*
    InitialConditions_gpu.cu -- GPU-accelerated initial conditions computation

    This file implements GPU kernels for computing initial conditions.

    Phase 1 Implementation:
    - CPU generates random numbers (GSL) for bit-for-bit reproducibility
    - CPU FFTW for FFT operations
    - GPU handles: velocity field computation, complex conjugate adjustment, subsampling

    Phase 2a Implementation:
    - CPU generates random numbers (GSL) for bit-for-bit reproducibility
    - cuFFT for FFT operations (data stays on GPU)
    - GPU handles: velocity field computation, complex conjugate adjustment, subsampling
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>

// GPU headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "indexing.h"
#include "indexing.cuh"
#include "dft.h"
#include "filtering.h"
#include "cosmology.h"
#include "rng.h"

#include "InitialConditions_gpu.h"

// ============================================================================
// GPU Kernel: Velocity field computation in k-space
// ============================================================================
// Computes velocity component from density field: v_k = i * k_component / k^2 * delta_k
// This is the main parallelizable operation in InitialConditions

__global__ void compute_velocity_kernel(
    cuFloatComplex *box,      // Input/output: k-space field
    int dimension,            // DIM
    int midpoint,             // MIDDLE
    int midpoint_para,        // MIDDLE_PARA
    float delta_k,            // DELTA_K
    float delta_k_para,       // DELTA_K_PARA
    float volume,             // VOLUME
    int component             // 0=x, 1=y, 2=z
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)dimension * dimension * (midpoint_para + 1);

    if (idx >= num_pixels) return;

    // Compute 3D indices from flattened index
    // C_INDEX(n_x, n_y, n_z) = z + (midpoint_para + 1) * (y + dimension * x)
    int n_z = idx % (midpoint_para + 1);
    unsigned long long remaining = idx / (midpoint_para + 1);
    int n_y = remaining % dimension;
    int n_x = remaining / dimension;

    // Compute wave vector components
    float k_x, k_y, k_z;

    if (n_x > midpoint)
        k_x = (n_x - dimension) * delta_k;
    else
        k_x = n_x * delta_k;

    if (n_y > midpoint)
        k_y = (n_y - dimension) * delta_k;
    else
        k_y = n_y * delta_k;

    k_z = n_z * delta_k_para;

    float k_sq = k_x * k_x + k_y * k_y + k_z * k_z;

    // Handle DC mode
    if (n_x == 0 && n_y == 0 && n_z == 0) {
        box[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    // Get the k component for this velocity direction
    float k_comp;
    if (component == 0) k_comp = k_x;
    else if (component == 1) k_comp = k_y;
    else k_comp = k_z;

    // Multiply by i * k_component / k^2 / VOLUME
    // i * (a + bi) = -b + ai
    cuFloatComplex val = box[idx];
    float factor = k_comp / k_sq / volume;
    // Multiply by i: (a + bi) * i = -b + ai
    box[idx] = make_cuFloatComplex(-cuCimagf(val) * factor, cuCrealf(val) * factor);
}

// ============================================================================
// GPU Kernel: Complex conjugate adjustment for Hermitian symmetry
// ============================================================================
// Enforces the complex conjugate relations required for a real-valued FFT result
// This operates on the k=0 and k=N/2 planes

__global__ void adjust_complex_conj_kernel(
    cuFloatComplex *box,
    int dimension,
    int midpoint,
    int midpoint_para
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // This kernel handles the edge cases for Hermitian symmetry
    // We'll launch this with enough threads to cover MIDDLE iterations

    if (idx >= (unsigned long long)midpoint) return;

    int i = idx + 1;  // i goes from 1 to MIDDLE-1

    // j corners (j = 0 or MIDDLE)
    for (int j = 0; j <= midpoint; j += midpoint) {
        for (int k = 0; k <= midpoint_para; k += midpoint_para) {
            unsigned long long src_idx = k + (midpoint_para + 1) * (j + (unsigned long long)dimension * (dimension - i));
            unsigned long long dst_idx = k + (midpoint_para + 1) * (j + (unsigned long long)dimension * i);
            cuFloatComplex src_val = box[src_idx];
            box[dst_idx] = make_cuFloatComplex(cuCrealf(src_val), -cuCimagf(src_val));
        }
    }

    // All of j (j from 1 to MIDDLE-1)
    for (int j = 1; j < midpoint; j++) {
        for (int k = 0; k <= midpoint_para; k += midpoint_para) {
            // HIRES_box[C_INDEX(i, j, k)] = conjf(HIRES_box[C_INDEX(DIM - i, DIM - j, k)])
            unsigned long long src_idx = k + (midpoint_para + 1) * ((dimension - j) + (unsigned long long)dimension * (dimension - i));
            unsigned long long dst_idx = k + (midpoint_para + 1) * (j + (unsigned long long)dimension * i);
            cuFloatComplex src_val = box[src_idx];
            box[dst_idx] = make_cuFloatComplex(cuCrealf(src_val), -cuCimagf(src_val));

            // HIRES_box[C_INDEX(i, DIM - j, k)] = conjf(HIRES_box[C_INDEX(DIM - i, j, k)])
            src_idx = k + (midpoint_para + 1) * (j + (unsigned long long)dimension * (dimension - i));
            dst_idx = k + (midpoint_para + 1) * ((dimension - j) + (unsigned long long)dimension * i);
            src_val = box[src_idx];
            box[dst_idx] = make_cuFloatComplex(cuCrealf(src_val), -cuCimagf(src_val));
        }
    }
}

__global__ void adjust_complex_conj_corners_kernel(
    cuFloatComplex *box,
    int dimension,
    int midpoint,
    int midpoint_para
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Handle i corners (i = 0 or MIDDLE)
    // j from 1 to MIDDLE-1

    if (idx >= (unsigned long long)(midpoint - 1)) return;

    int j = idx + 1;  // j goes from 1 to MIDDLE-1

    for (int i = 0; i <= midpoint; i += midpoint) {
        for (int k = 0; k <= midpoint_para; k += midpoint_para) {
            // HIRES_box[C_INDEX(i, j, k)] = conjf(HIRES_box[C_INDEX(i, DIM - j, k)])
            unsigned long long src_idx = k + (midpoint_para + 1) * ((dimension - j) + (unsigned long long)dimension * i);
            unsigned long long dst_idx = k + (midpoint_para + 1) * (j + (unsigned long long)dimension * i);
            cuFloatComplex src_val = box[src_idx];
            box[dst_idx] = make_cuFloatComplex(cuCrealf(src_val), -cuCimagf(src_val));
        }
    }
}

// ============================================================================
// GPU Kernel: Subsample high-res box to low-res
// ============================================================================

__global__ void subsample_box_kernel(
    float *hires_box,         // Input: high-res real-space box (with FFT padding)
    float *lowres_box,        // Output: low-res box (no padding)
    int hii_dim,              // HII_DIM
    int hii_d_para,           // HII_D_PARA
    int dim,                  // DIM
    int mid_para,             // MID_PARA
    float f_pixel_factor,     // DIM / HII_DIM
    float volume              // VOLUME for normalization
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)hii_dim * hii_dim * hii_d_para;

    if (idx >= num_pixels) return;

    // Compute 3D indices for low-res box
    int k = idx % hii_d_para;
    unsigned long long remaining = idx / hii_d_para;
    int j = remaining % hii_dim;
    int i = remaining / hii_dim;

    // Map to high-res indices
    int hi = (int)(i * f_pixel_factor + 0.5f);
    int hj = (int)(j * f_pixel_factor + 0.5f);
    int hk = (int)(k * f_pixel_factor + 0.5f);

    // R_FFT_INDEX(x, y, z) = z + 2 * (mid_para + 1) * (y + dim * x)
    unsigned long long hires_idx = hk + 2llu * (mid_para + 1) * (hj + (unsigned long long)dim * hi);

    lowres_box[idx] = hires_box[hires_idx] / volume;
}

// ============================================================================
// GPU Kernel: Copy hires density to output (with normalization)
// ============================================================================

__global__ void copy_hires_density_kernel(
    float *hires_box,         // Input: FFT result (with padding)
    float *output,            // Output: hires_density array (no padding)
    int dim,                  // DIM
    int d_para,               // D_PARA
    int mid_para,             // MID_PARA
    float volume              // VOLUME for normalization
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)dim * dim * d_para;

    if (idx >= num_pixels) return;

    // Compute 3D indices
    int k = idx % d_para;
    unsigned long long remaining = idx / d_para;
    int j = remaining % dim;
    int i = remaining / dim;

    // R_FFT_INDEX(x, y, z) = z + 2 * (mid_para + 1) * (y + dim * x)
    unsigned long long fft_idx = k + 2llu * (mid_para + 1) * (j + (unsigned long long)dim * i);

    output[idx] = hires_box[fft_idx] / volume;
}

// ============================================================================
// GPU Kernel: Store velocity to output array
// ============================================================================

__global__ void store_velocity_kernel(
    float *hires_box,         // Input: FFT result (with padding)
    float *output,            // Output: velocity array
    int dimension,            // DIM or HII_DIM
    int d_para,               // D_PARA or HII_D_PARA
    int dim,                  // DIM (for FFT indexing)
    int mid_para,             // MID_PARA
    float f_pixel_factor,     // pixel factor (1.0 for hires, DIM/HII_DIM for lowres)
    bool is_hires             // true for hires, false for lowres
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)dimension * dimension * d_para;

    if (idx >= num_pixels) return;

    // Compute 3D indices for output
    int k = idx % d_para;
    unsigned long long remaining = idx / d_para;
    int j = remaining % dimension;
    int i = remaining / dimension;

    unsigned long long fft_idx;
    if (is_hires) {
        // Direct copy
        fft_idx = k + 2llu * (mid_para + 1) * (j + (unsigned long long)dim * i);
    } else {
        // Subsample
        int hi = (int)(i * f_pixel_factor + 0.5f);
        int hj = (int)(j * f_pixel_factor + 0.5f);
        int hk = (int)(k * f_pixel_factor + 0.5f);
        fft_idx = hk + 2llu * (mid_para + 1) * (hj + (unsigned long long)dim * hi);
    }

    output[idx] = hires_box[fft_idx];
}

// ============================================================================
// GPU Kernel: Compute phi_1 component for 2LPT
// ============================================================================
// Computes: phi_1[k] = -k[i] * k[j] * delta_k / k^2 / VOLUME
// This is used to compute the second-order displacement field

__global__ void compute_phi1_kernel(
    cuFloatComplex *delta_k,  // Input: saved k-space density field
    cuFloatComplex *phi_1,    // Output: phi_1 component
    int dimension,            // DIM
    int midpoint,             // MIDDLE
    int midpoint_para,        // MIDDLE_PARA
    float delta_k_val,        // DELTA_K
    float delta_k_para,       // DELTA_K_PARA
    float volume,             // VOLUME
    int comp_i,               // first component (0=x, 1=y, 2=z)
    int comp_j                // second component (0=x, 1=y, 2=z)
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)dimension * dimension * (midpoint_para + 1);

    if (idx >= num_pixels) return;

    // Compute 3D indices from flattened index
    int n_z = idx % (midpoint_para + 1);
    unsigned long long remaining = idx / (midpoint_para + 1);
    int n_y = remaining % dimension;
    int n_x = remaining / dimension;

    // DC mode
    if (n_x == 0 && n_y == 0 && n_z == 0) {
        phi_1[idx] = make_cuFloatComplex(0.0f, 0.0f);
        return;
    }

    // Compute wave vector components
    float k_x, k_y, k_z;

    if (n_x > midpoint)
        k_x = (n_x - dimension) * delta_k_val;
    else
        k_x = n_x * delta_k_val;

    if (n_y > midpoint)
        k_y = (n_y - dimension) * delta_k_val;
    else
        k_y = n_y * delta_k_val;

    k_z = n_z * delta_k_para;

    float k_sq = k_x * k_x + k_y * k_y + k_z * k_z;

    // Get k components for the requested directions
    float k_arr[3] = {k_x, k_y, k_z};
    float k_i = k_arr[comp_i];
    float k_j = k_arr[comp_j];

    // phi_1[k] = -k[i] * k[j] * delta_k / k^2 / VOLUME
    cuFloatComplex val = delta_k[idx];
    float factor = -k_i * k_j / k_sq / volume;
    phi_1[idx] = make_cuFloatComplex(cuCrealf(val) * factor, cuCimagf(val) * factor);
}

// ============================================================================
// GPU Kernel: Accumulate 2LPT source term
// ============================================================================
// Computes: source += phi_ii * phi_jj - phi_ij^2
// This accumulates the Laplacian of phi_2 (eq. D13b from Scoccimarro 1998)

__global__ void accumulate_2lpt_source_kernel(
    float *source,            // Input/Output: accumulated source term (real space, with FFT padding)
    float *phi_ii,            // Input: diagonal component phi_ii (real space, no padding)
    float *phi_jj,            // Input: diagonal component phi_jj (real space, no padding)
    float *phi_ij,            // Input: cross component phi_ij (real space, with FFT padding)
    int dimension,            // DIM
    int d_para,               // D_PARA
    int mid_para              // MID_PARA
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)dimension * dimension * d_para;

    if (idx >= num_pixels) return;

    // Compute 3D indices
    int k = idx % d_para;
    unsigned long long remaining = idx / d_para;
    int j = remaining % dimension;
    int i = remaining / dimension;

    // R_INDEX (no padding) for phi_ii, phi_jj
    unsigned long long r_idx = k + (unsigned long long)d_para * (j + (unsigned long long)dimension * i);

    // R_FFT_INDEX (with padding) for source and phi_ij
    unsigned long long fft_idx = k + 2llu * (mid_para + 1) * (j + (unsigned long long)dimension * i);

    float val_ii = phi_ii[r_idx];
    float val_jj = phi_jj[r_idx];
    float val_ij = phi_ij[fft_idx];

    // Accumulate: source += phi_ii * phi_jj - phi_ij^2
    source[fft_idx] += val_ii * val_jj - val_ij * val_ij;
}

// ============================================================================
// GPU Kernel: Normalize 2LPT source and zero out buffer
// ============================================================================

__global__ void normalize_2lpt_source_kernel(
    float *source,            // Input/Output: source term (real space, with FFT padding)
    int dimension,            // DIM
    int d_para,               // D_PARA
    int mid_para,             // MID_PARA
    unsigned long long tot_num_pixels  // TOT_NUM_PIXELS for normalization
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)dimension * dimension * d_para;

    if (idx >= num_pixels) return;

    // Compute 3D indices
    int k = idx % d_para;
    unsigned long long remaining = idx / d_para;
    int j = remaining % dimension;
    int i = remaining / dimension;

    // R_FFT_INDEX (with padding)
    unsigned long long fft_idx = k + 2llu * (mid_para + 1) * (j + (unsigned long long)dimension * i);

    source[fft_idx] /= (float)tot_num_pixels;
}

__global__ void zero_fft_buffer_kernel(
    float *buffer,            // Buffer to zero (with FFT padding)
    int dimension,            // DIM
    int d_para,               // D_PARA
    int mid_para              // MID_PARA
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)dimension * dimension * d_para;

    if (idx >= num_pixels) return;

    // Compute 3D indices
    int k = idx % d_para;
    unsigned long long remaining = idx / d_para;
    int j = remaining % dimension;
    int i = remaining / dimension;

    // R_FFT_INDEX (with padding)
    unsigned long long fft_idx = k + 2llu * (mid_para + 1) * (j + (unsigned long long)dimension * i);

    buffer[fft_idx] = 0.0f;
}

// ============================================================================
// GPU Kernel: Copy cuFFT output to buffer without padding (for phi components)
// ============================================================================

__global__ void copy_cufft_to_no_padding_kernel(
    float *cufft_output,      // Input: cuFFT output (tightly packed)
    float *dest,              // Output: destination buffer (no padding)
    int dimension,            // DIM
    int d_para                // D_PARA
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long num_pixels = (unsigned long long)dimension * dimension * d_para;

    if (idx >= num_pixels) return;

    // Direct 1:1 copy - both are tightly packed
    dest[idx] = cufft_output[idx];
}

// ============================================================================
// Main GPU function: ComputeInitialConditions_gpu
// ============================================================================

extern "C" int ComputeInitialConditions_gpu(unsigned long long random_seed, InitialConditions *boxes) {
    int status;
    cudaError_t err;

    Try {
        LOG_DEBUG("ComputeInitialConditions_gpu: Starting GPU-accelerated computation");

        // Initialize CUDA context explicitly before any cuFFT calls.
        // On older GPUs (e.g., P100/Pascal), cuFFT may fail on the first call if
        // the CUDA context is not already initialized. This is a common CUDA idiom.
        err = cudaFree(0);
        if (err != cudaSuccess && err != cudaErrorInvalidValue) {
            LOG_ERROR("CUDA context initialization failed: %s", cudaGetErrorString(err));
            Throw(CUDAError);
        }

#if LOG_LEVEL >= DEBUG_LEVEL
        writeSimulationOptions(simulation_options_global);
        writeMatterOptions(matter_options_global);
        writeCosmoParams(cosmo_params_global);
#endif

        int n_x, n_y, n_z, i, j, k, ii;
        float k_x, k_y, k_z, k_mag, p, a, b;
        float f_pixel_factor;
        int dimension;

        // Initialize RNG (CPU - for bit-for-bit reproducibility with CPU version)
        gsl_rng *r[simulation_options_global->N_THREADS];
        seed_rng_threads(r, random_seed);
        omp_set_num_threads(simulation_options_global->N_THREADS);

        dimension = matter_options_global->PERTURB_ON_HIGH_RES ? simulation_options_global->DIM
                                                               : simulation_options_global->HII_DIM;

        // Define hi_dim for filter_box calls (must match CPU version)
        int hi_dim[3] = {simulation_options_global->DIM, simulation_options_global->DIM, (int)D_PARA};

        // Allocate CPU arrays
        fftwf_complex *HIRES_box =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
        fftwf_complex *HIRES_box_saved =
            (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

        if (!HIRES_box || !HIRES_box_saved) {
            LOG_ERROR("Failed to allocate CPU memory for HIRES boxes");
            Throw(MemoryAllocError);
        }

        f_pixel_factor = simulation_options_global->DIM / (float)simulation_options_global->HII_DIM;

        init_ps();

        // Check if the input hires density box is non-zero (user provided initial_density)
        bool non_zero_input = false;
        unsigned long long box_ct;
#pragma omp parallel shared(boxes, non_zero_input) \
    num_threads(simulation_options_global->N_THREADS)
        {
#pragma omp for
            for (box_ct = 0; box_ct < TOT_NUM_PIXELS; box_ct++) {
                if (!non_zero_input && boxes->hires_density[box_ct]) {
#pragma omp atomic write
                    non_zero_input = true;
                }
            }
        }

        float *hires_float = (float *)HIRES_box;

        if (non_zero_input) {
            // User provided initial_density: copy to HIRES_box with FFTW padding,
            // then R2C FFT to get k-space representation for velocity computation.
            LOG_DEBUG("Using provided hires_density (non-zero input detected)");
#pragma omp parallel shared(boxes, HIRES_box) private(i, j, k) \
    num_threads(simulation_options_global->N_THREADS)
            {
                unsigned long long int index_r, index_f;
#pragma omp for
                for (i = 0; i < hi_dim[0]; i++) {
                    for (j = 0; j < hi_dim[1]; j++) {
                        for (k = 0; k < hi_dim[2]; k++) {
                            index_r = grid_index_general(i, j, k, hi_dim);
                            index_f = grid_index_fftw_r(i, j, k, hi_dim);
                            *((float *)HIRES_box + index_f) =
                                boxes->hires_density[index_r] * VOLUME / TOT_NUM_PIXELS;
                        }
                    }
                }
            }

            int stat =
                dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->DIM,
                             D_PARA, simulation_options_global->N_THREADS, HIRES_box);
            if (stat > 0) Throw(stat);

            memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
            LOG_DEBUG("Saved provided density to k-space (HIRES_box_saved)");
        } else {
            // ============ CPU: Generate k-space Gaussian random field ============
            // This stays on CPU for bit-for-bit reproducibility with the CPU version
            // Use float pointer to access real/imag parts (CUDA-compatible, avoids C complex.h issues)
            LOG_DEBUG("Generating k-space Gaussian random field on CPU");

#pragma omp parallel shared(hires_float, r) private(n_x, n_y, n_z, k_x, k_y, k_z, k_mag, p, a, b) \
    num_threads(simulation_options_global->N_THREADS)
            {
                int thread_num = omp_get_thread_num();
#pragma omp for
                for (n_x = 0; n_x < simulation_options_global->DIM; n_x++) {
                    if (n_x > MIDDLE)
                        k_x = (n_x - simulation_options_global->DIM) * DELTA_K;
                    else
                        k_x = n_x * DELTA_K;

                    for (n_y = 0; n_y < simulation_options_global->DIM; n_y++) {
                        if (n_y > MIDDLE)
                            k_y = (n_y - simulation_options_global->DIM) * DELTA_K;
                        else
                            k_y = n_y * DELTA_K;

                        for (n_z = 0; n_z <= MIDDLE_PARA; n_z++) {
                            k_z = n_z * DELTA_K_PARA;
                            k_mag = sqrtf(k_x * k_x + k_y * k_y + k_z * k_z);
                            p = power_in_k(k_mag);

                            a = gsl_ran_ugaussian(r[thread_num]);
                            b = gsl_ran_ugaussian(r[thread_num]);

                            float scale = sqrtf(VOLUME * p / 2.0f);
                            unsigned long long idx = C_INDEX(n_x, n_y, n_z);
                            hires_float[2 * idx] = scale * a;      // real part
                            hires_float[2 * idx + 1] = scale * b;  // imag part
                        }
                    }
                }
            }
            LOG_DEBUG("Generated random field");

            // ============ CPU: Adjust complex conjugates ============
            // Keep on CPU for now (small operation, complex indexing)
            // Using float pointer to access real/imag parts directly (CUDA-compatible)

            float *box_float = (float *)HIRES_box;

        // Helper macro for accessing real and imag parts
        #define BOX_REAL(idx) box_float[2*(idx)]
        #define BOX_IMAG(idx) box_float[2*(idx) + 1]

        // corners - set to real-only (zero imag) or zero
        BOX_REAL(C_INDEX(0, 0, 0)) = 0; BOX_IMAG(C_INDEX(0, 0, 0)) = 0;

        BOX_IMAG(C_INDEX(0, 0, MIDDLE_PARA)) = 0;
        BOX_IMAG(C_INDEX(0, MIDDLE, 0)) = 0;
        BOX_IMAG(C_INDEX(0, MIDDLE, MIDDLE_PARA)) = 0;
        BOX_IMAG(C_INDEX(MIDDLE, 0, 0)) = 0;
        BOX_IMAG(C_INDEX(MIDDLE, 0, MIDDLE_PARA)) = 0;
        BOX_IMAG(C_INDEX(MIDDLE, MIDDLE, 0)) = 0;
        BOX_IMAG(C_INDEX(MIDDLE, MIDDLE, MIDDLE_PARA)) = 0;

#pragma omp parallel shared(box_float) private(i, j, k) num_threads(simulation_options_global->N_THREADS)
        {
#pragma omp for
            for (i = 1; i < MIDDLE; i++) {
                for (j = 0; j <= MIDDLE; j += MIDDLE) {
                    for (k = 0; k <= MIDDLE_PARA; k += MIDDLE_PARA) {
                        unsigned long long src_idx = C_INDEX((simulation_options_global->DIM) - i, j, k);
                        unsigned long long dst_idx = C_INDEX(i, j, k);
                        BOX_REAL(dst_idx) = BOX_REAL(src_idx);
                        BOX_IMAG(dst_idx) = -BOX_IMAG(src_idx);  // conjugate
                    }
                }
                for (j = 1; j < MIDDLE; j++) {
                    for (k = 0; k <= MIDDLE_PARA; k += MIDDLE_PARA) {
                        unsigned long long src_idx = C_INDEX((simulation_options_global->DIM) - i,
                                                             (simulation_options_global->DIM) - j, k);
                        unsigned long long dst_idx = C_INDEX(i, j, k);
                        BOX_REAL(dst_idx) = BOX_REAL(src_idx);
                        BOX_IMAG(dst_idx) = -BOX_IMAG(src_idx);

                        src_idx = C_INDEX((simulation_options_global->DIM) - i, j, k);
                        dst_idx = C_INDEX(i, (simulation_options_global->DIM) - j, k);
                        BOX_REAL(dst_idx) = BOX_REAL(src_idx);
                        BOX_IMAG(dst_idx) = -BOX_IMAG(src_idx);
                    }
                }
            }
        }

#pragma omp parallel shared(box_float) private(i, j, k) num_threads(simulation_options_global->N_THREADS)
        {
#pragma omp for
            for (i = 0; i <= MIDDLE; i += MIDDLE) {
                for (j = 1; j < MIDDLE; j++) {
                    for (k = 0; k <= MIDDLE_PARA; k += MIDDLE_PARA) {
                        unsigned long long src_idx = C_INDEX(i, (simulation_options_global->DIM) - j, k);
                        unsigned long long dst_idx = C_INDEX(i, j, k);
                        BOX_REAL(dst_idx) = BOX_REAL(src_idx);
                        BOX_IMAG(dst_idx) = -BOX_IMAG(src_idx);
                    }
                }
            }
        }

        #undef BOX_REAL
        #undef BOX_IMAG

        LOG_DEBUG("Adjusted complex conjugates");

            // Save the k-space field for later use
            memcpy(HIRES_box_saved, HIRES_box, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);
            LOG_SUPER_DEBUG("Saved k-space field");
        } // end else (normal random IC generation)

        // ============ cuFFT: Create plan for C2R transforms ============
        // Used by both paths (normal and initial_density) for lowres density and velocities
        cufftHandle fft_plan;
        cufftResult cufft_status;

        int fft_nx = simulation_options_global->DIM;
        int fft_ny = simulation_options_global->DIM;
        int fft_nz = (int)D_PARA;

        LOG_DEBUG("Creating cuFFT C2R plan for dims (%d, %d, %d)", fft_nx, fft_ny, fft_nz);

        cufft_status = cufftPlan3d(&fft_plan, fft_nx, fft_ny, fft_nz, CUFFT_C2R);
        if (cufft_status != CUFFT_SUCCESS) {
            LOG_ERROR("cuFFT plan creation failed: %d", cufft_status);
            Throw(CUDAError);
        }

        cufft_status = cufftSetStream(fft_plan, 0);
        if (cufft_status != CUFFT_SUCCESS) {
            LOG_WARNING("cufftSetStream failed: %d", cufft_status);
        }

        LOG_DEBUG("Created cuFFT C2R plan successfully");

        // Allocate GPU memory for FFT
        size_t kspace_bytes = KSPACE_NUM_PIXELS * sizeof(cufftComplex);
        size_t real_bytes_needed = TOT_NUM_PIXELS * sizeof(float);
        size_t real_bytes = (real_bytes_needed > kspace_bytes) ? real_bytes_needed : kspace_bytes;
        cufftComplex *d_kspace;
        float *d_realspace;

        err = cudaMalloc(&d_kspace, kspace_bytes);
        CATCH_CUDA_ERROR(err);
        err = cudaMalloc(&d_realspace, real_bytes);
        if (err != cudaSuccess) {
            cudaFree(d_kspace);
            CATCH_CUDA_ERROR(err);
        }
        LOG_DEBUG("Allocated GPU memory: d_kspace=%zu bytes, d_realspace=%zu bytes",
                  kspace_bytes, real_bytes);

        // ============ Hires density: C2R FFT and output copy ============
        // Skip when initial_density was provided (hires_density already set by caller)
        if (!non_zero_input) {
            err = cudaMemcpy(d_kspace, HIRES_box, kspace_bytes, cudaMemcpyHostToDevice);
            CATCH_CUDA_ERROR(err);

            cufft_status = cufftExecC2R(fft_plan, d_kspace, d_realspace);
            if (cufft_status != CUFFT_SUCCESS) {
                LOG_ERROR("cuFFT C2R execution failed: %d (kspace=%zu bytes, real=%zu bytes)",
                          cufft_status, kspace_bytes, real_bytes);
                cudaFree(d_kspace);
                cudaFree(d_realspace);
                cufftDestroy(fft_plan);
                Throw(CUDAError);
            }
            err = cudaDeviceSynchronize();
            CATCH_CUDA_ERROR(err);
            LOG_DEBUG("cuFFT execution complete");

            // Copy result back with FFT padding
            float *temp_real = (float *)malloc(real_bytes_needed);
            if (!temp_real) {
                LOG_ERROR("Failed to allocate temp buffer");
                Throw(MemoryAllocError);
            }
            err = cudaMemcpy(temp_real, d_realspace, real_bytes_needed, cudaMemcpyDeviceToHost);
            CATCH_CUDA_ERROR(err);

            hires_float = (float *)HIRES_box;
            #pragma omp parallel for collapse(2) num_threads(simulation_options_global->N_THREADS)
            for (int ix = 0; ix < simulation_options_global->DIM; ix++) {
                for (int iy = 0; iy < simulation_options_global->DIM; iy++) {
                    for (int iz = 0; iz < D_PARA; iz++) {
                        unsigned long long src_idx = iz + D_PARA * (iy + (unsigned long long)simulation_options_global->DIM * ix);
                        unsigned long long dst_idx = R_FFT_INDEX(ix, iy, iz);
                        hires_float[dst_idx] = temp_real[src_idx];
                    }
                }
            }
            free(temp_real);
            LOG_DEBUG("cuFFT to real space complete");

            // Copy hires density to output via GPU kernel
            {
                size_t fft_size = TOT_FFT_NUM_PIXELS * sizeof(float);
                size_t out_size = TOT_NUM_PIXELS * sizeof(float);

                float *d_hires_box, *d_output;
                err = cudaMalloc(&d_hires_box, fft_size);
                if (err != cudaSuccess) {
                    LOG_ERROR("CUDA malloc failed for d_hires_box: %s", cudaGetErrorString(err));
                    Throw(CUDAError);
                }

                err = cudaMalloc(&d_output, out_size);
                if (err != cudaSuccess) {
                    cudaFree(d_hires_box);
                    LOG_ERROR("CUDA malloc failed for d_output: %s", cudaGetErrorString(err));
                    Throw(CUDAError);
                }

                err = cudaMemcpy(d_hires_box, (float *)HIRES_box, fft_size, cudaMemcpyHostToDevice);
                CATCH_CUDA_ERROR(err);

                int threadsPerBlock = 256;
                int numBlocks = (TOT_NUM_PIXELS + threadsPerBlock - 1) / threadsPerBlock;

                copy_hires_density_kernel<<<numBlocks, threadsPerBlock>>>(
                    d_hires_box, d_output,
                    simulation_options_global->DIM,
                    D_PARA, MID_PARA, VOLUME
                );

                err = cudaDeviceSynchronize();
                CATCH_CUDA_ERROR(err);
                err = cudaGetLastError();
                CATCH_CUDA_ERROR(err);

                err = cudaMemcpy(boxes->hires_density, d_output, out_size, cudaMemcpyDeviceToHost);
                CATCH_CUDA_ERROR(err);

                cudaFree(d_hires_box);
                cudaFree(d_output);
            }
        } // end if (!non_zero_input)
        LOG_DEBUG("Saved hires_density");

        // ============ Create low-res density field ============
        memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

        if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
            filter_box(HIRES_box, hi_dim, 0,
                       L_FACTOR * simulation_options_global->BOX_LEN /
                           (simulation_options_global->HII_DIM + 0.0),
                       0., 0.);
        }

        // cuFFT: FFT to real space for lowres density
        err = cudaMemcpy(d_kspace, HIRES_box, kspace_bytes, cudaMemcpyHostToDevice);
        CATCH_CUDA_ERROR(err);

        cufft_status = cufftExecC2R(fft_plan, d_kspace, d_realspace);
        if (cufft_status != CUFFT_SUCCESS) {
            LOG_ERROR("cuFFT C2R execution failed for lowres: %d", cufft_status);
            Throw(CUDAError);
        }
        err = cudaDeviceSynchronize();
        CATCH_CUDA_ERROR(err);

        // Copy cuFFT output (tightly packed) back with FFT padding
        {
            float *temp_real2 = (float *)malloc(real_bytes);
            if (!temp_real2) {
                LOG_ERROR("Failed to allocate temp buffer for lowres");
                Throw(MemoryAllocError);
            }
            err = cudaMemcpy(temp_real2, d_realspace, real_bytes, cudaMemcpyDeviceToHost);
            CATCH_CUDA_ERROR(err);

            float *hires_float2 = (float *)HIRES_box;
            #pragma omp parallel for collapse(2) num_threads(simulation_options_global->N_THREADS)
            for (int ix = 0; ix < simulation_options_global->DIM; ix++) {
                for (int iy = 0; iy < simulation_options_global->DIM; iy++) {
                    for (int iz = 0; iz < D_PARA; iz++) {
                        unsigned long long src_idx = iz + D_PARA * (iy + (unsigned long long)simulation_options_global->DIM * ix);
                        unsigned long long dst_idx = R_FFT_INDEX(ix, iy, iz);
                        hires_float2[dst_idx] = temp_real2[src_idx];
                    }
                }
            }
            free(temp_real2);
        }

        // ============ GPU: Subsample to low-res ============
        {
            size_t fft_size = TOT_FFT_NUM_PIXELS * sizeof(float);
            size_t lowres_size = HII_TOT_NUM_PIXELS * sizeof(float);

            float *d_hires_box, *d_lowres_box;
            err = cudaMalloc(&d_hires_box, fft_size);
            CATCH_CUDA_ERROR(err);
            err = cudaMalloc(&d_lowres_box, lowres_size);
            if (err != cudaSuccess) {
                cudaFree(d_hires_box);
                CATCH_CUDA_ERROR(err);
            }

            err = cudaMemcpy(d_hires_box, (float *)HIRES_box, fft_size, cudaMemcpyHostToDevice);
            CATCH_CUDA_ERROR(err);

            int threadsPerBlock = 256;
            int numBlocks = (HII_TOT_NUM_PIXELS + threadsPerBlock - 1) / threadsPerBlock;

            subsample_box_kernel<<<numBlocks, threadsPerBlock>>>(
                d_hires_box, d_lowres_box,
                simulation_options_global->HII_DIM,
                HII_D_PARA,
                simulation_options_global->DIM,
                MID_PARA,
                f_pixel_factor,
                VOLUME
            );

            err = cudaDeviceSynchronize();
            CATCH_CUDA_ERROR(err);
            err = cudaGetLastError();
            CATCH_CUDA_ERROR(err);

            err = cudaMemcpy(boxes->lowres_density, d_lowres_box, lowres_size, cudaMemcpyDeviceToHost);
            CATCH_CUDA_ERROR(err);

            cudaFree(d_hires_box);
            cudaFree(d_lowres_box);
        }
        LOG_DEBUG("Created lowres_density");

        // ============ Velocity fields ============
        // Allocate GPU memory for velocity computation
        size_t kspace_size = KSPACE_NUM_PIXELS * sizeof(fftwf_complex);
        size_t fft_size = TOT_FFT_NUM_PIXELS * sizeof(float);

        cuFloatComplex *d_kspace_box;
        float *d_realspace_box, *d_output_box;

        err = cudaMalloc(&d_kspace_box, kspace_size);
        CATCH_CUDA_ERROR(err);
        err = cudaMalloc(&d_realspace_box, fft_size);
        if (err != cudaSuccess) {
            cudaFree(d_kspace_box);
            CATCH_CUDA_ERROR(err);
        }

        size_t vel_output_size;
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
            vel_output_size = TOT_NUM_PIXELS * sizeof(float);
        } else {
            vel_output_size = HII_TOT_NUM_PIXELS * sizeof(float);
        }

        err = cudaMalloc(&d_output_box, vel_output_size);
        if (err != cudaSuccess) {
            cudaFree(d_kspace_box);
            cudaFree(d_realspace_box);
            CATCH_CUDA_ERROR(err);
        }

        for (ii = 0; ii < 3; ii++) {
            memcpy(HIRES_box, HIRES_box_saved, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

            // ============ GPU: Compute velocity in k-space ============
            err = cudaMemcpy(d_kspace_box, HIRES_box, kspace_size, cudaMemcpyHostToDevice);
            CATCH_CUDA_ERROR(err);

            int threadsPerBlock = 256;
            int numBlocks = (KSPACE_NUM_PIXELS + threadsPerBlock - 1) / threadsPerBlock;

            compute_velocity_kernel<<<numBlocks, threadsPerBlock>>>(
                d_kspace_box,
                simulation_options_global->DIM,
                MIDDLE, MIDDLE_PARA,
                DELTA_K, DELTA_K_PARA,
                VOLUME,
                ii
            );

            err = cudaDeviceSynchronize();
            CATCH_CUDA_ERROR(err);
            err = cudaGetLastError();
            CATCH_CUDA_ERROR(err);

            err = cudaMemcpy(HIRES_box, d_kspace_box, kspace_size, cudaMemcpyDeviceToHost);
            CATCH_CUDA_ERROR(err);

            // Filter if needed
            if (!matter_options_global->PERTURB_ON_HIGH_RES) {
                if (simulation_options_global->DIM != simulation_options_global->HII_DIM) {
                    filter_box(HIRES_box, hi_dim, 0,
                               L_FACTOR * simulation_options_global->BOX_LEN /
                                   (simulation_options_global->HII_DIM + 0.0),
                               0., 0.);
                }
            }

            // cuFFT: FFT to real space for velocity
            err = cudaMemcpy(d_kspace, HIRES_box, kspace_bytes, cudaMemcpyHostToDevice);
            CATCH_CUDA_ERROR(err);

            cufft_status = cufftExecC2R(fft_plan, d_kspace, d_realspace);
            if (cufft_status != CUFFT_SUCCESS) {
                LOG_ERROR("cuFFT C2R execution failed for velocity %d: %d", ii, cufft_status);
                Throw(CUDAError);
            }
            err = cudaDeviceSynchronize();
            CATCH_CUDA_ERROR(err);

            // Copy cuFFT output (tightly packed) back with FFT padding
            {
                float *temp_real3 = (float *)malloc(real_bytes);
                if (!temp_real3) {
                    LOG_ERROR("Failed to allocate temp buffer for velocity");
                    Throw(MemoryAllocError);
                }
                err = cudaMemcpy(temp_real3, d_realspace, real_bytes, cudaMemcpyDeviceToHost);
                CATCH_CUDA_ERROR(err);

                float *hires_float3 = (float *)HIRES_box;
                #pragma omp parallel for collapse(2) num_threads(simulation_options_global->N_THREADS)
                for (int ix = 0; ix < simulation_options_global->DIM; ix++) {
                    for (int iy = 0; iy < simulation_options_global->DIM; iy++) {
                        for (int iz = 0; iz < D_PARA; iz++) {
                            unsigned long long src_idx = iz + D_PARA * (iy + (unsigned long long)simulation_options_global->DIM * ix);
                            unsigned long long dst_idx = R_FFT_INDEX(ix, iy, iz);
                            hires_float3[dst_idx] = temp_real3[src_idx];
                        }
                    }
                }
                free(temp_real3);
            }

            // ============ GPU: Store velocity to output ============
            err = cudaMemcpy(d_realspace_box, (float *)HIRES_box, fft_size, cudaMemcpyHostToDevice);
            CATCH_CUDA_ERROR(err);

            int vel_dimension, vel_d_para;
            unsigned long long vel_num_pixels;
            float *output_ptr;

            if (matter_options_global->PERTURB_ON_HIGH_RES) {
                vel_dimension = simulation_options_global->DIM;
                vel_d_para = D_PARA;
                vel_num_pixels = TOT_NUM_PIXELS;
                if (ii == 0) output_ptr = boxes->hires_vx;
                else if (ii == 1) output_ptr = boxes->hires_vy;
                else output_ptr = boxes->hires_vz;
            } else {
                vel_dimension = simulation_options_global->HII_DIM;
                vel_d_para = HII_D_PARA;
                vel_num_pixels = HII_TOT_NUM_PIXELS;
                if (ii == 0) output_ptr = boxes->lowres_vx;
                else if (ii == 1) output_ptr = boxes->lowres_vy;
                else output_ptr = boxes->lowres_vz;
            }

            numBlocks = (vel_num_pixels + threadsPerBlock - 1) / threadsPerBlock;

            store_velocity_kernel<<<numBlocks, threadsPerBlock>>>(
                d_realspace_box, d_output_box,
                vel_dimension, vel_d_para,
                simulation_options_global->DIM, MID_PARA,
                f_pixel_factor,
                matter_options_global->PERTURB_ON_HIGH_RES
            );

            err = cudaDeviceSynchronize();
            CATCH_CUDA_ERROR(err);
            err = cudaGetLastError();
            CATCH_CUDA_ERROR(err);

            err = cudaMemcpy(output_ptr, d_output_box, vel_output_size, cudaMemcpyDeviceToHost);
            CATCH_CUDA_ERROR(err);
        }

        cudaFree(d_kspace_box);
        cudaFree(d_realspace_box);
        cudaFree(d_output_box);

        LOG_DEBUG("Computed velocity fields");

        // ============ 2LPT: Second-order Lagrangian Perturbation Theory ============
        // Reference: Scoccimarro R., 1998, MNRAS, 299, 1097-1118 Appendix D
        if (matter_options_global->PERTURB_ALGORITHM == 2) {
            LOG_DEBUG("Starting 2LPT computation");

            // Define kernel launch parameters for 2LPT
            int threadsPerBlock = 256;
            int numBlocks_kspace = (KSPACE_NUM_PIXELS + threadsPerBlock - 1) / threadsPerBlock;
            int numBlocks_real = (TOT_NUM_PIXELS + threadsPerBlock - 1) / threadsPerBlock;

            // Allocate GPU memory for phi_1 components and 2LPT source
            // We need to store diagonal components temporarily, then compute cross components
            // phi_00 -> stored in d_phi_00 (tightly packed, no FFT padding)
            // phi_11 -> stored in d_phi_11
            // phi_22 -> stored in d_phi_22
            float *d_phi_00, *d_phi_11, *d_phi_22;
            cufftComplex *d_phi1_kspace;  // Reusable k-space buffer for phi_1 components

            err = cudaMalloc(&d_phi_00, real_bytes_needed);
            CATCH_CUDA_ERROR(err);
            err = cudaMalloc(&d_phi_11, real_bytes_needed);
            if (err != cudaSuccess) { cudaFree(d_phi_00); CATCH_CUDA_ERROR(err); }
            err = cudaMalloc(&d_phi_22, real_bytes_needed);
            if (err != cudaSuccess) { cudaFree(d_phi_00); cudaFree(d_phi_11); CATCH_CUDA_ERROR(err); }
            err = cudaMalloc(&d_phi1_kspace, kspace_bytes);
            if (err != cudaSuccess) { cudaFree(d_phi_00); cudaFree(d_phi_11); cudaFree(d_phi_22); CATCH_CUDA_ERROR(err); }

            LOG_DEBUG("Allocated 2LPT GPU memory");

            // Copy saved k-space density to GPU
            err = cudaMemcpy(d_kspace, HIRES_box_saved, kspace_bytes, cudaMemcpyHostToDevice);
            CATCH_CUDA_ERROR(err);

            // Compute diagonal phi_1 components (ii = 00, 11, 22)
            float *d_phi_diag[3] = {d_phi_00, d_phi_11, d_phi_22};
            for (int phi_comp = 0; phi_comp < 3; phi_comp++) {
                // Compute phi_1[k] = -k[i] * k[j] * delta_k / k^2 / VOLUME (with i=j)
                compute_phi1_kernel<<<numBlocks_kspace, threadsPerBlock>>>(
                    d_kspace, d_phi1_kspace,
                    simulation_options_global->DIM, MIDDLE, MIDDLE_PARA,
                    DELTA_K, DELTA_K_PARA, VOLUME,
                    phi_comp, phi_comp  // diagonal: i = j
                );
                err = cudaDeviceSynchronize();
                CATCH_CUDA_ERROR(err);

                // FFT to real space
                cufft_status = cufftExecC2R(fft_plan, d_phi1_kspace, d_realspace);
                if (cufft_status != CUFFT_SUCCESS) {
                    LOG_ERROR("cuFFT C2R failed for phi_%d%d: %d", phi_comp, phi_comp, cufft_status);
                    Throw(CUDAError);
                }
                err = cudaDeviceSynchronize();
                CATCH_CUDA_ERROR(err);

                // Copy to diagonal storage (tightly packed)
                copy_cufft_to_no_padding_kernel<<<numBlocks_real, threadsPerBlock>>>(
                    d_realspace, d_phi_diag[phi_comp],
                    simulation_options_global->DIM, D_PARA
                );
                err = cudaDeviceSynchronize();
                CATCH_CUDA_ERROR(err);
            }
            LOG_DEBUG("Computed diagonal phi_1 components");

            // Zero out HIRES_box for accumulating 2LPT source
            // Use d_realspace as the accumulator (with FFT padding)
            zero_fft_buffer_kernel<<<numBlocks_real, threadsPerBlock>>>(
                d_realspace,
                simulation_options_global->DIM, D_PARA, MID_PARA
            );
            err = cudaDeviceSynchronize();
            CATCH_CUDA_ERROR(err);

            // Copy zeroed buffer back to have a clean padded buffer
            // We'll use HIRES_box as the CPU-side accumulator
            memset(HIRES_box, 0, sizeof(fftwf_complex) * KSPACE_NUM_PIXELS);

            // Compute cross phi_1 components (ij = 01, 02, 12) and accumulate source
            int phi_directions[3][2] = {{0, 1}, {0, 2}, {1, 2}};
            for (int phi_comp = 0; phi_comp < 3; phi_comp++) {
                int comp_i = phi_directions[phi_comp][0];
                int comp_j = phi_directions[phi_comp][1];

                // Compute phi_1[k] = -k[i] * k[j] * delta_k / k^2 / VOLUME (cross term)
                compute_phi1_kernel<<<numBlocks_kspace, threadsPerBlock>>>(
                    d_kspace, d_phi1_kspace,
                    simulation_options_global->DIM, MIDDLE, MIDDLE_PARA,
                    DELTA_K, DELTA_K_PARA, VOLUME,
                    comp_i, comp_j
                );
                err = cudaDeviceSynchronize();
                CATCH_CUDA_ERROR(err);

                // FFT to real space - result goes to d_realspace temporarily
                // But we need d_realspace for accumulation, so use a different approach:
                // Copy d_realspace (accumulator) to host, do FFT, then accumulate

                // Actually, let's allocate a temporary buffer for the cross component FFT result
                float *d_phi_ij_real;
                err = cudaMalloc(&d_phi_ij_real, real_bytes);
                CATCH_CUDA_ERROR(err);

                cufft_status = cufftExecC2R(fft_plan, d_phi1_kspace, d_phi_ij_real);
                if (cufft_status != CUFFT_SUCCESS) {
                    LOG_ERROR("cuFFT C2R failed for phi_%d%d: %d", comp_i, comp_j, cufft_status);
                    cudaFree(d_phi_ij_real);
                    Throw(CUDAError);
                }
                err = cudaDeviceSynchronize();
                CATCH_CUDA_ERROR(err);

                // Accumulate: source += phi_ii * phi_jj - phi_ij^2
                // Note: d_phi_ij_real is tightly packed (cuFFT output), d_phi_diag are tightly packed
                // We need to handle this carefully - use d_realspace as source with FFT padding

                // First, copy the current accumulator state to d_realspace with padding
                // Actually, let's do the accumulation on GPU with a modified kernel that handles
                // the tightly packed phi_ij

                // For simplicity, let's copy phi_ij to d_realspace with padding, then accumulate
                // Copy phi_ij (tightly packed) to temp buffer and add padding to d_realspace
                float *temp_phi_ij = (float *)malloc(real_bytes_needed);
                if (!temp_phi_ij) {
                    cudaFree(d_phi_ij_real);
                    LOG_ERROR("Failed to allocate temp_phi_ij");
                    Throw(MemoryAllocError);
                }
                err = cudaMemcpy(temp_phi_ij, d_phi_ij_real, real_bytes_needed, cudaMemcpyDeviceToHost);
                CATCH_CUDA_ERROR(err);
                cudaFree(d_phi_ij_real);

                // Get diagonal components to host
                float *temp_phi_ii = (float *)malloc(real_bytes_needed);
                float *temp_phi_jj = (float *)malloc(real_bytes_needed);
                if (!temp_phi_ii || !temp_phi_jj) {
                    free(temp_phi_ij);
                    if (temp_phi_ii) free(temp_phi_ii);
                    LOG_ERROR("Failed to allocate temp_phi_ii/jj");
                    Throw(MemoryAllocError);
                }
                err = cudaMemcpy(temp_phi_ii, d_phi_diag[comp_i], real_bytes_needed, cudaMemcpyDeviceToHost);
                CATCH_CUDA_ERROR(err);
                err = cudaMemcpy(temp_phi_jj, d_phi_diag[comp_j], real_bytes_needed, cudaMemcpyDeviceToHost);
                CATCH_CUDA_ERROR(err);

                // Accumulate on CPU: HIRES_box += phi_ii * phi_jj - phi_ij^2
                float *hires_real = (float *)HIRES_box;
                #pragma omp parallel for collapse(2) num_threads(simulation_options_global->N_THREADS)
                for (int ix = 0; ix < simulation_options_global->DIM; ix++) {
                    for (int iy = 0; iy < simulation_options_global->DIM; iy++) {
                        for (int iz = 0; iz < D_PARA; iz++) {
                            unsigned long long r_idx = iz + D_PARA * (iy + (unsigned long long)simulation_options_global->DIM * ix);
                            unsigned long long fft_idx = R_FFT_INDEX(ix, iy, iz);
                            float val_ii = temp_phi_ii[r_idx];
                            float val_jj = temp_phi_jj[r_idx];
                            float val_ij = temp_phi_ij[r_idx];
                            hires_real[fft_idx] += val_ii * val_jj - val_ij * val_ij;
                        }
                    }
                }

                free(temp_phi_ij);
                free(temp_phi_ii);
                free(temp_phi_jj);
            }
            LOG_DEBUG("Accumulated 2LPT source term");

            // Normalize source by TOT_NUM_PIXELS
            float *hires_real = (float *)HIRES_box;
            #pragma omp parallel for collapse(2) num_threads(simulation_options_global->N_THREADS)
            for (int ix = 0; ix < simulation_options_global->DIM; ix++) {
                for (int iy = 0; iy < simulation_options_global->DIM; iy++) {
                    for (int iz = 0; iz < D_PARA; iz++) {
                        unsigned long long fft_idx = R_FFT_INDEX(ix, iy, iz);
                        hires_real[fft_idx] /= (float)TOT_NUM_PIXELS;
                    }
                }
            }

            // FFT source to k-space (R2C)
            // Need to create R2C plan
            cufftHandle fft_plan_r2c;
            cufft_status = cufftPlan3d(&fft_plan_r2c, fft_nx, fft_ny, fft_nz, CUFFT_R2C);
            if (cufft_status != CUFFT_SUCCESS) {
                LOG_ERROR("cuFFT R2C plan creation failed: %d", cufft_status);
                Throw(CUDAError);
            }

            // cuFFT R2C expects tightly-packed real input (DIM x DIM x D_PARA)
            // but HIRES_box has FFT padding (DIM x DIM x 2*(MID_PARA+1))
            // We need to extract the real data without padding first
            float *source_packed = (float *)malloc(real_bytes_needed);
            if (!source_packed) {
                LOG_ERROR("Failed to allocate source_packed for 2LPT R2C");
                cufftDestroy(fft_plan_r2c);
                Throw(MemoryAllocError);
            }
            float *hires_src = (float *)HIRES_box;
            #pragma omp parallel for collapse(2) num_threads(simulation_options_global->N_THREADS)
            for (int ix = 0; ix < simulation_options_global->DIM; ix++) {
                for (int iy = 0; iy < simulation_options_global->DIM; iy++) {
                    for (int iz = 0; iz < D_PARA; iz++) {
                        unsigned long long fft_idx = R_FFT_INDEX(ix, iy, iz);
                        unsigned long long packed_idx = iz + D_PARA * (iy + (unsigned long long)simulation_options_global->DIM * ix);
                        source_packed[packed_idx] = hires_src[fft_idx];
                    }
                }
            }

            // Copy tightly-packed source to GPU
            float *d_source_real;
            err = cudaMalloc(&d_source_real, real_bytes_needed);
            if (err != cudaSuccess) {
                free(source_packed);
                cufftDestroy(fft_plan_r2c);
                CATCH_CUDA_ERROR(err);
            }
            err = cudaMemcpy(d_source_real, source_packed, real_bytes_needed, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                free(source_packed);
                cudaFree(d_source_real);
                cufftDestroy(fft_plan_r2c);
                CATCH_CUDA_ERROR(err);
            }
            free(source_packed);

            cufft_status = cufftExecR2C(fft_plan_r2c, d_source_real, d_kspace);
            if (cufft_status != CUFFT_SUCCESS) {
                LOG_ERROR("cuFFT R2C execution failed: %d", cufft_status);
                cudaFree(d_source_real);
                cufftDestroy(fft_plan_r2c);
                Throw(CUDAError);
            }
            err = cudaDeviceSynchronize();
            CATCH_CUDA_ERROR(err);
            cudaFree(d_source_real);
            cufftDestroy(fft_plan_r2c);

            // Save phi_2 k-space for velocity computation
            err = cudaMemcpy(HIRES_box_saved, d_kspace, kspace_bytes, cudaMemcpyDeviceToHost);
            CATCH_CUDA_ERROR(err);

            LOG_DEBUG("FFT'd 2LPT source to k-space");

            // Scale phi_2 k-space by VOLUME to match the scaling convention used in the velocity kernel
            // The velocity kernel divides by VOLUME (needed for base velocity which has sqrt(VOLUME) scaling),
            // but phi_2 doesn't have that scaling, so we pre-multiply by VOLUME to compensate.
            {
                float *phi2_kspace = (float *)HIRES_box_saved;
                #pragma omp parallel for num_threads(simulation_options_global->N_THREADS)
                for (size_t i = 0; i < 2 * KSPACE_NUM_PIXELS; i++) {
                    phi2_kspace[i] *= VOLUME;
                }
            }

            // Compute 2LPT velocities (same as ZA but from phi_2)
            for (ii = 0; ii < 3; ii++) {
                // Copy phi_2 k-space (scaled by VOLUME) to GPU
                err = cudaMemcpy(d_kspace, HIRES_box_saved, kspace_bytes, cudaMemcpyHostToDevice);
                CATCH_CUDA_ERROR(err);

                // Compute velocity: v_k = i * k_component / k^2 * phi_2_k
                compute_velocity_kernel<<<numBlocks_kspace, threadsPerBlock>>>(
                    d_kspace,
                    simulation_options_global->DIM, MIDDLE, MIDDLE_PARA,
                    DELTA_K, DELTA_K_PARA, VOLUME,
                    ii
                );
                err = cudaDeviceSynchronize();
                CATCH_CUDA_ERROR(err);

                // Filter if needed (for lowres output)
                if (!matter_options_global->PERTURB_ON_HIGH_RES &&
                    simulation_options_global->DIM != simulation_options_global->HII_DIM) {
                    err = cudaMemcpy(HIRES_box, d_kspace, kspace_bytes, cudaMemcpyDeviceToHost);
                    CATCH_CUDA_ERROR(err);
                    filter_box(HIRES_box, hi_dim, 0,
                               L_FACTOR * simulation_options_global->BOX_LEN /
                                   (simulation_options_global->HII_DIM + 0.0),
                               0., 0.);
                    err = cudaMemcpy(d_kspace, HIRES_box, kspace_bytes, cudaMemcpyHostToDevice);
                    CATCH_CUDA_ERROR(err);
                }

                // FFT to real space
                cufft_status = cufftExecC2R(fft_plan, d_kspace, d_realspace);
                if (cufft_status != CUFFT_SUCCESS) {
                    LOG_ERROR("cuFFT C2R failed for 2LPT velocity %d: %d", ii, cufft_status);
                    Throw(CUDAError);
                }
                err = cudaDeviceSynchronize();
                CATCH_CUDA_ERROR(err);

                // Copy back and store in output
                float *temp_vel = (float *)malloc(real_bytes_needed);
                if (!temp_vel) {
                    LOG_ERROR("Failed to allocate temp buffer for 2LPT velocity");
                    Throw(MemoryAllocError);
                }
                err = cudaMemcpy(temp_vel, d_realspace, real_bytes_needed, cudaMemcpyDeviceToHost);
                CATCH_CUDA_ERROR(err);

                // Store to output arrays
                if (matter_options_global->PERTURB_ON_HIGH_RES) {
                    float *output_ptr = (ii == 0) ? boxes->hires_vx_2LPT :
                                        (ii == 1) ? boxes->hires_vy_2LPT : boxes->hires_vz_2LPT;
                    #pragma omp parallel for collapse(2) num_threads(simulation_options_global->N_THREADS)
                    for (int ix = 0; ix < simulation_options_global->DIM; ix++) {
                        for (int iy = 0; iy < simulation_options_global->DIM; iy++) {
                            for (int iz = 0; iz < D_PARA; iz++) {
                                unsigned long long r_idx = iz + D_PARA * (iy + (unsigned long long)simulation_options_global->DIM * ix);
                                output_ptr[r_idx] = temp_vel[r_idx];
                            }
                        }
                    }
                } else {
                    float *output_ptr = (ii == 0) ? boxes->lowres_vx_2LPT :
                                        (ii == 1) ? boxes->lowres_vy_2LPT : boxes->lowres_vz_2LPT;
                    #pragma omp parallel for collapse(2) num_threads(simulation_options_global->N_THREADS)
                    for (int ix = 0; ix < simulation_options_global->HII_DIM; ix++) {
                        for (int iy = 0; iy < simulation_options_global->HII_DIM; iy++) {
                            for (int iz = 0; iz < HII_D_PARA; iz++) {
                                int hi = (int)(ix * f_pixel_factor + 0.5f);
                                int hj = (int)(iy * f_pixel_factor + 0.5f);
                                int hk = (int)(iz * f_pixel_factor + 0.5f);
                                unsigned long long src_idx = hk + D_PARA * (hj + (unsigned long long)simulation_options_global->DIM * hi);
                                unsigned long long dst_idx = iz + HII_D_PARA * (iy + (unsigned long long)simulation_options_global->HII_DIM * ix);
                                output_ptr[dst_idx] = temp_vel[src_idx];
                            }
                        }
                    }
                }
                free(temp_vel);
            }

            // Free 2LPT GPU memory
            cudaFree(d_phi_00);
            cudaFree(d_phi_11);
            cudaFree(d_phi_22);
            cudaFree(d_phi1_kspace);

            LOG_DEBUG("Completed 2LPT computation");
        }

        // ============ Cleanup ============
        // Free cuFFT resources
        cufftDestroy(fft_plan);
        cudaFree(d_kspace);
        cudaFree(d_realspace);

        // Free FFTW resources (still used for memory allocation)
        fftwf_free(HIRES_box);
        fftwf_free(HIRES_box_saved);

        free_ps();
        free_rng_threads(r);

        LOG_DEBUG("ComputeInitialConditions_gpu: Complete");
    }  // End of Try{}

    Catch(status) { return (status); }
    return (0);
}
