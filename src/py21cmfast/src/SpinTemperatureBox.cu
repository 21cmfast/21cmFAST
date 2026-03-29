// Most of the following includes likely can be removed.
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

// GPU
#include <cuda.h>
#include <cuda_runtime.h>
// We use thrust for reduction
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h> // thrust::plus

#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"
#include "Constants.h"
#include "indexing.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "heating_helper_progs.h"
#include "elec_interp.h"
#include "interp_tables.h"
#include "debugging.h"
#include "cosmology.h"
#include "hmf.h"
#include "dft.h"
#include "filtering.h"
#include "thermochem.h"
#include "interpolation.h"

#include "cuda_utils.cuh"
#include "SpinTemperatureBox.h"


__device__ inline double EvaluateRGTable1D_f_gpu(double x, double x_min, double x_width, float *y_arr) {

    int idx = (int)floor((x - x_min) / x_width);

    // Clamp to valid range
    if (idx < 0) idx = 0;
    if (idx >= 399) idx = 398;

    double table_val = x_min + x_width * (float)idx;
    double interp_point = (x - table_val) / x_width;

    return y_arr[idx] * (1 - interp_point) + y_arr[idx + 1] * (interp_point);
}

template <unsigned int threadsPerBlock>
__device__ void warp_reduce(volatile double *sdata, unsigned int tid) {
    // Reduce by half
    // No syncing required with threads < 32
    if (threadsPerBlock >= 64) { sdata[tid] += sdata[tid + 32]; }
    if (threadsPerBlock >= 32) { sdata[tid] += sdata[tid + 16]; }
    if (threadsPerBlock >= 16) { sdata[tid] += sdata[tid + 8]; }
    if (threadsPerBlock >= 8) { sdata[tid] += sdata[tid + 4]; }
    if (threadsPerBlock >= 4) { sdata[tid] += sdata[tid + 2]; }
    if (threadsPerBlock >= 2) { sdata[tid] += sdata[tid + 1]; }
}

// As seen in talk by Mark Harris, NVIDIA.
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// https://www.youtube.com/watch?v=NrWhZMHrP4w
template <unsigned int threadsPerBlock>
__global__ void compute_and_reduce(
    double x_min, // reference
    double x_width, // reference
    float *y_arr, // reference
    float *dens_R_grid, // reference
    double zpp_growth_R_ct, // value
    float *sfrd_grid, // star formation rate density grid to be updated
    double *ave_sfrd_buf, // output buffer of length ceil(n / (threadsPerBlock * 2))
    unsigned long long num_pixels // length of input data
) {

    // An array to store intermediate summations
    // Shared between all threads in block
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x; // thread within current block
    unsigned int i = blockIdx.x * (threadsPerBlock * 2) + tid; // index of input data
    unsigned int gridSize = threadsPerBlock * 2 * gridDim.x;

    sdata[tid] = 0;

    // In bounds of gridSize, sum pairs of collapse fraction data together
    // And update the star formation rate density grid.
    double curr_dens_i;
    double curr_dens_j;
    double fcoll_i;
    double fcoll_j;

    while (i < num_pixels) {
        // Compute current density from density grid value * redshift-scaled growth factor
        curr_dens_i = dens_R_grid[i] * zpp_growth_R_ct;

        // Compute fraction of mass that has collapsed to form stars/other structures
        fcoll_i = exp(EvaluateRGTable1D_f_gpu(curr_dens_i, x_min, x_width, y_arr));

        // Update the shared buffer with the collapse fractions
        sdata[tid] += fcoll_i;

        // Update the relevant cells in the star formation rate density grid
        sfrd_grid[i] = (1. + curr_dens_i) * fcoll_i;

        // Repeat for i + threadsPerBlock
        if ((i + threadsPerBlock) < num_pixels) {
            curr_dens_j = dens_R_grid[i + threadsPerBlock] * zpp_growth_R_ct;
            fcoll_j = exp(EvaluateRGTable1D_f_gpu(curr_dens_j, x_min, x_width, y_arr));
            sdata[tid] += fcoll_j;
            sfrd_grid[i + threadsPerBlock] = (1. + curr_dens_j) * fcoll_j;
        }

        i += gridSize;
    }
    __syncthreads();

    // Reduce by half and sync (and repeat)
    if (threadsPerBlock >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (threadsPerBlock >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (threadsPerBlock >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    // Final reduction by separate kernel
    if (tid < 32) { warp_reduce<threadsPerBlock>(sdata, tid); }

    // The first thread of each block updates the block totals
    if (tid == 0) {
        ave_sfrd_buf[blockIdx.x] = sdata[0];
    }
}

extern "C" unsigned int init_sfrd_gpu_data(
    float *dens_R_grid, // input data
    float *sfrd_grid, // star formation rate density grid to be updated
    unsigned long long num_pixels, // length of input data
    unsigned int nbins, // nbins for sfrd_grid->y
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_ave_sfrd_buf
) {
    // Allocate device memory
    CALL_CUDA(cudaMalloc(d_y_arr, sizeof(float) * nbins)); // already pointers to pointers (no & needed)
    CALL_CUDA(cudaMalloc(d_dens_R_grid, sizeof(float) * num_pixels));
    CALL_CUDA(cudaMalloc(d_sfrd_grid, sizeof(float) * num_pixels));
    LOG_INFO("SFRD_conditional_table.y_arr and density and sfrd grids allocated on device.");

    // Initialise sfrd_grid to 0 (fill with byte=0)
    CALL_CUDA(cudaMemset(*d_sfrd_grid, 0, sizeof(float) * num_pixels)); // dereference the pointer to a pointer (*)
    LOG_INFO("sfrd grid initialised to 0.");

    // Get max threads/block for device
    int maxThreadsPerBlock;
    CALL_CUDA(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));

    // Set threads/block based on device max
    unsigned int threadsPerBlock;
    if (maxThreadsPerBlock >= 512) {
        threadsPerBlock = 512;
    } else if (maxThreadsPerBlock >= 256) {
        threadsPerBlock = 256;
    } else if (maxThreadsPerBlock >= 128) {
        threadsPerBlock = 128;
    } else if (maxThreadsPerBlock >= 64) {
        threadsPerBlock = 64;
    } else if (maxThreadsPerBlock >= 32) {
        threadsPerBlock = 32;
    } else {
        threadsPerBlock = 16;
    }

    // Allocate memory for SFRD sum buffer and initialise to 0 only for initial filter step;
    // reuse memory for remaining filter steps.
    // Ensure at least 1 block: run_global_evolution uses HII_DIM=1 (num_pixels=1),
    // which would give numBlocks=0 and an invalid kernel launch configuration.
    unsigned int numBlocks = ceil(num_pixels / (threadsPerBlock * 2));
    if (numBlocks == 0) numBlocks = 1;
    CALL_CUDA(cudaMalloc(d_ave_sfrd_buf, sizeof(double) * numBlocks)); // already pointer to a pointer (no & needed)
    LOG_INFO("SFRD sum reduction buffer allocated on device.");

    // Initialise buffer to 0 (fill with byte=0)
    CALL_CUDA(cudaMemset(*d_ave_sfrd_buf, 0, sizeof(double) * numBlocks)); // dereference the pointer to a pointer (*)
    LOG_INFO("SFRD sum reduction buffer initialised to 0.");

    return threadsPerBlock;
}

extern "C" double calculate_sfrd_from_grid_gpu(
    RGTable1D_f *SFRD_conditional_table, // input data
    float *dens_R_grid, // input data
    double *zpp_growth, // input data
    int R_ct, // filter step/loop iteration/spherical annuli (out of 40 by default)
    float *sfrd_grid, // star formation rate density grid to be updated
    unsigned long long num_pixels, // length of input data
    unsigned int threadsPerBlock, // computed in init function
    float *d_y_arr,
    float *d_dens_R_grid,
    float *d_sfrd_grid,
    double *d_ave_sfrd_buf,
    struct ScalingConstants *sc  // parameter added to match header declaration but not used in this implementation
) {
    // Get growth factor for current filter step
    double zpp_growth_R_ct = zpp_growth[R_ct];

    // Copy data from host to device
    CALL_CUDA(cudaMemcpy(d_y_arr, SFRD_conditional_table->y_arr, sizeof(float) * SFRD_conditional_table->n_bin, cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMemcpy(d_dens_R_grid, dens_R_grid, sizeof(float) * num_pixels, cudaMemcpyHostToDevice));
    LOG_INFO("SFRD_conditional_table.y_arr and density grid copied to device.");

    // Ensure at least 1 block (see allocation above for rationale).
    unsigned int numBlocks = ceil(num_pixels / (threadsPerBlock * 2));
    if (numBlocks == 0) numBlocks = 1;
    unsigned int smemSize = threadsPerBlock * sizeof(double); // shared memory

    // Invoke kernel
    switch (threadsPerBlock) {
        case 512:
            compute_and_reduce<512><<< numBlocks, threadsPerBlock, smemSize >>>(SFRD_conditional_table->x_min, SFRD_conditional_table->x_width, d_y_arr, d_dens_R_grid, zpp_growth_R_ct, d_sfrd_grid, d_ave_sfrd_buf, num_pixels);
            break;
        case 256:
            compute_and_reduce<256><<< numBlocks, threadsPerBlock, smemSize >>>(SFRD_conditional_table->x_min, SFRD_conditional_table->x_width, d_y_arr, d_dens_R_grid, zpp_growth_R_ct, d_sfrd_grid, d_ave_sfrd_buf, num_pixels);
            break;
        case 128:
            compute_and_reduce<128><<< numBlocks, threadsPerBlock, smemSize >>>(SFRD_conditional_table->x_min, SFRD_conditional_table->x_width, d_y_arr, d_dens_R_grid, zpp_growth_R_ct, d_sfrd_grid, d_ave_sfrd_buf, num_pixels);
            break;
        case 64:
            compute_and_reduce<64><<< numBlocks, threadsPerBlock, smemSize >>>(SFRD_conditional_table->x_min, SFRD_conditional_table->x_width, d_y_arr, d_dens_R_grid, zpp_growth_R_ct, d_sfrd_grid, d_ave_sfrd_buf, num_pixels);
            break;
        case 32:
            compute_and_reduce<32><<< numBlocks, threadsPerBlock, smemSize >>>(SFRD_conditional_table->x_min, SFRD_conditional_table->x_width, d_y_arr, d_dens_R_grid, zpp_growth_R_ct, d_sfrd_grid, d_ave_sfrd_buf, num_pixels);
            break;
        default:
            LOG_WARNING("Thread size invalid; defaulting to 256.");
            compute_and_reduce<256><<< numBlocks, 256, 256 * sizeof(double) >>>(SFRD_conditional_table->x_min, SFRD_conditional_table->x_width, d_y_arr, d_dens_R_grid, zpp_growth_R_ct, d_sfrd_grid, d_ave_sfrd_buf, num_pixels);
    }
    CALL_CUDA(cudaGetLastError());
    CALL_CUDA(cudaDeviceSynchronize());
    LOG_INFO("SpinTemperatureBox compute-and-reduce kernel called.");

    // Use thrust to reduce computed sums to one value.
    // Wrap device pointer in a thrust::device_ptr
    thrust::device_ptr<double> d_ave_sfrd_buf_ptr(d_ave_sfrd_buf);
    // Reduce final buffer sums to one value
    double ave_sfrd_buf = thrust::reduce(d_ave_sfrd_buf_ptr, d_ave_sfrd_buf_ptr + numBlocks, 0., thrust::plus<double>());
    CALL_CUDA(cudaGetLastError());
    LOG_INFO("SFRD sum reduced to single value by thrust::reduce operation.");

    // Copy results from device to host
    CALL_CUDA(cudaMemcpy(sfrd_grid, d_sfrd_grid, sizeof(float) * num_pixels, cudaMemcpyDeviceToHost));
    LOG_INFO("SFRD sum copied to host.");

    return ave_sfrd_buf;
}

extern "C" void free_sfrd_gpu_data(
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_ave_sfrd_buf
) {
    // Need to dereference the pointers to pointers (*)
    CALL_CUDA(cudaFree(*d_y_arr));
    CALL_CUDA(cudaFree(*d_dens_R_grid));
    CALL_CUDA(cudaFree(*d_sfrd_grid));
    CALL_CUDA(cudaFree(*d_ave_sfrd_buf));
    LOG_INFO("Device memory freed.");
}

// ============================================================================
// Phase 11.6a: Spectral integration GPU kernel
// ============================================================================
//
// Replaces the OpenMP per-pixel loop in ts_main that accumulates
// dxheat_dt_box, dxion_source_dt_box, dxlya_dt_box, dstarlya_dt_box
// (and optionally dstarlyLW_dt_box, dstarlya_cont_dt_box, dstarlya_inj_dt_box)
// from frequency integral table lookups per R iteration.
//
// The host manages the sequential R-loop; this kernel handles pixel parallelism.

// Flattened frequency integral tables on device.
// Layout: tbl[xidx * max_n_step + R_ct], where xidx in [0, x_int_NXHII)
// and R_ct in [0, N_STEP_TS).
struct SpectralIntegDeviceData {
    // Frequency integral tables (flattened [x_int_NXHII * N_STEP_TS])
    double *d_freq_int_heat_tbl;
    double *d_freq_int_ion_tbl;
    double *d_freq_int_lya_tbl;
    double *d_freq_int_heat_tbl_diff;
    double *d_freq_int_ion_tbl_diff;
    double *d_freq_int_lya_tbl_diff;

    // Per-pixel precomputed arrays
    int *d_m_xHII_low_box;
    float *d_inverse_val_box;

    // Per-pixel input: del_fcoll_Rct (uploaded per R for SOURCE_MODEL < 2)
    float *d_del_fcoll_Rct;
    float *d_del_fcoll_Rct_MINI;

    // Accumulation arrays (persistent across R iterations)
    double *d_dxheat_dt_box;
    double *d_dxion_source_dt_box;
    double *d_dxlya_dt_box;
    double *d_dstarlya_dt_box;
    double *d_dstarlyLW_dt_box;
    double *d_dstarlya_cont_dt_box;
    double *d_dstarlya_inj_dt_box;

    // Dimensions
    unsigned long long num_pixels;
    int n_step_ts;

    // Feature flags (copied from host options at init time)
    bool use_mini_halos;
    bool use_x_ray_heating;
    bool use_lya_heating;
};

__global__ void spectral_integration_kernel(
    int R_ct,
    int n_step_ts,
    unsigned long long num_pixels,
    // Per-pixel input
    float *d_del_fcoll_Rct,
    float *d_del_fcoll_Rct_MINI,
    int *d_m_xHII_low_box,
    float *d_inverse_val_box,
    // Frequency integral tables (flattened [x_int_NXHII * n_step_ts])
    double *d_freq_int_heat_tbl,
    double *d_freq_int_ion_tbl,
    double *d_freq_int_lya_tbl,
    double *d_freq_int_heat_tbl_diff,
    double *d_freq_int_ion_tbl_diff,
    double *d_freq_int_lya_tbl_diff,
    // Accumulation arrays (output, +=)
    double *d_dxheat_dt_box,
    double *d_dxion_source_dt_box,
    double *d_dxlya_dt_box,
    double *d_dstarlya_dt_box,
    double *d_dstarlyLW_dt_box,
    double *d_dstarlya_cont_dt_box,
    double *d_dstarlya_inj_dt_box,
    // R-dependent scalars (passed per kernel call)
    double z_edge_factor,
    double xray_R_factor,
    double avg_fix_term,
    double avg_fix_term_MINI,
    double F_STAR10,
    double L_X,
    double s_per_yr,
    double F_STAR7_MINI,
    double L_X_MINI,
    double dstarlya_dt_prefactor_R,
    double dstarlyLW_dt_prefactor_R,
    double dstarlyLW_dt_prefactor_MINI_R,
    double dstarlya_dt_prefactor_MINI_R,
    double dstarlya_cont_dt_prefactor_R,
    double dstarlya_inj_dt_prefactor_R,
    double dstarlya_cont_dt_prefactor_MINI_R,
    double dstarlya_inj_dt_prefactor_MINI_R,
    // Feature flags
    bool use_mini_halos,
    bool use_x_ray_heating,
    bool use_lya_heating
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    float fcoll = d_del_fcoll_Rct[idx];
    double sfr_term = (double)fcoll * z_edge_factor * avg_fix_term * F_STAR10;
    double xray_sfr = sfr_term * L_X * xray_R_factor * s_per_yr;

    double sfr_term_mini = 0.0;
    if (use_mini_halos) {
        sfr_term_mini = (double)d_del_fcoll_Rct_MINI[idx] * z_edge_factor *
                        avg_fix_term_MINI * F_STAR7_MINI;
        xray_sfr += sfr_term_mini * L_X_MINI * xray_R_factor * s_per_yr;

        d_dstarlyLW_dt_box[idx] +=
            sfr_term * dstarlyLW_dt_prefactor_R +
            sfr_term_mini * dstarlyLW_dt_prefactor_MINI_R;
    }

    int xidx = d_m_xHII_low_box[idx];
    double ival = (double)d_inverse_val_box[idx];

    // Table index: flattened [xidx][R_ct] -> xidx * n_step_ts + R_ct
    int tbl_idx = xidx * n_step_ts + R_ct;

    if (use_x_ray_heating) {
        d_dxheat_dt_box[idx] +=
            xray_sfr * (d_freq_int_heat_tbl_diff[tbl_idx] * ival +
                        d_freq_int_heat_tbl[tbl_idx]);
    }

    d_dxion_source_dt_box[idx] +=
        xray_sfr * (d_freq_int_ion_tbl_diff[tbl_idx] * ival +
                    d_freq_int_ion_tbl[tbl_idx]);

    d_dxlya_dt_box[idx] +=
        xray_sfr * (d_freq_int_lya_tbl_diff[tbl_idx] * ival +
                    d_freq_int_lya_tbl[tbl_idx]);

    d_dstarlya_dt_box[idx] +=
        sfr_term * dstarlya_dt_prefactor_R +
        sfr_term_mini * dstarlya_dt_prefactor_MINI_R;

    if (use_lya_heating) {
        d_dstarlya_cont_dt_box[idx] +=
            sfr_term * dstarlya_cont_dt_prefactor_R +
            sfr_term_mini * dstarlya_cont_dt_prefactor_MINI_R;
        d_dstarlya_inj_dt_box[idx] +=
            sfr_term * dstarlya_inj_dt_prefactor_R +
            sfr_term_mini * dstarlya_inj_dt_prefactor_MINI_R;
    }
}

// Initialise device data for spectral integration.
// Flattens 2D tables [x_int_NXHII][N_STEP_TS] into 1D device arrays.
// Allocates persistent accumulation buffers.
extern "C" SpectralIntegDeviceData *init_spectral_integration_gpu(
    unsigned long long num_pixels,
    int n_step_ts,
    // Host 2D tables (double*[x_int_NXHII], each row has n_step_ts entries)
    double **freq_int_heat_tbl,
    double **freq_int_ion_tbl,
    double **freq_int_lya_tbl,
    double **freq_int_heat_tbl_diff,
    double **freq_int_ion_tbl_diff,
    double **freq_int_lya_tbl_diff,
    // Host per-pixel arrays
    int *m_xHII_low_box,
    float *inverse_val_box,
    // Feature flags
    bool use_mini_halos,
    bool use_x_ray_heating,
    bool use_lya_heating
) {
    SpectralIntegDeviceData *dev = (SpectralIntegDeviceData *)malloc(sizeof(SpectralIntegDeviceData));
    dev->num_pixels = num_pixels;
    dev->n_step_ts = n_step_ts;
    dev->use_mini_halos = use_mini_halos;
    dev->use_x_ray_heating = use_x_ray_heating;
    dev->use_lya_heating = use_lya_heating;

    int nxhii = 14;  // x_int_NXHII
    size_t tbl_size = nxhii * n_step_ts * sizeof(double);
    size_t pix_size_d = num_pixels * sizeof(double);
    size_t pix_size_f = num_pixels * sizeof(float);
    size_t pix_size_i = num_pixels * sizeof(int);

    // Flatten and upload frequency integral tables
    double *h_flat = (double *)malloc(tbl_size);

    auto upload_table = [&](double **host_tbl, double **d_tbl) {
        for (int i = 0; i < nxhii; i++)
            memcpy(h_flat + i * n_step_ts, host_tbl[i], n_step_ts * sizeof(double));
        CALL_CUDA(cudaMalloc(d_tbl, tbl_size));
        CALL_CUDA(cudaMemcpy(*d_tbl, h_flat, tbl_size, cudaMemcpyHostToDevice));
    };

    upload_table(freq_int_heat_tbl, &dev->d_freq_int_heat_tbl);
    upload_table(freq_int_ion_tbl, &dev->d_freq_int_ion_tbl);
    upload_table(freq_int_lya_tbl, &dev->d_freq_int_lya_tbl);
    upload_table(freq_int_heat_tbl_diff, &dev->d_freq_int_heat_tbl_diff);
    upload_table(freq_int_ion_tbl_diff, &dev->d_freq_int_ion_tbl_diff);
    upload_table(freq_int_lya_tbl_diff, &dev->d_freq_int_lya_tbl_diff);
    free(h_flat);

    // Upload per-pixel index arrays
    CALL_CUDA(cudaMalloc(&dev->d_m_xHII_low_box, pix_size_i));
    CALL_CUDA(cudaMemcpy(dev->d_m_xHII_low_box, m_xHII_low_box, pix_size_i, cudaMemcpyHostToDevice));

    CALL_CUDA(cudaMalloc(&dev->d_inverse_val_box, pix_size_f));
    CALL_CUDA(cudaMemcpy(dev->d_inverse_val_box, inverse_val_box, pix_size_f, cudaMemcpyHostToDevice));

    // Allocate per-pixel input buffers (filled per R)
    CALL_CUDA(cudaMalloc(&dev->d_del_fcoll_Rct, pix_size_f));
    dev->d_del_fcoll_Rct_MINI = NULL;
    if (use_mini_halos) {
        CALL_CUDA(cudaMalloc(&dev->d_del_fcoll_Rct_MINI, pix_size_f));
    }

    // Allocate accumulation arrays and zero them
    if (use_x_ray_heating) {
        CALL_CUDA(cudaMalloc(&dev->d_dxheat_dt_box, pix_size_d));
        CALL_CUDA(cudaMemset(dev->d_dxheat_dt_box, 0, pix_size_d));
    } else {
        dev->d_dxheat_dt_box = NULL;
    }

    CALL_CUDA(cudaMalloc(&dev->d_dxion_source_dt_box, pix_size_d));
    CALL_CUDA(cudaMemset(dev->d_dxion_source_dt_box, 0, pix_size_d));

    CALL_CUDA(cudaMalloc(&dev->d_dxlya_dt_box, pix_size_d));
    CALL_CUDA(cudaMemset(dev->d_dxlya_dt_box, 0, pix_size_d));

    CALL_CUDA(cudaMalloc(&dev->d_dstarlya_dt_box, pix_size_d));
    CALL_CUDA(cudaMemset(dev->d_dstarlya_dt_box, 0, pix_size_d));

    dev->d_dstarlyLW_dt_box = NULL;
    if (use_mini_halos) {
        CALL_CUDA(cudaMalloc(&dev->d_dstarlyLW_dt_box, pix_size_d));
        CALL_CUDA(cudaMemset(dev->d_dstarlyLW_dt_box, 0, pix_size_d));
    }

    dev->d_dstarlya_cont_dt_box = NULL;
    dev->d_dstarlya_inj_dt_box = NULL;
    if (use_lya_heating) {
        CALL_CUDA(cudaMalloc(&dev->d_dstarlya_cont_dt_box, pix_size_d));
        CALL_CUDA(cudaMemset(dev->d_dstarlya_cont_dt_box, 0, pix_size_d));
        CALL_CUDA(cudaMalloc(&dev->d_dstarlya_inj_dt_box, pix_size_d));
        CALL_CUDA(cudaMemset(dev->d_dstarlya_inj_dt_box, 0, pix_size_d));
    }

    LOG_INFO("Spectral integration GPU data initialised (%llu pixels, %d R steps)",
             num_pixels, n_step_ts);
    return dev;
}

// Download accumulated results from device to host arrays.
extern "C" void download_spectral_integration_results(
    SpectralIntegDeviceData *dev,
    double *dxheat_dt_box,
    double *dxion_source_dt_box,
    double *dxlya_dt_box,
    double *dstarlya_dt_box,
    double *dstarlyLW_dt_box,
    double *dstarlya_cont_dt_box,
    double *dstarlya_inj_dt_box
) {
    size_t pix_size_d = dev->num_pixels * sizeof(double);

    if (dev->use_x_ray_heating && dev->d_dxheat_dt_box)
        CALL_CUDA(cudaMemcpy(dxheat_dt_box, dev->d_dxheat_dt_box, pix_size_d, cudaMemcpyDeviceToHost));

    CALL_CUDA(cudaMemcpy(dxion_source_dt_box, dev->d_dxion_source_dt_box, pix_size_d, cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(dxlya_dt_box, dev->d_dxlya_dt_box, pix_size_d, cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(dstarlya_dt_box, dev->d_dstarlya_dt_box, pix_size_d, cudaMemcpyDeviceToHost));

    if (dev->use_mini_halos && dev->d_dstarlyLW_dt_box)
        CALL_CUDA(cudaMemcpy(dstarlyLW_dt_box, dev->d_dstarlyLW_dt_box, pix_size_d, cudaMemcpyDeviceToHost));

    if (dev->use_lya_heating) {
        if (dev->d_dstarlya_cont_dt_box)
            CALL_CUDA(cudaMemcpy(dstarlya_cont_dt_box, dev->d_dstarlya_cont_dt_box, pix_size_d, cudaMemcpyDeviceToHost));
        if (dev->d_dstarlya_inj_dt_box)
            CALL_CUDA(cudaMemcpy(dstarlya_inj_dt_box, dev->d_dstarlya_inj_dt_box, pix_size_d, cudaMemcpyDeviceToHost));
    }

    LOG_INFO("Spectral integration results downloaded from GPU");
}

// Free all device memory for spectral integration.
extern "C" void free_spectral_integration_gpu(SpectralIntegDeviceData *dev) {
    CALL_CUDA(cudaFree(dev->d_freq_int_heat_tbl));
    CALL_CUDA(cudaFree(dev->d_freq_int_ion_tbl));
    CALL_CUDA(cudaFree(dev->d_freq_int_lya_tbl));
    CALL_CUDA(cudaFree(dev->d_freq_int_heat_tbl_diff));
    CALL_CUDA(cudaFree(dev->d_freq_int_ion_tbl_diff));
    CALL_CUDA(cudaFree(dev->d_freq_int_lya_tbl_diff));

    CALL_CUDA(cudaFree(dev->d_m_xHII_low_box));
    CALL_CUDA(cudaFree(dev->d_inverse_val_box));
    CALL_CUDA(cudaFree(dev->d_del_fcoll_Rct));
    if (dev->d_del_fcoll_Rct_MINI) CALL_CUDA(cudaFree(dev->d_del_fcoll_Rct_MINI));

    if (dev->d_dxheat_dt_box) CALL_CUDA(cudaFree(dev->d_dxheat_dt_box));
    CALL_CUDA(cudaFree(dev->d_dxion_source_dt_box));
    CALL_CUDA(cudaFree(dev->d_dxlya_dt_box));
    CALL_CUDA(cudaFree(dev->d_dstarlya_dt_box));
    if (dev->d_dstarlyLW_dt_box) CALL_CUDA(cudaFree(dev->d_dstarlyLW_dt_box));
    if (dev->d_dstarlya_cont_dt_box) CALL_CUDA(cudaFree(dev->d_dstarlya_cont_dt_box));
    if (dev->d_dstarlya_inj_dt_box) CALL_CUDA(cudaFree(dev->d_dstarlya_inj_dt_box));

    LOG_INFO("Spectral integration GPU data freed");
}

// Launch the spectral integration kernel for one R iteration.
// del_fcoll_Rct is the host array for the current R (uploaded H2D per call).
extern "C" void launch_spectral_integration_kernel(
    SpectralIntegDeviceData *dev,
    int R_ct,
    float *del_fcoll_Rct,
    float *del_fcoll_Rct_MINI,
    double z_edge_factor,
    double xray_R_factor,
    double avg_fix_term,
    double avg_fix_term_MINI,
    double F_STAR10,
    double L_X,
    double s_per_yr,
    double F_STAR7_MINI,
    double L_X_MINI,
    double dstarlya_dt_prefactor_R,
    double dstarlyLW_dt_prefactor_R,
    double dstarlyLW_dt_prefactor_MINI_R,
    double dstarlya_dt_prefactor_MINI_R,
    double dstarlya_cont_dt_prefactor_R,
    double dstarlya_inj_dt_prefactor_R,
    double dstarlya_cont_dt_prefactor_MINI_R,
    double dstarlya_inj_dt_prefactor_MINI_R
) {
    size_t pix_size_f = dev->num_pixels * sizeof(float);

    // Upload del_fcoll for this R
    CALL_CUDA(cudaMemcpy(dev->d_del_fcoll_Rct, del_fcoll_Rct, pix_size_f, cudaMemcpyHostToDevice));
    if (dev->use_mini_halos && del_fcoll_Rct_MINI) {
        CALL_CUDA(cudaMemcpy(dev->d_del_fcoll_Rct_MINI, del_fcoll_Rct_MINI, pix_size_f, cudaMemcpyHostToDevice));
    }

    unsigned int threads = 256;
    unsigned int blocks = (unsigned int)((dev->num_pixels + threads - 1) / threads);

    spectral_integration_kernel<<<blocks, threads>>>(
        R_ct,
        dev->n_step_ts,
        dev->num_pixels,
        dev->d_del_fcoll_Rct,
        dev->d_del_fcoll_Rct_MINI,
        dev->d_m_xHII_low_box,
        dev->d_inverse_val_box,
        dev->d_freq_int_heat_tbl,
        dev->d_freq_int_ion_tbl,
        dev->d_freq_int_lya_tbl,
        dev->d_freq_int_heat_tbl_diff,
        dev->d_freq_int_ion_tbl_diff,
        dev->d_freq_int_lya_tbl_diff,
        dev->d_dxheat_dt_box,
        dev->d_dxion_source_dt_box,
        dev->d_dxlya_dt_box,
        dev->d_dstarlya_dt_box,
        dev->d_dstarlyLW_dt_box,
        dev->d_dstarlya_cont_dt_box,
        dev->d_dstarlya_inj_dt_box,
        z_edge_factor,
        xray_R_factor,
        avg_fix_term,
        avg_fix_term_MINI,
        F_STAR10,
        L_X,
        s_per_yr,
        F_STAR7_MINI,
        L_X_MINI,
        dstarlya_dt_prefactor_R,
        dstarlyLW_dt_prefactor_R,
        dstarlyLW_dt_prefactor_MINI_R,
        dstarlya_dt_prefactor_MINI_R,
        dstarlya_cont_dt_prefactor_R,
        dstarlya_inj_dt_prefactor_R,
        dstarlya_cont_dt_prefactor_MINI_R,
        dstarlya_inj_dt_prefactor_MINI_R,
        dev->use_mini_halos,
        dev->use_x_ray_heating,
        dev->use_lya_heating
    );
    CALL_CUDA(cudaGetLastError());
}
