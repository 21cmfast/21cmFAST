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

#include "SpinTemperatureBox.h"


__device__ inline double EvaluateRGTable1D_f_gpu(double x, double x_min, double x_width, float *y_arr) {

    int idx = (int)floor((x - x_min) / x_width);

    double table_val = x_min + x_width * (float)idx;
    double interp_point = (x - table_val) / x_width;

    return y_arr[idx] * (1 - interp_point) + y_arr[idx + 1] * (interp_point);
}

template <unsigned int threadsPerBlock>
__device__ void warp_reduce(volatile double *sdata, unsigned int tid) {
    // Reduce by half
    // No syncing required with threads < 32
    if (threadsPerBlock >= 64) sdata[tid] += sdata[tid + 32];
    if (threadsPerBlock >= 32) sdata[tid] += sdata[tid + 16];
    if (threadsPerBlock >= 16) sdata[tid] += sdata[tid + 8];
    if (threadsPerBlock >= 8) sdata[tid] += sdata[tid + 4];
    if (threadsPerBlock >= 4) sdata[tid] += sdata[tid + 2];
    if (threadsPerBlock >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int threadsPerBlock>
__global__ void compute_and_reduce(
    double x_min, // reference
    double x_width, // reference
    float *y_arr, // reference
    float *dens_R_grid, // reference
    double zpp_growth_R_ct, // value
    float *sfrd_grid, // star formation rate density grid to be updated
    double *ave_sfrd_buf, // output buffer of length ceil(n / (threadsPerBlock * 2))
    unsigned int num_pixels // length of input data
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
        curr_dens_j = dens_R_grid[i + threadsPerBlock] * zpp_growth_R_ct;

        // Compute fraction of mass that has collapsed to form stars/other structures
        fcoll_i = exp(EvaluateRGTable1D_f_gpu(curr_dens_i, x_min, x_width, y_arr));
        fcoll_j = exp(EvaluateRGTable1D_f_gpu(curr_dens_j, x_min, x_width, y_arr));

        // Update the shared buffer with the collapse fractions
        sdata[tid] += fcoll_i + fcoll_j;

        // Update the relevant cells in the star formation rate density grid
        sfrd_grid[i] = (1. + curr_dens_i) * fcoll_i;
        sfrd_grid[i + threadsPerBlock] = (1. + curr_dens_j) * fcoll_j;

        i += gridSize;
    }
    __syncthreads();

    // Reduce by half and sync (and repeat)
    if (threadsPerBlock >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (threadsPerBlock >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (threadsPerBlock >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }

    // Final reduction by separate kernel
    if (tid < 32) warp_reduce<threadsPerBlock>(sdata, tid);

    // The first thread of each block updates the block totals
    if (tid == 0) ave_sfrd_buf[blockIdx.x] = sdata[0];
}

double calculate_sfrd_from_grid_gpu(
    RGTable1D_f *SFRD_conditional_table, // input data
    float *dens_R_grid, // input data
    double *zpp_growth, // input data
    int R_ct, // filter step/loop iteration/spherical annuli (out of 40 by default)
    float *sfrd_grid, // star formation rate density grid to be updated
    unsigned int num_pixels, // length of input data
    float *d_y_arr, float *d_dens_R_grid, float *d_sfrd_grid, double *d_ave_sfrd_buf // device pointers
) {
    cudaError_t err = cudaGetLastError();

    // Set bools for initial and final filtering steps to allow for memory reuse
    bool initial_filter_step = false;
    bool final_filter_step = false;

    // Default NUM_FILTER_STEPS_FOR_Ts = 40
    if (global_params.NUM_FILTER_STEPS_FOR_Ts - 1 == R_ct) {
        initial_filter_step = true;
    } else if (R_ct == 0) {
        final_filter_step = true;
    } else if (global_params.NUM_FILTER_STEPS_FOR_Ts == 1) {
        // Would case of NUM_FILTER_STEPS_FOR_Ts = 1 ever occur?
        initial_filter_step = true;
        final_filter_step = true;
    }

    // Get growth factor for current filter step
    double zpp_growth_R_ct = zpp_growth[R_ct];

    // ============================================================ <- these pointers need to persist across kernel calls, i.e. across loop iterations!
    // Device pointers are initialised before loop
    // if (initial_filter_step) {
    //     float *d_y_arr, *d_dens_R_grid, *d_sfrd_grid;
    //     double* d_ave_sfrd_buf;
    // }
    // ============================================================

    // Allocate device memory ------------------------------------------------------------------------------------------
    if (initial_filter_step) {
        err = cudaMalloc((void**)&d_y_arr, sizeof(float) * SFRD_conditional_table->n_bin);
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
            Throw(CUDAError);
        }
        err = cudaMalloc((void**)&d_dens_R_grid, sizeof(float) * num_pixels);
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
            Throw(CUDAError);
        }
        err = cudaMalloc((void**)&d_sfrd_grid, sizeof(float) * num_pixels);
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
            Throw(CUDAError);
        }
        LOG_INFO("SFRD_conditional_table.y_arr and density and sfrd grids allocated on device.");
    }

    // Copy data from host to device -----------------------------------------------------------------------------------
    err = cudaMemcpy(d_y_arr, SFRD_conditional_table->y_arr, sizeof(float) * SFRD_conditional_table->n_bin, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    err = cudaMemcpy(d_dens_R_grid, dens_R_grid, sizeof(float) * num_pixels, cudaMemcpyHostToDevice); // TODO: Does this change between filter steps?
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    // In initial filter step, set array to 0;
    // for future steps, previous array values will be written over
    if (initial_filter_step) {
        err = cudaMemset(d_sfrd_grid, 0, sizeof(float) * num_pixels); // fill with byte=0
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
            Throw(CUDAError);
        }
        LOG_INFO("SFRD_conditional_table.y_arr and density and sfrd grids copied to device.");
    } else {
        LOG_INFO("SFRD_conditional_table.y_arr and density grid copied to device.");
    }

    // Get max threads/block for device
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

    // Set threads/block based on device max
    int threadsPerBlock;
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
    int numBlocks = (num_pixels + threadsPerBlock - 1) / threadsPerBlock; // 91m & 256 -> 355959
    int smemSize = threadsPerBlock * sizeof(double); // shared memory

    // Allocate memory for SFRD sum buffer and initialise to 0 only for initial filter step;
    // reuse memory for remaining filter steps.
    unsigned int buffer_length = ceil(num_pixels / (threadsPerBlock * 2));
    if (initial_filter_step) {
        err = cudaMalloc((void**)&d_ave_sfrd_buf, sizeof(double) * buffer_length); // 91m & 256 -> 177979
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
            Throw(CUDAError);
        }
        LOG_INFO("SFRD sum reduction buffer allocated on device.");

        err = cudaMemset(d_ave_sfrd_buf, 0, sizeof(double) * buffer_length); // fill with byte=0
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
            Throw(CUDAError);
        }
        LOG_INFO("SFRD sum reduction buffer initialised to 0.");
    }

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
    LOG_INFO("SpinTemperatureBox compute-and-reduce kernel called.");

    // Only use during development?
    err = cudaDeviceSynchronize();
    CATCH_CUDA_ERROR(err);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("Kernel launch error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Use thrust to reduce computed sums to one value

    // Wrap device pointer in a thrust::device_ptr
    thrust::device_ptr<double> d_ave_sfrd_buf_ptr(d_ave_sfrd_buf);
    // Reduce final buffer sums to one value
    double ave_sfrd_buf = thrust::reduce(d_ave_sfrd_buf_ptr, d_ave_sfrd_buf_ptr + buffer_length, 0., thrust::plus<double>());
    LOG_INFO("SFRD sum reduced to single value by thrust::reduce operation.");

    // Copy results from device to host.
    err = cudaMemcpy(sfrd_grid, d_sfrd_grid, sizeof(float) * num_pixels, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    LOG_INFO("SFRD sum copied to host.");

    // Deallocate device memory on final filter step.
    if (final_filter_step) {
        err = cudaFree(d_y_arr);
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
            Throw(CUDAError);
        }
        err = cudaFree(d_dens_R_grid);
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
            Throw(CUDAError);
        }
        err = cudaFree(d_sfrd_grid);
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
            Throw(CUDAError);
        }
        err = cudaFree(d_ave_sfrd_buf);
        if (err != cudaSuccess) {
            LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
            Throw(CUDAError);
        }
        LOG_INFO("Device memory freed.");
    }

    return ave_sfrd_buf;
}
