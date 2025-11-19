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

    // Debug: Check for out of bounds access (nbins is typically 400)
    if (idx < 0 || idx >= 399) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            printf("=== WARNING: Table lookup idx=%d out of bounds! x=%f, x_min=%f, x_width=%f ===\n",
                   idx, x, x_min, x_width);
        }
        // Clamp to valid range
        if (idx < 0) idx = 0;
        if (idx >= 399) idx = 398;
    }

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

    // KERNEL PRINTF TEST: Simple unconditional output from first thread
    if (blockIdx.x == 0 && tid == 0) {
        printf("*** KERNEL PRINTF TEST: compute_and_reduce kernel is executing! ***\n");
    }

    // Debug output from first thread of first block
    if (blockIdx.x == 0 && tid == 0) {
        printf("=== DEVICE KERNEL EXECUTING! gridDim.x=%d, threadsPerBlock=%d, num_pixels=%llu ===\n",
               gridDim.x, threadsPerBlock, num_pixels);
        printf("=== First block will access: i=%u to %u, buffer index will be blockIdx.x=%d ===\n",
               i, i + gridSize, blockIdx.x);
        // Debug: Print table parameters and first/last entries
        printf("=== TABLE PARAMS: x_min=%f, x_width=%f, x_max=%f ===\n",
               x_min, x_width, x_min + x_width * 399);
        printf("=== TABLE VALUES: y_arr[0]=%f, y_arr[199]=%f, y_arr[398]=%f, y_arr[399]=%f ===\n",
               y_arr[0], y_arr[199], y_arr[398], y_arr[399]);
    }

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

        // Debug output from first thread for first 10 elements
        if (blockIdx.x == 0 && tid == 0 && i < 10) {
            int idx = (int)floor((curr_dens_i - x_min) / x_width);
            double table_val_at_idx = (idx >= 0 && idx < 399) ? y_arr[idx] : -999.0;
            printf("=== i=%u: curr_dens_i=%e, idx=%d, y_arr[idx]=%f, fcoll_i=%e, sfrd=%e ===\n",
                   i, curr_dens_i, idx, table_val_at_idx, fcoll_i, (float)((1. + curr_dens_i) * fcoll_i));
        }

        // Update the shared buffer with the collapse fractions
        sdata[tid] += fcoll_i;

        // Debug: First thread of first block reports first write
        if (blockIdx.x == 0 && tid == 0 && i == 0) {
            printf("=== About to write sfrd_grid[%u] = %f ===\n", i, (float)((1. + curr_dens_i) * fcoll_i));
        }

        // Update the relevant cells in the star formation rate density grid
        sfrd_grid[i] = (1. + curr_dens_i) * fcoll_i;

        // Repeat for i + threadsPerBlock
        if ((i + threadsPerBlock) < num_pixels) {
            curr_dens_j = dens_R_grid[i + threadsPerBlock] * zpp_growth_R_ct;
            fcoll_j = exp(EvaluateRGTable1D_f_gpu(curr_dens_j, x_min, x_width, y_arr));
            sdata[tid] += fcoll_j;

            // Debug: First thread of first block reports second write
            if (blockIdx.x == 0 && tid == 0 && i == 0) {
                printf("=== About to write sfrd_grid[%u] = %f ===\n", i + threadsPerBlock, (float)((1. + curr_dens_j) * fcoll_j));
            }

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
        // Debug: check buffer bounds before writing
        if (blockIdx.x < gridDim.x) {
            ave_sfrd_buf[blockIdx.x] = sdata[0];
            if (blockIdx.x == 0 || blockIdx.x == gridDim.x - 1) {
                printf("=== Block %d writing to ave_sfrd_buf[%d] = %f ===\n", blockIdx.x, blockIdx.x, sdata[0]);
            }
        } else {
            printf("=== ERROR: Block %d trying to write beyond gridDim.x=%d ===\n", blockIdx.x, gridDim.x);
        }
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
    unsigned int numBlocks = ceil(num_pixels / (threadsPerBlock * 2));
    fprintf(stderr, "=== init_sfrd_gpu_data: Allocating buffer with numBlocks=%u (num_pixels=%llu, threadsPerBlock=%u) ===\n",
            numBlocks, num_pixels, threadsPerBlock);
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
    fprintf(stderr, "=== INSIDE calculate_sfrd_from_grid_gpu! R_ct=%d, num_pixels=%llu, threadsPerBlock=%u ===\n", R_ct, num_pixels, threadsPerBlock);
    fprintf(stderr, "=== Device pointers: d_y_arr=%p, d_dens_R_grid=%p, d_sfrd_grid=%p, d_ave_sfrd_buf=%p ===\n",
            (void*)d_y_arr, (void*)d_dens_R_grid, (void*)d_sfrd_grid, (void*)d_ave_sfrd_buf);

    // Get growth factor for current filter step
    double zpp_growth_R_ct = zpp_growth[R_ct];

    // Copy data from host to device
    fprintf(stderr, "=== About to copy %u floats to d_y_arr ===\n", SFRD_conditional_table->n_bin);
    CALL_CUDA(cudaMemcpy(d_y_arr, SFRD_conditional_table->y_arr, sizeof(float) * SFRD_conditional_table->n_bin, cudaMemcpyHostToDevice));
    fprintf(stderr, "=== About to copy %llu floats to d_dens_R_grid ===\n", num_pixels);
    CALL_CUDA(cudaMemcpy(d_dens_R_grid, dens_R_grid, sizeof(float) * num_pixels, cudaMemcpyHostToDevice));
    fprintf(stderr, "=== Memcpy completed ===\n");
    LOG_INFO("SFRD_conditional_table.y_arr and density grid copied to device.");

    unsigned int numBlocks = ceil(num_pixels / (threadsPerBlock * 2));
    unsigned int smemSize = threadsPerBlock * sizeof(double); // shared memory

    fprintf(stderr, "=== Calculated numBlocks=%u for num_pixels=%llu, threadsPerBlock=%u ===\n",
            numBlocks, num_pixels, threadsPerBlock);

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
    fprintf(stderr, "=== Kernel dispatched, checking cudaGetLastError... ===\n");
    CALL_CUDA(cudaGetLastError());
    fprintf(stderr, "=== cudaGetLastError OK, now synchronizing... ===\n");
    CALL_CUDA(cudaDeviceSynchronize()); // Synchronize to flush printf and check for errors
    fprintf(stderr, "=== cudaDeviceSynchronize OK ===\n");
    LOG_INFO("SpinTemperatureBox compute-and-reduce kernel called.");

    // Use thrust to reduce computed sums to one value.
    // Wrap device pointer in a thrust::device_ptr
    fprintf(stderr, "=== About to call thrust::reduce on buffer with %u elements ===\n", numBlocks);
    thrust::device_ptr<double> d_ave_sfrd_buf_ptr(d_ave_sfrd_buf);
    fprintf(stderr, "=== Created thrust device_ptr ===\n");
    // Reduce final buffer sums to one value
    double ave_sfrd_buf = thrust::reduce(d_ave_sfrd_buf_ptr, d_ave_sfrd_buf_ptr + numBlocks, 0., thrust::plus<double>());
    fprintf(stderr, "=== thrust::reduce completed, checking error... ===\n");
    CALL_CUDA(cudaGetLastError());
    fprintf(stderr, "=== thrust::reduce result=%f ===\n", ave_sfrd_buf);
    // CALL_CUDA(cudaDeviceSynchronize()); // Only use during development
    LOG_INFO("SFRD sum reduced to single value by thrust::reduce operation.");

    // Copy results from device to host
    fprintf(stderr, "=== About to cudaMemcpy %llu elements from device to host ===\n", num_pixels);
    CALL_CUDA(cudaMemcpy(sfrd_grid, d_sfrd_grid, sizeof(float) * num_pixels, cudaMemcpyDeviceToHost));
    fprintf(stderr, "=== cudaMemcpy completed ===\n");
    LOG_INFO("SFRD sum copied to host.");

    return ave_sfrd_buf;
}

extern "C" void free_sfrd_gpu_data(
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_ave_sfrd_buf
) {
    fprintf(stderr, "=== Entering free_sfrd_gpu_data ===\n");
    fprintf(stderr, "=== Pointers: d_y_arr=%p (*=%p), d_dens_R_grid=%p (*=%p), d_sfrd_grid=%p (*=%p), d_ave_sfrd_buf=%p (*=%p) ===\n",
            (void*)d_y_arr, (void*)*d_y_arr, (void*)d_dens_R_grid, (void*)*d_dens_R_grid,
            (void*)d_sfrd_grid, (void*)*d_sfrd_grid, (void*)d_ave_sfrd_buf, (void*)*d_ave_sfrd_buf);

    // Need to dereference the pointers to pointers (*)
    fprintf(stderr, "=== Freeing d_y_arr ===\n");
    CALL_CUDA(cudaFree(*d_y_arr));
    fprintf(stderr, "=== Freeing d_dens_R_grid ===\n");
    CALL_CUDA(cudaFree(*d_dens_R_grid));
    fprintf(stderr, "=== Freeing d_sfrd_grid ===\n");
    CALL_CUDA(cudaFree(*d_sfrd_grid));
    fprintf(stderr, "=== Freeing d_ave_sfrd_buf ===\n");
    CALL_CUDA(cudaFree(*d_ave_sfrd_buf));
    fprintf(stderr, "=== All device memory freed ===\n");
    LOG_INFO("Device memory freed.");
}
