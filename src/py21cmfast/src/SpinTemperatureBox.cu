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
    if (tid == 0) { ave_sfrd_buf[blockIdx.x] = sdata[0]; }
}

unsigned int init_sfrd_gpu_data(
    float *dens_R_grid, // input data
    float *sfrd_grid, // star formation rate density grid to be updated
    unsigned long long num_pixels, // length of input data
    unsigned int nbins, // nbins for sfrd_grid->y
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_ave_sfrd_buf
) {
    cudaError_t err = cudaGetLastError();

    // Allocate device memory ------------------------------------------------------------------------------------------
    err = cudaMalloc((void**)d_y_arr, sizeof(float) * nbins); // already pointers to pointers (no & needed)
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    err = cudaMalloc((void**)d_dens_R_grid, sizeof(float) * num_pixels);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    err = cudaMalloc((void**)d_sfrd_grid, sizeof(float) * num_pixels);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    LOG_INFO("SFRD_conditional_table.y_arr and density and sfrd grids allocated on device.");

    // Initialise sfrd_grid to 0 (fill with byte=0) ----------------------------------------------------------------------
    err = cudaMemset(*d_sfrd_grid, 0, sizeof(float) * num_pixels); // dereference the pointer to a pointer (*)
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s: %p", cudaGetErrorString(err), d_sfrd_grid);
        Throw(CUDAError);
    }
    LOG_INFO("sfrd grid initialised to 0.");

    // Get max threads/block for device
    int maxThreadsPerBlock;
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);

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
    err = cudaMalloc((void**)d_ave_sfrd_buf, sizeof(double) * numBlocks); // already pointer to a pointer (no & needed) ...91m & 256 -> 177979
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    LOG_INFO("SFRD sum reduction buffer allocated on device.");

    // Initialise buffer to 0 (fill with byte=0)
    err = cudaMemset(*d_ave_sfrd_buf, 0, sizeof(double) * numBlocks); // dereference the pointer to a pointer (*)
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    LOG_INFO("SFRD sum reduction buffer initialised to 0.");

    return threadsPerBlock;
}

double calculate_sfrd_from_grid_gpu(
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
    double *d_ave_sfrd_buf
) {
    cudaError_t err = cudaGetLastError();

    // Get growth factor for current filter step
    double zpp_growth_R_ct = zpp_growth[R_ct];

    // Copy data from host to device -----------------------------------------------------------------------------------
    err = cudaMemcpy(d_y_arr, SFRD_conditional_table->y_arr, sizeof(float) * SFRD_conditional_table->n_bin, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    err = cudaMemcpy(d_dens_R_grid, dens_R_grid, sizeof(float) * num_pixels, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    LOG_INFO("SFRD_conditional_table.y_arr and density grid copied to device.");

    unsigned int numBlocks = ceil(num_pixels / (threadsPerBlock * 2));
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
    double ave_sfrd_buf = thrust::reduce(d_ave_sfrd_buf_ptr, d_ave_sfrd_buf_ptr + numBlocks, 0., thrust::plus<double>());
    LOG_INFO("SFRD sum reduced to single value by thrust::reduce operation.");

    // Copy results from device to host
    err = cudaMemcpy(sfrd_grid, d_sfrd_grid, sizeof(float) * num_pixels, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    LOG_INFO("SFRD sum copied to host.");

    return ave_sfrd_buf;
}

void free_sfrd_gpu_data(
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_ave_sfrd_buf
) {
    cudaError_t err = cudaGetLastError();

    // Need to dereference the pointers to pointers (*)
    err = cudaFree(*d_y_arr);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    err = cudaFree(*d_dens_R_grid);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    err = cudaFree(*d_sfrd_grid);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
    err = cudaFree(*d_ave_sfrd_buf);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    LOG_INFO("Device memory freed.");
}
