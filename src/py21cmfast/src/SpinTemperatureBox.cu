// Most of the following includes likely can be removed.
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <complex.h>
#include <fftw3.h>

// we use thrust for reduction
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

#include "SpinTemperatureBox.h"


__device__ inline double EvaluateRGTable1D_f_gpu(double x, RGTable1D_f *table) {

    double x_min = table->x_min;
    double x_width = table->x_width;

    int idx = (int)floor((x - x_min) / x_width);

    double table_val = x_min + x_width * (float)idx;
    double interp_point = (x - table_val) / x_width;

    return table->y_arr[idx] * (1 - interp_point) + table->y_arr[idx + 1] * (interp_point);
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
    RGTable1D_f *SFRD_conditional_table, // input data
    float *dens_R_grid, // input data
    double zpp_growth_R_ct, // input value
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
        fcoll_i = exp(EvaluateRGTable1D_f_gpu(curr_dens_i, &SFRD_conditional_table));
        fcoll_j = exp(EvaluateRGTable1D_f_gpu(curr_dens_j, &SFRD_conditional_table));

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
    if (tid < 32) warp_reduce(sdata, tid);

    // The first thread of each block updates the block totals
    if (tid == 0) ave_sfrd_buf[blockIdx.x] = sdata[0];
}

void calculate_sfrd_from_grid_gpu(
    RGTable1D_f *SFRD_conditional_table, // input data
    float *dens_R_grid, // input data
    double *zpp_growth, // input data
    int R_ct, // input data
    float *sfrd_grid, // star formation rate density grid to be updated
    double *ave_sfrd_buf, // final output (to be divided by HII_TOT_NUM_PIXELS)
    unsigned int num_pixels // length of input data
) {
    // Input data
    double zpp_growth_R_ct = zpp_growth[R_ct];

    // The kernel only needs access to some fields of the SFRD_conditional_table struct
    // so we allocate device memory and copy data only for required fields.

    // Create device pointers
    double *x_min, *x_width, *y_arr;
    // Allocate device memory
    cudaMalloc(&x_min, sizeof(double));
    cudaMalloc(&x_width, sizeof(double));
    cudaMalloc(&y_arr, sizeof(double) * SFRD_conditional_table->n_bin);
    // Copy data from host to device
    cudaMemcpy(x_min, &SFRD_conditional_table->x_min, sizeof(double), cudaMemcpyHostToDevice); // Can also pass in
    cudaMemcpy(x_width, &SFRD_conditional_table->x_width, sizeof(double), cudaMemcpyHostToDevice); // Can also pass in
    cudaMemcpy(y_arr, SFRD_conditional_table->y_arr, sizeof(double) * SFRD_conditional_table->n_bin, cudaMemcpyHostToDevice);

    // Allocate & populate device memory for other inputs.

    // Create device pointers
    float *d_dens_R_grid, *d_sfrd_grid;
    // Allocate device memory
    cudaMalloc(&d_dens_R_grid, sizeof(float) * num_pixels);
    cudaMalloc(&d_sfrd_grid, sizeof(float) * num_pixels);
    // Copy data from host to device
    cudaMemcpy(d_dens_R_grid, &dens_R_grid, sizeof(float) * num_pixels, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sfrd_grid, &sfrd_grid, sizeof(float) * num_pixels, cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
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
    int smemSize = threadsPerBlock * sizeof(double);  // shared memory

    // Allocate device memory for output buffer and set to 0
    double* d_ave_sfrd_buf;
    unsigned int buffer_length = ceil(num_pixels / (threadsPerBlock * 2));
    cudaMalloc(&d_ave_sfrd_buf, sizeof(double) * buffer_length); // 91m & 256 -> 177979
    // cudaMalloc((void**)&d_ave_sfrd_buf, sizeof(double) * buffer_length);
    cudaMemset(d_ave_sfrd_buf, 0, sizeof(double) * buffer_length); // fill with byte=0

    // Invoke kernel
    switch (threadsPerBlock) {
        case 512:
            compute_and_reduce<512><<< numBlocks, threadsPerBlock, smemSize >>>(&x_min, &x_width, &y_arr, &d_dens_R_grid, zpp_growth_R_ct, &d_sfrd_grid, &d_ave_sfrd_buf, num_pixels);
            break;
        case 256:
            compute_and_reduce<256><<< numBlocks, threadsPerBlock, smemSize >>>(&x_min, &x_width, &y_arr, &d_dens_R_grid, zpp_growth_R_ct, &d_sfrd_grid, &d_ave_sfrd_buf, num_pixels);
            break;
        case 128:
            compute_and_reduce<128><<< numBlocks, threadsPerBlock, smemSize >>>(&x_min, &x_width, &y_arr, &d_dens_R_grid, zpp_growth_R_ct, &d_sfrd_grid, &d_ave_sfrd_buf, num_pixels);
            break;
        case 64:
            compute_and_reduce<64><<< numBlocks, threadsPerBlock, smemSize >>>(&x_min, &x_width, &y_arr, &d_dens_R_grid, zpp_growth_R_ct, &d_sfrd_grid, &d_ave_sfrd_buf, num_pixels);
            break;
        case 32:
            compute_and_reduce<32><<< numBlocks, threadsPerBlock, smemSize >>>(&x_min, &x_width, &y_arr, &d_dens_R_grid, zpp_growth_R_ct, &d_sfrd_grid, &d_ave_sfrd_buf, num_pixels);
            break;
        default:
            // LOG_WARNING("Thread size invalid; defaulting to 256.")
            compute_and_reduce<256><<< numBlocks, 256, 256 * sizeof(double) >>>(&x_min, &x_width, &y_arr, &d_dens_R_grid, zpp_growth_R_ct, &d_sfrd_grid, &d_ave_sfrd_buf, num_pixels);
    }

    // Only use during development!
    err = cudaDeviceSynchronize();
    CATCH_CUDA_ERROR(err);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("Kernel launch error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Use thrust to reduce computed buffer values to one value.

    // Wrap device pointer in a thrust::device_ptr
    thrust::device_ptr<double> d_ave_sfrd_buf_ptr(d_ave_sfrd_buf);
    // Reduce final buffer values to one value
    ave_sfrd_buf = thrust::reduce(d_ave_sfrd_buf_ptr, d_ave_sfrd_buf_ptr + buffer_length, 0., thrust::plus<double>());

    // Copy results from device to host.
    err = cudaMemcpy(sfrd_grid, d_sfrd_grid, sizeof(float) * num_pixels, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Deallocate device memory.
    cudaFree(x_min);
    cudaFree(x_width);
    cudaFree(y_arr);
    cudaFree(d_dens_R_grid);
    cudaFree(d_sfrd_grid);
    cudaFree(d_ave_sfrd_buf);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
}
