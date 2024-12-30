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
#include "cuda_utils.cuh"


__device__ inline double EvaluateRGTable1D_f_gpu(double x, double x_min, double x_width, float *y_arr) {

    int idx = (int)floor((x - x_min) / x_width);

    double table_val = x_min + x_width * (float)idx;
    double interp_point = (x - table_val) / x_width;

    return y_arr[idx] * (1 - interp_point) + y_arr[idx + 1] * (interp_point);
}

__global__ void compute_sfrd_kernel(
    double x_min, // reference
    double x_width, // reference
    float *y_arr, // reference
    float *dens_R_grid, // reference
    double zpp_growth_R_ct, // value
    float *sfrd_grid, // star formation rate density grid to be updated
    double *fcoll_tmp, // temp buffer for later summation
    unsigned long long num_pixels // length of input data
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    double curr_dens;
    double fcoll;

   // Bound check (in case number of threads != multiple of block size)
    if (idx >= num_pixels) {
        return;
    }
    // Compute current density from density grid value * redshift-scaled growth factor
    curr_dens = dens_R_grid[idx] * zpp_growth_R_ct;

    // Compute fraction of mass that has collapsed to form stars/other structures
    fcoll = exp(EvaluateRGTable1D_f_gpu(curr_dens, x_min, x_width, y_arr));

    // Update the fcoll temp array
    fcoll_tmp[idx] = fcoll;

    // Update the relevant cells in the star formation rate density grid
    sfrd_grid[idx] = (1. + curr_dens) * fcoll;
}

void init_sfrd_gpu_data_simple(
    float *dens_R_grid, // input data
    float *sfrd_grid, // star formation rate density grid to be updated
    unsigned long long num_pixels, // length of input data
    unsigned int nbins, // nbins for sfrd_grid->y
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_fcoll_tmp
) {
    // Allocate device memory
    CALL_CUDA(cudaMalloc(d_y_arr, sizeof(float) * nbins)); // already pointers to pointers (no & needed)
    CALL_CUDA(cudaMalloc(d_dens_R_grid, sizeof(float) * num_pixels)); // TODO: check removal of (void**) since already pointers to pointers, so cast is redundant
    CALL_CUDA(cudaMalloc(d_sfrd_grid, sizeof(float) * num_pixels));
    CALL_CUDA(cudaMalloc(d_fcoll_tmp, sizeof(double) * num_pixels));
    LOG_INFO("SFRD_conditional_table.y_arr, density and sfrd grids, and fcoll temp array allocated on device.");

    // Initialise sfrd_grid to 0 (fill with byte=0)
    CALL_CUDA(cudaMemset(*d_sfrd_grid, 0, sizeof(float) * num_pixels)); // dereference the pointer to a pointer (*)
    CALL_CUDA(cudaMemset(*d_fcoll_tmp, 0, sizeof(double) * num_pixels));
    LOG_INFO("sfrd grid and fcoll temp array initialised to 0.");
}

double calculate_sfrd_gpu_simple(
    RGTable1D_f *SFRD_conditional_table, // input data
    float *dens_R_grid, // input data
    double *zpp_growth, // input data
    int R_ct, // filter step/loop iteration/spherical annuli (out of 40 by default)
    float *sfrd_grid, // star formation rate density grid to be updated
    unsigned long long num_pixels, // length of input data
    float *d_y_arr,
    float *d_dens_R_grid,
    float *d_sfrd_grid,
    double *d_fcoll_tmp
) {
    // Get growth factor for current filter step
    double zpp_growth_R_ct = zpp_growth[R_ct];

    // Copy data from host to device
    CALL_CUDA(cudaMemcpy(d_y_arr, SFRD_conditional_table->y_arr, sizeof(float) * SFRD_conditional_table->n_bin, cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMemcpy(d_dens_R_grid, dens_R_grid, sizeof(float) * num_pixels, cudaMemcpyHostToDevice));
    LOG_INFO("SFRD_conditional_table.y_arr and density grid copied to device.");

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
    unsigned int numBlocks = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

    compute_sfrd_kernel<<< numBlocks, threadsPerBlock >>>(SFRD_conditional_table->x_min, SFRD_conditional_table->x_width, d_y_arr, d_dens_R_grid, zpp_growth_R_ct, d_sfrd_grid, d_fcoll_tmp, num_pixels);
    // CALL_CUDA(cudaDeviceSynchronize());
    CALL_CUDA(cudaGetLastError());
    LOG_INFO("SpinTemperatureBox compute_sfrd_kernel called.");

    // Copy results from device to host
    CALL_CUDA(cudaMemcpy(sfrd_grid, d_sfrd_grid, sizeof(float) * num_pixels, cudaMemcpyDeviceToHost));
    LOG_INFO("SFRD sum copied to host.");

    // Use thrust to sum temp fcoll array for ave_sfrd
    // Wrap device pointer in a thrust::device_ptr
    thrust::device_ptr<double> d_fcoll_tmp_ptr(d_fcoll_tmp);
    // Sum temp fcoll array
    double sum_sfrd = thrust::reduce(d_fcoll_tmp_ptr, d_fcoll_tmp_ptr + num_pixels, 0., thrust::plus<double>());
    CALL_CUDA(cudaGetLastError());
    LOG_INFO("SFRD sum reduced to single value by thrust::reduce operation.");

    return sum_sfrd;
}

void free_sfrd_gpu_data_simple(
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_fcoll_tmp
) {
    // Need to dereference the pointers to pointers (*)
    CALL_CUDA(cudaFree(*d_y_arr));
    CALL_CUDA(cudaFree(*d_dens_R_grid));
    CALL_CUDA(cudaFree(*d_sfrd_grid));
    CALL_CUDA(cudaFree(*d_fcoll_tmp));
    LOG_INFO("Device memory freed.");
}
