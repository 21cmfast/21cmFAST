#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"

// GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
// We use thrust for reduction
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/functional.h> // thrust::plus

#include "Constants.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cosmology.h"
#include "hmf.h"
#include "indexing.h"
#include "dft.h"
#include "recombinations.h"
#include "debugging.h"
#include "heating_helper_progs.h"
#include "photoncons.h"
#include "thermochem.h"
#include "interp_tables.h"
#include "filtering.h"
#include "bubble_helper_progs.h"
#include "InitialConditions.h"

#include "IonisationBox.h"
#include "cuda_utils.cuh"


__device__ inline double EvaluateRGTable1D_f_gpu(double x, double x_min, double x_width, float *y_arr) {

    int idx = (int)floor((x - x_min) / x_width);

    double table_val = x_min + x_width * (float)idx;
    double interp_point = (x - table_val) / x_width;

    return y_arr[idx] * (1 - interp_point) + y_arr[idx + 1] * (interp_point);
}

// template <unsigned int threadsPerBlock>
__global__ void compute_Fcoll(
    cuFloatComplex *deltax_filtered, // fg_struct
    cuFloatComplex *xe_filtered, // fg_struct
    float *y_arr, // Nion_conditional_table1D
    double x_min, // Nion_conditional_table1D
    double x_width, // Nion_conditional_table1D
    double fract_float_err, // FRACT_FLOAT_ERR
    bool use_ts_fluct, // flag_options_global->USE_TS_FLUCT
    unsigned long long hii_tot_num_pixels, // HII_TOT_NUM_PIXELS
    long long hii_d, // HII_D
    long long hii_d_para, // HII_D_PARA
    long long hii_mid_para, // HII_MID_PARA
    float *Fcoll // box
) {
    // Get index of grids
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bound check
    if (idx >= hii_tot_num_pixels) {
        return;
    }

    // Get x, y, z from idx using HII_R_INDEX macro formula
    int z = idx % hii_d_para;
    unsigned long long remaining = idx / hii_d_para;
    int y = remaining % hii_d;
    int x = remaining / hii_d;

    // Get FFT index using HII_R_FFT_INDEX macro formula
    unsigned long long fft_idx = z + 2 * (hii_mid_para + 1) * (y + hii_d * x);

    // These clippings could be made in the calling function, using thrust, rather than here...

    // Clip the filtered grids to physical values
    // delta cannot be less than -1
    *((float *) deltax_filtered + fft_idx) = fmaxf(*((float *) deltax_filtered + fft_idx), -1. + fract_float_err);
    // <N_rec> cannot be less than zero
    // x_e has to be between zero and unity
    if (use_ts_fluct) {
        *((float *) xe_filtered + fft_idx) = fmaxf(*((float *) xe_filtered + fft_idx), 0.0);
        *((float *) xe_filtered + fft_idx) = fminf(*((float *) xe_filtered + fft_idx), 0.999);
    }

    // Compute collapse fraction
    Fcoll[idx] = exp(EvaluateRGTable1D_f_gpu(*((float *) deltax_filtered + fft_idx), x_min, x_width, y_arr));
}

// Warp-shuffle reduction as per these sources
// slides: https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
// code: https://github.com/olcf/cuda-training-series/blob/master/exercises/hw5/reductions.cu
// video: https://vimeo.com/428453188
__global__ void reduce_fcoll_ws(float *gdata, const size_t num_pixels, double *out) {
    __shared__ double sdata[32];

    int tid = threadIdx.x; // thread within block
    int idx = threadIdx.x + blockDim.x * blockIdx.x; // data index
    double val = 0.0f; // partial sum for thread
    unsigned mask = 0xFFFFFFFFU; // all threads participate
    int lane = threadIdx.x % warpSize; // lane within warp [0-31]
    int warpID = threadIdx.x / warpSize; // warp number [0...]

    // Grid-stride loop
    while (idx < num_pixels) {
        val += gdata[idx];
        idx += gridDim.x * blockDim.x;
    }

    // Warp-shuffle
    // "offset >>=" is a bitwise right shift operation (essentially integer division by 2)
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    // Thread 0 of each warp writes to shared memory
    if (lane == 0) {
        sdata[warpID] = val;
    }
    __syncthreads();

    // Final reduction by warp 0
    if (warpID == 0) {
        val = (tid < blockDim.x / warpSize) ? sdata[lane] : 0;

        // Warp-shuffle
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }

        // Thread 0 of warp 0 accumuates final result
        if (tid == 0) {
            atomicAdd(out, val);
        }
    }
}

void init_ionbox_gpu_data(
    fftwf_complex **d_deltax_filtered, // copies of pointers to pointers
    fftwf_complex **d_xe_filtered,
    float **d_y_arr,
    float **d_Fcoll,
    unsigned int nbins, // nbins for Nion_conditional_table1D->y
    unsigned long long hii_tot_num_pixels, // HII_TOT_NUM_PIXELS
    unsigned long long hii_kspace_num_pixels, // HII_KSPACE_NUM_PIXELS
    unsigned int *threadsPerBlock,
    unsigned int *numBlocks
) {
    CALL_CUDA(cudaMalloc((void**)d_deltax_filtered, sizeof(fftwf_complex) * hii_kspace_num_pixels)); // already pointers to pointers (no & needed)
    CALL_CUDA(cudaMemset(*d_deltax_filtered, 0, sizeof(fftwf_complex) * hii_kspace_num_pixels)); // dereference the pointer to a pointer (*)

    if (flag_options_global->USE_TS_FLUCT) {
        CALL_CUDA(cudaMalloc((void**)d_xe_filtered, sizeof(fftwf_complex) * hii_kspace_num_pixels));
        CALL_CUDA(cudaMemset(*d_xe_filtered, 0, sizeof(fftwf_complex) * hii_kspace_num_pixels));
    }

    CALL_CUDA(cudaMalloc((void**)d_y_arr, sizeof(float) * nbins));
    CALL_CUDA(cudaMemset(*d_y_arr, 0, sizeof(float) * nbins));

    CALL_CUDA(cudaMalloc((void**)d_Fcoll, sizeof(float) * hii_tot_num_pixels));
    CALL_CUDA(cudaMemset(*d_Fcoll, 0, sizeof(float) * hii_tot_num_pixels));

    LOG_INFO("Ionisation grids allocated on device.");
    LOG_INFO("Ionisation grids initialised on device.");

    // Get max threads/block for device
    int maxThreadsPerBlock;
    CALL_CUDA(cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0));

    // Set threads/block based on device max
    if (maxThreadsPerBlock >= 512) {
        *threadsPerBlock = 512;
    } else if (maxThreadsPerBlock >= 256) {
        *threadsPerBlock = 256;
    } else if (maxThreadsPerBlock >= 128) {
        *threadsPerBlock = 128;
    } else if (maxThreadsPerBlock >= 64) {
        *threadsPerBlock = 64;
    } else if (maxThreadsPerBlock >= 32) {
        *threadsPerBlock = 32;
    } else {
        *threadsPerBlock = 16;
    }

    *numBlocks = (hii_tot_num_pixels + *threadsPerBlock - 1) / *threadsPerBlock;
}

void calculate_fcoll_grid_gpu(
    IonizedBox *box, // for box->Fcoll
    fftwf_complex *h_deltax_filtered, // members of fg_struct
    fftwf_complex *h_xe_filtered,
    double *f_coll_grid_mean, // member of rspec
    fftwf_complex *d_deltax_filtered, // device pointers
    fftwf_complex *d_xe_filtered,
    float *d_Fcoll,
    float *d_y_arr,
    unsigned long long hii_tot_num_pixels, // HII_TOT_NUM_PIXELS
    unsigned long long hii_kspace_num_pixels, // HII_KSPACE_NUM_PIXELS
    unsigned int *threadsPerBlock,
    unsigned int *numBlocks
) {
    RGTable1D_f* Nion_conditional_table1D = get_Nion_conditional_table1D();

    // Copy grids from host to device
    CALL_CUDA(cudaMemcpy(d_deltax_filtered, h_deltax_filtered, sizeof(fftwf_complex) * hii_kspace_num_pixels, cudaMemcpyHostToDevice));
    if (flag_options_global->USE_TS_FLUCT) {
        CALL_CUDA(cudaMemcpy(d_xe_filtered, h_xe_filtered, sizeof(fftwf_complex) * hii_kspace_num_pixels, cudaMemcpyHostToDevice));
    }
    CALL_CUDA(cudaMemcpy(d_y_arr, Nion_conditional_table1D->y_arr, sizeof(float) * Nion_conditional_table1D->n_bin, cudaMemcpyHostToDevice));
    LOG_INFO("Ionisation grids copied to device.");

    // TODO: Can I pass these straight to kernel? (or access in kernel w/ Tiger's method)
    double fract_float_err = FRACT_FLOAT_ERR;
    bool use_ts_fluct = flag_options_global->USE_TS_FLUCT;
    long long hii_d = HII_D;
    long long hii_d_para = HII_D_PARA;
    long long hii_mid_para = HII_MID_PARA;

    // Invoke kernel
    compute_Fcoll<<< *numBlocks, *threadsPerBlock >>>(
        reinterpret_cast<cuFloatComplex *>(d_deltax_filtered),
        reinterpret_cast<cuFloatComplex *>(d_xe_filtered),
        d_y_arr,
        Nion_conditional_table1D->x_min,
        Nion_conditional_table1D->x_width,
        fract_float_err,
        use_ts_fluct,
        hii_tot_num_pixels,
        hii_d,
        hii_d_para,
        hii_mid_para,
        d_Fcoll
    );
    CALL_CUDA(cudaGetLastError());
    // CALL_CUDA(cudaDeviceSynchronize()); // Only use during development
    LOG_INFO("IonisationBox compute_Fcoll kernel called.");

    // // Use thrust to reduce computed sums to one value.
    // // Wrap device pointer in a thrust::device_ptr
    // thrust::device_ptr<float> d_Fcoll_ptr(d_Fcoll);
    // // Reduce final buffer sums to one value
    // double f_coll_grid_total = thrust::reduce(d_Fcoll_ptr, d_Fcoll_ptr + hii_tot_num_pixels, 0., thrust::plus<float>());
    // CALL_CUDA(cudaGetLastError());
    // // CALL_CUDA(cudaDeviceSynchronize()); // Only use during development
    // *f_coll_grid_mean = f_coll_grid_total / (double) hii_tot_num_pixels;
    // LOG_INFO("Fcoll sum reduced to single value by thrust::reduce operation.");

    // These are better off allocated/freed outside loop
    double *d_fcoll_sum;
    // double *h_fcoll_sum = (double*)malloc(sizeof(double));
    double *h_fcoll_sum;

    CALL_CUDA(cudaMalloc(&d_fcoll_sum, sizeof(double)));  // Allocate device space for sum
    CALL_CUDA(cudaMemset(d_fcoll_sum, 0, sizeof(double)));
    reduce_fcoll_ws<<< *numBlocks, *threadsPerBlock >>>(d_Fcoll, hii_tot_num_pixels, d_fcoll_sum);
    // CALL_CUDA(cudaDeviceSynchronize()); // Development only
    CALL_CUDA(cudaGetLastError());
    CALL_CUDA(cudaMemcpy(h_fcoll_sum, d_fcoll_sum, sizeof(double), cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaFree(d_fcoll_sum));
    *f_coll_grid_mean = *h_fcoll_sum / (double) hii_tot_num_pixels;
    // free(h_fcoll_sum);
    LOG_INFO("Fcoll grid mean computed by warp shuffle operation.");

    // Copy results from device to host
    CALL_CUDA(cudaMemcpy(box->Fcoll, d_Fcoll, sizeof(float) * hii_tot_num_pixels, cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(h_deltax_filtered, d_deltax_filtered, sizeof(fftwf_complex) * hii_kspace_num_pixels, cudaMemcpyDeviceToHost));
    if (flag_options_global->USE_TS_FLUCT) {
        CALL_CUDA(cudaMemcpy(h_xe_filtered, d_xe_filtered, sizeof(fftwf_complex) * hii_kspace_num_pixels, cudaMemcpyDeviceToHost));
    }
    LOG_INFO("Grids copied to host.");
}

void free_ionbox_gpu_data(
    fftwf_complex **d_deltax_filtered, // copies of pointers to pointers
    fftwf_complex **d_xe_filtered,
    float **d_y_arr,
    float **d_Fcoll
) {
    CALL_CUDA(cudaFree(*d_deltax_filtered)); // Need to dereference the pointers to pointers (*)
    if (flag_options_global->USE_TS_FLUCT) {
        CALL_CUDA(cudaFree(*d_xe_filtered));
    }
    CALL_CUDA(cudaFree(*d_y_arr));
    CALL_CUDA(cudaFree(*d_Fcoll));
    LOG_INFO("Device memory freed.");
}
