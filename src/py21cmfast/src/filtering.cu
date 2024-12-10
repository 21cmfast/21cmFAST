#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <fftw3.h>

// GPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
// #include <cufft.h>
// #include <cufftw.h>

#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"

#include "Constants.h"
#include "InputParameters.h"
#include "indexing.h"
#include "dft.h"
#include "filtering.h"

__device__ inline double real_tophat_filter(double kR) {
    // Second order taylor expansion around kR==0
    if (kR < 1e-4)
        return 1 - kR*kR/10;
    return 3.0*pow(kR, -3) * (sin(kR) - cos(kR)*kR);
}

__device__ inline double sharp_k_filter(double kR) {
    if (kR * 0.413566994 > 1)
       return 0.;
    return 1;
}

__device__ inline double gaussian_filter(double kR_squared) {
    return exp(-0.643 * 0.643 * kR_squared / 2.);
}

__device__ inline double exp_mfp_filter(double k, double R, double mfp, double exp_term) {
    double f;
    double kR = k * R;
    double ratio = mfp / R;

    // Second order taylor expansion around kR==0
    if (kR < 1e-4) {
        double ts_0 = 6 * pow(ratio, 3) - exp_term * (6 * pow(ratio, 3) + 6 * pow(ratio, 2) + 3 * ratio);
        return ts_0 + (exp_term * (2 * pow(ratio, 2) + 0.5 * ratio) - 2 * ts_0 * pow(ratio, 2)) * kR * kR;
    }
    // Davies & Furlanetto MFP-eps(r) window function
    f = (kR * kR * pow(ratio, 2) + 2 * ratio + 1) * ratio * cos(kR);
    f += (kR * kR * (pow(ratio, 2) - pow(ratio, 3)) + ratio + 1) * sin(kR) / kR;
    f *= exp_term;
    f -= 2 * pow(ratio, 2);
    f *= -3 * ratio/pow(pow(kR * ratio, 2) + 1, 2);
    return f;
}

__device__ inline double spherical_shell_filter(double k, double R_outer, double R_inner) {
    double kR_inner = k * R_inner;
    double kR_outer = k * R_outer;

    // Second order taylor expansion around kR_outer==0
    if (kR_outer < 1e-4)
        return 1. - kR_outer*kR_outer / 10 * \
                (pow(R_inner / R_outer, 5) - 1) / \
                (pow(R_inner / R_outer, 3) - 1);

    return 3.0 / (pow(kR_outer, 3) - pow(kR_inner, 3)) \
        * (sin(kR_outer) - cos(kR_outer) * kR_outer \
        -  sin(kR_inner) + cos(kR_inner) * kR_inner);
}

__global__ void filter_box_kernel(cuFloatComplex *box, int num_pixels, int dimension, int midpoint, int midpoint_para, double delta_k, float R, float R_param, double R_const, int filter_type) {

    // Get index of box (flattened k-box)
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Bound check (in case number of threads != multiple of block size)
    if (idx >= num_pixels) {
        return;
    }
    // Compute the 3D indices (n_x, n_y, n_z) for the k-box from the flattened index (idx)
    // Based on convenience macros in indexing.h
    int n_z = idx % (midpoint_para + 1);
    unsigned long long remaining = idx / (midpoint_para + 1);
    int n_y = remaining % dimension;
    int n_x = remaining / dimension;

    // Compute wave vector components
    float k_x = (n_x - dimension * (n_x > midpoint)) * delta_k; // Wrap around midpoint
    float k_y = (n_y - dimension * (n_y > midpoint)) * delta_k;
    float k_z = n_z * delta_k;

    // TODO: Try alternative vectorised coords & wave vector components?
    // int *cell_coords = (int[]) {idx % (midpoint_para + 1), (idx / (midpoint_para + 1)) % dimension, (idx / (midpoint_para + 1)) / dimension)};    // (as above and * delta_k to vector at end)
    // int *wave_vector = (float[]) { ... }

    // Compute squared magnitude of wave vector
    float k_mag_sq = k_x*k_x + k_y*k_y + k_z*k_z;

    float kR;
    if (filter_type == 0) { // real space top-hat
        kR = sqrt(k_mag_sq) * R;
        // box[idx] *= real_tophat_filter(kR);
        box[idx] = cuCmulf(box[idx], make_cuFloatComplex((float)real_tophat_filter(kR), 0.f));
    }
    else if (filter_type == 1) { // k-space top hat
        kR = sqrt(k_mag_sq) * R;
        // box[idx] *= sharp_k_filter(kR);
        box[idx] = cuCmulf(box[idx], make_cuFloatComplex((float)sharp_k_filter(kR), 0.f));
    }
    else if (filter_type == 2) { // gaussian
        kR = k_mag_sq * R * R;
        // box[idx] *= gaussian_filter(kR);
        box[idx] = cuCmulf(box[idx], make_cuFloatComplex((float)gaussian_filter(kR), 0.f));
    }
    else if (filter_type == 3) { // exponentially decaying tophat
        // box[idx] *= exp_mfp_filter(sqrt(k_mag_sq), R, R_param, R_const);
        box[idx] = cuCmulf(box[idx], make_cuFloatComplex((float)exp_mfp_filter(sqrt(k_mag_sq), R, R_param, R_const), 0.f));
    }
    else if (filter_type == 4) { //spherical shell
        // box[idx] *= spherical_shell_filter(sqrt(k_mag_sq), R, R_param);
        box[idx] = cuCmulf(box[idx], make_cuFloatComplex((float)spherical_shell_filter(sqrt(k_mag_sq), R, R_param), 0.f));
    }
}

void filter_box_gpu(fftwf_complex *box, int RES, int filter_type, float R, float R_param) {

    // Check for valid filter type
    if (filter_type < 0 || filter_type > 4) {
        LOG_WARNING("Filter type %i is undefined. Box is unfiltered.", filter_type);
        return;
    }

    // Get required values
    int dimension, midpoint, midpoint_para, num_pixels;
    switch(RES) {
        case 0:
            dimension = user_params_global->DIM;
            midpoint = MIDDLE;  // midpoint of x,y = DIM / 2
            midpoint_para = MID_PARA;  // midpoint of z = NON_CUBIC_FACTOR * HII_DIM / 2
            num_pixels = KSPACE_NUM_PIXELS;
            break;
        case 1:
            dimension = user_params_global->HII_DIM;
            midpoint = HII_MIDDLE;  // midpoint of x,y = HII_DIM / 2
            midpoint_para = HII_MID_PARA;  // midpoint of z = NON_CUBIC_FACTOR * HII_DIM / 2
            num_pixels = HII_KSPACE_NUM_PIXELS;
            break;
        default:
            LOG_ERROR("Resolution for filter functions must be 0(DIM) or 1(HII_DIM)");
            Throw(ValueError);
            break;
    }
    double delta_k = DELTA_K;
    double R_const;
    if (filter_type == 3) {
        R_const = exp(-R / R_param);
    }

    // Get size of flattened array
    size_t size = num_pixels * sizeof(fftwf_complex);

    cudaError_t err;

    // Allocate device memory
    fftwf_complex* d_box;
    err = cudaMalloc(&d_box, size);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Copy array from host to device
    err = cudaMemcpy(d_box, box, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Invoke kernel
    int threadsPerBlock = 256;
    int numBlocks = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;
    // d_box must be cast to cuFloatComplex (from fftwf_complex) for CUDA
    filter_box_kernel<<<numBlocks, threadsPerBlock>>>(reinterpret_cast<cuFloatComplex *>(d_box), num_pixels, dimension, midpoint, midpoint_para, delta_k, R, R_param, R_const, filter_type);

    // // Only use during development!
    err = cudaDeviceSynchronize();
    CATCH_CUDA_ERROR(err);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("Kernel launch error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Copy results from device to host
    err = cudaMemcpy(box, d_box, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Deallocate device memory
    err = cudaFree(d_box);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
}

// Test function to filter a box without computing a whole output box
int test_filter_gpu(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options
                    , float *input_box, double R, double R_param, int filter_flag, double *result) {
    int i,j,k;
    unsigned long long int ii;

    Broadcast_struct_global_all(user_params, cosmo_params, astro_params, flag_options);

    //setup the box
    fftwf_complex *box_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *box_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    for (i=0; i<user_params->HII_DIM; i++)
        for (j=0; j<user_params->HII_DIM; j++)
            for (k=0; k<HII_D_PARA; k++)
                *((float *)box_unfiltered + HII_R_FFT_INDEX(i,j,k)) = input_box[HII_R_INDEX(i,j,k)];

    dft_r2c_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, box_unfiltered);

    // Convert to CUDA complex type
    cuFloatComplex* box_unfiltered_cu = reinterpret_cast<cuFloatComplex*>(box_unfiltered);

    for(ii=0;ii<HII_KSPACE_NUM_PIXELS;ii++){
        // box_unfiltered[ii] /= (double)HII_TOT_NUM_PIXELS;
        box_unfiltered_cu[ii] = cuCdivf(box_unfiltered_cu[ii], make_cuFloatComplex((float)HII_TOT_NUM_PIXELS, 0.f));
    }

    memcpy(box_filtered, box_unfiltered, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

    filter_box_gpu(box_filtered, 1, filter_flag, R, R_param);

    dft_c2r_cube(user_params->USE_FFTW_WISDOM, user_params->HII_DIM, HII_D_PARA, user_params->N_THREADS, box_filtered);

    for (i=0; i<user_params->HII_DIM; i++)
        for (j=0; j<user_params->HII_DIM; j++)
            for (k=0; k<HII_D_PARA; k++)
                    result[HII_R_INDEX(i,j,k)] = *((float *)box_filtered + HII_R_FFT_INDEX(i,j,k));

    fftwf_free(box_unfiltered);
    fftwf_free(box_filtered);

    return 0;
}
