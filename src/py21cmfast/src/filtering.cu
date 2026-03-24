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
#include <cufft.h>

#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"

#include "Constants.h"
#include "InputParameters.h"
#include "indexing.h"
#include "indexing.cuh"
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

void filter_box_gpu(fftwf_complex *box, int box_dim[3], int filter_type, float R, float R_param,
                    float R_star) {

    // Upstream release-v4.2 introduced filter_type==5 (multiple-scattering window function)
    // with a new R_star parameter. The multi-scattering implementation uses GSL special
    // functions (gsl_sf_gammainv) and a hypergeometric-2F3 evaluator that are not yet
    // available on the GPU. When filter_type==5 is requested on the GPU dispatch path we
    // transparently delegate to the CPU implementation (filter_box_cpu). This is
    // documented in UPSTREAM.md / PLAN.md 11.10; it keeps the upstream physics correct
    // without requiring us to port multi-scattering to CUDA for this merge. R_star is
    // otherwise unused on the GPU path.
    if (filter_type == 5) {
        filter_box_cpu(box, box_dim, filter_type, R, R_param, R_star);
        return;
    }

    // Check for valid filter type
    if (filter_type < 0 || filter_type > 4) {
        LOG_WARNING("Filter type %i is undefined. Box is unfiltered.", filter_type);
        return;
    }

    // Derive required values from box_dim
    int dimension = box_dim[0];  // Assumes cubic in x,y
    int midpoint = dimension / 2;
    int midpoint_para = box_dim[2] / 2;
    int num_pixels = (unsigned long long)box_dim[0] * box_dim[1] * (box_dim[2] / 2 + 1);
    double delta_k = 2.0 * M_PI / simulation_options_global->BOX_LEN;
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

/* Device-to-device memcpy wrapper (callable from C code without CUDA headers) */
void device_memcpy(void *dst, void *src, unsigned long long size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) { LOG_ERROR("device_memcpy: %s", cudaGetErrorString(err)); Throw(CUDAError); }
}

/* Forward declaration */
void filter_box_gpu_inplace(void *d_box_v, int box_dim[3], int filter_type,
                            float R, float R_param);

/*
 * filter_and_transform_gpu — Filter k-space box and inverse-transform on GPU.
 *
 * Replaces the filter_box() + dft_c2r_cube() pair with a single GPU operation.
 * Input: host k-space complex array (unfiltered copy, will be modified).
 * Output: host real-space array (in-place, same buffer, padded FFTW layout).
 *
 * Uses a shared cuFFT plan — caller should pass a plan created with
 * create_cufft_c2r_plan(). If plan is 0, creates and destroys one internally.
 */
void filter_and_transform_gpu(fftwf_complex *box, int box_dim[3], int filter_type,
                              float R, float R_param, cufftHandle plan) {
    int dimension = box_dim[0];
    int midpoint_para = box_dim[2] / 2;
    int kspace_pixels = (unsigned long long)dimension * dimension * (midpoint_para + 1);
    int padded_dim_para = 2 * (midpoint_para + 1);
    size_t kspace_size = kspace_pixels * sizeof(cuFloatComplex);
    size_t real_padded_size = (size_t)dimension * dimension * padded_dim_para * sizeof(float);
    size_t alloc_size = real_padded_size > kspace_size ? real_padded_size : kspace_size;
    bool own_plan = false;

    cudaError_t err;
    cuFloatComplex *d_buf;

    err = cudaMalloc(&d_buf, alloc_size);
    if (err != cudaSuccess) { LOG_ERROR("cudaMalloc: %s", cudaGetErrorString(err)); Throw(CUDAError); }

    /* Upload k-space data */
    err = cudaMemcpy(d_buf, box, kspace_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_buf); LOG_ERROR("cudaMemcpy H2D: %s", cudaGetErrorString(err)); Throw(CUDAError); }

    /* Filter on device */
    filter_box_gpu_inplace(d_buf, box_dim, filter_type, R, R_param);

    /* cuFFT c2r inverse transform */
    if (plan == 0) {
        int n_fft[3] = {dimension, dimension, box_dim[2]};
        cufftResult cr = cufftPlanMany(&plan, 3, n_fft, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, 1);
        if (cr != CUFFT_SUCCESS) { cudaFree(d_buf); LOG_ERROR("cufftPlanMany: %d", cr); Throw(CUDAError); }
        own_plan = true;
    }

    cufftResult cr = cufftExecC2R(plan, (cufftComplex *)d_buf, (cufftReal *)d_buf);
    if (cr != CUFFT_SUCCESS) { cudaFree(d_buf); if (own_plan) cufftDestroy(plan); LOG_ERROR("cufftExecC2R: %d", cr); Throw(CUDAError); }
    cudaDeviceSynchronize();

    /* Download real-space result (padded layout, same as FFTW c2r output) */
    err = cudaMemcpy(box, d_buf, real_padded_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { cudaFree(d_buf); if (own_plan) cufftDestroy(plan); LOG_ERROR("cudaMemcpy D2H: %s", cudaGetErrorString(err)); Throw(CUDAError); }

    cudaFree(d_buf);
    if (own_plan) cufftDestroy(plan);
}

/*
 * create_cufft_c2r_plan — Create a reusable cuFFT c2r plan.
 * Call once before a loop, pass to filter_and_transform_gpu, destroy after.
 */
cufftHandle create_cufft_c2r_plan(int box_dim[3]) {
    cufftHandle plan;
    int n_fft[3] = {box_dim[0], box_dim[1], box_dim[2]};
    cufftResult cr = cufftPlanMany(&plan, 3, n_fft, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, 1);
    if (cr != CUFFT_SUCCESS) { LOG_ERROR("cufftPlanMany: %d", cr); Throw(CUDAError); }
    return plan;
}

void destroy_cufft_plan(cufftHandle plan) {
    cufftDestroy(plan);
}

/*
 * filter_and_transform_device — Filter + cuFFT c2r using persistent device buffers.
 *
 * d_source:  device pointer to unfiltered k-space data (not modified)
 * d_working: device pointer to working buffer (overwritten)
 * h_output:  host pointer to receive real-space result (padded FFTW layout)
 * plan:      pre-created cuFFT c2r plan
 *
 * Does: D2D copy source → working, filter kernel, cuFFT c2r, D2H result.
 * No cudaMalloc/cudaFree per call.
 */
void filter_and_transform_device(void *d_source_v, void *d_working_v,
                                 fftwf_complex *h_output, int box_dim[3],
                                 int filter_type, float R, float R_param,
                                 int plan_int) {
    cuFloatComplex *d_source = (cuFloatComplex *)d_source_v;
    cuFloatComplex *d_working = (cuFloatComplex *)d_working_v;
    cufftHandle plan = (cufftHandle)plan_int;

    int dimension = box_dim[0];
    int midpoint_para = box_dim[2] / 2;
    int kspace_pixels = (unsigned long long)dimension * dimension * (midpoint_para + 1);
    int padded_dim_para = 2 * (midpoint_para + 1);
    size_t kspace_size = kspace_pixels * sizeof(cuFloatComplex);
    size_t real_padded_size = (size_t)dimension * dimension * padded_dim_para * sizeof(float);

    cudaError_t err;

    /* D2D copy: unfiltered source → working buffer */
    err = cudaMemcpy(d_working, d_source, kspace_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) { LOG_ERROR("D2D: %s", cudaGetErrorString(err)); Throw(CUDAError); }

    /* Filter on device */
    filter_box_gpu_inplace(d_working, box_dim, filter_type, R, R_param);

    /* cuFFT c2r inverse transform */
    cufftResult cr = cufftExecC2R(plan, (cufftComplex *)d_working, (cufftReal *)d_working);
    if (cr != CUFFT_SUCCESS) { LOG_ERROR("cufftExecC2R: %d", cr); Throw(CUDAError); }
    cudaDeviceSynchronize();

    /* D2H: download real-space result */
    err = cudaMemcpy(h_output, d_working, real_padded_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { LOG_ERROR("D2H: %s", cudaGetErrorString(err)); Throw(CUDAError); }
}

/*
 * Device filter buffer management for IonisationBox R loop.
 * Uploads unfiltered k-space fields once, provides working buffer and cuFFT plan.
 */

void init_device_filter_buffers(struct DeviceFilterBuffers *bufs, int box_dim[3],
                                fftwf_complex **h_unfiltered_fields, int n_fields) {
    int dimension = box_dim[0];
    int midpoint_para = box_dim[2] / 2;
    int kspace_pixels = (unsigned long long)dimension * dimension * (midpoint_para + 1);
    int padded_dim_para = 2 * (midpoint_para + 1);
    size_t kspace_size = kspace_pixels * sizeof(cuFloatComplex);
    size_t real_padded_size = (size_t)dimension * dimension * padded_dim_para * sizeof(float);
    size_t alloc_size = real_padded_size > kspace_size ? real_padded_size : kspace_size;

    cudaError_t err;

    bufs->n_fields = n_fields;
    bufs->kspace_size = kspace_size;
    bufs->real_padded_size = real_padded_size;
    bufs->box_dim[0] = box_dim[0];
    bufs->box_dim[1] = box_dim[1];
    bufs->box_dim[2] = box_dim[2];
    bufs->d_deltax_real = NULL;
    bufs->d_xe_real = NULL;

    /* Allocate and upload each unfiltered field */
    for (int i = 0; i < n_fields; i++) {
        err = cudaMalloc(&bufs->d_fields[i], kspace_size);
        if (err != cudaSuccess) { LOG_ERROR("cudaMalloc field %d: %s", i, cudaGetErrorString(err)); Throw(CUDAError); }
        err = cudaMemcpy(bufs->d_fields[i], h_unfiltered_fields[i], kspace_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { LOG_ERROR("H2D field %d: %s", i, cudaGetErrorString(err)); Throw(CUDAError); }
    }

    /* Allocate working buffer (large enough for real-space padded output) */
    err = cudaMalloc(&bufs->d_working, alloc_size);
    if (err != cudaSuccess) { LOG_ERROR("cudaMalloc working: %s", cudaGetErrorString(err)); Throw(CUDAError); }

    /* Allocate persistent buffers for filtered real-space grids (Fcoll GPU path) */
    err = cudaMalloc(&bufs->d_deltax_real, alloc_size);
    if (err != cudaSuccess) { LOG_ERROR("cudaMalloc d_deltax_real: %s", cudaGetErrorString(err)); Throw(CUDAError); }
    if (astro_options_global->USE_TS_FLUCT) {
        err = cudaMalloc(&bufs->d_xe_real, alloc_size);
        if (err != cudaSuccess) { LOG_ERROR("cudaMalloc d_xe_real: %s", cudaGetErrorString(err)); Throw(CUDAError); }
    }

    /* Create cuFFT plan */
    int n_fft[3] = {dimension, dimension, box_dim[2]};
    cufftResult cr = cufftPlanMany(&bufs->plan, 3, n_fft, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, 1);
    if (cr != CUFFT_SUCCESS) { LOG_ERROR("cufftPlanMany: %d", cr); Throw(CUDAError); }
}

void free_device_filter_buffers(struct DeviceFilterBuffers *bufs) {
    for (int i = 0; i < bufs->n_fields; i++) {
        if (bufs->d_fields[i]) { cudaFree(bufs->d_fields[i]); bufs->d_fields[i] = NULL; }
    }
    if (bufs->d_working) { cudaFree(bufs->d_working); bufs->d_working = NULL; }
    if (bufs->d_deltax_real) { cudaFree(bufs->d_deltax_real); bufs->d_deltax_real = NULL; }
    if (bufs->d_xe_real) { cudaFree(bufs->d_xe_real); bufs->d_xe_real = NULL; }
    cufftDestroy(bufs->plan);
    bufs->n_fields = 0;
}

/*
 * filter_box_gpu_inplace — Filter k-space data already on device, in place.
 * Same as filter_box_gpu but operates on a device pointer, no H2D/D2H.
 */
void filter_box_gpu_inplace(void *d_box_v, int box_dim[3], int filter_type,
                            float R, float R_param) {
    cuFloatComplex *d_box = (cuFloatComplex *)d_box_v;
    if (filter_type < 0 || filter_type > 4) {
        LOG_WARNING("Filter type %i is undefined. Box is unfiltered.", filter_type);
        return;
    }

    int dimension = box_dim[0];
    int midpoint = dimension / 2;
    int midpoint_para = box_dim[2] / 2;
    int num_pixels = (unsigned long long)box_dim[0] * box_dim[1] * (box_dim[2] / 2 + 1);
    double delta_k = 2.0 * M_PI / simulation_options_global->BOX_LEN;
    double R_const = 0.0;
    if (filter_type == 3) {
        R_const = exp(-R / R_param);
    }

    int threadsPerBlock = 256;
    int numBlocks = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;
    filter_box_kernel<<<numBlocks, threadsPerBlock>>>(
        d_box, num_pixels, dimension, midpoint, midpoint_para,
        delta_k, R, R_param, R_const, filter_type);

    cudaError_t err = cudaDeviceSynchronize();
    CATCH_CUDA_ERROR(err);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("filter_box_gpu_inplace kernel error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }
}

/*
 * fill_Rbox_table_gpu — GPU-accelerated version of fill_Rbox_table.
 *
 * Uploads unfiltered k-space data to device once.
 * For each filter radius: device memcpy → filter kernel → cuFFT c2r → D2H.
 * Eliminates per-radius H2D transfers and CPU FFTW inverse transforms.
 */
void fill_Rbox_table_gpu(float **result, fftwf_complex *unfiltered_box, double *R_array,
                         int n_R, double min_value, double const_factor,
                         double *min_arr, double *average_arr, double *max_arr) {
    int i, j, k, R_ct;
    double R;
    double ave_buffer, min_out_R, max_out_R;
    int box_dim[3] = {simulation_options_global->HII_DIM, simulation_options_global->HII_DIM,
                      HII_D_PARA};
    int dim = box_dim[0];
    int dim_para = box_dim[2];
    int kspace_pixels = (unsigned long long)dim * dim * (dim_para / 2 + 1);
    int real_pixels = (unsigned long long)dim * dim * dim_para;
    size_t kspace_size = kspace_pixels * sizeof(cuFloatComplex);

    /* cuFFT c2r needs the real output buffer to be (dim * dim * (2*(dim_para/2+1))) floats
     * which is the padded FFTW layout. We allocate that on device. */
    int padded_dim_para = 2 * (dim_para / 2 + 1);
    size_t real_padded_size = (size_t)dim * dim * padded_dim_para * sizeof(float);

    cudaError_t err;

    /* Allocate device buffers */
    cuFloatComplex *d_unfiltered, *d_filtered;
    err = cudaMalloc(&d_unfiltered, kspace_size);
    if (err != cudaSuccess) { LOG_ERROR("cudaMalloc: %s", cudaGetErrorString(err)); Throw(CUDAError); }
    err = cudaMalloc(&d_filtered, real_padded_size > kspace_size ? real_padded_size : kspace_size);
    if (err != cudaSuccess) { cudaFree(d_unfiltered); LOG_ERROR("cudaMalloc: %s", cudaGetErrorString(err)); Throw(CUDAError); }

    /* Upload unfiltered k-space data once */
    err = cudaMemcpy(d_unfiltered, unfiltered_box, kspace_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { LOG_ERROR("cudaMemcpy H2D: %s", cudaGetErrorString(err)); Throw(CUDAError); }

    /* Create cuFFT c2r plan */
    cufftHandle plan_c2r;
    int n_fft[3] = {dim, dim, dim_para};
    cufftResult cufft_err = cufftPlanMany(&plan_c2r, 3, n_fft,
                                          NULL, 1, 0,   /* inembed, istride, idist (auto) */
                                          NULL, 1, 0,   /* onembed, ostride, odist (auto) */
                                          CUFFT_C2R, 1);
    if (cufft_err != CUFFT_SUCCESS) {
        LOG_ERROR("cufftPlanMany failed: %d", cufft_err);
        cudaFree(d_unfiltered); cudaFree(d_filtered);
        Throw(CUDAError);
    }

    /* Host buffer for real-space result (padded layout, same as FFTW) */
    float *h_real = (float *)malloc(real_padded_size);
    if (!h_real) {
        LOG_ERROR("malloc failed for h_real");
        cufftDestroy(plan_c2r); cudaFree(d_unfiltered); cudaFree(d_filtered);
        Throw(CUDAError);
    }

    for (R_ct = 0; R_ct < n_R; R_ct++) {
        R = R_array[R_ct];
        ave_buffer = 0;
        min_out_R = 1e20;
        max_out_R = -1e20;

        /* Copy unfiltered k-space to working buffer on device */
        err = cudaMemcpy(d_filtered, d_unfiltered, kspace_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) { LOG_ERROR("cudaMemcpy D2D: %s", cudaGetErrorString(err)); Throw(CUDAError); }

        /* Apply filter on device (skip if R is below cell size) */
        if (R > physconst.l_factor *
                    (simulation_options_global->BOX_LEN / simulation_options_global->HII_DIM)) {
            filter_box_gpu_inplace(d_filtered, box_dim, astro_options_global->HEAT_FILTER, R, 0.);
        }

        /* cuFFT c2r inverse transform on device */
        cufft_err = cufftExecC2R(plan_c2r, (cufftComplex *)d_filtered, (cufftReal *)d_filtered);
        if (cufft_err != CUFFT_SUCCESS) {
            LOG_ERROR("cufftExecC2R failed: %d", cufft_err);
            Throw(CUDAError);
        }
        cudaDeviceSynchronize();

        /* Download real-space result (padded layout) */
        err = cudaMemcpy(h_real, d_filtered, real_padded_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { LOG_ERROR("cudaMemcpy D2H: %s", cudaGetErrorString(err)); Throw(CUDAError); }

        /* Extract values — same logic as CPU fill_Rbox_table */
#pragma omp parallel private(i, j, k) num_threads(simulation_options_global->N_THREADS)
        {
            float curr;
            unsigned long long int index_r, index_f;
#pragma omp for reduction(+ : ave_buffer) reduction(max : max_out_R) reduction(min : min_out_R)
            for (i = 0; i < box_dim[0]; i++) {
                for (j = 0; j < box_dim[1]; j++) {
                    for (k = 0; k < box_dim[2]; k++) {
                        index_r = grid_index_general_d(i, j, k, box_dim);
                        index_f = (unsigned long long)i * dim * padded_dim_para
                                + (unsigned long long)j * padded_dim_para
                                + (unsigned long long)k;
                        curr = h_real[index_f];
                        /* Clamp before const_factor (matches CPU fill_Rbox_table) */
                        if (curr < min_value) {
                            curr = min_value;
                        }
                        curr *= const_factor;
                        result[R_ct][index_r] = curr;
                        ave_buffer += curr;
                        if (curr > max_out_R) max_out_R = curr;
                        if (curr < min_out_R) min_out_R = curr;
                    }
                }
            }
        }

        if (average_arr) average_arr[R_ct] = ave_buffer / (double)real_pixels;
        if (min_arr) min_arr[R_ct] = min_out_R;
        if (max_arr) max_arr[R_ct] = max_out_R;
    }

    free(h_real);
    cufftDestroy(plan_c2r);
    cudaFree(d_filtered);
    cudaFree(d_unfiltered);
}

// Test function to filter a box without computing a whole output box
//TODO: set device constants here
int test_filter_gpu(float *input_box, double R, double R_param, double R_star, int filter_flag,
                    double *result) {
    int i,j,k;
    unsigned long long int ii, jj;
    int box_dim[3] = {
        simulation_options_global->HII_DIM,
        simulation_options_global->HII_DIM,
        (int)(simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->HII_DIM)};

    //setup the box
    fftwf_complex *box_unfiltered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);
    fftwf_complex *box_filtered = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex)*HII_KSPACE_NUM_PIXELS);

    for (i=0; i<box_dim[0]; i++)
        for (j=0; j<box_dim[1]; j++)
            for (k=0; k<box_dim[2]; k++) {
                ii = grid_index_general_d(i, j, k, box_dim);
                jj = grid_index_fftw_r_d(i, j, k, box_dim);
                *((float *)box_unfiltered + jj) = input_box[ii];
            }

    dft_r2c_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM, HII_D_PARA, simulation_options_global->N_THREADS, box_unfiltered);

    // Convert to CUDA complex type
    cuFloatComplex* box_unfiltered_cu = reinterpret_cast<cuFloatComplex*>(box_unfiltered);

    for(ii=0;ii<HII_KSPACE_NUM_PIXELS;ii++){
        // box_unfiltered[ii] /= (double)HII_TOT_NUM_PIXELS;
        box_unfiltered_cu[ii] = cuCdivf(box_unfiltered_cu[ii], make_cuFloatComplex((float)HII_TOT_NUM_PIXELS, 0.f));
    }

    memcpy(box_filtered, box_unfiltered, sizeof(fftwf_complex) * HII_KSPACE_NUM_PIXELS);

    filter_box_gpu(box_filtered, box_dim, filter_flag, R, R_param, R_star);

    dft_c2r_cube(matter_options_global->USE_FFTW_WISDOM, simulation_options_global->HII_DIM, HII_D_PARA, simulation_options_global->N_THREADS, box_filtered);

    for (i=0; i<box_dim[0]; i++)
        for (j=0; j<box_dim[1]; j++)
            for (k=0; k<box_dim[2]; k++) {
                ii = grid_index_general_d(i, j, k, box_dim);
                jj = grid_index_fftw_r_d(i, j, k, box_dim);
                result[ii] = *((float *)box_filtered + jj);
            }

    fftwf_free(box_unfiltered);
    fftwf_free(box_filtered);

    return 0;
}
