// Re-write of perturb_field.c for being accessible within the MCMC

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <fftw3.h>

// GPU
#include <cuda.h>
#include <cuda_runtime.h>

#include "cexcept.h"
#include "exceptions.h"
#include "logger.h"
#include "Constants.h"
#include "indexing.h"
#include "InputParameters.h"
#include "OutputStructs.h"
#include "cosmology.h"
#include "dft.h"
#include "debugging.h"
#include "filtering.h"

#include "PerturbField.h"

// #define R_INDEX(x,y,z)((unsigned long long)((z)+D_PARA*((y)+D*(x))))
__device__ inline unsigned long long compute_R_INDEX(int i, int j, int k, int dim, long long d_para) {
    return k + d_para * (j + dim * i);
}

// #define HII_R_INDEX(x,y,z)((unsigned long long)((z)+HII_D_PARA*((y)+HII_D*(x))))
__device__ inline unsigned long long compute_HII_R_INDEX(int i, int j, int k, int hii_d, long long hii_d_para) {
    return k + hii_d_para * (j + hii_d * i);
}

// Is const needed as well as __restrict__?
__global__ void perturb_density_field_kernel(
    double *resampled_box,
    // const float* __restrict__ hires_density,
    // const float* __restrict__ hires_vx,
    // const float* __restrict__ hires_vy,
    // const float* __restrict__ hires_vz,
    // const float* __restrict__ lowres_vx,
    // const float* __restrict__ lowres_vy,
    // const float* __restrict__ lowres_vz,
    // const float* __restrict__ hires_vx_2LPT,
    // const float* __restrict__ hires_vy_2LPT,
    // const float* __restrict__ hires_vz_2LPT,
    // const float* __restrict__ lowres_vx_2LPT,
    // const float* __restrict__ lowres_vy_2LPT,
    // const float* __restrict__ lowres_vz_2LPT,
    float* hires_density,
    float* hires_vx,
    float* hires_vy,
    float* hires_vz,
    float* lowres_vx,
    float* lowres_vy,
    float* lowres_vz,
    float* hires_vx_2LPT,
    float* hires_vy_2LPT,
    float* hires_vz_2LPT,
    float* lowres_vx_2LPT,
    float* lowres_vy_2LPT,
    float* lowres_vz_2LPT,
    int dimension, int DIM,
    long long d_para, long long hii_d, long long hii_d_para,
    int non_cubic_factor,
    float f_pixel_factor, float init_growth_factor,
    bool perturb_on_high_res, bool use_2lpt
    ) {

    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < DIM * DIM * d_para) {

        // Get index of density cell
        int i = idx / (d_para * DIM);
        int j = (idx / d_para) % DIM;
        int k = idx % d_para;

        unsigned long long r_index = compute_R_INDEX(i, j, k, DIM, d_para);

        // Map index to location in units of box size
        double xf = (i + 0.5) / (DIM + 0.0);
        double yf = (j + 0.5) / (DIM + 0.0);
        double zf = (k + 0.5) / (d_para + 0.0);

        // Update locations
        unsigned long long HII_index;

        if (perturb_on_high_res) {
            // xf += __ldg(&hires_vx[r_index]);
            // yf += __ldg(&hires_vy[r_index]);
            // zf += __ldg(&hires_vz[r_index]);
            xf += hires_vx[r_index];
            yf += hires_vy[r_index];
            zf += hires_vz[r_index];
        }
        else {
            unsigned long long HII_i = (unsigned long long)(i / f_pixel_factor);
            unsigned long long HII_j = (unsigned long long)(j / f_pixel_factor);
            unsigned long long HII_k = (unsigned long long)(k / f_pixel_factor);
            HII_index = compute_HII_R_INDEX(HII_i, HII_j, HII_k, hii_d, hii_d_para);
            // xf += __ldg(&lowres_vx[HII_index]);
            // yf += __ldg(&lowres_vy[HII_index]);
            // zf += __ldg(&lowres_vz[HII_index]);
            xf += lowres_vx[HII_index];
            yf += lowres_vy[HII_index];
            zf += lowres_vz[HII_index];
        }

        // 2LPT (add second order corrections)
        if (use_2lpt) {
            if (perturb_on_high_res) {
                // xf -= __ldg(&hires_vx_2LPT[r_index]);
                // yf -= __ldg(&hires_vy_2LPT[r_index]);
                // zf -= __ldg(&hires_vz_2LPT[r_index]);
                xf -= hires_vx_2LPT[r_index];
                yf -= hires_vy_2LPT[r_index];
                zf -= hires_vz_2LPT[r_index];
            }
            else {
                // xf -= __ldg(&lowres_vx_2LPT[HII_index]);
                // yf -= __ldg(&lowres_vy_2LPT[HII_index]);
                // zf -= __ldg(&lowres_vz_2LPT[HII_index]);
                xf -= lowres_vx_2LPT[HII_index];
                yf -= lowres_vy_2LPT[HII_index];
                zf -= lowres_vz_2LPT[HII_index];
            }
        }

        // TODO: shared between threads?
        // Convert once to reduce overhead of multiple casts
        double dimension_double = (double)(dimension);
        double dimension_factored_double = dimension_double * (double)(non_cubic_factor);
        int dimension_factored = dimension * non_cubic_factor;

        // Scale coordinates back to grid size
        xf *= dimension_double;
        yf *= dimension_double;
        zf *= dimension_factored_double;

        // Wrap coordinates to keep them within valid boundaries
        xf = fmod(fmod(xf, dimension_double) + dimension_double, dimension_double);
        yf = fmod(fmod(yf, dimension_double) + dimension_double, dimension_double);
        zf = fmod(fmod(zf, dimension_factored_double) + dimension_factored_double, dimension_factored_double);

        // FROM NVIDIA DOCS:
        // __device__ doublenearbyint(double x) // Round the input argument to the nearest integer.
        // There are SO many double-to-int conversion intrinsics. How to know if should use any?

        // Get integer values for indices from double precision values
        int xi = xf;
        int yi = yf;
        int zi = zf;

        // Wrap index coordinates to ensure no out-of-bounds array access will be attempted
        xi = ((xi % dimension) + dimension) % dimension;
        yi = ((yi % dimension) + dimension) % dimension;
        zi = ((zi % dimension_factored) + dimension_factored) % dimension_factored;

        // Determine the fraction of the perturbed cell which overlaps with the 8 nearest grid cells,
        // based on the grid cell which contains the centre of the perturbed cell
        float d_x = fabs(xf - (double)(xi + 0.5)); // Absolute distances from grid cell centre to perturbed cell centre
        float d_y = fabs(yf - (double)(yi + 0.5)); // (also) The fractions of mass which will be moved to neighbouring cells
        float d_z = fabs(zf - (double)(zi + 0.5));

        // 8 neighbour cells-of-interest will be shifted left/down/behind if perturbed midpoint is in left/bottom/back corner of cell.
        if (xf < (double)(xi + 0.5)) {
            // If perturbed cell centre is less than the mid-point then update fraction
            // of mass in the cell and determine the cell centre of neighbour to be the
            // lowest grid point index
            d_x = 1. - d_x;
            xi -= 1;
            xi = (xi + dimension) % dimension; // Only this critera is possible as iterate back by one (we cannot exceed DIM)
        }
        if(yf < (double)(yi + 0.5)) {
            d_y = 1. - d_y;
            yi -= 1;
            yi = (yi + dimension) % dimension;
        }
        if(zf < (double)(zi + 0.5)) {
            d_z = 1. - d_z;
            zi -= 1;
            zi = (zi + (unsigned long long)(non_cubic_factor * dimension)) % (unsigned long long)(non_cubic_factor * dimension);
        }
        // The fractions of mass which will remain with perturbed cell
        float t_x = 1. - d_x;
        float t_y = 1. - d_y;
        float t_z = 1. - d_z;

        // Determine the grid coordinates of the 8 neighbouring cells.
        // Neighbours will be in positive direction; front/right/above cells (-> 2x2 cube, with perturbed cell bottom/left/back)
        // Takes into account the offset based on cell centre determined above
        int xp1 = (xi + 1) % dimension;
        int yp1 = (yi + 1) % dimension;
        int zp1 = (zi + 1) % (unsigned long long)(non_cubic_factor * dimension);

        // double scaled_density = 1 + init_growth_factor * __ldg(&hires_density[r_index]);
        double scaled_density = 1.0 + init_growth_factor * hires_density[r_index];

        if (perturb_on_high_res) {
            // Redistribute the mass over the 8 neighbouring cells according to cloud in cell
            // Cell mass = (1 + init_growth_factor * orig_density) * (proportion of mass to distribute)
            atomicAdd(&resampled_box[compute_R_INDEX(xi, yi, zi, DIM, d_para)], scaled_density * t_x * t_y * t_z);
            atomicAdd(&resampled_box[compute_R_INDEX(xp1, yi, zi, DIM, d_para)], scaled_density * d_x * t_y * t_z);
            atomicAdd(&resampled_box[compute_R_INDEX(xi, yp1, zi, DIM, d_para)], scaled_density * t_x * d_y * t_z);
            atomicAdd(&resampled_box[compute_R_INDEX(xp1, yp1, zi, DIM, d_para)], scaled_density * d_x * d_y * t_z);
            atomicAdd(&resampled_box[compute_R_INDEX(xi, yi, zp1, DIM, d_para)], scaled_density * t_x * t_y * d_z);
            atomicAdd(&resampled_box[compute_R_INDEX(xp1, yi, zp1, DIM, d_para)], scaled_density * d_x * t_y * d_z);
            atomicAdd(&resampled_box[compute_R_INDEX(xi, yp1, zp1, DIM, d_para)], scaled_density * t_x * d_y * d_z);
            atomicAdd(&resampled_box[compute_R_INDEX(xp1, yp1, zp1, DIM, d_para)], scaled_density * d_x * d_y * d_z);
        }
        else {
            atomicAdd(&resampled_box[compute_HII_R_INDEX(xi, yi, zi, hii_d, hii_d_para)], scaled_density * t_x * t_y * t_z);
            atomicAdd(&resampled_box[compute_HII_R_INDEX(xp1, yi, zi, hii_d, hii_d_para)], scaled_density * d_x * t_y * t_z);
            atomicAdd(&resampled_box[compute_HII_R_INDEX(xi, yp1, zi, hii_d, hii_d_para)], scaled_density * t_x * d_y * t_z);
            atomicAdd(&resampled_box[compute_HII_R_INDEX(xp1, yp1, zi, hii_d, hii_d_para)], scaled_density * d_x * d_y * t_z);
            atomicAdd(&resampled_box[compute_HII_R_INDEX(xi, yi, zp1, hii_d, hii_d_para)], scaled_density * t_x * t_y * d_z);
            atomicAdd(&resampled_box[compute_HII_R_INDEX(xp1, yi, zp1, hii_d, hii_d_para)], scaled_density * d_x * t_y * d_z);
            atomicAdd(&resampled_box[compute_HII_R_INDEX(xi, yp1, zp1, hii_d, hii_d_para)], scaled_density * t_x * d_y * d_z);
            atomicAdd(&resampled_box[compute_HII_R_INDEX(xp1, yp1, zp1, hii_d, hii_d_para)], scaled_density * d_x * d_y * d_z);
        }
    }
}

double* MapMass_gpu(
    UserParams *user_params, CosmoParams *cosmo_params, InitialConditions *boxes, double *resampled_box,
    int dimension, float f_pixel_factor, float init_growth_factor
) {
    // Makes the parameter structs visible to a variety of functions/macros
    // Do each time to avoid Python garbage collection issues
    Broadcast_struct_global_noastro(user_params, cosmo_params);

    // Box shapes from outputs.py and convenience macros
    size_t size_double, size_float;
    if(user_params->PERTURB_ON_HIGH_RES) {
        size_double = TOT_NUM_PIXELS * sizeof(double);
        size_float = TOT_NUM_PIXELS * sizeof(float);
    }
    else {
        size_double = HII_TOT_NUM_PIXELS * sizeof(double);
        size_float = HII_TOT_NUM_PIXELS * sizeof(float);
    }

    // Allocate device memory for output box and set to 0.
    double* d_resampled_box;
    cudaMalloc((void**)&d_resampled_box, size_double);
    cudaMemset(d_resampled_box, 0, size_double); // fills size_double bytes with byte=0

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Allocate device memory for density field
    float* hires_density;
    cudaMalloc(&hires_density, (TOT_NUM_PIXELS * sizeof(float))); // from 21cmFAST.h, outputs.py & indexing.h
    cudaMemcpy(hires_density, boxes->hires_density, (TOT_NUM_PIXELS * sizeof(float)), cudaMemcpyHostToDevice);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Allocate device memory and copy arrays to device as per user_params
    float* hires_vx; // floats as per 21cmFAST.h
    float* hires_vy;
    float* hires_vz;
    float* lowres_vx;
    float* lowres_vy;
    float* lowres_vz;
    float* hires_vx_2LPT;
    float* hires_vy_2LPT;
    float* hires_vz_2LPT;
    float* lowres_vx_2LPT;
    float* lowres_vy_2LPT;
    float* lowres_vz_2LPT;

    if (user_params->PERTURB_ON_HIGH_RES) {
        cudaMalloc(&hires_vx, size_float);
        cudaMalloc(&hires_vy, size_float);
        cudaMalloc(&hires_vz, size_float);
        cudaMemcpy(hires_vx, boxes->hires_vx, size_float, cudaMemcpyHostToDevice);
        cudaMemcpy(hires_vy, boxes->hires_vy, size_float, cudaMemcpyHostToDevice);
        cudaMemcpy(hires_vz, boxes->hires_vz, size_float, cudaMemcpyHostToDevice);
    }
    else {
        cudaMalloc(&lowres_vx, size_float);
        cudaMalloc(&lowres_vy, size_float);
        cudaMalloc(&lowres_vz, size_float);
        cudaMemcpy(lowres_vx, boxes->lowres_vx, size_float, cudaMemcpyHostToDevice);
        cudaMemcpy(lowres_vy, boxes->lowres_vy, size_float, cudaMemcpyHostToDevice);
        cudaMemcpy(lowres_vz, boxes->lowres_vz, size_float, cudaMemcpyHostToDevice);
    }
    if (user_params->USE_2LPT) {
        if (user_params->PERTURB_ON_HIGH_RES) {
            cudaMalloc(&hires_vx_2LPT, size_float);
            cudaMalloc(&hires_vy_2LPT, size_float);
            cudaMalloc(&hires_vz_2LPT, size_float);
            cudaMemcpy(hires_vx_2LPT, boxes->hires_vx_2LPT, size_float, cudaMemcpyHostToDevice);
            cudaMemcpy(hires_vy_2LPT, boxes->hires_vy_2LPT, size_float, cudaMemcpyHostToDevice);
            cudaMemcpy(hires_vz_2LPT, boxes->hires_vz_2LPT, size_float, cudaMemcpyHostToDevice);
        }
        else {
            cudaMalloc(&lowres_vx_2LPT, size_float);
            cudaMalloc(&lowres_vy_2LPT, size_float);
            cudaMalloc(&lowres_vz_2LPT, size_float);
            cudaMemcpy(lowres_vx_2LPT, boxes->lowres_vx_2LPT, size_float, cudaMemcpyHostToDevice);
            cudaMemcpy(lowres_vy_2LPT, boxes->lowres_vy_2LPT, size_float, cudaMemcpyHostToDevice);
            cudaMemcpy(lowres_vz_2LPT, boxes->lowres_vz_2LPT, size_float, cudaMemcpyHostToDevice);
        }
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Can't pass macro straight to kernel
    long long d_para = D_PARA;
    long long hii_d = HII_D;
    long long hii_d_para = HII_D_PARA;

    // Invoke kernel
    int threadsPerBlock = 256;
    int numBlocks = (TOT_NUM_PIXELS + threadsPerBlock - 1) / threadsPerBlock;
    perturb_density_field_kernel<<<numBlocks, threadsPerBlock>>>(
        d_resampled_box, hires_density, hires_vx, hires_vy, hires_vz, lowres_vx, lowres_vy, lowres_vz,
        hires_vx_2LPT, hires_vy_2LPT, hires_vz_2LPT, lowres_vx_2LPT, lowres_vy_2LPT, lowres_vz_2LPT,
        dimension, user_params->DIM, d_para, hii_d, hii_d_para, user_params->NON_CUBIC_FACTOR,
        f_pixel_factor, init_growth_factor, user_params->PERTURB_ON_HIGH_RES, user_params->USE_2LPT);

    // Only use during development!
    err = cudaDeviceSynchronize();
    CATCH_CUDA_ERROR(err);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("Kernel launch error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Copy results from device to host
    err = cudaMemcpy(resampled_box, d_resampled_box, size_double, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    // Deallocate device memory
    cudaFree(d_resampled_box);
    cudaFree(hires_density);

    if (user_params->PERTURB_ON_HIGH_RES) {
        cudaFree(hires_vx);
        cudaFree(hires_vy);
        cudaFree(hires_vz);
    }
    else {
        cudaFree(lowres_vx);
        cudaFree(lowres_vy);
        cudaFree(lowres_vz);
    }
    if (user_params->USE_2LPT) {
        if (user_params->PERTURB_ON_HIGH_RES) {
            cudaFree(hires_vx_2LPT);
            cudaFree(hires_vy_2LPT);
            cudaFree(hires_vz_2LPT);
        }
        else {
            cudaFree(lowres_vx_2LPT);
            cudaFree(lowres_vy_2LPT);
            cudaFree(lowres_vz_2LPT);
        }
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        LOG_ERROR("CUDA error: %s", cudaGetErrorString(err));
        Throw(CUDAError);
    }

    return resampled_box;
}
