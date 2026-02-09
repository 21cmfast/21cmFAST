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
    float velocity_scale, float velocity_scale_z,
    bool perturb_on_high_res, bool use_2lpt
    ) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < DIM * DIM * d_para) {

        // Get index of density cell
        int i = idx / (d_para * DIM);
        int j = (idx / d_para) % DIM;
        int k = idx % d_para;

        unsigned long long r_index = compute_R_INDEX(i, j, k, DIM, d_para);

        // ============================================================
        // CPU-MATCHING ALGORITHM: Start with integer position, apply
        // velocity displacement in dens_dim units, then scale to out_dim
        // ============================================================

        // Position initialization: integer cell index (matches CPU)
        double pos_x = (double)i;
        double pos_y = (double)j;
        double pos_z = (double)k;

        // Dimension ratios (matches CPU)
        double dim_ratio_vel = 1.0 / (double)f_pixel_factor;  // vel_dim / dens_dim
        double dim_ratio_out = (double)dimension / (double)DIM;  // out_dim / dens_dim

        // Velocity displacement factor in dens_dim units (matches CPU vdf)
        // CPU: vdf = velocity_scale * BOX_LEN / box_size * dens_dim
        // Since velocity_scale = (growth - init_growth) / BOX_LEN,
        // and box_size = BOX_LEN (cubic), we have:
        // vdf = velocity_scale * DIM
        double vdf_xy = (double)velocity_scale * (double)DIM;
        double vdf_z = (double)velocity_scale_z * (double)d_para;

        // Update locations
        unsigned long long HII_index;

        if (perturb_on_high_res) {
            // Apply velocity displacement in dens_dim units (matches CPU)
            pos_x += (double)hires_vx[r_index] * vdf_xy;
            pos_y += (double)hires_vy[r_index] * vdf_xy;
            pos_z += (double)hires_vz[r_index] * vdf_z;
        }
        else {
            // Match CPU resample_index exactly: idx_out = (int)(idx_in * dim_ratio + 0.5)
            int HII_i = (int)(i * dim_ratio_vel + 0.5);
            int HII_j = (int)(j * dim_ratio_vel + 0.5);
            int HII_k = (int)(k * dim_ratio_vel + 0.5);
            // Match CPU wrap_coord: wrap into [0, hii_d)
            while (HII_i >= hii_d) HII_i -= hii_d;
            while (HII_i < 0) HII_i += hii_d;
            while (HII_j >= hii_d) HII_j -= hii_d;
            while (HII_j < 0) HII_j += hii_d;
            while (HII_k >= hii_d_para) HII_k -= hii_d_para;
            while (HII_k < 0) HII_k += hii_d_para;
            HII_index = compute_HII_R_INDEX(HII_i, HII_j, HII_k, hii_d, hii_d_para);
            // Apply velocity displacement in dens_dim units (matches CPU)
            pos_x += (double)lowres_vx[HII_index] * vdf_xy;
            pos_y += (double)lowres_vy[HII_index] * vdf_xy;
            pos_z += (double)lowres_vz[HII_index] * vdf_z;
        }

        // 2LPT (add second order corrections) - NOTE: currently disabled
        if (use_2lpt) {
            // TODO: Need to compute velocity_displacement_factor_2LPT matching CPU
            // For now, 2LPT is disabled in the caller (use_2lpt = false)
            if (perturb_on_high_res) {
                pos_x -= (double)hires_vx_2LPT[r_index];
                pos_y -= (double)hires_vy_2LPT[r_index];
                pos_z -= (double)hires_vz_2LPT[r_index];
            }
            else {
                pos_x -= (double)lowres_vx_2LPT[HII_index];
                pos_y -= (double)lowres_vy_2LPT[HII_index];
                pos_z -= (double)lowres_vz_2LPT[HII_index];
            }
        }

        // Scale position to output grid dimensions (matches CPU: pos *= dim_ratio_out)
        pos_x *= dim_ratio_out;
        pos_y *= dim_ratio_out;
        pos_z *= dim_ratio_out;

        // Output dimensions for wrapping
        int out_dim_xy = dimension;
        int out_dim_z = dimension * non_cubic_factor;

        // ============================================================
        // CPU-MATCHING CIC: floor-based corner algorithm
        // ============================================================

        // Get base cell index using floor (matches CPU)
        int xi = (int)floor(pos_x);
        int yi = (int)floor(pos_y);
        int zi = (int)floor(pos_z);

        // Get +1 neighbor indices
        int xp1 = xi + 1;
        int yp1 = yi + 1;
        int zp1 = zi + 1;

        // Fractional distance from base cell (matches CPU: dist = pos - floor(pos))
        double d_x = pos_x - (double)xi;
        double d_y = pos_y - (double)yi;
        double d_z = pos_z - (double)zi;

        // Wrap base indices into [0, out_dim) (matches CPU wrap_coord)
        while (xi >= out_dim_xy) xi -= out_dim_xy;
        while (xi < 0) xi += out_dim_xy;
        while (yi >= out_dim_xy) yi -= out_dim_xy;
        while (yi < 0) yi += out_dim_xy;
        while (zi >= out_dim_z) zi -= out_dim_z;
        while (zi < 0) zi += out_dim_z;

        // Wrap +1 indices
        while (xp1 >= out_dim_xy) xp1 -= out_dim_xy;
        while (xp1 < 0) xp1 += out_dim_xy;
        while (yp1 >= out_dim_xy) yp1 -= out_dim_xy;
        while (yp1 < 0) yp1 += out_dim_xy;
        while (zp1 >= out_dim_z) zp1 -= out_dim_z;
        while (zp1 < 0) zp1 += out_dim_z;

        // CIC weights (matches CPU: t = 1 - d)
        double t_x = 1.0 - d_x;
        double t_y = 1.0 - d_y;
        double t_z = 1.0 - d_z;

        // double scaled_density = 1 + init_growth_factor * __ldg(&hires_density[r_index]);
        double scaled_density = 1.0 + init_growth_factor * hires_density[r_index];

        // Diagnostic output for sample cells (GPU) - matches CPU format
        // Note: Only works when !perturb_on_high_res (the common test case)
        if (!perturb_on_high_res && ((i == 0 && j == 0 && k == 0) || (i == 100 && j == 50 && k == 25))) {
            // Recompute velocity indices for diagnostic (matches the actual calculation above)
            int diag_HII_i = (int)(i * dim_ratio_vel + 0.5);
            int diag_HII_j = (int)(j * dim_ratio_vel + 0.5);
            int diag_HII_k = (int)(k * dim_ratio_vel + 0.5);
            // Apply wrapping for diagnostic
            while (diag_HII_i >= hii_d) diag_HII_i -= hii_d;
            while (diag_HII_j >= hii_d) diag_HII_j -= hii_d;
            while (diag_HII_k >= hii_d_para) diag_HII_k -= hii_d_para;
            // Compute velocity contribution for diagnostic
            double vel_contrib_x = (double)lowres_vx[HII_index] * vdf_xy;
            double vel_contrib_y = (double)lowres_vy[HII_index] * vdf_xy;
            double vel_contrib_z = (double)lowres_vz[HII_index] * vdf_z;
            printf("[DIAG_GPU] cell=(%d,%d,%d) dens_dim=%d out_dim=%d dim_ratio_out=%.9f\n",
                   i, j, k, DIM, dimension, dim_ratio_out);
            printf("[DIAG_GPU]   vel_idx=(%d,%d,%d) vel_index=%llu\n",
                   diag_HII_i, diag_HII_j, diag_HII_k, HII_index);
            printf("[DIAG_GPU]   vel_raw=(%.9f,%.9f,%.9f)\n",
                   lowres_vx[HII_index], lowres_vy[HII_index], lowres_vz[HII_index]);
            printf("[DIAG_GPU]   vdf=(%.9f,%.9f,%.9f)\n",
                   vdf_xy, vdf_xy, vdf_z);
            printf("[DIAG_GPU]   vel_contrib=(%.9f,%.9f,%.9f)\n",
                   vel_contrib_x, vel_contrib_y, vel_contrib_z);
            printf("[DIAG_GPU]   pos_init=(%d,%d,%d) pos_displaced=(%.9f,%.9f,%.9f)\n",
                   i, j, k,
                   (double)i + vel_contrib_x, (double)j + vel_contrib_y, (double)k + vel_contrib_z);
            printf("[DIAG_GPU]   pos_final=(%.9f,%.9f,%.9f) dens=%.9f\n",
                   pos_x, pos_y, pos_z, scaled_density);
            printf("[DIAG_GPU]   cic_base_raw=(%d,%d,%d) cic_base_wrapped=(%d,%d,%d)\n",
                   (int)floor(pos_x), (int)floor(pos_y), (int)floor(pos_z),
                   xi, yi, zi);
            printf("[DIAG_GPU]   cic_dist=(%.9f,%.9f,%.9f)\n",
                   d_x, d_y, d_z);
        }

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

extern "C" double* MapMass_gpu(
    InitialConditions *boxes, double *resampled_box,
    int dimension, float f_pixel_factor, float init_growth_factor,
    float velocity_scale, float velocity_scale_z
) {
    // Box shapes from outputs.py and convenience macros
    size_t size_double, size_float;
    if(matter_options_global->PERTURB_ON_HIGH_RES) {
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

    if (matter_options_global->PERTURB_ON_HIGH_RES) {
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
    if (false /* USE_2LPT obsolete */) {
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
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
        dimension, simulation_options_global->DIM, d_para, hii_d, hii_d_para, simulation_options_global->NON_CUBIC_FACTOR,
        f_pixel_factor, init_growth_factor, velocity_scale, velocity_scale_z,
        matter_options_global->PERTURB_ON_HIGH_RES, false /* USE_2LPT obsolete */);

    // // Only use during development!
    // err = cudaDeviceSynchronize();
    // CATCH_CUDA_ERROR(err);

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

    if (matter_options_global->PERTURB_ON_HIGH_RES) {
        cudaFree(hires_vx);
        cudaFree(hires_vy);
        cudaFree(hires_vz);
    }
    else {
        cudaFree(lowres_vx);
        cudaFree(lowres_vy);
        cudaFree(lowres_vz);
    }
    if (false /* USE_2LPT obsolete */) {
        if (matter_options_global->PERTURB_ON_HIGH_RES) {
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
