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

    // Clamp to valid range
    if (idx < 0) idx = 0;
    if (idx >= 399) idx = 398;

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
    if (tid == 0) {
        ave_sfrd_buf[blockIdx.x] = sdata[0];
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
    // Ensure at least 1 block: run_global_evolution uses HII_DIM=1 (num_pixels=1),
    // which would give numBlocks=0 and an invalid kernel launch configuration.
    unsigned int numBlocks = ceil(num_pixels / (threadsPerBlock * 2));
    if (numBlocks == 0) numBlocks = 1;
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
    // Get growth factor for current filter step
    double zpp_growth_R_ct = zpp_growth[R_ct];

    // Copy data from host to device
    CALL_CUDA(cudaMemcpy(d_y_arr, SFRD_conditional_table->y_arr, sizeof(float) * SFRD_conditional_table->n_bin, cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMemcpy(d_dens_R_grid, dens_R_grid, sizeof(float) * num_pixels, cudaMemcpyHostToDevice));
    LOG_INFO("SFRD_conditional_table.y_arr and density grid copied to device.");

    // Ensure at least 1 block (see allocation above for rationale).
    unsigned int numBlocks = ceil(num_pixels / (threadsPerBlock * 2));
    if (numBlocks == 0) numBlocks = 1;
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
    CALL_CUDA(cudaGetLastError());
    CALL_CUDA(cudaDeviceSynchronize());
    LOG_INFO("SpinTemperatureBox compute-and-reduce kernel called.");

    // Use thrust to reduce computed sums to one value.
    // Wrap device pointer in a thrust::device_ptr
    thrust::device_ptr<double> d_ave_sfrd_buf_ptr(d_ave_sfrd_buf);
    // Reduce final buffer sums to one value
    double ave_sfrd_buf = thrust::reduce(d_ave_sfrd_buf_ptr, d_ave_sfrd_buf_ptr + numBlocks, 0., thrust::plus<double>());
    CALL_CUDA(cudaGetLastError());
    LOG_INFO("SFRD sum reduced to single value by thrust::reduce operation.");

    // Copy results from device to host
    CALL_CUDA(cudaMemcpy(sfrd_grid, d_sfrd_grid, sizeof(float) * num_pixels, cudaMemcpyDeviceToHost));
    LOG_INFO("SFRD sum copied to host.");

    return ave_sfrd_buf;
}

extern "C" void free_sfrd_gpu_data(
    float **d_y_arr, // copies of pointers to pointers
    float **d_dens_R_grid,
    float **d_sfrd_grid,
    double **d_ave_sfrd_buf
) {
    // Need to dereference the pointers to pointers (*)
    CALL_CUDA(cudaFree(*d_y_arr));
    CALL_CUDA(cudaFree(*d_dens_R_grid));
    CALL_CUDA(cudaFree(*d_sfrd_grid));
    CALL_CUDA(cudaFree(*d_ave_sfrd_buf));
    LOG_INFO("Device memory freed.");
}

// ============================================================================
// Phase 11.6b: Spin temperature (get_Ts_fast) GPU kernel
// ============================================================================
//
// Replaces the per-pixel OpenMP loop that calls get_Ts_fast after the R-loop.
// Each thread computes one pixel's Ts, Tk, x_e, J_21_LW from the accumulated
// spectral integration arrays and the previous snapshot's spin temperature.

// Kappa tables (hardcoded from heating_helper_progs.c, 30 points each)
// All are uniformly spaced in log(T), with linear interpolation.
#define KAPPA_NPTS 30

// kappa_10 (H-H collisions): Allison & Dalgarno / Zygelman
__constant__ double d_kap10_logT0 = 0.0;
__constant__ double d_kap10_inv_dlogT = 1.0 / 0.317597943861;
__constant__ double d_kap10_y[KAPPA_NPTS] = {
    -29.6115227098, -29.6228184691, -29.5917673123, -29.4469989515,
    -29.1171430989, -28.5382192456, -27.7424388865, -26.8137036254,
    -25.8749225449, -25.0548322235, -24.4005076336, -23.8952335377,
    -23.5075651004, -23.201637629,  -22.9593758343, -22.7534867461,
    -22.5745752086, -22.4195690855, -22.2833176123, -22.1549519419,
    -22.0323282988, -21.9149994721, -21.800121439,  -21.6839502137,
    -21.5662434981, -21.4473595491, -21.3279560712, -21.2067614838,
    -21.0835560288, -20.9627928675
};

// kappa_10_pH (H-proton collisions)
__constant__ double d_kap_pH_logT0 = 0.0;
__constant__ double d_kap_pH_inv_dlogT = 1.0 / 0.341499570777;
__constant__ double d_kap_pH_y[KAPPA_NPTS] = {
    -21.6395565688, -21.5641675629, -21.5225112028, -21.5130514508,
    -21.5342522691, -21.5845293039, -21.6581396414, -21.7420392948,
    -21.8221380683, -21.8837908896, -21.9167553997, -21.9200173678,
    -21.8938574675, -21.8414464728, -21.7684762963, -21.6796222358,
    -21.5784701374, -21.4679438133, -21.3503236936, -21.2277666787,
    -21.1017425964, -20.9733966978, -20.8437244283, -20.7135746917,
    -20.583135408,  -20.4523507819, -20.3215504736, -20.1917429161,
    -20.0629513946, -19.9343540344
};

// kappa_10_elec (H-electron collisions)
__constant__ double d_kap_elec_logT0 = 0.0;
__constant__ double d_kap_elec_inv_dlogT = 1.0 / 0.396997429827;
__constant__ double d_kap_elec_y[KAPPA_NPTS] = {
    -22.1549007191, -21.9576919899, -21.760758435,  -21.5641795674,
    -21.3680349001, -21.1724124486, -20.9774403051, -20.78327367,
    -20.5901042551, -20.3981934669, -20.2078762485, -20.0195787458,
    -19.8339587914, -19.6518934427, -19.4745894649, -19.3043925781,
    -19.1444129787, -18.9986014565, -18.8720602784, -18.768679825,
    -18.6909581885, -18.6387511068, -18.6093755705, -18.5992098958,
    -18.6050625357, -18.6319366207, -18.7017996535, -18.8477153986,
    -19.0813436512, -19.408859606
};

// Device function: linear interpolation on log-spaced kappa table
__device__ inline double kappa_interp_gpu(
    double logT, const double *y, double logT0, double inv_dlogT, double extrap_power
) {
    if (logT < logT0) return exp(y[0]);
    double logT_last = logT0 + (KAPPA_NPTS - 1) / inv_dlogT;
    if (logT > logT_last) {
        // Power-law extrapolation
        double slope = (y[KAPPA_NPTS - 1] - y[KAPPA_NPTS - 2]) * inv_dlogT;
        return exp(y[KAPPA_NPTS - 1] + slope * (logT - logT_last));
    }
    int idx = (int)floor((logT - logT0) * inv_dlogT);
    if (idx >= KAPPA_NPTS - 1) idx = KAPPA_NPTS - 2;
    double frac = (logT - (logT0 + idx / inv_dlogT)) * inv_dlogT;
    return exp(y[idx] + frac * (y[idx + 1] - y[idx]));
}

__device__ inline double kappa_10_gpu(double Tk) {
    double logT = log(Tk);
    if (logT > d_kap10_y[0] + (KAPPA_NPTS - 1) / d_kap10_inv_dlogT + 1.0) {
        // Special: kappa_10 uses power-law with exponent 0.381
        double logT_last = (KAPPA_NPTS - 1) / d_kap10_inv_dlogT;
        return exp(d_kap10_y[KAPPA_NPTS - 1]) * pow(exp(logT) / exp(logT_last), 0.381);
    }
    return kappa_interp_gpu(logT, d_kap10_y, d_kap10_logT0, d_kap10_inv_dlogT, 0.381);
}

__device__ inline double kappa_10_pH_gpu(double Tk) {
    return kappa_interp_gpu(log(Tk), d_kap_pH_y, d_kap_pH_logT0, d_kap_pH_inv_dlogT, 0.0);
}

__device__ inline double kappa_10_elec_gpu(double Tk) {
    return kappa_interp_gpu(log(Tk), d_kap_elec_y, d_kap_elec_logT0, d_kap_elec_inv_dlogT, 0.0);
}

// Device function: Case-A recombination coefficient (from thermochem.c)
__device__ inline double alpha_A_gpu(double T) {
    double logT = log(T / 1.1604505e4);
    return exp(-28.6130338 - 0.72411256 * logT - 2.02604473e-2 * logT * logT -
               2.38086188e-3 * logT * logT * logT - 3.21260521e-4 * pow(logT, 4) -
               1.42150291e-5 * pow(logT, 5) + 4.98910892e-6 * pow(logT, 6) +
               5.75561414e-7 * pow(logT, 7) - 1.85676704e-8 * pow(logT, 8) -
               3.07113524e-9 * pow(logT, 9));
}

// Struct to pass all the per-snapshot constants to the Ts kernel
struct TsKernelConstants {
    double zp, dzp;
    double xray_prefactor, volunit_inv, lya_star_prefactor, Nb_zp;
    double Trad, Trad_inv, Ts_prefactor, xa_tilde_prefactor, xc_inverse;
    double dcomp_dzp_prefactor, hubble_zp, N_zp;
    double growth_zp, dgrowth_dzp, dt_dzp;
    double growth_factor_zp, inverse_growth_factor_z;
    double No, N_b0, H_FRAC, HE_FRAC;
    double CLUMPING_FACTOR;
    double A10, c_cms, lambda_21, k_B, h_p, T_21, m_p;
    double MAX_TK;
    bool use_x_ray_heating;
    bool use_mini_halos;
};

__global__ void compute_spin_temperature_kernel(
    TsKernelConstants c,
    unsigned long long num_pixels,
    // Input: accumulated spectral integration results
    double *d_dxheat_dt_box,
    double *d_dxion_source_dt_box,
    double *d_dxlya_dt_box,
    double *d_dstarlya_dt_box,
    double *d_dstarlyLW_dt_box,
    // Input: density and previous spin temp
    float *d_density,
    float *d_prev_spin_temperature,
    float *d_prev_kinetic_temp,
    float *d_prev_xray_ionised_fraction,
    // Output
    float *d_spin_temperature,
    float *d_kinetic_temp,
    float *d_xray_ionised_fraction,
    float *d_J_21_LW
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    double curr_delta = (double)d_density[idx] * c.growth_factor_zp * c.inverse_growth_factor_z;
    if (curr_delta <= -1.0) curr_delta = -1.0 + 1.0e-7;

    // Apply prefactors to accumulated boxes
    double dxheat_dt = 0.0;
    if (c.use_x_ray_heating && d_dxheat_dt_box != NULL)
        dxheat_dt = d_dxheat_dt_box[idx] * c.xray_prefactor * c.volunit_inv;

    double dxion_dt = d_dxion_source_dt_box[idx] * c.xray_prefactor * c.volunit_inv;
    double dxlya_dt = d_dxlya_dt_box[idx] * c.xray_prefactor * c.volunit_inv *
                      c.Nb_zp * (1.0 + curr_delta);
    double dstarlya_dt = d_dstarlya_dt_box[idx] * c.lya_star_prefactor * c.volunit_inv;

    double prev_Ts = (double)d_prev_spin_temperature[idx];
    double prev_Tk = (double)d_prev_kinetic_temp[idx];
    double prev_xe = (double)d_prev_xray_ionised_fraction[idx];

    // === get_Ts_fast logic ===

    double tau21 = (3.0 * c.h_p * c.A10 * c.c_cms * c.lambda_21 * c.lambda_21 /
                    32.0 / M_PI / c.k_B) *
                   ((1.0 - prev_xe) * c.N_zp) / prev_Ts / c.hubble_zp;
    double xCMB = (1.0 - exp(-tau21)) / tau21;

    // Electron density sink
    double dxion_sink_dt = alpha_A_gpu(prev_Tk) * c.CLUMPING_FACTOR * prev_xe * prev_xe *
                           c.H_FRAC * c.Nb_zp * (1.0 + curr_delta);
    double dxe_dzp = c.dt_dzp * (dxion_dt - dxion_sink_dt);

    // Adiabatic
    double dadia_dzp = 3.0 / (1.0 + c.zp);
    if (fabs(curr_delta) > 1.0e-7)
        dadia_dzp += c.dgrowth_dzp / (c.growth_zp * (1.0 / curr_delta + 1.0));
    dadia_dzp *= (2.0 / 3.0) * prev_Tk;

    // Species change
    double dspec_dzp = -dxe_dzp * prev_Tk / (1.0 + prev_xe);

    // Compton
    double dcomp_dzp = c.dcomp_dzp_prefactor * (prev_xe / (1.0 + prev_xe + c.HE_FRAC)) *
                       (c.Trad - prev_Tk);

    // X-ray heating
    double dxheat_dzp = 0.0;
    if (c.use_x_ray_heating)
        dxheat_dzp = dxheat_dt * c.dt_dzp * 2.0 / 3.0 / c.k_B / (1.0 + prev_xe);

    // Update x_e and Tk
    double x_e = prev_xe + dxe_dzp * c.dzp;
    if (x_e > 1.0) x_e = 1.0 - 1.0e-7;
    else if (x_e < 0.0) x_e = 0.0;

    double Tk = prev_Tk;
    if (Tk < c.MAX_TK)
        Tk += (dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp) * c.dzp;
    if (Tk < 0.0) Tk = c.Trad;

    // Spin temperature
    double T_inv = 1.0 / Tk;
    double T_inv_sq = T_inv * T_inv;

    double xc_fast = (1.0 + curr_delta) * c.xc_inverse *
                     ((1.0 - x_e) * c.No * kappa_10_gpu(Tk) +
                      x_e * c.N_b0 * kappa_10_elec_gpu(Tk) +
                      x_e * c.No * kappa_10_pH_gpu(Tk));

    double xi_power = c.Ts_prefactor * cbrt((1.0 + curr_delta) * (1.0 - x_e) * T_inv_sq);

    double J_alpha_tot = dstarlya_dt + dxlya_dt;
    double xa_tilde_fast_arg = c.xa_tilde_prefactor * J_alpha_tot *
                               pow(1.0 + 2.98394 * xi_power + 1.53583 * xi_power * xi_power +
                                       3.85289 * xi_power * xi_power * xi_power,
                                   -1.0);

    double TS_fast;
    if (J_alpha_tot > 1.0e-20) {
        // WF effect: iterative
        TS_fast = c.Trad;
        double TSold_fast = 0.0;
        double xa_tilde_fast;
        for (int iter = 0; iter < 100; iter++) {  // safety limit
            TSold_fast = TS_fast;
            xa_tilde_fast =
                (1.0 - 0.0631789 * T_inv + 0.115995 * T_inv_sq -
                 0.401403 * T_inv / TS_fast + 0.336463 * T_inv_sq / TS_fast) *
                xa_tilde_fast_arg;
            TS_fast = (xCMB + xa_tilde_fast + xc_fast) /
                      (xCMB * c.Trad_inv +
                       xa_tilde_fast * (T_inv + 0.405535 * T_inv / TS_fast -
                                        0.405535 * T_inv_sq) +
                       xc_fast * T_inv);
            if (fabs(TS_fast - TSold_fast) / TS_fast <= 1.0e-3) break;
        }
    } else {
        // Collisions only
        TS_fast = (xCMB + xc_fast) / (xCMB * c.Trad_inv + xc_fast * T_inv);
    }
    TS_fast = fabs(TS_fast);

    // Write outputs
    d_spin_temperature[idx] = (float)TS_fast;
    d_kinetic_temp[idx] = (float)Tk;
    d_xray_ionised_fraction[idx] = (float)x_e;
    if (c.use_mini_halos && d_J_21_LW != NULL && d_dstarlyLW_dt_box != NULL) {
        double dstarLW_dt = d_dstarlyLW_dt_box[idx] * c.lya_star_prefactor *
                            c.volunit_inv * c.h_p * 1e21;
        d_J_21_LW[idx] = (float)dstarLW_dt;
    }
}

// Launch the spin temperature kernel.
// Expects accumulated dxdt arrays on host (will upload them).
extern "C" void launch_spin_temperature_kernel(
    unsigned long long num_pixels,
    // Constants
    float zp, float dzp,
    double xray_prefactor, double volunit_inv, double lya_star_prefactor, double Nb_zp,
    double Trad, double Trad_inv, double Ts_prefactor, double xa_tilde_prefactor,
    double xc_inverse, double dcomp_dzp_prefactor, double hubble_zp, double N_zp,
    double growth_zp, double dgrowth_dzp, double dt_dzp,
    double growth_factor_zp, double inverse_growth_factor_z,
    double No_val, double N_b0_val, double H_FRAC_val, double HE_FRAC_val,
    double CLUMPING_FACTOR,
    double A10, double c_cms, double lambda_21, double k_B, double h_p, double T_21, double m_p,
    bool use_x_ray_heating, bool use_mini_halos,
    // Host arrays: accumulated spectral integration
    double *dxheat_dt_box, double *dxion_source_dt_box,
    double *dxlya_dt_box, double *dstarlya_dt_box, double *dstarlyLW_dt_box,
    // Host arrays: density and previous spin temp
    float *density, float *prev_spin_temperature,
    float *prev_kinetic_temp, float *prev_xray_ionised_fraction,
    // Host arrays: output
    float *spin_temperature, float *kinetic_temp,
    float *xray_ionised_fraction, float *J_21_LW
) {
    size_t pix_d = num_pixels * sizeof(double);
    size_t pix_f = num_pixels * sizeof(float);

    // Upload inputs
    double *d_dxheat = NULL, *d_dxion = NULL, *d_dxlya = NULL, *d_dstarlya = NULL;
    double *d_dstarlyLW = NULL;
    float *d_dens = NULL, *d_prev_Ts = NULL, *d_prev_Tk = NULL, *d_prev_xe = NULL;
    float *d_out_Ts = NULL, *d_out_Tk = NULL, *d_out_xe = NULL, *d_out_LW = NULL;

    if (use_x_ray_heating && dxheat_dt_box) {
        CALL_CUDA(cudaMalloc(&d_dxheat, pix_d));
        CALL_CUDA(cudaMemcpy(d_dxheat, dxheat_dt_box, pix_d, cudaMemcpyHostToDevice));
    }
    CALL_CUDA(cudaMalloc(&d_dxion, pix_d));
    CALL_CUDA(cudaMemcpy(d_dxion, dxion_source_dt_box, pix_d, cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc(&d_dxlya, pix_d));
    CALL_CUDA(cudaMemcpy(d_dxlya, dxlya_dt_box, pix_d, cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc(&d_dstarlya, pix_d));
    CALL_CUDA(cudaMemcpy(d_dstarlya, dstarlya_dt_box, pix_d, cudaMemcpyHostToDevice));
    if (use_mini_halos && dstarlyLW_dt_box) {
        CALL_CUDA(cudaMalloc(&d_dstarlyLW, pix_d));
        CALL_CUDA(cudaMemcpy(d_dstarlyLW, dstarlyLW_dt_box, pix_d, cudaMemcpyHostToDevice));
    }

    CALL_CUDA(cudaMalloc(&d_dens, pix_f));
    CALL_CUDA(cudaMemcpy(d_dens, density, pix_f, cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc(&d_prev_Ts, pix_f));
    CALL_CUDA(cudaMemcpy(d_prev_Ts, prev_spin_temperature, pix_f, cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc(&d_prev_Tk, pix_f));
    CALL_CUDA(cudaMemcpy(d_prev_Tk, prev_kinetic_temp, pix_f, cudaMemcpyHostToDevice));
    CALL_CUDA(cudaMalloc(&d_prev_xe, pix_f));
    CALL_CUDA(cudaMemcpy(d_prev_xe, prev_xray_ionised_fraction, pix_f, cudaMemcpyHostToDevice));

    // Allocate outputs
    CALL_CUDA(cudaMalloc(&d_out_Ts, pix_f));
    CALL_CUDA(cudaMalloc(&d_out_Tk, pix_f));
    CALL_CUDA(cudaMalloc(&d_out_xe, pix_f));
    if (use_mini_halos) {
        CALL_CUDA(cudaMalloc(&d_out_LW, pix_f));
    }

    // Build constants struct
    TsKernelConstants c;
    c.zp = zp; c.dzp = dzp;
    c.xray_prefactor = xray_prefactor; c.volunit_inv = volunit_inv;
    c.lya_star_prefactor = lya_star_prefactor; c.Nb_zp = Nb_zp;
    c.Trad = Trad; c.Trad_inv = Trad_inv;
    c.Ts_prefactor = Ts_prefactor; c.xa_tilde_prefactor = xa_tilde_prefactor;
    c.xc_inverse = xc_inverse; c.dcomp_dzp_prefactor = dcomp_dzp_prefactor;
    c.hubble_zp = hubble_zp; c.N_zp = N_zp;
    c.growth_zp = growth_zp; c.dgrowth_dzp = dgrowth_dzp; c.dt_dzp = dt_dzp;
    c.growth_factor_zp = growth_factor_zp; c.inverse_growth_factor_z = inverse_growth_factor_z;
    c.No = No_val; c.N_b0 = N_b0_val; c.H_FRAC = H_FRAC_val; c.HE_FRAC = HE_FRAC_val;
    c.CLUMPING_FACTOR = CLUMPING_FACTOR;
    c.A10 = A10; c.c_cms = c_cms; c.lambda_21 = lambda_21; c.k_B = k_B;
    c.h_p = h_p; c.T_21 = T_21; c.m_p = m_p;
    c.MAX_TK = 5e4;
    c.use_x_ray_heating = use_x_ray_heating;
    c.use_mini_halos = use_mini_halos;

    unsigned int threads = 256;
    unsigned int blocks = (unsigned int)((num_pixels + threads - 1) / threads);

    compute_spin_temperature_kernel<<<blocks, threads>>>(
        c, num_pixels,
        d_dxheat, d_dxion, d_dxlya, d_dstarlya, d_dstarlyLW,
        d_dens, d_prev_Ts, d_prev_Tk, d_prev_xe,
        d_out_Ts, d_out_Tk, d_out_xe, d_out_LW
    );
    CALL_CUDA(cudaGetLastError());
    CALL_CUDA(cudaDeviceSynchronize());

    // Download results
    CALL_CUDA(cudaMemcpy(spin_temperature, d_out_Ts, pix_f, cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(kinetic_temp, d_out_Tk, pix_f, cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(xray_ionised_fraction, d_out_xe, pix_f, cudaMemcpyDeviceToHost));
    if (use_mini_halos && d_out_LW)
        CALL_CUDA(cudaMemcpy(J_21_LW, d_out_LW, pix_f, cudaMemcpyDeviceToHost));

    // Free device memory
    if (d_dxheat) CALL_CUDA(cudaFree(d_dxheat));
    CALL_CUDA(cudaFree(d_dxion));
    CALL_CUDA(cudaFree(d_dxlya));
    CALL_CUDA(cudaFree(d_dstarlya));
    if (d_dstarlyLW) CALL_CUDA(cudaFree(d_dstarlyLW));
    CALL_CUDA(cudaFree(d_dens));
    CALL_CUDA(cudaFree(d_prev_Ts));
    CALL_CUDA(cudaFree(d_prev_Tk));
    CALL_CUDA(cudaFree(d_prev_xe));
    CALL_CUDA(cudaFree(d_out_Ts));
    CALL_CUDA(cudaFree(d_out_Tk));
    CALL_CUDA(cudaFree(d_out_xe));
    if (d_out_LW) CALL_CUDA(cudaFree(d_out_LW));

    LOG_INFO("Spin temperature GPU kernel complete");
}

// ============================================================================
// Phase 11.6a: Spectral integration GPU kernel
// ============================================================================
//
// Replaces the OpenMP per-pixel loop in ts_main that accumulates
// dxheat_dt_box, dxion_source_dt_box, dxlya_dt_box, dstarlya_dt_box
// (and optionally dstarlyLW_dt_box, dstarlya_cont_dt_box, dstarlya_inj_dt_box)
// from frequency integral table lookups per R iteration.
//
// The host manages the sequential R-loop; this kernel handles pixel parallelism.

// Flattened frequency integral tables on device.
// Layout: tbl[xidx * max_n_step + R_ct], where xidx in [0, x_int_NXHII)
// and R_ct in [0, N_STEP_TS).
struct SpectralIntegDeviceData {
    // Frequency integral tables (flattened [x_int_NXHII * N_STEP_TS])
    double *d_freq_int_heat_tbl;
    double *d_freq_int_ion_tbl;
    double *d_freq_int_lya_tbl;
    double *d_freq_int_heat_tbl_diff;
    double *d_freq_int_ion_tbl_diff;
    double *d_freq_int_lya_tbl_diff;

    // Per-pixel precomputed arrays
    int *d_m_xHII_low_box;
    float *d_inverse_val_box;

    // Per-pixel input: del_fcoll_Rct (uploaded per R for SOURCE_MODEL < 2)
    float *d_del_fcoll_Rct;
    float *d_del_fcoll_Rct_MINI;

    // Accumulation arrays (persistent across R iterations)
    double *d_dxheat_dt_box;
    double *d_dxion_source_dt_box;
    double *d_dxlya_dt_box;
    double *d_dstarlya_dt_box;
    double *d_dstarlyLW_dt_box;
    double *d_dstarlya_cont_dt_box;
    double *d_dstarlya_inj_dt_box;

    // Dimensions
    unsigned long long num_pixels;
    int n_step_ts;

    // Feature flags (copied from host options at init time)
    bool use_mini_halos;
    bool use_x_ray_heating;
    bool use_lya_heating;
};

__global__ void spectral_integration_kernel(
    int R_ct,
    int n_step_ts,
    unsigned long long num_pixels,
    // Per-pixel input
    float *d_del_fcoll_Rct,
    float *d_del_fcoll_Rct_MINI,
    int *d_m_xHII_low_box,
    float *d_inverse_val_box,
    // Frequency integral tables (flattened [x_int_NXHII * n_step_ts])
    double *d_freq_int_heat_tbl,
    double *d_freq_int_ion_tbl,
    double *d_freq_int_lya_tbl,
    double *d_freq_int_heat_tbl_diff,
    double *d_freq_int_ion_tbl_diff,
    double *d_freq_int_lya_tbl_diff,
    // Accumulation arrays (output, +=)
    double *d_dxheat_dt_box,
    double *d_dxion_source_dt_box,
    double *d_dxlya_dt_box,
    double *d_dstarlya_dt_box,
    double *d_dstarlyLW_dt_box,
    double *d_dstarlya_cont_dt_box,
    double *d_dstarlya_inj_dt_box,
    // R-dependent scalars (passed per kernel call)
    double z_edge_factor,
    double xray_R_factor,
    double avg_fix_term,
    double avg_fix_term_MINI,
    double F_STAR10,
    double L_X,
    double s_per_yr,
    double F_STAR7_MINI,
    double L_X_MINI,
    double dstarlya_dt_prefactor_R,
    double dstarlyLW_dt_prefactor_R,
    double dstarlyLW_dt_prefactor_MINI_R,
    double dstarlya_dt_prefactor_MINI_R,
    double dstarlya_cont_dt_prefactor_R,
    double dstarlya_inj_dt_prefactor_R,
    double dstarlya_cont_dt_prefactor_MINI_R,
    double dstarlya_inj_dt_prefactor_MINI_R,
    // Feature flags
    bool use_mini_halos,
    bool use_x_ray_heating,
    bool use_lya_heating
) {
    unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    float fcoll = d_del_fcoll_Rct[idx];
    double sfr_term = (double)fcoll * z_edge_factor * avg_fix_term * F_STAR10;
    double xray_sfr = sfr_term * L_X * xray_R_factor * s_per_yr;

    double sfr_term_mini = 0.0;
    if (use_mini_halos) {
        sfr_term_mini = (double)d_del_fcoll_Rct_MINI[idx] * z_edge_factor *
                        avg_fix_term_MINI * F_STAR7_MINI;
        xray_sfr += sfr_term_mini * L_X_MINI * xray_R_factor * s_per_yr;

        d_dstarlyLW_dt_box[idx] +=
            sfr_term * dstarlyLW_dt_prefactor_R +
            sfr_term_mini * dstarlyLW_dt_prefactor_MINI_R;
    }

    int xidx = d_m_xHII_low_box[idx];
    double ival = (double)d_inverse_val_box[idx];

    // Table index: flattened [xidx][R_ct] -> xidx * n_step_ts + R_ct
    int tbl_idx = xidx * n_step_ts + R_ct;

    if (use_x_ray_heating) {
        d_dxheat_dt_box[idx] +=
            xray_sfr * (d_freq_int_heat_tbl_diff[tbl_idx] * ival +
                        d_freq_int_heat_tbl[tbl_idx]);
    }

    d_dxion_source_dt_box[idx] +=
        xray_sfr * (d_freq_int_ion_tbl_diff[tbl_idx] * ival +
                    d_freq_int_ion_tbl[tbl_idx]);

    d_dxlya_dt_box[idx] +=
        xray_sfr * (d_freq_int_lya_tbl_diff[tbl_idx] * ival +
                    d_freq_int_lya_tbl[tbl_idx]);

    d_dstarlya_dt_box[idx] +=
        sfr_term * dstarlya_dt_prefactor_R +
        sfr_term_mini * dstarlya_dt_prefactor_MINI_R;

    if (use_lya_heating) {
        d_dstarlya_cont_dt_box[idx] +=
            sfr_term * dstarlya_cont_dt_prefactor_R +
            sfr_term_mini * dstarlya_cont_dt_prefactor_MINI_R;
        d_dstarlya_inj_dt_box[idx] +=
            sfr_term * dstarlya_inj_dt_prefactor_R +
            sfr_term_mini * dstarlya_inj_dt_prefactor_MINI_R;
    }
}

// Initialise device data for spectral integration.
// Flattens 2D tables [x_int_NXHII][N_STEP_TS] into 1D device arrays.
// Allocates persistent accumulation buffers.
extern "C" SpectralIntegDeviceData *init_spectral_integration_gpu(
    unsigned long long num_pixels,
    int n_step_ts,
    // Host 2D tables (double*[x_int_NXHII], each row has n_step_ts entries)
    double **freq_int_heat_tbl,
    double **freq_int_ion_tbl,
    double **freq_int_lya_tbl,
    double **freq_int_heat_tbl_diff,
    double **freq_int_ion_tbl_diff,
    double **freq_int_lya_tbl_diff,
    // Host per-pixel arrays
    int *m_xHII_low_box,
    float *inverse_val_box,
    // Feature flags
    bool use_mini_halos,
    bool use_x_ray_heating,
    bool use_lya_heating
) {
    SpectralIntegDeviceData *dev = (SpectralIntegDeviceData *)malloc(sizeof(SpectralIntegDeviceData));
    dev->num_pixels = num_pixels;
    dev->n_step_ts = n_step_ts;
    dev->use_mini_halos = use_mini_halos;
    dev->use_x_ray_heating = use_x_ray_heating;
    dev->use_lya_heating = use_lya_heating;

    int nxhii = 14;  // x_int_NXHII
    size_t tbl_size = nxhii * n_step_ts * sizeof(double);
    size_t pix_size_d = num_pixels * sizeof(double);
    size_t pix_size_f = num_pixels * sizeof(float);
    size_t pix_size_i = num_pixels * sizeof(int);

    // Flatten and upload frequency integral tables
    double *h_flat = (double *)malloc(tbl_size);

    auto upload_table = [&](double **host_tbl, double **d_tbl) {
        for (int i = 0; i < nxhii; i++)
            memcpy(h_flat + i * n_step_ts, host_tbl[i], n_step_ts * sizeof(double));
        CALL_CUDA(cudaMalloc(d_tbl, tbl_size));
        CALL_CUDA(cudaMemcpy(*d_tbl, h_flat, tbl_size, cudaMemcpyHostToDevice));
    };

    upload_table(freq_int_heat_tbl, &dev->d_freq_int_heat_tbl);
    upload_table(freq_int_ion_tbl, &dev->d_freq_int_ion_tbl);
    upload_table(freq_int_lya_tbl, &dev->d_freq_int_lya_tbl);
    upload_table(freq_int_heat_tbl_diff, &dev->d_freq_int_heat_tbl_diff);
    upload_table(freq_int_ion_tbl_diff, &dev->d_freq_int_ion_tbl_diff);
    upload_table(freq_int_lya_tbl_diff, &dev->d_freq_int_lya_tbl_diff);
    free(h_flat);

    // Upload per-pixel index arrays
    CALL_CUDA(cudaMalloc(&dev->d_m_xHII_low_box, pix_size_i));
    CALL_CUDA(cudaMemcpy(dev->d_m_xHII_low_box, m_xHII_low_box, pix_size_i, cudaMemcpyHostToDevice));

    CALL_CUDA(cudaMalloc(&dev->d_inverse_val_box, pix_size_f));
    CALL_CUDA(cudaMemcpy(dev->d_inverse_val_box, inverse_val_box, pix_size_f, cudaMemcpyHostToDevice));

    // Allocate per-pixel input buffers (filled per R)
    CALL_CUDA(cudaMalloc(&dev->d_del_fcoll_Rct, pix_size_f));
    dev->d_del_fcoll_Rct_MINI = NULL;
    if (use_mini_halos) {
        CALL_CUDA(cudaMalloc(&dev->d_del_fcoll_Rct_MINI, pix_size_f));
    }

    // Allocate accumulation arrays and zero them
    if (use_x_ray_heating) {
        CALL_CUDA(cudaMalloc(&dev->d_dxheat_dt_box, pix_size_d));
        CALL_CUDA(cudaMemset(dev->d_dxheat_dt_box, 0, pix_size_d));
    } else {
        dev->d_dxheat_dt_box = NULL;
    }

    CALL_CUDA(cudaMalloc(&dev->d_dxion_source_dt_box, pix_size_d));
    CALL_CUDA(cudaMemset(dev->d_dxion_source_dt_box, 0, pix_size_d));

    CALL_CUDA(cudaMalloc(&dev->d_dxlya_dt_box, pix_size_d));
    CALL_CUDA(cudaMemset(dev->d_dxlya_dt_box, 0, pix_size_d));

    CALL_CUDA(cudaMalloc(&dev->d_dstarlya_dt_box, pix_size_d));
    CALL_CUDA(cudaMemset(dev->d_dstarlya_dt_box, 0, pix_size_d));

    dev->d_dstarlyLW_dt_box = NULL;
    if (use_mini_halos) {
        CALL_CUDA(cudaMalloc(&dev->d_dstarlyLW_dt_box, pix_size_d));
        CALL_CUDA(cudaMemset(dev->d_dstarlyLW_dt_box, 0, pix_size_d));
    }

    dev->d_dstarlya_cont_dt_box = NULL;
    dev->d_dstarlya_inj_dt_box = NULL;
    if (use_lya_heating) {
        CALL_CUDA(cudaMalloc(&dev->d_dstarlya_cont_dt_box, pix_size_d));
        CALL_CUDA(cudaMemset(dev->d_dstarlya_cont_dt_box, 0, pix_size_d));
        CALL_CUDA(cudaMalloc(&dev->d_dstarlya_inj_dt_box, pix_size_d));
        CALL_CUDA(cudaMemset(dev->d_dstarlya_inj_dt_box, 0, pix_size_d));
    }

    LOG_INFO("Spectral integration GPU data initialised (%llu pixels, %d R steps)",
             num_pixels, n_step_ts);
    return dev;
}

// Download accumulated results from device to host arrays.
extern "C" void download_spectral_integration_results(
    SpectralIntegDeviceData *dev,
    double *dxheat_dt_box,
    double *dxion_source_dt_box,
    double *dxlya_dt_box,
    double *dstarlya_dt_box,
    double *dstarlyLW_dt_box,
    double *dstarlya_cont_dt_box,
    double *dstarlya_inj_dt_box
) {
    size_t pix_size_d = dev->num_pixels * sizeof(double);

    if (dev->use_x_ray_heating && dev->d_dxheat_dt_box)
        CALL_CUDA(cudaMemcpy(dxheat_dt_box, dev->d_dxheat_dt_box, pix_size_d, cudaMemcpyDeviceToHost));

    CALL_CUDA(cudaMemcpy(dxion_source_dt_box, dev->d_dxion_source_dt_box, pix_size_d, cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(dxlya_dt_box, dev->d_dxlya_dt_box, pix_size_d, cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(dstarlya_dt_box, dev->d_dstarlya_dt_box, pix_size_d, cudaMemcpyDeviceToHost));

    if (dev->use_mini_halos && dev->d_dstarlyLW_dt_box)
        CALL_CUDA(cudaMemcpy(dstarlyLW_dt_box, dev->d_dstarlyLW_dt_box, pix_size_d, cudaMemcpyDeviceToHost));

    if (dev->use_lya_heating) {
        if (dev->d_dstarlya_cont_dt_box)
            CALL_CUDA(cudaMemcpy(dstarlya_cont_dt_box, dev->d_dstarlya_cont_dt_box, pix_size_d, cudaMemcpyDeviceToHost));
        if (dev->d_dstarlya_inj_dt_box)
            CALL_CUDA(cudaMemcpy(dstarlya_inj_dt_box, dev->d_dstarlya_inj_dt_box, pix_size_d, cudaMemcpyDeviceToHost));
    }

    LOG_INFO("Spectral integration results downloaded from GPU");
}

// Free all device memory for spectral integration.
extern "C" void free_spectral_integration_gpu(SpectralIntegDeviceData *dev) {
    CALL_CUDA(cudaFree(dev->d_freq_int_heat_tbl));
    CALL_CUDA(cudaFree(dev->d_freq_int_ion_tbl));
    CALL_CUDA(cudaFree(dev->d_freq_int_lya_tbl));
    CALL_CUDA(cudaFree(dev->d_freq_int_heat_tbl_diff));
    CALL_CUDA(cudaFree(dev->d_freq_int_ion_tbl_diff));
    CALL_CUDA(cudaFree(dev->d_freq_int_lya_tbl_diff));

    CALL_CUDA(cudaFree(dev->d_m_xHII_low_box));
    CALL_CUDA(cudaFree(dev->d_inverse_val_box));
    CALL_CUDA(cudaFree(dev->d_del_fcoll_Rct));
    if (dev->d_del_fcoll_Rct_MINI) CALL_CUDA(cudaFree(dev->d_del_fcoll_Rct_MINI));

    if (dev->d_dxheat_dt_box) CALL_CUDA(cudaFree(dev->d_dxheat_dt_box));
    CALL_CUDA(cudaFree(dev->d_dxion_source_dt_box));
    CALL_CUDA(cudaFree(dev->d_dxlya_dt_box));
    CALL_CUDA(cudaFree(dev->d_dstarlya_dt_box));
    if (dev->d_dstarlyLW_dt_box) CALL_CUDA(cudaFree(dev->d_dstarlyLW_dt_box));
    if (dev->d_dstarlya_cont_dt_box) CALL_CUDA(cudaFree(dev->d_dstarlya_cont_dt_box));
    if (dev->d_dstarlya_inj_dt_box) CALL_CUDA(cudaFree(dev->d_dstarlya_inj_dt_box));

    LOG_INFO("Spectral integration GPU data freed");
}

// Launch the spectral integration kernel for one R iteration.
// del_fcoll_Rct is the host array for the current R (uploaded H2D per call).
extern "C" void launch_spectral_integration_kernel(
    SpectralIntegDeviceData *dev,
    int R_ct,
    float *del_fcoll_Rct,
    float *del_fcoll_Rct_MINI,
    double z_edge_factor,
    double xray_R_factor,
    double avg_fix_term,
    double avg_fix_term_MINI,
    double F_STAR10,
    double L_X,
    double s_per_yr,
    double F_STAR7_MINI,
    double L_X_MINI,
    double dstarlya_dt_prefactor_R,
    double dstarlyLW_dt_prefactor_R,
    double dstarlyLW_dt_prefactor_MINI_R,
    double dstarlya_dt_prefactor_MINI_R,
    double dstarlya_cont_dt_prefactor_R,
    double dstarlya_inj_dt_prefactor_R,
    double dstarlya_cont_dt_prefactor_MINI_R,
    double dstarlya_inj_dt_prefactor_MINI_R
) {
    size_t pix_size_f = dev->num_pixels * sizeof(float);

    // Upload del_fcoll for this R
    CALL_CUDA(cudaMemcpy(dev->d_del_fcoll_Rct, del_fcoll_Rct, pix_size_f, cudaMemcpyHostToDevice));
    if (dev->use_mini_halos && del_fcoll_Rct_MINI) {
        CALL_CUDA(cudaMemcpy(dev->d_del_fcoll_Rct_MINI, del_fcoll_Rct_MINI, pix_size_f, cudaMemcpyHostToDevice));
    }

    unsigned int threads = 256;
    unsigned int blocks = (unsigned int)((dev->num_pixels + threads - 1) / threads);

    spectral_integration_kernel<<<blocks, threads>>>(
        R_ct,
        dev->n_step_ts,
        dev->num_pixels,
        dev->d_del_fcoll_Rct,
        dev->d_del_fcoll_Rct_MINI,
        dev->d_m_xHII_low_box,
        dev->d_inverse_val_box,
        dev->d_freq_int_heat_tbl,
        dev->d_freq_int_ion_tbl,
        dev->d_freq_int_lya_tbl,
        dev->d_freq_int_heat_tbl_diff,
        dev->d_freq_int_ion_tbl_diff,
        dev->d_freq_int_lya_tbl_diff,
        dev->d_dxheat_dt_box,
        dev->d_dxion_source_dt_box,
        dev->d_dxlya_dt_box,
        dev->d_dstarlya_dt_box,
        dev->d_dstarlyLW_dt_box,
        dev->d_dstarlya_cont_dt_box,
        dev->d_dstarlya_inj_dt_box,
        z_edge_factor,
        xray_R_factor,
        avg_fix_term,
        avg_fix_term_MINI,
        F_STAR10,
        L_X,
        s_per_yr,
        F_STAR7_MINI,
        L_X_MINI,
        dstarlya_dt_prefactor_R,
        dstarlyLW_dt_prefactor_R,
        dstarlyLW_dt_prefactor_MINI_R,
        dstarlya_dt_prefactor_MINI_R,
        dstarlya_cont_dt_prefactor_R,
        dstarlya_inj_dt_prefactor_R,
        dstarlya_cont_dt_prefactor_MINI_R,
        dstarlya_inj_dt_prefactor_MINI_R,
        dev->use_mini_halos,
        dev->use_x_ray_heating,
        dev->use_lya_heating
    );
    CALL_CUDA(cudaGetLastError());
}
