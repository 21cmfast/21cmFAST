#include <math.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <curand.h> // host-side header file
#include <curand_kernel.h> // device-side header file

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <iostream>

#include "Constants.h"
#include "interpolation_types.h"
#include "Stochasticity.h"

// #include "tiger_checks.h"
#include "cuda_utils.cuh"
#include "Stochasticity.cuh"
#include "DeviceConstants.cuh"
#include "hmf.cu"
#include "interp_tables.cu"




#ifndef MAX_DELTAC_FRAC
#define MAX_DELTAC_FRAC (float)0.99 // max delta/deltac for the mass function integrals
#endif

#ifndef DELTA_MIN
#define DELTA_MIN -1 // minimum delta for Lagrangian mass function integrals
#endif

#ifndef MAX_HALO_CELL
#define MAX_HALO_CELL (int)1e5
#endif

void validate_thrust()
{
    // Create a host vector with some values
    thrust::host_vector<int> h_vec(5);
    h_vec[0] = 1;
    h_vec[1] = 2;
    h_vec[2] = 3;
    h_vec[3] = 4;
    h_vec[4] = 5;

    // Transfer data from host to device
    thrust::device_vector<int> d_vec = h_vec;

    // Calculate the sum of all elements in the device vector
    int sum = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

    // Print the result
    std::cout << "Sum is: " << sum << std::endl; // Should print "Sum is: 15"
}

void condense_device_vector()
{
    // Step 1: Create a device vector with some elements, including -1
    thrust::device_vector<int> d_vec(10);
    d_vec[0] = 1;
    d_vec[1] = -1;
    d_vec[2] = 3;
    d_vec[3] = -1;
    d_vec[4] = 5;
    d_vec[5] = 6;
    d_vec[6] = -1;
    d_vec[7] = 7;
    d_vec[8] = -1;
    d_vec[9] = 9;

    // Step 2: Use thrust::remove_if to remove all occurrences of -1
    thrust::device_vector<int>::iterator new_end = thrust::remove(d_vec.begin(), d_vec.end(), -1);

    // Step 3: Resize the vector to remove the trailing elements after the "new_end" iterator
    d_vec.erase(new_end, d_vec.end());

    // Step 4: Copy the result to the host to check
    thrust::host_vector<int> h_vec = d_vec;

    // Step 5: Print the result
    std::cout << "Condensed Vector: ";
    for (size_t i = 0; i < h_vec.size(); i++)
    {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;
}

int condenseDeviceArray(int *d_array, int original_size)
{
    // Wrap the raw device pointer into a thrust device pointer
    thrust::device_ptr<int> d_array_ptr(d_array);

    // Remove elements with value 0
    // thrust::device_vector<int>::iterator new_end = thrust::remove(d_array_ptr, d_array_ptr + original_size, 0);
    // thrust::device_ptr<int> new_end = thrust::remove(d_array_ptr, d_array_ptr + original_size, 0);
    auto new_end = thrust::remove(d_array_ptr, d_array_ptr + original_size, 0);

    // Calculate the number of valid elements
    int valid_size = new_end - d_array_ptr;

    // Print results (on host side)
    std::cout << "Valid elements count: " << valid_size << "\n";
    return valid_size;
}

int filterWithMask(float *d_data, int *d_mask, int original_size)
{
    // Wrap the raw pointers into thrust device pointers
    thrust::device_ptr<float> d_data_ptr(d_data);
    thrust::device_ptr<int> d_mask_ptr(d_mask);

    // Use the mask to select only elements that correspond to a value of 1 in the mask
    auto end = thrust::copy_if(d_data_ptr, d_data_ptr + original_size, d_mask_ptr, d_data_ptr, thrust::identity<float>());

    // Calculate the new valid size after filtering
    int valid_size = end - d_data_ptr;

    // Optionally, print the number of valid elements
    std::cout << "Valid elements count: " << valid_size << "\n";

    return valid_size;
}

// 11-30: the following implementation works (before using any global params on gpu)
__device__ void stoc_set_consts_cond(struct HaloSamplingConstants *const_struct, float cond_val, int HMF, double x_min, double x_width, float *d_y_arr, int n_bin)
{
    double m_exp, n_exp;
    // Here the condition is a mass, volume is the Lagrangian volume and delta_l is set by the
    // redshift difference which represents the difference in delta_crit across redshifts
    if (const_struct->from_catalog){
        const_struct->M_cond = cond_val;
        const_struct->lnM_cond = log(cond_val);
        const_struct->sigma_cond = EvaluateSigma(const_struct->lnM_cond, x_min, x_width, d_y_arr, n_bin); //todo: update this function using global tables in constant memory
        // mean stellar mass of this halo mass, used for stellar z correlations
        const_struct->cond_val = const_struct->lnM_cond;
        // condition delta is the previous delta crit
        const_struct->delta = get_delta_crit(HMF, const_struct->sigma_cond, const_struct->growth_in) / const_struct->growth_in * const_struct->growth_out; //todo: update this function using global variables in constant memory
    }
    // Here the condition is a cell of a given density, the volume/mass is given by the grid parameters
    else
    {
        // since the condition mass/sigma is already set all we need is delta
        const_struct->delta = cond_val;
        const_struct->cond_val = cond_val;
    }
    // Get expected N and M from interptables
    // the splines don't work well for cells above Deltac, but there CAN be cells above deltac, since this calculation happens
    // before the overlap, and since the smallest dexm mass is M_cell*(1.01^3) there *could* be a cell above Deltac not in a halo
    // NOTE: all this does is prevent integration errors below since these cases are also dealt with in stoc_sample
    if (const_struct->delta > MAX_DELTAC_FRAC * get_delta_crit(d_user_params.HMF, const_struct->sigma_cond, const_struct->growth_out)){
        const_struct->expected_M = const_struct->M_cond;
        const_struct->expected_N = 1;
    }
    else if (const_struct->delta <= DELTA_MIN){
        const_struct->expected_M = 0;
        const_struct->expected_N = 0;
    }
    else
    {
        n_exp = EvaluateNhalo(const_struct->cond_val, const_struct->growth_out, const_struct->lnM_min,
                              const_struct->lnM_max_tb, const_struct->M_cond, const_struct->sigma_cond, const_struct->delta);
        m_exp = EvaluateMcoll(const_struct->cond_val, const_struct->growth_out, const_struct->lnM_min,
                              const_struct->lnM_max_tb, const_struct->M_cond, const_struct->sigma_cond, const_struct->delta);
        const_struct->expected_N = n_exp * const_struct->M_cond;
        const_struct->expected_M = m_exp * const_struct->M_cond;
    }
    return;
}

__device__ double sample_dndM_inverse(double condition, struct HaloSamplingConstants *hs_constants, curandState *state)
{
    double p_in, result;
    p_in = curand_uniform_double(state);
    result = EvaluateNhaloInv(condition, p_in);
    result = fmin(1.0, fmax(0.0, result)); // clip in case of extrapolation
    result = result * hs_constants->M_cond;
    return result;
}

__device__ double remove_random_halo(curandState *state, int n_halo, int *idx, double *M_prog, float *M_out){
    double last_M_del;
    int random_idx;
    do {
        random_idx = (int)(curand_uniform(state) * n_halo);
    } while (M_out[random_idx] == 0);
    last_M_del = M_out[random_idx];
    *M_prog -= last_M_del;
    M_out[random_idx] = 0; // zero mass halos are skipped and not counted

    *idx = random_idx;
    return last_M_del;
}

__device__ void fix_mass_sample(curandState *state, double exp_M, int *n_halo_pt, double *M_tot_pt, float *M_out){
    // Keep the last halo if it brings us closer to the expected mass
    // This is done by addition or subtraction over the limit to balance
    // the bias of the last halo being larger
    int random_idx;
    double last_M_del;
    int sel = curand(state) % 2;
    // bool sel = gsl_rng_uniform_int(rng, 2);
    // int sel = 1;
    if (sel)
    {
        if (fabs(*M_tot_pt - M_out[*n_halo_pt - 1] - exp_M) < fabs(*M_tot_pt - exp_M))
        {
            *M_tot_pt -= M_out[*n_halo_pt - 1];
            // here we remove by setting the counter one lower so it isn't read
            (*n_halo_pt)--; // increment has preference over dereference
        }
    }
    else
    {
        do {
            // here we remove by setting halo mass to zero, skipping it during the consolidation
            last_M_del = remove_random_halo(state, *n_halo_pt, &random_idx, M_tot_pt, M_out);
        } while (*M_tot_pt > exp_M);

        // if the sample with the last subtracted halo is closer to the expected mass, keep it
        // LOG_ULTRA_DEBUG("Deciding to keep last halo M %.3e tot %.3e exp %.3e",last_M_del,*M_tot_pt,exp_M);
        if (fabs(*M_tot_pt + last_M_del - exp_M) < fabs(*M_tot_pt - exp_M))
        {
            M_out[random_idx] = last_M_del;
            *M_tot_pt += last_M_del;
        }
    }
}

__device__ int stoc_mass_sample(struct HaloSamplingConstants *hs_constants, curandState *state, int *n_halo_out, float *M_out, int *further_process){
    double exp_M = hs_constants->expected_M;

    // The mass-limited sampling as-is has a slight bias to producing too many halos,
    //   which is independent of density or halo mass,
    //   this factor reduces the total expected mass to bring it into line with the CMF
    // exp_M *= user_params_global->HALOMASS_CORRECTION;
    exp_M *= d_user_params.HALOMASS_CORRECTION;

    int n_halo_sampled = 0;
    double M_prog = 0;
    double M_sample;

    double tbl_arg = hs_constants->cond_val;

    // tmp (start)
    M_sample = sample_dndM_inverse(tbl_arg, hs_constants, state);

    M_prog += M_sample;
    // tmp (end)

    // while (M_prog < exp_M){
    //     M_sample = sample_dndM_inverse(tbl_arg, hs_constants, state);

    //     M_prog += M_sample;
    //     M_out[n_halo_sampled++] = M_sample;
    // }
    // todo: enable fix_mass_sample
    // The above sample is above the expected mass, by up to 100%. I wish to make the average mass equal to exp_M
    // fix_mass_sample(state, exp_M, &n_halo_sampled, &M_prog, M_out);

    *n_halo_out = n_halo_sampled;
    if (M_prog < exp_M){
        *further_process = 1;
        return 1;
    }
    *M_out = M_sample;
    return 0;
}

__device__ int stoc_sample(struct HaloSamplingConstants *hs_constants, curandState *state, int *n_halo_out, float *M_out, int *further_process){
    // TODO: really examine the case for number/mass sampling
    // The poisson sample fails spectacularly for high delta (from_catalogs or dense cells)
    //   and excludes the correlation between number and mass (e.g many small halos or few large ones)
    // The mass sample underperforms at low exp_M/M_max by excluding stochasticity in the total collapsed fraction
    //   and excluding larger halos (e.g if exp_M is 0.1*M_max we can effectively never sample the large halos)
    // i.e there is some case for a delta cut between these two methods however I have no intuition for the exact levels

    int err;

    // If the expected mass is below our minimum saved mass, don't bother calculating
    // NOTE: some of these conditions are redundant with set_consts_cond()
    if (hs_constants->delta <= DELTA_MIN || hs_constants->expected_M < d_user_params.SAMPLER_MIN_MASS)
    {
        *n_halo_out = 0;
        return 0;
    }
    // if delta is above critical, form one big halo
    if (hs_constants->delta >= MAX_DELTAC_FRAC * get_delta_crit(d_user_params.HMF, hs_constants->sigma_cond, hs_constants->growth_out)){
        *n_halo_out = 1;

        // Expected mass takes into account potential dexm overlap
        M_out[0] = hs_constants->expected_M;
        return 0;
    }

    // todo: implement callee functions for SAMPLE_METHOD (1,2,3)
    // We always use Number-Limited sampling for grid-based cases
    if (d_user_params.SAMPLE_METHOD == 1 || !hs_constants->from_catalog)
    {
        // err = stoc_halo_sample(hs_constants, rng, n_halo_out, M_out);
        return 0;
    }
    else if (d_user_params.SAMPLE_METHOD == 0)
    {
        err = stoc_mass_sample(hs_constants, state, n_halo_out, M_out, further_process);
    }
    else if (d_user_params.SAMPLE_METHOD == 2)
    {
        // err = stoc_partition_sample(hs_constants, rng, n_halo_out, M_out);
        return 0;
    }
    else if (d_user_params.SAMPLE_METHOD == 3)
    {
        // err = stoc_split_sample(hs_constants, rng, n_halo_out, M_out);
        return 0;
    }
    else
    {
        printf("Invalid sampling method \n");
        return 0;
        // todo: check how to throw error in cuda
        // LOG_ERROR("Invalid sampling method");
        // Throw(ValueError);
    }
    if (*n_halo_out > MAX_HALO_CELL)
    {
        printf("too many halos in conditin, buffer overflow");
        // todo: check how to throw error in cuda
        // LOG_ERROR("too many halos in condition, buffer overflow");
        // Throw(ValueError);
    }
    return err;
}

// todo: implement condense_sparse_halolist
// // todo: just copied the original function here, need to verify it works with cuda
// __device__ void condense_sparse_halolist(HaloField *halofield, unsigned long long int *istart_threads, unsigned long long int *nhalo_threads)
// {
//     int i = 0;
//     unsigned long long int count_total = 0;
//     for (i = 0; i < user_params_global->N_THREADS; i++)
//     {
//         memmove(&halofield->halo_masses[count_total], &halofield->halo_masses[istart_threads[i]], sizeof(float) * nhalo_threads[i]);
//         memmove(&halofield->star_rng[count_total], &halofield->star_rng[istart_threads[i]], sizeof(float) * nhalo_threads[i]);
//         memmove(&halofield->sfr_rng[count_total], &halofield->sfr_rng[istart_threads[i]], sizeof(float) * nhalo_threads[i]);
//         memmove(&halofield->xray_rng[count_total], &halofield->xray_rng[istart_threads[i]], sizeof(float) * nhalo_threads[i]);
//         memmove(&halofield->halo_coords[3 * count_total], &halofield->halo_coords[3 * istart_threads[i]], sizeof(int) * 3 * nhalo_threads[i]);
//         LOG_SUPER_DEBUG("Moved array (start,count) (%llu, %llu) to position %llu", istart_threads[i], nhalo_threads[i], count_total);
//         count_total += nhalo_threads[i];
//     }
//     halofield->n_halos = count_total;

//     // replace the rest with zeros for clarity
//     memset(&halofield->halo_masses[count_total], 0, (halofield->buffer_size - count_total) * sizeof(float));
//     memset(&halofield->halo_coords[3 * count_total], 0, 3 * (halofield->buffer_size - count_total) * sizeof(int));
//     memset(&halofield->star_rng[count_total], 0, (halofield->buffer_size - count_total) * sizeof(float));
//     memset(&halofield->sfr_rng[count_total], 0, (halofield->buffer_size - count_total) * sizeof(float));
//     memset(&halofield->xray_rng[count_total], 0, (halofield->buffer_size - count_total) * sizeof(float));
//     LOG_SUPER_DEBUG("Set %llu elements beyond %llu to zero", halofield->buffer_size - count_total, count_total);
// }

// todo: implement set_prop_rng
// __device__ void set_prop_rng(gsl_rng *rng, bool from_catalog, double *interp, double *input, double *output)
// {
//     double rng_star, rng_sfr, rng_xray;

//     // Correlate properties by interpolating between the sampled and descendant gaussians
//     rng_star = astro_params_global->SIGMA_STAR > 0. ? gsl_ran_ugaussian(rng) : 0.;
//     rng_sfr = astro_params_global->SIGMA_SFR_LIM > 0. ? gsl_ran_ugaussian(rng) : 0.;
//     rng_xray = astro_params_global->SIGMA_LX > 0. ? gsl_ran_ugaussian(rng) : 0.;

//     if (from_catalog)
//     {
//         // this transforms the sample to one from the multivariate Gaussian, conditioned on the first sample
//         rng_star = sqrt(1 - interp[0] * interp[0]) * rng_star + interp[0] * input[0];
//         rng_sfr = sqrt(1 - interp[1] * interp[1]) * rng_sfr + interp[1] * input[1];
//         rng_xray = sqrt(1 - interp[2] * interp[2]) * rng_xray + interp[2] * input[2];
//     }

//     output[0] = rng_star;
//     output[1] = rng_sfr;
//     output[2] = rng_xray;
//     return;
// }

// kernel function
__global__ void setup_random_states(curandState *d_states, unsigned long long int random_seed){
    // get thread idx
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(random_seed, ind, 0, &d_states[ind]);
}

__global__ void update_halo_constants(float *d_halo_masses, float *d_y_arr, double x_min, double x_width,
                                      unsigned long long int n_halos, int n_bin, struct HaloSamplingConstants d_hs_constants,
                                      int HMF, curandState *d_states, 
                                      float *d_halo_masses_out, float *star_rng_out,
                                      float *sfr_rng_out, float *xray_rng_out, float *halo_coords_out, int *d_sum_check,
                                      int *further_process)
{
    // Define shared memory for block-level reduction
    __shared__ int shared_check[256];
    
    // get thread idx
    int tid = threadIdx.x;
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= n_halos)
    {
        return;
    }

    float M = d_halo_masses[ind];

    int n_prog; // the value will be updated after calling stoc_sample

    // set condition-dependent variables for sampling
    stoc_set_consts_cond(&d_hs_constants, M, HMF, x_min, x_width, d_y_arr, n_bin);

    // tmp: just to verify the tables have been copied correctly
    if (ind == 0)
    {
        printf("The first element of Nhalo y_arr: %e (%e) \n", d_Nhalo_yarr[0], d_Nhalo_table.y_arr[0]);
        printf("The nhalo table n_bin: %d\n", d_Nhalo_table.n_bin);
        printf("The nhalo_inv table nx_bin: %d\n", d_Nhalo_inv_table.nx_bin);
        printf("HII_DIM: %d \n", d_user_params.HII_DIM);
        printf("test params: %f \n", d_test_params);
        printf("A_VCB: %f \n", d_astro_params.A_VCB);
        printf("SIGMA_8: %f \n", d_cosmo_params.SIGMA_8);
    }

    // todo: each thread across different blocks has unique random state
    // curand_init(seed, threadIdx.x, 0, &d_states[threadIdx.x]);
    // curandState local_state = d_states[threadIdx.x];
    curandState local_state = d_states[ind];
    // tmp: for validation only
    // sample_dndM_inverse(0.38, &d_hs_constants, &local_state);
    // int tmp1 = 20;
    // double tmp2 = 681273355217.0;
    // float tmp3 = 101976856.0; 
    // remove_random_halo(&local_state, 59, &tmp1, &tmp2, &tmp3);
    int check = stoc_sample(&d_hs_constants, &local_state, &n_prog, &d_halo_masses_out[ind], &further_process[ind]);
    d_states[ind] = local_state;

    shared_check[tid] = check;
    __syncthreads();

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_check[tid] += shared_check[tid + stride];
        }
        __syncthreads(); // Ensure all threads have completed each stage of reduction
    }

    // Write the result from each block to the global sum 
    if (tid == 0)
    {
        atomicAdd(d_sum_check, shared_check[0]);
    }

    // Sample the CMF set by the descendant
    // stoc_sample(&hs_constants, &local_state, &n_prog, prog_buf);

    // double sigma = EvaluateSigma(log(M), x_min, x_width, d_y_arr, n_bin);
    // double delta = get_delta_crit(HMF, sigma, d_hs_constants.growth_in)\
    //                                 / d_hs_constants.growth_in * d_hs_constants.growth_out;

    return;
}

__global__ void update_halo_constants_multi(float *d_halo_masses, float *d_y_arr, double x_min, double x_width,
                                      unsigned long long int n_halos, int n_bin, struct HaloSamplingConstants d_hs_constants,
                                      int HMF, curandState *d_states,
                                      float *d_halo_masses_out, float *star_rng_out,
                                      float *sfr_rng_out, float *xray_rng_out, float *halo_coords_out, int *d_sum_check,
                                      int *further_process)
{
    // Define shared memory for block-level reduction
    __shared__ int shared_check[256];

    // get thread idx
    int tid = threadIdx.x;
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= n_halos)
    {
        return;
    }

    float M = d_halo_masses[ind];

    int n_prog; // the value will be updated after calling stoc_sample

    // set condition-dependent variables for sampling
    stoc_set_consts_cond(&d_hs_constants, M, HMF, x_min, x_width, d_y_arr, n_bin);

    // todo: each thread across different blocks has unique random state
    curandState local_state = d_states[ind];
    int check = stoc_sample(&d_hs_constants, &local_state, &n_prog, &d_halo_masses_out[ind], &further_process[ind]);
    d_states[ind] = local_state;

    shared_check[tid] = check;
    __syncthreads();

    // Perform reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        if (tid < stride)
        {
            shared_check[tid] += shared_check[tid + stride];
        }
        __syncthreads(); // Ensure all threads have completed each stage of reduction
    }

    // Write the result from each block to the global sum
    if (tid == 0)
    {
        atomicAdd(d_sum_check, shared_check[0]);
    }

    return;
}

// function to launch kernel grids
int updateHaloOut(float *halo_masses, unsigned long long int n_halos, float *y_arr, int n_bin_y, double x_min, double x_width,
                  struct HaloSamplingConstants hs_constants, unsigned long long int n_buffer)
{
    // allocate memory and copy halo_masses to the device
    size_t size_halo = sizeof(float) * n_halos;
    float *d_halo_masses;
    CALL_CUDA(cudaMalloc(&d_halo_masses, size_halo));
    CALL_CUDA(cudaMemcpy(d_halo_masses, halo_masses, size_halo, cudaMemcpyHostToDevice));

    // allocate memory and copy y_arr of sigma_table to the device
    size_t size_yarr = sizeof(float) * n_bin_y;
    float *d_y_arr;
    CALL_CUDA(cudaMalloc(&d_y_arr, size_yarr));
    CALL_CUDA(cudaMemcpy(d_y_arr, y_arr, size_yarr, cudaMemcpyHostToDevice));

    // allocate memory for d_check_sum (tmp)
    int *d_sum_check;
    CALL_CUDA(cudaMalloc((void **)&d_sum_check, sizeof(int)));
    CALL_CUDA(cudaMemset(d_sum_check, 0, sizeof(int)));

    // allocate memory for out halos
    size_t buffer_size = sizeof(float) * n_buffer * 2;
    float *d_halo_masses_out;
    CALL_CUDA(cudaMalloc(&d_halo_masses_out, buffer_size));

    float *star_rng_out;
    CALL_CUDA(cudaMalloc(&star_rng_out, buffer_size));

    float *sfr_rng_out;
    CALL_CUDA(cudaMalloc(&sfr_rng_out, buffer_size));

    float *xray_rng_out;
    CALL_CUDA(cudaMalloc(&xray_rng_out, buffer_size));

    float *halo_coords_out;
    CALL_CUDA(cudaMalloc(&halo_coords_out, buffer_size * 3));

    // allocate memory to store list of halo index need further process
    int *d_further_process;
    CALL_CUDA(cudaMalloc(&d_further_process, size_halo));

    // get parameters needed by the kernel
    int HMF = user_params_global->HMF;

    // define threads layout
    int n_threads = 256;
    int n_blocks = (int)((n_halos + 255) / 256);
    int total_threads = n_threads * n_blocks;

    // Allocate memory for RNG states
    curandState *d_states;
    CALL_CUDA(cudaMalloc((void **)&d_states, total_threads * sizeof(curandState)));

    // setup random states
    setup_random_states<<<n_blocks, n_threads>>>(d_states, 1234ULL);

    // Check kernel launch errors
    CALL_CUDA(cudaGetLastError());

    // launch kernel grid
    update_halo_constants<<<n_blocks, n_threads>>>(d_halo_masses, d_y_arr, x_min, x_width, n_halos, n_bin_y, hs_constants, HMF, d_states, d_halo_masses_out, star_rng_out,
                                                       sfr_rng_out, xray_rng_out, halo_coords_out, d_sum_check, d_further_process);

    // Check kernel launch errors
    CALL_CUDA(cudaGetLastError());

    CALL_CUDA(cudaDeviceSynchronize());

    // filtered halos
    int n_filter_halo = filterWithMask(d_halo_masses, d_further_process, n_halos);
    float *h_filter_halos;
    CALL_CUDA(cudaHostAlloc((void **)&h_filter_halos, sizeof(float)*n_filter_halo, cudaHostAllocDefault));
    CALL_CUDA(cudaMemcpy(h_filter_halos, d_halo_masses, sizeof(float)*n_filter_halo, cudaMemcpyDeviceToHost));

    // launch second kernel
    


    // copy data from device to host
    int h_sum_check;
    CALL_CUDA(cudaMemcpy(&h_sum_check, d_sum_check, sizeof(int), cudaMemcpyDeviceToHost));

    float *h_halo_masses_out;
    CALL_CUDA(cudaHostAlloc((void **)&h_halo_masses_out, buffer_size, cudaHostAllocDefault));
    CALL_CUDA(cudaMemcpy(h_halo_masses_out, d_halo_masses_out, buffer_size, cudaMemcpyDeviceToHost));

    // Free device memory
    CALL_CUDA(cudaFree(d_halo_masses));
    CALL_CUDA(cudaFree(d_y_arr));
    CALL_CUDA(cudaFree(d_states));
    CALL_CUDA(cudaFree(d_halo_masses_out));
    CALL_CUDA(cudaFree(star_rng_out));
    CALL_CUDA(cudaFree(sfr_rng_out));
    CALL_CUDA(cudaFree(xray_rng_out));
    CALL_CUDA(cudaFree(halo_coords_out));
    CALL_CUDA(cudaFree(d_further_process));

    validate_thrust();

    condense_device_vector();

    return 0;
}
