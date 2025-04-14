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
#include <thrust/fill.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cassert>

#include "Constants.h"
#include "interpolation_types.h"
#include "Stochasticity.h"

#include "cuda_utils.cuh"
#include "Stochasticity.cuh"
#include "DeviceConstants.cuh"
#include "device_rng.cuh"
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

// int condenseDeviceArray(float *d_array, int original_size, float mask_value)
// {
//     // Wrap the raw device pointer into a thrust device pointer
//     thrust::device_ptr<float> d_array_ptr(d_array);

//     // Remove elements with mask value 
//     // i.e.move elements not equal to mask value to the beginning of the array without changing order
//     auto new_end = thrust::remove(d_array_ptr, d_array_ptr + original_size, mask_value);

//     // Calculate the number of valid elements
//     int valid_size = new_end - d_array_ptr;

//     // Fill the remaining space with mask value
//     thrust::fill(new_end, d_array_ptr + original_size, mask_value);

//     // Print results (on host side)
//     // std::cout << "Valid elements count: " << valid_size << "\n";
//     return valid_size;
// }

template <typename T>
int condenseDeviceArray(T *d_array, int original_size, T mask_value)
{
    // Wrap the raw device pointer into a thrust device pointer
    thrust::device_ptr<T> d_array_ptr(d_array);

    // Remove elements with mask value
    auto new_end = thrust::remove(d_array_ptr, d_array_ptr + original_size, mask_value);

    // Calculate the number of valid elements
    int valid_size = new_end - d_array_ptr;

    // Fill the remaining space with mask value
    thrust::fill(new_end, d_array_ptr + original_size, mask_value);

    return valid_size;
}

void testCondenseDeviceArray()
{
    // Input data
    float h_array[] = {1.0f, 0.0f, 2.0f, 3.0f, 0.0f, 4.0f};
    float mask_value = 0.0f;
    int original_size = 6;

    // Expected outputs
    float expected_array[] = {1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f};
    int expected_valid_size = 4;

    // Allocate and copy to device
    float *d_array;
    cudaMalloc(&d_array, original_size * sizeof(float));
    cudaMemcpy(d_array, h_array, original_size * sizeof(float), cudaMemcpyHostToDevice);

    // Call the function from Stochasticity.cu
    int valid_size = condenseDeviceArray(d_array, original_size, mask_value);

    // Copy the results back to the host
    float h_result[original_size];
    cudaMemcpy(h_result, d_array, original_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate the results
    assert(valid_size == expected_valid_size);
    for (int i = 0; i < original_size; ++i)
    {
        assert(h_result[i] == expected_array[i]);
    }

    std::cout << "Test passed: condenseDeviceArray\n";

    // Free device memory
    cudaFree(d_array);
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
    // std::cout << "Valid elements count: " << valid_size << "\n";

    return valid_size;
}

void testFilterWithMask()
{
    // Input arrays
    float h_data[] = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f}; // Input data
    int h_mask[] = {1, 0, 1, 0, 1};                  // Mask array
    int original_size = 5;

    // Expected outputs
    float expected_data[] = {1.1f, 3.3f, 5.5f}; // Expected filtered data
    int expected_size = 3;                      // Number of valid elements

    // Allocate device memory
    float *d_data;
    int *d_mask;
    cudaMalloc(&d_data, original_size * sizeof(float));
    cudaMalloc(&d_mask, original_size * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_data, h_data, original_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, original_size * sizeof(int), cudaMemcpyHostToDevice);

    // Call the function
    int valid_size = filterWithMask(d_data, d_mask, original_size);

    // Copy the filtered data back to host
    float h_result[original_size];
    cudaMemcpy(h_result, d_data, original_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate the size of the filtered array
    assert(valid_size == expected_size);

    // Validate the filtered elements
    for (int i = 0; i < valid_size; ++i)
    {
        assert(h_result[i] == expected_data[i]);
    }

    // Print success message
    std::cout << "Test passed: filterWithMask\n";

    // Free device memory
    cudaFree(d_data);
    cudaFree(d_mask);
}

void countElements(const int *array, int size, const std::vector<int> &values_to_count)
{
    // Initialize a frequency array to count occurrences
    int count[values_to_count.size()] = {0};

    // Iterate through the input array
    for (int i = 0; i < size; ++i)
    {
        // Find the index of the value in values_to_count
        for (size_t j = 0; j < values_to_count.size(); ++j)
        {
            if (array[i] == values_to_count[j])
            {
                count[j]++;
                break;
            }
        }
    }

    // Print the results
    for (size_t i = 0; i < values_to_count.size(); ++i)
    {
        std::cout << "Value " << values_to_count[i] << ": " << count[i] << " occurrences\n";
    }
}

// decide the number of sparsity
int getSparsity(int n_buffer, int n_halo){
    if (n_halo > 0){
        int power = floor(log2(n_buffer / n_halo));
        int sparsity = 1 << power;
        return sparsity;
    }

}

// initialize device array with given value
void initializeArray(int *d_array, int n_elements, int value){
    thrust::device_ptr<int> d_array_ptr(d_array);
    thrust::fill(d_array_ptr, d_array_ptr + n_elements, value);
}

// void getKernelAttr(){
//     cudaFuncAttributes attr;
//     cudaFuncGetAttributes(&attr, myKernel);
//     printf("Kernel Shared Memory per Block: %zu bytes\n", attr.sharedSizeBytes);
//     printf("Kernel Registers per Thread: %d\n", attr.numRegs);
//     printf("Kernel Max Threads per Block: %d\n", attr.maxThreadsPerBlock);
// }

struct GridLayout{
    int n_threads;
    int n_blocks;
};
// calculate workload 
// todo: add more checks on sparsity
GridLayout getWorkload(int sparsity, unsigned long long int n_halos){
    GridLayout res;
    int n_threads, n_blocks;
    if (sparsity != 0 && 256 % sparsity == 0){
        n_threads = 256;
    }
    else {
        n_threads = std::min(sparsity,512);
    }
    res.n_threads = n_threads;
    n_blocks = (n_halos * sparsity + n_threads -1)/n_threads;
    res.n_blocks = n_blocks;
    return res;
}

// 11-30: the following implementation works (before using any global params on gpu)
__device__ void stoc_set_consts_cond(struct HaloSamplingConstants *const_struct, float cond_val, int HMF, double x_min, double x_width, float *d_y_arr, int n_bin, double *expected_mass)
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
    *expected_mass = const_struct->expected_M;
    return;
}

__device__ double sample_dndM_inverse(double condition, struct HaloSamplingConstants *hs_constants, curandState *state)
{
    double p_in, result;
    p_in = curand_uniform_double(state);
    // printf("curand uniform random number: %f\n", p_in);
    result = EvaluateNhaloInv(condition, p_in);
    result = fmin(1.0, fmax(0.0, result)); // clip in case of extrapolation
    result = result * hs_constants->M_cond;
    return result;
}

__device__ double remove_random_halo(curandState *state, int n_halo, int *idx, float *M_prog, float *M_out){
    double last_M_del;
    int random_idx;
    do {
        random_idx = (int)(curand_uniform(state) * n_halo);
    } while (M_out[random_idx] == 0.0f);
    last_M_del = M_out[random_idx];
    *M_prog -= last_M_del;
    M_out[random_idx] = 0.0f; // -1 mass halos are skipped and not counted

    *idx = random_idx;
    return last_M_del;
}

__device__ void fix_mass_sample(curandState *state, double exp_M, float *M_prog, float *M_out, int write_limit, int *n_prog){
    // Keep the last halo if it brings us closer to the expected mass
    // This is done by addition or subtraction over the limit to balance
    // the bias of the last halo being larger
    int random_idx;
    double last_M_del;
    int sel = curand(state) % 2;
    // int sel = 1; //tmp: implement the first case 
    if (sel)
    {
        if (fabs(*M_prog - M_out[write_limit] - exp_M) < fabs(*M_prog - exp_M))
        {
            // *M_tot_pt -= M_out[*n_halo_pt - 1];
            // here we remove by setting the counter one lower so it isn't read
            M_out[write_limit] = 0.0f;
            (*n_prog)--;
        }
    }
    else
    {
        do {
            // here we remove by setting halo mass to zero, skipping it during the consolidation
            last_M_del = remove_random_halo(state, write_limit+1, &random_idx, M_prog, M_out);
        } while (*M_prog > exp_M);

        // if the sample with the last subtracted halo is closer to the expected mass, keep it
        // LOG_ULTRA_DEBUG("Deciding to keep last halo M %.3e tot %.3e exp %.3e",last_M_del,*M_tot_pt,exp_M);
        if (fabs(*M_prog + last_M_del - exp_M) < fabs(*M_prog - exp_M))
        {
            M_out[random_idx] = last_M_del;
            *M_prog += last_M_del;
        }
        
    }
}

__device__ int stoc_mass_sample(struct HaloSamplingConstants *hs_constants, curandState *state, float *M_out){
    double exp_M = hs_constants->expected_M;

    // The mass-limited sampling as-is has a slight bias to producing too many halos,
    //   which is independent of density or halo mass,
    //   this factor reduces the total expected mass to bring it into line with the CMF
    // exp_M *= user_params_global->HALOMASS_CORRECTION;
    exp_M *= d_user_params.HALOMASS_CORRECTION;

    // int n_halo_sampled = 0;
    // double M_prog = 0;
    // double M_sample;

    double tbl_arg = hs_constants->cond_val;

    // tmp (start)
    double M_sample = sample_dndM_inverse(tbl_arg, hs_constants, state);

    // M_prog += M_sample;
    // tmp (end)

    // while (M_prog < exp_M){
    //     M_sample = sample_dndM_inverse(tbl_arg, hs_constants, state);

    //     M_prog += M_sample;
    //     M_out[n_halo_sampled++] = M_sample;
    // }
    // todo: enable fix_mass_sample
    // The above sample is above the expected mass, by up to 100%. I wish to make the average mass equal to exp_M
    // fix_mass_sample(state, exp_M, &n_halo_sampled, &M_prog, M_out);

    // *n_halo_out = n_halo_sampled;
    // if (M_prog < exp_M){
    //     *further_process = 1;
    //     return 1;
    // }
    *M_out = M_sample;
    return 0;
}

__device__ int stoc_sample(struct HaloSamplingConstants *hs_constants, curandState *state, float *M_out, int *sampleCondition){
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
        // *n_halo_out = 0;
        *sampleCondition = 0;
        return 0;
    }
    // if delta is above critical, form one big halo
    if (hs_constants->delta >= MAX_DELTAC_FRAC * get_delta_crit(d_user_params.HMF, hs_constants->sigma_cond, hs_constants->growth_out)){
        // *n_halo_out = 1;

        // Expected mass takes into account potential dexm overlap
        *M_out = hs_constants->expected_M;
        *sampleCondition = 1;
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
        err = stoc_mass_sample(hs_constants, state, M_out);
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
    // if (*n_halo_out > MAX_HALO_CELL)
    // {
    //     printf("too many halos in conditin, buffer overflow\n");
    //     // todo: check how to throw error in cuda
    //     // LOG_ERROR("too many halos in condition, buffer overflow");
    //     // Throw(ValueError);
    // }
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

__device__ void set_prop_rng(curandState *state, bool from_catalog, double *interp, float *input, float *output)
{
    float rng_star, rng_sfr, rng_xray;

    // Correlate properties by interpolating between the sampled and descendant gaussians
    rng_star = d_astro_params.SIGMA_STAR > 0. ? curand_normal(state) : 0.;
    rng_sfr = d_astro_params.SIGMA_SFR_LIM > 0. ? curand_normal(state) : 0.;
    rng_xray = d_astro_params.SIGMA_LX > 0. ? curand_normal(state) : 0.;

    if (from_catalog)
    {
        // this transforms the sample to one from the multivariate Gaussian, conditioned on the first sample
        rng_star = sqrt(1 - interp[0] * interp[0]) * rng_star + interp[0] * input[0];
        rng_sfr = sqrt(1 - interp[1] * interp[1]) * rng_sfr + interp[1] * input[1];
        rng_xray = sqrt(1 - interp[2] * interp[2]) * rng_xray + interp[2] * input[2];
    }

    output[0] = rng_star;
    output[1] = rng_sfr;
    output[2] = rng_xray;
    return;
}

__global__ void update_halo_constants(float *d_halo_masses, float *d_star_rng_in, float *d_sfr_rng_in, float *d_xray_rng_in, 
                                      int *d_halo_coords_in, float *d_y_arr, double x_min, double x_width,
                                      unsigned long long int n_halos, int n_bin, struct HaloSamplingConstants d_hs_constants,
                                      int HMF,
                                      float *d_halo_masses_out, float *d_star_rng_out,
                                      float *d_sfr_rng_out, float *d_xray_rng_out, int *d_halo_coords_out, int *d_sum_check,
                                      int *d_further_process, int *d_nprog_predict, int sparsity, unsigned long long int write_offset, 
                                      double *expected_mass, int *d_n_prog, int offset_shared)
{
    // Define shared memory for block-level reduction
    extern __shared__ float shared_memory[];
    // __shared__ float shared_mass[256];

    // partition shared memory
    float *shared_mass = shared_memory;
    float *shared_prop_rng = shared_memory + offset_shared;

    // get local thread idx
    int tid = threadIdx.x;

    // initialize shared_mass
    shared_mass[tid] = 0.0f;

    // initialize shared_prop_rng
    for (int i=0;i<3;i++){
        shared_prop_rng[tid+i*offset_shared] = 0.0f;
    }
    

    // get global thread idx
    int ind = blockIdx.x * blockDim.x + threadIdx.x;
    
    // get halo idx
    int hid = ind / sparsity;
    if (hid >= n_halos)
    {
        // printf("Out of halo range.\n");
        return;
    }

    // get halo mass
    float M = d_halo_masses[hid];

    // get stoc properties from in halo
    float prop_in[3] = {d_star_rng_in[hid], d_sfr_rng_in[hid], d_xray_rng_in[hid]};

    // get correction
    double corr_arr[3] = {d_hs_constants.corr_star, d_hs_constants.corr_sfr, d_hs_constants.corr_xray};

    // get coordinate
    int coords_in[3] = {d_halo_coords_in[hid*3], d_halo_coords_in[hid*3+1], d_halo_coords_in[hid*3+2]};

    // idx of d_halo_masses_out and other halo field arrays
    int out_id = write_offset + ind;

    // set condition-dependent variables for sampling
    stoc_set_consts_cond(&d_hs_constants, M, HMF, x_min, x_width, d_y_arr, n_bin, &expected_mass[hid]);
    // if (hid == 1){
    //     printf("check here. \n");
    // }

    // if (hid == 2){
    //     printf("check here. \n");
    // }

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
        // printf("number of rng states: %d\n", g_numRNGStates);
        // // tiger tmp: debug (start)
        // double res1, res2, res3, res4;
        // res1 = EvaluateNhaloInv(18.694414138793945, 0.0046723012881037529);
        // printf("tmp res1 on gpu: %.17f \n", res1);
        // res2 = EvaluateNhaloInv(20.084152221679688, 0.32153863360286256);
        // printf("tmp res2 on gpu: %.17f \n", res2);
        // res3 = EvaluateNhaloInv(26.806314468383789, 0.8698794976081996);
        // printf("tmp res3 on gpu: %.17f \n", res3);
        // res4 = EvaluateNhaloInv(19.00053596496582, 0.83130413049947305);
        // printf("tmp res4 on gpu: %.17f \n", res4);
        // // tiger tmp: debug (end)
    }
    
    curandState local_state = d_randStates[ind];
    // if (blockIdx.x > 100000){
    //     // printf("check here. \n");
    // }
    // tmp: for validation only
    // sample_dndM_inverse(0.38, &d_hs_constants, &local_state);
    // int tmp1 = 20;
    // double tmp2 = 681273355217.0;
    // float tmp3 = 101976856.0;
    // remove_random_halo(&local_state, 59, &tmp1, &tmp2, &tmp3);

    // check sample condition
    // condition 0: no sampling; condition 1: use expected_M; condition 2: sampling
    int sampleCondition = 2;
    stoc_sample(&d_hs_constants, &local_state, &shared_mass[tid], &sampleCondition);

    // get stochastic halo properties
    set_prop_rng(&local_state, true, corr_arr, prop_in, &shared_prop_rng[tid*3]);

    

    __syncthreads();

    if (tid % sparsity == 0){
        if (sampleCondition == 0){
            d_n_prog[hid] = 0;
        }
        if (sampleCondition == 1){
            if(shared_mass[tid] >= d_user_params.SAMPLER_MIN_MASS){
                d_halo_masses_out[out_id] = shared_mass[tid];
                d_n_prog[hid] = 1;
                d_star_rng_out[out_id] = shared_prop_rng[3 * tid];
                d_sfr_rng_out[out_id] = shared_prop_rng[3 * tid + 1];
                d_xray_rng_out[out_id] = shared_prop_rng[3 * tid + 2];
                d_halo_coords_out[out_id*3] = coords_in[0];
                d_halo_coords_out[out_id*3+1] = coords_in[1];
                d_halo_coords_out[out_id*3+2] = coords_in[2];

            }
        }
        if (sampleCondition == 2){
            float Mprog = 0.0;
            int write_limit = 0;
            int meetCondition = 0;

            for (int i = 0; i < sparsity; ++i){
                Mprog += shared_mass[tid + i];
                if (Mprog >= d_hs_constants.expected_M)
                {
                    write_limit = i;
                    meetCondition = 1;
                    break;
                }
                }
            
            if (meetCondition){
                // correct the mass samples
                int n_prog = write_limit +1;
                
                fix_mass_sample(&local_state, d_hs_constants.expected_M, &Mprog, &shared_mass[tid], write_limit, &n_prog);

                // record number of progenitors
                d_n_prog[hid] = min(100,n_prog);

                for (int i = 0; i < write_limit + 1; ++i)
                {
                    if(shared_mass[tid + i] < d_user_params.SAMPLER_MIN_MASS) continue;
                    // write the final mass sample to array in global memory
                    d_halo_masses_out[out_id + i] = shared_mass[tid + i];
                    d_star_rng_out[out_id + i] = shared_prop_rng[3*(tid +i)];
                    d_sfr_rng_out[out_id + i] = shared_prop_rng[3*(tid+i) + 1];
                    d_xray_rng_out[out_id + i] = shared_prop_rng[3*(tid+i) + 2];
                    d_halo_coords_out[(out_id+i) * 3] = coords_in[0];
                    d_halo_coords_out[(out_id+i) * 3 + 1] = coords_in[1];
                    d_halo_coords_out[(out_id+i) * 3 + 2] = coords_in[2];
                }
            }
            else{
                d_further_process[hid] = 1;
                d_nprog_predict[hid] = ceil(d_hs_constants.expected_M * sparsity / Mprog);

            }
        }
    }

        // Perform reduction within the block
        // for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
        // {
        //     if (tid < stride)
        //     {
        //         shared_check[tid] += shared_check[tid + stride];
        //     }
        //     __syncthreads(); // Ensure all threads have completed each stage of reduction
        // }

        // Write the result from each block to the global sum
        // if (tid == 0)
        // {
        //     atomicAdd(d_sum_check, shared_check[0]);
        // }

        // Sample the CMF set by the descendant
        // stoc_sample(&hs_constants, &local_state, &n_prog, prog_buf);

        // double sigma = EvaluateSigma(log(M), x_min, x_width, d_y_arr, n_bin);
        // double delta = get_delta_crit(HMF, sigma, d_hs_constants.growth_in)\
    //                                 / d_hs_constants.growth_in * d_hs_constants.growth_out;

    d_randStates[ind] = local_state;
    return;
}

// function to launch kernel grids
int updateHaloOut(float *halo_masses, float *star_rng, float *sfr_rng, float *xray_rng, int *halo_coords,
                  unsigned long long int n_halos, float *y_arr, int n_bin_y, double x_min, double x_width,
                  struct HaloSamplingConstants hs_constants, unsigned long long int n_buffer, HaloField *halofield_out)
{
    // allocate memory and copy halo data to the device (halo in)
    size_t size_halo = sizeof(float) * n_halos;
    float *d_halo_masses;
    CALL_CUDA(cudaMalloc(&d_halo_masses, size_halo));
    CALL_CUDA(cudaMemcpy(d_halo_masses, halo_masses, size_halo, cudaMemcpyHostToDevice));

    float *d_star_rng;
    CALL_CUDA(cudaMalloc(&d_star_rng, size_halo));
    CALL_CUDA(cudaMemcpy(d_star_rng, star_rng, size_halo, cudaMemcpyHostToDevice));

    float *d_sfr_rng;
    CALL_CUDA(cudaMalloc(&d_sfr_rng, size_halo));
    CALL_CUDA(cudaMemcpy(d_sfr_rng, sfr_rng, size_halo, cudaMemcpyHostToDevice));

    float *d_xray_rng;
    CALL_CUDA(cudaMalloc(&d_xray_rng, size_halo));
    CALL_CUDA(cudaMemcpy(d_xray_rng, xray_rng, size_halo, cudaMemcpyHostToDevice));

    int *d_halo_coords;
    size_t size_halo_coords = 3 * sizeof(int) * n_halos;
    CALL_CUDA(cudaMalloc(&d_halo_coords, size_halo_coords));
    CALL_CUDA(cudaMemcpy(d_halo_coords, halo_coords, size_halo_coords, cudaMemcpyHostToDevice));

    // allocate memory and copy y_arr of sigma_table to the device
    size_t size_yarr = sizeof(float) * n_bin_y;
    float *d_y_arr;
    CALL_CUDA(cudaMalloc(&d_y_arr, size_yarr));
    CALL_CUDA(cudaMemcpy(d_y_arr, y_arr, size_yarr, cudaMemcpyHostToDevice));

    // allocate memory for d_check_sum (tmp)
    int *d_sum_check;
    CALL_CUDA(cudaMalloc((void **)&d_sum_check, sizeof(int)));
    CALL_CUDA(cudaMemset(d_sum_check, 0, sizeof(int)));

    // allocate memory to store list of halo index need further process
    int *d_further_process;
    CALL_CUDA(cudaMalloc(&d_further_process, sizeof(int)*n_halos));
    CALL_CUDA(cudaMemset(d_further_process, 0, sizeof(int)*n_halos));

    // allocate memory to store number of progenitors per halo
    int *d_n_prog;
    CALL_CUDA(cudaMalloc(&d_n_prog, sizeof(int) * n_halos));
    initializeArray(d_n_prog, n_halos, 32);

    // allocate memory to store estimated n_prog after the first kernel launch
    int *d_nprog_predict;
    CALL_CUDA(cudaMalloc(&d_nprog_predict, sizeof(int) * n_halos));
    CALL_CUDA(cudaMemset(d_nprog_predict, 0, sizeof(int) * n_halos));

    // tmp: check expected_M
    double *d_expected_mass, *h_expected_mass;
    CALL_CUDA(cudaMalloc(&d_expected_mass, sizeof(double) * n_halos));
    CALL_CUDA(cudaMemset(d_expected_mass, 0, sizeof(double) * n_halos));
    CALL_CUDA(cudaHostAlloc((void **)&h_expected_mass, sizeof(double) * n_halos, cudaHostAllocDefault));

    // get parameters needed by the kernel
    int HMF = user_params_global->HMF;

    // set buffer size (hard-coded)
    int scale = 5;
    size_t d_n_buffer = n_halos * scale;
    size_t buffer_size = sizeof(float) * d_n_buffer;

    // allocate memory for out halos (just allocate once at each call of this grid launch function)
    float *d_halo_masses_out;
    CALL_CUDA(cudaMalloc(&d_halo_masses_out, buffer_size));
    CALL_CUDA(cudaMemset(d_halo_masses_out, 0, buffer_size));
    // initializeArray(d_halo_masses_out, d_n_buffer, -1.2f);

    float *d_star_rng_out;
    CALL_CUDA(cudaMalloc(&d_star_rng_out, buffer_size));
    CALL_CUDA(cudaMemset(d_star_rng_out, 0, buffer_size));
    // initializeArray(d_halo_masses_out, d_n_buffer, -1.2f);

    float *d_sfr_rng_out;
    CALL_CUDA(cudaMalloc(&d_sfr_rng_out, buffer_size));
    CALL_CUDA(cudaMemset(d_sfr_rng_out, 0, buffer_size));

    float *d_xray_rng_out;
    CALL_CUDA(cudaMalloc(&d_xray_rng_out, buffer_size));
    CALL_CUDA(cudaMemset(d_xray_rng_out, 0, buffer_size));

    int *d_halo_coords_out;
    CALL_CUDA(cudaMalloc(&d_halo_coords_out, sizeof(int) * d_n_buffer * 3));
    initializeArray(d_halo_coords_out, d_n_buffer * 3, -1000);

    // initiate n_halo check
    unsigned long long int n_halo_check = n_halos;

    // initiate offset for writing output data
    unsigned long long int write_offset = 0;

    // initialize n filter halo
    unsigned long long int n_halos_tbp = n_halos;

    // initialize number of progenitors processed
    unsigned long long int n_processed_prog;

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, update_halo_constants);
    printf("Kernel Shared Memory per Block: %zu bytes\n", attr.sharedSizeBytes);
    printf("Kernel Registers per Thread: %d\n", attr.numRegs);
    printf("Kernel Max Threads per Block: %d\n", attr.maxThreadsPerBlock);

    // start with 4 threads work with one halo
    int sparsity = 4;

    // Check if sparsity is smaller than scale
    if (sparsity >= scale)
    {
        throw std::runtime_error("'sparsity' must be smaller than 'scale'.");
    }

    // initial kernel grid
    GridLayout grids = getWorkload(sparsity, n_halos);

    // launch kernel grid
    while (n_halos_tbp > 0){
    size_t shared_size = grids.n_threads * sizeof(float) * 4;
    int offset_shared = grids.n_threads;
    printf("start launching kernel function.\n");
    update_halo_constants<<<grids.n_blocks, grids.n_threads, shared_size>>>(d_halo_masses, d_star_rng, d_sfr_rng, d_xray_rng, d_halo_coords,
                                                       d_y_arr, x_min, x_width, n_halos_tbp, n_bin_y, hs_constants, HMF, d_halo_masses_out, d_star_rng_out,
                                                       d_sfr_rng_out, d_xray_rng_out, d_halo_coords_out, d_sum_check, d_further_process, d_nprog_predict, sparsity, write_offset, d_expected_mass, 
                                                       d_n_prog, offset_shared);

    // Check kernel launch errors
    CALL_CUDA(cudaGetLastError());

    CALL_CUDA(cudaDeviceSynchronize());

    // filter device halo masses in-place
    n_halos_tbp = filterWithMask(d_halo_masses, d_further_process, n_halos_tbp);
    printf("The number of halos for further processing: %d \n", n_halos_tbp);

    // // tmp 2025-01-19: check d_halo_masses_out writing out 
    // float *h_halo_masses_out_check;
    // CALL_CUDA(cudaHostAlloc((void **)&h_halo_masses_out_check, buffer_size, cudaHostAllocDefault));
    // CALL_CUDA(cudaMemcpy(h_halo_masses_out_check, d_halo_masses_out, buffer_size, cudaMemcpyDeviceToHost));

    // number of progenitors per halo
    int *h_n_prog;
    CALL_CUDA(cudaHostAlloc((void **)&h_n_prog, sizeof(int)*n_halos, cudaHostAllocDefault));
    CALL_CUDA(cudaMemcpy(h_n_prog, d_n_prog, sizeof(int)*n_halos, cudaMemcpyDeviceToHost));

    // Values to count
    std::vector<int> values_to_count = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100,32};

    // Count and display occurrences
    countElements(h_n_prog, n_halos, values_to_count);

    // condense halo mass array on the device
    n_processed_prog = condenseDeviceArray(d_halo_masses_out, d_n_buffer, 0.0f);
    printf("The number of progenitors written in out halo field so far: %d \n", n_processed_prog);
    
    // condense other halo field arrays on the device
    unsigned long long int n_processed_star_rng = condenseDeviceArray(d_star_rng_out, d_n_buffer, 0.0f);
    printf("The number of star prop rng written in out halo field so far: %d \n", n_processed_star_rng);

    unsigned long long int n_processed_sfr_rng = condenseDeviceArray(d_sfr_rng_out, d_n_buffer, 0.0f);
    printf("The number of sfr prop rng written in out halo field so far: %d \n", n_processed_sfr_rng);

    unsigned long long int n_processed_xray_rng = condenseDeviceArray(d_xray_rng_out, d_n_buffer, 0.0f);
    printf("The number of xray prop rng written in out halo field so far: %d \n", n_processed_xray_rng);

    unsigned long long int n_processed_coords = condenseDeviceArray(d_halo_coords_out, d_n_buffer*3, -1000);
    printf("The number of halo coords written in out halo field so far: %d \n", n_processed_coords);

    // tmp: the following is just needed for debugging purpose
    // float *h_filter_halos;
    // CALL_CUDA(cudaHostAlloc((void **)&h_filter_halos, sizeof(float) * n_halos_tbp, cudaHostAllocDefault));
    // CALL_CUDA(cudaMemcpy(h_filter_halos, d_halo_masses, sizeof(float) * n_halos_tbp, cudaMemcpyDeviceToHost));

    // int *h_nprog_predict;
    // CALL_CUDA(cudaHostAlloc((void **)&h_nprog_predict, sizeof(int) * n_halos, cudaHostAllocDefault));
    // CALL_CUDA(cudaMemcpy(h_nprog_predict, d_nprog_predict, sizeof(int) * n_halos, cudaMemcpyDeviceToHost));

    if (n_halos_tbp > 0){
        // update sparsity value
        unsigned long long int available_n_buffer = d_n_buffer - n_processed_prog;
        sparsity = getSparsity(available_n_buffer, n_halos_tbp);

        // check max threadblock size
        int device;
        CALL_CUDA(cudaGetDevice(&device));
        cudaDeviceProp deviceProp;
        CALL_CUDA(cudaGetDeviceProperties(&deviceProp, device));
        int max_threads_pb = deviceProp.maxThreadsPerBlock;

        // sparsity should not exceed the max threads per block
        // sparsity = 256;
        sparsity = std::min(sparsity, 512);

        // reset grids layout
        grids = getWorkload(sparsity, n_halos_tbp);

        // update write offset
        write_offset = n_processed_prog;

        // reset mask array
        CALL_CUDA(cudaMemset(d_further_process, 0, sizeof(int) * n_halos));
        
        // copy data from device to host
        int h_sum_check;
        CALL_CUDA(cudaMemcpy(&h_sum_check, d_sum_check, sizeof(int), cudaMemcpyDeviceToHost));
    }
    // tmp: for debug only
    // CALL_CUDA(cudaFreeHost(h_filter_halos));
    // CALL_CUDA(cudaFreeHost(h_sum_check));

    }

    // write data back to the host
    halofield_out->n_halos = n_processed_prog;
    size_t out_size = sizeof(float) * n_processed_prog;
    
    // float *h_halo_masses_out;
    // CALL_CUDA(cudaHostAlloc((void **)&h_halo_masses_out, out_size, cudaHostAllocDefault));
    CALL_CUDA(cudaGetLastError());
    CALL_CUDA(cudaDeviceSynchronize());

    CALL_CUDA(cudaMemcpy(halofield_out->halo_masses, d_halo_masses_out, out_size, cudaMemcpyDeviceToHost));

    
    CALL_CUDA(cudaMemcpy(halofield_out->star_rng, d_star_rng_out, out_size, cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(halofield_out->sfr_rng, d_sfr_rng_out, out_size, cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpy(halofield_out->xray_rng, d_xray_rng_out, out_size, cudaMemcpyDeviceToHost));

    size_t out_coords_size = sizeof(int) * n_processed_prog * 3;
    CALL_CUDA(cudaMemcpy(halofield_out->halo_coords, d_halo_coords_out, out_coords_size, cudaMemcpyDeviceToHost));

    
    // Free device memory
    CALL_CUDA(cudaFree(d_halo_masses));
    CALL_CUDA(cudaFree(d_y_arr));
    CALL_CUDA(cudaFree(d_halo_masses_out));
    CALL_CUDA(cudaFree(d_star_rng_out));
    CALL_CUDA(cudaFree(d_sfr_rng_out));
    CALL_CUDA(cudaFree(d_xray_rng_out));
    CALL_CUDA(cudaFree(d_halo_coords_out));
    CALL_CUDA(cudaFree(d_further_process));

    validate_thrust();

    condense_device_vector();

    testCondenseDeviceArray();

    testFilterWithMask();

    CALL_CUDA(cudaGetLastError());
    CALL_CUDA(cudaDeviceSynchronize());
    printf("After synchronization. \n");
    return 0;
}
