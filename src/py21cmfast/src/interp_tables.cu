#include <cuda_runtime.h>

#include <math.h>
// #include <stdio.h>

// #include "InputParameters.h"
#include "interpolation_types.h"

#include "cuda_utils.cuh"
#include "interp_tables.cuh"
#include "DeviceConstants.cuh"

#include "interpolation.cu"

// define relevant variables stored in constant memory
__constant__ RGTable1D d_Nhalo_table;
__constant__ RGTable1D d_Mcoll_table;
__constant__ RGTable2D d_Nhalo_inv_table;

// specify a max size of yarr
const int device_n_max = 200;
__constant__ double d_Nhalo_yarr[device_n_max];
__constant__ double d_Mcoll_yarr[device_n_max];


// copy tables to gpu
void copyTablesToDevice(RGTable1D h_Nhalo_table, RGTable1D h_Mcoll_table, RGTable2D h_Nhalo_inv_table)
{
    // copy Nhalo table and its member y_arr
    size_t size_Nhalo_yarr = sizeof(double) * h_Nhalo_table.n_bin;
    // get a copy of the Nhalo table
    RGTable1D h_Nhalo_table_to_device = h_Nhalo_table;
    if (h_Nhalo_table.n_bin > device_n_max){
        // double *d_Nhalo_yarr;
        // todo: declare device yarr (not using constant)
        return;
    }
    else{
        CALL_CUDA(cudaMemcpyToSymbol(d_Nhalo_yarr, h_Nhalo_table.y_arr, size_Nhalo_yarr, 0, cudaMemcpyHostToDevice));
        // get memory address on the device
        double *d_Nhalo_yarr_device;
        CALL_CUDA(cudaGetSymbolAddress((void **)&d_Nhalo_yarr_device, d_Nhalo_yarr));

        h_Nhalo_table_to_device.y_arr = d_Nhalo_yarr_device;
    }
    CALL_CUDA(cudaMemcpyToSymbol(d_Nhalo_table, &h_Nhalo_table_to_device, sizeof(RGTable1D), 0, cudaMemcpyHostToDevice));

    // copy Mcoll table and its member y_arr
    size_t size_Mcoll_yarr = sizeof(double) * h_Mcoll_table.n_bin;
    // get a copy of Mcoll table
    RGTable1D h_Mcoll_table_to_device = h_Mcoll_table;
    if (h_Mcoll_table.n_bin > device_n_max){
        return;
    }
    else{
        CALL_CUDA(cudaMemcpyToSymbol(d_Mcoll_yarr, h_Mcoll_table.y_arr, size_Mcoll_yarr, 0, cudaMemcpyHostToDevice));
        // get memory address on the device
        double *d_Mcoll_yarr_device;
        CALL_CUDA(cudaGetSymbolAddress((void **)&d_Mcoll_yarr_device, d_Mcoll_yarr));

        h_Mcoll_table_to_device.y_arr = d_Mcoll_yarr_device;
    }
    CALL_CUDA(cudaMemcpyToSymbol(d_Mcoll_table, &h_Mcoll_table_to_device, sizeof(RGTable1D), 0, cudaMemcpyHostToDevice));

    // copy Nhalo_inv table and its member flatten_data
    size_t size_Nhalo_inv_flatten_data = sizeof(double) * h_Nhalo_inv_table.nx_bin * h_Nhalo_inv_table.ny_bin;
    // get a copy of Nhalo_inv_table
    RGTable2D h_Nhalo_inv_table_to_device = h_Nhalo_inv_table;

    double *d_Nhalo_flatten_data;
    CALL_CUDA(cudaMalloc(&d_Nhalo_flatten_data, size_Nhalo_inv_flatten_data));
    CALL_CUDA(cudaMemcpy(d_Nhalo_flatten_data, h_Nhalo_inv_table.flatten_data, size_Nhalo_inv_flatten_data, cudaMemcpyHostToDevice));

    double **d_z_arr, **z_arr_to_device;
    size_t size_z_arr = sizeof(double *) * h_Nhalo_inv_table.nx_bin;
    CALL_CUDA(cudaHostAlloc((void **)&z_arr_to_device, size_z_arr, cudaHostAllocDefault));
    // get the address of flatten data on the device
    int i;
    for (i=0;i<h_Nhalo_inv_table.nx_bin;i++){
        z_arr_to_device[i] = &d_Nhalo_flatten_data[i * h_Nhalo_inv_table.ny_bin];
    }

    CALL_CUDA(cudaMalloc(&d_z_arr, size_z_arr));
    CALL_CUDA(cudaMemcpy(d_z_arr, z_arr_to_device, size_z_arr, cudaMemcpyHostToDevice));

    // free data after it's been copied to the device
    CALL_CUDA(cudaFreeHost(z_arr_to_device));

    h_Nhalo_inv_table_to_device.flatten_data = d_Nhalo_flatten_data;
    h_Nhalo_inv_table_to_device.z_arr = d_z_arr;

    CALL_CUDA(cudaMemcpyToSymbol(d_Nhalo_inv_table, &h_Nhalo_inv_table_to_device, sizeof(RGTable2D), 0, cudaMemcpyHostToDevice));
}

// assume use interpolation table is true at this stage, add the check later
// todo: double check whether I should use float or double or x, it's been mixed used in c code
__device__ double EvaluateSigma(float x, double x_min, double x_width, float *y_arr, int n_bin)
{
    // using log units to make the fast option faster and the slow option slower
    // return EvaluateRGTable1D_f(lnM, table);
    int idx = (int)floor((x - x_min) / x_width);
    if (idx < 0 || idx >= n_bin - 1)
    {
        return 0.0; // Out-of-bounds handling
    }

    double table_val = x_min + x_width * (float)idx;
    double interp_point = (x - table_val) / x_width;

    return y_arr[idx] * (1 - interp_point) + y_arr[idx + 1] * (interp_point);
}

__device__ double extrapolate_dNdM_inverse(double condition, double lnp)
{
    double x_min = d_Nhalo_inv_table.x_min;
    double x_width = d_Nhalo_inv_table.x_width;
    // printf("condition: %f; lnp: %f \n", condition, lnp); //tmp
    int x_idx = (int)floor((condition - x_min) / x_width);
    double x_table = x_min + x_idx * x_width;
    double interp_point_x = (condition - x_table) / x_width;

    double extrap_point_y = (lnp - d_user_params.MIN_LOGPROB) / d_Nhalo_inv_table.y_width;

    // find the log-mass at the edge of the table for this condition
    double xlimit = d_Nhalo_inv_table.z_arr[x_idx][0] * (interp_point_x) + d_Nhalo_inv_table.z_arr[x_idx + 1][0] * (1 - interp_point_x);
    double xlimit_m1 = d_Nhalo_inv_table.z_arr[x_idx][1] * (interp_point_x) + d_Nhalo_inv_table.z_arr[x_idx + 1][1] * (1 - interp_point_x);

    double result = xlimit + (xlimit_m1 - xlimit) * (extrap_point_y);

    return result;
}

__device__ double EvaluateNhaloInv(double condition, double prob)
{
    if (prob == 0.)
        return 1.; // q == 1 -> condition mass
    double lnp = log(prob);
    if (lnp < d_user_params.MIN_LOGPROB)
        return extrapolate_dNdM_inverse(condition, lnp);
    return EvaluateRGTable2D(condition, lnp, &d_Nhalo_inv_table);
}

__device__ double EvaluateMcoll(double condition, double growthf, double lnMmin, double lnMmax, double M_cond, double sigma, double delta)
{
    if (d_user_params.USE_INTERPOLATION_TABLES)
        return EvaluateRGTable1D(condition, &d_Mcoll_table);
    // todo: implement Mcoll_Conditional
    return 0;
}

__device__ double EvaluateNhalo(double condition, double growthf, double lnMmin, double lnMmax, double M_cond, double sigma, double delta)
{
    if (d_user_params.USE_INTERPOLATION_TABLES)
        return EvaluateRGTable1D(condition, &d_Nhalo_table);
    // todo: implement Nhalo_Conditional
    return 0;
}
