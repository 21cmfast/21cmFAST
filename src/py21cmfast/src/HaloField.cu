#ifndef _HALOFIELD_CU
#define _HALOFIELD_CU

#include <cuda_runtime.h>

#include "DeviceConstants.cuh"
#include "HaloField.cuh"

// define relevant variables stored in constant memory
__constant__ UserParams d_user_params;
__constant__ CosmoParams d_cosmo_params;
__constant__ AstroParams d_astro_params;
__constant__ double d_test_params;

void updateGlobalParams(UserParams *h_user_params, CosmoParams *h_cosmo_params, AstroParams *h_astro_params){
    cudaMemcpyToSymbol(d_user_params, h_user_params, sizeof(UserParams), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_cosmo_params, h_cosmo_params, sizeof(CosmoParams), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_astro_params, h_astro_params, sizeof(AstroParams), 0, cudaMemcpyHostToDevice);
    double test_data = 5.5;
    cudaMemcpyToSymbol(d_test_params, &test_data, sizeof(double), 0, cudaMemcpyHostToDevice);
}

#endif
