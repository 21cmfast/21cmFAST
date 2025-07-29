#ifndef _HALOFIELD_CU
#define _HALOFIELD_CU

#include <cuda_runtime.h>

#include "DeviceConstants.cuh"
#include "HaloField.cuh"

// define relevant variables stored in constant memory
__constant__ MatterOptions d_matter_options;
__constant__ SimulationOptions d_simulation_options;
__constant__ CosmoParams d_cosmo_params;
__constant__ AstroParams d_astro_params;

void updateGlobalParams(SimulationOptions *h_simulation_options, MatterOptions * h_matter_options, CosmoParams *h_cosmo_params, AstroParams *h_astro_params){
    cudaMemcpyToSymbol(d_simulation_options, h_simulation_options, sizeof(SimulationOptions), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_matter_options, h_matter_options, sizeof(MatterOptions), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_cosmo_params, h_cosmo_params, sizeof(CosmoParams), 0, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_astro_params, h_astro_params, sizeof(AstroParams), 0, cudaMemcpyHostToDevice);
}

#endif
