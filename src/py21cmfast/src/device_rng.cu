#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include "cuda_utils.cuh"

// Define this before including device_rng.cuh to avoid extern declarations
#define _DEVICE_RNG_CU_IMPL
#include "device_rng.cuh"

// Now define the actual device variables
__device__ curandState *d_randStates = nullptr;
__device__ int d_numStates = 0;

// initiate random states
// use the same random seed, different sub-sequence, and with offset of 0
__global__ void initRandStates(unsigned long long int random_seed, int totalStates)
{
    // get thread idx
    int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind < totalStates){
        curand_init(random_seed, ind, 0, &d_randStates[ind]);
    }
}

// Function to initialize RNG states.
void init_rand_states(unsigned long long int seed, int numStates)
{
    // ensure previously allocated random states on the device are freed before allocating new ones
    free_rand_states();

    CALL_CUDA(cudaMemcpyToSymbol(d_numStates, &numStates, sizeof(int), 0, cudaMemcpyHostToDevice));

    curandState *tmpPtr = nullptr;
    CALL_CUDA(cudaMalloc((void **)&tmpPtr, numStates * sizeof(curandState)));
    CALL_CUDA(cudaMemcpyToSymbol(d_randStates, &tmpPtr, sizeof(tmpPtr), 0, cudaMemcpyHostToDevice));
    tmpPtr = nullptr;

    // define kernel grids
    int threadsPerBlock = 256;
    int blocks = (numStates + threadsPerBlock - 1) / threadsPerBlock;

    // launch kernel function
    initRandStates<<<blocks, threadsPerBlock>>>(seed, numStates);
    CALL_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();
}

void free_rand_states()
{
    // CRITICAL: Wait for ALL GPU kernels to finish before freeing d_randStates
    // This ensures no kernel is still using d_randStates when we deallocate it
    cudaDeviceSynchronize();

    // copy device pointer/variable to the host
    curandState *h_randStates = nullptr;
    int h_numStates = 0;
    CALL_CUDA(cudaMemcpyFromSymbol(&h_randStates, d_randStates, sizeof(d_randStates), 0, cudaMemcpyDeviceToHost));
    CALL_CUDA(cudaMemcpyFromSymbol(&h_numStates, d_numStates, sizeof(int), 0, cudaMemcpyDeviceToHost));

    if (h_randStates){
        CALL_CUDA(cudaFree(h_randStates));
        h_randStates = nullptr;
        CALL_CUDA(cudaMemcpyToSymbol(d_randStates, &h_randStates, sizeof(h_randStates), 0, cudaMemcpyHostToDevice));
    }

    if (h_numStates){
        h_numStates = 0;
        CALL_CUDA(cudaMemcpyToSymbol(d_numStates, &h_numStates, sizeof(int), 0, cudaMemcpyHostToDevice));
    }
}
