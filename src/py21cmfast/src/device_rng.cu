#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>

#include "cuda_utils.cuh"
#include "device_rng.cuh"

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

        // todo: add the following block to debug 
        if (ind < 2)
        {
            printf("temp check rng init.\n");
            printf("Thread %d: d = %u, v0 = %u, boxmuller_flag = %d, boxmuller_extra = %f\n",
                   ind, d_randStates[ind].d, d_randStates[ind].v[0],
                   d_randStates[ind].boxmuller_flag, d_randStates[ind].boxmuller_extra);
        }
    }
}

// Function to initialize RNG states.
void init_rand_states(unsigned long long int seed, int numStates)
{
    CALL_CUDA(cudaMemcpyToSymbol(d_numStates, &numStates, sizeof(int), 0, cudaMemcpyHostToDevice));

    // todo: add the following block to debug
    curandState *checkPtr0 = nullptr;
    CALL_CUDA(cudaMemcpyFromSymbol(&checkPtr0, d_randStates, sizeof(checkPtr0), 0, cudaMemcpyDeviceToHost));
    printf("init device pointer = %p\n", checkPtr0);

    curandState *tmpPtr = nullptr;
    CALL_CUDA(cudaMalloc((void **)&tmpPtr, numStates * sizeof(curandState)));
    CALL_CUDA(cudaMemcpyToSymbol(d_randStates, &tmpPtr, sizeof(tmpPtr), 0, cudaMemcpyHostToDevice));
    tmpPtr = nullptr;

    // todo: add the following block to debug (verify device pointer has been updated successfully)
    curandState *checkPtr = nullptr;
    CALL_CUDA(cudaMemcpyFromSymbol(&checkPtr, d_randStates, sizeof(checkPtr), 0, cudaMemcpyDeviceToHost));
    printf("updated device pointer = %p\n", checkPtr);

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

    if (h_numStates != 0){
        h_numStates = 0;
        CALL_CUDA(cudaMemcpyToSymbol(d_numStates, &h_numStates, sizeof(int), 0, cudaMemcpyHostToDevice));
    }
}
