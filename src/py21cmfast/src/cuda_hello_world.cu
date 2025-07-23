#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_utils.cuh"
#include "cuda_hello_world.cuh"

__global__ void hello_kernel() {
    printf("Hello World from GPU! BlockIdx: %d, ThreadIdx: %d\n", blockIdx.x, threadIdx.x);
}

int call_cuda() {
    hello_kernel<<<3, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}

// more members of deviceprop can be found in cura_runtime_api documentation
void print_key_device_properties(){
    int device;
    CALL_CUDA(cudaGetDevice(&device));
    cudaDeviceProp deviceProp;
    CALL_CUDA(cudaGetDeviceProperties(&deviceProp, device));
    printf("Device name: %s\n", deviceProp.name);
    printf("Total global memory: %zu bytes \n", deviceProp.totalGlobalMem);
    printf("Shared memory per block: %zu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Registers per block: %d\n", deviceProp.regsPerBlock);
    printf("Warp size: %d \n", deviceProp.warpSize);
    printf("Memory pitch: %zu bytes \n", deviceProp.memPitch);
    printf("Max threads per block: %d \n", deviceProp.maxThreadsPerBlock);
    printf("Total constant memory: %zu bytes \n", deviceProp.totalConstMem);
}
