#include <stdio.h>

__global__ void hello_kernel() {
    printf("Hello World from GPU! BlockIdx: %d, ThreadIdx: %d\n", blockIdx.x, threadIdx.x);
}

extern "C" int call_cuda() {
    hello_kernel<<<3, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}