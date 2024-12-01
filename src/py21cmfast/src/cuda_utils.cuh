#ifndef _CUDA_UTILS_CUH
#define _CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <stdio.h>

#define CALL_CUDA(x)                                                                    \
    do                                                                                  \
    {                                                                                   \
        cudaError_t err = (x);                                                          \
        if (err != cudaSuccess)                                                         \
        {                                                                               \
            printf("Error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    } while (0)

#endif