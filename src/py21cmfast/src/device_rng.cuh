#ifndef _DEVICE_RNG_CUH
#define _DEVICE_RNG_CUH

#ifdef __CUDACC__
#include <curand_kernel.h>
// Declare the device variables as extern so that they can be shared across CUDA files.
extern __device__ curandState *d_randStates;
extern __device__ int d_numStates;
#endif


#ifdef __cplusplus
extern "C"
{
#endif
    // Function prototypes.
    void init_rand_states(unsigned long long int seed, int numStates);
    void free_rand_states();

#ifdef __cplusplus
}
#endif

#endif 
