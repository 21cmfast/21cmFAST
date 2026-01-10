/*
    InitialConditions_gpu.h -- GPU-accelerated initial conditions computation

    This header declares the GPU version of ComputeInitialConditions.
    The GPU implementation offloads parallelizable computations (k-space operations,
    velocity field calculations, subsampling) to the GPU while keeping FFTs on CPU.
*/

#ifndef _INITCONDITIONS_GPU_H
#define _INITCONDITIONS_GPU_H

#include "OutputStructs.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
    ComputeInitialConditions_gpu - GPU-accelerated initial conditions computation

    This function generates the initial conditions using GPU acceleration:
    - Gaussian random field generation (CPU, for reproducibility)
    - Power spectrum multiplication (GPU kernel)
    - Complex conjugate adjustment (GPU kernel)
    - Velocity field computation (GPU kernel)
    - Filtering/subsampling (GPU kernel)
    - FFT operations (CPU, using FFTW)

    Parameters:
        random_seed: Seed for random number generation
        boxes: Output structure for initial conditions

    Returns:
        0 on success, error code on failure
*/
int ComputeInitialConditions_gpu(unsigned long long random_seed, InitialConditions *boxes);

#ifdef __cplusplus
}
#endif

#endif /* _INITCONDITIONS_GPU_H */
