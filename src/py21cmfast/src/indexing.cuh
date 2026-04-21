/*
    indexing.cuh -- CUDA device functions and macros for indexing 3D arrays.

    These are the GPU equivalents of the inline functions in indexing.h,
    plus compatibility macros for hi-res box indexing that were removed
    from indexing.h in upstream.
*/
#ifndef _INDEXING_CUH
#define _INDEXING_CUH

#include <cuda_runtime.h>
#include "InputParameters.h"

// -------------------------------------------------------------------------------------
// Hi-res box macros (removed from upstream indexing.h, needed for GPU code)
// Note: Some macros like D_PARA are still in indexing.h, so we guard against redefinition
// -------------------------------------------------------------------------------------
#ifndef TWOPI
#define TWOPI (2.0 * M_PI)
#endif

#ifndef DELTA_K
#define DELTA_K (TWOPI / simulation_options_global->BOX_LEN)
#endif

#ifndef DELTA_K_PARA
#define DELTA_K_PARA \
    (TWOPI / (simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->BOX_LEN))
#endif

#ifndef D
#define D (long long)simulation_options_global->DIM
#endif

// D_PARA is still defined in indexing.h, don't redefine

#ifndef MIDDLE
#define MIDDLE (simulation_options_global->DIM / 2)
#endif

#ifndef MIDDLE_PARA
#define MIDDLE_PARA \
    (simulation_options_global->NON_CUBIC_FACTOR * simulation_options_global->DIM / 2)
#endif

#ifndef MID
#define MID ((long long)MIDDLE)
#endif

#ifndef MID_PARA
#define MID_PARA ((long long)MIDDLE_PARA)
#endif

#ifndef L_FACTOR
#define L_FACTOR (simulation_options_global->BOX_LEN / (double)simulation_options_global->DIM)
#endif

// INDEXING MACROS for hi-res boxes
#ifndef C_INDEX
// for 3D complex array
#define C_INDEX(x, y, z) ((unsigned long long)((z) + (MID_PARA + 1llu) * ((y) + D * (x))))
#endif

#ifndef R_FFT_INDEX
// for 3D real array with the FFT padding
#define R_FFT_INDEX(x, y, z) \
    ((unsigned long long)((z) + 2llu * (MID_PARA + 1llu) * ((y) + D * (x))))
#endif

#ifndef R_INDEX
// for 3D real array with no padding
#define R_INDEX(x, y, z) ((unsigned long long)((z) + D_PARA * ((y) + D * (x))))
#endif

// -------------------------------------------------------------------------------------
// Device-compatible indexing functions
// -------------------------------------------------------------------------------------

// Indexing a 3D array stored in a 1D array
__device__ __host__ inline unsigned long long grid_index_general_d(int x, int y, int z, int dim[3]) {
    return (z) + (dim[2] + 0llu) * (y + (dim[1] + 0llu) * x);
}

// Indexing a 3D array stored in a 1D array, where the 3D array is the real-space
// representation of a Fourier Transform (with padding on the last axis)
__device__ __host__ inline unsigned long long grid_index_fftw_r_d(int x, int y, int z, int dim[3]) {
    return (z) + 2llu * (dim[2] / 2 + 1llu) * (y + (dim[1] + 0llu) * x);
}

// Indexing a 3D array stored in a 1D array, where the 3D array is the complex-space
// representation of a Fourier Transform (with padding on the last axis)
__device__ __host__ inline unsigned long long grid_index_fftw_c_d(int x, int y, int z, int dim[3]) {
    return (z) + (dim[2] / 2 + 1llu) * (y + (dim[1] + 0llu) * x);
}

#endif // _INDEXING_CUH
