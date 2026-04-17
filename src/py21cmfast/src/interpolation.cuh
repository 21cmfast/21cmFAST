#ifndef _INTERPOLATION_CUH
#define _INTERPOLATION_CUH

#include <stdbool.h>
#include "interpolation_types.h"

#ifdef __CUDA_ARCH__

__device__ double EvaluateRGTable1D(double x, RGTable1D *table);
__device__ double EvaluateRGTable2D(double x, double y, RGTable2D *table);

#endif

#endif
