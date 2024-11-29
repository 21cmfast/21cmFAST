#ifndef _INTERP_TABLES_CUH
#define _INTERP_TABLES_CUH

#include "interpolation_types.h"

#ifdef __CUDA_ARCH__
__device__ double EvaluateSigma(float x, double x_min, double x_width, float *y_arr, int n_bin);
__device__ double EvaluateNhaloInv();
__device__ double extrapolate_dNdM_inverse();
__device__ double EvaluateMcoll();
__device__ double EvaluateNhalo();
#endif

#ifdef __cplusplus
extern "C"
{
#endif
    void copyTablesToDevice(RGTable1D h_Nhalo_table, RGTable1D h_Mcoll_table, RGTable2D h_Nhalo_inv_table);
#ifdef __cplusplus
}
#endif

#endif
