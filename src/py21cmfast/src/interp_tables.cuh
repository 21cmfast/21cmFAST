#ifndef _INTERP_TABLES_CUH
#define _INTERP_TABLES_CUH

#include "interpolation_types.h"

#ifdef __CUDA_ARCH__
__device__ double EvaluateSigma(float x, double x_min, double x_width, float *y_arr, int n_bin);
__device__ double extrapolate_dNdM_inverse(double condition, double lnp);
__device__ double EvaluateNhaloInv(double condition, double prob);
__device__ double EvaluateMcoll(double condition, double growthf, double lnMmin, double lnMmax, double M_cond, double sigma, double delta);
__device__ double EvaluateNhalo(double condition, double growthf, double lnMmin, double lnMmax, double M_cond, double sigma, double delta);
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
