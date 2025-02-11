#ifndef _TIGER_CHECK_H
#define _TIGER_CHECK_H
#include <fftw3.h>

#include "Stochasticity.h"

#ifdef __cplusplus
extern "C"
{
#endif
    int checkComplextype(fftwf_complex *box, int total_elements, int xy_dim, int z_dim, int midpoint, int RES, int filter_type, float R, float R_param);
    // int updateHaloOut(float *halo_masses, unsigned long long int n_halos, float *y_arr, int n_bin_y, double x_min, double x_width, struct HaloSamplingConstants hs_constants);
#ifdef __cplusplus
}
#endif

#endif // TIGER_CHECK_H
