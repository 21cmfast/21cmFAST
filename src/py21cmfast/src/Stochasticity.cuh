#ifndef _STOCHASTICITY_CUH
#define _STOCHASTICITY_CUH

#ifdef __cplusplus
extern "C"
{
#endif
    int updateHaloOut(float *halo_masses, unsigned long long int n_halos, float *y_arr, int n_bin_y, double x_min, double x_width,
                      struct HaloSamplingConstants hs_constants, unsigned long long int n_buffer);
#ifdef __cplusplus
}
#endif

#endif