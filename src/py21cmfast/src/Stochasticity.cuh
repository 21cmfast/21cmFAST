#ifndef _STOCHASTICITY_CUH
#define _STOCHASTICITY_CUH

#define HALO_CUDA_THREAD_FACTOR (int) (4)

#ifdef __cplusplus
extern "C"
{
#endif
    int updateHaloOut(float *halo_masses, float *star_rng, float *sfr_rng, float *xray_rng, int *halo_coords,
                      unsigned long long int n_halos, float *y_arr, int n_bin_y, double x_min, double x_width,
                      struct HaloSamplingConstants hs_constants, unsigned long long int n_buffer, HaloField *halofield_out);
#ifdef __cplusplus
}
#endif

#endif