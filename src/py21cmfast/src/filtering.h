#ifndef _FILTERING_H
#define _FILTERING_H

#include <fftw3.h>
// #include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
void filter_box(fftwf_complex *box, int RES, int filter_type, float R, float R_param);
void filter_box_cpu(fftwf_complex *box, int RES, int filter_type, float R, float R_param);
void filter_box_gpu(fftwf_complex *box, int RES, int filter_type, float R, float R_param);
int test_filter(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options
                    , float *input_box, double R, double R_param, int filter_flag, double *result);
int test_filter_cpu(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options
                    , float *input_box, double R, double R_param, int filter_flag, double *result);
int test_filter_gpu(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options
                    , float *input_box, double R, double R_param, int filter_flag, double *result);

#ifdef __cplusplus
}
#endif
#endif
