#ifndef _FILTERING_H
#define _FILTERING_H

#include <fftw3.h>

void filter_box(fftwf_complex *box, int RES, int filter_type, float R, float R_param);
int test_filter(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params,
                FlagOptions *flag_options, float *input_box, double R, double R_param,
                int filter_flag, double *result);
double filter_function(double k, int filter_type);
double dwdm_filter(double k, double R, int filter_type);

#endif
