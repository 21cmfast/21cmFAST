#ifndef _FILTERING_H
#define _FILTERING_H

#include <fftw3.h>

void filter_box(fftwf_complex *box, int RES, int filter_type, float R);
void filter_box_annulus(fftwf_complex *box, int RES, float R_inner, float R_outer);
void filter_box_mfp(fftwf_complex *box, int RES, float R, float mfp);

int test_mfp_filter(UserParams *user_params, CosmoParams *cosmo_params, AstroParams *astro_params, FlagOptions *flag_options
                    , float *input_box, double R, double mfp, double *result);

#endif
