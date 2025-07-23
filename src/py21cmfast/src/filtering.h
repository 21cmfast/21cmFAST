#ifndef _FILTERING_H
#define _FILTERING_H

#include <complex.h>
#include <fftw3.h>

#ifdef __cplusplus
extern "C" {
#endif

void filter_box(fftwf_complex *box, int RES, int filter_type, float R, float R_param);
void filter_box_cpu(fftwf_complex *box, int RES, int filter_type, float R, float R_param);
void filter_box_gpu(fftwf_complex *box, int RES, int filter_type, float R, float R_param);
int test_filter(float *input_box, double R, double R_param, int filter_flag, double *result);
int test_filter_cpu(float *input_box, double R, double R_param, int filter_flag, double *result);
int test_filter_gpu(float *input_box, double R, double R_param, int filter_flag, double *result);
double filter_function(double k, int filter_type);
double dwdm_filter(double k, double R, int filter_type);

#ifdef __cplusplus
}
#endif
#endif
