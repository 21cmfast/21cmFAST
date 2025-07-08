#ifndef _FILTERING_H
#define _FILTERING_H

#include <fftw3.h>

#include "InputParameters.h"

void filter_box(fftwf_complex *box, int RES, int filter_type, float R, float R_param, float r_star);
int test_filter(float *input_box, double R, double R_param, double r_star, int filter_flag, double *result);
double filter_function(double k, int filter_type);
double dwdm_filter(double k, double R, int filter_type);

#endif
