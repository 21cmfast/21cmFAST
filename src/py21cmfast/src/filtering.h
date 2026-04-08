#ifndef _FILTERING_H
#define _FILTERING_H

#include <fftw3.h>

#include "InputParameters.h"

void filter_box(fftwf_complex *box, int box_dim[3], int filter_type, float R, float R_param,
                float R_star);
int test_filter(float *input_box, double R, double R_param, double R_star, int filter_flag,
                double *result);
double filter_function(double k, int filter_type);
double dwdm_filter(double k, double R, int filter_type);

#endif
