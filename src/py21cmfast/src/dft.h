/* Function prototypes and definitions used in the fourier transforms */
#ifndef _DFT_H
#define _DFT_H

#include <fftw3.h>
#include "InputParameters.h"

int dft_c2r_cube(bool use_wisdom, int dim, int dim_los, int n_threads, fftwf_complex *box);
int dft_r2c_cube(bool use_wisdom, int dim, int dim_los, int n_threads, fftwf_complex *box);
int CreateFFTWWisdoms(UserParams *user_params, CosmoParams *cosmo_params);

#endif