#ifndef _FILTERING_H
#define _FILTERING_H

#include <fftw3.h>

void filter_box(fftwf_complex *box, int RES, int filter_type, float R);
void filter_box_annulus(fftwf_complex *box, int RES, float R_inner, float R_outer);
void filter_box_mfp(fftwf_complex *box, int RES, float R, float mfp);

#endif