#ifndef _SUBCELL_RSD_H
#define _SUBCELL_RSD_H

#include "InputParameters.h"
#include "OutputStructs.h"

double apply_subcell_rsds(IonizedBox *ionized_box, BrightnessTemp *box, float redshift,
                          TsBox *spin_temp, float T_rad, float *v, float H);
double compute_rsds(float *box_in, float *los_displacement, int I1, int J1, int K1, int n_subcells, int n_threads, float *box_out);

#endif
