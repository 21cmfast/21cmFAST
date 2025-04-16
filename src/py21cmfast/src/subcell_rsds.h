#ifndef _SUBCELL_RSD_H
#define _SUBCELL_RSD_H

#include "InputParameters.h"
#include "OutputStructs.h"

double apply_subcell_rsds(IonizedBox *ionized_box, BrightnessTemp *box, float redshift,
                          TsBox *spin_temp, float T_rad, float *v, float H);

#endif
