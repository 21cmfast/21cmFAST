/* definitions and function prototypes regarding the brightness temperature box */
#ifndef _BTEMP_H
#define _BTEMP_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputeBrightnessTemp(float redshift, TsBox *spin_temp, IonizedBox *ionized_box,
                          PerturbedField *perturb_field, BrightnessTemp *box);

#endif
