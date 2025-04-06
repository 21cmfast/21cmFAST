/* definitions and function prototypes regarding the brightness temperature box */
#ifndef _BTEMP_H
#define _BTEMP_H

#include "InputParameters.h"
#include "OutputStructs.h"

int ComputeBrightnessTemp(float redshift, MatterParams *matter_params, MatterFlags *matter_flags,
                          CosmoParams *cosmo_params, AstroParams *astro_params,
                          AstroFlags *astro_flags, TsBox *spin_temp, IonizedBox *ionized_box,
                          PerturbedField *perturb_field, BrightnessTemp *box);

#endif
