/* definitions and function prototypes regarding the brightness temperature box */
#ifndef _BTEMP_H
#define _BTEMP_H

#include "InputParameters.h"
#include "OutputStructs.h"

#ifdef __cplusplus
extern "C" {
#endif
int ComputeBrightnessTemp(float redshift, UserParams *user_params, CosmoParams *cosmo_params,
                           AstroParams *astro_params, FlagOptions *flag_options,
                           TsBox *spin_temp, IonizedBox *ionized_box,
                           PerturbedField *perturb_field, BrightnessTemp *box);

#ifdef __cplusplus
}
#endif
#endif
